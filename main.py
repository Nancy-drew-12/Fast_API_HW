from fastapi import Body, FastAPI, HTTPException, Query

from dataset import get_info, load_dataset, to_rows
from model import evaluate_model, load_churn_model, predict_churn, save_churn_model, train_churn_model
from preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    features_to_dataframe,
    get_split_info,
    prepare_data,
    split_data,
)
from schemas import DatasetRowChurn, FeatureVectorChurn, PredictionResponseChurn, TrainingConfigChurn

app = FastAPI()

try:
    df = load_dataset()   # читаем датасет один раз при старте сервиса
except FileNotFoundError:
    df = None             # файла нет — сервис всё равно запустится

try:
    model_bundle = load_churn_model()   # пробуем поднять модель, сохранённую в прошлый раз
except Exception:
    model_bundle = None                 # файл повреждён — начинаем без модели


def require_dataset():
    """Проверка: датасет загружен и не пустой."""
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Датасет не загружен или пуст")
    return df


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


CLIENT_EXAMPLE = {
    "monthly_fee": 25.5,
    "usage_hours": 12,
    "support_requests": 2,
    "account_age_months": 8,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "tablet",
    "payment_method": "card",
    "autopay_enabled": 0,
}

PREDICT_EXAMPLES = {
    "one_client": {
        "summary": "Один клиент",
        "value": CLIENT_EXAMPLE,
    },
    "several_clients": {
        "summary": "Несколько клиентов",
        "value": [
            CLIENT_EXAMPLE,
            {**CLIENT_EXAMPLE, "usage_hours": 35, "failed_payments": 0,
             "account_age_months": 30, "autopay_enabled": 1},
        ],
    },
}


@app.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses={
        200: {
            "description": "Предсказанный класс и вероятности",
            "content": {"application/json": {"example": {
                "churn": 1, "probability_no_churn": 0.38, "probability_churn": 0.62,
            }}},
        },
        400: {"description": "Модель ещё не обучена"},
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(openapi_examples=PREDICT_EXAMPLES),
):
    if model_bundle is None:
        raise HTTPException(
            status_code=400,
            detail="Модель ещё не обучена. Сначала вызовите POST /model/train",
        )

    is_single = not isinstance(features, list)
    clients = [features] if is_single else features

    X = features_to_dataframe(clients)
    predictions = predict_churn(model_bundle["pipeline"], X)

    return predictions[0] if is_single else predictions


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1, le=100)) -> list[DatasetRowChurn]:
    return to_rows(require_dataset().head(n))


@app.get("/dataset/info")
def dataset_info():
    return get_info(require_dataset())


@app.get("/dataset/split-info")
def dataset_split_info():
    return get_split_info(require_dataset())


TRAIN_EXAMPLES = {
    "logreg_default": {
        "summary": "Логистическая регрессия по умолчанию",
        "value": {"model_type": "logreg", "hyperparameters": {}},
    },
    "logreg_custom": {
        "summary": "Логистическая регрессия с регуляризацией C=0.1",
        "value": {"model_type": "logreg", "hyperparameters": {"C": 0.1}},
    },
    "random_forest": {
        "summary": "Случайный лес",
        "value": {"model_type": "random_forest",
                  "hyperparameters": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5}},
    },
}


@app.post("/model/train")
def train_model(config: TrainingConfigChurn = Body(openapi_examples=TRAIN_EXAMPLES)):
    global model_bundle
    data = require_dataset()

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    try:
        pipeline, params = train_churn_model(X_train, y_train, config.model_type, config.hyperparameters)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Неверные гиперпараметры: {error}")

    metrics = evaluate_model(pipeline, X_test, y_test)
    model_bundle = save_churn_model(pipeline, metrics, config.model_type, params)

    return {
        "status": "model trained and saved",
        "model_type": config.model_type,
        "hyperparameters": params,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": model_bundle["trained_at"],
        "metrics": metrics,
    }


@app.get("/model/status")
def model_status():
    if model_bundle is None:
        return {"is_trained": False, "model_type": None, "hyperparameters": None,
                "trained_at": None, "metrics": None}
    return {
        "is_trained": True,
        "model_type": model_bundle.get("model_type", "logreg"),
        "hyperparameters": model_bundle.get("hyperparameters", {}),
        "trained_at": model_bundle["trained_at"],
        "metrics": model_bundle["metrics"],
    }


@app.get("/model/schema")
def model_schema():
    properties = FeatureVectorChurn.model_json_schema()["properties"]

    allowed_values = {}
    if model_bundle is not None:
        encoder = (model_bundle["pipeline"]
                   .named_steps["preprocessor"]
                   .named_transformers_["cat"]
                   .named_steps["onehot"])
        for column, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
            allowed_values[column] = [str(value) for value in categories]

    features = []
    for name in FEATURE_COLUMNS:
        feature = {
            "name": name,
            "type": properties[name]["type"],
            "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
        }
        if name in allowed_values:
            feature["known_values"] = allowed_values[name]
        features.append(feature)

    return {
        "n_features": len(features),
        "features": features,
        "example": CLIENT_EXAMPLE,
    }