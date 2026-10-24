from typing import Literal

from fastapi import Body, FastAPI, Query

from dataset import TARGET, get_info, load_dataset, to_rows
from errors import (
    INTERNAL_EXAMPLE,
    NO_DATASET_EXAMPLE,
    NO_MODEL_EXAMPLE,
    VALIDATION_EXAMPLE,
    ErrorResponse,
    ServiceError,
    error_example,
    register_error_handlers,
)
from history import add_training_record, best_record, filter_history, load_history
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

app = FastAPI(title="Churn prediction service")
register_error_handlers(app)

try:
    df = load_dataset()   # читаем датасет один раз при старте сервиса
except FileNotFoundError:
    df = None             # файла нет — сервис всё равно запустится

try:
    model_bundle = load_churn_model()   # пробуем поднять модель, сохранённую в прошлый раз
except Exception:
    model_bundle = None                 # файл повреждён — начинаем без модели


def require_dataset():
    """Проверка: датасет загружен, не пустой и содержит все нужные колонки."""
    if df is None or df.empty:
        raise ServiceError("dataset_empty", "Датасет не загружен или пуст")

    missing = [col for col in FEATURE_COLUMNS + [TARGET] if col not in df.columns]
    if missing:
        raise ServiceError(
            "dataset_invalid",
            "В датасете не хватает нужных колонок",
            details={"missing_columns": missing},
        )
    return df


def require_model():
    """Проверка: модель обучена."""
    if model_bundle is None:
        raise ServiceError("model_not_trained", "Модель ещё не обучена. Сначала вызовите POST /model/train")
    return model_bundle


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
        400: {
            "model": ErrorResponse,
            "description": "Модель ещё не обучена",
            "content": {"application/json": {"example": NO_MODEL_EXAMPLE}},
        },
        422: {
            "model": ErrorResponse,
            "description": "Неверные данные клиента: не тот тип, не хватает или лишний признак",
            "content": {"application/json": {"examples": {
                "wrong_type": {"summary": "Неверный тип значения", "value": VALIDATION_EXAMPLE},
                "missing_feature": {"summary": "Не хватает признака", "value": error_example(
                    "validation_error", "Неверные входные данные",
                    [{"field": "region", "problem": "Field required", "input": {"...": "..."}}])},
                "extra_feature": {"summary": "Лишний признак", "value": error_example(
                    "validation_error", "Неверные входные данные",
                    [{"field": "age", "problem": "Extra inputs are not permitted", "input": 30}])},
            }}},
        },
        500: {
            "model": ErrorResponse,
            "description": "Ошибка при расчёте предсказания",
            "content": {"application/json": {"example": error_example(
                "prediction_failed", "Не удалось получить предсказание модели")}},
        },
    },
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(openapi_examples=PREDICT_EXAMPLES),
):
    bundle = require_model()

    is_single = not isinstance(features, list)
    clients = [features] if is_single else features

    try:
        X = features_to_dataframe(clients)
        predictions = predict_churn(bundle["pipeline"], X)
    except Exception as error:
        raise ServiceError(
            "prediction_failed", "Не удалось получить предсказание модели",
            status_code=500, details=str(error),
        )

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


@app.post(
    "/model/train",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Пустой или некорректный датасет, неверные гиперпараметры",
            "content": {"application/json": {"examples": {
                "dataset_empty": {"summary": "Пустой датасет", "value": NO_DATASET_EXAMPLE},
                "dataset_invalid": {"summary": "Не хватает колонок", "value": error_example(
                    "dataset_invalid", "В датасете не хватает нужных колонок",
                    {"missing_columns": ["region"]})},
                "bad_hyperparameters": {"summary": "Неверные гиперпараметры", "value": error_example(
                    "invalid_hyperparameters", "Неверные гиперпараметры модели",
                    "RandomForestClassifier.__init__() got an unexpected keyword argument 'abc'")},
            }}},
        },
        422: {
            "model": ErrorResponse,
            "description": "Неизвестный тип модели или неверный формат конфигурации",
            "content": {"application/json": {"example": error_example(
                "validation_error", "Неверные входные данные",
                [{"field": "model_type", "problem": "Input should be 'logreg' or 'random_forest'",
                  "input": "xgboost"}])}},
        },
        500: {
            "model": ErrorResponse,
            "description": "Непредвиденная ошибка",
            "content": {"application/json": {"example": INTERNAL_EXAMPLE}},
        },
    },
)
def train_model(config: TrainingConfigChurn = Body(openapi_examples=TRAIN_EXAMPLES)):
    global model_bundle
    data = require_dataset()

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    try:
        pipeline, params = train_churn_model(X_train, y_train, config.model_type, config.hyperparameters)
    except (TypeError, ValueError) as error:
        raise ServiceError("invalid_hyperparameters", "Неверные гиперпараметры модели", details=str(error))

    metrics = evaluate_model(pipeline, X_test, y_test)
    model_bundle = save_churn_model(pipeline, metrics, config.model_type, params)

    add_training_record({
        "trained_at": model_bundle["trained_at"],
        "model_type": config.model_type,
        "hyperparameters": params,
        "metrics": metrics,
    })

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


@app.get("/model/metrics")
def model_metrics(
    limit: int = Query(5, ge=1, le=100, description="Сколько последних записей вернуть"),
    model_type: Literal["logreg", "random_forest"] | None = Query(None, description="Фильтр по типу модели"),
):
    history = load_history()
    if not history:
        raise ServiceError("no_training_history", "Модель ещё ни разу не обучалась. Вызовите POST /model/train")

    filtered = filter_history(history, model_type)

    return {
        "latest": history[-1],
        "total_trainings": len(history),
        "filter": {"model_type": model_type},
        "best_by_f1": best_record(filtered, "f1"),
        "best_by_roc_auc": best_record(filtered, "roc_auc"),
        "history": list(reversed(filtered[-limit:])),
    }