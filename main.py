from fastapi import FastAPI, HTTPException, Query

from dataset import get_info, load_dataset, to_rows
from model import evaluate_model, load_churn_model, save_churn_model, train_churn_model
from preprocessing import get_split_info, prepare_data, split_data
from schemas import DatasetRowChurn, FeatureVectorChurn

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


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1, le=100)) -> list[DatasetRowChurn]:
    return to_rows(require_dataset().head(n))


@app.get("/dataset/info")
def dataset_info():
    return get_info(require_dataset())


@app.get("/dataset/split-info")
def dataset_split_info():
    return get_split_info(require_dataset())


@app.post("/model/train")
def train_model():
    global model_bundle
    data = require_dataset()

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = train_churn_model(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    model_bundle = save_churn_model(pipeline, metrics)

    return {
        "status": "model trained and saved",
        "model": "LogisticRegression",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": model_bundle["trained_at"],
        "metrics": metrics,
    }


@app.get("/model/status")
def model_status():
    if model_bundle is None:
        return {"is_trained": False, "trained_at": None, "metrics": None}
    return {
        "is_trained": True,
        "trained_at": model_bundle["trained_at"],
        "metrics": model_bundle["metrics"],
    }