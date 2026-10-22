from fastapi import FastAPI, HTTPException, Query

from dataset import get_info, load_dataset, to_rows
from model import evaluate_model, train_churn_model
from preprocessing import get_split_info, prepare_data, split_data
from schemas import DatasetRowChurn, FeatureVectorChurn

app = FastAPI()

try:
    df = load_dataset()   # читаем датасет один раз при старте сервиса
except FileNotFoundError:
    df = None             # файла нет — сервис всё равно запустится

model = None              # здесь будет лежать обученный pipeline


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
    global model
    data = require_dataset()

    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_churn_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    return {
        "status": "model trained",
        "model": "LogisticRegression",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
    }