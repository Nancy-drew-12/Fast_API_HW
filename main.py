from fastapi import FastAPI, Query

from dataset import get_info, load_dataset, to_rows
from schemas import DatasetRowChurn, FeatureVectorChurn

app = FastAPI()

df = load_dataset()   # читаем датасет один раз при старте сервиса


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1, le=100)) -> list[DatasetRowChurn]:
    return to_rows(df.head(n))


@app.get("/dataset/info")
def dataset_info():
    return get_info(df)