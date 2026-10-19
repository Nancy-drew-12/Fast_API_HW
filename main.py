from fastapi import FastAPI

from schemas import FeatureVectorChurn

app = FastAPI()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features