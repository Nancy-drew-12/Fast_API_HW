import time

from fastapi import APIRouter, Body

from app.api.deps import require_model
from app.api.examples import PREDICT_EXAMPLES, PREDICT_RESPONSES
from app.core.errors import ServiceError
from app.core.logging import logger
from app.ml.model import predict_churn
from app.ml.preprocessing import features_to_dataframe
from app.schemas import FeatureVectorChurn, PredictionResponseChurn

router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictionResponseChurn | list[PredictionResponseChurn],
    responses=PREDICT_RESPONSES,
)
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn] = Body(openapi_examples=PREDICT_EXAMPLES),
):
    bundle = require_model()

    is_single = not isinstance(features, list)
    clients = [features] if is_single else features

    start = time.perf_counter()
    try:
        predictions = predict_churn(bundle["pipeline"], features_to_dataframe(clients))
    except Exception as error:
        raise ServiceError(
            "prediction_failed", "Не удалось получить предсказание модели",
            status_code=500, details=str(error),
        )
    duration_ms = (time.perf_counter() - start) * 1000

    churn_count = sum(p["churn"] for p in predictions)
    logger.info("Predict: %d client(s), %d predicted churn, %.1f ms", len(predictions), churn_count, duration_ms)

    return predictions[0] if is_single else predictions
