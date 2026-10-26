from fastapi import APIRouter, Body, Query

from app.api.deps import require_dataset
from app.api.examples import CLIENT_EXAMPLE, TRAIN_EXAMPLES, TRAIN_RESPONSES
from app.core.config import FEATURE_COLUMNS, NUMERIC_FEATURES
from app.core.errors import ServiceError
from app.core.logging import logger
from app.core.state import state
from app.ml.history import add_training_record, best_record, filter_history, load_history
from app.ml.model import evaluate_model, known_categories, save_churn_model, train_churn_model
from app.ml.preprocessing import prepare_data, split_data
from app.schemas import FeatureVectorChurn, TrainingConfigChurn
from app.schemas.churn import ModelType

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train", responses=TRAIN_RESPONSES)
def train_model(config: TrainingConfigChurn = Body(openapi_examples=TRAIN_EXAMPLES)):
    data = require_dataset()

    logger.info("Training started: model_type=%s, hyperparameters=%s", config.model_type, config.hyperparameters)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    try:
        pipeline, params = train_churn_model(X_train, y_train, config.model_type, config.hyperparameters)
    except (TypeError, ValueError) as error:
        raise ServiceError("invalid_hyperparameters", "Неверные гиперпараметры модели", details=str(error))

    metrics = evaluate_model(pipeline, X_test, y_test)
    state.model_bundle = save_churn_model(pipeline, metrics, config.model_type, params)
    trained_at = state.model_bundle["trained_at"]
    logger.info("Training finished: model_type=%s, metrics=%s", config.model_type, metrics)

    add_training_record({
        "trained_at": trained_at,
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
        "trained_at": trained_at,
        "metrics": metrics,
    }


@router.get("/status")
def model_status():
    bundle = state.model_bundle
    if bundle is None:
        return {"is_trained": False, "model_type": None, "hyperparameters": None,
                "trained_at": None, "metrics": None}
    return {
        "is_trained": True,
        "model_type": bundle["model_type"],
        "hyperparameters": bundle["hyperparameters"],
        "trained_at": bundle["trained_at"],
        "metrics": bundle["metrics"],
    }


@router.get("/schema")
def model_schema():
    properties = FeatureVectorChurn.model_json_schema()["properties"]
    known = known_categories(state.model_bundle["pipeline"]) if state.model_bundle else {}

    features = []
    for name in FEATURE_COLUMNS:
        feature = {
            "name": name,
            "type": properties[name]["type"],
            "kind": "numeric" if name in NUMERIC_FEATURES else "categorical",
        }
        if name in known:
            feature["known_values"] = known[name]
        features.append(feature)

    return {"n_features": len(features), "features": features, "example": CLIENT_EXAMPLE}


@router.get("/metrics")
def model_metrics(
    limit: int = Query(5, ge=1, le=100, description="Сколько последних записей вернуть"),
    model_type: ModelType | None = Query(None, description="Фильтр по типу модели"),
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
