from fastapi import APIRouter

from app.core.state import state

router = APIRouter(tags=["health"])


@router.get("/")
def root():
    return {"message": "ml churn service is running"}


@router.get("/health")
def health():
    df, bundle = state.dataset, state.model_bundle
    dataset_loaded = df is not None and not df.empty
    model_loaded = bundle is not None
    return {
        "status": "ok" if dataset_loaded and model_loaded else "degraded",
        "dataset_loaded": dataset_loaded,
        "dataset_rows": int(df.shape[0]) if dataset_loaded else 0,
        "model_loaded": model_loaded,
        "model_type": bundle["model_type"] if model_loaded else None,
        "model_trained_at": bundle["trained_at"] if model_loaded else None,
        "service_started_at": state.started_at,
    }
