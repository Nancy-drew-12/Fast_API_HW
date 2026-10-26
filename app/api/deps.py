"""Проверки, которые нужны нескольким эндпоинтам."""
import pandas as pd

from app.core.errors import ServiceError
from app.core.state import state
from app.ml.dataset import missing_columns


def require_dataset() -> pd.DataFrame:
    df = state.dataset
    if df is None or df.empty:
        raise ServiceError("dataset_empty", "Датасет не загружен или пуст")

    missing = missing_columns(df)
    if missing:
        raise ServiceError(
            "dataset_invalid",
            "В датасете не хватает нужных колонок",
            details={"missing_columns": missing},
        )
    return df


def require_model() -> dict:
    if state.model_bundle is None:
        raise ServiceError("model_not_trained", "Модель ещё не обучена. Сначала вызовите POST /model/train")
    return state.model_bundle
