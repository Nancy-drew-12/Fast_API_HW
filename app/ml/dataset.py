from pathlib import Path

import pandas as pd

from app.core import config
from app.schemas import DatasetRowChurn


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or config.DATA_PATH)


def to_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    """Строки DataFrame -> объекты DatasetRowChurn (с проверкой типов)."""
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Каких нужных колонок не хватает в датасете."""
    return [col for col in config.FEATURE_COLUMNS + [config.TARGET] if col not in df.columns]


def get_info(df: pd.DataFrame) -> dict:
    churn_counts = df[config.TARGET].value_counts().sort_index()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "features": [col for col in df.columns if col != config.TARGET],
        "churn_distribution": {str(k): int(v) for k, v in churn_counts.items()},
    }
