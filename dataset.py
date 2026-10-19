from pathlib import Path

import pandas as pd

from schemas import DatasetRowChurn

DATA_PATH = Path(__file__).parent / "data" / "churn_dataset.csv"
TARGET = "churn"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Читает CSV и возвращает DataFrame."""
    return pd.read_csv(path)


def to_rows(df: pd.DataFrame) -> list[DatasetRowChurn]:
    """Превращает строки DataFrame в объекты DatasetRowChurn (с проверкой типов)."""
    return [DatasetRowChurn(**row) for row in df.to_dict(orient="records")]


def get_info(df: pd.DataFrame) -> dict:
    """Краткая сводка по датасету."""
    churn_counts = df[TARGET].value_counts().sort_index()
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "features": [col for col in df.columns if col != TARGET],
        "churn_distribution": {str(k): int(v) for k, v in churn_counts.items()},
    }