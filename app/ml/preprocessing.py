import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет X от y и заполняет пропуски."""
    df = df.dropna(subset=[TARGET])

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int)

    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].fillna(X[NUMERIC_FEATURES].median())
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna("unknown")
    return X, y


def features_to_dataframe(clients: list) -> pd.DataFrame:
    """Список FeatureVectorChurn -> DataFrame с тем же порядком колонок, что при обучении."""
    return pd.DataFrame([client.model_dump() for client in clients], columns=FEATURE_COLUMNS)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Train/test со стратификацией: доля оттока одинакова в обеих частях."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def class_distribution(y: pd.Series) -> dict:
    counts = y.value_counts().sort_index()
    shares = y.value_counts(normalize=True).sort_index()
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares": {str(k): round(float(v), 3) for k, v in shares.items()},
    }


def get_split_info(df: pd.DataFrame) -> dict:
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "churn_train": class_distribution(y_train),
        "churn_test": class_distribution(y_test),
    }
