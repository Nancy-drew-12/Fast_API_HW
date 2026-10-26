import numpy as np

from app.core.config import FEATURE_COLUMNS, TARGET
from app.ml.dataset import load_dataset
from app.ml.preprocessing import features_to_dataframe, prepare_data, split_data
from app.schemas import FeatureVectorChurn


def test_real_dataset_loads_and_has_all_columns():
    df = load_dataset()
    assert not df.empty
    assert set(FEATURE_COLUMNS + [TARGET]).issubset(df.columns)


def test_prepare_data_separates_x_and_y(synthetic_df):
    X, y = prepare_data(synthetic_df)
    assert list(X.columns) == FEATURE_COLUMNS      # тот же набор и порядок
    assert TARGET not in X.columns
    assert len(X) == len(y) == len(synthetic_df)


def test_prepare_data_fills_missing_values(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "usage_hours"] = np.nan
    df.loc[1, "region"] = np.nan
    X, _ = prepare_data(df)
    assert X.isna().sum().sum() == 0
    assert X.loc[1, "region"] == "unknown"


def test_split_keeps_sizes_and_class_balance(synthetic_df):
    X, y = prepare_data(synthetic_df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 240 and len(X_test) == 60      # 80 / 20
    assert abs(y_train.mean() - y_test.mean()) < 0.05      # stratify сохранил долю оттока


def test_split_is_reproducible(synthetic_df):
    X, y = prepare_data(synthetic_df)
    first = split_data(X, y)[1].index.tolist()
    second = split_data(X, y)[1].index.tolist()
    assert first == second                                  # random_state зафиксирован


def test_features_to_dataframe_uses_training_column_order(client_example):
    reversed_order = dict(reversed(list(client_example.items())))
    X = features_to_dataframe([FeatureVectorChurn(**reversed_order)])
    assert list(X.columns) == FEATURE_COLUMNS
