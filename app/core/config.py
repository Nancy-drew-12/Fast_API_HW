"""Пути к файлам и общие настройки сервиса — всё в одном месте."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "churn_dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
HISTORY_PATH = BASE_DIR / "models" / "training_history.json"

TARGET = "churn"

NUMERIC_FEATURES = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]
CATEGORICAL_FEATURES = ["region", "device_type", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TEST_SIZE = 0.2
RANDOM_STATE = 42
