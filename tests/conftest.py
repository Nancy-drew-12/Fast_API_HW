import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api.examples import CLIENT_EXAMPLE
from app.core import config
from app.core.state import state
from app.main import app


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Маленький синтетический датасет с фиксированным seed — всегда одинаковый."""
    rng = np.random.default_rng(42)
    n = 300
    usage_hours = rng.uniform(0, 40, n).round(1)
    failed_payments = rng.integers(0, 4, n)
    churn = ((usage_hours < 15) | (failed_payments >= 2)).astype(int)
    return pd.DataFrame({
        "monthly_fee": rng.uniform(5, 50, n).round(2),
        "usage_hours": usage_hours,
        "support_requests": rng.integers(0, 5, n),
        "account_age_months": rng.integers(1, 36, n),
        "failed_payments": failed_payments,
        "region": rng.choice(["europe", "america", "asia", "africa"], n),
        "device_type": rng.choice(["mobile", "desktop", "tablet"], n),
        "payment_method": rng.choice(["card", "paypal", "crypto"], n),
        "autopay_enabled": rng.integers(0, 2, n),
        "churn": churn,
    })


@pytest.fixture
def client(synthetic_df, tmp_path, monkeypatch):
    """TestClient на синтетических данных. Модель и история пишутся во временную папку,
    поэтому тесты не трогают настоящие файлы в models/."""
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "churn_model.joblib")
    monkeypatch.setattr(config, "HISTORY_PATH", tmp_path / "training_history.json")
    monkeypatch.setattr(state, "dataset", synthetic_df)
    monkeypatch.setattr(state, "model_bundle", None)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client_example() -> dict:
    return dict(CLIENT_EXAMPLE)
