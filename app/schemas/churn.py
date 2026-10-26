from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ModelType = Literal["logreg", "random_forest"]


class FeatureVectorChurn(BaseModel):
    """Признаки одного клиента. Лишние поля — ошибка 422, а не тихое игнорирование."""
    model_config = ConfigDict(extra="forbid")

    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: int
    probability_no_churn: float
    probability_churn: float


class TrainingConfigChurn(BaseModel):
    model_type: ModelType = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
