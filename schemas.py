from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FeatureVectorChurn(BaseModel):
    model_config = ConfigDict(extra="forbid")   # лишние признаки — ошибка, а не тихое игнорирование

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
    churn: int                    # предсказанный класс: 1 — уйдёт, 0 — останется
    probability_no_churn: float   # вероятность класса 0
    probability_churn: float      # вероятность класса 1


class TrainingConfigChurn(BaseModel):
    model_type: Literal["logreg", "random_forest"] = "logreg"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)