"""Примеры запросов и ответов для Swagger (/docs)."""
from typing import Any

from app.core.errors import ErrorResponse


def error_example(code: str, message: str, details: Any = None) -> dict:
    return {"code": code, "message": message, "details": details}


def error_response(description: str, example: dict | None = None, examples: dict | None = None) -> dict:
    """Описание ответа с ошибкой для параметра responses= в декораторе эндпоинта."""
    content = {"example": example} if examples is None else {"examples": examples}
    return {"model": ErrorResponse, "description": description, "content": {"application/json": content}}


NO_DATASET = error_example("dataset_empty", "Датасет не загружен или пуст")
NO_MODEL = error_example("model_not_trained", "Модель ещё не обучена. Сначала вызовите POST /model/train")
INTERNAL = error_example("internal_error", "Внутренняя ошибка сервиса. Попробуйте позже")
VALIDATION = error_example(
    "validation_error", "Неверные входные данные",
    [{"field": "usage_hours", "problem": "Input should be a valid number", "input": "много"}],
)

CLIENT_EXAMPLE = {
    "monthly_fee": 25.5,
    "usage_hours": 12,
    "support_requests": 2,
    "account_age_months": 8,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "tablet",
    "payment_method": "card",
    "autopay_enabled": 0,
}

PREDICT_EXAMPLES = {
    "one_client": {"summary": "Один клиент", "value": CLIENT_EXAMPLE},
    "several_clients": {
        "summary": "Несколько клиентов",
        "value": [
            CLIENT_EXAMPLE,
            {**CLIENT_EXAMPLE, "usage_hours": 35, "failed_payments": 0,
             "account_age_months": 30, "autopay_enabled": 1},
        ],
    },
}

TRAIN_EXAMPLES = {
    "logreg_default": {
        "summary": "Логистическая регрессия по умолчанию",
        "value": {"model_type": "logreg", "hyperparameters": {}},
    },
    "logreg_custom": {
        "summary": "Логистическая регрессия с регуляризацией C=0.1",
        "value": {"model_type": "logreg", "hyperparameters": {"C": 0.1}},
    },
    "random_forest": {
        "summary": "Случайный лес",
        "value": {"model_type": "random_forest",
                  "hyperparameters": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 5}},
    },
}

PREDICT_RESPONSES = {
    200: {
        "description": "Предсказанный класс и вероятности",
        "content": {"application/json": {"example": {
            "churn": 1, "probability_no_churn": 0.38, "probability_churn": 0.62,
        }}},
    },
    400: error_response("Модель ещё не обучена", NO_MODEL),
    422: error_response(
        "Неверные данные клиента: не тот тип, не хватает или лишний признак",
        examples={
            "wrong_type": {"summary": "Неверный тип значения", "value": VALIDATION},
            "missing_feature": {"summary": "Не хватает признака", "value": error_example(
                "validation_error", "Неверные входные данные",
                [{"field": "region", "problem": "Field required", "input": {"...": "..."}}])},
            "extra_feature": {"summary": "Лишний признак", "value": error_example(
                "validation_error", "Неверные входные данные",
                [{"field": "age", "problem": "Extra inputs are not permitted", "input": 30}])},
        },
    ),
    500: error_response("Ошибка при расчёте предсказания", error_example(
        "prediction_failed", "Не удалось получить предсказание модели")),
}

TRAIN_RESPONSES = {
    400: error_response(
        "Пустой или некорректный датасет, неверные гиперпараметры",
        examples={
            "dataset_empty": {"summary": "Пустой датасет", "value": NO_DATASET},
            "dataset_invalid": {"summary": "Не хватает колонок", "value": error_example(
                "dataset_invalid", "В датасете не хватает нужных колонок", {"missing_columns": ["region"]})},
            "bad_hyperparameters": {"summary": "Неверные гиперпараметры", "value": error_example(
                "invalid_hyperparameters", "Неверные гиперпараметры модели",
                "RandomForestClassifier.__init__() got an unexpected keyword argument 'abc'")},
        },
    ),
    422: error_response("Неизвестный тип модели или неверный формат конфигурации", error_example(
        "validation_error", "Неверные входные данные",
        [{"field": "model_type", "problem": "Input should be 'logreg' or 'random_forest'", "input": "xgboost"}])),
    500: error_response("Непредвиденная ошибка", INTERNAL),
}
