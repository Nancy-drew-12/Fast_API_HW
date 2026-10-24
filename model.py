from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path(__file__).parent / "models" / "churn_model.joblib"

MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

DEFAULT_HYPERPARAMETERS = {
    "logreg": {"max_iter": 1000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 200, "class_weight": "balanced", "random_state": 42},
}


def create_model(model_type: str, hyperparameters: dict):
    """Создаёт модель нужного типа: параметры по умолчанию + то, что прислал пользователь."""
    params = {**DEFAULT_HYPERPARAMETERS[model_type], **hyperparameters}
    model = MODEL_CLASSES[model_type](**params)
    return model, params


def build_pipeline(classifier) -> Pipeline:
    """Предобработка + модель в одном объекте."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logreg",
    hyperparameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Обучает pipeline выбранного типа и возвращает его вместе с итоговыми гиперпараметрами."""
    classifier, params = create_model(model_type, hyperparameters or {})
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    return pipeline, params


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Считает метрики качества на тестовой выборке."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]   # вероятность класса 1 — нужна для ROC-AUC
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "precision": round(precision_score(y_test, y_pred), 3),
        "recall": round(recall_score(y_test, y_pred), 3),
        "f1": round(f1_score(y_test, y_pred), 3),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
    }


def save_churn_model(
    pipeline: Pipeline,
    metrics: dict,
    model_type: str,
    hyperparameters: dict,
    path: Path = MODEL_PATH,
) -> dict:
    """Сохраняет модель вместе с типом, гиперпараметрами, датой обучения и метриками."""
    bundle = {
        "pipeline": pipeline,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)   # создаём папку models, если её нет
    joblib.dump(bundle, path)
    return bundle


def load_churn_model(path: Path = MODEL_PATH) -> dict | None:
    """Загружает модель с диска. Если файла нет — возвращает None."""
    if not path.exists():
        return None
    return joblib.load(path)


def predict_churn(pipeline: Pipeline, X: pd.DataFrame) -> list[dict]:
    """Класс и вероятности классов для каждого клиента."""
    classes = pipeline.predict(X)            # [0, 1, ...]
    probas = pipeline.predict_proba(X)       # [[p0, p1], [p0, p1], ...]
    return [
        {
            "churn": int(cls),
            "probability_no_churn": round(float(p[0]), 3),
            "probability_churn": round(float(p[1]), 3),
        }
        for cls, p in zip(classes, probas)
    ]