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

from app.core import config

MODEL_CLASSES = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

DEFAULT_HYPERPARAMETERS = {
    "logreg": {"max_iter": 1000, "class_weight": "balanced"},
    "random_forest": {"n_estimators": 200, "class_weight": "balanced", "random_state": config.RANDOM_STATE},
}


def create_model(model_type: str, hyperparameters: dict):
    """Параметры по умолчанию + то, что прислал пользователь (его значения важнее)."""
    params = {**DEFAULT_HYPERPARAMETERS[model_type], **hyperparameters}
    return MODEL_CLASSES[model_type](**params), params


def build_pipeline(classifier) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "logreg",
    hyperparameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    classifier, params = create_model(model_type, hyperparameters or {})
    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)
    return pipeline, params


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
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
    path: Path | None = None,
) -> dict:
    """Сохраняет модель вместе с типом, гиперпараметрами, датой обучения и метриками."""
    bundle = {
        "pipeline": pipeline,
        "model_type": model_type,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    path = path or config.MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_churn_model(path: Path | None = None) -> dict | None:
    path = path or config.MODEL_PATH
    if not path.exists():
        return None
    return joblib.load(path)


def predict_churn(pipeline: Pipeline, X: pd.DataFrame) -> list[dict]:
    classes = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
    return [
        {
            "churn": int(cls),
            "probability_no_churn": round(float(p[0]), 3),
            "probability_churn": round(float(p[1]), 3),
        }
        for cls, p in zip(classes, probas)
    ]


def known_categories(pipeline: Pipeline) -> dict[str, list[str]]:
    """Значения категориальных признаков, которые модель видела при обучении."""
    encoder = (pipeline.named_steps["preprocessor"]
               .named_transformers_["cat"]
               .named_steps["onehot"])
    return {
        column: [str(value) for value in categories]
        for column, categories in zip(config.CATEGORICAL_FEATURES, encoder.categories_)
    }
