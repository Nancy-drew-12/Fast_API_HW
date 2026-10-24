import pytest

from model import evaluate_model, load_churn_model, predict_churn, save_churn_model, train_churn_model
from preprocessing import prepare_data, split_data


@pytest.fixture
def splits(synthetic_df):
    X, y = prepare_data(synthetic_df)
    return split_data(X, y)


@pytest.mark.parametrize("model_type", ["logreg", "random_forest"])
def test_train_returns_pipeline_and_params(splits, model_type):
    X_train, X_test, y_train, y_test = splits
    pipeline, params = train_churn_model(X_train, y_train, model_type)
    assert hasattr(pipeline, "predict_proba")
    assert params["class_weight"] == "balanced"            # дефолты подставились


def test_user_hyperparameters_override_defaults(splits):
    X_train, _, y_train, _ = splits
    _, params = train_churn_model(X_train, y_train, "logreg", {"C": 0.1, "max_iter": 500})
    assert params["C"] == 0.1
    assert params["max_iter"] == 500


def test_invalid_hyperparameter_raises(splits):
    X_train, _, y_train, _ = splits
    with pytest.raises(TypeError):
        train_churn_model(X_train, y_train, "random_forest", {"abc": 1})


def test_metrics_are_in_valid_range(splits):
    X_train, X_test, y_train, y_test = splits
    pipeline, _ = train_churn_model(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())
    assert metrics["roc_auc"] > 0.7                        # на синтетике зависимость есть — модель её находит


def test_training_is_reproducible(splits):
    X_train, X_test, y_train, y_test = splits
    first = evaluate_model(train_churn_model(X_train, y_train, "random_forest")[0], X_test, y_test)
    second = evaluate_model(train_churn_model(X_train, y_train, "random_forest")[0], X_test, y_test)
    assert first == second


def test_predict_probabilities_sum_to_one(splits):
    X_train, X_test, y_train, _ = splits
    pipeline, _ = train_churn_model(X_train, y_train)
    for prediction in predict_churn(pipeline, X_test.head(10)):
        assert prediction["churn"] in (0, 1)
        assert prediction["probability_churn"] + prediction["probability_no_churn"] == pytest.approx(1, abs=0.002)


def test_save_and_load_roundtrip(splits, tmp_path):
    X_train, X_test, y_train, y_test = splits
    pipeline, params = train_churn_model(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test)
    path = tmp_path / "model.joblib"

    save_churn_model(pipeline, metrics, "logreg", params, path=path)
    bundle = load_churn_model(path)

    assert bundle["model_type"] == "logreg"
    assert bundle["metrics"] == metrics
    assert (bundle["pipeline"].predict(X_test) == pipeline.predict(X_test)).all()


def test_load_returns_none_when_no_file(tmp_path):
    assert load_churn_model(tmp_path / "missing.joblib") is None
