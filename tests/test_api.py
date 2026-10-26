import pandas as pd

import main


# ---------- полный сценарий ----------

def test_root(client):
    assert client.get("/").json() == {"message": "ml churn service is running"}


def test_health_before_and_after_training(client):
    before = client.get("/health").json()
    assert before["dataset_loaded"] is True
    assert before["model_loaded"] is False
    assert before["status"] == "degraded"

    client.post("/model/train", json={})
    after = client.get("/health").json()
    assert after["model_loaded"] is True
    assert after["status"] == "ok"


def test_dataset_info_on_synthetic_data(client):
    info = client.get("/dataset/info").json()
    assert info["n_rows"] == 300
    assert info["n_columns"] == 10


def test_full_scenario_train_status_predict(client, client_example):
    # 1. обучение
    response = client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {}})
    assert response.status_code == 200
    body = response.json()
    assert body["train_size"] == 240 and body["test_size"] == 60
    assert "roc_auc" in body["metrics"]

    # 2. статус
    status = client.get("/model/status").json()
    assert status["is_trained"] is True
    assert status["model_type"] == "logreg"
    assert status["metrics"] == body["metrics"]

    # 3. предсказание для одного клиента
    prediction = client.post("/predict", json=client_example).json()
    assert prediction["churn"] in (0, 1)
    assert 0 <= prediction["probability_churn"] <= 1

    # 4. предсказание для списка
    predictions = client.post("/predict", json=[client_example, client_example]).json()
    assert isinstance(predictions, list) and len(predictions) == 2


def test_model_reacts_to_risky_and_loyal_clients(client, client_example):
    client.post("/model/train", json={})
    risky = {**client_example, "usage_hours": 2, "failed_payments": 3}
    loyal = {**client_example, "usage_hours": 38, "failed_payments": 0}
    p_risky = client.post("/predict", json=risky).json()["probability_churn"]
    p_loyal = client.post("/predict", json=loyal).json()["probability_churn"]
    assert p_risky > p_loyal


def test_metrics_history_and_filter(client):
    client.post("/model/train", json={"model_type": "logreg"})
    client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"max_depth": 4}})

    metrics = client.get("/model/metrics").json()
    assert metrics["total_trainings"] == 2
    assert metrics["latest"]["model_type"] == "random_forest"

    only_logreg = client.get("/model/metrics", params={"model_type": "logreg"}).json()
    assert [r["model_type"] for r in only_logreg["history"]] == ["logreg"]


def test_schema_lists_all_features(client):
    schema = client.get("/model/schema").json()
    assert schema["n_features"] == 9


# ---------- обработка ошибок ----------

def test_predict_without_model_returns_400(client, client_example):
    response = client.post("/predict", json=client_example)
    assert response.status_code == 400
    assert response.json()["code"] == "model_not_trained"


def test_metrics_without_history_returns_400(client):
    response = client.get("/model/metrics")
    assert response.status_code == 400
    assert response.json()["code"] == "no_training_history"


def test_predict_with_wrong_type_returns_422(client, client_example):
    client.post("/model/train", json={})
    response = client.post("/predict", json={**client_example, "usage_hours": "много"})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["details"][0]["field"] == "usage_hours"


def test_predict_with_missing_feature_returns_422(client, client_example):
    client.post("/model/train", json={})
    client_example.pop("region")
    response = client.post("/predict", json=client_example)
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "region"


def test_predict_with_extra_feature_returns_422(client, client_example):
    client.post("/model/train", json={})
    response = client.post("/predict", json={**client_example, "age": 30})
    assert response.status_code == 422
    assert response.json()["details"][0]["field"] == "age"


def test_train_with_unknown_model_returns_422(client):
    response = client.post("/model/train", json={"model_type": "xgboost"})
    assert response.status_code == 422


def test_train_with_invalid_hyperparameters_returns_400(client):
    response = client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"abc": 1}})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_hyperparameters"


def test_train_on_empty_dataset_returns_400(client, monkeypatch):
    monkeypatch.setattr(main, "df", pd.DataFrame())
    response = client.post("/model/train", json={})
    assert response.status_code == 400
    assert response.json()["code"] == "dataset_empty"


def test_train_on_dataset_without_column_returns_400(client, monkeypatch, synthetic_df):
    monkeypatch.setattr(main, "df", synthetic_df.drop(columns=["region"]))
    response = client.post("/model/train", json={})
    assert response.status_code == 400
    assert response.json()["details"]["missing_columns"] == ["region"]


def test_unknown_url_returns_404_in_our_format(client):
    response = client.get("/abc")
    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
