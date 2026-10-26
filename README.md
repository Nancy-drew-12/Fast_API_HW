# Churn Prediction Service

A REST API built with **FastAPI** and **scikit-learn** that trains a customer churn model and predicts the probability that a client will leave.

The service lets you:

- look at the training dataset;
- train a model (logistic regression or random forest) with your own hyperparameters;
- get predictions for one client or a batch of clients;
- compare metrics across training runs.

It also has a health check, a single error format, logging, tests and a Docker image.

## Tech stack

Python 3.13 · FastAPI · Pydantic v2 · scikit-learn · pandas · joblib · pytest · Docker

## Project structure

```
Fast_API_HW/
├── app/
│   ├── main.py              # creates the FastAPI app, connects routers, loads data on startup
│   ├── api/                 # HTTP layer: endpoints only
│   │   ├── health.py        #   GET /, GET /health
│   │   ├── dataset.py       #   GET /dataset/*
│   │   ├── model.py         #   POST /model/train, GET /model/status|schema|metrics
│   │   ├── predict.py       #   POST /predict
│   │   ├── deps.py          #   shared checks: dataset loaded? model trained?
│   │   └── examples.py      #   request/response examples for Swagger
│   ├── core/                # settings and infrastructure
│   │   ├── config.py        #   file paths, feature lists, split settings
│   │   ├── errors.py        #   ServiceError + global exception handlers
│   │   ├── logging.py       #   log format
│   │   └── state.py         #   dataset and model kept in memory
│   ├── ml/                  # ML logic, independent of FastAPI
│   │   ├── dataset.py       #   load CSV, dataset summary
│   │   ├── preprocessing.py #   X/y split, missing values, train/test split
│   │   ├── model.py         #   sklearn Pipeline, training, metrics, save/load
│   │   └── history.py       #   training history in JSON
│   └── schemas/
│       └── churn.py         # Pydantic models for requests and responses
├── data/churn_dataset.csv
├── models/                  # trained model and history (created automatically, not in git)
├── tests/                   # pytest: preprocessing, model, API
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

The layers depend in one direction only: `api` → `ml` → `core`. The ML code never imports FastAPI, so you can reuse it in a notebook or a batch job.

## Dataset: `data/churn_dataset.csv`

The file has one row per client: 2,000 rows and 10 columns.

| Column | Type | Description | Values |
|---|---|---|---|
| `monthly_fee` | float | Monthly subscription fee | 5 – 50 |
| `usage_hours` | float | Hours of service usage per month | 0 – 40 |
| `support_requests` | int | Number of support tickets | 0 – 4 |
| `account_age_months` | int | How long the client has been with us | 1 – 35 |
| `failed_payments` | int | Number of failed payments | 0 – 2 |
| `region` | str | Client region | `europe`, `america`, `asia`, `africa` |
| `device_type` | str | Main device | `mobile`, `desktop`, `tablet` |
| `payment_method` | str | Payment method | `card`, `paypal`, `crypto` |
| `autopay_enabled` | int | Autopay is on | 0 / 1 |
| **`churn`** | int | **Target**: the client left | 0 / 1 (about 20% are 1) |

**Preprocessing**

- Rows without a target are dropped.
- Numeric gaps are filled with the median and the features are scaled with `StandardScaler`.
- Categorical gaps are filled with `"unknown"` and the features go through `OneHotEncoder(handle_unknown="ignore")`. A category the model has never seen does not break a prediction.
- The data is split 80/20 with stratification by `churn` and `random_state=42`.
- The classes are imbalanced, so both models use `class_weight="balanced"`.

## Run locally

```bash
git clone https://github.com/Nancy-drew-12/Fast_API_HW.git
cd Fast_API_HW

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000
- Interactive docs (Swagger): http://127.0.0.1:8000/docs

## Run in Docker

```bash
docker build -t churn-service .
docker run -p 8001:8000 churn-service
```

The service is then available at http://127.0.0.1:8001/docs.

The image does not contain a trained model, because `models/` is excluded by `.dockerignore`. A fresh container therefore starts with `"status": "degraded"`. Call `POST /model/train` once and the status becomes `"ok"`.

## Run tests

```bash
python -m pytest -v
```

The tests use a small synthetic dataset and write the model into a temporary folder, so they never touch the real files in `models/`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | The service is running |
| GET | `/health` | Status: dataset loaded, model loaded, when it was trained |
| GET | `/dataset/preview?n=5` | First `n` rows of the dataset (1–100) |
| GET | `/dataset/info` | Size, list of features, class balance |
| GET | `/dataset/split-info` | Train/test sizes and churn share in each |
| POST | `/model/train` | Train and save a model |
| GET | `/model/status` | Current model: type, hyperparameters, metrics |
| GET | `/model/schema` | Expected features and the category values the model knows |
| GET | `/model/metrics?limit=5&model_type=` | Training history and the best run by F1 and ROC-AUC |
| POST | `/predict` | Predict churn for one client or a list |

## Request examples

### Train a model — `POST /model/train`

Train with the default settings (logistic regression):

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "logreg", "hyperparameters": {}}'
```

Train a random forest with your own hyperparameters:

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 300, "max_depth": 6}}'
```

Response:

```json
{
  "status": "model trained and saved",
  "model_type": "logreg",
  "hyperparameters": {"max_iter": 1000, "class_weight": "balanced"},
  "train_size": 1600,
  "test_size": 400,
  "trained_at": "2026-10-26T15:52:49",
  "metrics": {"accuracy": 0.49, "precision": 0.175, "recall": 0.407, "f1": 0.244, "roc_auc": 0.469}
}
```

- `model_type` must be `logreg` or `random_forest`.
- Your `hyperparameters` are merged on top of the defaults.
- The model is saved to `models/churn_model.joblib`, and every run is appended to `models/training_history.json`.

> The metrics above are modest. ROC-AUC is close to 0.5 because the features in this course dataset carry little signal. The focus of the project is the service around the model.

### Predict — `POST /predict`

Predict for one client:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "monthly_fee": 25.5, "usage_hours": 12, "support_requests": 2,
        "account_age_months": 8, "failed_payments": 1, "region": "europe",
        "device_type": "tablet", "payment_method": "card", "autopay_enabled": 0
      }'
```

```json
{"churn": 1, "probability_no_churn": 0.492, "probability_churn": 0.508}
```

Predict for several clients by sending a list:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '[
        {"monthly_fee": 25.5, "usage_hours": 12, "support_requests": 2, "account_age_months": 8,
         "failed_payments": 1, "region": "europe", "device_type": "tablet",
         "payment_method": "card", "autopay_enabled": 0},
        {"monthly_fee": 25.5, "usage_hours": 35, "support_requests": 2, "account_age_months": 30,
         "failed_payments": 0, "region": "europe", "device_type": "tablet",
         "payment_method": "card", "autopay_enabled": 1}
      ]'
```

```json
[
  {"churn": 1, "probability_no_churn": 0.492, "probability_churn": 0.508},
  {"churn": 0, "probability_no_churn": 0.563, "probability_churn": 0.437}
]
```

## Errors

Every error has the same shape:

```json
{"code": "model_not_trained", "message": "Модель ещё не обучена. Сначала вызовите POST /model/train", "details": null}
```

| HTTP | `code` | When |
|---|---|---|
| 400 | `model_not_trained` | `/predict` is called before training |
| 400 | `dataset_empty` / `dataset_invalid` | The dataset is missing, empty or lacks columns |
| 400 | `invalid_hyperparameters` | sklearn rejected a hyperparameter |
| 400 | `no_training_history` | `/model/metrics` is called before any training |
| 404 | `not_found` | Unknown URL |
| 422 | `validation_error` | Wrong type, missing feature or extra feature; `details` names the field |
| 500 | `prediction_failed` / `internal_error` | Unexpected error; the traceback goes to the log only |
