# Architecture

This document describes how the credit card fraud detection system is organized and why each main decision was made. It is kept short on purpose and will be updated as the project evolves.

## Overview

The system trains a classifier on the Kaggle Credit Card Fraud dataset, scores new transactions, stores every transaction and prediction in PostgreSQL, and shows the results in a Streamlit dashboard.

There are two ways to score transactions:

- **Real-time:** a FastAPI endpoint receives one transaction and returns a decision, a fraud probability, and an explanation.
- **Batch:** a command processes a CSV file with many transactions and stores the results.

Both entry points use the same core package, so the preprocessing and scoring logic exists in one place only.

## Data flow

```
                       TRAINING
  creditcard.csv ──> core: preprocess ──> core: train ──> core: evaluate
                                                              │
                                     ┌────────────────────────┤
                                     v                        v
                              model file (joblib)     MLflow (runs, metrics,
                              + threshold                model versions)
                                     │                        │
                                     │                        v
                                     │            PostgreSQL: experiments
                                     v
                       SCORING
  API request (1 transaction) ──┐
                                ├──> core: preprocess ──> model ──> threshold ──> SHAP explanation
  Batch CSV (many rows) ────────┘                                                     │
                                                                                      v
                                        PostgreSQL: transactions + predictions
                                                              │
                                                              v
                                                   Streamlit dashboard
```

## Project structure

```
credit-card-fraud-detection/
├── src/fraud_detection/   # core package: preprocessing, training, evaluation,
│                          # threshold selection, scoring, explanation, database access
├── api/                   # thin FastAPI app that calls the core package
├── dashboard/             # thin Streamlit app that reads from PostgreSQL
├── scripts/               # entry points such as training and batch scoring
├── tests/                 # pytest tests
├── docs/                  # documentation, including this file
├── Dockerfile
└── docker-compose.yml     # api, dashboard, postgres, mlflow
```

The `api/` and `dashboard/` folders contain no business logic. They only receive input, call the core package, and return or display the result.

## Database

| Table | Purpose | Main columns |
|---|---|---|
| `transactions` | Raw transaction history | id, time, v1 to v28, amount, class (real label, when known) |
| `predictions` | One prediction per transaction per model version | id, transaction_id (FK to transactions), model_version, probability, decision, threshold, explanation, created_at |
| `experiments` | Summary of each training run | id, model, parameters, pr_auc, recall, precision, f1, created_at |

`predictions` has a unique constraint on `(transaction_id, model_version)`, so the same model cannot score the same transaction twice, but a new model version can score it again for comparison.

## API

- `POST /predict` receives one transaction and returns `decision`, `probability`, `threshold`, `model_version`, and `explanation`.
- `GET /health` reports whether the service and the model are ready.

Error handling:

- **422** for invalid client input (FastAPI and Pydantic default validation).
- **5xx** for server faults, such as the model not loaded or the database unavailable.
- All errors use the same body format:

```json
{
  "error": "short error name",
  "details": "human readable description"
}
```

## Decisions

| Decision | Reason |
|---|---|
| Kaggle Credit Card Fraud dataset | Public, well known, and realistic class imbalance (about 0.17% fraud in about 285k transactions). |
| One core package with thin `api/` and `dashboard/` | Avoids duplicated logic between real-time and batch scoring, and makes the core easy to test. |
| Real-time API and batch CSV scoring | Covers the two common ways fraud models are used in practice. |
| PostgreSQL for transactions and predictions | Keeps a history that the dashboard can query and allows comparing model versions on the same data. |
| Model saved with joblib | Simplest way to persist and load a scikit-learn compatible model. |
| MLflow from the first version | Tracks parameters and metrics of every run and versions the models, so results are reproducible. |
| PR-AUC, recall, precision, F1 as metrics | Accuracy is misleading with 0.17% fraud; these metrics focus on the minority class. |
| Threshold chosen by a cost function | The default 0.5 threshold ignores business impact. The threshold minimizes the total cost of missed fraud plus blocked legitimate customers. |
| SHAP explanations per transaction | A blocked customer or an analyst needs to know why a transaction was flagged. |
| Same error body for 422 and 5xx | Clients handle every error the same way. |
| Docker and docker compose | Anyone can run the full system with one command. |
| pytest and GitHub Actions CI | Every change is tested automatically before it is merged. |

## Open items

- **TODO:** Define the cost values for a missed fraud and for a blocked legitimate transaction.
- **TODO:** Define the exact model candidates to compare (for example, a linear baseline and a gradient boosting model).
- **TODO:** Define the SHAP approach details: which explainer, how many top features are returned, and how the explanation is stored.
