# Credit Card Fraud Detection

An end-to-end machine learning system that detects fraudulent credit card transactions. It trains and tracks models, scores transactions in real time through an API or in batch from a CSV file, explains each decision, stores the results in PostgreSQL, and shows them in a dashboard.

## Problem

Card fraud is rare but expensive. A model that misses fraud costs the bank money, and a model that blocks too many legitimate purchases frustrates customers. Because fraud is less than 1% of all transactions, a model can reach 99.8% accuracy by never predicting fraud at all, which makes accuracy a useless metric here.

This project focuses on:

- Metrics that measure performance on the rare fraud class.
- A decision threshold chosen by business cost, not by default.
- An explanation for every decision, so a flagged transaction can be reviewed.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- About 285,000 transactions made by European cardholders over two days.
- About 0.17% of transactions are fraud.
- Features `V1` to `V28` are anonymized PCA components, plus `Time`, `Amount`, and the label `Class` (1 = fraud).

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in `data/`.

## Approach

1. **Preprocessing:** scale `Time` and `Amount`, split data into train and test sets with stratification to keep the fraud ratio.
2. **Training:** compare model candidates, tracking every run in MLflow.
3. **Evaluation:** PR-AUC, recall, precision, and F1 on the fraud class.
4. **Threshold selection:** pick the threshold that minimizes the total cost of missed fraud plus blocked legitimate transactions.
5. **Explanation:** SHAP values show which features pushed each transaction toward fraud.
6. **Serving:** a FastAPI endpoint for single transactions and a batch command for CSV files.
7. **Storage and monitoring:** transactions and predictions are stored in PostgreSQL and displayed in a Streamlit dashboard.

### Architecture

```
creditcard.csv ──> preprocess ──> train ──> evaluate ──> model + threshold (MLflow)
                                                               │
API request / batch CSV ──> preprocess ──> model ──> threshold ──> SHAP
                                                               │
                                                PostgreSQL ──> Streamlit
```

See [docs/architecture.md](docs/architecture.md) for the full design and the reasons behind each decision.

## Results

> Placeholder. This section will be filled after the experiments are run.

| Model | PR-AUC | Recall | Precision | F1 | Threshold |
|---|---|---|---|---|---|
| TBD | - | - | - | - | - |

## Tech stack

Python, scikit-learn, SHAP, MLflow, FastAPI, Pydantic, PostgreSQL, Streamlit, Docker, pytest, GitHub Actions, uv.

## How to run

### Requirements

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose, or
- Python 3.12 and [uv](https://docs.astral.sh/uv/) for local development.

### With Docker

```bash
git clone https://github.com/smacedo7/credit-card-fraud-detection.git
cd credit-card-fraud-detection
docker compose up --build
```

Services:

| Service | URL |
|---|---|
| API docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
| MLflow | http://localhost:5000 |

### Locally with uv

```bash
uv sync                                   # install dependencies
uv run python scripts/train.py            # train and register a model
uv run uvicorn api.main:app --reload      # start the API
uv run streamlit run dashboard/app.py     # start the dashboard
uv run pytest                             # run the tests
```

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"time": 0, "amount": 149.62, "v1": -1.36, "v2": -0.07, "...": "..."}'
```

Example response:

```json
{
  "decision": "fraud",
  "probability": 0.87,
  "threshold": 0.35,
  "model_version": "1",
  "explanation": {"V14": -0.42, "V4": 0.31, "V12": -0.18}
}
```

## Project structure

```
credit-card-fraud-detection/
├── src/fraud_detection/   # core logic: preprocessing, training, evaluation, scoring
├── api/                   # FastAPI application
├── dashboard/             # Streamlit application
├── scripts/               # training and batch scoring entry points
├── tests/                 # pytest tests
├── docs/                  # architecture and design notes
├── data/                  # dataset (not versioned)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Author

Samuel Macedo Correia

- [LinkedIn](https://www.linkedin.com/in/samuelmacedoo/)
- [GitHub](https://github.com/smacedo7)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
