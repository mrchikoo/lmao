<div align="center">

# Credit Card Fraud Detection

**End-to-end ML pipeline that trains imbalance-aware fraud classifiers and serves them through a FastAPI REST API.**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7)
![LightGBM](https://img.shields.io/badge/LightGBM-02A04A)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

</div>

## Overview

Fraud is rare: in the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset only 492 of 284,807 transactions (0.17%) are fraudulent. This project loads and validates that data, engineers a few features, balances the training set with SMOTE, compares four models using precision-recall metrics rather than accuracy, and exposes the trained XGBoost model as a REST service.

## Features

- **Data validation**: checks for the `Time`, `Amount`, `Class` and `V1`-`V28` columns and a binary `Class` label.
- **Feature engineering**: `Log_Amount` (log1p of `Amount`), `Hour` and `Is_Night` (before 06:00 or from 22:00) derived from `Time`; `Time` and `Amount` are dropped. Features are scaled with `RobustScaler`.
- **Imbalance handling**: SMOTE applied to the training split only (stratified 80/20 split).
- **Model comparison**: Logistic Regression, Random Forest, XGBoost and LightGBM, scored with 5-fold stratified cross-validation on PR-AUC (`average_precision`).
- **Evaluation**: PR-AUC, F1, precision, recall and confusion matrices on the held-out test set, plus precision-recall curve and confusion-matrix plots.
- **Serving**: FastAPI app with single and batch prediction endpoints and a health check.
- **Docker** image for the API.

## Tech Stack

Python 3.11, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, FastAPI, Uvicorn, Pydantic v2, matplotlib, seaborn, joblib, pytest, httpx.

## Getting Started

### 1. Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Download `creditcard.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/` (the file is git-ignored).

### 3. Train and evaluate

```bash
python -m src.train --data data/creditcard.csv --output models/
```

Options: `--output` (default `models/`) and `--cv-folds` (default `5`). Training also runs the test-set evaluation. Outputs in `models/`:

- `<model>.joblib` for each model (`logistic_regression`, `random_forest`, `xgboost`, `lightgbm`)
- `metrics.json` (cross-validation PR-AUC mean and std)
- `test_metrics.json`, `pr_curves.png`, `confusion_matrices.png`

### 4. Run the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The API loads `models/xgboost.joblib` at startup. If the file is missing, the server still starts, `/health` reports `model_loaded: false`, and prediction endpoints return `503`.

### 5. Run the tests

```bash
pytest
```

### Docker

The image copies the `models/` directory, so train first, then build:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "healthy", "model_loaded": <bool>}` |
| POST | `/predict` | Score one transaction. Body: `{"features": [31 floats]}` |
| POST | `/predict/batch` | Score many. Body: `{"instances": [[31 floats], ...]}` |

Requests with a feature count other than 31 are rejected with `422`. The response contains `prediction`, `is_fraud`, `fraud_probability` and `legitimate_probability`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.5, ...]}'
```

Interactive docs are available at `http://localhost:8000/docs`.

> **Note:** the 31 features are `V1`-`V28` followed by `Log_Amount`, `Hour`, `Is_Night`, in the already-scaled form used for training. The fitted scaler is not saved by the training script, so the API expects pre-processed input.

## Results

Reference results from the author's run on the held-out test set (SMOTE applied to training data only). Exact numbers will vary with library versions and environment; `models/test_metrics.json` holds the values from your own run.

| Model               | PR-AUC | F1 Score | Precision | Recall |
|---------------------|--------|----------|-----------|--------|
| Logistic Regression | 0.72   | 0.71     | 0.83      | 0.62   |
| Random Forest       | 0.84   | 0.83     | 0.91      | 0.76   |
| XGBoost             | 0.86   | 0.84     | 0.93      | 0.77   |
| LightGBM            | 0.85   | 0.83     | 0.92      | 0.76   |

## Project Structure

```
├── src/
│   ├── data_loader.py       # Data ingestion and schema validation
│   ├── preprocessing.py     # Feature engineering, split and scaling
│   ├── train.py             # Model training with SMOTE and cross-validation
│   ├── evaluate.py          # Metrics and comparison plots
│   ├── predict.py           # Inference wrapper (FraudPredictor)
│   └── api.py               # FastAPI application
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_api.py
├── data/                    # Place creditcard.csv here
├── Dockerfile
├── requirements.txt
└── README.md
```
