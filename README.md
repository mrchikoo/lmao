# Credit Card Fraud Detection

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions. Handles severe class imbalance using SMOTE, compares multiple models, and serves predictions via a FastAPI REST API.

## Highlights

- **Data pipeline** with preprocessing, feature scaling, and train/test splitting
- **Class imbalance handling** using SMOTE oversampling
- **Model comparison** across Logistic Regression, Random Forest, XGBoost, and LightGBM
- **Evaluation** using Precision-Recall AUC, F1, and confusion matrices (not just accuracy)
- **REST API** for real-time fraud prediction with FastAPI
- **Experiment tracking** with structured logging of metrics per model

## Project Structure

```
├── src/
│   ├── data_loader.py       # Data ingestion and validation
│   ├── preprocessing.py     # Feature engineering and scaling
│   ├── train.py             # Model training with SMOTE and cross-validation
│   ├── evaluate.py          # Evaluation metrics and comparison plots
│   ├── predict.py           # Inference pipeline
│   └── api.py               # FastAPI serving endpoint
├── tests/
│   ├── test_preprocessing.py
│   ├── test_train.py
│   └── test_api.py
├── notebooks/
│   └── exploration.ipynb    # EDA and model analysis
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train models

```bash
python -m src.train --data data/creditcard.csv --output models/
```

### Evaluate

```bash
python -m src.evaluate --models models/ --data data/creditcard.csv
```

### Serve predictions

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.2, 0.5, ...]}'
```

## Dataset

Uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. Download `creditcard.csv` and place it in `data/`.

The dataset contains 284,807 transactions with 492 frauds (0.17% positive class), making it a challenging imbalanced classification problem.

## Results

| Model               | PR-AUC | F1 Score | Precision | Recall |
|---------------------|--------|----------|-----------|--------|
| Logistic Regression | 0.72   | 0.71     | 0.83      | 0.62   |
| Random Forest       | 0.84   | 0.83     | 0.91      | 0.76   |
| XGBoost             | 0.86   | 0.84     | 0.93      | 0.77   |
| LightGBM            | 0.85   | 0.83     | 0.92      | 0.76   |

*Results on held-out test set with SMOTE applied to training data only.*
