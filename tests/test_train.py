import numpy as np
import pytest

from src.train import apply_smote, get_models


@pytest.fixture
def imbalanced_data():
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.array([0] * 990 + [1] * 10)
    return X, y


def test_smote_balances_classes(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_smote(X, y)
    unique, counts = np.unique(y_res, return_counts=True)
    assert counts[0] == counts[1]


def test_smote_preserves_features(imbalanced_data):
    X, y = imbalanced_data
    X_res, y_res = apply_smote(X, y)
    assert X_res.shape[1] == X.shape[1]


def test_get_models_returns_at_least_two():
    models = get_models()
    assert len(models) >= 2
    assert "logistic_regression" in models
    assert "random_forest" in models
