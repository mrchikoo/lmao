import numpy as np
import pandas as pd
import pytest

from src.preprocessing import preprocess, _engineer_features, get_feature_names


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    data = {f"V{i}": np.random.randn(n) for i in range(1, 29)}
    data["Time"] = np.random.uniform(0, 172800, n)
    data["Amount"] = np.random.exponential(100, n)
    data["Class"] = np.array([0] * 190 + [1] * 10)
    return pd.DataFrame(data)


def test_engineer_features_adds_columns(sample_df):
    result = _engineer_features(sample_df)
    assert "Log_Amount" in result.columns
    assert "Hour" in result.columns
    assert "Is_Night" in result.columns
    assert "Time" not in result.columns
    assert "Amount" not in result.columns


def test_preprocess_shapes(sample_df):
    X_train, X_test, y_train, y_test = preprocess(sample_df, test_size=0.2)
    assert X_train.shape[0] == 160
    assert X_test.shape[0] == 40
    assert X_train.shape[1] == X_test.shape[1]
    assert len(y_train) == 160
    assert len(y_test) == 40


def test_preprocess_stratification(sample_df):
    _, _, y_train, y_test = preprocess(sample_df, test_size=0.2)
    train_ratio = y_train.sum() / len(y_train)
    test_ratio = y_test.sum() / len(y_test)
    assert abs(train_ratio - test_ratio) < 0.05


def test_get_feature_names(sample_df):
    names = get_feature_names(sample_df)
    assert "Class" not in names
    assert "Log_Amount" in names
