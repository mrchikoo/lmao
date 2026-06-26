"""Feature engineering and preprocessing pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _engineer_features(df)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_test = _scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Hour"] = df["Time"].apply(lambda t: (t // 3600) % 24)
    df["Is_Night"] = df["Hour"].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df = df.drop(columns=["Time", "Amount"])
    return df


def _scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def get_feature_names(df: pd.DataFrame) -> list[str]:
    df = _engineer_features(df)
    return [c for c in df.columns if c != "Class"]
