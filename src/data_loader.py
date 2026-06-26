"""Data loading and validation for credit card fraud detection."""

from pathlib import Path

import pandas as pd


def load_data(filepath: str | Path) -> pd.DataFrame:
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    required_columns = {"Time", "Amount", "Class"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    v_columns = [f"V{i}" for i in range(1, 29)]
    missing_v = set(v_columns) - set(df.columns)
    if missing_v:
        raise ValueError(f"Missing PCA feature columns: {missing_v}")

    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class column must contain only 0 and 1")


def get_summary(df: pd.DataFrame) -> dict:
    fraud_count = int(df["Class"].sum())
    total = len(df)
    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legitimate_count": total - fraud_count,
        "fraud_percentage": round(fraud_count / total * 100, 4),
        "features": len(df.columns) - 1,
    }
