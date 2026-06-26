"""Model training with SMOTE oversampling and cross-validation."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_models() -> dict:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=100, random_state=42, eval_metric="aucpr",
            use_label_encoder=False,
        )
    except ImportError:
        logger.warning("XGBoost not installed, skipping")

    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            is_unbalance=True, random_state=42, verbose=-1,
        )
    except ImportError:
        logger.warning("LightGBM not installed, skipping")

    return models


def apply_smote(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        "SMOTE: %d -> %d samples (fraud: %d -> %d)",
        len(y_train), len(y_resampled),
        y_train.sum(), y_resampled.sum(),
    )
    return X_resampled, y_resampled


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> dict:
    X_resampled, y_resampled = apply_smote(X_train, y_train)
    models = get_models()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        scores = cross_val_score(
            model, X_resampled, y_resampled, cv=cv, scoring="average_precision", n_jobs=-1
        )
        model.fit(X_resampled, y_resampled)
        results[name] = {
            "model": model,
            "cv_pr_auc_mean": float(np.mean(scores)),
            "cv_pr_auc_std": float(np.std(scores)),
        }
        logger.info("%s CV PR-AUC: %.4f (+/- %.4f)", name, np.mean(scores), np.std(scores))

    return results


def save_models(results: dict, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for name, result in results.items():
        model_path = output_dir / f"{name}.joblib"
        joblib.dump(result["model"], model_path)
        metrics[name] = {
            "cv_pr_auc_mean": result["cv_pr_auc_mean"],
            "cv_pr_auc_std": result["cv_pr_auc_std"],
        }
        logger.info("Saved %s to %s", name, model_path)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data", required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", default="models/", help="Output directory for models")
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    from src.data_loader import load_data
    from src.preprocessing import preprocess

    df = load_data(args.data)
    X_train, X_test, y_train, y_test = preprocess(df)

    results = train_and_evaluate(X_train, y_train, cv_folds=args.cv_folds)
    save_models(results, args.output)

    from src.evaluate import evaluate_all
    evaluate_all(results, X_test, y_test, output_dir=args.output)


if __name__ == "__main__":
    main()
