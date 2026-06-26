"""Inference pipeline for fraud detection."""

from pathlib import Path

import joblib
import numpy as np


class FraudPredictor:
    def __init__(self, model_path: str | Path):
        self.model = joblib.load(model_path)

    def predict(self, features: list[float]) -> dict:
        X = np.array(features).reshape(1, -1)
        prob = self.model.predict_proba(X)[0]
        prediction = int(self.model.predict(X)[0])

        return {
            "prediction": prediction,
            "is_fraud": bool(prediction == 1),
            "fraud_probability": float(prob[1]),
            "legitimate_probability": float(prob[0]),
        }

    def predict_batch(self, features_list: list[list[float]]) -> list[dict]:
        X = np.array(features_list)
        probs = self.model.predict_proba(X)
        preds = self.model.predict(X)

        return [
            {
                "prediction": int(pred),
                "is_fraud": bool(pred == 1),
                "fraud_probability": float(prob[1]),
                "legitimate_probability": float(prob[0]),
            }
            for pred, prob in zip(preds, probs)
        ]
