"""FastAPI endpoint for real-time fraud prediction."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.predict import FraudPredictor

MODEL_PATH = Path("models/xgboost.joblib")
predictor: FraudPredictor | None = None

EXPECTED_FEATURES = 31


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    if MODEL_PATH.exists():
        predictor = FraudPredictor(MODEL_PATH)
    else:
        logging.warning("Model file not found at %s. Train a model first.", MODEL_PATH)
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def validate_feature_length(cls, v: list[float]) -> list[float]:
        if len(v) != EXPECTED_FEATURES:
            raise ValueError(f"Expected {EXPECTED_FEATURES} features, got {len(v)}")
        return v


class BatchRequest(BaseModel):
    instances: list[list[float]]


class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    fraud_probability: float
    legitimate_probability: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.predict(request.features)


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.predict_batch(request.instances)
