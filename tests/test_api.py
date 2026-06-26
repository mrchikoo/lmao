import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_no_model(client):
    response = client.post("/predict", json={"features": [0.0] * 31})
    assert response.status_code == 503


def test_predict_invalid_features(client):
    response = client.post("/predict", json={"features": [0.0] * 5})
    assert response.status_code == 422
