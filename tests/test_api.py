# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.api import app
import sys
from pathlib import Path

# Find project root
current = Path(__file__).resolve().parent
while current.name != "industrial-robot-predictive-mtce":
    current = current.parent
    if current.parent == current:
        raise RuntimeError("Cannot find project root folder")

PROJECT_ROOT = current
sys.path.insert(0, str(PROJECT_ROOT))

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to RobotGuard AI API"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "predictor_loaded" in response.json()

def test_generate_data_endpoint():
    # We use sample mode to make it faster if it runs synchronously (though it's background)
    response = client.post("/generate-data?mode=sample")
    assert response.status_code == 200
    assert "started" in response.json()["status"]

def test_clean_data_endpoint():
    response = client.post("/clean-data?mode=sample")
    assert response.status_code == 200
    assert "started" in response.json()["status"]

def test_build_features_endpoint():
    response = client.post("/build-features")
    assert response.status_code == 200
    assert "started" in response.json()["status"]

def test_train_endpoint():
    response = client.post("/train?mode=sample")
    assert response.status_code == 200
    assert "started" in response.json()["status"]

def test_predict_endpoint_fail_no_model():
    # If model is not found, it should return 503 or 500
    sample_input = {
        "vibration": 0.5,
        "torque": 10.0,
        "temperature": 45.0,
        "current": 2.0,
        "joint_angle": 90.0,
        "hours_since_service": 100.0
    }
    response = client.post("/predict", json=sample_input)
    # Depending on whether models exist in /models, this might be 200 or 503
    if response.status_code == 200:
        assert "predicted_rul_hours" in response.json()
        assert "failure_probabilities" in response.json()
    else:
        assert response.status_code in [503, 500]

def test_invalid_input_predict():
    invalid_input = {
        "vibration": "not-a-number",
        "torque": 10.0
    }
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422  # Unprocessable Entity (validation error)
