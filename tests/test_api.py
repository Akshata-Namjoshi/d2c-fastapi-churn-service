from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test if the API is up and running."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "churn-predictor"}

def test_prediction_flow():
    """Test a valid prediction request."""
    payload = {
        "customer_id": "TEST_001",
        "recency": 10,
        "frequency": 5,
        "monetary": 500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert data["risk_level"] in ["Low", "Medium", "High"]

def test_invalid_input():
    """Test input validation (Recency cannot be negative)."""
    payload = {
        "customer_id": "TEST_BAD",
        "recency": -5, 
        "frequency": 5,
        "monetary": 500.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
