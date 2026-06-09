from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

# 1. SETUP

app = FastAPI(
    title="Churn Prediction API",
    description="Microservice for scoring customer retention risk.",
    version="1.0.0"
)

# Load the model at startup
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    model = joblib.load(model_path)
except Exception as e:
    model = None
    print(f"⚠️ Warning: Model not found at {model_path}. Run train_model.py first.")

# 2. INPUT VALIDATION (Pydantic)

class CustomerInput(BaseModel):
    customer_id: str = Field(..., json_schema_extra={"example": "CUST_001"})
    recency: int = Field(..., gt=0, description="Days since last order")
    frequency: int = Field(..., ge=0, description="Total number of orders")
    monetary: float = Field(..., ge=0.0, description="Total lifetime spend")

class PredictionOutput(BaseModel):
    customer_id: str
    churn_probability: float
    risk_level: str
    risk_explanation: str

# 3. HELPER FUNCTIONS

def get_risk_level(prob):
    if prob < 0.3: return "Low"
    if prob < 0.7: return "Medium"
    return "High"

def explain_risk(features, prob):
    reasons = []
    if features['recency'] > 90:
        reasons.append("High inactivity duration")
    if features['frequency'] < 2:
        reasons.append("Low purchase frequency")
    
    if not reasons and prob > 0.5:
        return "Complex behavioral pattern detected"
    elif not reasons:
        return "Customer is active and healthy"
        
    return "; ".join(reasons)

# 4. ENDPOINTS


@app.get("/health")
def health_check():
    """Returns the service status."""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "service": "churn-predictor"}

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(input_data: CustomerInput):
    """Predicts churn risk for a single customer."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Isolate the exact dictionary data payload
    data_dict = input_data.model_dump()

    # 2. Extract ONLY the numeric features the model pipeline expects (drop customer_id)
    features = pd.DataFrame([{
        "recency": data_dict["recency"],
        "frequency": data_dict["frequency"],
        "monetary": data_dict["monetary"]
    }])

    # 3. Calculate predictions cleanly using the filtered DataFrame
    prob = model.predict_proba(features)[0][1]  # Probability of Churn (Class 1)
    risk_lvl = get_risk_level(prob)

    # 4. Return matching fields aligned with your PredictionOutput schema
    return {
        "customer_id": input_data.customer_id,
        "churn_probability": float(prob),
        "risk_level": risk_lvl,
        "risk_explanation": explain_risk(data_dict, prob),
    }



@app.post("/batch_predict")
def batch_predict(inputs: list[CustomerInput]):
    """Handles multiple customers at once."""
    results = []
    for customer in inputs:
        # Re-use the single prediction logic
        # (In production, we would vectorize this for speed)
        res = predict_churn(customer)
        results.append(res)
    return {"batch_results": results}
