# 📊 D2C Customer Churn Intelligence & Retention API

A production-ready FastAPI microservice that leverages a Machine Learning pipeline to predict customer churn risks in real-time. Built as the Capstone Project completion module, this service uses Recency, Frequency, and Monetary (RFM) behavioral patterns to score customer retention risks instantly.

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Environment Setup
Clone the repository and ensure your local virtual environment is active:
```bash
# Activate your virtual environment (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model
Generate the synthetic pipeline artifact (`model.pkl`) using the automated training script:
```bash
python train_model.py
```

### 3. Launch the API Server
Spin up the local Uvicorn development server with hot-reloading enabled:
```bash
uvicorn app.main:app --reload
```
The service will be live at: `http://127.0.0.1:8000`

### 4. Execute the Test Suite
Validate endpoint integrity, data validations, and model predictions cleanly:
```bash
pytest
```

---

## 🛠️ API Architecture & Schema Blueprint

### Data Contract Validation Models (Pydantic V2)
The service enforces strict data compliance. Non-numeric identifier columns like `customer_id` are cleanly filtered out before hitting the model array layer to prevent server pipeline crashes.

#### Request Schema (`POST /predict`)
```json
{
  "customer_id": "CUST_001",
  "recency": 120,
  "frequency": 1,
  "monetary": 50.0
}
```

#### Response Schema (`200 OK`)
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.84,
  "risk_level": "High",
  "risk_explanation": "High inactivity duration; Low purchase frequency"
}
```

---

## 🌐 Endpoints

 `GET /health`: Returns systemic availability and model instantiation status.
 `POST /predict`: Evaluates single customer profile data contracts.
 `POST /batch_predict`: Vectorizes multi-customer batch retention risks.
 `GET /docs`: Interactive visual Swagger UI suite.
