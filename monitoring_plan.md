# 📈 Operational Monitoring & Observability Framework

This blueprint details the orchestration strategy for tracking system performance, data health, and conceptual drift across production deployments of the Churn API.

---

## 🛠️ Infrastructure & System Observability

### 1. Application Performance Monitoring (APM)
 Tooling: Prometheus + Grafana
 Core Service Metrics:
   Latency: Track p50, p95, and p99 inference request windows. Target inference time: `< 50ms`.
   Throughput: Volumetric tracking via Requests Per Second (RPS).
   Error Ingestion Rate: Strict alert triggers on spike clusters of `HTTP 5xx` (Server Error) responses.

### 2. Health Probe Design
The configured `/health` micro-probe verifies deep operational health:
 Level 1: Validates ASGI web-server responsiveness.
 Level 2: Checks memory footprint stability of the active `app/model.pkl` serialization artifact.

---

## 🧠 ML-Specific Post-Deployment Observability

To prevent unannounced precision decay after deployment, data pipelines must log incoming features to analyze metric shifts over time:

### 1. Data Integrity & Validation Fails
 Metric: Type-error volume and missing schema boundaries caught by Pydantic.
 Impact: Spikes indicate upstream data mutation or corrupted payload events.

### 2. Feature Drift Assessment
 Method: Weekly evaluation of incoming `recency`, `frequency`, and `monetary` metrics against training data baseline metrics using a Kolmogorov-Smirnov (KS) test.
 Action Threshold: If feature variance deviates beyond $\alpha = 0.05$, flag a drift notification alert.

### 3. Conceptual Model Decay (Concept Drift)
 Telemetry: Log predicted probability values alongside delayed ground-truth transactional churn data over rolling 30, 60, and 90-day intervals.
 Metric Triggers: If overall ROC-AUC metrics fall below `0.70`, initiate automated pipeline retraining routines.

---

## 🪵 Production Logging Schema

Production JSON structured logs are funneled cleanly to target storage solutions:

```json
{
  "timestamp": "2026-06-09T18:04:23Z",
  "level": "INFO",
  "endpoint": "/predict",
  "latency_ms": 14.2,
  "payload": {
    "recency": 120,
    "frequency": 1,
    "monetary": 50.0
  },
  "inference": {
    "probability": 0.84,
    "risk_level": "High"
  }
}
