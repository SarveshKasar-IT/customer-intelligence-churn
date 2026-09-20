## 🔌 REST API (FastAPI + Docker)

The trained model is also served as a REST API, so other systems can request predictions without using the Streamlit UI.

```
Client ──POST /predict──▶ FastAPI (Pydantic validation)
                              │
                              ▼
                     src/preprocessing.py  ──▶  Random Forest  ──▶  probability, risk level
                     (same features as training)
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and model status |
| `POST /predict` | Churn probability, risk level, and recommendation for one customer |
| `POST /predict/batch` | Same, for up to 1,000 customers |
| `GET /docs` | Interactive Swagger documentation |

Inputs are validated: an unknown contract type, a tenure outside 0-72 months, or a missing field returns HTTP 422.

### Run locally

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 285.0
}'
```

```json
{
  "churn_probability": 0.6946,
  "will_churn": true,
  "risk_level": "MEDIUM",
  "threshold": 0.32,
  "recommendation": "Moderate churn risk. Monitor this customer and offer targeted engagement or support."
}
```

### Tests and CI

```bash
pip install -r requirements-api.txt -r requirements-dev.txt
pytest
```

GitHub Actions runs the tests on every push, then builds the Docker image and checks that `/health` responds.

### Engineering notes

- **Training/serving parity.** The first version of the app encoded one customer at a time with `pd.get_dummies(drop_first=True)`. With a single row, every text column has one value, so `drop_first` removed it and all categorical inputs (contract, internet service, payment method, ...) were ignored. Preprocessing now lives in `src/preprocessing.py`, gives each categorical column its full list of allowed values, and is used by both the API and the Streamlit app. A test checks that features for all 7,043 customers are identical to the training features, and that predicting one customer at a time matches batch predictions.
- **Pinned scikit-learn.** The model was saved with scikit-learn 1.9.0, so `requirements-api.txt` pins that version.
- **Known limitations.** The risk bands (Medium from 0.40, High from 0.70) are fixed and separate from the F1-optimized decision threshold (0.32), so a customer can be "likely to churn" with a Medium label. Model performance is limited by the Telco dataset (F1 about 0.59).
