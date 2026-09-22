# Customer Intelligence & Churn Prediction

An end-to-end machine learning system for predicting customer churn, explaining model predictions, and serving real-time churn predictions through a production-ready FastAPI API and Streamlit interface.

The project covers the complete ML lifecycle:

**Data → EDA → Feature Engineering → Model Training → Model Comparison → Class Imbalance → Threshold Optimization → SHAP Explainability → MLflow Tracking → FastAPI → Streamlit → CI/CD → Cloud Deployment**

---

## 🚀 Live Application

## 🚀 Live Application

### Streamlit Dashboard

**[Open the Live Streamlit App](https://customer-intelligence-churn-iuch3usa5hdn6pxn8lkuav.streamlit.app/)**

The Streamlit application provides an interactive interface where users can enter customer information and receive:

- Churn probability
- Churn prediction
- Risk level
- Retention recommendation

The Streamlit application communicates with the deployed FastAPI backend rather than loading the model directly.

### FastAPI Backend

**[Live API](https://e-commerce-analytics-platform.onrender.com)**

**[Interactive API Documentation](https://e-commerce-analytics-platform.onrender.com/docs)**

**[API Health Check](https://e-commerce-analytics-platform.onrender.com/health)**

### Streamlit Dashboard

The Streamlit application provides an interactive interface where users can enter customer information and receive:

* Churn probability
* Churn prediction
* Risk level
* Retention recommendation

The Streamlit application communicates with the deployed FastAPI backend rather than loading the model directly.

**Live API:**
https://e-commerce-analytics-platform.onrender.com

**API Documentation:**
https://e-commerce-analytics-platform.onrender.com/docs

**Health Check:**
https://e-commerce-analytics-platform.onrender.com/health

> Add the Streamlit Community Cloud URL above once the final Streamlit deployment URL is confirmed.

---

# 📌 Project Overview

Customer churn occurs when an existing customer stops using a company's service.

The objective of this project is to build a machine learning system that estimates the probability that a telecom customer will churn and converts that probability into an actionable risk category.

The project uses the **Telco Customer Churn dataset** containing **7,043 customer records**.

The target variable is:

```text
Churn
```

with:

```text
No  → Customer stayed
Yes → Customer churned
```

The dataset contains **21 original columns**, including customer demographics, tenure, services, contract information, billing information, and churn status.

---

# 🎯 Business Objective

The system is designed to help a business identify customers who may be at higher risk of leaving.

Instead of treating every customer equally, the application produces an individual churn probability and risk classification.

Example:

```text
Customer
   ↓
Customer attributes
   ↓
Feature preprocessing
   ↓
XGBoost model
   ↓
Churn probability
   ↓
Risk classification
   ↓
Retention recommendation
```

The prediction can then support downstream customer-retention workflows.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit App     │
                    │  Customer Interface │
                    └──────────┬──────────┘
                               │
                               │ HTTP POST
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │                     │
                    │ /predict            │
                    │ /predict/batch      │
                    │ /health             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Shared Preprocessing │
                    │                     │
                    │ Encoding            │
                    │ Feature Alignment   │
                    │ Validation          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Balanced XGBoost    │
                    │ Churn Model         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Probability         │
                    │ + Threshold         │
                    │ + Risk Level        │
                    └─────────────────────┘
```

The model is trained offline and serialized using Joblib.

The production API loads the serialized model and applies the same preprocessing logic used during training.

---

# 🧠 Machine Learning Pipeline

```text
Raw Dataset
     │
     ▼
Data Understanding
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ├── TotalCharges conversion
     └── Tenure groups
     │
     ▼
Train/Test Split
     │
     │ Stratified 80/20
     ▼
Feature Encoding
     │
     ▼
Model Training
     │
     ├── Random Forest
     ├── Balanced Random Forest
     ├── XGBoost
     └── Balanced XGBoost
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1
     ├── ROC-AUC
     └── PR-AUC
     │
     ▼
Threshold Analysis
     │
     ▼
SHAP Explainability
     │
     ▼
MLflow Experiment Tracking
     │
     ▼
Production Model
     │
     ▼
FastAPI
     │
     ▼
Streamlit
```

---

# 📊 Dataset

Dataset:

**Telco Customer Churn**

Shape:

```text
7,043 rows
21 columns
```

Important feature groups include:

### Customer information

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`
* `tenure`

### Services

* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`

### Contract and billing

* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`

### Target

* `Churn`

---

# ⚙️ Feature Engineering

The production preprocessing pipeline is centralized in:

```text
src/preprocessing.py
```

This is important because the model must receive exactly the same feature representation during training and inference.

The preprocessing pipeline performs:

### 1. Total Charges conversion

`TotalCharges` is converted to numeric values and missing values are handled.

### 2. Tenure grouping

Customer tenure is converted into categorical tenure groups.

### 3. Categorical validation

Expected categorical levels are explicitly checked.

Unexpected values are rejected rather than silently encoded.

### 4. One-hot encoding

Categorical variables are converted into numerical features.

### 5. Feature alignment

The resulting dataframe is reindexed to the exact feature columns expected by the trained model.

The final production model uses:

```text
33 features
```

---

# 🤖 Models Evaluated

Stage 3 compared four configurations:

1. Random Forest
2. Balanced Random Forest
3. XGBoost
4. Balanced XGBoost

Class imbalance was explicitly investigated because the dataset contains substantially more non-churn customers than churn customers.

---

# 📈 Stage 3 Model Results

Evaluation was performed on the current stratified evaluation split.

| Model                  | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ---------------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Random Forest          |   0.7906 |    0.6330 | 0.5027 | 0.5604 |  0.8244 | 0.6147 |
| Balanced Random Forest |   0.7637 |    0.5469 | 0.6390 | 0.5894 |  0.8271 | 0.6166 |
| XGBoost                |   0.7991 |    0.6596 | 0.5027 | 0.5706 |  0.8424 | 0.6550 |
| Balanced XGBoost       |   0.7530 |    0.5235 | 0.7754 | 0.6250 |  0.8429 | 0.6594 |

These metrics should be interpreted together rather than relying on accuracy alone.

For a churn problem, recall and precision are particularly important because the business may care about identifying customers who are actually at risk while controlling the number of unnecessary retention interventions.

---

# 🎚️ Threshold Optimization

A classification model normally converts a probability into a binary prediction using a threshold.

Instead of automatically assuming:

```text
0.50
```

the project evaluated different thresholds using F1 score.

The resulting F1-optimized thresholds on the current evaluation split were:

| Model                  | F1-optimized threshold | Best F1 |
| ---------------------- | ---------------------: | ------: |
| Random Forest          |                   0.25 |  0.6089 |
| Balanced Random Forest |                   0.35 |  0.6204 |
| XGBoost                |                   0.30 |  0.6309 |
| Balanced XGBoost       |                   0.55 |  0.6321 |

The production model currently uses:

```text
Balanced XGBoost
Threshold = 0.55
```

### Important evaluation note

The threshold above was selected from the current evaluation split. For a stricter production validation process, threshold selection should ideally be performed on a separate validation set and then evaluated once on an untouched test set.

Therefore, `0.55` is documented as the **F1-optimized threshold on the current evaluation split**, rather than being described as a universally optimal threshold.

---

# 🚀 Production Model

The deployed API currently uses:

```text
XGBClassifier
```

with:

```text
33 features
```

and:

```text
Decision threshold = 0.55
```

Model file:

```text
models/stage3_xgboost_balanced.pkl
```

Threshold file:

```text
models/stage3_xgboost_balanced_threshold.txt
```

---

# 🔍 SHAP Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to investigate which features contribute to model predictions.

SHAP analysis is implemented in:

```text
src/shap_analysis.py
```

Generated artifacts include:

```text
data/processed/shap/
├── shap_global_bar.png
├── shap_beeswarm.png
├── shap_feature_importance.csv
└── shap_individual_waterfall.png
```

### Global explanation

The global SHAP analysis helps identify features that have the largest overall contribution across the evaluated customer sample.

### Beeswarm plot

The beeswarm visualization shows:

* Feature importance
* Direction of contribution
* Distribution of SHAP values

### Individual explanation

The waterfall plot demonstrates how individual feature contributions combine to move a prediction away from the model's baseline.

---

# 📊 MLflow Experiment Tracking

MLflow was used to track the Stage 3 experiments.

Experiment:

```text
Customer Churn - Stage 3
```

Tracked model configurations include:

* Random Forest
* Balanced Random Forest
* XGBoost
* Balanced XGBoost
* SHAP explainability analysis

Tracked information includes model metrics and experiment artifacts.

The project also includes:

```text
src/log_shap_mlflow.py
```

for logging SHAP artifacts to MLflow.

---

# 🌐 FastAPI

The model is served through FastAPI.

Main API files:

```text
api/
├── main.py
└── schemas.py
```

### Health endpoint

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model": "XGBClassifier",
  "n_features": 33,
  "threshold": 0.55
}
```

### Single prediction

```http
POST /predict
```

The endpoint accepts customer information and returns:

* Churn probability
* Churn prediction
* Risk level
* Threshold
* Retention recommendation

### Batch prediction

```http
POST /predict/batch
```

Supports prediction for multiple customer records.

### Interactive API documentation

FastAPI automatically provides Swagger documentation at:

```text
/docs
```

---

# 🛡️ API Validation

The API uses Pydantic schemas to validate incoming customer data.

Validation helps prevent malformed requests from reaching the model.

The API validates the expected Telco customer fields and categorical values before inference.

---

# 🎯 Risk Classification

The API converts churn probability into business-friendly risk levels.

Conceptually:

```text
Customer Data
      ↓
Churn Probability
      ↓
Risk Classification
      ↓
Retention Recommendation
```

The API currently uses:

```text
HIGH
MEDIUM
LOW
```

risk categories.

Recommendations are generated based on the resulting risk level.

---

# 🖥️ Streamlit Application

The Streamlit application is implemented in:

```text
app.py
```

The application acts as the user-facing frontend.

Instead of directly loading the model, it sends customer information to the FastAPI service.

```text
Streamlit
    │
    │ HTTP request
    ▼
FastAPI
    │
    ▼
XGBoost
    │
    ▼
Prediction
    │
    ▼
Streamlit
```

This separation makes the application architecture closer to a production ML system.

---

# 🧪 Testing

The project includes automated tests using Pytest.

Test areas include:

* API health endpoint
* Prediction endpoint
* Batch prediction
* Input validation
* Preprocessing
* Feature alignment
* Production model loading

Current test result:

```text
18 passed
```

Run locally with:

```powershell
pytest -q
```

---

# 🔄 Continuous Integration

GitHub Actions is configured through:

```text
.github/workflows/ci.yml
```

The CI pipeline runs the automated test suite when changes are pushed to GitHub.

This provides an automated quality check before deployment.

---

# ☁️ Deployment

The FastAPI backend is deployed on Render.

Production API:

```text
https://e-commerce-analytics-platform.onrender.com
```

Render deployment uses:

```text
Build:
pip install -r requirements-api.txt

Start:
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

The production environment includes the required XGBoost runtime dependency.

---

# 🐳 Docker

The repository contains a Dockerfile for containerized API deployment.

Docker configuration is designed to install the API dependencies and start the FastAPI application.

Example:

```powershell
docker build -t customer-churn-api .
```

Then:

```powershell
docker run -p 8000:8000 customer-churn-api
```

> Docker was not locally verified during the current development cycle because Docker Desktop was not installed on the development machine. The Docker configuration is included in the repository, but this README does not claim a successful local Docker build.

---

# 📁 Project Structure

```text
customer-intelligence-churn/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   │   └── telco_customer_churn.csv
│   │
│   └── processed/
│       ├── stage3_model_comparison.csv
│       ├── stage3_threshold_analysis.csv
│       └── shap/
│
├── models/
│   ├── customer_churn_random_forest.pkl
│   ├── churn_threshold.txt
│   ├── stage3_xgboost_balanced.pkl
│   └── stage3_xgboost_balanced_threshold.txt
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model_service.py
│   ├── stage3_experiment.py
│   ├── shap_analysis.py
│   └── log_shap_mlflow.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api.py
│   └── test_preprocessing.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── app.py
├── requirements.txt
├── requirements-api.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

# 🛠️ Technology Stack

| Technology     | Purpose                     |
| -------------- | --------------------------- |
| Python         | Core development            |
| Pandas         | Data manipulation           |
| NumPy          | Numerical operations        |
| Scikit-learn   | ML utilities and evaluation |
| Random Forest  | Baseline ensemble model     |
| XGBoost        | Gradient boosting model     |
| SHAP           | Model explainability        |
| MLflow         | Experiment tracking         |
| FastAPI        | REST API                    |
| Pydantic       | API validation              |
| Streamlit      | User interface              |
| Joblib         | Model serialization         |
| Pytest         | Automated testing           |
| GitHub Actions | CI                          |
| Docker         | Containerization            |
| Render         | API deployment              |

---

# ▶️ Run Locally

## 1. Clone the repository

```powershell
git clone https://github.com/SarveshKasar-IT/customer-intelligence-churn.git
cd customer-intelligence-churn
```

## 2. Create a virtual environment

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

## 3. Install dependencies

For the full development environment:

```powershell
pip install -r requirements.txt
```

For API development:

```powershell
pip install -r requirements-api.txt
```

For testing:

```powershell
pip install -r requirements-dev.txt
```

## 4. Run tests

```powershell
pytest -q
```

## 5. Run FastAPI

```powershell
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 6. Run Streamlit

```powershell
streamlit run app.py
```

---

# 🔬 Reproduce Stage 3 Experiments

The Stage 3 experiment script is:

```text
src/stage3_experiment.py
```

Run:

```powershell
python src/stage3_experiment.py
```

The experiment compares the four model configurations and generates evaluation artifacts.

---

# 🔍 Reproduce SHAP Analysis

Run:

```powershell
python src/shap_analysis.py
```

This generates the SHAP artifacts under:

```text
data/processed/shap/
```

---

# 📋 API Example

Example customer request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 450.0
}
```

The response contains information similar to:

```json
{
  "churn_probability": 0.90,
  "will_churn": true,
  "risk_level": "HIGH",
  "threshold": 0.55,
  "recommendation": "High churn risk. Consider a personalized offer, a service review, or a customer-support call."
}
```

The exact probability depends on the supplied customer attributes.

---

# 💡 Key Technical Learnings

### 1. Accuracy alone is not enough

The dataset is imbalanced, so accuracy does not fully describe churn detection performance.

Precision, recall, F1, ROC-AUC, and PR-AUC provide additional information.

### 2. Class imbalance changes model behavior

Balanced training increased the model's ability to identify churn cases, while also changing precision and accuracy.

This demonstrates the practical trade-off between false positives and false negatives.

### 3. Probability threshold matters

A model's predicted probability does not automatically have to use `0.50` as the decision boundary.

Different thresholds produce different precision/recall trade-offs.

### 4. Preprocessing must be consistent

The same feature engineering and encoding logic must be used during both training and production inference.

The shared preprocessing module helps prevent training/serving mismatches.

### 5. Explainability is important

SHAP helps move from:

```text
"The customer is likely to churn."
```

toward:

```text
"Which features contributed to this prediction?"
```

### 6. Experiment tracking improves reproducibility

MLflow records experiments, metrics, and artifacts so model development can be compared systematically.

### 7. ML deployment is more than saving a model

A production ML system requires:

```text
Model
+
Preprocessing
+
Validation
+
API
+
Testing
+
CI
+
Deployment
+
Monitoring/Tracking
```

---

# 📌 Project Development Journey

The project evolved through multiple stages.

### Stage 1 — Data & ML

* Dataset exploration
* Data cleaning
* EDA
* Feature engineering
* Baseline machine learning
* Model evaluation

### Stage 2 — Production API

* FastAPI
* Pydantic validation
* Shared preprocessing
* API testing
* Docker configuration
* GitHub Actions
* Render deployment
* Streamlit → API integration

### Stage 3 — Model Improvement

* XGBoost
* Class imbalance analysis
* Balanced XGBoost
* Precision/Recall analysis
* PR-AUC
* Threshold optimization
* SHAP
* MLflow
* Production model upgrade

---

# 📌 Current Production Configuration

```text
Model:
Balanced XGBoost

Model class:
XGBClassifier

Features:
33

Threshold:
0.55

API:
FastAPI

Frontend:
Streamlit

Deployment:
Render

Testing:
Pytest

CI:
GitHub Actions

Explainability:
SHAP

Experiment tracking:
MLflow
```

---

# 💼 Resume Description

**Customer Churn Intelligence Platform | Python, XGBoost, FastAPI, Streamlit, SHAP, MLflow**

* Built an end-to-end customer churn prediction system using the Telco Customer Churn dataset, covering preprocessing, feature engineering, model comparison, class-imbalance analysis, threshold optimization, and evaluation using precision, recall, F1, ROC-AUC, and PR-AUC.
* Improved the modeling pipeline by evaluating Random Forest and XGBoost configurations, including class-balanced training, and deployed a balanced XGBoost model through a FastAPI REST service.
* Implemented shared production preprocessing, Pydantic validation, automated Pytest coverage, GitHub Actions CI, SHAP explainability, MLflow experiment tracking, Docker configuration, and Render deployment.
* Integrated the FastAPI backend with a Streamlit frontend to provide real-time churn probability, risk classification, and retention recommendations.

---

# 🎤 Interview Summary

A concise way to explain the project:

> "I built an end-to-end customer churn prediction platform using the Telco Customer Churn dataset. I started with data understanding and feature engineering, then compared Random Forest and XGBoost models while investigating class imbalance and probability thresholds. I added SHAP for explainability and MLflow for experiment tracking. For deployment, I separated the model from the UI by serving it through FastAPI, added input validation and automated tests, connected the API to Streamlit, configured GitHub Actions CI, and deployed the API on Render. The current production model is a balanced XGBoost classifier using 33 features and a threshold of 0.55 selected for F1 on the current evaluation split."

---

# 👨‍💻 Author

**Sarvesh Kasar**

GitHub:
https://github.com/SarveshKasar-IT

Project Repository:
https://github.com/SarveshKasar-IT/customer-intelligence-churn
