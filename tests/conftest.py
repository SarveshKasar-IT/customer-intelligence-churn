from pathlib import Path

import pandas as pd
import pytest

from src.model_service import ChurnModel

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_customer_churn.csv"

BASE_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 285.0,
}


@pytest.fixture(scope="session")
def churn_model():
    return ChurnModel()


@pytest.fixture(scope="session")
def dataset():
    if not DATA_PATH.exists():
        pytest.skip("training CSV not available")
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def base_customer():
    return dict(BASE_CUSTOMER)
