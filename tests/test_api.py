import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan so the model is loaded
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "XGBClassifier"
    assert body["n_features"] == 33


def test_predict_returns_expected_fields(client, base_customer):
    r = client.post("/predict", json=base_customer)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert isinstance(body["will_churn"], bool)
    assert body["recommendation"]


def test_predict_responds_to_contract_type(client, base_customer):
    monthly = client.post("/predict", json=base_customer).json()["churn_probability"]
    two_year = client.post(
        "/predict", json={**base_customer, "Contract": "Two year"}
    ).json()["churn_probability"]
    assert two_year < monthly


@pytest.mark.parametrize(
    "change",
    [
        {"Contract": "Three year"},
        {"tenure": 100},
        {"tenure": -1},
        {"MonthlyCharges": -5},
        {"SeniorCitizen": 2},
        {"gender": "Other"},
    ],
)
def test_invalid_input_returns_422(client, base_customer, change):
    r = client.post("/predict", json={**base_customer, **change})
    assert r.status_code == 422


def test_missing_field_returns_422(client, base_customer):
    base_customer.pop("Contract")
    assert client.post("/predict", json=base_customer).status_code == 422


def test_batch_predict(client, base_customer):
    payload = {"customers": [base_customer, {**base_customer, "Contract": "Two year"}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["predictions"][1]["churn_probability"] < body["predictions"][0]["churn_probability"]


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
