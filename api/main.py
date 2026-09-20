"""Customer churn prediction API.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request

from src.model_service import ChurnModel

from .schemas import BatchRequest, BatchResponse, Customer, Health, Prediction


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.churn_model = ChurnModel()  # load once at startup
    yield


app = FastAPI(
    title="Customer Churn API",
    description="Predicts churn probability and risk level for telecom customers.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=Health)
def health(request: Request):
    m: ChurnModel = request.app.state.churn_model
    return Health(
        status="ok",
        model=m.model_name,
        n_features=len(m.columns),
        threshold=round(m.threshold, 4),
    )


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request):
    m: ChurnModel = request.app.state.churn_model
    frame = pd.DataFrame([customer.model_dump()])
    return m.predict(frame)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(payload: BatchRequest, request: Request):
    m: ChurnModel = request.app.state.churn_model
    frame = pd.DataFrame([c.model_dump() for c in payload.customers])
    predictions = m.predict(frame)
    return BatchResponse(count=len(predictions), predictions=predictions)
