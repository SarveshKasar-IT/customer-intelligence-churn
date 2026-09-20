"""Loads the trained model once and turns customer data into predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .preprocessing import build_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "customer_churn_random_forest.pkl"
THRESHOLD_PATH = ROOT / "models" / "churn_threshold.txt"

# Same risk bands the Streamlit app uses.
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40

RECOMMENDATIONS = {
    "HIGH": (
        "High churn risk. Consider a personalized offer, a service review, "
        "or a customer-support call."
    ),
    "MEDIUM": (
        "Moderate churn risk. Monitor this customer and offer targeted "
        "engagement or support."
    ),
    "LOW": "Low churn risk. Continue normal engagement and service monitoring.",
}


def risk_level(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "HIGH"
    if probability >= MEDIUM_RISK:
        return "MEDIUM"
    return "LOW"


class ChurnModel:
    def __init__(self, model_path: Path = MODEL_PATH, threshold_path: Path = THRESHOLD_PATH):
        self.model = joblib.load(model_path)
        self.model_name = type(self.model).__name__
        self.columns = list(self.model.feature_names_in_)
        self.threshold = (
            float(Path(threshold_path).read_text().strip())
            if Path(threshold_path).exists()
            else 0.5
        )

    def probabilities(self, customers: pd.DataFrame) -> np.ndarray:
        features = build_features(customers, self.columns)
        return self.model.predict_proba(features)[:, 1]

    def predict(self, customers: pd.DataFrame) -> list[dict]:
        results = []
        for probability in self.probabilities(customers):
            probability = float(probability)
            level = risk_level(probability)
            results.append(
                {
                    "churn_probability": round(probability, 4),
                    "will_churn": probability >= self.threshold,
                    "risk_level": level,
                    "threshold": round(self.threshold, 4),
                    "recommendation": RECOMMENDATIONS[level],
                }
            )
        return results
