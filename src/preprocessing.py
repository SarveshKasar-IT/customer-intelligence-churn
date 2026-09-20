"""Feature preparation shared by the API and the Streamlit app.

Why this file exists
--------------------
The original app built features for one customer at a time with
pd.get_dummies(drop_first=True). With a single row, every text column has only
one value, so drop_first removed it and every categorical input (Contract,
InternetService, PaymentMethod, ...) became zero. The model then only saw
tenure, MonthlyCharges and TotalCharges.

Here every categorical column is given its full list of allowed values before
encoding, so one row and many rows produce identical features.
"""
from __future__ import annotations

import pandas as pd

TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]

_YES_NO = ["No", "Yes"]
_YES_NO_NO_INTERNET = ["No", "No internet service", "Yes"]

# Levels are listed in alphabetical order, the same order pd.get_dummies uses,
# so drop_first removes the same baseline value as it did during training.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": _YES_NO_NO_INTERNET,
    "OnlineBackup": _YES_NO_NO_INTERNET,
    "DeviceProtection": _YES_NO_NO_INTERNET,
    "TechSupport": _YES_NO_NO_INTERNET,
    "StreamingTV": _YES_NO_NO_INTERNET,
    "StreamingMovies": _YES_NO_NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def build_features(customers: pd.DataFrame, columns) -> pd.DataFrame:
    """Turn raw customer rows into the exact feature columns the model expects.

    Parameters
    ----------
    customers : raw Telco-style columns (customerID / Churn are ignored if present).
    columns   : the model's training columns, i.e. model.feature_names_in_.
    """
    df = customers.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    df = df.drop(columns=["customerID", "Churn"], errors="ignore")

    for column, levels in CATEGORY_LEVELS.items():
        original = df[column]
        unexpected = original[~original.isin(levels) & original.notna()].unique()
        if len(unexpected):
            raise ValueError(f"Unexpected value(s) in '{column}': {list(unexpected)}")
        df[column] = pd.Categorical(original, categories=levels)

    if df["tenure_group"].isna().any():
        raise ValueError("tenure must be between 0 and 72 months")

    encoded = pd.get_dummies(df, drop_first=True)
    return encoded.reindex(columns=list(columns), fill_value=0)
