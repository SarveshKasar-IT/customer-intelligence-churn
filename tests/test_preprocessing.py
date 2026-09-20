"""Training/serving parity tests.

The reference function below is the preprocessing the model was trained with
(copied from the original app.py). The new build_features() must match it.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_features


def training_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    data = data.drop(columns=["customerID", "Churn"], errors="ignore")
    return pd.get_dummies(data, drop_first=True)


def test_columns_match_the_model(churn_model, dataset):
    reference = training_features(dataset)
    assert list(reference.columns) == churn_model.columns


def test_full_dataset_features_identical_to_training(churn_model, dataset):
    new = build_features(dataset, churn_model.columns)
    reference = training_features(dataset)[churn_model.columns]
    pd.testing.assert_frame_equal(new.astype(float), reference.astype(float))


def test_one_row_at_a_time_matches_batch(churn_model, dataset):
    """The bug the original app had: single rows must give the same result as batch."""
    sample = dataset.sample(200, random_state=0)
    batch = churn_model.model.predict_proba(
        training_features(dataset).loc[sample.index, churn_model.columns]
    )[:, 1]
    one_by_one = np.array(
        [churn_model.probabilities(sample.loc[[i]])[0] for i in sample.index]
    )
    np.testing.assert_allclose(one_by_one, batch, rtol=0, atol=1e-9)


def test_categorical_inputs_change_the_prediction(churn_model, base_customer):
    """Regression test: changing Contract used to have no effect at all."""
    monthly = churn_model.probabilities(pd.DataFrame([base_customer]))[0]
    two_year = churn_model.probabilities(
        pd.DataFrame([{**base_customer, "Contract": "Two year"}])
    )[0]
    assert two_year != monthly
    assert two_year < monthly  # longer contracts should lower churn risk


def test_unknown_category_is_rejected(churn_model, base_customer):
    bad = pd.DataFrame([{**base_customer, "Contract": "Three year"}])
    with pytest.raises(ValueError, match="Contract"):
        build_features(bad, churn_model.columns)


def test_tenure_out_of_range_is_rejected(churn_model, base_customer):
    bad = pd.DataFrame([{**base_customer, "tenure": 100}])
    with pytest.raises(ValueError, match="tenure"):
        build_features(bad, churn_model.columns)
