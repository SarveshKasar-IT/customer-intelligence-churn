
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/customer_churn_random_forest.pkl"
THRESHOLD_PATH = "models/churn_threshold.txt"
DATA_PATH = "data/raw/telco_customer_churn.csv"


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    return model, threshold


@st.cache_data
def load_training_columns():

    df = pd.read_csv(DATA_PATH)

    # Same preprocessing used during model training

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    df = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    df_encoded = pd.get_dummies(
        df,
        drop_first=True
    )

    return df_encoded.columns.tolist()


try:

    model, threshold = load_model()
    training_columns = load_training_columns()

except Exception as e:

    st.error(
        "Unable to load the model or dataset."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    """
    This application predicts whether a telecommunications customer
    is likely to churn based on their demographic, service, contract,
    and billing information.
    """
)

st.info(
    f"Current classification threshold: {threshold:.2f}"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Customer Information")

st.sidebar.write(
    "Enter the customer's information below."
)


# ============================================================
# CUSTOMER DETAILS
# ============================================================

gender = st.sidebar.selectbox(
    "Gender",
    ["Female", "Male"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.sidebar.selectbox(
    "Partner",
    ["No", "Yes"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["No", "Yes"]
)

tenure = st.sidebar.number_input(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=12,
    step=1
)


# ============================================================
# SERVICE INFORMATION
# ============================================================

st.sidebar.subheader("Services")

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["No", "Yes"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["No internet service", "No", "Yes"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["No internet service", "No", "Yes"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["No internet service", "No", "Yes"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["No internet service", "No", "Yes"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["No internet service", "No", "Yes"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["No internet service", "No", "Yes"]
)


# ============================================================
# CONTRACT AND BILLING
# ============================================================

st.sidebar.subheader("Contract & Billing")

contract = st.sidebar.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["No", "Yes"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=monthly_charges * tenure,
    step=10.0
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

customer_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# ============================================================
# CREATE TENURE GROUP
# ============================================================

customer_data["tenure_group"] = pd.cut(
    customer_data["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)


# ============================================================
# ENCODE INPUT
# ============================================================

customer_encoded = pd.get_dummies(
    customer_data,
    drop_first=True
)


# Make sure input has exactly the same features as training data

customer_encoded = customer_encoded.reindex(
    columns=training_columns,
    fill_value=0
)


# ============================================================
# PREDICTION
# ============================================================

st.header("Prediction")

if st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
):

    probability = model.predict_proba(
        customer_encoded
    )[0][1]

    prediction = int(
        probability >= threshold
    )


    # ========================================================
    # DISPLAY PROBABILITY
    # ========================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Churn Probability",
            f"{probability:.1%}"
        )

    with col2:

        if prediction == 1:

            st.metric(
                "Prediction",
                "⚠️ CHURN"
            )

        else:

            st.metric(
                "Prediction",
                "✅ NO CHURN"
            )

    with col3:

        if probability >= 0.70:

            risk = "HIGH"

        elif probability >= 0.40:

            risk = "MEDIUM"

        else:

            risk = "LOW"

        st.metric(
            "Risk Level",
            risk
        )


    # ========================================================
    # RESULT MESSAGE
    # ========================================================

    if prediction == 1:

        st.error(
            f"""
            ⚠️ This customer is predicted to be at risk of churn.

            Estimated churn probability: {probability:.1%}

            The business should consider proactive retention actions.
            """
        )

        st.subheader(
            "Recommended Retention Actions"
        )

        st.write(
            """
            • Contact the customer proactively

            • Review pricing and plan suitability

            • Consider a loyalty or retention offer

            • Investigate service quality issues

            • Encourage migration to a longer-term contract

            • Provide additional customer support
            """
        )

    else:

        st.success(
            f"""
            ✅ This customer is predicted to remain with the company.

            Estimated churn probability: {probability:.1%}
            """
        )

        st.info(
            """
            Continue normal customer engagement and monitor the
            customer periodically.
            """
        )


# ============================================================
# CUSTOMER INPUT SUMMARY
# ============================================================

st.header("Customer Summary")

summary = pd.DataFrame({
    "Attribute": [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure",
        "Internet Service",
        "Contract",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"
    ],

    "Value": [
        gender,
        "Yes" if senior_citizen == 1 else "No",
        partner,
        dependents,
        f"{tenure} months",
        internet_service,
        contract,
        payment_method,
        f"${monthly_charges:.2f}",
        f"${total_charges:.2f}"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        f"""
        **Model:** {type(model).__name__}

        **Classification Threshold:** {threshold:.2f}

        **Test ROC-AUC:** {0.0 if "final_roc_auc" not in globals() else final_roc_auc:.4f}

        This model was developed using customer demographic,
        service, contract, and billing information.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction | Machine Learning Project"
)
