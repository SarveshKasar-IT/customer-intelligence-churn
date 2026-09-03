
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Header */
    .hero {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #1f2937 0%,
            #111827 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        text-align: center;
        min-height: 120px;
    }

    .metric-title {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #111827;
    }

    /* Prediction cards */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-top: 1rem;
    }

    .risk-high {
        border-left: 6px solid #dc2626;
    }

    .risk-medium {
        border-left: 6px solid #f59e0b;
    }

    .risk-low {
        border-left: 6px solid #16a34a;
    }

    .risk-label {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .probability {
        font-size: 2.4rem;
        font-weight: 800;
    }

    /* Section headers */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        color: #111827;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.65rem;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/customer_churn_random_forest.pkl"
THRESHOLD_PATH = "models/churn_threshold.txt"
DATA_PATH = "data/raw/telco_customer_churn.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)


def load_threshold():
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, "r") as f:
            return float(f.read().strip())
    return 0.50


try:
    model = load_model()
    df_raw = load_training_data()
    churn_threshold = load_threshold()

except Exception as e:
    st.error("Unable to load the model or project files.")
    st.exception(e)
    st.stop()


# ============================================================
# RECREATE TRAINING FEATURES
# ============================================================

df_training = df_raw.copy()

df_training["TotalCharges"] = pd.to_numeric(
    df_training["TotalCharges"],
    errors="coerce"
)

df_training["TotalCharges"] = df_training["TotalCharges"].fillna(0)

df_training["Churn"] = df_training["Churn"].map({
    "No": 0,
    "Yes": 1
})

df_training = df_training.drop(
    columns=["customerID", "Churn"]
)

df_training["tenure_group"] = pd.cut(
    df_training["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=[
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]
)

df_training_encoded = pd.get_dummies(
    df_training,
    drop_first=True
)

training_columns = df_training_encoded.columns


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎯 Customer Profile")

    st.caption(
        "Enter customer information to estimate "
        "the probability of churn."
    )

    st.markdown("---")

    st.markdown("### 👤 Personal Information")

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    st.markdown("### 📱 Services")

    tenure = st.slider(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    st.markdown("### 💳 Billing")

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges ($)",
        min_value=0.0,
        max_value=10000.0,
        value=monthly_charges * max(tenure, 1),
        step=10.0
    )

    st.markdown("---")

    predict_button = st.button(
        "🔮 Predict Churn Risk",
        type="primary",
        use_container_width=True
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown("""
<div class="hero">

    <h1>📊 Customer Churn Intelligence</h1>

    <p>
        Machine learning powered customer retention and churn
        risk prediction platform.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# TOP INFORMATION CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">🤖 Model</div>
        <div class="metric-value">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">🎯 Prediction</div>
        <div class="metric-value">Churn Probability</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">⚖️ Decision Threshold</div>
        <div class="metric-value">{churn_threshold:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">📈 Risk Levels</div>
        <div class="metric-value">3 Levels</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 How it works</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter a customer's profile using the sidebar. "
    "The trained Random Forest model estimates their churn "
    "probability and classifies the customer as Low, Medium, "
    "or High risk."
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({
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

    # --------------------------------------------------------
    # TENURE GROUP
    # --------------------------------------------------------

    input_data["tenure_group"] = pd.cut(
        input_data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    # --------------------------------------------------------
    # ONE-HOT ENCODING
    # --------------------------------------------------------

    input_encoded = pd.get_dummies(
        input_data,
        drop_first=True
    )

    # Match training features exactly
    input_encoded = input_encoded.reindex(
        columns=training_columns,
        fill_value=0
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    churn_probability = model.predict_proba(
        input_encoded
    )[0][1]

    churn_prediction = int(
        churn_probability >= churn_threshold
    )

    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if churn_probability >= 0.70:
        risk_level = "HIGH"
        risk_class = "risk-high"
        risk_icon = "🔴"

    elif churn_probability >= 0.40:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
        risk_icon = "🟠"

    else:
        risk_level = "LOW"
        risk_class = "risk-low"
        risk_icon = "🟢"


    # ========================================================
    # RESULT HEADER
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Churn Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="prediction-card {risk_class}">

            <div style="display:flex;
                        justify-content:space-between;
                        align-items:center;">

                <div>
                    <div style="color:#6b7280;
                                font-size:0.9rem;">
                        Predicted Risk Level
                    </div>

                    <div class="risk-label">
                        {risk_icon} {risk_level} RISK
                    </div>
                </div>

                <div style="text-align:right;">

                    <div style="color:#6b7280;
                                font-size:0.9rem;">
                        Churn Probability
                    </div>

                    <div class="probability">
                        {churn_probability:.1%}
                    </div>

                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Risk Probability</div>',
        unsafe_allow_html=True
    )

    st.progress(
        float(churn_probability),
        text=f"Estimated churn probability: {churn_probability:.1%}"
    )


    # ========================================================
    # PREDICTION MESSAGE
    # ========================================================

    if churn_prediction == 1:

        st.error(
            f"⚠️ **Customer is predicted to churn.** "
            f"The model estimates a {churn_probability:.1%} "
            f"probability of churn."
        )

    else:

        st.success(
            f"✅ **Customer is predicted to stay.** "
            f"The estimated churn probability is "
            f"{churn_probability:.1%}."
        )


    # ========================================================
    # RETENTION RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Recommended Retention Actions</div>',
        unsafe_allow_html=True
    )

    if risk_level == "HIGH":

        recommendations = [
            "📞 Contact the customer proactively.",
            "🎁 Offer a personalized retention incentive.",
            "💳 Review pricing and billing concerns.",
            "📋 Consider moving the customer to a longer-term contract.",
            "🛠️ Check whether service or technical issues are affecting satisfaction."
        ]

    elif risk_level == "MEDIUM":

        recommendations = [
            "📧 Send a personalized engagement offer.",
            "🎁 Consider a targeted discount or loyalty benefit.",
            "📊 Monitor customer activity and service usage.",
            "📞 Consider a proactive customer-success contact."
        ]

    else:

        recommendations = [
            "⭐ Continue providing consistent service.",
            "🎁 Consider loyalty rewards.",
            "📈 Monitor the customer for changes in behavior.",
            "💬 Maintain regular customer engagement."
        ]

    for recommendation in recommendations:
        st.write(recommendation)


    # ========================================================
    # CUSTOMER SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">👤 Customer Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Tenure",
            f"{tenure} months"
        )

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:,.2f}"
        )

    with summary_col2:

        st.metric(
            "Total Charges",
            f"${total_charges:,.2f}"
        )

        st.metric(
            "Contract",
            contract
        )

    with summary_col3:

        st.metric(
            "Internet Service",
            internet_service
        )

        st.metric(
            "Payment Method",
            payment_method
        )


    # ========================================================
    # MODEL DECISION
    # ========================================================

    with st.expander("🔬 View model decision details"):

        st.write(
            f"**Churn probability:** "
            f"{churn_probability:.4f}"
        )

        st.write(
            f"**Decision threshold:** "
            f"{churn_threshold:.4f}"
        )

        st.write(
            f"**Predicted class:** "
            f"{'Churn' if churn_prediction == 1 else 'No Churn'}"
        )

        st.write(
            "**Model:** Random Forest Classifier"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Intelligence • Machine Learning Project • "
    "Random Forest Classification"
)

st.caption(
    "⚠️ Predictions are model estimates and should be used "
    "as decision-support information."
)
