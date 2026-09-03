
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

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

.main {
    padding-top: 1rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

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

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    color: #111827;
}

section[data-testid="stSidebar"] {
    background-color: #f8fafc;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/customer_churn_random_forest.pkl"
THRESHOLD_PATH = "models/churn_threshold.txt"
DATA_PATH = "data/raw/telco_customer_churn.csv"
IMPORTANCE_PATH = "models/feature_importance.csv"


# ============================================================
# LOAD MODEL / DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_training_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_feature_importance():

    if os.path.exists(IMPORTANCE_PATH):
        return pd.read_csv(IMPORTANCE_PATH)

    return None


def load_threshold():

    if os.path.exists(THRESHOLD_PATH):

        with open(THRESHOLD_PATH, "r") as f:
            return float(f.read().strip())

    return 0.50


try:

    model = load_model()
    df_raw = load_training_data()
    feature_importance = load_feature_importance()
    churn_threshold = load_threshold()

except Exception as e:

    st.error("Unable to load project files.")

    st.exception(e)

    st.stop()


# ============================================================
# PREPROCESS DATA
# ============================================================

df = df_raw.copy()

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

df = df.drop(
    columns=["customerID"]
)

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

df_encoded = pd.get_dummies(
    df,
    drop_first=True
)

X = df_encoded.drop(
    columns=["Churn"]
)

y = df_encoded["Churn"]


# ============================================================
# TRAINING COLUMN REFERENCE
# ============================================================

training_columns = X.columns


# ============================================================
# MODEL PERFORMANCE
# ============================================================

@st.cache_data
def calculate_model_metrics():

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= churn_threshold
    ).astype(int)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    return metrics, cm, fpr, tpr


metrics, confusion, fpr, tpr = calculate_model_metrics()


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
        0,
        72,
        12
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
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
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
        Machine learning powered customer retention
        and churn risk prediction platform.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================

tab1, tab2 = st.tabs([
    "🔮 Churn Prediction",
    "📈 Model Insights"
])


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with col2:

        st.metric(
            "📊 ROC-AUC",
            f"{metrics['ROC-AUC']:.3f}"
        )

    with col3:

        st.metric(
            "⚖️ Threshold",
            f"{churn_threshold:.2f}"
        )

    with col4:

        st.metric(
            "📁 Customers",
            f"{len(df_raw):,}"
        )


    st.markdown(
        '<div class="section-title">🔍 How it works</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Enter a customer's profile in the sidebar. "
        "The Random Forest model estimates churn probability "
        "and classifies the customer according to the optimized "
        "decision threshold."
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

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


        input_encoded = pd.get_dummies(

            input_data,

            drop_first=True
        )


        input_encoded = input_encoded.reindex(

            columns=training_columns,

            fill_value=0
        )


        churn_probability = model.predict_proba(

            input_encoded

        )[0][1]


        churn_prediction = int(

            churn_probability >= churn_threshold

        )


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


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '🎯 Churn Prediction Result'
            '</div>',
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


        st.progress(

            float(churn_probability),

            text=(
                f"Estimated churn probability: "
                f"{churn_probability:.1%}"
            )
        )


        if churn_prediction == 1:

            st.error(

                f"⚠️ **Customer is predicted to churn.** "
                f"The model estimates a "
                f"{churn_probability:.1%} probability of churn."
            )

        else:

            st.success(

                f"✅ **Customer is predicted to stay.** "
                f"The estimated churn probability is "
                f"{churn_probability:.1%}."
            )


        # ====================================================
        # RETENTION ACTIONS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '💡 Recommended Retention Actions'
            '</div>',
            unsafe_allow_html=True
        )


        if risk_level == "HIGH":

            recommendations = [

                "📞 Contact the customer proactively.",

                "🎁 Offer a personalized retention incentive.",

                "💳 Review pricing and billing concerns.",

                "📋 Consider moving the customer to "
                "a longer-term contract.",

                "🛠️ Check for service or technical issues."

            ]

        elif risk_level == "MEDIUM":

            recommendations = [

                "📧 Send a personalized engagement offer.",

                "🎁 Consider a targeted loyalty benefit.",

                "📊 Monitor customer activity.",

                "📞 Consider proactive customer-success contact."

            ]

        else:

            recommendations = [

                "⭐ Continue providing consistent service.",

                "🎁 Consider loyalty rewards.",

                "📈 Monitor customer behavior.",

                "💬 Maintain regular engagement."

            ]


        for recommendation in recommendations:

            st.write(recommendation)


        # ====================================================
        # CUSTOMER SUMMARY
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '👤 Customer Summary'
            '</div>',
            unsafe_allow_html=True
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            st.metric(
                "Tenure",
                f"{tenure} months"
            )

            st.metric(
                "Monthly Charges",
                f"${monthly_charges:,.2f}"
            )


        with c2:

            st.metric(
                "Total Charges",
                f"${total_charges:,.2f}"
            )

            st.metric(
                "Contract",
                contract
            )


        with c3:

            st.metric(
                "Internet Service",
                internet_service
            )

            st.metric(
                "Payment Method",
                payment_method
            )


        with st.expander(
            "🔬 View model decision details"
        ):

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
# TAB 2 — MODEL INSIGHTS
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">'
        '📈 Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Performance is evaluated on the held-out test set "
        "using the same preprocessing pipeline and saved model."
    )


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

    m1, m2, m3, m4, m5 = st.columns(5)


    with m1:

        st.metric(
            "Accuracy",
            f"{metrics['Accuracy']:.1%}"
        )


    with m2:

        st.metric(
            "Precision",
            f"{metrics['Precision']:.1%}"
        )


    with m3:

        st.metric(
            "Recall",
            f"{metrics['Recall']:.1%}"
        )


    with m4:

        st.metric(
            "F1 Score",
            f"{metrics['F1 Score']:.1%}"
        )


    with m5:

        st.metric(
            "ROC-AUC",
            f"{metrics['ROC-AUC']:.3f}"
        )


    # ========================================================
    # CONFUSION MATRIX + ROC
    # ========================================================

    chart_col1, chart_col2 = st.columns(2)


    with chart_col1:

        st.markdown("### Confusion Matrix")

        fig, ax = plt.subplots()

        ax.imshow(confusion)

        ax.set_title("Confusion Matrix")

        ax.set_xlabel("Predicted")

        ax.set_ylabel("Actual")

        ax.set_xticks([0, 1])

        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["No Churn", "Churn"]
        )

        ax.set_yticklabels(
            ["No Churn", "Churn"]
        )


        for i in range(2):

            for j in range(2):

                ax.text(
                    j,
                    i,
                    str(confusion[i, j]),
                    ha="center",
                    va="center",
                    fontsize=14
                )


        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    with chart_col2:

        st.markdown("### ROC Curve")

        fig, ax = plt.subplots()

        ax.plot(
            fpr,
            tpr,
            label=f"AUC = {metrics['ROC-AUC']:.3f}"
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random"
        )

        ax.set_xlabel("False Positive Rate")

        ax.set_ylabel("True Positive Rate")

        ax.set_title("Receiver Operating Characteristic")

        ax.legend()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔎 Top Churn Drivers'
        '</div>',
        unsafe_allow_html=True
    )


    if feature_importance is not None:

        importance = feature_importance.copy()


        # Identify likely feature/value columns
        feature_col = None
        importance_col = None


        for col in importance.columns:

            lower = col.lower()

            if "feature" in lower:

                feature_col = col

            if (
                "importance" in lower
                or "value" in lower
            ):

                importance_col = col


        if feature_col is None:

            feature_col = importance.columns[0]


        if importance_col is None:

            importance_col = importance.columns[1]


        importance = importance.sort_values(
            importance_col,
            ascending=False
        ).head(10)


        fig, ax = plt.subplots()

        ax.barh(
            importance[feature_col][::-1],
            importance[importance_col][::-1]
        )

        ax.set_title(
            "Top 10 Most Important Features"
        )

        ax.set_xlabel(
            "Feature Importance"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)


        with st.expander(
            "📋 View feature importance table"
        ):

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True
            )


    else:

        st.info(
            "Feature importance file was not found."
        )


    # ========================================================
    # BUSINESS INTERPRETATION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '💼 Business Interpretation'
        '</div>',
        unsafe_allow_html=True
    )


    st.info(
        """
        **How this model can support a retention team:**

        • Identify customers with elevated churn probability.

        • Prioritize high-risk customers for proactive outreach.

        • Use customer characteristics and service information
          to understand potential churn drivers.

        • Combine model predictions with customer-service
          knowledge before taking retention actions.

        • Use the probability score to prioritize limited
          retention resources.
        """
    )


    st.caption(
        "Model: Random Forest Classifier • "
        "Evaluation: held-out test set • "
        f"Decision threshold: {churn_threshold:.2f}"
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Intelligence • "
    "Machine Learning Portfolio Project"
)

st.caption(
    "⚠️ Predictions are model estimates and should be "
    "used as decision-support information."
)
