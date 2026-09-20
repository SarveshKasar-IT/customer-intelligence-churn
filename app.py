import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st

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

st.markdown(
    """
    <style>

    /* Main application */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .hero {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 100%
        );
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        color: #111827;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-top: 0;
    }

    /* Prediction result */
    .prediction-card {
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .prediction-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 0.5rem;
    }

    .prediction-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }

    /* Risk cards */
    .risk-high {
        padding: 1.2rem;
        border-radius: 14px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        text-align: center;
    }

    .risk-medium {
        padding: 1.2rem;
        border-radius: 14px;
        background: #fffbeb;
        border: 1px solid #fde68a;
        text-align: center;
    }

    .risk-low {
        padding: 1.2rem;
        border-radius: 14px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        text-align: center;
    }

    .risk-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    /* Information boxes */
    .info-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
    }

    .info-title {
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.3rem;
    }

    .info-text {
        color: #64748b;
        margin: 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        padding: 2rem 0 1rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/customer_churn_random_forest.pkl"
THRESHOLD_PATH = "models/churn_threshold.txt"
DATA_PATH = "data/raw/telco_customer_churn.csv"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.csv"

# Local development API URL.
# During deployment, this can be replaced through an environment variable.
API_URL = os.getenv(
    "MODEL_API_URL",
    "http://127.0.0.1:8000/predict"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD THRESHOLD
# ============================================================

@st.cache_data
def load_threshold():

    if not os.path.exists(THRESHOLD_PATH):
        return 0.50

    with open(THRESHOLD_PATH, "r") as file:
        return float(file.read().strip())


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found: {DATA_PATH}")
        st.stop()

    data = pd.read_csv(DATA_PATH)

    return data


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

@st.cache_data
def load_feature_importance():

    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        return None

    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


# ============================================================
# LOAD EVERYTHING
# ============================================================

model = load_model()
final_threshold = load_threshold()
df = load_data()
feature_importance = load_feature_importance()


# ============================================================
# RECREATE TRAINING PREPROCESSING
# ============================================================

def prepare_training_features(data):

    data = data.copy()

    # Convert TotalCharges to numeric
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # Create tenure group
    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months"
        ]
    )

    # Remove ID and target
    data = data.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    # One-hot encoding
    data = pd.get_dummies(
        data,
        drop_first=True
    )

    return data


# ============================================================
# CREATE TRAINING FEATURE COLUMNS
# ============================================================

training_features = prepare_training_features(df)

training_columns = training_features.columns.tolist()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            📊 Customer Churn Intelligence
        </div>
        <p class="hero-subtitle">
            Predict customer churn risk using machine learning
            and support data-driven retention decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

prediction_tab, insights_tab = st.tabs(
    [
        "🔮 Churn Prediction",
        "📈 Model Insights"
    ]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with prediction_tab:

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title("👤 Customer Profile")

    st.sidebar.markdown(
        "Enter customer information below."
    )

    gender = st.sidebar.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    senior_citizen = st.sidebar.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    partner = st.sidebar.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.sidebar.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

    tenure = st.sidebar.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=12
    )

    phone_service = st.sidebar.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.sidebar.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.sidebar.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.sidebar.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

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
        ["Yes", "No"]
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
        "Monthly Charges ($)",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.sidebar.number_input(
        "Total Charges ($)",
        min_value=0.0,
        value=840.0,
        step=10.0
    )

    st.sidebar.markdown("---")

    predict_button = st.sidebar.button(
        "🔮 Predict Churn Risk",
        use_container_width=True,
        type="primary"
    )


    # --------------------------------------------------------
    # MAIN PREDICTION AREA
    # --------------------------------------------------------

    st.subheader("Customer Risk Assessment")

    st.write(
        "Enter customer information in the sidebar and click "
        "**Predict Churn Risk** to generate a prediction."
    )

    if predict_button:

        # ----------------------------------------------------
        # CREATE CUSTOMER INPUT
        # ----------------------------------------------------

        customer_input = pd.DataFrame(
            [{
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }]
        )


        # ----------------------------------------------------
        # CALL FASTAPI PREDICTION SERVICE
        # ----------------------------------------------------

        try:

            response = requests.post(
                API_URL,
                json=customer_input.iloc[0].to_dict(),
                timeout=10
            )

            response.raise_for_status()

            prediction_result = response.json()

            churn_probability = prediction_result[
                "churn_probability"
            ]

            churn_prediction = int(
                prediction_result["will_churn"]
            )

            risk_level = prediction_result[
                "risk_level"
            ]

            final_threshold = prediction_result[
                "threshold"
            ]

            recommendation = prediction_result.get(
                "recommendation",
                ""
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI prediction "
                "service. Please make sure FastAPI is running "
                "at http://127.0.0.1:8000."
            )

            st.stop()

        except requests.exceptions.Timeout:

            st.error(
                "The FastAPI prediction service took too long "
                "to respond."
            )

            st.stop()

        except requests.exceptions.RequestException as e:

            st.error(
                f"Prediction API error: {e}"
            )

            st.stop()

        except (KeyError, ValueError, TypeError) as e:

            st.error(
                f"Unexpected prediction response from API: {e}"
            )

            st.stop()


        # ----------------------------------------------------
        # RISK CARD STYLE
        # ----------------------------------------------------

        if risk_level == "HIGH":

            risk_class = "risk-high"
            risk_icon = "🔴"

        elif risk_level == "MEDIUM":

            risk_class = "risk-medium"
            risk_icon = "🟡"

        else:

            risk_class = "risk-low"
            risk_icon = "🟢"


        # ----------------------------------------------------
        # TOP METRICS
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability:.1%}"
            )

        with col2:

            st.metric(
                "Model Decision",
                "Likely to Churn"
                if churn_prediction == 1
                else "Likely to Stay"
            )

        with col3:

            st.metric(
                "Decision Threshold",
                f"{final_threshold:.2f}"
            )


        # ----------------------------------------------------
        # RISK CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="{risk_class}">
                <div>Predicted Risk Level</div>
                <div class="risk-label">
                    {risk_icon} {risk_level} RISK
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PROBABILITY BAR
        # ----------------------------------------------------

        st.subheader("Churn Probability")

        st.progress(
            float(churn_probability)
        )

        st.caption(
            f"Estimated probability that this customer will "
            f"churn: {churn_probability:.2%}"
        )


        # ----------------------------------------------------
        # RETENTION RECOMMENDATION
        # ----------------------------------------------------

        st.subheader("💡 Recommended Action")

        if recommendation:

            if risk_level == "HIGH":

                st.error(recommendation)

            elif risk_level == "MEDIUM":

                st.warning(recommendation)

            else:

                st.success(recommendation)

        else:

            if risk_level == "HIGH":

                st.error(
                    "High churn risk detected. Consider proactive "
                    "retention action such as a personalized offer, "
                    "service review, or customer-support intervention."
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    "Moderate churn risk detected. Consider monitoring "
                    "this customer and providing targeted engagement "
                    "or service-support options."
                )

            else:

                st.success(
                    "Low churn risk detected. Continue normal customer "
                    "engagement and service monitoring."
                )


        # ----------------------------------------------------
        # CUSTOMER SUMMARY
        # ----------------------------------------------------

        st.subheader("👤 Customer Summary")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.markdown(
                f"""
                **Gender:** {gender}

                **Senior Citizen:** {
                    "Yes" if senior_citizen == 1 else "No"
                }

                **Partner:** {partner}

                **Dependents:** {dependents}

                **Tenure:** {tenure} months

                **Contract:** {contract}

                **Internet Service:** {internet_service}
                """
            )

        with summary_col2:

            st.markdown(
                f"""
                **Monthly Charges:** ${monthly_charges:,.2f}

                **Total Charges:** ${total_charges:,.2f}

                **Payment Method:** {payment_method}

                **Paperless Billing:** {paperless_billing}

                **Tech Support:** {tech_support}

                **Online Security:** {online_security}

                **Device Protection:** {device_protection}
                """
            )


        # ----------------------------------------------------
        # MODEL DECISION DETAILS
        # ----------------------------------------------------

        st.subheader("⚙️ Model Decision Details")

        if churn_prediction == 1:

            st.info(
                f"The model classified this customer as likely "
                f"to churn because the estimated probability "
                f"({churn_probability:.2%}) is above the saved "
                f"classification threshold ({final_threshold:.2%})."
            )

        else:

            st.info(
                f"The model classified this customer as likely "
                f"to stay because the estimated probability "
                f"({churn_probability:.2%}) is below the saved "
                f"classification threshold ({final_threshold:.2%})."
            )


# ============================================================
# MODEL INSIGHTS TAB
# ============================================================

with insights_tab:

    st.subheader("📈 Model Performance & Insights")

    st.write(
        "This section provides an overview of the model's "
        "performance and the features that contributed most "
        "to its predictions."
    )


    # --------------------------------------------------------
    # PREPARE EVALUATION DATA
    # --------------------------------------------------------

    evaluation_data = df.copy()

    evaluation_data["Churn"] = (
        evaluation_data["Churn"]
        .map({"No": 0, "Yes": 1})
    )

    X_eval = prepare_training_features(
        evaluation_data
    )

    y_eval = evaluation_data["Churn"]

    X_train_eval, X_test_eval, y_train_eval, y_test_eval = (
        train_test_split(
            X_eval,
            y_eval,
            test_size=0.20,
            random_state=42,
            stratify=y_eval
        )
    )


    # --------------------------------------------------------
    # MAKE SURE COLUMNS MATCH MODEL INPUT
    # --------------------------------------------------------

    X_test_eval = X_test_eval.reindex(
        columns=training_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_probability = model.predict_proba(
        X_test_eval
    )[:, 1]

    y_prediction = (
        y_probability >= final_threshold
    ).astype(int)


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test_eval,
        y_prediction
    )

    precision = precision_score(
        y_test_eval,
        y_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test_eval,
        y_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test_eval,
        y_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test_eval,
        y_probability
    )


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Accuracy",
            f"{accuracy:.2%}"
        )

    with c2:

        st.metric(
            "Precision",
            f"{precision:.2%}"
        )

    with c3:

        st.metric(
            "Recall",
            f"{recall:.2%}"
        )

    with c4:

        st.metric(
            "F1 Score",
            f"{f1:.2%}"
        )

    with c5:

        st.metric(
            "ROC-AUC",
            f"{roc_auc:.4f}"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test_eval,
        y_prediction
    )

    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:

        fig_cm, ax_cm = plt.subplots(
            figsize=(6, 4)
        )

        ax_cm.imshow(cm)

        ax_cm.set_title(
            "Confusion Matrix"
        )

        ax_cm.set_xlabel(
            "Predicted Label"
        )

        ax_cm.set_ylabel(
            "Actual Label"
        )

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])

        ax_cm.set_xticklabels(
            ["Stay", "Churn"]
        )

        ax_cm.set_yticklabels(
            ["Stay", "Churn"]
        )

        for i in range(2):

            for j in range(2):

                ax_cm.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )

        fig_cm.tight_layout()

        st.pyplot(fig_cm)

        plt.close(fig_cm)


    with cm_col2:

        st.markdown(
            """
            **How to read the confusion matrix**

            **True Negative (TN):** Customer predicted to stay
            and actually stayed.

            **False Positive (FP):** Customer predicted to churn
            but actually stayed.

            **False Negative (FN):** Customer predicted to stay
            but actually churned.

            **True Positive (TP):** Customer predicted to churn
            and actually churned.
            """
        )


    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(
        y_test_eval,
        y_probability
    )

    fig_roc, ax_roc = plt.subplots(
        figsize=(8, 5)
    )

    ax_roc.plot(
        fpr,
        tpr,
        label=f"Random Forest (AUC = {roc_auc:.3f})"
    )

    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax_roc.set_xlabel(
        "False Positive Rate"
    )

    ax_roc.set_ylabel(
        "True Positive Rate"
    )

    ax_roc.set_title(
        "ROC Curve"
    )

    ax_roc.legend()

    ax_roc.grid(alpha=0.2)

    fig_roc.tight_layout()

    st.pyplot(fig_roc)

    plt.close(fig_roc)


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🔎 Top Churn Drivers")

    if feature_importance is not None:

        fi = feature_importance.copy()

        # Try to identify columns automatically
        feature_col = None
        importance_col = None

        for col in fi.columns:

            if col.lower() in [
                "feature",
                "features",
                "feature_name"
            ]:

                feature_col = col

            if col.lower() in [
                "importance",
                "feature_importance"
            ]:

                importance_col = col


        if feature_col is not None and importance_col is not None:

            fi = fi.sort_values(
                importance_col,
                ascending=False
            ).head(15)

            fi_plot = fi.sort_values(
                importance_col,
                ascending=True
            )


            fig_fi, ax_fi = plt.subplots(
                figsize=(9, 6)
            )

            ax_fi.barh(
                fi_plot[feature_col],
                fi_plot[importance_col]
            )

            ax_fi.set_xlabel(
                "Importance"
            )

            ax_fi.set_ylabel(
                "Feature"
            )

            ax_fi.set_title(
                "Top 15 Feature Importances"
            )

            fig_fi.tight_layout()

            st.pyplot(fig_fi)

            plt.close(fig_fi)


            st.dataframe(
                fi,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Feature importance file was found, "
                "but its columns could not be identified."
            )

    else:

        st.info(
            "Feature importance file is not available."
        )


    # --------------------------------------------------------
    # BUSINESS INTERPRETATION
    # --------------------------------------------------------

    st.subheader("💼 Business Interpretation")

    st.markdown(
        """
        The model can help a telecom business prioritize
        customers for retention efforts.

        **Potential uses:**

        - Identify customers with higher estimated churn risk.
        - Prioritize proactive customer engagement.
        - Investigate customers with short tenure and high charges.
        - Review contract and payment patterns.
        - Provide targeted retention offers.
        - Monitor service-related customer segments.

        **Important:** Feature importance represents predictive
        relationships learned by the model. It does not prove
        that a feature directly causes customer churn.
        """
    )


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("🤖 Model Information")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.markdown(
            """
            **Algorithm**

            Random Forest Classifier
            """
        )

    with info2:

        st.markdown(
            f"""
            **Classification Threshold**

            {final_threshold:.4f}
            """
        )

    with info3:

        st.markdown(
            f"""
            **Training Features**

            {len(training_columns)}
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Customer Churn Intelligence • Machine Learning + Streamlit
        <br>
        Built as an end-to-end customer churn prediction project.
    </div>
    """,
    unsafe_allow_html=True
)