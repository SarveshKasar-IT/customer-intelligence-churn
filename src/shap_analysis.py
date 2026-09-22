from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.preprocessing import build_features


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/telco_customer_churn.csv")
MODEL_PATH = Path("models/stage3_xgboost_balanced.pkl")
PRODUCTION_MODEL_PATH = Path("models/customer_churn_random_forest.pkl")

OUTPUT_DIR = Path("data/processed/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 1000


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("SHAP EXPLAINABILITY - CUSTOMER CHURN")
print("=" * 70)

print("\n[1/6] Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
).fillna(0)

df["Churn"] = (
    df["Churn"]
    .map({"Yes": 1, "No": 0})
    .astype(int)
)

X = df.drop(columns=["Churn"])

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. LOAD EXPERIMENTAL MODEL
# ============================================================

print("\n[2/6] Loading Stage 3 XGBoost Balanced model...")

model = joblib.load(MODEL_PATH)

production_model = joblib.load(PRODUCTION_MODEL_PATH)

production_columns = list(
    production_model.feature_names_in_
)

print(f"Model: {type(model).__name__}")
print(f"Expected features: {len(production_columns)}")


# ============================================================
# 3. APPLY PRODUCTION PREPROCESSING
# ============================================================

print("\n[3/6] Applying production preprocessing...")

X_processed = build_features(
    X,
    production_columns
)

print(
    f"Processed dataset shape: {X_processed.shape}"
)

if X_processed.shape[1] != len(production_columns):
    raise ValueError(
        "Feature count does not match production representation."
    )

# Use a manageable sample for SHAP
if len(X_processed) > SAMPLE_SIZE:
    X_shap = X_processed.sample(
        SAMPLE_SIZE,
        random_state=RANDOM_STATE
    )
else:
    X_shap = X_processed.copy()

print(
    f"SHAP sample shape: {X_shap.shape}"
)


# ============================================================
# 4. CREATE SHAP EXPLAINER
# ============================================================

print("\n[4/6] Creating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_shap)

# SHAP 0.52 can return different structures depending
# on the model/API combination.
if isinstance(shap_values, list):
    shap_values_for_plot = shap_values[1]
else:
    shap_values_for_plot = shap_values

print(
    f"SHAP values shape: {shap_values_for_plot.shape}"
)


# ============================================================
# 5. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\n[5/6] Creating global SHAP explanations...")

# ------------------------------------------------------------
# SHAP BAR PLOT
# ------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values_for_plot,
    X_shap,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

bar_path = OUTPUT_DIR / "shap_global_bar.png"

plt.savefig(
    bar_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# SHAP BEESWARM
# ------------------------------------------------------------

plt.figure()

shap.summary_plot(
    shap_values_for_plot,
    X_shap,
    show=False,
    max_display=20
)

plt.tight_layout()

beeswarm_path = OUTPUT_DIR / "shap_beeswarm.png"

plt.savefig(
    beeswarm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# GLOBAL IMPORTANCE TABLE
# ------------------------------------------------------------

importance = pd.DataFrame(
    {
        "feature": X_shap.columns,
        "mean_abs_shap": (
            abs(shap_values_for_plot)
            .mean(axis=0)
        ),
    }
)

importance = (
    importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_path = (
    OUTPUT_DIR /
    "shap_feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# 6. INDIVIDUAL CUSTOMER EXPLANATION
# ============================================================

print("\n[6/6] Creating individual customer explanation...")

customer_index = 0

customer_row = X_shap.iloc[
    customer_index:customer_index + 1
]

customer_shap = shap_values_for_plot[
    customer_index
]

# Waterfall plot
explanation = shap.Explanation(
    values=customer_shap,
    base_values=explainer.expected_value,
    data=customer_row.iloc[0].values,
    feature_names=customer_row.columns.tolist(),
)

plt.figure()

shap.plots.waterfall(
    explanation,
    max_display=15,
    show=False
)

plt.tight_layout()

waterfall_path = (
    OUTPUT_DIR /
    "shap_individual_waterfall.png"
)

plt.savefig(
    waterfall_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DISPLAY TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP CHURN DRIVERS")
print("=" * 70)

print(
    importance.head(15).to_string(index=False)
)


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(f"  {bar_path}")
print(f"  {beeswarm_path}")
print(f"  {importance_path}")
print(f"  {waterfall_path}")

print("\nModel analyzed:")
print("  XGBoost Balanced")

print("\nProduction preprocessing:")
print("  33 features")

print("\nNext:")
print("  Review SHAP results and integrate explainability")
print("=" * 70)