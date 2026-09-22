from pathlib import Path
import mlflow
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Customer Churn - Stage 3"

SHAP_DIR = Path("data/processed/shap")


# ============================================================
# MLflow setup
# ============================================================

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# Validate SHAP artifacts
# ============================================================

required_files = [
    SHAP_DIR / "shap_global_bar.png",
    SHAP_DIR / "shap_beeswarm.png",
    SHAP_DIR / "shap_feature_importance.csv",
    SHAP_DIR / "shap_individual_waterfall.png",
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing SHAP artifact: {file_path}"
        )


# ============================================================
# Read feature importance
# ============================================================

importance_path = (
    SHAP_DIR / "shap_feature_importance.csv"
)

importance = pd.read_csv(importance_path)


# ============================================================
# Create MLflow run
# ============================================================

with mlflow.start_run(
    run_name="SHAP Explainability - XGBoost Balanced"
) as run:

    mlflow.log_param(
        "explained_model",
        "XGBoost Balanced"
    )

    mlflow.log_param(
        "feature_representation",
        "Production 33 features"
    )

    mlflow.log_param(
        "shap_sample_size",
        1000
    )

    mlflow.log_metric(
        "top_feature_mean_abs_shap",
        float(
            importance.iloc[0]["mean_abs_shap"]
        )
    )

    # Log all SHAP artifacts
    mlflow.log_artifact(
        str(SHAP_DIR / "shap_global_bar.png"),
        artifact_path="shap"
    )

    mlflow.log_artifact(
        str(SHAP_DIR / "shap_beeswarm.png"),
        artifact_path="shap"
    )

    mlflow.log_artifact(
        str(SHAP_DIR / "shap_feature_importance.csv"),
        artifact_path="shap"
    )

    mlflow.log_artifact(
        str(SHAP_DIR / "shap_individual_waterfall.png"),
        artifact_path="shap"
    )

    print("=" * 70)
    print("SHAP ARTIFACTS LOGGED TO MLFLOW")
    print("=" * 70)

    print(f"\nMLflow Run ID: {run.info.run_id}")

    print("\nLogged:")
    print("  ✓ Global SHAP bar plot")
    print("  ✓ SHAP beeswarm plot")
    print("  ✓ Feature importance CSV")
    print("  ✓ Individual waterfall plot")

    print("\nExperiment:")
    print(f"  {EXPERIMENT_NAME}")

    print("=" * 70)