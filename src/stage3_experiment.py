"""
Stage 3 - Customer Churn Model Experiments

Experiments:
1. Random Forest baseline
2. Random Forest with class balancing
3. XGBoost
4. XGBoost with class balancing

Tracks experiments using MLflow.

Important:
- Uses the SAME preprocessing function as production.
- Does NOT modify the deployed FastAPI model.
- Does NOT modify Streamlit.
"""

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from xgboost import XGBClassifier

from src.preprocessing import build_features


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/raw/telco_customer_churn.csv"
)

PRODUCTION_MODEL_PATH = Path(
    "models/customer_churn_random_forest.pkl"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

MLFLOW_EXPERIMENT_NAME = (
    "Customer Churn - Stage 3"
)

# Use SQLite as the local MLflow tracking backend.
# This is the recommended backend for current MLflow versions.
MLFLOW_TRACKING_URI = (
    "sqlite:///mlflow.db"
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)

print("=" * 70)
print("STAGE 3 - MODEL IMPROVEMENT + MLFLOW")
print("=" * 70)

print(
    f"\nMLflow experiment: "
    f"{MLFLOW_EXPERIMENT_NAME}"
)

print(
    f"MLflow tracking URI: "
    f"{MLFLOW_TRACKING_URI}"
)

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/8] Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset shape: {df.shape}"
)


# ============================================================
# 3. CLEAN DATA
# ============================================================

print("\n[2/8] Cleaning dataset...")

if "TotalCharges" in df.columns:

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

if "customerID" in df.columns:

    df = df.drop(
        columns=["customerID"]
    )

if "Churn" not in df.columns:

    raise ValueError(
        "Target column 'Churn' not found."
    )

df["Churn"] = (
    df["Churn"]
    .astype(str)
    .str.strip()
    .map({
        "Yes": 1,
        "No": 0
    })
)

df = df.dropna(
    subset=["Churn"]
)

df["Churn"] = df["Churn"].astype(int)


print(
    f"Cleaned dataset shape: {df.shape}"
)

print("\nTarget distribution:")

print(
    df["Churn"].value_counts()
)

print("\nTarget percentages:")

print(
    (
        df["Churn"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# 4. X / y
# ============================================================

X = df.drop(
    columns=["Churn"]
)

y = df["Churn"]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

print(
    "\n[3/8] Creating train/test split..."
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
)


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows : {len(X_test)}"
)


# ============================================================
# 6. PRODUCTION-COMPATIBLE PREPROCESSING
# ============================================================

print(
    "\n[4/8] Applying production preprocessing..."
)


# Load the production Random Forest only to retrieve
# its exact feature names.
production_model = joblib.load(
    PRODUCTION_MODEL_PATH
)

production_columns = list(
    production_model.feature_names_in_
)


print(
    f"Production model expects "
    f"{len(production_columns)} features."
)


X_train_processed = build_features(
    X_train,
    production_columns
)

X_test_processed = build_features(
    X_test,
    production_columns
)


print(
    f"Processed training shape: "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape: "
    f"{X_test_processed.shape}"
)


if X_train_processed.shape[1] != len(
    production_columns
):

    raise ValueError(
        "Feature count mismatch between "
        "Stage 3 preprocessing and production model."
    )


print(
    "\nSUCCESS: Stage 3 is using the "
    "same feature representation as production."
)


# ============================================================
# 7. CLASS IMBALANCE
# ============================================================

print(
    "\n[5/8] Calculating class imbalance..."
)


negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative_count /
    positive_count
)


print(
    f"Non-churn customers: "
    f"{negative_count}"
)

print(
    f"Churn customers    : "
    f"{positive_count}"
)

print(
    f"scale_pos_weight   : "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================

print(
    "\n[6/8] Creating models..."
)


models = {

    "Random Forest Baseline":
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=None,
        ),

    "Random Forest Balanced":
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),

    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

    "XGBoost Balanced":
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
}


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    model
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"TRAINING: {name}"
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=name
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train_processed,
            y_train
        )


        # ----------------------------------------------------
        # Predict probabilities
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_test_processed
            )[:, 1]
        )


        # ----------------------------------------------------
        # Default threshold
        # ----------------------------------------------------

        default_threshold = 0.50

        predictions = (
            probabilities >=
            default_threshold
        ).astype(int)


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = (
            average_precision_score(
                y_test,
                probabilities
            )
        )


        # ----------------------------------------------------
        # Log common parameters
        # ----------------------------------------------------

        mlflow.log_params({

            "model_name": name,

            "model_type":
                type(model).__name__,

            "random_state":
                RANDOM_STATE,

            "test_size":
                TEST_SIZE,

            "n_features":
                X_train_processed.shape[1],

            "training_rows":
                X_train_processed.shape[0],

            "testing_rows":
                X_test_processed.shape[0],

            "positive_training_samples":
                int(positive_count),

            "negative_training_samples":
                int(negative_count),

            "class_ratio":
                float(scale_pos_weight),

            "default_threshold":
                default_threshold,
        })


        # ----------------------------------------------------
        # Log model-specific parameters
        # ----------------------------------------------------

        if isinstance(
            model,
            RandomForestClassifier
        ):

            mlflow.log_params({

                "n_estimators":
                    model.n_estimators,

                "max_depth":
                    str(model.max_depth),

                "class_weight":
                    str(model.class_weight),

                "max_features":
                    str(model.max_features),

            })


        elif isinstance(
            model,
            XGBClassifier
        ):

            mlflow.log_params({

                "n_estimators":
                    model.n_estimators,

                "max_depth":
                    model.max_depth,

                "learning_rate":
                    model.learning_rate,

                "subsample":
                    model.subsample,

                "colsample_bytree":
                    model.colsample_bytree,

                "scale_pos_weight":
                    model.scale_pos_weight,

                "objective":
                    model.objective,

                "eval_metric":
                    model.eval_metric,

            })


        # ----------------------------------------------------
        # Log primary metrics
        # ----------------------------------------------------

        mlflow.log_metrics({

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

        })


        # ----------------------------------------------------
        # Threshold analysis
        # ----------------------------------------------------

        thresholds = [
            0.20,
            0.25,
            0.30,
            0.32,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ]


        threshold_results = []


        for threshold in thresholds:

            threshold_predictions = (
                probabilities >= threshold
            ).astype(int)


            threshold_precision = (
                precision_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                )
            )

            threshold_recall = (
                recall_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                )
            )

            threshold_f1 = (
                f1_score(
                    y_test,
                    threshold_predictions,
                    zero_division=0
                )
            )


            threshold_results.append({

                "Threshold":
                    threshold,

                "Precision":
                    threshold_precision,

                "Recall":
                    threshold_recall,

                "F1":
                    threshold_f1,

                "Churn_Predictions":
                    int(
                        threshold_predictions.sum()
                    ),

            })


        threshold_df = pd.DataFrame(
            threshold_results
        )


        # ----------------------------------------------------
        # Best threshold according to F1
        # ----------------------------------------------------

        best_row = threshold_df.loc[
            threshold_df["F1"].idxmax()
        ]


        best_threshold = float(
            best_row["Threshold"]
        )

        best_threshold_f1 = float(
            best_row["F1"]
        )

        best_threshold_precision = float(
            best_row["Precision"]
        )

        best_threshold_recall = float(
            best_row["Recall"]
        )


        mlflow.log_metrics({

            "best_f1_threshold":
                best_threshold,

            "best_threshold_f1":
                best_threshold_f1,

            "precision_at_best_threshold":
                best_threshold_precision,

            "recall_at_best_threshold":
                best_threshold_recall,

        })


        # ----------------------------------------------------
        # Save threshold analysis
        # ----------------------------------------------------

        safe_name = (
            name
            .lower()
            .replace(" ", "_")
        )


        threshold_path = Path(
            "data/processed"
        ) / (
            f"threshold_{safe_name}.csv"
        )


        threshold_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        threshold_df.to_csv(
            threshold_path,
            index=False
        )


        mlflow.log_artifact(
            str(threshold_path)
        )


        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions
        )


        cm_df = pd.DataFrame(

            cm,

            index=[
                "Actual_No_Churn",
                "Actual_Churn",
            ],

            columns=[
                "Predicted_No_Churn",
                "Predicted_Churn",
            ],
        )


        cm_path = Path(
            "data/processed"
        ) / (
            f"confusion_matrix_{safe_name}.csv"
        )


        cm_df.to_csv(
            cm_path
        )


        mlflow.log_artifact(
            str(cm_path)
        )


        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report = classification_report(

            y_test,

            predictions,

            target_names=[
                "No Churn",
                "Churn",
            ],

            zero_division=0,

        )


        report_path = Path(
            "data/processed"
        ) / (
            f"classification_report_{safe_name}.txt"
        )


        report_path.write_text(
            report,
            encoding="utf-8"
        )


        mlflow.log_artifact(
            str(report_path)
        )


        # ----------------------------------------------------
        # Log trained model
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )


        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1       : {f1:.4f}"
        )

        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC   : {pr_auc:.4f}"
        )

        print(
            f"Best F1 threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Best F1: "
            f"{best_threshold_f1:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

        print(
            "\nClassification Report:"
        )

        print(report)


        run_id = (
            mlflow.active_run()
            .info
            .run_id
        )


        print(
            f"\nMLflow Run ID: "
            f"{run_id}"
        )


    return {

        "Model":
            name,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "ROC_AUC":
            roc_auc,

        "PR_AUC":
            pr_auc,

        "Best_Threshold":
            best_threshold,

        "Best_Threshold_F1":
            best_threshold_f1,

        "ModelObject":
            model,

        "Probabilities":
            probabilities,

        "Run_ID":
            run_id,

    }


# ============================================================
# 10. TRAIN ALL MODELS
# ============================================================

print(
    "\n[7/8] Training models + logging to MLflow..."
)


results = []


for model_name, model in models.items():

    result = evaluate_model(
        model_name,
        model
    )

    results.append(
        result
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 70
)


comparison = pd.DataFrame([

    {

        "Model":
            result["Model"],

        "Accuracy":
            result["Accuracy"],

        "Precision":
            result["Precision"],

        "Recall":
            result["Recall"],

        "F1":
            result["F1"],

        "ROC_AUC":
            result["ROC_AUC"],

        "PR_AUC":
            result["PR_AUC"],

        "Best_Threshold":
            result["Best_Threshold"],

        "Best_Threshold_F1":
            result["Best_Threshold_F1"],

    }

    for result in results

])


print(
    comparison.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# 12. SAVE COMPARISON
# ============================================================

output_dir = Path(
    "data/processed"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


comparison_path = (
    output_dir /
    "stage3_model_comparison.csv"
)


comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 13. SAVE XGBOOST BALANCED MODEL
# ============================================================

print(
    "\n[8/8] Saving experimental XGBoost model..."
)


xgb_result = next(

    result

    for result in results

    if result["Model"]
    == "XGBoost Balanced"

)


xgb_model_path = Path(
    "models/stage3_xgboost_balanced.pkl"
)


joblib.dump(
    xgb_result["ModelObject"],
    xgb_model_path
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STAGE 3 + MLFLOW COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nMLflow experiment:"
)

print(
    f"  {MLFLOW_EXPERIMENT_NAME}"
)


print(
    "\nTracked:"
)

print(
    "  ✓ Model parameters"
)

print(
    "  ✓ Dataset information"
)

print(
    "  ✓ Class imbalance"
)

print(
    "  ✓ Accuracy"

)

print(
    "  ✓ Precision"
)

print(
    "  ✓ Recall"
)

print(
    "  ✓ F1"
)

print(
    "  ✓ ROC-AUC"
)

print(
    "  ✓ PR-AUC"
)

print(
    "  ✓ Threshold analysis"
)

print(
    "  ✓ Confusion matrices"
)

print(
    "  ✓ Classification reports"
)

print(
    "  ✓ Trained models"
)


print(
    "\nSaved:"
)

print(
    f"  {comparison_path}"
)

print(
    f"  {xgb_model_path}"
)


print(
    "\nProduction API:"
)

print(
    "  UNCHANGED"
)

print(
    "\nStreamlit:"
)

print(
    "  UNCHANGED"
)

print(
    "\nNext:"
)

print(
    "  SHAP explainability"
)

print(
    "=" * 70
)