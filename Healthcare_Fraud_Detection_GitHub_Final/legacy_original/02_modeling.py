
# ============================================================
# PROJECT 1 — STEP 2
# PROVIDER-LEVEL HEALTHCARE FRAUD-RISK DETECTION
# Model Training, Evaluation and Results Export
# ============================================================

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. SETTINGS AND AUTOMATIC FILE LOCATION
# ------------------------------------------------------------

RANDOM_STATE = 42
TARGET_RECALL = 0.70

def find_project_root():
    current = Path.cwd().resolve()

    for folder in [current, *current.parents]:
        for candidate in [
            folder / "outputs" / "provider_level_features.csv",
            folder / "provider_level_features.csv"
        ]:
            if candidate.is_file():
                return folder, candidate

    raise FileNotFoundError(
        "Cannot find provider_level_features.csv. "
        "Run Step 1 (feature engineering) first, "
        "or open Jupyter from your project folder."
    )

ROOT, DATA_FILE = find_project_root()

OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

print("Project folder:", ROOT)
print("Dataset:", DATA_FILE)

# ------------------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

required = {"Provider", "Label"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["Provider"].isna().any():
    raise ValueError("Provider identifiers contain missing values.")

if df["Provider"].duplicated().any():
    raise ValueError(
        "Duplicate providers detected. "
        "The dataset must contain one row per provider."
    )

if df["Label"].isna().any():
    raise ValueError("Missing provider labels detected.")

if not set(df["Label"].unique()).issubset({0, 1}):
    raise ValueError("Label must contain only 0 and 1.")

if df["Label"].nunique() != 2:
    raise ValueError("Both label classes are required.")

feature_cols = [
    col for col in df.columns
    if col not in ["Provider", "PotentialFraud", "Label"]
]

X = df[feature_cols].copy()
y = df["Label"].astype(int)

non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()

if non_numeric:
    raise ValueError(
        f"Non-numeric model features detected: {non_numeric}"
    )

X = X.replace([np.inf, -np.inf], np.nan)

empty_features = X.columns[X.isna().all()].tolist()

if empty_features:
    raise ValueError(
        f"Features contain only missing values: {empty_features}"
    )

print("\nDATASET SUMMARY")
print("Providers:", len(df))
print("Features:", len(feature_cols))
print("Positive historical labels:", int(y.sum()))
print("Positive label rate:", round(y.mean() * 100, 2), "%")

# ------------------------------------------------------------
# 3. TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=43,
    stratify=y_dev
)

assert set(X_train.index).isdisjoint(X_val.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_val.index).isdisjoint(X_test.index)

print("\nDATA SPLIT")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ------------------------------------------------------------
# 4. DEFINE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ),

    "Gradient Boosting": make_pipeline(
        SimpleImputer(strategy="median"),
        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )
}

# ------------------------------------------------------------
# 5. SELECT THRESHOLDS USING VALIDATION DATA ONLY
# ------------------------------------------------------------

def select_threshold(y_true, scores, target_recall):
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    valid = np.where(
        (tpr >= target_recall) &
        np.isfinite(thresholds)
    )[0]

    if len(valid) == 0:
        raise ValueError("No valid threshold found.")

    # Lowest FPR among thresholds meeting target recall.
    best_fpr = fpr[valid].min()
    candidates = valid[np.isclose(fpr[valid], best_fpr)]

    # Prefer the highest threshold when FPR is tied.
    best = candidates[np.argmax(thresholds[candidates])]

    return float(thresholds[best])

# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

results = []
prediction_rows = []
trained_models = {}
thresholds = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("MODEL:", name)
    print("=" * 55)

    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    test_scores = model.predict_proba(X_test)[:, 1]

    threshold = select_threshold(
        y_val,
        val_scores,
        TARGET_RECALL
    )

    predictions = (test_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    results.append({
        "Model": name,
        "Threshold": threshold,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "FalsePositiveRate": fpr,
        "TruePositive": int(tp),
        "FalsePositive": int(fp),
        "TrueNegative": int(tn),
        "FalseNegative": int(fn)
    })

    trained_models[name] = model
    thresholds[name] = threshold

    print("ROC-AUC:", round(roc_auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("False Positive Rate:", round(fpr, 4))
    print("Validation-selected threshold:", round(threshold, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Historical Negative",
                "Historical Positive"
            ],
            zero_division=0
        )
    )

    for idx, score, pred in zip(
        X_test.index,
        test_scores,
        predictions
    ):
        prediction_rows.append({
            "Provider": df.loc[idx, "Provider"],
            "Model": name,
            "RiskScore": float(score),
            "PredictedFlag": int(pred),
            "ActualProviderLabel": int(y.loc[idx])
        })

# ------------------------------------------------------------
# 7. COMPARE MODELS AT VALIDATION-SELECTED THRESHOLDS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

baseline_row = results_df.loc[
    results_df["Model"] == "Logistic Regression"
].iloc[0]

primary_row = results_df.loc[
    results_df["Model"] == "Gradient Boosting"
].iloc[0]

baseline_fpr = float(baseline_row["FalsePositiveRate"])
primary_fpr = float(primary_row["FalsePositiveRate"])

fpr_difference_pp = (baseline_fpr - primary_fpr) * 100

relative_reduction = (
    (baseline_fpr - primary_fpr) / baseline_fpr * 100
    if baseline_fpr > 0 else None
)

print("\nFALSE POSITIVE RATE COMPARISON")
print("Target validation recall:", TARGET_RECALL)
print("Baseline test recall:", round(float(baseline_row["Recall"]), 4))
print("Primary test recall:", round(float(primary_row["Recall"]), 4))
print("Baseline test FPR:", round(baseline_fpr, 4))
print("Primary test FPR:", round(primary_fpr, 4))
print("FPR difference (percentage points):", round(fpr_difference_pp, 2))

if relative_reduction is not None:
    print("Relative FPR reduction (%):", round(relative_reduction, 2))

print(
    "\nNOTE: Both thresholds were chosen to meet the same "
    "validation recall target. Actual test recalls may differ. "
    "This is not a strict matched-test-recall comparison."
)

# ------------------------------------------------------------
# 8. SAVE RESULTS
# ------------------------------------------------------------

results_df.to_csv(
    OUT / "model_comparison.csv",
    index=False
)

predictions_df = pd.DataFrame(prediction_rows)

predictions_df.to_csv(
    OUT / "test_predictions_all_models.csv",
    index=False
)

# Primary model predictions for downstream scripts.
primary_predictions = predictions_df.loc[
    predictions_df["Model"] == "Gradient Boosting"
].copy()

primary_predictions.to_csv(
    OUT / "test_predictions.csv",
    index=False
)

# Save the complete fitted pipeline.
with open(OUT / "primary_model.pkl", "wb") as f:
    pickle.dump(trained_models["Gradient Boosting"], f)

X_train.to_csv(
    OUT / "X_train.csv",
    index=False
)

summary = {
    "n_providers_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_validation": int(len(X_val)),
    "n_test": int(len(X_test)),
    "positive_label_rate": float(y.mean()),
    "target_validation_recall": TARGET_RECALL,
    "models": {
        row["Model"]: {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in row.items()
            if key != "Model"
        }
        for row in results
    },
    "fpr_difference_percentage_points": fpr_difference_pp,
    "relative_fpr_reduction_pct": relative_reduction,
    "comparison_note": (
        "Thresholds selected on validation data. "
        "Test recalls may differ; not a strict "
        "matched-test-recall comparison."
    )
}

with open(OUT / "results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nFILES SAVED TO:", OUT)
print("1. model_comparison.csv")
print("2. test_predictions_all_models.csv")
print("3. test_predictions.csv")
print("4. primary_model.pkl")
print("5. X_train.csv")
print("6. results_summary.json")

print("\nSTEP 2 COMPLETED SUCCESSFULLY")
