
"""
Healthcare Provider Fraud-Risk Detection
Step 3: Model Evaluation and Explainability

This script:
1. Loads a previously trained provider-level classification model.
2. Evaluates global feature importance using permutation importance.
3. Generates ROC and precision-recall curves.
4. Saves evaluation metrics, feature importance, and visualizations.

The analysis evaluates predictions of the historical PotentialFraud label.
Permutation importance measures global model reliance on features;
it does not establish causal effects or individual prediction explanations.
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Supports scripts stored in the repository root or in src/.
PROJECT_DIR = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"src", "scripts"}
    else SCRIPT_DIR
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_input(filename):
    """
    Locate a required input in common project directories.
    """
    search_dirs = [
        PROJECT_DIR,
        PROJECT_DIR / "data",
        PROJECT_DIR / "models",
        PROJECT_DIR / "outputs",
        PROJECT_DIR / "artifacts",
        PROJECT_DIR / "legacy_original",
    ]

    for directory in search_dirs:
        path = directory / filename
        if path.is_file():
            return path

    searched = "\n".join(str(p / filename) for p in search_dirs)
    raise FileNotFoundError(
        f"Required file '{filename}' was not found.\n"
        f"Checked:\n{searched}\n\n"
        "Run the training pipeline first or restore the required artifact."
    )


# ------------------------------------------------------------
# 2. Load model and data
# ------------------------------------------------------------

model_path = find_input("primary_model.pkl")
train_path = find_input("X_train.csv")
test_path = find_input("test_predictions.csv")

with model_path.open("rb") as file:
    model = pickle.load(file)

X_train = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

feature_cols = X_train.columns.tolist()

required_cols = feature_cols + ["Label"]
missing_cols = [c for c in required_cols if c not in test_df.columns]

if missing_cols:
    raise ValueError(
        f"Missing columns in test_predictions.csv: {missing_cols}"
    )

if X_train.empty or test_df.empty:
    raise ValueError("Training features and test data must not be empty.")

X_test = test_df.loc[:, feature_cols].copy()
y_test = test_df["Label"]

if y_test.isna().any():
    raise ValueError("Test labels contain missing values.")

if not set(y_test.unique()).issubset({0, 1}):
    raise ValueError("Test labels must be binary values 0 and 1.")

if y_test.nunique() != 2:
    raise ValueError("ROC-AUC requires both label classes in the test set.")

if X_test.isna().any().any():
    raise ValueError(
        "Test features contain missing values. "
        "Use the same preprocessing pipeline as model training."
    )

if not hasattr(model, "predict_proba"):
    raise TypeError(
        "The loaded model must support predict_proba()."
    )

if not hasattr(model, "classes_"):
    raise TypeError(
        "The loaded model does not expose class labels."
    )

classes = list(model.classes_)

if 1 not in classes:
    raise ValueError(
        "The trained model must include positive class 1."
    )

positive_index = classes.index(1)

print("Model and datasets loaded successfully.")
print(f"Training providers: {len(X_train):,}")
print(f"Test providers: {len(X_test):,}")
print(f"Features: {len(feature_cols)}")


# ------------------------------------------------------------
# 3. Calculate test-set predictions and metrics
# ------------------------------------------------------------

y_score = model.predict_proba(X_test)[:, positive_index]

roc_auc = roc_auc_score(y_test, y_score)
average_precision = average_precision_score(y_test, y_score)
test_prevalence = float(y_test.mean())

metrics = {
    "test_providers": int(len(X_test)),
    "feature_count": len(feature_cols),
    "roc_auc": float(roc_auc),
    "average_precision": float(average_precision),
    "test_positive_label_rate": test_prevalence,
}

with (OUTPUT_DIR / "explainability_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

print(f"\nTest ROC-AUC: {roc_auc:.4f}")
print(f"Test Average Precision: {average_precision:.4f}")


# ------------------------------------------------------------
# 4. Global permutation importance
# ------------------------------------------------------------

print("\nCalculating permutation importance...")

perm_result = permutation_importance(
    estimator=model,
    X=X_test,
    y=y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values(
    by="importance_mean",
    ascending=False,
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False,
)

print("\nTop 10 features by permutation importance:")
print(importance_df.head(10).to_string(index=False))


# ------------------------------------------------------------
# 5. Feature importance visualization
# ------------------------------------------------------------

top_features = importance_df.head(12).iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    top_features["feature"],
    top_features["importance_mean"],
    xerr=top_features["importance_std"],
    color="#2E5C8A",
    capsize=3,
)

ax.axvline(0, color="gray", linewidth=0.8)

ax.set_xlabel("Mean decrease in ROC-AUC after permutation")
ax.set_title("Global Feature Importance — Provider-Level Model")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# 6. ROC and precision-recall curves
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_score)
precision, recall, _ = precision_recall_curve(y_test, y_score)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ROC curve
axes[0].plot(
    fpr,
    tpr,
    color="#2E5C8A",
    linewidth=2,
    label=f"Model (ROC-AUC = {roc_auc:.3f})",
)

axes[0].plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random classifier",
)

axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve — Held-Out Test Set")
axes[0].legend()

# Precision-recall curve
axes[1].plot(
    recall,
    precision,
    color="#2E5C8A",
    linewidth=2,
    label=f"Model (AP = {average_precision:.3f})",
)

axes[1].axhline(
    y=test_prevalence,
    linestyle="--",
    color="gray",
    label="No-skill baseline",
)

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Held-Out Test Set")
axes[1].legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "roc_pr_curves.png",
    dpi=150,
    bbox_inches="tight",
)

plt.close(fig)


# ------------------------------------------------------------
# 7. Completion summary
# ------------------------------------------------------------

print("\nAnalysis completed successfully.")
print(f"Outputs saved in: {OUTPUT_DIR}")

for filename in [
    "explainability_metrics.json",
    "feature_importance.csv",
    "feature_importance.png",
    "roc_pr_curves.png",
]:
    print(f"  - {filename}")
