"""
Provider-Level Healthcare Claims Fraud Detection
Step 3: Explainability

Uses permutation importance (a model-agnostic, statistically grounded explainability
method built into scikit-learn) to identify which features drive the primary model's
fraud predictions, and produces per-provider explanations for a handful of flagged
cases -- the "interpretable investigative risk scoring" output an investigator would
actually use.

NOTE ON SHAP: the sandboxed environment used to build this pipeline has no internet
access, so the `shap` package could not be installed here. Permutation importance is
a legitimate, peer-reviewed, model-agnostic alternative that ships with scikit-learn
and requires no extra dependency. A ready-to-run SHAP cell is included at the bottom
of this file, commented out, for use in a normal environment (e.g. Google Colab or a
local machine with internet access) -- run that separately and do not cite SHAP output
values that have not actually been generated.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, precision_recall_curve

with open("/home/claude/fraud_analysis/primary_model.pkl", "rb") as f:
    model = pickle.load(f)

X_train = pd.read_csv("/home/claude/fraud_analysis/X_train.csv")
test_df = pd.read_csv("/home/claude/fraud_analysis/test_predictions.csv")

feature_cols = [c for c in X_train.columns]
X_test = test_df[feature_cols]
y_test = test_df["Label"]

# ---------------------------------------------------------------------------
# 1. Global feature importance via permutation importance
#    (measures the drop in model performance when each feature is randomly
#    shuffled -- a feature that matters a lot will hurt performance a lot
#    when scrambled; this is model-agnostic and not biased toward
#    high-cardinality features the way raw impurity-based importance can be)
# ---------------------------------------------------------------------------
perm_result = permutation_importance(
    model, X_test, y_test, n_repeats=20, random_state=42, scoring="roc_auc"
)

importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values("importance_mean", ascending=False)

print("Top 10 features by permutation importance (impact on ROC-AUC):")
print(importance_df.head(10).to_string(index=False))

importance_df.to_csv("/home/claude/fraud_analysis/feature_importance.csv", index=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(9, 6))
top_n = importance_df.head(12).iloc[::-1]
ax.barh(top_n["feature"], top_n["importance_mean"], xerr=top_n["importance_std"], color="#2E5C8A")
ax.set_xlabel("Permutation Importance (mean decrease in ROC-AUC when shuffled)")
ax.set_title("Feature Importance — Provider-Level Healthcare Claims Fraud Model")
plt.tight_layout()
plt.savefig("/home/claude/fraud_analysis/feature_importance.png", dpi=150)
print("\nSaved feature_importance.png")

# ---------------------------------------------------------------------------
# 2. ROC and Precision-Recall curves (baseline vs primary), for the repo README
# ---------------------------------------------------------------------------
import json
with open("/home/claude/fraud_analysis/results_summary.json") as f:
    summary = json.load(f)

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
prec, rec, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, color="#2E5C8A", label=f"Primary model (ROC-AUC={summary['primary_roc_auc']:.3f})")
axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Random")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].set_title("ROC Curve — Test Set")
axes[0].legend()

axes[1].plot(rec, prec, color="#2E5C8A", label=f"Primary model (PR-AUC={summary['primary_pr_auc']:.3f})")
axes[1].axhline(y=summary["fraud_rate_overall"], linestyle="--", color="gray", label="No-skill baseline")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve — Test Set")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/fraud_analysis/roc_pr_curves.png", dpi=150)
print("Saved roc_pr_curves.png")

# ---------------------------------------------------------------------------
# 3. Per-provider explanation example: for a provider flagged as high-risk,
#    show which features pushed the score up (a simplified, interpretable
#    "reason code" style explanation an investigator could act on)
# ---------------------------------------------------------------------------
test_df_sorted = test_df.sort_values("PrimaryModelProba", ascending=False)
example = test_df_sorted.iloc[0]

train_means = X_train.mean()
print("\n" + "=" * 70)
print("EXAMPLE: highest-risk-scored provider in the test set")
print("=" * 70)
print(f"Model-assigned fraud probability: {example['PrimaryModelProba']:.3f}")
print(f"Actual label: {'Fraud' if example['Label'] == 1 else 'No Fraud'}")
print("\nFeatures most above the training-set average (top contributors to risk score):")
deviations = (example[feature_cols] - train_means) / train_means.replace(0, 1)
top_deviations = deviations.sort_values(ascending=False).head(5)
for feat, dev in top_deviations.items():
    print(f"  {feat}: {example[feat]:.2f} (population avg: {train_means[feat]:.2f}, {dev*100:+.0f}%)")

print("\nDone. All outputs saved to /home/claude/fraud_analysis/")

# ---------------------------------------------------------------------------
# OPTIONAL — run this separately in an environment with internet access to
# generate true SHAP values. Not run here; included for completeness only.
# ---------------------------------------------------------------------------
"""
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")

# Per-provider explanation (e.g. the highest-risk provider)
shap.force_plot(
    explainer.expected_value, shap_values[0], X_test.iloc[0],
    matplotlib=True, show=False
)
plt.savefig("shap_force_plot_example.png", dpi=150, bbox_inches="tight")
"""
