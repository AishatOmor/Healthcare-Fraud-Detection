"""
Provider-Level Healthcare Claims Fraud Detection
Step 2: Baseline and Primary Model Training + Evaluation

Trains a simple Logistic Regression baseline and a Gradient Boosted Trees primary
model on the provider-level features built in step 1. Reports the metrics needed
to substantiate an NIW petition's technical claims: precision, recall, ROC-AUC,
PR-AUC, and a false-positive-rate comparison between the two models AT A MATCHED
RECALL LEVEL, so the comparison is apples-to-apples rather than an arbitrary
threshold pick.

IMPORTANT: All numbers printed by this script are real, computed from the actual
data. Nothing here is a placeholder or an assumed result.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report
)
import json

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load features and split by PROVIDER (the label's grain) into train/test
# ---------------------------------------------------------------------------
df = pd.read_csv("/home/claude/fraud_analysis/provider_level_features.csv")

feature_cols = [c for c in df.columns if c not in ("Provider", "PotentialFraud", "Label")]
X = df[feature_cols].fillna(0)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"Train set: {X_train.shape[0]} providers ({y_train.mean():.4f} fraud rate)")
print(f"Test set:  {X_test.shape[0]} providers ({y_test.mean():.4f} fraud rate)")

# ---------------------------------------------------------------------------
# 2. Baseline model: Logistic Regression on standardized features
#    (This is the "rule-based / simple baseline" referenced in the petition.)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
baseline.fit(X_train_scaled, y_train)
baseline_proba = baseline.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 3. Primary model: Gradient Boosted Trees
#    (scikit-learn's GradientBoostingClassifier — xgboost was unavailable in
#    this sandboxed environment with no internet access; this is a standard,
#    well-established substitute using the same underlying boosted-tree family.)
# ---------------------------------------------------------------------------
primary = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
)
primary.fit(X_train, y_train)
primary_proba = primary.predict_proba(X_test)[:, 1]

# Also fit a Random Forest for comparison / robustness check
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 4. Evaluation: ROC-AUC, PR-AUC (average precision)
# ---------------------------------------------------------------------------
results = {}
for name, proba in [("Baseline (Logistic Regression)", baseline_proba),
                     ("Primary (Gradient Boosted Trees)", primary_proba),
                     ("Random Forest (robustness check)", rf_proba)]:
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    results[name] = {"ROC_AUC": auc, "PR_AUC": ap}
    print(f"\n{name}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC (Average Precision): {ap:.4f}")

# ---------------------------------------------------------------------------
# 5. False-positive-rate comparison AT A MATCHED RECALL LEVEL
#    This directly answers the petition's open question: "30% compared with
#    what, at what recall?" — here we fix recall and report the actual FPR.
# ---------------------------------------------------------------------------
def fpr_at_recall(y_true, y_proba, target_recall):
    """Find the false positive rate at the threshold achieving >= target_recall."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # find smallest threshold (i.e. lowest index where tpr first reaches target)
    idx = np.searchsorted(tpr, target_recall)
    idx = min(idx, len(tpr) - 1)
    return fpr[idx], tpr[idx], thresholds[idx]

print("\n" + "=" * 70)
print("FALSE POSITIVE RATE COMPARISON AT MATCHED RECALL (target recall = 0.70)")
print("=" * 70)

target_recall = 0.70
baseline_fpr, baseline_actual_recall, baseline_thresh = fpr_at_recall(y_test, baseline_proba, target_recall)
primary_fpr, primary_actual_recall, primary_thresh = fpr_at_recall(y_test, primary_proba, target_recall)

print(f"\nBaseline (Logistic Regression):")
print(f"  At recall = {baseline_actual_recall:.3f}, false positive rate = {baseline_fpr:.4f} ({baseline_fpr*100:.2f}%)")
print(f"  Threshold: {baseline_thresh:.4f}")

print(f"\nPrimary (Gradient Boosted Trees):")
print(f"  At recall = {primary_actual_recall:.3f}, false positive rate = {primary_fpr:.4f} ({primary_fpr*100:.2f}%)")
print(f"  Threshold: {primary_thresh:.4f}")

if baseline_fpr > 0:
    relative_reduction = (baseline_fpr - primary_fpr) / baseline_fpr * 100
    pct_point_reduction = (baseline_fpr - primary_fpr) * 100
    print(f"\n>>> Relative FPR reduction: {relative_reduction:.1f}%")
    print(f">>> Percentage-point FPR reduction: {pct_point_reduction:.2f} percentage points")
else:
    relative_reduction = None
    pct_point_reduction = (baseline_fpr - primary_fpr) * 100

# ---------------------------------------------------------------------------
# 6. Confusion matrix and classification report at the matched-recall threshold
# ---------------------------------------------------------------------------
primary_preds_at_thresh = (primary_proba >= primary_thresh).astype(int)
cm = confusion_matrix(y_test, primary_preds_at_thresh)
print(f"\nPrimary model confusion matrix at recall={primary_actual_recall:.3f} threshold:")
print(f"                 Predicted No-Fraud   Predicted Fraud")
print(f"Actual No-Fraud       {cm[0][0]:>6}              {cm[0][1]:>6}")
print(f"Actual Fraud          {cm[1][0]:>6}              {cm[1][1]:>6}")

print("\nFull classification report (primary model, 0.5 threshold):")
print(classification_report(y_test, (primary_proba >= 0.5).astype(int), target_names=["No Fraud", "Fraud"]))

# ---------------------------------------------------------------------------
# 7. Save everything needed for reporting + downstream explainability script
# ---------------------------------------------------------------------------
summary = {
    "n_providers_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "fraud_rate_overall": float(y.mean()),
    "baseline_roc_auc": float(results["Baseline (Logistic Regression)"]["ROC_AUC"]),
    "baseline_pr_auc": float(results["Baseline (Logistic Regression)"]["PR_AUC"]),
    "primary_roc_auc": float(results["Primary (Gradient Boosted Trees)"]["ROC_AUC"]),
    "primary_pr_auc": float(results["Primary (Gradient Boosted Trees)"]["PR_AUC"]),
    "rf_roc_auc": float(results["Random Forest (robustness check)"]["ROC_AUC"]),
    "rf_pr_auc": float(results["Random Forest (robustness check)"]["PR_AUC"]),
    "target_recall": target_recall,
    "baseline_fpr_at_target_recall": float(baseline_fpr),
    "baseline_actual_recall": float(baseline_actual_recall),
    "primary_fpr_at_target_recall": float(primary_fpr),
    "primary_actual_recall": float(primary_actual_recall),
    "relative_fpr_reduction_pct": float(relative_reduction) if relative_reduction is not None else None,
    "pct_point_fpr_reduction": float(pct_point_reduction),
}

with open("/home/claude/fraud_analysis/results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save test set + predictions for the explainability script
X_test.assign(Label=y_test.values, PrimaryModelProba=primary_proba).to_csv(
    "/home/claude/fraud_analysis/test_predictions.csv", index=False
)

import pickle
with open("/home/claude/fraud_analysis/primary_model.pkl", "wb") as f:
    pickle.dump(primary, f)
X_train.to_csv("/home/claude/fraud_analysis/X_train.csv", index=False)

print("\nSaved: results_summary.json, test_predictions.csv, primary_model.pkl, X_train.csv")
