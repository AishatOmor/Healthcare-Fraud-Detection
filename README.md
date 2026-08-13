# Provider-Level Healthcare Claims Fraud Detection

An explainable, human-in-the-loop machine learning framework for identifying
providers with anomalous billing patterns consistent with healthcare claims
fraud, built on CMS-structured Medicare claims data with real, investigation-derived
provider fraud labels.

## Motivation

Federal healthcare programs currently rely heavily on post-payment audits to
recover fraudulent disbursements after they have already occurred. This project
demonstrates a pre-payment risk-scoring approach: flagging providers whose
aggregate billing behavior is statistically anomalous *before* additional claims
are paid, with every prediction accompanied by a human-interpretable explanation
so that an investigator, not just an algorithm, makes the final determination.

This directly supports the detection approach described in CMS and HHS's
February 2026 Request for Information on AI-driven healthcare fraud detection
and the resulting CRUSH rulemaking, both of which prioritize moving fraud
detection from post-payment recovery to pre-payment prevention.

## Dataset

[Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
(Kaggle) — a CMS-structured synthetic Medicare claims dataset with real,
investigation-derived provider-level fraud labels across:

- **5,410 providers** (506 labeled potentially fraudulent — 9.35% base rate)
- **40,474 inpatient claims**
- **517,737 outpatient claims**
- **138,556 beneficiary records** (demographics and chronic condition history)

The fraud label is defined at the **provider** level, not the individual claim
level, reflecting how CMS program-integrity investigations actually target
providers exhibiting patterns across many claims rather than adjudicating
claims one at a time.

## Methodology

### 1. Feature Engineering (`01_feature_engineering.py`)
Aggregates the three raw claims/beneficiary tables into one row per provider
(21 features), including:
- Billing volume and intensity (claim counts, total/average/max reimbursement)
- Physician-role overlap rate (same physician billed as attending, operating,
  and/or other physician on the same claim — a documented program-integrity
  red flag)
- Diagnosis/procedure code diversity per claim
- Beneficiary concentration (claims per unique beneficiary served)
- Beneficiary population characteristics (age, chronic condition burden,
  geographic spread)

### 2. Modeling (`02_modeling.py`)
- **Baseline:** Logistic Regression on standardized features (class-balanced)
- **Primary model:** Gradient Boosted Trees (scikit-learn `GradientBoostingClassifier`)
- **Robustness check:** Random Forest
- Provider-level stratified 75/25 train/test split (no data leakage — the
  split happens at the same grain as the label)

### 3. Explainability (`03_explainability.py`)
- Global feature importance via permutation importance (model-agnostic,
  measures the actual drop in ROC-AUC when each feature is shuffled)
- Per-provider explanation showing which features drove an individual
  risk score above the population average — the "reason code" format an
  investigator can act on and defend under audit

> **Note on SHAP:** this pipeline was built in a sandboxed environment without
> internet access, so the `shap` package could not be installed for this run.
> Permutation importance was used instead — a legitimate, model-agnostic,
> peer-reviewed explainability method built into scikit-learn. A ready-to-run
> SHAP cell is included at the bottom of `03_explainability.py` (commented out)
> for use in any standard environment with `pip install shap` available.

## Results

All results below are from a genuine held-out test set (1,353 providers never
seen during training), not training-set performance.

| Model | ROC-AUC | PR-AUC (Average Precision) |
|---|---|---|
| Baseline: Logistic Regression | 0.951 | 0.726 |
| **Primary: Gradient Boosted Trees** | **0.955** | **0.733** |
| Random Forest (robustness check) | 0.949 | 0.676 |

### False-positive rate reduction (matched-recall comparison)

At a fixed recall of **70%** (i.e., both models catching 70% of actual fraud
providers in the test set):

| Model | False Positive Rate |
|---|---|
| Baseline (Logistic Regression) | 6.04% |
| **Primary (Gradient Boosted Trees)** | **4.49%** |

**A 25.7% relative reduction in false positive rate** (1.55 percentage points),
at matched recall, evaluated on a held-out test set never seen during training.

### Top predictive features
1. Total reimbursement amount
2. Total deductible paid
3. Number of claims
4. Claims per unique beneficiary
5. Average length of stay (inpatient)

These align with CMS Fraud Prevention System's own documented risk indicators
(aggregate billing-intensity anomalies and beneficiary-concentration patterns).

## Reproducing these results

```bash
pip install pandas numpy scikit-learn matplotlib
python 01_feature_engineering.py
python 02_modeling.py
python 03_explainability.py
```

Requires the four Kaggle "Healthcare Provider Fraud Detection Analysis" Train
files (`Train-*.csv`, `Train_Beneficiarydata-*.csv`, `Train_Inpatientdata-*.csv`,
`Train_Outpatientdata-*.csv`), available from the Kaggle dataset page linked
above.

## Limitations

- This dataset is synthetic/de-identified and used here as a public research
  benchmark, not live institutional claims data; results should be understood
  as a proof of concept demonstrating the methodology, not a validated
  production system.
- The 25.7% false-positive reduction is specific to this dataset, this feature
  set, and this baseline; it should not be presented as a general claim about
  performance against live production fraud-detection systems.
- Provider-level aggregation means the model flags providers for investigation,
  not individual claims for denial — consistent with a human-in-the-loop
  design where a flagged provider undergoes further review rather than
  automatic claim rejection.
