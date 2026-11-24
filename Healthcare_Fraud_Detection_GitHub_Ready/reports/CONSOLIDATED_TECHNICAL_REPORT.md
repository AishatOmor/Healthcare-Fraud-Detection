# Healthcare Provider Fraud-Risk Detection: Consolidated Technical Research Report

**Research status:** Retrospective, exploratory provider-level classification benchmark. **Date:** November 18, 2026.  
**Dataset:** Kaggle Healthcare Provider Fraud Detection Analysis; beneficiary, inpatient, outpatient and provider label training tables.  
**Intended audience:** Independent ML researchers, healthcare payment-integrity specialists,.

## 1. Executive summary
This project examines whether historical provider-claim aggregates distinguish providers assigned the dataset's binary `PotentialFraud` label. It does not establish that individual claims are fraudulent or that an algorithm can prospectively block improper payments. The completed experiments include baseline reproduction, a provider-disjoint 60/20/20 train/validation/test protocol, threshold selection on validation data, paired bootstrap comparisons, and a review-capacity simulation. The final held-out test set contains **1,082 providers**, of whom **101** have positive historical labels.

## 2. Research question and scope
Can models rank historically labeled providers for human review using aggregated retrospective records? This is a **provider prioritization** problem. A prepayment claims-review system is a distinct future application requiring claim-level labels, as-of features, independent validation, and operational governance.

## 3. Data and provenance
Training inputs comprise `Train-1542865627584.csv` (provider labels), `Train_Beneficiarydata-1542865627584.csv`, `Train_Inpatientdata-1542865627584.csv`, and `Train_Outpatientdata-1542865627584.csv`. The engineered provider benchmark contains **5,410 providers**. Labels are historical `PotentialFraud` indicators; label adjudication provenance and investigation-confirmation standards have not been independently verified. Avoid calling positive labels confirmed fraud.

## 4. Methods
- Provider-disjoint stratified train/validation/test: **3,246 / 1,082 / 1,082** providers; random split, not chronological.
- Balanced logistic regression: median imputation and standardization; gradient boosting: median imputation.
- Model thresholds selected **only on validation** for target validation recall ≥70%; test labels used for final evaluation.
- Paired bootstrap over held-out providers: 600 resamples; exploratory percentile intervals.
- Retrospective top-K review simulation and claim-volume subgroup descriptions.

## 5. Held-out evaluation
| Measure | Logistic regression | Gradient boosting |
|---|---:|---:|
| ROC-AUC | 0.9593 | 0.9570 |
| PR-AUC | 0.7489 | 0.6922 |
| Recall | 81.19% | 77.23% |
| Precision | 50.00% | 55.32% |
| False-positive rate | 8.36% | 6.42% |
| True positives | 82 | 78 |
| False positives | 82 | 63 |
| False negatives | 19 | 23 |

Gradient boosting produces **19 fewer false positives** but **4 fewer true positives** at the selected thresholds. The test recalls differ, so these results do **not** support a matched-test-recall false-positive reduction claim. Logistic regression has higher PR-AUC on this split.

## 6. Uncertainty and interpretability
The paired gradient-boosting-minus-logistic-regression ROC-AUC difference is **-0.0022**, with 95% bootstrap interval **[-0.0103, 0.0050]**. PR-AUC difference is **-0.0567**, with interval **[-0.1227, 0.0119]**. Both intervals include zero. Phase 3 permutation feature-importance outputs are available for inspection; importance measures model reliance, not causation or proof of wrongdoing.

## 7. Temporal feasibility assessment — new in Phase 5
The source claims contain date fields:

| Source     | DateField    |   Rows | Earliest   | Latest     |   MissingDates |   DistinctProviders |
|:-----------|:-------------|-------:|:-----------|:-----------|---------------:|--------------------:|
| Inpatient  | ClaimStartDt |  40474 | 2008-11-27 | 2009-12-31 |              0 |                2092 |
| Inpatient  | ClaimEndDt   |  40474 | 2009-01-01 | 2009-12-31 |              0 |                2092 |
| Outpatient | ClaimStartDt | 517737 | 2008-12-12 | 2009-12-31 |              0 |                5012 |
| Outpatient | ClaimEndDt   | 517737 | 2008-12-28 | 2009-12-31 |              0 |                5012 |

**Feasible next experiment:** build timestamped provider history using claims strictly before an index claim date, with payment/outcome fields restricted to information demonstrably available at that date. Use a later time window for evaluation, and exclude future-claim aggregates from every prediction. **Key constraint:** provider-level labels are not timestamped claim outcomes. Even a chronological feature split cannot validate claim-level prepayment detection or prevent retrospective label leakage without trustworthy label dates and definitions. A meaningful forward-in-time study would require outcome timing and preferably multiple periods of longitudinal data.

## 8. Feature-readiness and data-leakage gate
All **21** audited engineered aggregates are currently **not verified as claim-time-safe**. Potentially problematic features include historical aggregate counts and reimbursement summaries that may include records after a hypothetical index claim. This does not invalidate retrospective benchmarking, but prevents prospective deployment claims. See `phase4_feature_availability_audit.csv` for per-feature remediation.

## 9. Review-capacity simulation
At a fixed review capacity of 100 providers, the prior Phase 4 analysis found **66** historical positive labels among the logistic regression top 100 and **63** among gradient boosting's top 100. These are ranking metrics, not confirmed recoveries, investigated cases, or dollar savings. Review thresholds, human capacity, and the harm of false allegations must be considered.

## 10. Human oversight and operational design (proposal, not implemented)
A possible workflow would be: receive claim → retrieve strictly prior provider history → compute risk score → route a limited subset to trained reviewers → record disposition and appeal outcomes → monitor drift and disparate error rates. Any such system would require payer authorization, access controls, audit logs, privacy impact assessment, outcome validation, and monitoring. **No institutional deployment, partnership, or prospective pilot is documented here.**

## 11. Reproducibility and review package
- Notebooks: `01_original_model_reproduction_executed.ipynb` through `04_matched_recall_feature_audit_executed.ipynb`.
- Phase 4 datasets: model metrics, paired bootstrap intervals, top-K capacity, review queue, feature audit, and claim-volume slices.
- Service-date experiments: scripts in `src/` and aggregate CSV outputs.
- Independent reviewers should rerun the notebooks, check dependency versions, verify the source-data license/provenance, and examine the split and features before endorsing conclusions.

## November 23, 2026 — service-date proxy extension
Executed on 558,211 training claims. Four features count earlier claims, earlier inpatient claims, earlier active service dates and inpatient share. Same-day claims are excluded from each claim's history. One hundred sampled provider-date histories matched direct counts. Payment amounts and annual beneficiary totals are excluded. Provider-disjoint 60/20/20 splits use snapshots before July 1, September 1 and November 1, 2009 respectively. Validation ROC-AUC 0.917608 and AP 0.597170; held-out ROC-AUC 0.916846 and AP 0.552958. Top 100 captures 51 of 101 positive-labeled providers. These are retrospective provider labels with unknown ascertainment dates, not prospective outcomes; date-ordered snapshots do not establish real-time validity. Service start dates are not submission or availability timestamps. The demonstration ranks claims from held-out providers using inherited provider scores, not claim-specific fraud probabilities. It exports up to ten claims per day (610 rows), without beneficiary identifiers. Scores are not calibrated for operational use. No actual investigator review or prevented payments occurred. The Excel dashboard includes both experiments. Source script and CSV outputs provide reproducible evidence of this run, not independent reproduction.

## Claim-specific review and chronological sensitivity extension (November 23, 2026)
The executed script `src/06_claim_review.py` uses the same provider-disjoint split as the service-date experiment. A fixed model fitted on July 1 historical snapshots is evaluated on the same 1,082 held-out providers (101 positive labels) at September, October, November and December cutoffs. ROC-AUC is 0.9163, 0.9197, 0.9168 and 0.9223; average precision is 0.5249, 0.5211, 0.5530 and 0.5688. These repeated observations are correlated and the labels have no verified adjudication dates. This is chronological sensitivity, not prospective outcome validation.

The claim review demonstration uses 18,672 held-out claims in November–December (61 days). A fixed illustrative review index is 0.8 times the historical provider model score plus 0.1 each for diagnosis and procedure counts exceeding the 95th percentile of the corresponding inpatient/outpatient training reference before July. These weights are illustrative, not optimized or validated; coding complexity is not evidence of fraud. The queue contains 610 claims (10 per day). Of 10,770 provider-day groups, 491 have distinct claim scores. No claim-level precision, recall or savings are claimed. The index is not a fraud probability.

For each claim the script computes the largest absolute provider-score change when one historical feature is replaced with its training median. This local sensitivity is not SHAP, causal attribution or an additive decomposition. The queue shows coding flags, historical features, sensitivity, daily rank and a pending human review status. Reviewers should verify source documentation and applicable payment rules, record disposition and rationale, escalate only when warranted, and never deny solely on this score. The queue is a simulation, not evidence of actual reviewer use.

The feature eligibility CSV distinguishes conditional service-date proxies from excluded payment-derived fields. No feature is certified as available at adjudication without actual submission/receipt timestamps and data lineage. Production validation requires timestamped claim versions, adjudicated outcomes, access authorization, calibration and reviewer evaluation. The Excel dashboard summarizes both experiments; CSV outputs contain detailed metrics. 
