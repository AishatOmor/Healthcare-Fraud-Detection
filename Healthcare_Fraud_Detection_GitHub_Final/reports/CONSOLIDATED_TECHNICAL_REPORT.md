# Healthcare Provider Fraud-Risk Detection: Consolidated Technical Research Report

**Research status:** Retrospective, exploratory provider-level classification benchmark. **Date:** November 18, 2026.  
**Dataset:** Kaggle Healthcare Provider Fraud Detection Analysis; beneficiary, inpatient, outpatient and provider label training tables.  
**Intended audience:** Independent ML researchers, healthcare payment-integrity specialists, and legal counsel reviewing technical evidence.

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
- Phase 5: `research_dashboard.html` (interactive **standalone HTML**, not a Power BI `.pbix` file), `temporal_feasibility.csv`, and Power BI build instructions.
- Independent reviewers should rerun the notebooks, check dependency versions, verify the source-data license/provenance, and examine the split and features before endorsing conclusions.
