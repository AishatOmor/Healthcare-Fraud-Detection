# Project 1 — Phase 4: Expert Technical Brief

**Date:** November 18, 2026  
**Purpose:** Independent technical assessment of an exploratory, historical provider-level fraud-risk classification benchmark. This is **not** a prospective prepayment claim adjudication model.

## Dataset and evaluation design
- Source: Kaggle Healthcare Provider Fraud Detection Analysis (training beneficiary, inpatient, outpatient, and provider label CSVs).
- Provider-level engineered benchmark: 5,410 providers; binary `PotentialFraud` labels.
- Stratified, provider-disjoint train/validation/test split, 60%/20%/20% (3,246/1,082/1,082), with fixed seeds.
- Two models: balanced logistic regression with median imputation and standardization; gradient boosting with median imputation.
- Decision thresholds selected on validation for at least 70% recall; held-out test labels used only for evaluation.
- The 70% recall requirement is met on validation, not guaranteed to match exactly on test.

## Held-out benchmark results
| Metric | Logistic regression | Gradient boosting |
|---|---:|---:|
| ROC-AUC | 0.9593 | 0.9570 |
| PR-AUC | 0.7489 | 0.6922 |
| Recall | 81.19% | 77.23% |
| False-positive rate | 8.36% | 6.42% |
| Precision | 50.00% | 55.32% |
| False positives | 82 | 63 |
| True positives | 82 | 78 |

At these validation-selected thresholds, gradient boosting produces 19 fewer false positives **and** 4 fewer true positives. Do not characterize this as an equal-recall FPR improvement.

## Paired bootstrap (600 resamples of held-out providers)
- Gradient boosting minus logistic regression ROC-AUC: -0.0022; 95% percentile CI [-0.0103, 0.0050].
- Gradient boosting minus logistic regression PR-AUC: -0.0567; 95% percentile CI [-0.1227, 0.0119].
- Both intervals include zero. Bootstrap intervals describe sampling uncertainty on this historical benchmark; they do not account for dataset shift or retrospective feature leakage.

## Review-capacity simulation
- At top 100 scored providers, logistic regression ranks 66 fraud-labeled providers; gradient boosting ranks 63.
- These are historical label-enrichment figures, not confirmed investigation outcomes or monetary savings.

## Critical limitations and next validation gates
1. **Retrospective feature availability:** All 21 engineered provider aggregates must be rebuilt as timestamped, strictly prior history before claiming prepayment usefulness. See `phase4_feature_availability_audit.csv`.
2. **Label semantics:** Provider-level `PotentialFraud` is not a verified fraudulent individual claim; confirm how labels were assigned and whether they reflect investigations or proxies.
3. **Time-based generalization:** Current random provider holdout is not a forward-in-time evaluation. A temporal holdout and as-of data joins are necessary.
4. **External validation:** No independent payer, CMS, or operational pilot has verified these results.
5. **Operational safety:** Investigators would need a review policy, false-positive cost assessment, human oversight, privacy/security controls, drift monitoring, and appeal/escalation procedures.
6. **Subgroup reliability:** Claim-volume slices are exploratory and may have small counts; avoid unqualified subgroup claims.

## Power BI implementation specification
Use these CSV files from `outputs/`:
- `phase4_matched_target_model_metrics.csv`: KPI cards and comparison matrix.
- `phase4_paired_model_difference_intervals.csv`: uncertainty interval comparison.
- `phase4_provider_review_simulation.csv`: sortable **historical provider-level benchmark** review queue. Provider ID, model, score, flag, claim count, historical label.
- `phase4_review_capacity.csv`: capacity vs. labeled-positive capture curve.
- `phase4_claim_volume_slices.csv`: claim-volume sensitivity matrix (show sample counts).
- `phase4_feature_availability_audit.csv`: feature readiness and leakage remediation table.

**Suggested pages:** (1) Benchmark performance; (2) Review capacity; (3) Provider scoring simulation; (4) Data availability and leakage audit. Label each page **Retrospective research benchmark — not production prepayment validation**.

## RFE evidence framing
This work documents reproducible technical progress, trade-off evaluation, and explicit limitations. It does not establish national importance, adoption, actual cost savings, or institutional endorsement by itself. Because the analyses postdate the original filing, counsel should distinguish post-filing progress from qualifying evidence of circumstances existing at filing. Independent reviewers should assess the code, data provenance, and conclusions rather than sign claims they cannot verify.
