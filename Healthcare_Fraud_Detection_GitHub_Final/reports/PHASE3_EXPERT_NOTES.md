# Project 1 — Phase 3 Technical Review Notes

**Prepared:** November 18, 2026. **Status:** Exploratory historical provider-level benchmark; not independently validated in a production payer environment.

## Changes in this phase
- Executable Jupyter notebook: `03_leakage_uncertainty_powerbi_executed.ipynb`.
- Train/validation/test provider-disjoint splits and thresholds chosen on validation only.
- Bootstrap percentile uncertainty for held-out ROC-AUC and PR-AUC (400 resamples; fixed fitted model).
- Validation-set permutation feature importance using average precision.
- Five Power BI-ready CSV exports in `outputs/`.

## Evidence boundaries
- Provider-level `PotentialFraud` labels are not individual claim adjudications.
- Historical aggregates are not proven available at prepayment decision time; prospective/temporal leakage is unresolved.
- Random provider holdout is not an independent external or prospective test.
- Different observed test recall prevents a strict matched-recall comparison of false-positive rates.
- No claim of institutional adoption, CMS endorsement, live deployment, or confirmed prevented losses.

## Recommended Power BI pages
1. **Model quality:** ROC-AUC, PR-AUC, recall, precision, FPR, confusion counts, and bootstrap intervals by model (`powerbi_phase3_model_metrics.csv`, `powerbi_phase3_bootstrap_intervals.csv`).
2. **Provider review queue:** provider identifier, score, risk band, threshold flag; label prominently as **benchmark provider-level review simulation** (`powerbi_phase3_provider_scores.csv`).
3. **Risk distribution:** provider counts by score band and benchmark label (`powerbi_phase3_risk_bands.csv`).
4. **Explainability:** feature ranking by validation-set permutation PR-AUC drop (`powerbi_phase3_feature_importance.csv`).

## Expert evaluation questions
- Are data sources and fraud labels suitable for this modeling claim?
- Are the split and threshold protocol appropriate for this benchmark?
- Which engineered features are only available retrospectively?
- What evidence would be necessary for a prospective claim-level pilot?
- Are limitations and performance results reported accurately?

This analysis was completed after the original petition filing date. Any use in an RFE response should distinguish new progress from evidence of circumstances existing at filing, with legal counsel review.
