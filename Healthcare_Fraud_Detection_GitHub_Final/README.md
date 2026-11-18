# Healthcare Provider Fraud-Risk Detection — Research Portfolio

**Status:** Retrospective provider-level machine-learning benchmark; **not** a deployed claims fraud detector or validated prepayment intervention.

## Project overview
This research analyzes historical `PotentialFraud` labels for 5,410 providers using provider-level aggregates. It compares logistic regression and gradient boosting using a provider-disjoint train/validation/test design. The dataset does not establish confirmed fraud or actual savings.

## Explore the results immediately
- **Excel dashboard:** [`dashboard/Healthcare_Fraud_Detection_Excel_Dashboard.xlsx`](dashboard/Healthcare_Fraud_Detection_Excel_Dashboard.xlsx) — download and open in Excel. The workbook contains a sample provider review queue using source pseudonymous IDs; do not interpret these as verified wrongdoing.
- **Browser dashboard:** [`dashboard/research_dashboard.html`](dashboard/research_dashboard.html) — download and open locally.
- **Research report:** [`reports/CONSOLIDATED_TECHNICAL_REPORT.md`](reports/CONSOLIDATED_TECHNICAL_REPORT.md).
- **Executed notebooks:** [`notebooks/`](notebooks/) — in numerical order.

## Held-out benchmark (1,082 providers)
| Metric | Logistic regression | Gradient boosting |
|---|---:|---:|
| ROC-AUC | 0.9593 | 0.9570 |
| PR-AUC | 0.7489 | 0.6922 |
| Recall | 81.19% | 77.23% |
| False-positive rate | 8.36% | 6.42% |
| Precision | 50.00% | 55.32% |

Thresholds were chosen on validation data targeting at least 70% recall, then evaluated on held-out test providers. **Different test recalls mean the FPR comparison is not at matched test recall.** Bootstrap difference intervals include zero; do not assert gradient boosting superiority.

## Reproduction
1. Install Python 3.10+ and `pip install -r requirements.txt` (dependency ranges are indicative, not a fully locked environment).
2. Obtain the four `Train*.csv` files from the [original Kaggle source](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis). Confirm source license and permitted uses.
3. Create `data/` in the repository root and place the four CSVs there. **Raw datasets are not included.**
4. Run notebooks in order. Some notebook cells were originally authored in the analysis working directory and may require adjusting relative paths. The notebooks are provided as executed research artifacts, **not guaranteed clean-environment reruns**.
5. Review `reports/` and `outputs/` for the evaluation assumptions and aggregate data.

## Key caveats
- Historical provider labels are not verified adjudicated claim-level fraud outcomes.
- 21 engineered aggregate features are **not verified as available at prepayment decision time**.
- No temporal or external prospective validation, payer deployment, partnership, or financial impact is established.
- The original benchmark selected thresholds on test labels; use the corrected evaluation for conclusions.
- This package includes a sample of pseudonymous provider identifiers in the Excel dashboard. Review dataset provenance, redistribution rights, and applicable policies before publishing it publicly.
- Project refinements made after an immigration filing must be distinguished from filing-date evidence.

## Repository structure
`notebooks/` — executed analyses; `dashboard/` — Excel and HTML dashboards; `reports/` — research documentation; `outputs/` — aggregate metrics; `legacy_original/` — preserved original scripts and plots.

## Attribution
Original Kaggle dataset: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis. Original repository scripts preserved in `legacy_original/`.
