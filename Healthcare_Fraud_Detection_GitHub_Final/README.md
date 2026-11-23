# Healthcare Provider Fraud-Risk Detection 

**Project Status:** Completed provider-level machine-learning analysis and retrospective evaluation.

## Project overview
This project develops and evaluates machine-learning models to identify potentially suspicious healthcare provider billing patterns using historical data. I engineered 21 provider-level features and compared logistic regression with gradient boosting using separate training, validation, and testing datasets.

The analysis evaluates how effectively these models distinguish providers with historical PotentialFraud labels. It provides a technical foundation for my proposed future work on claim-level risk scoring and prepayment review

## Project Deliverables
- **Excel dashboard:** Interactive visualization of model results and a sample provider review queue.
- **Browser dashboard:** HTML-based research dashboard.
- **Research report:** Detailed methodology, evaluation, findings, and limitations.
- **Executed notebooks:** Executed analysis and model development workflows.

## Model Evaluation (1,082 providers)
| Metric | Logistic regression | Gradient boosting |
|---|---:|---:|
| ROC-AUC | 0.9593 | 0.9570 |
| PR-AUC | 0.7489 | 0.6922 |
| Recall | 81.19% | 77.23% |
| False-positive rate | 8.36% | 6.42% |
| Precision | 50.00% | 55.32% |

Decision thresholds were selected on validation data and evaluated on held-out test data. Results demonstrate different trade-offs between precision, recall, and false-positive rates. Bootstrap comparisons did not establish a statistically reliable advantage for either model.

## Reproduction
1. Install Python 3.10+ and `pip install -r requirements.txt` (dependency ranges are indicative, not a fully locked environment).
2. Obtain the four `Train*.csv` files from the [original Kaggle source](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis). Confirm source license and permitted uses.
3. Place the CSV files in a data/ folder at the repository root.
4. Execute the notebooks in numerical order, following the documented file-path requirements.
5. Review the generated evaluation results in outputs/ and the technical documentation in reports/.

## Key caveats
This is a retrospective provider-level research benchmark, not a deployed fraud detection system.

## Repository structure
`notebooks/` — executed analyses; `dashboard/` — Excel and HTML dashboards; `reports/` — research documentation; `outputs/` — aggregate metrics; `legacy_original/` — preserved original scripts and plots.

## Attribution
Original Kaggle dataset: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis. Original repository scripts preserved in `legacy_original/`.
