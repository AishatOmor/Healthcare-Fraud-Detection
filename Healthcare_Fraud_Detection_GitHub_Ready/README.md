# Healthcare Fraud-Risk Detection

An explainable machine-learning research portfolio for provider risk ranking and capacity-constrained claim review.

## Overview
The project analyzes 558,211 historical claims and 5,410 providers from the [Healthcare Provider Fraud Detection Analysis dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis). It compares retrospective provider classifiers, constructs strictly earlier service-date histories, and demonstrates a claim-specific review index for human review.

## Results
| Experiment | ROC-AUC | Average precision |
|---|---:|---:|
| Retrospective logistic regression | 0.9593 | 0.7489 |
| Retrospective gradient boosting | 0.9570 | 0.6922 |
| Service-date provider model (November cutoff) | 0.9168 | 0.5530 |

The retrospective benchmark and service-date experiment use different feature sets and split procedures; scores are not a controlled head-to-head comparison. The claim review simulation ranks 18,672 claims and selects 610 for hypothetical review (10 per day).

## Explore
- [Technical report](reports/CONSOLIDATED_TECHNICAL_REPORT.md)
- [Excel dashboard](dashboard/Healthcare_Fraud_Detection_Excel_Dashboard.xlsx)
- [Executed notebooks](notebooks/)
- [Source code](src/) and [aggregate results](outputs/)

## Reproduce
From the repository root, use Python 3.13 and install dependencies: `python -m pip install -r requirements.txt`. Download the four training CSV files from Kaggle and place them in `data/` without renaming. Run `python run_research.py`. The runner executes the four notebooks in order, followed by both service-date scripts. Historical exploratory notebook results are retained for transparency; use corrected validation results for conclusions. Generated row-level files remain local and should not be published without verifying data permissions.

## Interpretation
The target is a retrospective provider-level PotentialFraud label, not confirmed claim fraud. Service dates are proxies, not verified submission timestamps. The claim review index is an illustrative heuristic, not a calibrated fraud probability. No automatic denial, prospective validation, actual investigator adoption or savings is established. Operational use requires authorized timestamped data, outcome validation, calibration, governance and human review.

## Repository layout
- `notebooks/`: historical benchmark and corrected evaluation
- `src/`: service-date and claim-review experiments
- `outputs/`: aggregate results
- `dashboard/`: Excel summary
- `reports/`: technical methodology and limitations
- `legacy_original/`: archived original implementation (not the supported reproduction entry point)

Raw data is not redistributed. Check the original dataset terms before use.
