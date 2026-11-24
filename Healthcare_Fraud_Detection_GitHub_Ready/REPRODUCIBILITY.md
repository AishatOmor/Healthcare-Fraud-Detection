# Reproducibility notes

The first three notebooks completed in a fresh output run using the installed Python 3.13 environment. The fourth notebook was started but the overall run reached its execution time limit; a full end-to-end rerun has not been verified. Both service-date scripts were rerun separately and reproduced the published aggregate metrics. Dependency versions are pinned to the installed environment, not independently tested in a newly provisioned environment.

Run `python run_research.py` from the repository root after installing requirements and placing the four training CSVs in `data/`. The runner executes notebooks without overwriting their saved historical outputs. Local row-level outputs should not be published.
