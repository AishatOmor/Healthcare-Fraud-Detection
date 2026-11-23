
"""
Healthcare Provider Fraud-Risk Detection
Step 1: Provider-Level Feature Engineering

Creates provider-level features from the Kaggle Healthcare
Provider Fraud Detection Analysis training dataset.

The target, PotentialFraud, is defined at provider level.
Claims are therefore aggregated to one record per provider.

The resulting features describe historical billing patterns.
They are not validated claim-level prepayment indicators.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1. Project paths and input files
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_DIR = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() in {"src", "scripts"}
    else SCRIPT_DIR
)

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "labels": "Train-1542865627584.csv",
    "beneficiary": "Train_Beneficiarydata-1542865627584.csv",
    "inpatient": "Train_Inpatientdata-1542865627584.csv",
    "outpatient": "Train_Outpatientdata-1542865627584.csv",
}


def find_input(filename):
    """Find a source file in common repository directories."""

    directories = [
        PROJECT_DIR / "data",
        PROJECT_DIR / "data" / "raw",
        PROJECT_DIR,
    ]

    for directory in directories:
        path = directory / filename
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Cannot find {filename}.\n"
        "Download the original Kaggle training CSV files "
        "and place them in the repository's data/ folder."
    )


def require_columns(df, columns, table_name):
    """Check that the source table has its required columns."""

    missing = sorted(set(columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def parse_dates(df, columns):
    """Convert source date columns to datetime."""

    for column in columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    return df


def convert_numeric(df, columns):
    """Convert numeric claim fields and preserve missing values."""

    for column in columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


print("Loading source datasets...")

tables = {
    name: pd.read_csv(find_input(filename), low_memory=False)
    for name, filename in FILES.items()
}

labels = tables["labels"]
beneficiary = tables["beneficiary"]
inpatient = tables["inpatient"]
outpatient = tables["outpatient"]

for name, df in tables.items():
    print(f"{name}: {len(df):,} records")


# ------------------------------------------------------------
# 2. Validate source tables
# ------------------------------------------------------------

diag_cols = [
    f"ClmDiagnosisCode_{i}" for i in range(1, 11)
]

proc_cols = [
    f"ClmProcedureCode_{i}" for i in range(1, 7)
]

common_cols = [
    "BeneID",
    "ClaimID",
    "ClaimStartDt",
    "ClaimEndDt",
    "Provider",
    "InscClaimAmtReimbursed",
    "AttendingPhysician",
    "OperatingPhysician",
    "OtherPhysician",
    "DeductibleAmtPaid",
    "ClmAdmitDiagnosisCode",
] + diag_cols

require_columns(
    labels,
    ["Provider", "PotentialFraud"],
    "Provider labels",
)

require_columns(
    inpatient,
    common_cols + proc_cols + ["AdmissionDt", "DischargeDt"],
    "Inpatient",
)

require_columns(
    outpatient,
    common_cols,
    "Outpatient",
)

require_columns(
    beneficiary,
    [
        "BeneID",
        "DOB",
        "DOD",
        "RenalDiseaseIndicator",
        "State",
    ],
    "Beneficiary",
)

if labels["Provider"].isna().any():
    raise ValueError("Provider labels contain missing provider IDs.")

if labels["Provider"].duplicated().any():
    raise ValueError("Provider labels contain duplicate provider IDs.")

if beneficiary["BeneID"].isna().any():
    raise ValueError("Beneficiary data contain missing beneficiary IDs.")

if beneficiary["BeneID"].duplicated().any():
    raise ValueError("Beneficiary data contain duplicate beneficiary IDs.")

valid_labels = {"Yes", "No"}

if not labels["PotentialFraud"].isin(valid_labels).all():
    raise ValueError(
        "PotentialFraud must contain only Yes or No."
    )

for name, df in [
    ("Inpatient", inpatient),
    ("Outpatient", outpatient),
]:
    if df["Provider"].isna().any():
        raise ValueError(f"{name} contains missing provider IDs.")

    if df["BeneID"].isna().any():
        raise ValueError(f"{name} contains missing beneficiary IDs.")

    if df["ClaimID"].isna().any():
        raise ValueError(f"{name} contains missing claim IDs.")

if not inpatient.empty and not outpatient.empty:
    duplicate_claims = pd.concat(
        [inpatient["ClaimID"], outpatient["ClaimID"]],
        ignore_index=True,
    ).duplicated()

    if duplicate_claims.any():
        raise ValueError(
            "Duplicate ClaimID values found across claim tables."
        )


# ------------------------------------------------------------
# 3. Combine inpatient and outpatient claims
# ------------------------------------------------------------

inpatient = inpatient.copy()
outpatient = outpatient.copy()

inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"

inpatient_cols = common_cols + [
    "ClaimType",
] + proc_cols + [
    "AdmissionDt",
    "DischargeDt",
]

outpatient_cols = common_cols + ["ClaimType"]

inpatient_claims = inpatient[inpatient_cols].copy()
outpatient_claims = outpatient[outpatient_cols].copy()

# These fields are not present in the outpatient table.
for column in proc_cols + ["AdmissionDt", "DischargeDt"]:
    outpatient_claims[column] = np.nan

claims = pd.concat(
    [inpatient_claims, outpatient_claims],
    ignore_index=True,
)

claims = parse_dates(
    claims,
    [
        "ClaimStartDt",
        "ClaimEndDt",
        "AdmissionDt",
        "DischargeDt",
    ],
)

claims = convert_numeric(
    claims,
    [
        "InscClaimAmtReimbursed",
        "DeductibleAmtPaid",
    ],
)

claims["ClaimDurationDays"] = (
    claims["ClaimEndDt"] - claims["ClaimStartDt"]
).dt.days

claims["LengthOfStayDays"] = (
    claims["DischargeDt"] - claims["AdmissionDt"]
).dt.days

for column in ["ClaimDurationDays", "LengthOfStayDays"]:
    if (claims[column].dropna() < 0).any():
        raise ValueError(
            f"Negative values detected in {column}. "
            "Review source dates before proceeding."
        )

# Number of populated diagnosis and procedure fields.
claims["NumDiagCodes"] = (
    claims[diag_cols].notna().sum(axis=1)
)

claims["NumProcCodes"] = (
    claims[proc_cols].notna().sum(axis=1)
)

# Same physician identifier appears in multiple roles.
# Descriptive indicator only; not proof of improper billing.
claims["PhysicianRoleOverlap"] = (
    (
        (claims["AttendingPhysician"] == claims["OperatingPhysician"])
        & claims["AttendingPhysician"].notna()
    )
    |
    (
        (claims["AttendingPhysician"] == claims["OtherPhysician"])
        & claims["AttendingPhysician"].notna()
    )
    |
    (
        (claims["OperatingPhysician"] == claims["OtherPhysician"])
        & claims["OperatingPhysician"].notna()
    )
).astype(int)

print(f"Combined claims: {len(claims):,}")


# ------------------------------------------------------------
# 4. Prepare beneficiary information
# ------------------------------------------------------------

beneficiary = beneficiary.copy()

beneficiary = parse_dates(
    beneficiary,
    ["DOB", "DOD"],
)

# Approximate age using the historical dataset reference year.
# This is a retrospective feature, not an as-of-claim age.
REFERENCE_YEAR = 2009

beneficiary["Age"] = (
    REFERENCE_YEAR - beneficiary["DOB"].dt.year
)

beneficiary.loc[
    ~beneficiary["Age"].between(0, 120),
    "Age",
] = np.nan

chronic_cols = [
    column
    for column in beneficiary.columns
    if column.startswith("ChronicCond_")
]

if not chronic_cols:
    raise ValueError(
        "No chronic condition columns were found."
    )

# Source convention: 1 = condition present, 2 = absent.
# Unexpected or missing values remain missing.
for column in chronic_cols:
    raw = pd.to_numeric(
        beneficiary[column],
        errors="coerce",
    )

    beneficiary[column] = raw.map({
        1: 1.0,
        2: 0.0,
    })

    if beneficiary[column].isna().any():
        raise ValueError(
            f"Missing or unexpected codes in {column}."
        )

beneficiary["ChronicConditionCount"] = (
    beneficiary[chronic_cols].sum(axis=1)
)

beneficiary["IsDeceased"] = (
    beneficiary["DOD"].notna().astype(int)
)

renal_values = (
    beneficiary["RenalDiseaseIndicator"]
    .astype("string")
    .str.strip()
    .str.upper()
)

beneficiary["RenalDiseaseIndicator"] = renal_values.map({
    "Y": 1.0,
    "0": 0.0,
})

if beneficiary["RenalDiseaseIndicator"].isna().any():
    raise ValueError(
        "Unexpected or missing RenalDiseaseIndicator values."
    )

bene_features = beneficiary[
    [
        "BeneID",
        "Age",
        "ChronicConditionCount",
        "IsDeceased",
        "RenalDiseaseIndicator",
        "State",
    ]
].copy()

before_merge = len(claims)

claims = claims.merge(
    bene_features,
    on="BeneID",
    how="left",
    validate="many_to_one",
    indicator=True,
)

unmatched = int((claims["_merge"] == "left_only").sum())

if unmatched:
    raise ValueError(
        f"{unmatched:,} claims have no matching beneficiary record."
    )

claims = claims.drop(columns="_merge")

if len(claims) != before_merge:
    raise RuntimeError(
        "Beneficiary merge unexpectedly changed the claim count."
    )


# ------------------------------------------------------------
# 5. Aggregate to provider level
# ------------------------------------------------------------

claims["IsInpatient"] = (
    claims["ClaimType"] == "Inpatient"
).astype(int)

agg = claims.groupby("Provider").agg(
    NumClaims=("ClaimID", "count"),
    NumUniqueBeneficiaries=("BeneID", "nunique"),
    TotalReimbursed=("InscClaimAmtReimbursed", "sum"),
    AvgReimbursedPerClaim=("InscClaimAmtReimbursed", "mean"),
    StdReimbursedPerClaim=("InscClaimAmtReimbursed", "std"),
    MaxReimbursedClaim=("InscClaimAmtReimbursed", "max"),
    TotalDeductible=("DeductibleAmtPaid", "sum"),
    AvgClaimDurationDays=("ClaimDurationDays", "mean"),
    AvgLengthOfStayDays=("LengthOfStayDays", "mean"),
    AvgNumDiagCodes=("NumDiagCodes", "mean"),
    AvgNumProcCodes=("NumProcCodes", "mean"),
    PhysicianRoleOverlapRate=("PhysicianRoleOverlap", "mean"),
    NumUniqueAttendingPhysicians=("AttendingPhysician", "nunique"),
    NumUniqueOperatingPhysicians=("OperatingPhysician", "nunique"),
    PctInpatient=("IsInpatient", "mean"),
    AvgBeneficiaryAge=("Age", "mean"),
    AvgChronicConditions=("ChronicConditionCount", "mean"),
    PctDeceasedBeneficiaries=("IsDeceased", "mean"),
    PctRenalDisease=("RenalDiseaseIndicator", "mean"),
    NumUniqueStates=("State", "nunique"),
).reset_index()

# Claim volume relative to unique beneficiaries.
agg["ClaimsPerBeneficiary"] = (
    agg["NumClaims"]
    / agg["NumUniqueBeneficiaries"].replace(0, np.nan)
)

# Preserve original feature conventions for compatibility.
agg["AvgLengthOfStayDays"] = (
    agg["AvgLengthOfStayDays"].fillna(0)
)

agg["StdReimbursedPerClaim"] = (
    agg["StdReimbursedPerClaim"].fillna(0)
)


# ------------------------------------------------------------
# 6. Attach provider-level labels
# ------------------------------------------------------------

dataset = labels[
    ["Provider", "PotentialFraud"]
].merge(
    agg,
    on="Provider",
    how="left",
    validate="one_to_one",
    indicator=True,
)

missing_providers = int(
    (dataset["_merge"] == "left_only").sum()
)

if missing_providers:
    raise ValueError(
        f"{missing_providers} labeled providers have no claims."
    )

dataset = dataset.drop(columns="_merge")

dataset["Label"] = (
    dataset["PotentialFraud"] == "Yes"
).astype(int)

feature_cols = [
    column
    for column in dataset.columns
    if column not in [
        "Provider",
        "PotentialFraud",
        "Label",
    ]
]

if len(feature_cols) != 21:
    raise ValueError(
        f"Expected 21 features, found {len(feature_cols)}."
    )

if dataset["Provider"].duplicated().any():
    raise ValueError(
        "Final dataset contains duplicate providers."
    )

# Do not silently impute unexpected missing feature values.
missing_features = dataset[feature_cols].isna().sum()
missing_features = missing_features[
    missing_features > 0
]

if not missing_features.empty:
    raise ValueError(
        "Missing feature values remain. Review the data or "
        "apply documented training-only imputation:\n"
        + missing_features.to_string()
    )

if not np.isfinite(
    dataset[feature_cols].to_numpy(dtype=float)
).all():
    raise ValueError(
        "Non-finite feature values found."
    )


# ------------------------------------------------------------
# 7. Save final dataset
# ------------------------------------------------------------

output_path = OUTPUT_DIR / "provider_level_features.csv"

dataset.to_csv(
    output_path,
    index=False,
)

print("\nFeature engineering completed successfully.")
print(f"Providers: {len(dataset):,}")
print(f"Features: {len(feature_cols)}")
print(
    "PotentialFraud positive-label rate: "
    f"{dataset['Label'].mean():.4f}"
)
print(f"Saved dataset: {output_path}")

print("\nEngineered features:")
for index, feature in enumerate(feature_cols, start=1):
    print(f"{index:02d}. {feature}")
