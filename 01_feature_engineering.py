"""
Provider-Level Healthcare Claims Fraud Detection
Step 1: Feature Engineering

Builds provider-level aggregate features from the CMS-style Inpatient, Outpatient,
and Beneficiary claims tables (Kaggle "Healthcare Provider Fraud Detection Analysis"
dataset). The prediction target (PotentialFraud) is defined at the PROVIDER level,
not the individual claim level, so all features must be aggregated up to one row
per provider before modeling.
"""

import pandas as pd
import numpy as np

DATA_DIR = "/mnt/user-data/uploads"

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------
labels = pd.read_csv(f"{DATA_DIR}/Train-1542865627584.csv")
beneficiary = pd.read_csv(f"{DATA_DIR}/Train_Beneficiarydata-1542865627584.csv")
inpatient = pd.read_csv(f"{DATA_DIR}/Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv(f"{DATA_DIR}/Train_Outpatientdata-1542865627584.csv")

print(f"Providers (labels):   {labels.shape}")
print(f"Beneficiary records:  {beneficiary.shape}")
print(f"Inpatient claims:     {inpatient.shape}")
print(f"Outpatient claims:    {outpatient.shape}")

# ---------------------------------------------------------------------------
# 2. Combine inpatient + outpatient into one claims table
#    (mark claim type so we can compute the inpatient/outpatient mix per provider)
# ---------------------------------------------------------------------------
inpatient = inpatient.copy()
outpatient = outpatient.copy()
inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"

diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
proc_cols_ip = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

common_cols = [
    "BeneID", "ClaimID", "ClaimStartDt", "ClaimEndDt", "Provider",
    "InscClaimAmtReimbursed", "AttendingPhysician", "OperatingPhysician",
    "OtherPhysician", "DeductibleAmtPaid", "ClmAdmitDiagnosisCode", "ClaimType"
] + diag_cols

claims = pd.concat(
    [
        inpatient[common_cols + proc_cols_ip + ["AdmissionDt", "DischargeDt"]],
        outpatient[common_cols].assign(
            **{c: np.nan for c in proc_cols_ip}, AdmissionDt=np.nan, DischargeDt=np.nan
        ),
    ],
    ignore_index=True,
)

claims["ClaimStartDt"] = pd.to_datetime(claims["ClaimStartDt"])
claims["ClaimEndDt"] = pd.to_datetime(claims["ClaimEndDt"])
claims["AdmissionDt"] = pd.to_datetime(claims["AdmissionDt"])
claims["DischargeDt"] = pd.to_datetime(claims["DischargeDt"])
claims["ClaimDurationDays"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days
claims["LengthOfStayDays"] = (claims["DischargeDt"] - claims["AdmissionDt"]).dt.days

# Diagnosis / procedure code counts per claim (how many distinct codes billed)
claims["NumDiagCodes"] = claims[diag_cols].notna().sum(axis=1)
claims["NumProcCodes"] = claims[proc_cols_ip].notna().sum(axis=1)

# Physician-role overlap flag: same physician billed in more than one role on
# the same claim. This is a documented red flag in CMS program-integrity
# guidance (self-referral / role-stacking patterns).
claims["PhysicianRoleOverlap"] = (
    ((claims["AttendingPhysician"] == claims["OperatingPhysician"]) & claims["AttendingPhysician"].notna())
    | ((claims["AttendingPhysician"] == claims["OtherPhysician"]) & claims["AttendingPhysician"].notna())
    | ((claims["OperatingPhysician"] == claims["OtherPhysician"]) & claims["OperatingPhysician"].notna())
).astype(int)

print(f"\nCombined claims table: {claims.shape}")

# ---------------------------------------------------------------------------
# 3. Beneficiary-level features (age, chronic conditions) joined onto claims
# ---------------------------------------------------------------------------
beneficiary = beneficiary.copy()
beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"])
beneficiary["DOD"] = pd.to_datetime(beneficiary["DOD"])
# Approximate age as of the dataset's reference year (2009, per the Kaggle
# dataset documentation) since DOD is only populated for deceased beneficiaries.
beneficiary["Age"] = 2009 - beneficiary["DOB"].dt.year

chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]
# Chronic condition flags are coded 1 = yes, 2 = no in this dataset; convert to 0/1
for c in chronic_cols:
    beneficiary[c] = (beneficiary[c] == 1).astype(int)
beneficiary["ChronicConditionCount"] = beneficiary[chronic_cols].sum(axis=1)
beneficiary["IsDeceased"] = beneficiary["DOD"].notna().astype(int)

bene_features = beneficiary[
    ["BeneID", "Age", "ChronicConditionCount", "IsDeceased", "RenalDiseaseIndicator", "State"]
].copy()
bene_features["RenalDiseaseIndicator"] = (bene_features["RenalDiseaseIndicator"] == "Y").astype(int)

claims = claims.merge(bene_features, on="BeneID", how="left")

# ---------------------------------------------------------------------------
# 4. Aggregate claims up to one row per PROVIDER (the label's grain)
# ---------------------------------------------------------------------------
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
    PctInpatient=("ClaimType", lambda x: (x == "Inpatient").mean()),
    AvgBeneficiaryAge=("Age", "mean"),
    AvgChronicConditions=("ChronicConditionCount", "mean"),
    PctDeceasedBeneficiaries=("IsDeceased", "mean"),
    PctRenalDisease=("RenalDiseaseIndicator", "mean"),
    NumUniqueStates=("State", "nunique"),
).reset_index()

# Claims-per-beneficiary concentration: high values suggest a small pool of
# beneficiaries being billed repeatedly, a documented pattern in provider-level
# fraud rings.
agg["ClaimsPerBeneficiary"] = agg["NumClaims"] / agg["NumUniqueBeneficiaries"]

# Fill claim-type-specific NaNs (providers with zero inpatient claims have no
# length-of-stay to average) with 0 rather than dropping the provider.
agg["AvgLengthOfStayDays"] = agg["AvgLengthOfStayDays"].fillna(0)
agg["StdReimbursedPerClaim"] = agg["StdReimbursedPerClaim"].fillna(0)

# ---------------------------------------------------------------------------
# 5. Attach the fraud label
# ---------------------------------------------------------------------------
dataset = agg.merge(labels, on="Provider", how="inner")
dataset["Label"] = (dataset["PotentialFraud"] == "Yes").astype(int)

print(f"\nFinal provider-level dataset: {dataset.shape}")
print(f"Fraud rate: {dataset['Label'].mean():.4f}")

dataset.to_csv("/home/claude/fraud_analysis/provider_level_features.csv", index=False)
print("\nSaved to provider_level_features.csv")
print("\nFeature columns:")
print([c for c in dataset.columns if c not in ("Provider", "PotentialFraud", "Label")])
