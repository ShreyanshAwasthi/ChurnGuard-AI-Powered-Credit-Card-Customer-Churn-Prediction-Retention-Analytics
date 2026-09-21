"""
data_quality.py
----------------
Data quality & governance layer.

Runs a set of validation rules against the raw ingested data, quarantines
rows that fail, and produces a scorecard describing exactly what was wrong
and how much of the data was affected. This mirrors the kind of governance
checks a bank's data engineering team would run before any data reaches an
analytics or ML workload.

Design choice: every check is a pure function that takes a DataFrame and
returns a boolean Series (True = row is valid for that check). That makes
each rule independently unit-testable (see tests/test_data_quality.py).
"""

from dataclasses import dataclass, field
import pandas as pd
import numpy as np


VALID_GENDERS = {"M", "F"}
REQUIRED_COLUMNS = [
    "customer_id", "age", "gender", "income_category", "months_on_book",
    "total_relationship_count", "months_inactive_12_mon", "credit_limit",
    "total_revolving_bal", "total_trans_amt", "total_trans_ct",
    "avg_utilization_ratio", "attrition_flag",
]


@dataclass
class QualityReport:
    total_rows_in: int
    null_rows_flagged: int
    duplicate_rows_flagged: int
    out_of_range_flagged: int
    total_rows_quarantined: int
    total_rows_clean: int
    quality_score_pct: float
    issues_detail: dict = field(default_factory=dict)

    def as_dict(self):
        d = self.__dict__.copy()
        d.pop("issues_detail")
        return d


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trims whitespace and standardizes casing on categorical columns."""
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    for col in ["education_level", "marital_status", "income_category", "card_category"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def check_required_columns_present(df: pd.DataFrame) -> list:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def check_nulls(df: pd.DataFrame) -> pd.Series:
    """True where a row has no nulls in any required column."""
    return df[REQUIRED_COLUMNS].notna().all(axis=1)


def check_duplicates(df: pd.DataFrame) -> pd.Series:
    """True for the first occurrence of each customer_id; False for repeats."""
    return ~df.duplicated(subset=["customer_id"], keep="first")


def check_ranges(df: pd.DataFrame) -> pd.Series:
    """True where all numeric fields fall within plausible real-world bounds."""
    valid = pd.Series(True, index=df.index)
    valid &= df["age"].between(18, 100)
    valid &= df["credit_limit"] >= 0
    valid &= df["total_revolving_bal"] >= 0
    valid &= df["avg_utilization_ratio"].between(0, 1.0001)
    valid &= df["months_inactive_12_mon"].between(0, 12)
    valid &= df["total_relationship_count"].between(1, 10)
    valid &= df["gender"].isin(VALID_GENDERS)
    return valid


def run_quality_checks(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """
    Runs the full validation suite.

    Returns:
        clean_df:  rows that passed every check (whitespace/case normalized)
        quarantined_df: rows that failed at least one check, with a reason column
        report: QualityReport scorecard summarizing the run
    """
    missing_cols = check_required_columns_present(df)
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {missing_cols}")

    total_in = len(df)
    df_norm = normalize_text_columns(df)

    nulls_ok = check_nulls(df_norm)
    dup_ok = check_duplicates(df_norm)
    range_ok = check_ranges(df_norm)

    all_ok = nulls_ok & dup_ok & range_ok

    reasons = pd.Series([""] * len(df_norm), index=df_norm.index)
    reasons[~nulls_ok] += "null_value;"
    reasons[~dup_ok] += "duplicate_customer_id;"
    reasons[~range_ok] += "out_of_range;"

    clean_df = df_norm[all_ok].copy()
    quarantined_df = df_norm[~all_ok].copy()
    quarantined_df["quarantine_reason"] = reasons[~all_ok]

    report = QualityReport(
        total_rows_in=total_in,
        null_rows_flagged=int((~nulls_ok).sum()),
        duplicate_rows_flagged=int((~dup_ok).sum()),
        out_of_range_flagged=int((~range_ok).sum()),
        total_rows_quarantined=int((~all_ok).sum()),
        total_rows_clean=int(all_ok.sum()),
        quality_score_pct=round(100 * all_ok.sum() / total_in, 2) if total_in else 0.0,
    )

    return clean_df, quarantined_df, report


if __name__ == "__main__":
    from pathlib import Path
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_customers.csv"
    df = pd.read_csv(raw_path)
    clean_df, quarantined_df, report = run_quality_checks(df)
    print(report.as_dict())
    print(f"\nSample quarantine reasons:\n{quarantined_df['quarantine_reason'].value_counts()}")
