"""
generate_data.py
-----------------
Generates a synthetic credit card customer dataset for the ChurnGuard project.

Why synthetic data?
Real bank customer data is confidential and never publicly available. This
generator produces a dataset with the same structure, feature relationships,
and data-quality problems you'd expect in a real banking data warehouse feed,
so the full pipeline (data quality -> ETL -> feature engineering ->
segmentation -> modeling -> dashboard) can be built, tested, and demonstrated
end-to-end without using anyone else's data or dataset license.

The churn relationship injected below is based on well-documented, publicly
known industry churn drivers for credit card products:
  - long inactivity (months with no activity)
  - declining transaction count/amount quarter-over-quarter
  - fewer product relationships with the bank
  - low engagement (few contacts, low transaction count)
  - high revolving balance relative to credit limit (credit-stressed customers)

A fixed random seed makes this fully reproducible.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_CUSTOMERS = 10000

RAW_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_customers.csv"


def generate_customers(n=N_CUSTOMERS, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    customer_id = 700000000 + np.arange(n)  # mimics a bank-style 9-digit customer number

    age = rng.normal(46, 8, n).clip(21, 73).astype(int)
    gender = rng.choice(["M", "F"], size=n, p=[0.53, 0.47])
    dependent_count = rng.integers(0, 6, n)

    education = rng.choice(
        ["High School", "Graduate", "Post-Graduate", "Uneducated", "Doctorate", "Unknown"],
        size=n, p=[0.22, 0.32, 0.15, 0.10, 0.06, 0.15]
    )
    marital_status = rng.choice(["Married", "Single", "Divorced", "Unknown"], size=n, p=[0.46, 0.39, 0.09, 0.06])
    income_category = rng.choice(
        ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"],
        size=n, p=[0.35, 0.18, 0.14, 0.15, 0.10, 0.08]
    )
    card_category = rng.choice(["Blue", "Silver", "Gold", "Platinum"], size=n, p=[0.93, 0.055, 0.011, 0.004])

    months_on_book = rng.integers(12, 57, n)
    total_relationship_count = rng.integers(1, 7, n)  # number of products held
    months_inactive_12_mon = rng.integers(0, 7, n)
    contacts_count_12_mon = rng.integers(0, 7, n)

    credit_limit = np.round(rng.gamma(2.0, 4200, n) + 1500, 2).clip(1500, 34500)
    avg_utilization_ratio = rng.beta(1.5, 4, n)
    total_revolving_bal = np.round(credit_limit * avg_utilization_ratio, 2)
    avg_open_to_buy = np.round((credit_limit - total_revolving_bal).clip(min=0), 2)

    total_trans_ct = rng.integers(10, 140, n)
    total_trans_amt = np.round(total_trans_ct * rng.normal(55, 20, n).clip(5, None), 2)

    total_amt_chng_q4_q1 = np.round(rng.normal(0.75, 0.25, n).clip(0, 3.5), 3)
    total_ct_chng_q4_q1 = np.round(rng.normal(0.7, 0.25, n).clip(0, 3.5), 3)

    # ---- Churn probability model (the "ground truth" business logic) ----
    # Standardize the key drivers, then combine with weights reflecting their
    # real-world relative importance to attrition risk.
    z_inactive = (months_inactive_12_mon - months_inactive_12_mon.mean()) / months_inactive_12_mon.std()
    z_ct_decline = -(total_ct_chng_q4_q1 - total_ct_chng_q4_q1.mean()) / total_ct_chng_q4_q1.std()
    z_amt_decline = -(total_amt_chng_q4_q1 - total_amt_chng_q4_q1.mean()) / total_amt_chng_q4_q1.std()
    z_low_relationship = -(total_relationship_count - total_relationship_count.mean()) / total_relationship_count.std()
    z_low_trans_ct = -(total_trans_ct - total_trans_ct.mean()) / total_trans_ct.std()
    z_high_util = (avg_utilization_ratio - avg_utilization_ratio.mean()) / avg_utilization_ratio.std()
    z_contacts = (contacts_count_12_mon - contacts_count_12_mon.mean()) / contacts_count_12_mon.std()

    logit = (
        -1.8
        + 0.85 * z_inactive
        + 1.05 * z_ct_decline
        + 0.55 * z_amt_decline
        + 0.65 * z_low_relationship
        + 0.60 * z_low_trans_ct
        + 0.30 * z_high_util
        + 0.20 * z_contacts
        + rng.normal(0, 0.35, n)  # irreducible noise so the model isn't trivially perfect
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    attrition_flag = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "dependent_count": dependent_count,
        "education_level": education,
        "marital_status": marital_status,
        "income_category": income_category,
        "card_category": card_category,
        "months_on_book": months_on_book,
        "total_relationship_count": total_relationship_count,
        "months_inactive_12_mon": months_inactive_12_mon,
        "contacts_count_12_mon": contacts_count_12_mon,
        "credit_limit": credit_limit,
        "total_revolving_bal": total_revolving_bal,
        "avg_open_to_buy": avg_open_to_buy,
        "total_trans_amt": total_trans_amt,
        "total_trans_ct": total_trans_ct,
        "total_amt_chng_q4_q1": total_amt_chng_q4_q1,
        "total_ct_chng_q4_q1": total_ct_chng_q4_q1,
        "avg_utilization_ratio": np.round(avg_utilization_ratio, 4),
        "attrition_flag": attrition_flag,
    })

    return df


def inject_data_quality_issues(df: pd.DataFrame, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Deliberately dirties a small % of rows so the data-quality layer has
    real issues to detect, log, and quarantine -- mirroring a real raw
    banking data feed (nulls, duplicates, out-of-range values, bad types).
    """
    rng = np.random.default_rng(seed + 1)
    df = df.copy()
    n = len(df)

    # 1. Nulls in a few columns (~1.5% of rows each)
    for col in ["income_category", "months_on_book", "total_trans_amt"]:
        idx = rng.choice(n, size=int(n * 0.015), replace=False)
        df.loc[idx, col] = np.nan

    # 2. Duplicate customer records (~0.5%)
    dup_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    # 3. Out-of-range / impossible values (~0.3%)
    bad_idx = rng.choice(len(df), size=int(n * 0.003), replace=False)
    df.loc[bad_idx[: len(bad_idx) // 2], "credit_limit"] = -100.0
    df.loc[bad_idx[len(bad_idx) // 2:], "age"] = 999

    # 4. Inconsistent categorical casing / whitespace (~1%)
    text_idx = rng.choice(len(df), size=int(n * 0.01), replace=False)
    df.loc[text_idx, "gender"] = df.loc[text_idx, "gender"].str.lower() + "  "

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle


def main():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customers()
    df_dirty = inject_data_quality_issues(df)
    df_dirty.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated {len(df_dirty):,} raw records -> {RAW_DATA_PATH}")
    print(f"True churn rate (before dirtying): {df['attrition_flag'].mean():.2%}")


if __name__ == "__main__":
    main()
