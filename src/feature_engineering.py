"""
feature_engineering.py
-----------------------
Derives credit-card-specific behavioral features from the cleaned customer
table. These are the signals that actually explain churn -- raw columns
like "credit_limit" alone don't; how a customer's *behavior* is trending
does.
"""

import pandas as pd
import numpy as np


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Already-provided utilization ratio, kept for clarity/consistency
    df["revolving_util_ratio"] = (
        df["total_revolving_bal"] / df["credit_limit"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)

    # Average value per transaction -- low avg value + high count can mean
    # a very different customer than high avg value + low count.
    df["avg_trans_value"] = (
        df["total_trans_amt"] / df["total_trans_ct"].replace(0, np.nan)
    ).fillna(0)

    # Spend normalized by tenure -- comparable across customers regardless
    # of how long they've held the card.
    df["trans_amt_per_month"] = (
        df["total_trans_amt"] / df["months_on_book"].replace(0, np.nan)
    ).fillna(0)

    # A simple composite engagement score: more products, more transactions,
    # more recent activity, fewer inactive months = more engaged.
    # Each component is min-max scaled to [0, 1] before combining.
    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    engagement = (
        0.35 * minmax(df["total_relationship_count"])
        + 0.35 * minmax(df["total_trans_ct"])
        + 0.30 * (1 - minmax(df["months_inactive_12_mon"]))
    )
    df["engagement_score"] = engagement.round(3)

    # Simple binary flag used for quick filtering/segmentation rules
    df["inactivity_risk_flag"] = (df["months_inactive_12_mon"] >= 4).astype(int)

    return df


FEATURE_COLUMNS_FOR_MODEL = [
    "age", "dependent_count", "months_on_book", "total_relationship_count",
    "months_inactive_12_mon", "contacts_count_12_mon", "credit_limit",
    "total_revolving_bal", "avg_open_to_buy", "total_trans_amt", "total_trans_ct",
    "total_amt_chng_q4_q1", "total_ct_chng_q4_q1", "avg_utilization_ratio",
    "revolving_util_ratio", "avg_trans_value", "trans_amt_per_month",
    "engagement_score", "inactivity_risk_flag",
]

CATEGORICAL_COLUMNS = ["gender", "education_level", "marital_status", "income_category", "card_category"]


if __name__ == "__main__":
    from pathlib import Path
    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_customers.csv"
    from data_quality import run_quality_checks
    df = pd.read_csv(raw_path)
    clean_df, _, _ = run_quality_checks(df)
    feat_df = add_engineered_features(clean_df)
    print(feat_df[["customer_id", "revolving_util_ratio", "avg_trans_value",
                    "trans_amt_per_month", "engagement_score", "inactivity_risk_flag"]].head())
