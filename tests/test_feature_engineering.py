"""
tests/test_feature_engineering.py
Unit tests for derived feature calculations.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from feature_engineering import add_engineered_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "customer_id": [1, 2, 3],
        "credit_limit": [10000.0, 5000.0, 0.0],
        "total_revolving_bal": [2000.0, 500.0, 0.0],
        "total_trans_amt": [4000.0, 1000.0, 0.0],
        "total_trans_ct": [40, 20, 0],
        "months_on_book": [24, 12, 0],
        "total_relationship_count": [5, 2, 1],
        "months_inactive_12_mon": [1, 4, 6],
    })


def test_revolving_util_ratio_computed_correctly(sample_df):
    result = add_engineered_features(sample_df)
    assert result.loc[0, "revolving_util_ratio"] == pytest.approx(0.2)
    assert result.loc[1, "revolving_util_ratio"] == pytest.approx(0.1)


def test_zero_credit_limit_does_not_divide_by_zero(sample_df):
    result = add_engineered_features(sample_df)
    # customer 3 has credit_limit == 0 -- ratio should be 0, not NaN or inf
    assert result.loc[2, "revolving_util_ratio"] == 0


def test_avg_trans_value_computed_correctly(sample_df):
    result = add_engineered_features(sample_df)
    assert result.loc[0, "avg_trans_value"] == pytest.approx(100.0)


def test_zero_transaction_count_does_not_divide_by_zero(sample_df):
    result = add_engineered_features(sample_df)
    assert result.loc[2, "avg_trans_value"] == 0


def test_zero_months_on_book_does_not_divide_by_zero(sample_df):
    result = add_engineered_features(sample_df)
    assert result.loc[2, "trans_amt_per_month"] == 0


def test_engagement_score_is_bounded_between_0_and_1(sample_df):
    result = add_engineered_features(sample_df)
    assert result["engagement_score"].between(0, 1).all()


def test_high_relationship_low_inactivity_scores_higher_engagement(sample_df):
    result = add_engineered_features(sample_df)
    # customer 1: high relationship count, low inactivity -> most engaged
    # customer 3: low relationship count, high inactivity -> least engaged
    assert result.loc[0, "engagement_score"] > result.loc[2, "engagement_score"]


def test_inactivity_risk_flag_threshold(sample_df):
    result = add_engineered_features(sample_df)
    assert result.loc[0, "inactivity_risk_flag"] == 0  # 1 month inactive
    assert result.loc[2, "inactivity_risk_flag"] == 1  # 6 months inactive
