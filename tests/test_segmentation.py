"""
tests/test_segmentation.py
Unit tests for K-Means customer segmentation.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate_customers
from feature_engineering import add_engineered_features
from segmentation import segment_customers, N_CLUSTERS, ARCHETYPES


def _prepared_df(n=300):
    df = generate_customers(n=n, seed=7)
    return add_engineered_features(df)


def test_segment_column_is_added():
    df = _prepared_df()
    result = segment_customers(df)
    assert "value_risk_segment" in result.columns


def test_all_rows_get_a_segment_assigned():
    df = _prepared_df()
    result = segment_customers(df)
    assert result["value_risk_segment"].notna().all()


def test_exactly_n_clusters_produce_n_distinct_labels():
    df = _prepared_df(n=500)  # enough rows for 4 well-separated clusters
    result = segment_customers(df)
    assert result["value_risk_segment"].nunique() == N_CLUSTERS


def test_segment_labels_are_all_valid_archetypes():
    df = _prepared_df()
    result = segment_customers(df)
    assert set(result["value_risk_segment"].unique()).issubset(set(ARCHETYPES.keys()))


def test_row_count_is_preserved():
    df = _prepared_df()
    result = segment_customers(df)
    assert len(result) == len(df)
