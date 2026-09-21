"""
tests/test_data_quality.py
Unit tests for the data quality & governance rules. These run on small,
hand-built DataFrames -- no MySQL connection required, so `pytest` works
out of the box on any machine.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_quality import (
    run_quality_checks, check_nulls, check_duplicates, check_ranges,
    normalize_text_columns, REQUIRED_COLUMNS,
)


def _base_row(**overrides):
    row = {
        "customer_id": 1, "age": 40, "gender": "M", "dependent_count": 2,
        "education_level": "Graduate", "marital_status": "Married",
        "income_category": "$60K - $80K", "card_category": "Blue",
        "months_on_book": 36, "total_relationship_count": 3,
        "months_inactive_12_mon": 2, "contacts_count_12_mon": 2,
        "credit_limit": 8000.0, "total_revolving_bal": 1200.0,
        "avg_open_to_buy": 6800.0, "total_trans_amt": 3500.0,
        "total_trans_ct": 45, "total_amt_chng_q4_q1": 0.7,
        "total_ct_chng_q4_q1": 0.65, "avg_utilization_ratio": 0.15,
        "attrition_flag": 0,
    }
    row.update(overrides)
    return row


def make_df(rows):
    return pd.DataFrame(rows)


class TestCheckNulls:
    def test_valid_row_passes(self):
        df = make_df([_base_row()])
        assert check_nulls(df).iloc[0] == True  # noqa: E712

    def test_null_in_required_column_fails(self):
        df = make_df([_base_row(income_category=None)])
        assert check_nulls(df).iloc[0] == False  # noqa: E712


class TestCheckDuplicates:
    def test_unique_ids_all_pass(self):
        df = make_df([_base_row(customer_id=1), _base_row(customer_id=2)])
        assert check_duplicates(df).all()

    def test_duplicate_id_second_occurrence_fails(self):
        df = make_df([_base_row(customer_id=1), _base_row(customer_id=1)])
        result = check_duplicates(df)
        assert result.iloc[0] == True and result.iloc[1] == False  # noqa: E712


class TestCheckRanges:
    @pytest.mark.parametrize("overrides", [
        {"age": 150},
        {"age": 5},
        {"credit_limit": -100},
        {"avg_utilization_ratio": 1.5},
        {"months_inactive_12_mon": 20},
        {"total_relationship_count": 0},
        {"gender": "X"},
    ])
    def test_out_of_range_values_fail(self, overrides):
        df = make_df([_base_row(**overrides)])
        assert check_ranges(df).iloc[0] == False  # noqa: E712

    def test_valid_row_passes_range_check(self):
        df = make_df([_base_row()])
        assert check_ranges(df).iloc[0] == True  # noqa: E712


class TestNormalizeTextColumns:
    def test_gender_is_trimmed_and_uppercased(self):
        df = make_df([_base_row(gender="  m  ")])
        normalized = normalize_text_columns(df)
        assert normalized["gender"].iloc[0] == "M"


class TestRunQualityChecks:
    def test_all_clean_rows_pass_through(self):
        df = make_df([_base_row(customer_id=i) for i in range(1, 6)])
        clean_df, quarantined_df, report = run_quality_checks(df)
        assert len(clean_df) == 5
        assert len(quarantined_df) == 0
        assert report.quality_score_pct == 100.0

    def test_mixed_quality_rows_are_split_correctly(self):
        rows = [
            _base_row(customer_id=1),                      # clean
            _base_row(customer_id=2, age=999),              # out of range
            _base_row(customer_id=3, income_category=None), # null
            _base_row(customer_id=3, income_category=None), # duplicate of above
        ]
        df = make_df(rows)
        clean_df, quarantined_df, report = run_quality_checks(df)
        assert report.total_rows_in == 4
        assert report.total_rows_clean == 1
        assert report.total_rows_quarantined == 3
        assert report.out_of_range_flagged == 1
        assert report.null_rows_flagged == 2

    def test_missing_required_column_raises(self):
        df = make_df([_base_row()]).drop(columns=["age"])
        with pytest.raises(ValueError):
            run_quality_checks(df)
