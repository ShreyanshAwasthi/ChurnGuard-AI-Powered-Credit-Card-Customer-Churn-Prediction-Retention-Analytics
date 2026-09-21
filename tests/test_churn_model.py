"""
tests/test_churn_model.py
Tests for churn model training and evaluation. Uses a small synthetic
sample (not the full 10k-row dataset and no MySQL connection needed) so
this stays fast and runnable anywhere.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate_customers
from feature_engineering import add_engineered_features
from segmentation import segment_customers
from churn_model import train_and_evaluate, get_models, FEATURE_COLUMNS_FOR_MODEL, CATEGORICAL_COLUMNS


@pytest.fixture(scope="module")
def small_dataset():
    df = generate_customers(n=800, seed=3)
    df = add_engineered_features(df)
    df = segment_customers(df)
    return df


def test_get_models_returns_three_named_models():
    models = get_models()
    assert set(models.keys()) == {"logistic_regression", "random_forest", "xgboost"}


def test_required_feature_columns_exist_in_generated_data(small_dataset):
    missing = [c for c in FEATURE_COLUMNS_FOR_MODEL + CATEGORICAL_COLUMNS if c not in small_dataset.columns]
    assert missing == []


def test_train_and_evaluate_returns_expected_keys(small_dataset):
    outcome = train_and_evaluate(small_dataset)
    expected_keys = {
        "results_by_model", "best_model_name", "best_pipeline",
        "top_features", "confusion_matrix", "classification_report",
        "all_customer_scores",
    }
    assert expected_keys.issubset(outcome.keys())


def test_all_three_models_produce_metrics(small_dataset):
    outcome = train_and_evaluate(small_dataset)
    assert set(outcome["results_by_model"].keys()) == {"logistic_regression", "random_forest", "xgboost"}
    for metrics in outcome["results_by_model"].values():
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0


def test_best_model_is_selected_by_highest_roc_auc(small_dataset):
    outcome = train_and_evaluate(small_dataset)
    best_auc = outcome["results_by_model"][outcome["best_model_name"]]["roc_auc"]
    all_aucs = [m["roc_auc"] for m in outcome["results_by_model"].values()]
    assert best_auc == max(all_aucs)


def test_churn_risk_scores_are_valid_probabilities(small_dataset):
    outcome = train_and_evaluate(small_dataset)
    scores = outcome["all_customer_scores"]
    assert len(scores) == len(small_dataset)
    assert scores.between(0, 1).all()


def test_top_features_are_nonempty(small_dataset):
    outcome = train_and_evaluate(small_dataset)
    assert len(outcome["top_features"]) > 0
