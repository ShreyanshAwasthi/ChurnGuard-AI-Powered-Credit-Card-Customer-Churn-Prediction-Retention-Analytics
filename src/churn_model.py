"""
churn_model.py
--------------
Trains and compares three classifiers (Logistic Regression, Random Forest,
XGBoost) to predict customer attrition, selects the best by ROC-AUC, and
saves the trained model + metrics + feature importances for the pipeline
and dashboard to consume.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLUMNS_FOR_MODEL, CATEGORICAL_COLUMNS

RANDOM_SEED = 42
TARGET = "attrition_flag"


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), FEATURE_COLUMNS_FOR_MODEL),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
    ])


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            eval_metric="logloss", random_state=RANDOM_SEED,
            scale_pos_weight=(1 - 0.22) / 0.22,  # roughly (majority/minority) ratio
        ),
    }


def train_and_evaluate(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS_FOR_MODEL + CATEGORICAL_COLUMNS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor()
    results = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
        }
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"[MODEL] {name}: {metrics}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n[MODEL] Best model by ROC-AUC: {best_name}")

    # Feature importance from the best tree-based model (fallback to
    # logistic regression coefficients if best model has no importances)
    feature_names = (
        FEATURE_COLUMNS_FOR_MODEL
        + list(best_pipe.named_steps["preprocess"]
               .named_transformers_["cat"].get_feature_names_out(CATEGORICAL_COLUMNS))
    )
    model_step = best_pipe.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    else:
        importances = np.abs(model_step.coef_[0])
    top_features = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(10)
    )

    # Confusion matrix + full classification report for the writeup
    y_pred_best = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best).tolist()
    report = classification_report(y_test, y_pred_best, output_dict=True)

    # Score every customer (not just the test split) so the whole book of
    # business gets a risk score for the feature store / dashboard.
    all_proba = best_pipe.predict_proba(X)[:, 1]

    return {
        "results_by_model": results,
        "best_model_name": best_name,
        "best_pipeline": best_pipe,
        "top_features": top_features,
        "confusion_matrix": cm,
        "classification_report": report,
        "all_customer_scores": pd.Series(all_proba, index=df.index, name="churn_risk_score"),
    }


def save_artifacts(outcome: dict, outputs_dir: Path):
    outputs_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(outcome["best_pipeline"], outputs_dir / "churn_model.pkl")

    metrics_payload = {
        "best_model": outcome["best_model_name"],
        "results_by_model": outcome["results_by_model"],
        "top_features": outcome["top_features"].round(4).to_dict(),
        "confusion_matrix": outcome["confusion_matrix"],
        "classification_report": outcome["classification_report"],
    }
    with open(outputs_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[MODEL] Saved model -> {outputs_dir / 'churn_model.pkl'}")
    print(f"[MODEL] Saved metrics -> {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    from data_quality import run_quality_checks
    from feature_engineering import add_engineered_features
    from segmentation import segment_customers

    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_customers.csv"
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"

    df = pd.read_csv(raw_path)
    clean_df, _, _ = run_quality_checks(df)
    feat_df = add_engineered_features(clean_df)
    seg_df = segment_customers(feat_df)

    outcome = train_and_evaluate(seg_df)
    save_artifacts(outcome, outputs_dir)

    print("\nTop 10 churn drivers:")
    print(outcome["top_features"])
