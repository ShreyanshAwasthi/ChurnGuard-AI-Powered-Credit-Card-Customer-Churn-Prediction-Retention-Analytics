"""
pipeline.py
-----------
Runs the entire ChurnGuard pipeline end to end, in order:

  1. Generate synthetic raw data (skipped if it already exists)
  2. ETL: extract -> data quality checks -> load raw + clean tables to MySQL
  3. Feature engineering: derive behavioral features
  4. Segmentation: K-Means value-risk clustering
  5. Modeling: train/compare Logistic Regression, Random Forest, XGBoost
  6. Feature store: write engineered features + risk scores to MySQL
  7. Dashboard: export flat CSV for Power BI/Tableau + a standalone HTML dashboard

Usage:
    python src/pipeline.py

Requires a MySQL server running and reachable using the credentials in .env
(see .env.example and the README's Setup section).
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from sqlalchemy import create_engine, text

import config
import generate_data
from etl import run_etl
from feature_engineering import add_engineered_features
from segmentation import segment_customers
from churn_model import train_and_evaluate, save_artifacts
from dashboard_export import build_feature_store_table, write_feature_store, build_html_dashboard


def reset_tables(engine):
    """
    Clears prior pipeline output so the run is idempotent -- re-running
    the pipeline (e.g. after regenerating data) won't hit duplicate primary
    key errors. Deletes in FK-safe order: children before parents.
    """
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM customer_features"))
        conn.execute(text("DELETE FROM clean_customers"))
        conn.execute(text("DELETE FROM raw_customers"))
        conn.execute(text("DELETE FROM data_quality_log"))
    print("[RESET] Cleared previous pipeline output from MySQL tables.")


def main():
    start = time.time()
    print("=" * 60)
    print("ChurnGuard: Credit Card Customer Attrition Analytics")
    print("=" * 60)

    if not config.RAW_DATA_PATH.exists():
        print("\n[STEP 1/7] Generating synthetic raw data...")
        generate_data.main()
    else:
        print("\n[STEP 1/7] Raw data already exists, skipping generation.")

    engine = create_engine(config.SQLALCHEMY_DATABASE_URL)
    print("\n[STEP 2/7] Resetting tables for a clean, idempotent run...")
    reset_tables(engine)

    print("\n[STEP 3/7] Running ETL (extract -> data quality -> load)...")
    clean_df = run_etl(engine=engine, write_to_db=True)

    print("\n[STEP 4/7] Engineering behavioral features...")
    feat_df = add_engineered_features(clean_df)

    print("\n[STEP 5/7] Segmenting customers (K-Means)...")
    seg_df = segment_customers(feat_df)
    print(seg_df["value_risk_segment"].value_counts().to_string())

    print("\n[STEP 6/7] Training churn models (Logistic Regression, Random Forest, XGBoost)...")
    outcome = train_and_evaluate(seg_df)
    save_artifacts(outcome, config.OUTPUTS_DIR)

    print("\n[STEP 7/7] Writing feature store + building dashboard...")
    feature_store_df = build_feature_store_table(seg_df, outcome["all_customer_scores"])
    write_feature_store(feature_store_df, engine=engine)
    summary = build_html_dashboard(seg_df, feature_store_df, config.OUTPUTS_DIR / "dashboard.html")

    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Best model     : {outcome['best_model_name']}")
    print(f"ROC-AUC        : {outcome['results_by_model'][outcome['best_model_name']]['roc_auc']}")
    print(f"Churn rate     : {summary['overall_churn_rate_pct']}%")
    print(f"High-risk count: {summary['high_risk_customers']:,}")
    print(f"Revenue at risk: ${summary['total_revenue_at_risk']:,.0f}")
    print(f"Dashboard      : {config.OUTPUTS_DIR / 'dashboard.html'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
