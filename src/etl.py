"""
etl.py
------
Extract  : read the raw CSV data feed
Transform: normalize + run data quality checks (src/data_quality.py)
Load     : write raw rows to `raw_customers`, clean rows to `clean_customers`,
           and log the run's scorecard to `data_quality_log`.

This is the layer that would, in production, run on a schedule against a
real incoming data feed (e.g. a nightly extract from the core banking system).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from sqlalchemy import create_engine

import config
from data_quality import run_quality_checks


def load_raw_csv() -> pd.DataFrame:
    if not config.RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at {config.RAW_DATA_PATH}. "
            "Run `python src/generate_data.py` first."
        )
    return pd.read_csv(config.RAW_DATA_PATH)


def run_etl(engine=None, write_to_db: bool = True) -> pd.DataFrame:
    """
    Runs the full ETL step. Returns the clean DataFrame either way; only
    writes to MySQL if write_to_db=True (lets tests/CI run without a DB).
    """
    print("[ETL] Extracting raw data...")
    raw_df = load_raw_csv()
    print(f"[ETL] Loaded {len(raw_df):,} raw rows.")

    print("[ETL] Running data quality checks...")
    clean_df, quarantined_df, report = run_quality_checks(raw_df)
    print(f"[ETL] Quality score: {report.quality_score_pct}% "
          f"({report.total_rows_clean:,} clean / {report.total_rows_quarantined:,} quarantined)")

    if write_to_db:
        engine = engine or create_engine(config.SQLALCHEMY_DATABASE_URL)
        print("[ETL] Loading raw_customers table...")
        raw_df.to_sql("raw_customers", engine, if_exists="append", index=False,
                      chunksize=1000)

        print("[ETL] Loading clean_customers table...")
        clean_cols = [c for c in clean_df.columns if c != "quarantine_reason"]
        clean_df[clean_cols].to_sql(
            "clean_customers", engine, if_exists="append", index=False, chunksize=1000
        )

        print("[ETL] Logging data quality scorecard...")
        report_df = pd.DataFrame([report.as_dict()])
        report_df.to_sql("data_quality_log", engine, if_exists="append", index=False)

        # Save quarantined rows locally for inspection/audit (not loaded to DB)
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        quarantined_df.to_csv(config.PROCESSED_DIR / "quarantined_rows.csv", index=False)
        print(f"[ETL] Quarantined rows written to {config.PROCESSED_DIR / 'quarantined_rows.csv'}")

    return clean_df


if __name__ == "__main__":
    run_etl()
