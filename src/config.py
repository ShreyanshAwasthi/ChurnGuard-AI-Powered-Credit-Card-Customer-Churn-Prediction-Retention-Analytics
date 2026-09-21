"""
config.py
---------
Central configuration, loaded from environment variables (via a .env file).
Never hardcode credentials in the pipeline code -- copy .env.example to .env
and fill in your own local MySQL credentials before running anything.
"""

import os
from pathlib import Path
from urllib.parse import quote_plus
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "churnguard_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "churnguard")

# quote_plus escapes special characters (@, :, etc.) so the password is
# parsed correctly inside the connection URL.
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{quote_plus(DB_PASSWORD)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "credit_card_customers.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

HIGH_RISK_THRESHOLD = 0.5  # churn_risk_score >= this is flagged "high risk"
RANDOM_SEED = 42
