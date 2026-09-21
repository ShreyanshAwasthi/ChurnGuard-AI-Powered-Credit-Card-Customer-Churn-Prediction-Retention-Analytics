# ChurnGuard AI: Credit Card Customer Attrition Prediction & Retention Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosted%20Trees-green)
![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end, AI-powered analytics pipeline that predicts which credit card
customers are likely to churn, explains *why*, and turns that into a
prioritized, revenue-ranked watchlist a retention team could actually act on —
covering the full lifecycle from raw data to a governed feature store to a
BI-ready dashboard, not just a notebook that trains a model.

> **On the data:** this project uses a synthetically generated dataset (see
> `src/generate_data.py`) built to mirror real-world credit card customer
> attributes and well-documented industry churn drivers — inactivity, declining
> transaction activity, low product relationship count, high utilization. Real
> bank customer data is never public, so the generator produces a realistic,
> fully reproducible dataset (including intentionally dirty records for the
> data-quality layer to catch) instead of depending on a third-party dataset.
> Every metric below comes from an actual run of this pipeline.

---

## Table of Contents
- [Business Objective](#business-objective)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Dashboard & Reporting](#dashboard--reporting)
- [Design Decisions](#design-decisions)
- [License](#license)

---

## Business Objective

Credit cards are a high-margin, high-priority growth product for banks —
revenue comes from interest on revolving balances, fees, and cross-sell to
active cardholders. A churned cardholder doesn't just stop using one product;
it erases all future revenue from that relationship at once. ChurnGuard AI
builds the analytics layer a retention team needs to intervene **before** that
happens: who is at risk, how much revenue is at stake, and what's driving the
risk.

## Architecture

```
Raw data (CSV)
     │
     ▼
┌───────────────────────┐     ┌────────────────────────┐
│   Data Quality Layer    │────▶│    data_quality_log      │  (MySQL)
│  (null/dup/range        │     │    (scorecard per run)   │
│   checks + quarantine)  │     └────────────────────────┘
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│   raw_customers          │  (MySQL — as-is ingestion)
│   clean_customers         │  (MySQL — validated records only)
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Feature Engineering     │  utilization ratio, engagement score,
│  (src/feature_           │  spend trend, avg transaction value...
│   engineering.py)        │
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Segmentation (K-Means)  │  4 value-risk segments
└───────────┬─────────────┘
            ▼
┌───────────────────────┐
│  Churn Modeling (AI)     │  Logistic Regression / Random Forest / XGBoost
│  (src/churn_model.py)    │  → best model selected by ROC-AUC
└───────────┬─────────────┘
            ▼
┌───────────────────────┐     ┌────────────────────────────┐
│  customer_features        │──▶│  SQL views (feature store)   │  → Power BI / Tableau
│  (MySQL feature store)    │   │  vw_customer_360,            │  → outputs/dashboard.html
└───────────────────────┘     │  vw_retention_watchlist,      │
                               │  vw_segment_summary           │
                               └────────────────────────────┘
```

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.12 |
| Data manipulation | Pandas, NumPy |
| Database / feature store | MySQL 8, SQLAlchemy, PyMySQL |
| Machine learning | Scikit-learn (Logistic Regression, Random Forest, K-Means), XGBoost |
| Reporting / BI | Power BI / Tableau (via SQL views), Plotly (bundled HTML dashboard) |
| Testing | Pytest |

---

## Results

*(from an actual run of this pipeline — see `outputs/metrics.json`)*

**Data Quality:** 94.82% of ingested records passed all validation checks
(10,050 raw rows in → 9,529 clean, 521 quarantined for nulls, duplicate
customer IDs, or out-of-range values).

**Model Comparison:**

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.836** | 0.745 | 0.472 | **0.782** | 0.589 |
| XGBoost | 0.827 | 0.762 | 0.493 | 0.724 | 0.587 |
| Random Forest | 0.822 | 0.793 | 0.554 | 0.587 | 0.570 |

Logistic Regression was selected as the production model — it catches **78.2%
of customers who actually churn** (recall), which matters more than raw
accuracy for a retention use case, where missing an at-risk high-value
customer is far more costly than a false alarm.

**Top churn drivers** (by feature importance): transaction count decline (Q4
vs Q1), overall engagement score, months inactive, transaction amount
decline, and total transaction count — consistent with the hypothesis that
*declining engagement*, not any single static attribute, drives attrition.

**Customer Segments** (K-Means, 4 segments):

| Segment | Customers | Actual Churn Rate | Revenue at Risk |
|---|---|---|---|
| High-Value At-Risk | 2,381 | **43.6%** | $15.3M |
| High-Value Engaged | 2,353 | 25.2% | $10.4M |
| Low-Value Dormant | 2,586 | 16.6% | $8.1M |
| Low-Value Active | 2,209 | **7.5%** | $3.9M |

The **High-Value At-Risk** segment churns at nearly 6x the rate of Low-Value
Active customers — exactly the group a retention team should prioritize, and
exactly what `vw_retention_watchlist` surfaces, ranked by revenue exposure.

---

## Project Structure

```
churnguard-ai/
├── README.md
├── requirements.txt
├── .env.example              # copy to .env and fill in your MySQL credentials
├── .gitignore
├── pytest.ini
├── LICENSE
├── data/
│   └── raw/                  # generated synthetic dataset lands here
├── sql/
│   ├── 01_schema.sql                # run first: creates all tables
│   ├── 02_feature_store_views.sql   # run after the first pipeline run
│   └── 03_dashboard_queries.sql     # paste into Power BI / Tableau
├── src/
│   ├── config.py               # loads DB credentials from .env
│   ├── generate_data.py        # synthetic data generator
│   ├── data_quality.py         # validation rules + scorecard
│   ├── etl.py                  # extract -> quality checks -> load to MySQL
│   ├── feature_engineering.py  # derived behavioral features
│   ├── segmentation.py         # K-Means value-risk segmentation
│   ├── churn_model.py          # trains/evaluates LR, RF, XGBoost
│   ├── dashboard_export.py     # feature store write + HTML dashboard
│   └── pipeline.py             # runs everything above, in order
├── tests/                      # pytest suite, 36 tests, no DB required
├── powerbi/                    # .pbix file + dashboard screenshot go here
└── outputs/                    # generated: model, metrics, dashboard (gitignored)
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- MySQL Server 8.0+ installed and running locally ([Windows](https://dev.mysql.com/downloads/installer/) / [Mac](https://dev.mysql.com/downloads/mysql/) / `sudo apt install mysql-server` on Linux)

### 1. Clone and install dependencies
```bash
git clone https://github.com/<your-username>/churnguard-ai.git
cd churnguard-ai
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up MySQL
```sql
CREATE USER 'churnguard_user'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON churnguard.* TO 'churnguard_user'@'localhost';
FLUSH PRIVILEGES;
```
```bash
mysql -u churnguard_user -p < sql/01_schema.sql
```

### 3. Configure credentials
```bash
cp .env.example .env
# edit .env with your DB_USER / DB_PASSWORD
```

### 4. Run the entire pipeline (one command)
```bash
python src/pipeline.py
```
This generates the synthetic dataset (first run only), runs data quality
checks, loads MySQL tables, engineers features, segments customers, trains
all three models, writes the feature store, and builds the dashboard — done
in under 10 seconds.

### 5. Build the feature-store views (run once, after the first pipeline run)
```bash
mysql -u churnguard_user -p < sql/02_feature_store_views.sql
```

### 6. View the results
- **Dashboard:** open `outputs/dashboard.html` in any browser
- **Power BI / Tableau:** connect to your local MySQL `churnguard` database and
  use the queries in `sql/03_dashboard_queries.sql`, or import
  `outputs/dashboard_export.csv` directly
- **Model + metrics:** `outputs/churn_model.pkl` and `outputs/metrics.json`

---

## Running Tests

```bash
pytest
```
36 tests covering data quality rules, feature engineering, segmentation, and
model training — all run on small in-memory samples, so no MySQL connection
is required to run the suite.

---

## Dashboard & Reporting

`.pbix` and Tableau workbook files are proprietary binary formats that only
their desktop apps can write, so this project instead ships everything needed
to build the real dashboard in minutes:
- `sql/03_dashboard_queries.sql` — paste directly into Power BI's or Tableau's
  custom SQL connection
- `outputs/dashboard_export.csv` — a flat file ready for direct import
- `outputs/dashboard.html` — a working interactive Plotly dashboard, viewable
  immediately with no BI tool installed
- `powerbi/` — where the actual `.pbix` file and a dashboard screenshot live
  once built (see `powerbi/README.md` for the build steps)

---

## Design Decisions

- **Why synthetic data?** Full control over reproducibility, no dataset
  licensing ambiguity, and the ability to demonstrate the data-quality layer
  against known, intentionally-injected issues.
- **Why is ROC-AUC ~0.84 and not 0.99?** Near-perfect separation on customer
  churn is a red flag for data leakage. The noise injected into the synthetic
  churn labels keeps this realistic — comparable to published churn-modeling
  case studies in banking and telecom.
- **Why does Logistic Regression beat XGBoost here?** The underlying churn
  probability was generated as a linear combination of standardized features,
  which favors a linear model. On real-world data with genuine non-linear
  interactions, tree-based models typically pull ahead — the pipeline already
  benchmarks all three, so swapping in real data doesn't require code changes.

---

## License

This project is licensed under the [MIT License](LICENSE).
