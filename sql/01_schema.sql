-- ============================================================
-- ChurnGuard: Credit Card Customer Attrition Analytics
-- Schema: raw ingestion layer + cleaned layer + model/reporting tables
-- Run this once against an empty database:
--   mysql -u <user> -p < sql/01_schema.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS churnguard
    CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE churnguard;

-- ------------------------------------------------------------
-- 1. RAW LAYER — mirrors the incoming data feed as-is.
--    No constraints beyond types, so nothing is rejected here;
--    the data-quality layer decides what's clean vs. quarantined.
-- ------------------------------------------------------------
DROP TABLE IF EXISTS raw_customers;
CREATE TABLE raw_customers (
    row_id                      INT AUTO_INCREMENT PRIMARY KEY,
    customer_id                 BIGINT,
    age                         INT,
    gender                      VARCHAR(10),
    dependent_count             INT,
    education_level             VARCHAR(30),
    marital_status              VARCHAR(20),
    income_category             VARCHAR(30),
    card_category               VARCHAR(20),
    months_on_book              FLOAT,
    total_relationship_count    INT,
    months_inactive_12_mon      INT,
    contacts_count_12_mon       INT,
    credit_limit                DECIMAL(12,2),
    total_revolving_bal         DECIMAL(12,2),
    avg_open_to_buy             DECIMAL(12,2),
    total_trans_amt             DECIMAL(12,2),
    total_trans_ct              INT,
    total_amt_chng_q4_q1        DECIMAL(6,3),
    total_ct_chng_q4_q1         DECIMAL(6,3),
    avg_utilization_ratio       DECIMAL(6,4),
    attrition_flag              TINYINT,
    ingested_at                 TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 2. DATA QUALITY LOG — every pipeline run writes its scorecard here,
--    so quality is tracked over time, not just checked once.
-- ------------------------------------------------------------
DROP TABLE IF EXISTS data_quality_log;
CREATE TABLE data_quality_log (
    run_id              INT AUTO_INCREMENT PRIMARY KEY,
    run_timestamp        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_rows_in        INT,
    null_rows_flagged    INT,
    duplicate_rows_flagged INT,
    out_of_range_flagged INT,
    total_rows_quarantined INT,
    total_rows_clean     INT,
    quality_score_pct    DECIMAL(5,2)
);

-- ------------------------------------------------------------
-- 3. CLEANED / CURATED LAYER — output of the ETL + data quality steps.
--    This is the table the feature engineering and modeling code reads from.
-- ------------------------------------------------------------
DROP TABLE IF EXISTS clean_customers;
CREATE TABLE clean_customers (
    customer_id                 BIGINT PRIMARY KEY,
    age                         INT,
    gender                      VARCHAR(10),
    dependent_count             INT,
    education_level             VARCHAR(30),
    marital_status              VARCHAR(20),
    income_category             VARCHAR(30),
    card_category               VARCHAR(20),
    months_on_book              INT,
    total_relationship_count    INT,
    months_inactive_12_mon      INT,
    contacts_count_12_mon       INT,
    credit_limit                DECIMAL(12,2),
    total_revolving_bal         DECIMAL(12,2),
    avg_open_to_buy             DECIMAL(12,2),
    total_trans_amt             DECIMAL(12,2),
    total_trans_ct              INT,
    total_amt_chng_q4_q1        DECIMAL(6,3),
    total_ct_chng_q4_q1         DECIMAL(6,3),
    avg_utilization_ratio       DECIMAL(6,4),
    attrition_flag              TINYINT,
    loaded_at                   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 4. FEATURE STORE — engineered features + segment, refreshed each run.
--    This is what a "data platform powering analytics" looks like: a
--    single governed table that both the ML code and BI tool read from.
-- ------------------------------------------------------------
DROP TABLE IF EXISTS customer_features;
CREATE TABLE customer_features (
    customer_id                 BIGINT PRIMARY KEY,
    revolving_util_ratio        DECIMAL(6,4),
    avg_trans_value             DECIMAL(12,2),
    trans_amt_per_month         DECIMAL(12,2),
    engagement_score            DECIMAL(6,3),
    inactivity_risk_flag        TINYINT,
    value_risk_segment          VARCHAR(30),
    churn_risk_score            DECIMAL(6,4),
    revenue_at_risk             DECIMAL(14,2),
    updated_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES clean_customers(customer_id)
);

-- ------------------------------------------------------------
-- Helpful indexes for the reporting queries in 03_dashboard_queries.sql
-- ------------------------------------------------------------
CREATE INDEX idx_features_segment ON customer_features (value_risk_segment);
CREATE INDEX idx_features_risk    ON customer_features (churn_risk_score);
CREATE INDEX idx_clean_attrition  ON clean_customers (attrition_flag);
