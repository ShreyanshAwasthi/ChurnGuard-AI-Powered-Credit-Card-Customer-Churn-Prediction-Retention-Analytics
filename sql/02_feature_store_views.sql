-- ============================================================
-- ChurnGuard: Feature-store style SQL views
-- These sit on top of clean_customers + customer_features and give
-- analytics/BI consumers a stable, documented interface instead of
-- querying raw tables directly (this IS the "data platform" layer).
-- Run after 01_schema.sql and after the pipeline has populated the tables.
-- ============================================================

USE churnguard;

-- View 1: single wide table joining cleaned attributes with engineered
-- features and risk score — the main table Power BI / Tableau connects to.
CREATE OR REPLACE VIEW vw_customer_360 AS
SELECT
    c.customer_id,
    c.age,
    c.gender,
    c.income_category,
    c.card_category,
    c.months_on_book,
    c.total_relationship_count,
    c.months_inactive_12_mon,
    c.credit_limit,
    c.total_revolving_bal,
    c.total_trans_amt,
    c.total_trans_ct,
    c.attrition_flag,
    f.revolving_util_ratio,
    f.engagement_score,
    f.inactivity_risk_flag,
    f.value_risk_segment,
    f.churn_risk_score,
    f.revenue_at_risk
FROM clean_customers c
JOIN customer_features f ON c.customer_id = f.customer_id;

-- View 2: early-warning watchlist — high-risk, high-value customers,
-- ranked by revenue exposure, for the retention team to action first.
CREATE OR REPLACE VIEW vw_retention_watchlist AS
SELECT
    c.customer_id,
    c.card_category,
    f.churn_risk_score,
    c.credit_limit,
    f.revenue_at_risk,
    f.value_risk_segment,
    c.months_inactive_12_mon
FROM clean_customers c
JOIN customer_features f ON c.customer_id = f.customer_id
WHERE f.churn_risk_score >= 0.5
ORDER BY f.revenue_at_risk DESC;

-- View 3: segment-level summary for the dashboard's top-level KPIs.
CREATE OR REPLACE VIEW vw_segment_summary AS
SELECT
    f.value_risk_segment,
    COUNT(*)                                   AS customer_count,
    ROUND(AVG(c.attrition_flag) * 100, 2)      AS actual_churn_rate_pct,
    ROUND(AVG(f.churn_risk_score) * 100, 2)    AS avg_predicted_risk_pct,
    ROUND(SUM(f.revenue_at_risk), 2)           AS total_revenue_at_risk,
    ROUND(AVG(c.credit_limit), 2)              AS avg_credit_limit
FROM clean_customers c
JOIN customer_features f ON c.customer_id = f.customer_id
GROUP BY f.value_risk_segment
ORDER BY total_revenue_at_risk DESC;
