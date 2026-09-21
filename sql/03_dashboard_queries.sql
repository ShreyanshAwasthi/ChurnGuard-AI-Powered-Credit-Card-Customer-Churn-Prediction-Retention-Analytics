-- ============================================================
-- ChurnGuard: Dashboard queries
-- Paste these directly into Power BI's "Get Data -> MySQL database ->
-- Advanced options -> SQL statement" box, or Tableau's custom SQL
-- connection, to build each panel.
-- ============================================================

USE churnguard;

-- Panel 1: Headline KPIs (churn rate, at-risk customers, revenue exposure)
SELECT
    COUNT(*)                                              AS total_customers,
    ROUND(AVG(attrition_flag) * 100, 2)                   AS overall_churn_rate_pct,
    SUM(CASE WHEN f.churn_risk_score >= 0.5 THEN 1 ELSE 0 END)      AS high_risk_customers,
    ROUND(SUM(CASE WHEN f.churn_risk_score >= 0.5 THEN f.revenue_at_risk ELSE 0 END), 2) AS total_revenue_at_risk
FROM clean_customers c
JOIN customer_features f ON c.customer_id = f.customer_id;

-- Panel 2: Churn rate & predicted risk by segment (bar chart)
SELECT * FROM vw_segment_summary;

-- Panel 3: Retention watchlist table (sortable grid visual)
SELECT * FROM vw_retention_watchlist LIMIT 200;

-- Panel 4: Churn rate by card category
SELECT
    card_category,
    COUNT(*) AS customers,
    ROUND(AVG(attrition_flag) * 100, 2) AS churn_rate_pct
FROM clean_customers
GROUP BY card_category
ORDER BY churn_rate_pct DESC;

-- Panel 5: Churn rate by months inactive (line/trend chart)
SELECT
    months_inactive_12_mon,
    COUNT(*) AS customers,
    ROUND(AVG(attrition_flag) * 100, 2) AS churn_rate_pct
FROM clean_customers
GROUP BY months_inactive_12_mon
ORDER BY months_inactive_12_mon;

-- Panel 6: Data quality trend over pipeline runs (governance panel)
SELECT
    run_timestamp,
    total_rows_in,
    total_rows_quarantined,
    quality_score_pct
FROM data_quality_log
ORDER BY run_timestamp;
