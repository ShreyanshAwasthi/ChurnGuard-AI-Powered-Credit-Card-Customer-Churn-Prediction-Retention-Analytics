"""
dashboard_export.py
--------------------
Two things happen here:

1. Writes the engineered features + segment + churn risk score + revenue-at-
   risk for every customer into the `customer_features` table in MySQL --
   this is the feature store that Power BI / Tableau connects to via the
   SQL views in sql/02_feature_store_views.sql.

2. Generates a self-contained interactive HTML dashboard (Plotly) as a
   quick-look substitute for an actual Power BI / Tableau file, since those
   are proprietary desktop-app file formats. Open outputs/dashboard.html in
   any browser to see it -- and use sql/03_dashboard_queries.sql to build
   the real Power BI/Tableau version by pointing at the MySQL feature store.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def build_feature_store_table(seg_df: pd.DataFrame, churn_scores: pd.Series) -> pd.DataFrame:
    df = seg_df.copy()
    df["churn_risk_score"] = churn_scores.round(4)
    df["revenue_at_risk"] = (df["churn_risk_score"] * df["credit_limit"]).round(2)

    feature_cols = [
        "customer_id", "revolving_util_ratio", "avg_trans_value",
        "trans_amt_per_month", "engagement_score", "inactivity_risk_flag",
        "value_risk_segment", "churn_risk_score", "revenue_at_risk",
    ]
    return df[feature_cols]


def write_feature_store(feature_store_df: pd.DataFrame, engine=None):
    engine = engine or create_engine(config.SQLALCHEMY_DATABASE_URL)
    feature_store_df.to_sql("customer_features", engine, if_exists="append", index=False, chunksize=1000)
    print(f"[DASHBOARD] Wrote {len(feature_store_df):,} rows to customer_features.")


def build_html_dashboard(seg_df: pd.DataFrame, feature_store_df: pd.DataFrame, output_path: Path,
                          high_risk_threshold: float = 0.5):
    merged = seg_df.merge(feature_store_df, on="customer_id", suffixes=("", "_fs"))

    overall_churn_rate = merged["attrition_flag"].mean() * 100
    high_risk = merged[merged["churn_risk_score"] >= high_risk_threshold]
    total_revenue_at_risk = high_risk["revenue_at_risk"].sum()

    segment_summary = (
        merged.groupby("value_risk_segment")
        .agg(customers=("customer_id", "count"),
             actual_churn_rate=("attrition_flag", "mean"),
             avg_predicted_risk=("churn_risk_score", "mean"),
             total_revenue_at_risk=("revenue_at_risk", "sum"))
        .reset_index()
        .sort_values("total_revenue_at_risk", ascending=False)
    )

    inactivity_trend = (
        merged.groupby("months_inactive_12_mon")["attrition_flag"]
        .mean().reset_index()
    )

    card_trend = (
        merged.groupby("card_category")["attrition_flag"]
        .agg(["mean", "count"]).reset_index()
        .sort_values("mean", ascending=False)
    )

    watchlist = (
        high_risk.sort_values("revenue_at_risk", ascending=False)
        .head(15)[["customer_id", "value_risk_segment", "churn_risk_score", "credit_limit", "revenue_at_risk"]]
    )

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Churn Rate & Revenue-at-Risk by Segment",
            "Churn Rate by Months Inactive",
            "Churn Rate by Card Category",
            "Top 15 Retention Watchlist (by Revenue at Risk)",
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "table"}]],
        vertical_spacing=0.15,
    )

    fig.add_trace(go.Bar(x=segment_summary["value_risk_segment"],
                          y=segment_summary["actual_churn_rate"] * 100,
                          name="Churn Rate %", marker_color="#d9534f"), row=1, col=1)

    fig.add_trace(go.Scatter(x=inactivity_trend["months_inactive_12_mon"],
                              y=inactivity_trend["attrition_flag"] * 100,
                              mode="lines+markers", name="Churn Rate % by Inactivity",
                              line=dict(color="#f0ad4e")), row=1, col=2)

    fig.add_trace(go.Bar(x=card_trend["card_category"], y=card_trend["mean"] * 100,
                          name="Churn Rate % by Card", marker_color="#5bc0de"), row=2, col=1)

    fig.add_trace(go.Table(
        header=dict(values=["Customer ID", "Segment", "Risk Score", "Credit Limit", "Revenue at Risk"],
                    fill_color="#333", font=dict(color="white")),
        cells=dict(values=[watchlist["customer_id"], watchlist["value_risk_segment"],
                           watchlist["churn_risk_score"], watchlist["credit_limit"],
                           watchlist["revenue_at_risk"]])
    ), row=2, col=2)

    fig.update_layout(
        title=f"ChurnGuard — Credit Card Customer Attrition Dashboard<br>"
              f"<sup>Overall churn rate: {overall_churn_rate:.1f}% | "
              f"High-risk customers: {len(high_risk):,} | "
              f"Total revenue at risk: ${total_revenue_at_risk:,.0f}</sup>",
        height=900, showlegend=False,
        template="plotly_white",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"[DASHBOARD] Wrote interactive dashboard -> {output_path}")

    # Also export the flat table Power BI/Tableau would import directly
    csv_path = output_path.parent / "dashboard_export.csv"
    merged.to_csv(csv_path, index=False)
    print(f"[DASHBOARD] Wrote flat export for Power BI/Tableau -> {csv_path}")

    return {
        "overall_churn_rate_pct": round(overall_churn_rate, 2),
        "high_risk_customers": len(high_risk),
        "total_revenue_at_risk": round(float(total_revenue_at_risk), 2),
    }


if __name__ == "__main__":
    from data_quality import run_quality_checks
    from feature_engineering import add_engineered_features
    from segmentation import segment_customers
    from churn_model import train_and_evaluate

    raw_path = config.RAW_DATA_PATH
    df = pd.read_csv(raw_path)
    clean_df, _, _ = run_quality_checks(df)
    feat_df = add_engineered_features(clean_df)
    seg_df = segment_customers(feat_df)
    outcome = train_and_evaluate(seg_df)

    feature_store_df = build_feature_store_table(seg_df, outcome["all_customer_scores"])
    write_feature_store(feature_store_df)

    summary = build_html_dashboard(seg_df, feature_store_df, config.OUTPUTS_DIR / "dashboard.html")
    print(summary)
