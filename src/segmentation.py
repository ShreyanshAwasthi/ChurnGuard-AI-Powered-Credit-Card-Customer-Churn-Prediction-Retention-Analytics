"""
segmentation.py
----------------
Segments customers into value-risk tiers using K-Means clustering on
engagement, spend, and credit-utilization features. A churn signal means
something different for a high-value dormant customer than for a low-limit
occasional user, so segmentation happens before/alongside the churn model,
not as an afterthought.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

SEGMENTATION_FEATURES = [
    "credit_limit", "total_trans_amt", "total_trans_ct",
    "engagement_score", "revolving_util_ratio", "months_inactive_12_mon",
]

N_CLUSTERS = 4
RANDOM_SEED = 42

# Each archetype's target position in (value, engagement) space, used to
# match each of the 4 K-Means clusters to the business label that best
# describes it. This guarantees 4 distinct, meaningful segment names
# regardless of exactly how the clusters happen to fall.
ARCHETYPES = {
    "High-Value Engaged":  (1, 1),
    "High-Value At-Risk":  (1, -1),
    "Low-Value Active":    (-1, 1),
    "Low-Value Dormant":   (-1, -1),
}


def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[SEGMENTATION_FEATURES].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Profile each cluster on the two axes that matter for the business
    # story: how valuable the customer is (credit_limit) and how engaged
    # they are (engagement_score). Standardize these across cluster means
    # so they're comparable to the (-1, +1) archetype coordinates above.
    profile = df.groupby("cluster")[["credit_limit", "engagement_score"]].mean()
    profile_z = (profile - profile.mean()) / profile.std(ddof=0)

    archetype_names = list(ARCHETYPES.keys())
    archetype_coords = np.array(list(ARCHETYPES.values()))
    cluster_coords = profile_z[["credit_limit", "engagement_score"]].values

    # Optimal one-to-one assignment (Hungarian algorithm): match each
    # cluster to the closest archetype such that no two clusters share
    # a label -- this is what guarantees 4 distinct segment names.
    cost = np.linalg.norm(
        cluster_coords[:, None, :] - archetype_coords[None, :, :], axis=2
    )
    cluster_order, archetype_order = linear_sum_assignment(cost)
    label_map = {
        profile.index[c]: archetype_names[a]
        for c, a in zip(cluster_order, archetype_order)
    }

    df["value_risk_segment"] = df["cluster"].map(label_map)
    df = df.drop(columns=["cluster"])

    return df


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from data_quality import run_quality_checks
    from feature_engineering import add_engineered_features

    raw_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "credit_card_customers.csv"
    df = pd.read_csv(raw_path)
    clean_df, _, _ = run_quality_checks(df)
    feat_df = add_engineered_features(clean_df)
    seg_df = segment_customers(feat_df)
    print(seg_df["value_risk_segment"].value_counts())
    print(seg_df.groupby("value_risk_segment")["attrition_flag"].mean().round(3))
