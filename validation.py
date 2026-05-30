import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

try:
    from .clustering import KMeansFromScratch
except (ImportError, ValueError):
    from clustering import KMeansFromScratch

def calculate_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate mean Silhouette Score across all samples from scratch in NumPy."""
    N = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= N:
        return 0.0

    # Compute pairwise Euclidean distance matrix
    dist_matrix = np.sqrt(np.maximum(0, np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)))

    silhouette_vals = np.zeros(N)

    for i in range(N):
        own_cluster = labels[i]
        own_mask = (labels == own_cluster)
        own_mask[i] = False  # exclude self

        if np.sum(own_mask) == 0:
            silhouette_vals[i] = 0.0
            continue

        # Mean intra-cluster distance (a_i)
        a_i = np.mean(dist_matrix[i, own_mask])

        # Mean nearest-cluster distance (b_i)
        other_clusters = [c for c in unique_labels if c != own_cluster]
        b_i_candidates = [
            np.mean(dist_matrix[i, labels == c]) for c in other_clusters if np.sum(labels == c) > 0
        ]
        b_i = min(b_i_candidates) if b_i_candidates else 0.0

        denom = max(a_i, b_i)
        silhouette_vals[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(silhouette_vals))


def calculate_wcss_elbow(X: np.ndarray, max_k: int = 8, seed: int = 42) -> Dict[str, List[float]]:
    """Compute WCSS across multiple K values for Elbow Curve heuristic."""
    k_range = list(range(2, min(max_k + 1, X.shape[0])))
    wcss_list = []

    for k in k_range:
        kmeans = KMeansFromScratch(n_clusters=k, seed=seed)
        _, _, inertia = kmeans.fit_predict(X)
        wcss_list.append(round(inertia, 2))

    return {
        "k_values": k_range,
        "wcss": wcss_list
    }


class SegmentProfiler:
    """Generates business profiles and human-interpretable personas for customer clusters."""

    SEGMENT_NAMES = {
        0: "Champions (High Value, Active)",
        1: "Loyal At-Risk (High Spend, Lapsed)",
        2: "Frequent Budget Shoppers",
        3: "Hibernating / Low Engagement"
    }

    def profile_segments(self, df_rfm: pd.DataFrame, labels: np.ndarray) -> List[Dict[str, Any]]:
        df = df_rfm.copy()
        df["cluster"] = labels

        profiles = []
        for cluster_id in sorted(np.unique(labels)):
            cluster_data = df[df["cluster"] == cluster_id]
            size = len(cluster_data)
            pct = round(size / len(df) * 100, 1)

            recency_mean = round(float(cluster_data["recency"].mean()), 1)
            frequency_mean = round(float(cluster_data["frequency"].mean()), 1)
            monetary_mean = round(float(cluster_data["monetary"].mean()), 2)
            aov_mean = round(float(cluster_data["avg_order_value"].mean()), 2)
            disc_col = "discount_affinity" if "discount_affinity" in cluster_data.columns else ("discount_ratio" if "discount_ratio" in cluster_data.columns else None)
            discount_mean = round(float(cluster_data[disc_col].mean() * 100), 1) if disc_col else 0.0

            persona_name = self.SEGMENT_NAMES.get(cluster_id % 4, f"Segment {cluster_id + 1}")

            profiles.append({
                "cluster_id": int(cluster_id),
                "persona_name": persona_name,
                "customer_count": int(size),
                "customer_pct": pct,
                "avg_recency_days": recency_mean,
                "avg_frequency_orders": frequency_mean,
                "avg_monetary_spend": monetary_mean,
                "avg_order_value": aov_mean,
                "avg_discount_affinity_pct": discount_mean,
            })

        return profiles
