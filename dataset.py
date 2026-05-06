import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List

CATEGORIES = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Books", "Sports"]

def generate_synthetic_transactions(n_customers: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic e-commerce transaction dataset with distinct customer behavioral archetypes."""
    np.random.seed(seed)
    random.seed(seed)

    base_date = datetime(2026, 8, 1)
    records = []

    # Define 4 underlying customer personas for synthetic data generation
    # 1. Champions: High monetary, high frequency, recent
    # 2. Loyal At-Risk: High monetary, high frequency, old recency
    # 3. Frequent Budget: Low monetary, high frequency, recent
    # 4. Hibernating / One-Time: Low monetary, low frequency, old recency

    for cid in range(1, n_customers + 1):
        archetype = np.random.choice(["champion", "loyal_at_risk", "frequent_budget", "hibernating"], p=[0.25, 0.25, 0.25, 0.25])

        if archetype == "champion":
            num_orders = np.random.randint(8, 25)
            avg_amount = np.random.uniform(150, 400)
            max_days_ago = np.random.randint(1, 30)
            discount_prob = 0.2
        elif archetype == "loyal_at_risk":
            num_orders = np.random.randint(6, 18)
            avg_amount = np.random.uniform(120, 350)
            max_days_ago = np.random.randint(90, 240)
            discount_prob = 0.3
        elif archetype == "frequent_budget":
            num_orders = np.random.randint(10, 30)
            avg_amount = np.random.uniform(20, 60)
            max_days_ago = np.random.randint(1, 45)
            discount_prob = 0.6
        else: # hibernating
            num_orders = np.random.randint(1, 3)
            avg_amount = np.random.uniform(15, 75)
            max_days_ago = np.random.randint(120, 365)
            discount_prob = 0.5

        for o in range(num_orders):
            days_ago = max_days_ago + np.random.randint(0, 30 * o + 1)
            inv_date = base_date - timedelta(days=days_ago)
            amount = max(5.0, round(np.random.normal(avg_amount, avg_amount * 0.25), 2))
            quantity = np.random.randint(1, 5)
            category = random.choice(CATEGORIES)
            discount = round(random.choice([0.0, 0.1, 0.15, 0.2, 0.3]) if random.random() < discount_prob else 0.0, 2)

            records.append({
                "customer_id": f"CUST-{cid:04d}",
                "invoice_date": inv_date,
                "amount": amount,
                "quantity": quantity,
                "category": category,
                "discount": discount
            })

    df = pd.DataFrame(records)
    return df


def extract_rfm_features(df: pd.DataFrame, reference_date: datetime = datetime(2026, 8, 1)) -> pd.DataFrame:
    """Extract RFM (Recency, Frequency, Monetary) and behavioral features per customer."""
    # Ensure invoice_date is datetime
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    # Aggregations per customer
    rfm = df.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (reference_date - x.max()).days),
        frequency=("invoice_date", "count"),
        monetary=("amount", "sum"),
        avg_order_value=("amount", "mean"),
        category_diversity=("category", "nunique"),
        discount_ratio=("discount", lambda x: (x > 0).mean())
    ).reset_index()

    # Recency cannot be negative
    rfm["recency"] = rfm["recency"].clip(lower=0)

    return rfm


def prepare_clustering_matrix(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    """
    Log transform right-skewed features and apply StandardScaler normalization.
    Returns: (scaled_X_matrix, feature_names, scaler_params)
    """
    feature_cols = ["recency", "frequency", "monetary", "avg_order_value", "category_diversity", "discount_ratio"]
    X_raw = rfm_df[feature_cols].copy()

    # Apply log1p transformation to Recency, Frequency, Monetary, and AOV
    X_log = X_raw.copy()
    for col in ["recency", "frequency", "monetary", "avg_order_value"]:
        X_log[col] = np.log1p(X_raw[col])

    # Standard Scaling: (X - mean) / std
    means = X_log.mean().to_dict()
    stds = X_log.std().to_dict()
    # Handle zero std
    for col in feature_cols:
        if stds[col] == 0:
            stds[col] = 1.0

    X_scaled = (X_log - X_log.mean()) / X_log.std()

    scaler_params = {"means": means, "stds": stds}
    return X_scaled.values, feature_cols, scaler_params
