import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Campaign Action Rules per Segment Persona
CAMPAIGN_RULES: Dict[str, Dict[str, Any]] = {
    "Champions (High Value, Active)": {
        "action": "VIP Loyalty Access & Exclusive Early Product Drops",
        "description": "Invite to VIP club, grant early product access, offer premium customer support. Avoid margin-slashing discounts.",
        "expected_cr_generic": 0.12,
        "expected_cr_personalized": 0.28,
        "aov_multiplier_generic": 1.0,
        "aov_multiplier_personalized": 1.15
    },
    "Loyal At-Risk (High Value, Churn Risk)": {
        "action": "20% Win-Back Re-Engagement Coupon & Personalized Outreach",
        "description": "Send urgent win-back email with high-value discount coupon (20% off) valid for 7 days.",
        "expected_cr_generic": 0.04,
        "expected_cr_personalized": 0.18,
        "aov_multiplier_generic": 0.9,
        "aov_multiplier_personalized": 1.10
    },
    "Frequent Budget (Low Spend, Active)": {
        "action": "Cross-Sell / Upsell Bundle Deals & $75 Free Shipping Threshold",
        "description": "Encourage basket size growth using complementary product bundle recommendations and minimum spend thresholds.",
        "expected_cr_generic": 0.08,
        "expected_cr_personalized": 0.22,
        "aov_multiplier_generic": 1.0,
        "aov_multiplier_personalized": 1.35
    },
    "Hibernating (Low Value, Inactive)": {
        "action": "Low-Cost Automated Email Flash Sale Retargeting",
        "description": "Run automated low-cost email retargeting with clear 'We Miss You' message and clearance items.",
        "expected_cr_generic": 0.02,
        "expected_cr_personalized": 0.07,
        "aov_multiplier_generic": 0.8,
        "aov_multiplier_personalized": 0.95
    }
}

class PersonalizationEngine:
    """Generates personalized marketing campaign recommendations and simulates revenue lift."""

    def generate_segment_recommendations(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach targeted marketing actions to segment profiles."""
        recommendations = []
        for p in profiles:
            persona = p.get("persona") or p.get("persona_name", "Hibernating (Low Value, Inactive)")
            rule = CAMPAIGN_RULES.get(persona, CAMPAIGN_RULES["Hibernating (Low Value, Inactive)"])

            rec = dict(p)
            rec["recommended_action"] = rule["action"]
            rec["action_description"] = rule["description"]
            recommendations.append(rec)

        return recommendations

    def simulate_revenue_impact(self, rfm_df: pd.DataFrame, labels: np.ndarray, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simulate total expected revenue comparing a non-segmented baseline campaign
        against targeted segment-personalized campaigns.
        """
        df = rfm_df.copy()
        df["cluster"] = labels

        total_customers = len(df)
        total_baseline_revenue = 0.0
        total_personalized_revenue = 0.0
        segment_simulation_details = []

        for p in profiles:
            c_id = p["cluster_id"]
            persona = p.get("persona") or p.get("persona_name", "Hibernating (Low Value, Inactive)")
            rule = CAMPAIGN_RULES.get(persona, CAMPAIGN_RULES["Hibernating (Low Value, Inactive)"])

            cluster_df = df[df["cluster"] == c_id]
            n_cust = len(cluster_df)
            hist_aov = float(cluster_df["avg_order_value"].mean())

            # Baseline Campaign Metrics
            base_cr = rule["expected_cr_generic"]
            base_aov = hist_aov * rule["aov_multiplier_generic"]
            base_rev = n_cust * base_cr * base_aov

            # Personalized Campaign Metrics
            pers_cr = rule["expected_cr_personalized"]
            pers_aov = hist_aov * rule["aov_multiplier_personalized"]
            pers_rev = n_cust * pers_cr * pers_aov

            total_baseline_revenue += base_rev
            total_personalized_revenue += pers_rev

            segment_simulation_details.append({
                "persona": persona,
                "customer_count": n_cust,
                "baseline_cr": f"{base_cr * 100:.1f}%",
                "personalized_cr": f"{pers_cr * 100:.1f}%",
                "baseline_revenue": round(base_rev, 2),
                "personalized_revenue": round(pers_rev, 2),
                "segment_revenue_lift": round(pers_rev - base_rev, 2)
            })

        revenue_lift_dollars = total_personalized_revenue - total_baseline_revenue
        revenue_lift_pct = (revenue_lift_dollars / max(1.0, total_baseline_revenue)) * 100.0

        return {
            "total_customers": total_customers,
            "baseline_campaign_revenue": round(total_baseline_revenue, 2),
            "personalized_campaign_revenue": round(total_personalized_revenue, 2),
            "revenue_lift_dollars": round(revenue_lift_dollars, 2),
            "revenue_lift_percentage": round(revenue_lift_pct, 2),
            "segment_breakdown": segment_simulation_details
        }
