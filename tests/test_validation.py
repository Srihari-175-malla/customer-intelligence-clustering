import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    from customer_segmentation.dataset import (
        generate_synthetic_transactions,
        extract_rfm_features,
        prepare_clustering_matrix,
    )
    from customer_segmentation.clustering import KMeansFromScratch
    from customer_segmentation.validation import calculate_silhouette_score, SegmentProfiler
    from customer_segmentation.personalization import PersonalizationEngine
except ImportError:
    from dataset import (
        generate_synthetic_transactions,
        extract_rfm_features,
        prepare_clustering_matrix,
    )
    from clustering import KMeansFromScratch
    from validation import calculate_silhouette_score, SegmentProfiler
    from personalization import PersonalizationEngine


class TestValidationAndPersonalization(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_transactions(n_customers=40, seed=42)
        self.rfm = extract_rfm_features(self.df)
        self.X_scaled, _, _ = prepare_clustering_matrix(self.rfm)
        self.kmeans = KMeansFromScratch(n_clusters=3, seed=42)
        self.labels, _, _ = self.kmeans.fit_predict(self.X_scaled)

    def test_silhouette_score(self):
        sil = calculate_silhouette_score(self.X_scaled, self.labels)
        self.assertGreaterEqual(sil, -1.0)
        self.assertLessEqual(sil, 1.0)

    def test_segment_profiler_and_revenue_sim(self):
        profiler = SegmentProfiler()
        profiles = profiler.profile_segments(self.rfm, self.labels)
        self.assertEqual(len(profiles), 3)

        engine = PersonalizationEngine()
        recs = engine.generate_segment_recommendations(profiles)
        self.assertEqual(len(recs), 3)
        self.assertIn("recommended_action", recs[0])

        sim = engine.simulate_revenue_impact(self.rfm, self.labels, profiles)
        self.assertGreater(sim["personalized_campaign_revenue"], 0)


if __name__ == "__main__":
    unittest.main()
