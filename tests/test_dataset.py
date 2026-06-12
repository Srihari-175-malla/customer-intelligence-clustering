import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from customer_segmentation.dataset import (
        generate_synthetic_transactions,
        extract_rfm_features,
        prepare_clustering_matrix,
    )
except ImportError:
    from dataset import (
        generate_synthetic_transactions,
        extract_rfm_features,
        prepare_clustering_matrix,
    )


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_transactions(n_customers=50, seed=42)

    def test_synthetic_generation(self):
        self.assertGreater(len(self.df), 50)
        self.assertIn("customer_id", self.df.columns)
        self.assertIn("amount", self.df.columns)

    def test_rfm_extraction(self):
        rfm = extract_rfm_features(self.df)
        self.assertEqual(len(rfm), 50)
        self.assertIn("recency", rfm.columns)
        self.assertIn("frequency", rfm.columns)
        self.assertIn("monetary", rfm.columns)
        self.assertTrue((rfm["recency"] >= 0).all())

    def test_matrix_scaling(self):
        rfm = extract_rfm_features(self.df)
        X_scaled, cols, params = prepare_clustering_matrix(rfm)
        self.assertEqual(X_scaled.shape[0], 50)
        self.assertEqual(X_scaled.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
