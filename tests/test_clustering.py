import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

try:
    from customer_segmentation.clustering import (
        KMeansFromScratch,
        HierarchicalAgglomerativeFromScratch,
        GaussianMixtureFromScratch,
    )
except ImportError:
    from clustering import (
        KMeansFromScratch,
        HierarchicalAgglomerativeFromScratch,
        GaussianMixtureFromScratch,
    )


class TestClustering(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        c1 = np.random.randn(25, 4) + 5
        c2 = np.random.randn(25, 4) - 5
        self.X = np.vstack([c1, c2])

    def test_kmeans(self):
        kmeans = KMeansFromScratch(n_clusters=2, seed=42)
        labels, centroids, inertia = kmeans.fit_predict(self.X)
        self.assertEqual(len(labels), 50)
        self.assertEqual(centroids.shape, (2, 4))
        self.assertGreater(inertia, 0.0)

    def test_hierarchical(self):
        agg = HierarchicalAgglomerativeFromScratch(n_clusters=2, linkage="average")
        labels = agg.fit_predict(self.X)
        self.assertEqual(len(labels), 50)
        self.assertEqual(len(np.unique(labels)), 2)

    def test_gmm(self):
        gmm = GaussianMixtureFromScratch(n_clusters=2, seed=42)
        labels, probs, means = gmm.fit_predict(self.X)
        self.assertEqual(len(labels), 50)
        self.assertEqual(probs.shape, (50, 2))


if __name__ == "__main__":
    unittest.main()
