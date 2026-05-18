import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class KMeansFromScratch:
    """K-Means++ Clustering algorithm built conceptually from scratch in NumPy."""

    def __init__(self, n_clusters: int = 4, max_iter: int = 300, tol: float = 1e-4, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def _init_centroids_pp(self, X: np.ndarray) -> np.ndarray:
        """K-means++ centroid initialization."""
        np.random.seed(self.seed)
        N, D = X.shape
        centroids = np.empty((self.n_clusters, D))

        # First centroid chosen uniformly at random
        first_idx = np.random.randint(N)
        centroids[0] = X[first_idx]

        # Select remaining centroids with probability proportional to D(x)^2
        for k in range(1, self.n_clusters):
            # Compute distance squared to nearest existing centroid
            dist_sq = np.min(np.sum((X[:, np.newaxis, :] - centroids[:k, :]) ** 2, axis=2), axis=1)
            probs = dist_sq / np.sum(dist_sq)
            next_idx = np.random.choice(N, p=probs)
            centroids[k] = X[next_idx]

        return centroids

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fit K-Means model and return (labels, centroids, inertia_wcss)."""
        N, D = X.shape
        self.centroids = self._init_centroids_pp(X)

        for iteration in range(self.max_iter):
            # Assign samples to nearest centroid
            distances = np.sum((X[:, np.newaxis, :] - self.centroids) ** 2, axis=2) # (N, K)
            self.labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.empty_like(self.centroids)
            for k in range(self.n_clusters):
                cluster_pts = X[self.labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = cluster_pts.mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.randint(N)]

            # Check convergence
            center_shift = np.sum((new_centroids - self.centroids) ** 2)
            self.centroids = new_centroids

            if center_shift < self.tol:
                break

        # Compute Within-Cluster Sum of Squares (Inertia)
        self.inertia_ = float(np.sum((X - self.centroids[self.labels]) ** 2))
        return self.labels, self.centroids, self.inertia_


class HierarchicalAgglomerativeFromScratch:
    """Agglomerative Hierarchical Clustering built conceptually from scratch."""

    def __init__(self, n_clusters: int = 4, linkage: str = "average"):
        self.n_clusters = n_clusters
        self.linkage = linkage.lower()

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        # Start with each point in its own cluster
        clusters: Dict[int, List[int]] = {i: [i] for i in range(N)}

        # Precompute pairwise distance matrix
        dist_matrix = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2) # Euclidean squared

        # Merge clusters until n_clusters remain
        while len(clusters) > self.n_clusters:
            keys = list(clusters.keys())
            min_dist = float("inf")
            merge_pair = (keys[0], keys[1])

            for i in range(len(keys)):
                c1_idx = keys[i]
                c1_pts = clusters[c1_idx]

                for j in range(i + 1, len(keys)):
                    c2_idx = keys[j]
                    c2_pts = clusters[c2_idx]

                    sub_dists = dist_matrix[np.ix_(c1_pts, c2_pts)]
                    if self.linkage == "complete":
                        d = np.max(sub_dists)
                    elif self.linkage == "single":
                        d = np.min(sub_dists)
                    else: # average
                        d = np.mean(sub_dists)

                    if d < min_dist:
                        min_dist = d
                        merge_pair = (c1_idx, c2_idx)

            # Merge pair
            keep_id, remove_id = merge_pair
            clusters[keep_id].extend(clusters[remove_id])
            del clusters[remove_id]

        labels = np.zeros(N, dtype=int)
        for cluster_label, (cid, pts) in enumerate(clusters.items()):
            labels[pts] = cluster_label

        return labels


class GaussianMixtureFromScratch:
    """Gaussian Mixture Model (GMM) with Expectation-Maximization (EM) algorithm from scratch."""

    def __init__(self, n_clusters: int = 4, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def fit_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit GMM via EM algorithm. Returns (hard_labels, soft_probabilities, means)."""
        np.random.seed(self.seed)
        N, D = X.shape

        # Initialize parameters using K-Means++
        kmeans = KMeansFromScratch(n_clusters=self.n_clusters, seed=self.seed)
        labels, means, _ = kmeans.fit_predict(X)

        weights = np.ones(self.n_clusters) / self.n_clusters
        covariances = np.array([np.eye(D) for _ in range(self.n_clusters)])

        log_likelihood_old = -float("inf")
        responsibilities = np.zeros((N, self.n_clusters))

        for iteration in range(self.max_iter):
            # --- E-Step: Compute soft responsibilities gamma ---
            for k in range(self.n_clusters):
                diff = X - means[k]
                # Regularized covariance for numerical stability
                cov_reg = covariances[k] + 1e-6 * np.eye(D)
                inv_cov = np.linalg.inv(cov_reg)
                det_cov = np.linalg.det(cov_reg)

                norm_const = 1.0 / np.sqrt((2 * np.pi) ** D * max(det_cov, 1e-12))
                exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
                responsibilities[:, k] = weights[k] * norm_const * np.exp(np.clip(exponent, -50, 50))

            resp_sum = np.sum(responsibilities, axis=1, keepdims=True)
            resp_sum = np.where(resp_sum == 0, 1e-12, resp_sum)
            responsibilities = responsibilities / resp_sum

            # --- M-Step: Update weights, means, and covariances ---
            Nk = np.sum(responsibilities, axis=0) # (K,)
            weights = Nk / N

            for k in range(self.n_clusters):
                if Nk[k] > 0:
                    means[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / Nk[k]
                    diff = X - means[k]
                    covariances[k] = (responsibilities[:, k:k+1] * diff).T @ diff / Nk[k] + 1e-5 * np.eye(D)

            # Check convergence via log-likelihood
            log_likelihood = np.sum(np.log(resp_sum))
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        hard_labels = np.argmax(responsibilities, axis=1)
        return hard_labels, responsibilities, means
