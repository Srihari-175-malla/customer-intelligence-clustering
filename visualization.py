import numpy as np
from typing import Dict, List, Any
from sklearn.decomposition import PCA

def compute_pca_2d_projection(X_scaled: np.ndarray, labels: np.ndarray, customer_ids: List[str]) -> Dict[str, Any]:
    """Compute 2D PCA projection for cluster scatter plot visualization."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = [round(float(v) * 100, 2) for v in pca.explained_variance_ratio_]

    points = []
    for i in range(len(customer_ids)):
        points.append({
            "customer_id": customer_ids[i],
            "pc1": round(float(X_pca[i, 0]), 3),
            "pc2": round(float(X_pca[i, 1]), 3),
            "cluster": int(labels[i])
        })

    # Compute centroids in PCA space
    centroids = []
    for c_id in np.unique(labels):
        mask = (labels == c_id)
        if np.sum(mask) > 0:
            c_pc1 = float(np.mean(X_pca[mask, 0]))
            c_pc2 = float(np.mean(X_pca[mask, 1]))
            centroids.append({
                "cluster": int(c_id),
                "pc1": round(c_pc1, 3),
                "pc2": round(c_pc2, 3)
            })

    return {
        "variance_explained": var_explained,
        "points": points,
        "centroids": centroids
    }
