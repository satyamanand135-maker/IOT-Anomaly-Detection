"""
density_methods.py : DBSCAN used as an anomaly scorer.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .base import BaseDetector


def knee_eps(kdist_sorted: np.ndarray) -> float:
    """Elbow of the sorted k-distance curve = point farthest below the chord joining its ends."""
    y = (kdist_sorted - kdist_sorted[0]) / max(kdist_sorted[-1] - kdist_sorted[0], 1e-12)
    x = np.linspace(0, 1, len(y))
    return float(kdist_sorted[int(np.argmax(x - y))])


class DBSCANDetector(BaseDetector):
    """DBSCAN (Ester et al. 1996). Points labelled noise (-1) are anomalies.

    DBSCAN yields only a hard noise label, so a continuous score is defined as
        score = dist(x, nearest core point) / eps
    = 0 for core points, <= 1 for border points, > 1 exactly for noise points. Native rule: score > 1.

    PARAMETER SELECTION (the core difficulty in high dimensions): eps is chosen by the
    k-distance elbow (k = min_samples) [default] or a quantile of the k-distance; min_samples
    is fixed (default 10) rather than tuned, to avoid using labels. In high dimensions the
    k-distance curve flattens (distance concentration), so the elbow becomes ill-defined and
    DBSCAN tends to degenerate to "everything noise" or "one cluster"; we record this in
    `extras` (n_clusters, noise_fraction, degenerate).
    """
    name, family = "DBSCAN", "density"

    def __init__(self, eps: float | None = None, min_samples: int = 10,
                 eps_strategy: str = "knee", quantile: float = 0.90):
        super().__init__()
        self.eps, self.min_samples = eps, min_samples
        self.eps_strategy, self.quantile = eps_strategy, quantile

    def fit_score(self, X):
        X = self._check(X)
        ms = max(2, min(self.min_samples, len(X) - 1))
        nn = NearestNeighbors(n_neighbors=ms).fit(X)
        kd = np.sort(nn.kneighbors(None)[0][:, -1])
        if self.eps is not None:
            eps = float(self.eps)
        elif self.eps_strategy == "quantile":
            eps = float(np.quantile(kd, self.quantile))
        else:
            eps = knee_eps(kd)
        eps = max(eps, 1e-9)
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        labels, core = db.labels_, db.core_sample_indices_
        n_clusters = len(set(labels) - {-1})
        noise_frac = float((labels == -1).mean())
        self.extras.update({"eps": eps, "min_samples": ms, "n_clusters": n_clusters,
                            "noise_fraction": noise_frac,
                            "degenerate": bool(noise_frac > 0.5 or n_clusters == 0)})
        if len(core) == 0:                            # no core point: fall back to k-distance
            score = nn.kneighbors(None)[0][:, -1] / eps
        else:
            cnn = NearestNeighbors(n_neighbors=1).fit(X[core])
            score = cnn.kneighbors(X)[0][:, 0] / eps  # exactly 0 for core points
        self._native = score > 1.0
        return score
