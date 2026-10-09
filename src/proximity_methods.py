"""
proximity_methods.py : KNN distance score and Local Outlier Factor.
Both rely on neighbourhood distances, which concentrate in high dimensions
(Beyer et al. 1999; Aggarwal et al. 2001): the contrast between nearest and farthest
neighbour shrinks, so scores may become less discriminative as d grows.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from .base import BaseDetector


class KNNDetector(BaseDetector):
    """Distance to the k-th nearest neighbour (Ramaswamy et al. 2000); 'mean'/'median'
    aggregate the k distances instead. Self excluded. Higher distance = more anomalous
    (no inversion needed)."""
    name, family = "KNN", "proximity"

    def __init__(self, k: int = 10, method: str = "largest", metric: str = "euclidean"):
        super().__init__()
        self.k, self.method, self.metric = k, method, metric

    def fit_score(self, X):
        X = self._check(X)
        k = max(1, min(self.k, len(X) - 1))
        nn = NearestNeighbors(n_neighbors=k, metric=self.metric).fit(X)
        dist, _ = nn.kneighbors(None)              # None => exclude the point itself
        self.extras["k"] = k
        return {"largest": dist[:, -1], "mean": dist.mean(axis=1),
                "median": np.median(dist, axis=1)}[self.method]


class LOFDetector(BaseDetector):
    """Local Outlier Factor (Breunig et al. 2000): ratio of the average local reachability
    density of a point's k neighbours to its own. LOF ~ 1 inliers, >> 1 outliers.
    sklearn exposes negative_outlier_factor_ (higher = more normal), so
    score = -negative_outlier_factor_ (inversion). Native rule: LOF > 1.5 (common heuristic)."""
    name, family = "LOF", "proximity"

    def __init__(self, k: int = 20, metric: str = "euclidean"):
        super().__init__()
        self.k, self.metric = k, metric

    def fit_score(self, X):
        X = self._check(X)
        k = max(2, min(self.k, len(X) - 1))
        lof = LocalOutlierFactor(n_neighbors=k, metric=self.metric, novelty=False).fit(X)
        s = -lof.negative_outlier_factor_
        self.extras["k"] = k
        self._native = s > 1.5
        return s
