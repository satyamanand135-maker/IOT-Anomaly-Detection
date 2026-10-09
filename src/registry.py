"""registry.py : algorithm registry with FIXED default hyper-parameters.
Defaults are deliberately not tuned per condition (tuning would require labels);
sensitivity to k / eps is studied separately in experiments.run_sensitivity()."""
from __future__ import annotations

from .density_methods import DBSCANDetector
from .proximity_methods import KNNDetector, LOFDetector
from .statistical_methods import GrubbsDetector, MahalanobisDetector
from .tree_methods import ExtendedIsolationForestDetector, IsolationForestDetector

ALGORITHMS = {
    "Mahalanobis": MahalanobisDetector,
    "Grubbs": GrubbsDetector,
    "KNN": KNNDetector,
    "LOF": LOFDetector,
    "DBSCAN": DBSCANDetector,
    "IsolationForest": IsolationForestDetector,
    "ExtendedIF": ExtendedIsolationForestDetector,
}
FAMILY = {"Mahalanobis": "statistical", "Grubbs": "statistical", "KNN": "proximity",
          "LOF": "proximity", "DBSCAN": "density", "IsolationForest": "tree",
          "ExtendedIF": "tree"}
SEEDED = {"Mahalanobis", "IsolationForest", "ExtendedIF"}


def make_detector(name: str, params: dict | None = None, seed: int = 0):
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name}; choose from {list(ALGORITHMS)}")
    params = dict(params or {})
    if name in SEEDED:
        params.setdefault("random_state", seed)
    return ALGORITHMS[name](**params)
