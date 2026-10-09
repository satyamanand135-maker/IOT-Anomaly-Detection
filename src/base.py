"""Common detector interface. CONTRACT: fit_score(X) returns one float per row,
HIGHER = MORE ANOMALOUS. Any inversion/normalisation is done inside the detector
and documented in its docstring."""
from __future__ import annotations
import numpy as np


class BaseDetector:
    name = "base"
    family = "base"

    def __init__(self):
        self.extras: dict = {}          # diagnostics (condition number, eps, backend ...)
        self._native: np.ndarray | None = None

    def fit_score(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def native_predict(self) -> np.ndarray | None:
        """Boolean flags from the algorithm's own decision rule (if it has one), after fit_score."""
        return self._native

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN/inf: impute before detection")
        return X
