"""
tree_methods.py : Isolation Forest and Extended Isolation Forest.

Isolation Forest : scikit-learn implementation (Liu, Ting & Zhou 2008).
Extended IF      : Hariri, Kind & Brunner (2021), IEEE TKDE, doi:10.1109/TKDE.2019.2947676.
    Standard IF splits on one random feature with a random threshold (axis-parallel cuts),
    producing score artefacts along axes. EIF splits with random HYPERPLANES: a random
    normal vector n ~ N(0, I) (with d-1-ext coordinates zeroed, ext = "extension level")
    and a random intercept point p drawn uniformly from the node's bounding box; a point x
    goes left iff (x - p) . n <= 0.

EIF BACKENDS (`backend="auto"` picks the first available and records it in `extras`):
    1. isotree  - Cortes' C++ library (pip install isotree). Configured for EIF-style splits
                  (ndim = extension_level+1, ntry=1, gain-based pickers disabled, normal
                  coefficients). It is a *reputable EIF-capable* implementation but not
                  byte-identical to Hariri's reference code (e.g. intercept sampling).
    2. eif      - Hariri's reference package (pip install eif; needs a C compiler).
    3. numpy    - our own faithful re-implementation of Hariri et al.'s algorithm
                  (below); slower but dependency-free. Unit-tested in tests/.
Because backends differ slightly, the backend used is stored with every result.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseDetector

EULER = 0.5772156649


def c_factor(n: float) -> float:
    """Average path length of unsuccessful search in a BST (normalising constant)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (math.log(n - 1.0) + EULER) - 2.0 * (n - 1.0) / n


class IsolationForestDetector(BaseDetector):
    """sklearn IsolationForest. score = -score_samples(X) (sklearn returns the *negated*
    anomaly score of the paper, so we flip it back: higher = more anomalous)."""
    name, family = "IsolationForest", "tree"

    def __init__(self, n_estimators: int = 200, max_samples: int | str = 256,
                 random_state: int = 0):
        super().__init__()
        self.n_estimators, self.max_samples, self.random_state = n_estimators, max_samples, random_state

    def fit_score(self, X):
        X = self._check(X)
        ms = self.max_samples if isinstance(self.max_samples, str) else min(self.max_samples, len(X))
        m = IsolationForest(n_estimators=self.n_estimators, max_samples=ms,
                            random_state=self.random_state, n_jobs=1).fit(X)
        return -m.score_samples(X)


# --------------------------------------------------------------------------- #
# Own EIF implementation (Hariri et al. 2021, Algorithms 1-3)
# --------------------------------------------------------------------------- #
class _EIFTree:
    def __init__(self, X: np.ndarray, height_limit: int, ext: int, rng: np.random.Generator):
        d = X.shape[1]
        self.normal, self.offset, self.left, self.right, self.size = [], [], [], [], []
        self.ext, self.d, self.rng, self.hlim = ext, d, rng, height_limit
        self._build(X, 0)
        self.normal = np.asarray(self.normal)
        self.offset = np.asarray(self.offset)
        self.left = np.asarray(self.left)
        self.right = np.asarray(self.right)
        self.size = np.asarray(self.size)

    def _new_node(self):
        self.normal.append(np.zeros(self.d)); self.offset.append(0.0)
        self.left.append(-1); self.right.append(-1); self.size.append(0)
        return len(self.left) - 1

    def _build(self, X, depth) -> int:
        node = self._new_node()
        n = len(X)
        if depth >= self.hlim or n <= 1:
            self.size[node] = n
            return node
        lo, hi = X.min(axis=0), X.max(axis=0)
        if np.all(hi == lo):
            self.size[node] = n
            return node
        nvec = self.rng.normal(size=self.d)
        n_zero = self.d - 1 - self.ext                       # zero out coordinates (extension level)
        if n_zero > 0:
            nvec[self.rng.choice(self.d, size=n_zero, replace=False)] = 0.0
        p = self.rng.uniform(lo, hi)                          # intercept inside the bounding box
        b = float(p @ nvec)
        go_left = X @ nvec <= b
        if go_left.all() or (~go_left).all():                 # degenerate cut -> leaf
            self.size[node] = n
            return node
        self.normal[node], self.offset[node] = nvec, b
        self.left[node] = self._build(X[go_left], depth + 1)
        self.right[node] = self._build(X[~go_left], depth + 1)
        return node

    def path_lengths(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        node = np.zeros(n, dtype=int)
        depth = np.zeros(n)
        active = self.left[node] != -1
        while active.any():
            idx = np.flatnonzero(active)
            nd = node[idx]
            go_left = np.einsum("ij,ij->i", X[idx], self.normal[nd]) <= self.offset[nd]
            node[idx] = np.where(go_left, self.left[nd], self.right[nd])
            depth[idx] += 1
            active = self.left[node] != -1
        adj = np.array([c_factor(s) for s in self.size[node]])
        return depth + adj


class _NumpyEIF:
    def __init__(self, n_trees=100, sample_size=256, ext=None, seed=0):
        self.n_trees, self.sample_size, self.ext, self.seed = n_trees, sample_size, ext, seed

    def fit(self, X):
        n, d = X.shape
        self.psi = min(self.sample_size, n)
        ext = d - 1 if self.ext is None else int(np.clip(self.ext, 0, d - 1))
        hlim = int(math.ceil(math.log2(max(self.psi, 2))))
        rng = np.random.default_rng([self.seed, 505])
        self.trees = []
        for _ in range(self.n_trees):
            sub = X[rng.choice(n, size=self.psi, replace=False)]
            self.trees.append(_EIFTree(sub, hlim, ext, rng))
        return self

    def score(self, X):
        h = np.mean([t.path_lengths(X) for t in self.trees], axis=0)
        return 2.0 ** (-h / c_factor(self.psi))              # in (0,1); higher = more anomalous


class ExtendedIsolationForestDetector(BaseDetector):
    """Extended Isolation Forest. Score s(x,psi) = 2^(-E[h(x)]/c(psi)) in (0,1);
    higher = more anomalous, no inversion needed. extension_level=None => fully extended
    (hyperplanes use all d coordinates), the recommended default in Hariri et al."""
    name, family = "ExtendedIF", "tree"

    def __init__(self, n_estimators: int = 100, max_samples: int = 256,
                 extension_level: int | None = None, backend: str = "auto",
                 random_state: int = 0):
        super().__init__()
        self.n_estimators, self.max_samples = n_estimators, max_samples
        self.extension_level, self.backend, self.random_state = extension_level, backend, random_state

    def _try_isotree(self, X):
        from isotree import IsolationForest as ITF  # noqa
        n, d = X.shape
        ext = d - 1 if self.extension_level is None else self.extension_level
        m = ITF(ntrees=self.n_estimators, sample_size=min(self.max_samples, n),
                ndim=int(ext) + 1, ntry=1, coefs="normal", prob_pick_avg_gain=0,
                prob_pick_pooled_gain=0, penalize_range=False, standardize_data=False,
                missing_action="fail", random_seed=self.random_state, nthreads=1).fit(X)
        return np.asarray(m.predict(X), dtype=float)

    def _try_eif(self, X):
        import eif  # noqa
        n, d = X.shape
        ext = d - 1 if self.extension_level is None else self.extension_level
        f = eif.iForest(X, ntrees=self.n_estimators, sample_size=min(self.max_samples, n),
                        ExtensionLevel=int(ext))
        return np.asarray(f.compute_paths(X_in=X), dtype=float)

    def fit_score(self, X):
        X = self._check(X)
        order = ["isotree", "eif", "numpy"] if self.backend == "auto" else [self.backend]
        err = {}
        for b in order:
            try:
                if b == "isotree":
                    s = self._try_isotree(X)
                elif b == "eif":
                    s = self._try_eif(X)
                else:
                    s = _NumpyEIF(self.n_estimators, self.max_samples,
                                  self.extension_level, self.random_state).fit(X).score(X)
                self.extras["backend"] = b
                return s
            except (ImportError, TypeError, ValueError) as exc:   # missing package / API mismatch
                err[b] = f"{type(exc).__name__}: {exc}"
                self.extras["backend_errors"] = err
        raise RuntimeError(f"No EIF backend available: {err}")
