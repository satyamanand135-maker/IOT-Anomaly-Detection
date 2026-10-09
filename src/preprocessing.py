"""
preprocessing.py
================
Reusable, leakage-aware preprocessing for multivariate sensor data.

LEAKAGE POLICY (important for the report)
-----------------------------------------
Anomaly detection here is UNSUPERVISED: detectors never see ground-truth labels.
Imputers and scalers are therefore fitted on the *unlabelled* experimental data
only (transductive setting: fit on the data to be scored, which is what a
practitioner can do), and NEVER on information derived from anomaly labels.
For a deployment/hold-out setting use `Preprocessor.fit(X_train)` followed by
`Preprocessor.transform(X_new)`: statistics come from the training split only.
Dimension-selection by variance and correlation filtering are also fitted on
the fitting split only.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# --------------------------------------------------------------------------- #
# Column helpers
# --------------------------------------------------------------------------- #
# Priority order used when choosing 1, 2, 4 or 8 derived features per sensor.
# Design choice (documented): steady-state response first, then the fastest
# (alpha=0.1) EMA transients, then slower ones. Nested subsets => 16 < 32 < 64 < 128.
DERIVED_PRIORITY = ["DR", "EMAi0.1", "EMAd0.1", "EMAi0.01",
                    "EMAd0.01", "abs_DR", "EMAi0.001", "EMAd0.001"]
N_SENSORS = 16


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"Sensor\d+_", str(c))]


def sensor_of(col: str) -> int:
    m = re.match(r"Sensor(\d+)_", str(col))
    return int(m.group(1)) if m else -1


def sensor_groups(columns: Iterable[str]) -> list[np.ndarray]:
    """Column-index groups, one per sensor. Columns without a sensor id form singleton groups."""
    columns = list(columns)
    groups: dict[int, list[int]] = {}
    for j, c in enumerate(columns):
        s = sensor_of(c)
        groups.setdefault(s if s > 0 else -(j + 1), []).append(j)
    return [np.array(v) for v in groups.values()]


# --------------------------------------------------------------------------- #
# Missing-value analysis
# --------------------------------------------------------------------------- #
def missing_value_report(df: pd.DataFrame, cols: list[str] | None = None) -> dict:
    cols = cols or feature_columns(df)
    miss = df[cols].isna()
    per_col = miss.sum().sort_values(ascending=False)
    return {
        "total_cells": int(miss.size),
        "missing_cells": int(miss.values.sum()),
        "missing_pct": float(miss.values.mean() * 100),
        "rows_with_missing": int(miss.any(axis=1).sum()),
        "cols_with_missing": int((per_col > 0).sum()),
        "per_column": per_col[per_col > 0].to_frame("n_missing"),
    }


# --------------------------------------------------------------------------- #
# Imputation & scaling factories
# --------------------------------------------------------------------------- #
def make_imputer(name: str, n_neighbors: int = 5):
    name = (name or "none").lower()
    if name in ("mean", "median"):
        return SimpleImputer(strategy=name, keep_empty_features=True)
    if name == "knn":
        return KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    if name == "none":
        return None
    raise ValueError(f"unknown imputer '{name}' (mean|median|knn|none)")


def make_scaler(name: str):
    name = (name or "none").lower()
    table = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
    if name == "none":
        return None
    if name not in table:
        raise ValueError(f"unknown scaler '{name}' (standard|robust|minmax|none)")
    return table[name]()


class Preprocessor:
    """imputer -> scaler, fitted on `fit()` data only."""

    def __init__(self, imputer: str = "median", scaler: str = "standard",
                 n_neighbors: int = 5):
        self.imputer_name, self.scaler_name, self.n_neighbors = imputer, scaler, n_neighbors

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X = np.asarray(X, dtype=float)
        self.imputer_ = make_imputer(self.imputer_name, self.n_neighbors)
        Xi = self.imputer_.fit_transform(X) if self.imputer_ is not None else X
        if np.isnan(Xi).any():
            raise ValueError("NaNs remain: choose an imputer (mean|median|knn).")
        self.scaler_ = make_scaler(self.scaler_name)
        if self.scaler_ is not None:
            self.scaler_.fit(Xi)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xi = self.imputer_.transform(X) if self.imputer_ is not None else X
        return self.scaler_.transform(Xi) if self.scaler_ is not None else Xi

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Feature diagnostics
# --------------------------------------------------------------------------- #
def variance_analysis(df: pd.DataFrame, cols: list[str] | None = None,
                      near_zero: float = 1e-8) -> pd.DataFrame:
    cols = cols or feature_columns(df)
    v = df[cols].var()
    out = pd.DataFrame({"feature": cols, "variance": v.values,
                        "std": np.sqrt(v.values),
                        "coef_variation": (df[cols].std() / df[cols].mean().abs().replace(0, np.nan)).values})
    out["near_zero_variance"] = out["variance"] < near_zero
    return out.sort_values("variance", ascending=False).reset_index(drop=True)


def high_correlation_pairs(df: pd.DataFrame, cols: list[str] | None = None,
                           threshold: float = 0.95) -> pd.DataFrame:
    cols = cols or feature_columns(df)
    c = df[cols].corr().abs()
    iu = np.triu_indices_from(c, k=1)
    pairs = pd.DataFrame({"feature_a": np.array(cols)[iu[0]],
                          "feature_b": np.array(cols)[iu[1]],
                          "abs_corr": c.values[iu]})
    return pairs[pairs.abs_corr >= threshold].sort_values("abs_corr", ascending=False).reset_index(drop=True)


def drop_correlated(df_fit: pd.DataFrame, cols: list[str], threshold: float = 0.95) -> list[str]:
    """Greedy filter fitted on df_fit ONLY; returns the kept columns."""
    c = df_fit[cols].corr().abs().values
    keep: list[int] = []
    for j in range(len(cols)):
        if all(c[j, k] < threshold for k in keep):
            keep.append(j)
    return [cols[j] for j in keep]


def pca_reduce(X_fit: np.ndarray, X: np.ndarray, n_components: int | float,
               seed: int = 0) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=seed).fit(X_fit)
    return pca.transform(X), pca


# --------------------------------------------------------------------------- #
# Dimensionality selection (for the high-dimensionality experiment)
# --------------------------------------------------------------------------- #
def select_dimensions(columns: list[str], n_dims: int, strategy: str = "feature_per_sensor",
                      X_ref: pd.DataFrame | None = None) -> list[str]:
    """Choose `n_dims` of the 128 features.

    feature_per_sensor (default, meaningful): keep ALL 16 sensors and use the first
        n_dims/16 derived features of each sensor (priority list above). Nested subsets
        16 (DeltaR only) < 32 < 64 < 128. Sensor coverage is constant, so dimensionality
        changes feature richness, not which physical sensors exist.
        CAVEAT: added derived features are strongly correlated with existing ones, so
        the *intrinsic* dimensionality grows less than the nominal one.
    sensor_subset: first n_dims/8 sensors with all 8 derived features.
    variance: the n_dims highest-variance columns (needs X_ref; fit on fit-split only).
    first_n: first n_dims columns (naive baseline).
    """
    columns = list(columns)
    if n_dims >= len(columns):
        return columns
    if strategy == "feature_per_sensor":
        if n_dims % N_SENSORS:
            raise ValueError("feature_per_sensor needs n_dims to be a multiple of 16")
        per = n_dims // N_SENSORS
        wanted = [f"Sensor{s}_{d}" for s in range(1, N_SENSORS + 1) for d in DERIVED_PRIORITY[:per]]
        missing = [w for w in wanted if w not in columns]
        if missing:
            raise ValueError(f"expected columns not found: {missing[:3]} ...")
        return wanted
    if strategy == "sensor_subset":
        n_s = max(1, n_dims // 8)
        return [c for c in columns if 1 <= sensor_of(c) <= n_s]
    if strategy == "variance":
        if X_ref is None:
            raise ValueError("variance strategy needs X_ref")
        return list(X_ref[columns].var().sort_values(ascending=False).index[:n_dims])
    if strategy == "first_n":
        return columns[:n_dims]
    raise ValueError(f"unknown strategy {strategy}")


# --------------------------------------------------------------------------- #
# Context encoding (for contextual anomaly detection)
# --------------------------------------------------------------------------- #
def encode_context(gas_class: np.ndarray, concentration: np.ndarray,
                   n_classes: int = 6, weight: float = 1.0) -> np.ndarray:
    """Contextual attributes -> numeric block [one-hot gas | z-scored log10 concentration].

    Chandola et al. (2009) split attributes into *contextual* (here gas, concentration)
    and *behavioural* (the sensor features). Appending this block lets a standard detector
    notice behaviour that is inconsistent with the context. Statistics use no anomaly labels."""
    gas = np.asarray(gas_class).astype(int)
    onehot = np.zeros((len(gas), n_classes))
    onehot[np.arange(len(gas)), np.clip(gas - 1, 0, n_classes - 1)] = 1.0
    lc = np.log10(np.asarray(concentration, dtype=float))
    lc = (lc - lc.mean()) / (lc.std() if lc.std() > 0 else 1.0)
    return weight * np.hstack([onehot, lc[:, None]])


# --------------------------------------------------------------------------- #
# Sensor dropout simulation
# --------------------------------------------------------------------------- #
def simulate_dropout(X: np.ndarray, columns: list[str], fraction: float,
                     strategy: str = "random", seed: int = 0,
                     burst_length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_with_NaN, missing_mask). `fraction` ~ share of values lost.

    random      : individual values lost independently (exact count).
    sensor      : a (row, sensor) block is lost -> all of that sensor's features missing.
    consecutive : burst outages of `burst_length` consecutive observations per sensor
                  (observation order = stream order of the experimental set).
    """
    X = np.array(X, dtype=float, copy=True)
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)
    if fraction <= 0:
        return X, mask
    rng = np.random.default_rng([seed, 101])
    groups = sensor_groups(columns)

    if strategy == "random":
        k = int(round(fraction * n * d))
        mask.flat[rng.choice(n * d, size=k, replace=False)] = True
    elif strategy == "sensor":
        n_blocks = n * len(groups)
        k = int(round(fraction * n_blocks))
        for b in rng.choice(n_blocks, size=k, replace=False):
            r, g = divmod(int(b), len(groups))
            mask[r, groups[g]] = True
    elif strategy == "consecutive":
        target = int(round(fraction * n))
        for g in groups:
            m = np.zeros(n, dtype=bool)
            while m.sum() < target:
                s = int(rng.integers(0, n))
                m[s:s + burst_length] = True
            mask[np.ix_(m, g)] = True
    else:
        raise ValueError(f"unknown dropout strategy '{strategy}'")
    X[mask] = np.nan
    return X, mask


# --------------------------------------------------------------------------- #
# Irregular sampling simulation + resampling
# --------------------------------------------------------------------------- #
def simulate_irregular_sampling(n: int, drop_fraction: float, seed: int = 0,
                                mode: str = "random", burst_length: int = 5) -> np.ndarray:
    """Boolean keep-mask over n regular ticks. SIMULATED: the UCI data has no raw
    irregular timestamps, so we thin a regular grid at random ('random') or in
    gaps ('bursty') to mimic lost/late packets."""
    keep = np.ones(n, dtype=bool)
    if drop_fraction <= 0:
        return keep
    rng = np.random.default_rng([seed, 202])
    target = int(round(drop_fraction * n))
    if mode == "random":
        keep[rng.choice(n, size=target, replace=False)] = False
    elif mode == "bursty":
        while (~keep).sum() < target:
            s = int(rng.integers(0, n))
            keep[s:s + burst_length] = False
    else:
        raise ValueError(f"unknown irregular mode '{mode}'")
    return keep


def fill_missing_stream(X: np.ndarray, method: str = "linear") -> np.ndarray:
    """Column-wise fill along the observation order (time axis).

    ffill  : last observation carried forward (leading gaps back-filled).
    linear : linear interpolation (also extrapolates at the edges by nearest).
    pchip  : monotone cubic (shape preserving; avoids overshoot of splines).
    nearest: nearest valid observation.
    Remaining NaN (all-NaN columns) fall back to the column median, then 0."""
    df = pd.DataFrame(np.asarray(X, dtype=float))
    if method == "ffill":
        df = df.ffill().bfill()
    elif method in ("linear", "nearest", "pchip"):
        try:
            df = df.interpolate(method=method, limit_direction="both")
        except Exception:  # e.g. too few points for pchip
            df = df.interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError(f"unknown interpolation '{method}'")
    df = df.fillna(df.median()).fillna(0.0)
    return df.to_numpy()


STREAM_FILL_METHODS = ("ffill", "linear", "pchip", "nearest")
