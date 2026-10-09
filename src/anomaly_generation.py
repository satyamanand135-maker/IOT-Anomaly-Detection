"""
anomaly_generation.py
=====================
Reproducible, fully-logged anomaly injection with ground truth.

The ORIGINAL data is never modified: every function works on copies.

ASSUMPTIONS (state these in the report)
---------------------------------------
* The UCI file is a table of processed measurements, not a raw timestamped IoT stream.
  "Stream order" below = the row order of the experimental subset (rows are kept in their
  original file order). It is a *constructed* ordering used to define windows; it is not
  claimed to be physical time.
* POINT: feature-level perturbations of individual rows.
      zscore         x_j <- x_j + s * sigma_j * (+-1)
      multiplicative x_j <- x_j * (1 + s)          (scale dependent: documented caveat)
      extreme_replace x_j <- mu_j + (+-1)(s + U(0,1)) * sigma_j
  mu_j, sigma_j come from the clean experimental subset (generator knowledge, never
  given to the detectors). Only `n_features` features per row are altered (default 4):
  a FIXED number, so in higher dimensions the anomalous subspace is a smaller fraction
  of all features. This design choice makes the dilution aspect of the curse of
  dimensionality visible; use `feature_fraction` to scale k with d instead.
* CONTEXTUAL: context = (gas class, concentration). The behavioural vector of a row is
  replaced by a REAL measurement taken from a different gas context (a donor outside the
  experimental subset), while the row keeps its own context labels. Globally the vector is
  a plausible reading (it exists in the data); relative to its labelled context it is
  inconsistent. Detectors only see it as anomalous if the context is provided to them.
* COLLECTIVE: non-overlapping windows of `window_length` consecutive rows receive the SAME
  coordinated shift on a feature subset (a persistent sensor-profile shift, strength
  `collective_strength` x sigma, typically milder than point anomalies). The whole window is
  labelled anomalous. Row-wise detectors see each row of the window individually: a stated
  limitation (no window-level scoring is performed).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

ANOMALY_TYPES = ("point", "contextual", "collective", "mixed")


@dataclass
class InjectionConfig:
    anomaly_type: str = "point"            # point | contextual | collective | mixed
    ratio: float = 0.05                    # fraction of rows labelled anomalous
    strength: float = 3.0                  # point strength in sigmas
    n_features: int = 4                    # features altered per point anomaly / per window
    feature_fraction: float | None = None  # if set, k = max(1, round(fraction * d))
    point_modes: tuple = ("zscore", "extreme_replace", "multiplicative")
    window_length: int = 10
    collective_strength: float = 2.0
    contextual_strategy: str = "swap_gas"  # swap_gas | swap_concentration
    contextual_mix: float = 1.0            # 1 = full donor vector; <1 blends with original
    seed: int = 0

    def k_features(self, d: int) -> int:
        if self.feature_fraction is not None:
            return int(min(d, max(1, round(self.feature_fraction * d))))
        return int(min(d, self.n_features))


@dataclass
class InjectedData:
    X: np.ndarray                 # injected copy
    y: np.ndarray                 # 0 normal / 1 anomaly
    anomaly_type: np.ndarray      # 'normal' | 'point' | 'contextual' | 'collective'
    log: pd.DataFrame             # one row per modified observation (mechanism trace)
    config: InjectionConfig = field(default_factory=InjectionConfig)

    def to_frame(self, columns: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(self.X, columns=columns)
        out["anomaly_label"] = self.y
        out["anomaly_type"] = self.anomaly_type
        return out


def clean_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(~np.isfinite(sd) | (sd == 0), 1.0, sd)
    return mu, sd


# --------------------------------------------------------------------------- #
# Individual mechanisms
# --------------------------------------------------------------------------- #
def _inject_point(X, idx, rng, cfg, mu, sd, log):
    d = X.shape[1]
    k = cfg.k_features(d)
    for i in idx:
        feats = rng.choice(d, size=k, replace=False)
        mode = cfg.point_modes[int(rng.integers(len(cfg.point_modes)))]
        sign = rng.choice([-1.0, 1.0], size=k)
        if mode == "zscore":
            X[i, feats] = X[i, feats] + sign * cfg.strength * sd[feats]
        elif mode == "multiplicative":
            X[i, feats] = X[i, feats] * (1.0 + cfg.strength)
        elif mode == "extreme_replace":
            X[i, feats] = mu[feats] + sign * (cfg.strength + rng.random(k)) * sd[feats]
        else:
            raise ValueError(f"unknown point mode {mode}")
        log.append({"row": int(i), "type": "point", "mode": mode,
                    "features": feats.tolist(), "strength_sigma": cfg.strength})


def _inject_contextual(X, idx, rng, cfg, gas, conc, pool_X, pool_gas, pool_conc, log):
    if pool_X is None or len(pool_X) == 0:
        raise ValueError("contextual injection needs a donor pool (pool_X, pool_gas, pool_conc)")
    for i in idx:
        if cfg.contextual_strategy == "swap_gas":
            cand = np.flatnonzero(pool_gas != gas[i])
        elif cfg.contextual_strategy == "swap_concentration":
            ratio = np.maximum(pool_conc, conc[i]) / np.minimum(pool_conc, conc[i])
            cand = np.flatnonzero((pool_gas == gas[i]) & (ratio >= 5.0))
        else:
            raise ValueError(f"unknown contextual strategy {cfg.contextual_strategy}")
        if len(cand) == 0:  # fall back to any other gas
            cand = np.flatnonzero(pool_gas != gas[i])
        j = int(rng.choice(cand))
        lam = cfg.contextual_mix
        X[i] = (1 - lam) * X[i] + lam * pool_X[j]
        log.append({"row": int(i), "type": "contextual", "mode": cfg.contextual_strategy,
                    "own_gas": int(gas[i]), "donor_gas": int(pool_gas[j]),
                    "own_conc": float(conc[i]), "donor_conc": float(pool_conc[j])})


def _choose_windows(n, n_windows, L, rng, forbidden=None):
    """Pick non-overlapping, non-adjacent windows (start indices) when possible."""
    n_blocks = n // L
    blocks = np.arange(n_blocks)
    if forbidden is not None:
        ok = [b for b in blocks if not forbidden[b * L:(b + 1) * L].any()]
        blocks = np.array(ok, dtype=int)
    even = blocks[blocks % 2 == 0]
    pool = even if len(even) >= n_windows else blocks
    n_windows = min(n_windows, len(pool))
    chosen = rng.choice(pool, size=n_windows, replace=False)
    return np.sort(chosen) * L


def _inject_collective(X, starts, L, rng, cfg, sd, log):
    d = X.shape[1]
    k = cfg.k_features(d)
    for s in starts:
        feats = rng.choice(d, size=k, replace=False)
        sign = rng.choice([-1.0, 1.0], size=k)
        shift = sign * cfg.collective_strength * sd[feats]
        X[s:s + L][:, feats] += shift          # identical shift for the entire window
        log.append({"row": int(s), "type": "collective", "mode": "window_shift",
                    "window_length": L, "features": feats.tolist(),
                    "strength_sigma": cfg.collective_strength})


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def inject_anomalies(X_clean: np.ndarray, cfg: InjectionConfig,
                     gas: np.ndarray | None = None, conc: np.ndarray | None = None,
                     pool_X: np.ndarray | None = None, pool_gas: np.ndarray | None = None,
                     pool_conc: np.ndarray | None = None) -> InjectedData:
    """Inject anomalies into a COPY of X_clean and return labels + trace."""
    if cfg.anomaly_type not in ANOMALY_TYPES:
        raise ValueError(f"anomaly_type must be one of {ANOMALY_TYPES}")
    rng = np.random.default_rng([cfg.seed, 303])
    X = np.array(X_clean, dtype=float, copy=True)
    n, d = X.shape
    mu, sd = clean_stats(X_clean)
    y = np.zeros(n, dtype=int)
    types = np.array(["normal"] * n, dtype=object)
    log: list[dict] = []
    n_anom = int(round(cfg.ratio * n))
    L = cfg.window_length

    shares = {cfg.anomaly_type: n_anom} if cfg.anomaly_type != "mixed" else {
        "collective": n_anom // 3, "contextual": n_anom // 3, "point": n_anom - 2 * (n_anom // 3)}

    taken = np.zeros(n, dtype=bool)
    if shares.get("collective", 0) > 0:        # windows first: they need contiguous free blocks
        n_win = max(1, int(round(shares["collective"] / L)))
        starts = _choose_windows(n, n_win, L, rng)
        _inject_collective(X, starts, L, rng, cfg, sd, log)
        for s in starts:
            taken[s:s + L] = True
            types[s:s + L] = "collective"
    for t in ("contextual", "point"):
        m = shares.get(t, 0)
        if m <= 0:
            continue
        free = np.flatnonzero(~taken)
        idx = rng.choice(free, size=min(m, len(free)), replace=False)
        if t == "point":
            _inject_point(X, idx, rng, cfg, mu, sd, log)
        else:
            if gas is None or conc is None:
                raise ValueError("contextual anomalies need gas and conc arrays")
            _inject_contextual(X, idx, rng, cfg, np.asarray(gas), np.asarray(conc),
                               pool_X, None if pool_gas is None else np.asarray(pool_gas),
                               None if pool_conc is None else np.asarray(pool_conc), log)
        taken[idx] = True
        types[idx] = t

    y[taken] = 1
    return InjectedData(X=X, y=y, anomaly_type=types, log=pd.DataFrame(log), config=cfg)


def inject_into_dataframe(df: pd.DataFrame, feature_cols: list[str], cfg: InjectionConfig,
                          pool_df: pd.DataFrame | None = None,
                          gas_col: str = "Class", conc_col: str = "Concentration") -> pd.DataFrame:
    """DataFrame convenience wrapper. Returns a NEW frame with anomaly_label / anomaly_type."""
    gas = df[gas_col].to_numpy() if gas_col in df else None
    conc = df[conc_col].to_numpy() if conc_col in df else None
    inj = inject_anomalies(
        df[feature_cols].to_numpy(float), cfg, gas, conc,
        None if pool_df is None else pool_df[feature_cols].to_numpy(float),
        None if pool_df is None else pool_df[gas_col].to_numpy(),
        None if pool_df is None else pool_df[conc_col].to_numpy())
    out = df.copy()
    out[feature_cols] = inj.X
    out["anomaly_label"] = inj.y
    out["anomaly_type"] = inj.anomaly_type
    return out


def add_sensor_noise(X: np.ndarray, level: float, sigma: np.ndarray, seed: int = 0) -> np.ndarray:
    """Additive Gaussian measurement noise: N(0, (level * sigma_j)^2) per feature.
    level = 0.05 means noise std equals 5 % of each feature's clean std."""
    if level <= 0:
        return np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng([seed, 404])
    return X + rng.normal(size=X.shape) * (level * sigma)


def config_to_dict(cfg: InjectionConfig) -> dict:
    return asdict(cfg)
