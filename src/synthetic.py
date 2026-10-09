"""
synthetic.py : STAND-IN data for code testing and offline demos ONLY.

It mimics the *schema and rough structure* of the UCI gas-sensor table (6 gases,
5 concentration levels, 10 batches, 16 sensors x 8 derived features, batch drift,
correlated EMA features). It is NOT the UCI data; NO scientific result may be reported
from it. Real experiments must use data/processed/gas_sensor_drift_parsed.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import EXPECTED_BATCH_COUNTS, GAS_NAMES, build_feature_names


def make_standin(n_total: int | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = dict(EXPECTED_BATCH_COUNTS)
    if n_total is not None:
        scale = n_total / sum(counts.values())
        counts = {b: max(20, int(c * scale)) for b, c in counts.items()}
    gas_sig = rng.normal(size=(6, 16))                    # gas-specific sensor signature
    sens_gain = rng.uniform(0.5, 2.0, 16)
    rows = []
    for b, n in counts.items():
        drift = 1 + 0.04 * b * rng.normal(size=16)         # slow multiplicative drift per batch
        g = rng.integers(1, 7, n)
        c = rng.choice([10, 50, 100, 500, 1000], n)
        base = gas_sig[g - 1] * (c[:, None] ** 0.4) * sens_gain * drift
        dr = base + rng.normal(scale=0.3, size=(n, 16)) * (c[:, None] ** 0.2)
        feats = np.empty((n, 128))
        for s in range(16):
            x = dr[:, s]
            cols = [x, np.abs(x), 0.8 * x + rng.normal(0, 0.2, n), 0.5 * x + rng.normal(0, 0.2, n),
                    0.3 * x + rng.normal(0, 0.2, n), -0.8 * x + rng.normal(0, 0.2, n),
                    -0.5 * x + rng.normal(0, 0.2, n), -0.3 * x + rng.normal(0, 0.2, n)]
            feats[:, s * 8:(s + 1) * 8] = np.column_stack(cols)
        d = pd.DataFrame(feats, columns=build_feature_names())
        d.insert(0, "Concentration", c.astype(float))
        d.insert(0, "Gas", [GAS_NAMES[i] for i in g])
        d.insert(0, "Class", g)
        d.insert(0, "Batch", b)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)
