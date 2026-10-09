"""pipeline.py : single entry point used by the Streamlit dashboard (and usable from notebooks)
to run ONE detector on arbitrary tabular sensor data. No labels are used for fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluation import evaluate_scores, normalize_scores, threshold_mad, threshold_top_k
from .preprocessing import (STREAM_FILL_METHODS, Preprocessor, drop_correlated,
                            fill_missing_stream, pca_reduce)
from .registry import make_detector

IMPUTERS = ["none", "mean", "median", "knn"] + list(STREAM_FILL_METHODS)


def run_detection(df: pd.DataFrame, columns: list[str], algorithm: str, params: dict | None = None,
                  imputer: str = "median", scaler: str = "standard", pca_components: int | None = None,
                  corr_threshold: float | None = None, threshold_mode: str = "top_percent",
                  contamination: float = 0.05, label_col: str | None = None,
                  context_block: np.ndarray | None = None, seed: int = 0) -> dict:
    """Returns dict(result_df, n_anomalies, pct, X_processed, extras, metrics|None)."""
    X_df = df[columns].apply(pd.to_numeric, errors="coerce")
    if corr_threshold:
        columns = drop_correlated(X_df.fillna(X_df.median()), columns, corr_threshold)
        X_df = X_df[columns]
    X = X_df.to_numpy(float)
    if imputer in STREAM_FILL_METHODS:
        X = fill_missing_stream(X, imputer) if np.isnan(X).any() else X
        Z = Preprocessor("none", scaler).fit_transform(X)
    else:
        Z = Preprocessor(imputer, scaler).fit_transform(X)
    if pca_components:
        Z, _ = pca_reduce(Z, Z, min(pca_components, Z.shape[1], Z.shape[0] - 1), seed)
    if context_block is not None:
        Z = np.hstack([Z, context_block])

    det = make_detector(algorithm, params, seed)
    s = det.fit_score(Z)
    native = det.native_predict()
    if threshold_mode == "native" and native is not None:
        pred = native.astype(int)
    elif threshold_mode == "mad" or (threshold_mode == "native" and native is None):
        pred = threshold_mad(s)
    else:
        pred = threshold_top_k(s, int(round(contamination * len(s))))

    res = pd.DataFrame({"anomaly_score": s, "score_norm": normalize_scores(s),
                        "prediction": pred, "label": np.where(pred == 1, "anomaly", "normal")}, index=df.index)
    out = {"result_df": res, "n_anomalies": int(pred.sum()), "pct": float(pred.mean() * 100),
           "X_processed": Z, "extras": det.extras, "columns_used": columns, "metrics": None}
    if label_col and label_col in df:
        y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
        if set(np.unique(y)) <= {0, 1} and 0 < y.sum() < len(y):
            out["metrics"] = evaluate_scores(y, s, y.mean())
            out["y"] = y
    return out
