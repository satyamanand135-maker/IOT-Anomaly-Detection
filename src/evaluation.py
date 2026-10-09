"""
evaluation.py
=============
Standardised score handling + metrics.

SCORE CONVENTION
----------------
Every detector returns a raw `anomaly_score` where higher = more anomalous
(inversions are documented per detector, e.g. IsolationForest: -score_samples,
LOF: -negative_outlier_factor_). For cross-algorithm plots a rank-normalised copy in
[0,1] is also produced: score_norm = (rank - 1)/(n - 1), ties averaged. ROC-AUC and PR-AUC
are rank-based, hence unaffected by this monotone transform.

SCORE -> PREDICTION (two rules, both reported)
-----------------------------------------------
top_k : flag the k = round(ratio * n) highest scores. Uses the TRUE anomaly ratio (an
        "oracle contamination" assumption, standard for comparing rankings). With k equal to
        the true count FP == FN, so precision == recall == F1 for this rule.
mad   : data-driven, no oracle: flag if robust z = (s - median) / (1.4826 * MAD) > 3.5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


def clean_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float).copy()
    finite = np.isfinite(s)
    if not finite.all():
        hi = s[finite].max() if finite.any() else 0.0
        lo = s[finite].min() if finite.any() else 0.0
        s[np.isposinf(s)] = hi
        s[np.isneginf(s)] = lo
        s[np.isnan(s)] = lo
    return s


def normalize_scores(s: np.ndarray, method: str = "rank") -> np.ndarray:
    s = clean_scores(s)
    if len(s) < 2:
        return np.zeros_like(s)
    if method == "rank":
        return (rankdata(s, method="average") - 1) / (len(s) - 1)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)


def threshold_top_k(scores: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Top-k flags; ties broken by a fixed random permutation (no index-order bias)."""
    s = clean_scores(scores)
    k = int(max(0, min(k, len(s))))
    jitter = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((jitter, -s))          # primary: -score, secondary: random
    pred = np.zeros(len(s), dtype=int)
    pred[order[:k]] = 1
    return pred


def threshold_mad(scores: np.ndarray, z: float = 3.5) -> np.ndarray:
    s = clean_scores(scores)
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    if mad == 0:
        mad = np.std(s) if np.std(s) > 0 else 1.0
    return ((s - med) / mad > z).astype(int)


def classification_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_scores(y: np.ndarray, scores: np.ndarray, ratio: float | None = None) -> dict:
    """All metrics for one detector on one labelled set."""
    y = np.asarray(y).astype(int)
    s = clean_scores(scores)
    out: dict = {}
    two_classes = len(np.unique(y)) == 2
    out["roc_auc"] = roc_auc_score(y, s) if two_classes else np.nan
    out["pr_auc"] = average_precision_score(y, s) if two_classes else np.nan
    k = int(round((ratio if ratio is not None else y.mean()) * len(y)))
    out.update(classification_metrics(y, threshold_top_k(s, k)))
    m = classification_metrics(y, threshold_mad(s))
    out.update({"f1_mad": m["f1"], "precision_mad": m["precision"],
                "recall_mad": m["recall"], "fpr_mad": m["fpr"]})
    return out


@dataclass
class DetectionResult:
    """Standard output: algorithm, anomaly_score, prediction, ground_truth."""
    algorithm: str
    anomaly_score: np.ndarray
    prediction: np.ndarray
    ground_truth: np.ndarray | None = None
    score_norm: np.ndarray | None = None
    extras: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        d = {"algorithm": self.algorithm, "anomaly_score": self.anomaly_score,
             "score_norm": self.score_norm, "prediction": self.prediction}
        if self.ground_truth is not None:
            d["ground_truth"] = self.ground_truth
        return pd.DataFrame(d)


def make_result(name: str, scores: np.ndarray, y: np.ndarray | None = None,
                ratio: float | None = None, n_flag: int | None = None, extras: dict | None = None
                ) -> DetectionResult:
    s = clean_scores(scores)
    if n_flag is None:
        n_flag = int(round((ratio if ratio is not None else (y.mean() if y is not None else 0.05)) * len(s)))
    return DetectionResult(name, s, threshold_top_k(s, n_flag), y, normalize_scores(s), extras or {})


def roc_points(y, scores):
    return roc_curve(y, clean_scores(scores))


def pr_points(y, scores):
    return precision_recall_curve(y, clean_scores(scores))
