"""Core unit tests (synthetic stand-in data / toy data). Run: python tests/test_core.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.anomaly_generation import InjectionConfig, inject_anomalies
from src.evaluation import evaluate_scores, normalize_scores, threshold_top_k
from src.pipeline import run_detection
from src.preprocessing import (Preprocessor, feature_columns, select_dimensions,
                               simulate_dropout, simulate_irregular_sampling, fill_missing_stream)
from src.registry import ALGORITHMS, make_detector
from src.synthetic import make_standin
from sklearn.metrics import roc_auc_score

df = make_standin(2500)
cols = feature_columns(df)
X = df[cols].to_numpy()


def test_injection_labels_and_isolation():
    X0 = X.copy()
    for t in ["point", "contextual", "collective", "mixed"]:
        cfg = InjectionConfig(anomaly_type=t, ratio=0.10, seed=3)
        pool = df.sample(800, random_state=1)
        inj = inject_anomalies(X, cfg, df.Class.to_numpy(), df.Concentration.to_numpy(),
                               pool[cols].to_numpy(), pool.Class.to_numpy(), pool.Concentration.to_numpy())
        assert np.array_equal(X, X0), "original data was modified"
        assert abs(inj.y.mean() - 0.10) < 0.02, (t, inj.y.mean())
        assert set(inj.anomaly_type[inj.y == 0]) == {"normal"}
        assert (inj.anomaly_type[inj.y == 1] != "normal").all()
        changed = np.any(inj.X != X, axis=1)
        assert changed[inj.y == 1].mean() > 0.95 and not changed[inj.y == 0].any(), t
        inj2 = inject_anomalies(X, cfg, df.Class.to_numpy(), df.Concentration.to_numpy(),
                                pool[cols].to_numpy(), pool.Class.to_numpy(), pool.Concentration.to_numpy())
        assert np.array_equal(inj.y, inj2.y) and np.array_equal(inj.X, inj2.X), "not reproducible"
    print("PASS injection: labels, untouched original, reproducible (4 types)")


def test_collective_windows_contiguous():
    inj = inject_anomalies(X, InjectionConfig(anomaly_type="collective", ratio=0.1, window_length=10, seed=1))
    runs, cur = [], 0
    for v in inj.y:
        cur = cur + 1 if v else 0
        if not v and cur == 0:
            pass
    idx = np.flatnonzero(inj.y)
    assert len(idx) % 10 == 0
    starts = idx[::10]
    assert all((idx[i:i + 10] == np.arange(s, s + 10)).all() for i, s in zip(range(0, len(idx), 10), starts))
    print("PASS collective: windows are contiguous blocks of 10, fully labelled")


def test_contextual_donor_is_other_gas():
    pool = df.sample(800, random_state=1)
    inj = inject_anomalies(X, InjectionConfig(anomaly_type="contextual", ratio=0.05, seed=2),
                           df.Class.to_numpy(), df.Concentration.to_numpy(), pool[cols].to_numpy(),
                           pool.Class.to_numpy(), pool.Concentration.to_numpy())
    assert (inj.log.own_gas != inj.log.donor_gas).all()
    print("PASS contextual: donor always from a different gas context")


def test_dropout_and_irregular():
    for strat in ["random", "sensor", "consecutive"]:
        Xm, mask = simulate_dropout(X, cols, 0.2, strat, seed=0)
        assert np.isnan(Xm).sum() == mask.sum()
        assert 0.12 < mask.mean() < 0.30, (strat, mask.mean())
    keep = simulate_irregular_sampling(1000, 0.3, 0)
    assert (~keep).sum() == 300
    Xf = fill_missing_stream(np.where(np.random.default_rng(0).random((200, 5)) < .2, np.nan, 1.0), "linear")
    assert np.isfinite(Xf).all()
    print("PASS dropout strategies, irregular sampling, interpolation")


def test_dimension_selection_nested():
    sets = {d: set(select_dimensions(cols, d)) for d in [16, 32, 64, 128]}
    assert sets[16] < sets[32] < sets[64] < sets[128] and len(sets[64]) == 64
    print("PASS dimension subsets are nested 16<32<64<128")


def test_no_leakage_scaler_fit_on_train_only():
    tr, te = X[:1500], X[1500:]
    p = Preprocessor("median", "standard").fit(tr)
    assert np.allclose(p.scaler_.mean_, tr.mean(axis=0))
    assert not np.allclose(p.scaler_.mean_, X.mean(axis=0))
    print("PASS scaler statistics come from the fit split only")


def test_score_orientation_planted_outlier():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(600, 8)); Z[0] = 12.0
    for name in ALGORITHMS:
        s = make_detector(name, seed=0).fit_score(Z)
        assert s.argmax() == 0 or np.argsort(-s).tolist().index(0) < 3, f"{name}: planted outlier not top-ranked"
    print("PASS all 7 detectors rank a planted outlier at the top (higher = more anomalous)")


def test_eif_numpy_discriminates():
    rng = np.random.default_rng(1)
    Z = np.vstack([rng.normal(size=(500, 6)), rng.normal(loc=5, size=(25, 6))])
    y = np.r_[np.zeros(500), np.ones(25)]
    for be in ["numpy"]:
        det = make_detector("ExtendedIF", {"backend": be}, 0)
        auc = roc_auc_score(y, det.fit_score(Z))
        assert auc > 0.95, auc
    print(f"PASS ExtendedIF (numpy backend) AUC={auc:.3f} on a toy cluster/outlier set")


def test_evaluation_helpers():
    y = np.r_[np.zeros(95), np.ones(5)]
    s = np.r_[np.zeros(95), np.ones(5)] + 0.0
    m = evaluate_scores(y, s, 0.05)
    assert m["roc_auc"] == 1 and m["f1"] == 1 and m["fpr"] == 0 and m["fnr"] == 0
    assert threshold_top_k(np.arange(10.0), 3).sum() == 3
    n = normalize_scores(np.array([5., 1., 3.]))
    assert n.argmax() == 0 and n.min() == 0 and n.max() == 1
    print("PASS metrics, top-k thresholding, rank normalisation")


def test_pipeline_end_to_end():
    d = df.copy()
    inj = inject_anomalies(d[cols].to_numpy(), InjectionConfig(anomaly_type="point", ratio=0.05, seed=0))
    d[cols] = inj.X; d["anomaly_label"] = inj.y
    for alg in ALGORITHMS:
        out = run_detection(d, cols[:64], alg, imputer="knn" if alg == "KNN" else "median", label_col="anomaly_label",
                            contamination=0.05)
        assert out["metrics"] is not None and len(out["result_df"]) == len(d)
    print("PASS pipeline: 7 detectors end-to-end with labels")


if __name__ == "__main__":
    for f in [test_injection_labels_and_isolation, test_collective_windows_contiguous, test_contextual_donor_is_other_gas,
              test_dropout_and_irregular, test_dimension_selection_nested, test_no_leakage_scaler_fit_on_train_only,
              test_score_orientation_planted_outlier, test_eif_numpy_discriminates, test_evaluation_helpers,
              test_pipeline_end_to_end]:
        f()
    print("\nAll core tests passed.")
