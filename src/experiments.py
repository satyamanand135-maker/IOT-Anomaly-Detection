"""
experiments.py
==============
Configurable experiment framework.

PIPELINE FOR ONE CONDITION (prepare_dataset)
--------------------------------------------
 1. sample n rows of the validated UCI table (kept in original file order = "stream order");
    unsampled rows form a donor pool for contextual anomalies
 2. select `n_dims` features (nested, sensor-aware; or PCA)
 3. inject anomalies (ground truth)        -> src.anomaly_generation
 4. add measurement noise (level x sigma_j) -> after injection: noise corrupts everything a
                                              sensor reports, anomalies included
 5. simulate sensor dropout (NaN)            -> after injection: values are lost in transmission
 6. simulate irregular sampling (rows lost)  -> SIMULATED, source has no raw timestamps
 7. impute / interpolate (mean|median|knn | ffill|linear|pchip|nearest)
 8. scale (standard|robust|minmax), fitted on the unlabeled experimental data
 9. optional PCA to n_dims; optional context block for contextual anomalies
10. run detectors -> scores -> predictions -> metrics (on observed, non-interpolated rows)

NOTE ON ORDER: the project brief lists dropout/resampling before injection. We inject first
and degrade afterwards because anomalies are properties of what the sensor *measures*, while
dropout/irregularity are properties of *transmission*; this lets dropout actually destroy
anomalous readings, which is exactly what RQ5/RQ6 ask about. Documented deviation.

CLI
---
    python -m src.experiments --quick            # fast development run
    python -m src.experiments --full             # final experiments (long)
    python -m src.experiments --quick --standin  # code test on STAND-IN data (NOT UCI)
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .anomaly_generation import (InjectionConfig, add_sensor_noise, clean_stats,
                                 inject_anomalies)
from .evaluation import evaluate_scores
from .preprocessing import (STREAM_FILL_METHODS, Preprocessor, encode_context,
                            feature_columns, fill_missing_stream, pca_reduce,
                            select_dimensions, simulate_dropout,
                            simulate_irregular_sampling)
from .registry import ALGORITHMS, FAMILY, make_detector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
PARSED_CSV = PROJECT_ROOT / "data" / "processed" / "gas_sensor_drift_parsed.csv"

QUICK_MODE = True  # default when imported; the CLI overrides it


@dataclass(frozen=True)
class Condition:
    anomaly_type: str = "point"
    n_dims: int = 64
    ratio: float = 0.05
    noise: float = 0.0
    dropout: float = 0.0
    dropout_strategy: str = "random"
    imputer: str = "median"
    scaler: str = "standard"
    irregular_drop: float = 0.0
    irregular_mode: str = "random"
    dim_strategy: str = "feature_per_sensor"
    context_mode: str = "auto"          # auto | none | augment
    strength: float = 3.0
    n_samples: int = 2000
    seed: int = 0
    tag: str = "main"
    alg_params: str = ""                # JSON: {"KNN": {"k": 5}}

    def cid(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


COND_FIELDS = [f.name for f in fields(Condition)]
EXTRA_COLS = ["cov_cond", "eps", "noise_fraction", "n_clusters", "degenerate",
              "eif_backend", "error"]
METRIC_COLS = ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "fpr", "fnr",
               "f1_mad", "precision_mad", "recall_mad", "fpr_mad", "tn", "fp", "fn", "tp"]
RESULT_COLUMNS = (["cid"] + COND_FIELDS + ["algorithm", "family"] + METRIC_COLS +
                  ["n_eval", "n_anom", "fit_time_s"] + EXTRA_COLS)


# --------------------------------------------------------------------------- #
# One condition -> analysis-ready matrix
# --------------------------------------------------------------------------- #
def prepare_dataset(df_all: pd.DataFrame, cond: Condition) -> dict:
    all_cols = feature_columns(df_all)
    rng = np.random.default_rng([cond.seed, 1])
    N = len(df_all)
    n = min(cond.n_samples, N - 1)
    idx = np.sort(rng.choice(N, size=n, replace=False))
    pool_mask = np.ones(N, dtype=bool)
    pool_mask[idx] = False
    sample, pool = df_all.iloc[idx], df_all[pool_mask]

    cols = all_cols if cond.dim_strategy == "pca" else select_dimensions(
        all_cols, cond.n_dims, cond.dim_strategy, X_ref=sample)
    X_clean = sample[cols].to_numpy(float)
    gas, conc = sample["Class"].to_numpy(), sample["Concentration"].to_numpy()

    needs_pool = cond.anomaly_type in ("contextual", "mixed")
    cfg = InjectionConfig(anomaly_type=cond.anomaly_type, ratio=cond.ratio,
                          strength=cond.strength, seed=cond.seed)
    inj = inject_anomalies(
        X_clean, cfg, gas, conc,
        pool[cols].to_numpy(float) if needs_pool else None,
        pool["Class"].to_numpy() if needs_pool else None,
        pool["Concentration"].to_numpy() if needs_pool else None)

    _, sd = clean_stats(X_clean)
    X = add_sensor_noise(inj.X, cond.noise, sd, cond.seed)
    X, _ = simulate_dropout(X, cols, cond.dropout, cond.dropout_strategy, cond.seed)
    keep = simulate_irregular_sampling(n, cond.irregular_drop, cond.seed, cond.irregular_mode)
    if (~keep).any():
        X[~keep] = np.nan

    if cond.imputer in STREAM_FILL_METHODS:
        if np.isnan(X).any():
            X = fill_missing_stream(X, cond.imputer)
        Z = Preprocessor("none", cond.scaler).fit_transform(X)
    else:
        Z = Preprocessor(cond.imputer, cond.scaler).fit_transform(X)

    if cond.dim_strategy == "pca":
        Z, _ = pca_reduce(Z, Z, cond.n_dims, cond.seed)

    use_ctx = cond.context_mode == "augment" or (
        cond.context_mode == "auto" and cond.anomaly_type in ("contextual", "mixed"))
    if use_ctx:
        Z = np.hstack([Z, encode_context(gas, conc)])

    return {"X": Z, "y": inj.y, "types": inj.anomaly_type, "eval_mask": keep,
            "columns": cols, "context_used": use_ctx, "log": inj.log,
            "X_injected_raw": inj.X, "gas": gas, "conc": conc}


# --------------------------------------------------------------------------- #
# Run detectors on one condition
# --------------------------------------------------------------------------- #
def run_condition(df_all: pd.DataFrame, cond: Condition, algorithms: list[str],
                  keep_scores: bool = False):
    warnings.filterwarnings("ignore")
    data = prepare_dataset(df_all, cond)
    X, y, ev = data["X"], data["y"], data["eval_mask"]
    params = json.loads(cond.alg_params) if cond.alg_params else {}
    rows, scores_out = [], {}
    for name in algorithms:
        row = {"cid": cond.cid(), **asdict(cond), "algorithm": name, "family": FAMILY[name],
               "n_eval": int(ev.sum()), "n_anom": int(y[ev].sum())}
        t0 = time.perf_counter()
        try:
            det = make_detector(name, params.get(name), seed=cond.seed)
            s = det.fit_score(X)
            row.update(evaluate_scores(y[ev], s[ev], cond.ratio))
            ex = det.extras
            row.update({"cov_cond": ex.get("cov_cond"), "eps": ex.get("eps"),
                        "noise_fraction": ex.get("noise_fraction"),
                        "n_clusters": ex.get("n_clusters"), "degenerate": ex.get("degenerate"),
                        "eif_backend": ex.get("backend")})
            if keep_scores:
                scores_out[name] = s
        except Exception as exc:  # never lose a whole sweep to one failing detector
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        row["fit_time_s"] = time.perf_counter() - t0
        rows.append(row)
    return (rows, scores_out, data) if keep_scores else rows


# --------------------------------------------------------------------------- #
# Experiment designs
# --------------------------------------------------------------------------- #
LEVELS_FULL = dict(anomaly_type=["point", "contextual", "collective"], n_dims=[16, 32, 64, 128],
                   ratio=[0.01, 0.05, 0.10, 0.20], noise=[0.0, 0.05, 0.10, 0.20],
                   dropout=[0.0, 0.10, 0.20])
LEVELS_QUICK = dict(anomaly_type=["point", "contextual", "collective"], n_dims=[16, 64],
                    ratio=[0.05, 0.10], noise=[0.0, 0.10], dropout=[0.0, 0.10])
BASELINE = dict(n_dims=64, ratio=0.05, noise=0.0, dropout=0.0)


def build_main_conditions(quick: bool, design: str, n_samples: int, seeds: list[int],
                          max_conditions: int | None = None) -> list[Condition]:
    L = LEVELS_QUICK if quick else LEVELS_FULL
    combos: list[dict] = []
    if design == "factorial":
        for t, d, r, nz, dr in itertools.product(L["anomaly_type"], L["n_dims"], L["ratio"],
                                                 L["noise"], L["dropout"]):
            combos.append(dict(anomaly_type=t, n_dims=d, ratio=r, noise=nz, dropout=dr))
    elif design == "ofat":          # one-factor-at-a-time around BASELINE
        seen = set()
        for t in L["anomaly_type"]:
            for factor, key in [("n_dims", "n_dims"), ("ratio", "ratio"),
                                ("noise", "noise"), ("dropout", "dropout")]:
                for v in L[factor]:
                    c = dict(BASELINE, anomaly_type=t)
                    c[key] = v
                    sig = tuple(sorted(c.items()))
                    if sig not in seen:
                        seen.add(sig)
                        combos.append(c)
    else:
        raise ValueError("design must be 'ofat' or 'factorial'")
    if max_conditions and len(combos) > max_conditions:
        rng = np.random.default_rng(0)
        combos = [combos[i] for i in sorted(rng.choice(len(combos), max_conditions, replace=False))]
    return [Condition(n_samples=n_samples, seed=s, tag="main", **c) for c in combos for s in seeds]


def build_extra_conditions(quick: bool, n_samples: int, seeds: list[int]) -> list[Condition]:
    """RQ5/RQ6 (dropout x strategy x imputer), preprocessing (scalers), irregular sampling,
    sensitivity (k, eps) and contextual-with/without-context experiments."""
    out: list[Condition] = []
    types = ["point"] if quick else ["point", "contextual", "collective"]
    drops = [0.10] if quick else [0.10, 0.20, 0.30]
    strategies = ["random", "consecutive"] if quick else ["random", "sensor", "consecutive"]
    imputers = ["median", "knn", "linear"] if quick else ["mean", "median", "knn", "ffill", "linear"]
    for s in seeds:
        base = dict(n_samples=n_samples, seed=s)
        for t, dr, st, im in itertools.product(types, drops, strategies, imputers):
            out.append(Condition(tag="imputation", anomaly_type=t, dropout=dr,
                                 dropout_strategy=st, imputer=im, **base))
        for t, sc, dr in itertools.product(types, ["standard", "robust", "minmax"], [0.0, 0.10]):
            out.append(Condition(tag="preproc", anomaly_type=t, scaler=sc, dropout=dr, **base))
        for t, ir, it, mode in itertools.product(["point", "collective"], [0.0, 0.2, 0.4],
                                                 ["ffill", "linear", "pchip"], ["random", "bursty"]):
            if ir == 0.0 and (it != "linear" or mode != "random"):
                continue
            out.append(Condition(tag="irregular", anomaly_type=t, irregular_drop=ir, imputer=it,
                                 irregular_mode=mode, **base))
        for cm in ["none", "augment"]:
            out.append(Condition(tag="context", anomaly_type="contextual", context_mode=cm, **base))
        for dim in ([16, 64] if quick else [16, 32, 64, 128]):
            for k in [3, 5, 10, 20, 50]:
                out.append(Condition(tag="sensitivity", anomaly_type="point", n_dims=dim,
                                     alg_params=json.dumps({"KNN": {"k": k}, "LOF": {"k": k}}), **base))
            for q in [0.80, 0.90, 0.95]:
                out.append(Condition(tag="sensitivity", anomaly_type="point", n_dims=dim,
                                     alg_params=json.dumps({"DBSCAN": {"eps_strategy": "quantile",
                                                                       "quantile": q}}), **base))
    return out


SENSITIVITY_ALGS = ["KNN", "LOF", "DBSCAN"]


# --------------------------------------------------------------------------- #
# Runner (resumable, optionally parallel)
# --------------------------------------------------------------------------- #
def _alg_subset(cond: Condition, algorithms: list[str]) -> list[str]:
    return [a for a in SENSITIVITY_ALGS if a in algorithms] if cond.tag == "sensitivity" else algorithms


def run_experiments(df_all: pd.DataFrame, conditions: list[Condition], algorithms: list[str],
                    out_csv: Path, n_jobs: int = 1, resume: bool = True, verbose: bool = True):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if resume and out_csv.exists():
        old = pd.read_csv(out_csv, usecols=["cid", "algorithm"])
        done = set(old.cid + "|" + old.algorithm)
    todo = []
    for c in conditions:
        algs = [a for a in _alg_subset(c, algorithms) if f"{c.cid()}|{a}" not in done]
        if algs:
            todo.append((c, algs))
    if verbose:
        print(f"[experiments] {len(conditions)} conditions, {len(todo)} to run "
              f"({sum(len(a) for _, a in todo)} detector runs) -> {out_csv}")
    first = not out_csv.exists() or not resume
    t0 = time.time()

    def _write(rows):
        nonlocal first
        pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS).to_csv(
            out_csv, mode="w" if first else "a", header=first, index=False)
        first = False

    if n_jobs == 1:
        for i, (c, algs) in enumerate(todo, 1):
            _write(run_condition(df_all, c, algs))
            if verbose and (i % 5 == 0 or i == len(todo)):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} conditions  elapsed {el:.0f}s  ETA {el / i * (len(todo) - i):.0f}s")
    else:
        from joblib import Parallel, delayed
        gen = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(run_condition)(df_all, c, algs) for c, algs in todo)
        for i, rows in enumerate(gen, 1):
            _write(rows)
            if verbose and (i % 10 == 0 or i == len(todo)):
                print(f"  {i}/{len(todo)} conditions  elapsed {time.time() - t0:.0f}s")
    return pd.read_csv(out_csv)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true", help="QUICK_MODE=True (development)")
    g.add_argument("--full", action="store_true", help="QUICK_MODE=False (final experiments)")
    ap.add_argument("--design", choices=["ofat", "factorial"], default=None,
                    help="default: factorial for --quick, ofat for --full")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of repetitions")
    ap.add_argument("--max-conditions", type=int, default=None)
    ap.add_argument("--no-extras", action="store_true", help="skip imputation/irregular/sensitivity sweeps")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS))
    ap.add_argument("--standin", action="store_true",
                    help="use synthetic STAND-IN data (code test only; results go to results_standin/)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results (no resume)")
    a = ap.parse_args(argv)

    quick = a.quick
    design = a.design or ("factorial" if quick else "ofat")
    n_samples = a.n_samples or (800 if quick else 3000)
    seeds = list(range(a.seeds or (1 if quick else 5)))

    if a.standin:
        from .synthetic import make_standin
        df = make_standin(6000, seed=0)
        out = PROJECT_ROOT / "results_standin" / "experiment_results.csv"
        print("!! STAND-IN synthetic data - NOT the UCI dataset. Do not report these results.")
    else:
        if not PARSED_CSV.exists():
            print(f"Missing {PARSED_CSV}. Run `python -m src.data_loader` first.")
            return 1
        df = pd.read_csv(PARSED_CSV)
        out = RESULTS_DIR / "experiment_results.csv"

    conds = build_main_conditions(quick, design, n_samples, seeds, a.max_conditions)
    if not a.no_extras:
        conds += build_extra_conditions(quick, n_samples, seeds)
    run_experiments(df, conds, a.algorithms, out, n_jobs=a.n_jobs, resume=not a.fresh)
    print("Done. Next: python -m src.analysis" + (" --standin" if a.standin else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
