"""
analysis.py : turns experiment_results.csv into tables, figures and the PRACTICAL
METHOD-SELECTION GUIDE. Every number written is computed from the results file;
nothing is hard-coded. Re-run after any new experiments.

    python -m src.analysis            # results/experiment_results.csv
    python -m src.analysis --standin  # results_standin/ (code test only)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

from . import visualization as viz
from .experiments import Condition, prepare_dataset, run_condition
from .registry import ALGORITHMS

ROOT = Path(__file__).resolve().parents[1]
ALGS = list(ALGORITHMS)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def summarize(df: pd.DataFrame, by: list[str], metrics=("roc_auc", "pr_auc", "f1", "f1_mad", "fpr", "fnr")):
    agg = df.groupby(by + ["algorithm"])[list(metrics)].agg(["mean", "std", "count"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def best_in_subset(sub: pd.DataFrame, metric="roc_auc", top=3):
    """Rank algorithms in a subset; test #1 vs #2 with a paired Wilcoxon over matched conditions."""
    if sub.empty:
        return None
    piv = sub.pivot_table(index="cid", columns="algorithm", values=metric, aggfunc="mean")
    mean = piv.mean().sort_values(ascending=False)
    sd = piv.std()
    ranking = [(a, mean[a], sd[a], sub[sub.algorithm == a]["pr_auc"].mean()) for a in mean.index[:top]]
    note = "single algorithm only"
    if len(mean) >= 2:
        a, b = mean.index[0], mean.index[1]
        diff = (piv[a] - piv[b]).dropna()
        if len(diff) >= 6 and (diff != 0).any():
            try:
                p = stats.wilcoxon(diff).pvalue
                note = (f"{a} vs {b}: mean {metric} diff {diff.mean():+.3f} (Wilcoxon p={p:.3g}, n={len(diff)}) -> "
                        + ("significant" if p < 0.05 else "NOT significant: treat as tie"))
            except ValueError:
                note = f"{a} vs {b}: diff {diff.mean():+.3f} (test not computable)"
        else:
            note = f"{a} vs {b}: diff {(mean[a] - mean[b]):+.3f} (only {len(diff)} paired conditions: no test)"
    return ranking, note, len(piv)


def win_rates(df: pd.DataFrame, metric="roc_auc") -> pd.DataFrame:
    piv = df.pivot_table(index="cid", columns="algorithm", values=metric, aggfunc="mean").dropna()
    winners = piv.idxmax(axis=1).value_counts(normalize=True)
    ranks = piv.rank(axis=1, ascending=False).mean()
    out = pd.DataFrame({"win_rate": winners, "mean_rank": ranks}).fillna({"win_rate": 0.0})
    return out.sort_values("mean_rank")


def friedman(df: pd.DataFrame, metric="roc_auc"):
    piv = df.pivot_table(index="cid", columns="algorithm", values=metric, aggfunc="mean").dropna()
    if len(piv) < 3 or piv.shape[1] < 3:
        return None
    stat, p = stats.friedmanchisquare(*[piv[c] for c in piv.columns])
    return {"statistic": float(stat), "p_value": float(p), "n_blocks": int(len(piv)), "k_algorithms": int(piv.shape[1])}


# --------------------------------------------------------------------------- #
# Practical method-selection guide (data-driven)
# --------------------------------------------------------------------------- #
def build_guide(main: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    dims = sorted(main.n_dims.unique())
    lo_d, hi_d = dims[0], dims[-1]
    noises = sorted(main.noise.unique())
    ratios = sorted(main.ratio.unique())
    rules = [
        (f"Low dimensional (n_dims={lo_d}), no noise", (main.n_dims == lo_d) & (main.noise == 0) & (main.dropout == 0)),
        (f"High dimensional (n_dims={hi_d}), no noise", (main.n_dims == hi_d) & (main.noise == 0) & (main.dropout == 0)),
        (f"High dimensional (n_dims={hi_d}) + high noise (>= {noises[-1]:g} sigma)",
         (main.n_dims == hi_d) & (main.noise == noises[-1]) & (main.dropout == 0)),
        (f"Low dimensional (n_dims={lo_d}) + high noise (>= {noises[-1]:g} sigma)",
         (main.n_dims == lo_d) & (main.noise == noises[-1]) & (main.dropout == 0)),
        ("Point anomalies", main.anomaly_type == "point"),
        ("Contextual anomalies (context provided)", main.anomaly_type == "contextual"),
        ("Collective anomalies", main.anomaly_type == "collective"),
        (f"Low anomaly ratio (<= {ratios[0]:g})", main.ratio == ratios[0]),
        (f"High anomaly ratio (>= {ratios[-1]:g})", main.ratio == ratios[-1]),
        ("Complete data (no dropout)", main.dropout == 0),
        ("Sensor dropout present (median-imputed)", main.dropout > 0),
    ]
    if extra is not None and len(extra):
        imp = extra[extra.tag == "imputation"]
        if len(imp):
            rules.append(("Dropout + KNN imputation", None))
            rules.append(("Dropout + median imputation", None))
    rows = []
    for name, mask in rules:
        if mask is None:
            imp_name = "knn" if "KNN imputation" in name else "median"
            sub = extra[(extra.tag == "imputation") & (extra.imputer == imp_name)]
        else:
            sub = main[mask]
        res = best_in_subset(sub)
        if res is None:
            rows.append({"condition": name, "recommended": "no runs", "mean_roc_auc": np.nan, "reason (measured)": "",
                         "limitations": ""})
            continue
        ranking, note, nblocks = res
        rec = ", ".join(f"{a} ({m:.3f})" for a, m, _, _ in ranking)
        worst = sub.groupby("algorithm")["roc_auc"].mean().sort_values().index[0]
        deg = sub[(sub.algorithm == "DBSCAN")]["degenerate"].astype(float).mean() if "degenerate" in sub else np.nan
        lim = [f"{nblocks} runs; fixed default hyper-parameters; injected (not natural) anomalies; weakest here: {worst}"]
        if not np.isnan(deg) and deg > 0.25:
            lim.append(f"DBSCAN degenerate in {deg:.0%} of runs")
        rows.append({"condition": name, "recommended": rec, "mean_roc_auc": ranking[0][1],
                     "reason (measured)": note, "limitations": "; ".join(lim)})
    return pd.DataFrame(rows)


def guide_to_markdown(g: pd.DataFrame) -> str:
    out = ["| Condition | Recommended methods (mean ROC-AUC) | Evidence | Limitations |", "|---|---|---|---|"]
    for _, r in g.iterrows():
        out.append(f"| {r['condition']} | {r['recommended']} | {r['reason (measured)']} | {r['limitations']} |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Curse-of-dimensionality diagnostics (computed on clean, scaled data)
# --------------------------------------------------------------------------- #
def dimensionality_diagnostics(df_all: pd.DataFrame, dims=(16, 32, 64, 128), n=600, seeds=(0, 1, 2)):
    rows = []
    for d in dims:
        for s in seeds:
            c = Condition(anomaly_type="point", ratio=0.0, n_dims=d, n_samples=n, seed=s, tag="diag")
            X = prepare_dataset(df_all, c)["X"]
            D = pdist(X)
            dm = D.mean()
            # relative contrast per point: (farthest - nearest)/nearest  (Beyer et al. 1999)
            from scipy.spatial.distance import squareform
            S = squareform(D); np.fill_diagonal(S, np.nan)
            rc = np.nanmean((np.nanmax(S, 1) - np.nanmin(S, 1)) / np.nanmin(S, 1))
            ev = PCA().fit(X).explained_variance_ratio_
            rows.append({"n_dims": d, "seed": s, "relative_contrast": rc, "cv_of_distances": D.std() / dm,
                         "pca_components_95pct_var": int(np.searchsorted(np.cumsum(ev), 0.95) + 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_analysis(results_csv: Path, out_dir: Path, df_all: pd.DataFrame | None = None, standin=False):
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(results_csv)
    R = R[R.error.isna()] if "error" in R else R
    main = R[R.tag == "main"]
    log: list[str] = []
    note = "STAND-IN DATA - code test only, NOT UCI results\n\n" if standin else ""

    # ---- overall + RQ7
    summarize(main, ["anomaly_type"]).to_csv(tab_dir / "summary_by_anomaly_type.csv", index=False)
    wr = win_rates(main)
    wr.to_csv(tab_dir / "rq7_win_rates_mean_ranks.csv")
    fr = friedman(main)
    (tab_dir / "rq7_friedman.json").write_text(json.dumps(fr, indent=2))
    for by in ["n_dims", "noise", "ratio", "dropout"]:
        summarize(main, [by]).to_csv(tab_dir / f"summary_by_{by}.csv", index=False)
    viz.plot_metric_bars(main, "roc_auc", "anomaly_type", fig_dir / "bar_roc_auc_by_anomaly_type.png")
    viz.plot_metric_bars(main, "pr_auc", "anomaly_type", fig_dir / "bar_pr_auc_by_anomaly_type.png")

    # ---- performance vs factor (RQ1, RQ2, RQ4, RQ5) + heatmaps
    for factor, fname in [("n_dims", "dimensionality"), ("noise", "noise"), ("ratio", "anomaly_ratio"),
                          ("dropout", "sensor_dropout")]:
        if main[factor].nunique() > 1:
            sub = main
            viz.plot_performance_vs(sub, factor, "roc_auc", col="anomaly_type", path=fig_dir / f"perf_vs_{fname}_roc_auc.png")
            viz.plot_performance_vs(sub, factor, "pr_auc", col="anomaly_type", path=fig_dir / f"perf_vs_{fname}_pr_auc.png")
            _, tab = viz.plot_algorithm_heatmap(sub, factor, "roc_auc", fig_dir / f"heatmap_algorithm_x_{fname}.png")
            tab.to_csv(tab_dir / f"heatmap_algorithm_x_{fname}.csv")
    # robustness to noise / dropout: relative drop (RQ2, RQ5)
    for factor in ["noise", "dropout"]:
        lv = sorted(main[factor].unique())
        if len(lv) > 1:
            t = main.pivot_table(index="algorithm", columns=factor, values="roc_auc", aggfunc="mean")
            drop = ((t[lv[0]] - t[lv[-1]]) / t[lv[0]]).rename(f"relative_roc_auc_drop_{lv[0]:g}_to_{lv[-1]:g}")
            drop.sort_values().to_frame().to_csv(tab_dir / f"robustness_{factor}.csv")
            log.append(f"robustness_{factor}.csv written")

    # ---- imputation / preproc / irregular / context / sensitivity
    extra = R[R.tag != "main"]
    imp = R[R.tag == "imputation"]
    if len(imp):
        t = imp.pivot_table(index="algorithm", columns=["dropout_strategy", "imputer"], values="roc_auc", aggfunc="mean")
        t.to_csv(tab_dir / "rq6_imputation_roc_auc.csv")
        viz.plot_performance_vs(imp.assign(imputer=imp.imputer.astype(str)), "imputer", "roc_auc", col="dropout_strategy",
                                path=fig_dir / "rq6_imputation_effect.png")
        viz.plot_algorithm_heatmap(imp, "imputer", "roc_auc", fig_dir / "rq6_heatmap_algorithm_x_imputer.png")
    pre = R[R.tag == "preproc"]
    if len(pre):
        viz.plot_algorithm_heatmap(pre, "scaler", "roc_auc", fig_dir / "heatmap_algorithm_x_scaler.png")[1].to_csv(
            tab_dir / "preprocessing_scaler_roc_auc.csv")
    irr = R[R.tag == "irregular"]
    if len(irr):
        t = irr.pivot_table(index="algorithm", columns=["irregular_drop", "imputer"], values="roc_auc", aggfunc="mean")
        t.to_csv(tab_dir / "irregular_sampling_roc_auc.csv")
        viz.plot_performance_vs(irr, "irregular_drop", "roc_auc", col="imputer", path=fig_dir / "irregular_sampling_effect.png")
    ctx = R[R.tag == "context"]
    if len(ctx):
        viz.plot_algorithm_heatmap(ctx, "context_mode", "roc_auc", fig_dir / "contextual_with_vs_without_context.png")[1].to_csv(
            tab_dir / "contextual_context_mode_roc_auc.csv")
    sens = R[R.tag == "sensitivity"].copy()
    if len(sens):
        sens["k"] = sens.alg_params.apply(lambda s: (json.loads(s).get("KNN", {}) or {}).get("k", np.nan) if isinstance(s, str) else np.nan)
        sens["dbscan_quantile"] = sens.alg_params.apply(lambda s: (json.loads(s).get("DBSCAN", {}) or {}).get("quantile", np.nan) if isinstance(s, str) else np.nan)
        ks = sens[sens.algorithm.isin(["KNN", "LOF"]) & sens.k.notna()]
        if len(ks):
            viz.plot_performance_vs(ks, "k", "roc_auc", col="n_dims", path=fig_dir / "sensitivity_k_knn_lof.png",
                                    title="KNN / LOF: ROC-AUC vs k (columns = n_dims)")
            ks.pivot_table(index=["algorithm", "n_dims"], columns="k", values="roc_auc").to_csv(tab_dir / "sensitivity_k.csv")
        db = sens[(sens.algorithm == "DBSCAN") & sens.dbscan_quantile.notna()]
        if len(db):
            db.groupby(["n_dims", "dbscan_quantile"])[["roc_auc", "noise_fraction", "n_clusters"]].mean().to_csv(
                tab_dir / "sensitivity_dbscan_eps.csv")

    # ---- Mahalanobis / DBSCAN / timing diagnostics (RQ1)
    diag = main.groupby(["n_dims", "algorithm"])[["fit_time_s"]].mean().reset_index()
    diag.to_csv(tab_dir / "runtime_by_dims.csv", index=False)
    mh = main[main.algorithm == "Mahalanobis"].groupby("n_dims")["cov_cond"].median()
    mh.to_frame("median_covariance_condition_number").to_csv(tab_dir / "mahalanobis_condition_number.csv")
    db = main[main.algorithm == "DBSCAN"].groupby("n_dims")[["degenerate", "noise_fraction", "n_clusters"]].mean()
    db.to_csv(tab_dir / "dbscan_degeneracy_by_dims.csv")
    if df_all is not None:
        dd = dimensionality_diagnostics(df_all, dims=sorted(main.n_dims.unique()))
        dd.groupby("n_dims").mean(numeric_only=True).drop(columns="seed").to_csv(tab_dir / "curse_of_dimensionality_diagnostics.csv")
        log.append("curse_of_dimensionality_diagnostics.csv written")

    # ---- guide
    guide = build_guide(main, extra)
    guide.to_csv(tab_dir / "method_selection_guide.csv", index=False)
    md = note + "# Practical method-selection guide (generated from measured results)\n\n" + guide_to_markdown(guide)
    md += "\n\n## RQ7: is there a consistent best algorithm?\n\n" + wr.round(3).to_markdown() + "\n"
    if fr:
        md += (f"\nFriedman test over {fr['n_blocks']} matched conditions and {fr['k_algorithms']} algorithms: "
               f"chi2={fr['statistic']:.2f}, p={fr['p_value']:.3g}.\n")
    top = wr.win_rate.max()
    md += (f"\nLargest win-rate by ROC-AUC: {wr.win_rate.idxmax()} = {top:.0%} of conditions. "
           + ("Different algorithms win in different conditions."
              if top < 0.999 else "One algorithm wins in every condition (check coverage).") + "\n")
    (out_dir / "results_summary.md").write_text(md)
    return {"guide": guide, "win_rates": wr, "friedman": fr, "log": log}


def representative_figures(df_all: pd.DataFrame, fig_dir: Path, n_samples=1500, seed=0):
    """ROC / PR / confusion matrices / PCA-t-SNE-UMAP / parallel coordinates for one baseline
    condition per anomaly type (+ one noisy variant)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    for t in ["point", "contextual", "collective"]:
        c = Condition(anomaly_type=t, n_dims=64, ratio=0.05, noise=0.05, n_samples=n_samples, seed=seed, tag="rep")
        rows, scores, data = run_condition(df_all, c, ALGS, keep_scores=True)
        ev = data["eval_mask"]
        y = data["y"][ev]
        sc = {k: v[ev] for k, v in scores.items()}
        viz.plot_roc_curves(y, sc, fig_dir / f"roc_{t}.png", f"ROC curves - {t} anomalies (64-D, 5% anomalies, 5% noise)")
        viz.plot_pr_curves(y, sc, fig_dir / f"pr_{t}.png", f"Precision-Recall - {t} anomalies")
        viz.plot_confusion_matrices(y, sc, c.ratio, fig_dir / f"confusion_{t}.png")
        best = max(rows, key=lambda r: r.get("roc_auc") or 0)["algorithm"]
        viz.plot_parallel_coordinates(data["X"], data["y"], data["columns"], fig_dir / f"parallel_coordinates_{t}.png",
                                      title=f"Parallel coordinates - {t} anomalies")
        for m in ["pca", "tsne"]:
            viz.plot_embedding(data["X"], data["y"], data["types"], scores[best], m, fig_dir / f"{m}_{t}.png", seed)
        try:
            viz.plot_embedding(data["X"], data["y"], data["types"], scores[best], "umap", fig_dir / f"umap_{t}.png", seed)
        except Exception as exc:  # umap-learn missing/unavailable
            print(f"[umap skipped] {type(exc).__name__}: {exc}")
        print(f"[figures] {t}: best algorithm in this single run = {best}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--no-representative", action="store_true")
    a = ap.parse_args(argv)
    if a.standin:
        from .synthetic import make_standin
        out, df_all = ROOT / "results_standin", make_standin(6000, seed=0)
        csv = out / "experiment_results.csv"
    else:
        out, csv = ROOT / "results", ROOT / "results" / "experiment_results.csv"
        df_all = pd.read_csv(ROOT / "data" / "processed" / "gas_sensor_drift_parsed.csv")
    if not csv.exists():
        raise SystemExit(f"{csv} not found - run src.experiments first")
    res = run_analysis(csv, out, df_all, standin=a.standin)
    print(guide_to_markdown(res["guide"]))
    if not a.no_representative:
        representative_figures(df_all, out / "figures")
    print(f"\nTables -> {out/'tables'}\nFigures -> {out/'figures'}\nSummary -> {out/'results_summary.md'}")


if __name__ == "__main__":
    main()
