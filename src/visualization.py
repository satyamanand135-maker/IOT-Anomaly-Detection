"""
visualization.py : publication-style figures. Every function saves a PNG (dpi 200)
when `path` is given and returns the matplotlib Figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from .evaluation import normalize_scores, pr_points, roc_points, threshold_top_k

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
ALG_ORDER = ["Mahalanobis", "Grubbs", "KNN", "LOF", "DBSCAN", "IsolationForest", "ExtendedIF"]
PALETTE = dict(zip(ALG_ORDER, sns.color_palette("colorblind", len(ALG_ORDER))))
TYPE_COLORS = {"normal": "#9aa5b1", "point": "#d1495b", "contextual": "#edae49",
               "collective": "#00798c"}


def _save(fig, path):
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


# ----------------------------- EDA / data views ---------------------------- #
def plot_correlation_heatmap(df, cols, path=None, title="Feature correlation (Pearson)"):
    c = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(c, cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
                xticklabels=len(cols) <= 40, yticklabels=len(cols) <= 40,
                cbar_kws={"label": "correlation"})
    ax.set_title(title); ax.set_xlabel("feature"); ax.set_ylabel("feature")
    return _save(fig, path)


def plot_feature_distributions(df, cols, path=None, n=12, seed=0, title="Feature distributions"):
    rng = np.random.default_rng(seed)
    cols = list(rng.choice(cols, size=min(n, len(cols)), replace=False)) if len(cols) > n else list(cols)
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        sns.histplot(df[c].dropna(), kde=True, ax=ax, color="#3d5a80", bins=40)
        ax.set_title(c, fontsize=9); ax.set_xlabel("value"); ax.set_ylabel("count")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _save(fig, path)


def plot_category_counts(df, col, path=None, title=None, hue=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = sorted(df[col].unique())
    sns.countplot(data=df, x=col, hue=hue, order=order, ax=ax,
                  color=None if hue else "#3d5a80")
    ax.set_title(title or f"Observations per {col}"); ax.set_xlabel(col); ax.set_ylabel("count")
    for p in ax.patches if not hue else []:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    return _save(fig, path)


def plot_sensor_stats(df, cols, path=None):
    """Per-sensor mean |z| of ΔR-type features and std (sensor-level statistics)."""
    sens = sorted({int(c.split("_")[0].replace("Sensor", "")) for c in cols})
    rows = []
    for s in sens:
        sc = [c for c in cols if c.startswith(f"Sensor{s}_")]
        z = (df[sc] - df[sc].mean()) / df[sc].std().replace(0, 1)
        rows.append({"sensor": s, "mean_abs_z": z.abs().mean().mean(),
                     "mean_feature_std": df[sc].std().mean(), "n_features": len(sc)})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(t.sensor, t.mean_feature_std, color="#3d5a80"); ax[0].set_yscale("log")
    ax[0].set_title("Mean feature std per sensor (log scale)"); ax[0].set_xlabel("sensor"); ax[0].set_ylabel("std")
    ax[1].bar(t.sensor, t.mean_abs_z, color="#ee6c4d")
    ax[1].set_title("Mean |z| per sensor"); ax[1].set_xlabel("sensor"); ax[1].set_ylabel("mean |z|")
    fig.tight_layout()
    _save(fig, path)
    return fig, t


def plot_prior_outliers(df, cols, path=None, thr=3.5):
    """Natural (pre-injection) outliers: share of |robust z| > thr per feature."""
    med = df[cols].median()
    mad = (df[cols] - med).abs().median().replace(0, np.nan) * 1.4826
    rz = ((df[cols] - med) / mad).abs()
    frac = (rz > thr).mean().fillna(0).sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(range(len(frac)), frac.values * 100, color="#3d5a80")
    ax[0].set_title(f"Natural outliers per feature before injection (|robust z| > {thr})")
    ax[0].set_xlabel("feature rank"); ax[0].set_ylabel("% of observations")
    row_frac = (rz > thr).mean(axis=1)
    ax[1].hist(row_frac * 100, bins=40, color="#ee6c4d")
    ax[1].set_title("Share of extreme features per observation"); ax[1].set_xlabel("% of features"); ax[1].set_ylabel("observations")
    fig.tight_layout()
    return _save(fig, path), frac


def plot_parallel_coordinates(X, y, cols=None, path=None, max_rows=600, max_cols=12, seed=0,
                              title="Parallel coordinates (scaled features)"):
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    cols = list(cols) if cols is not None else []
    cols += [f"ctx{i}" for i in range(len(cols), d)]      # e.g. appended context block
    keep_c = np.sort(rng.choice(d, size=min(max_cols, d), replace=False))
    anom = np.flatnonzero(y == 1)
    norm = np.flatnonzero(y == 0)
    sel = np.concatenate([rng.choice(norm, size=min(max_rows, len(norm)), replace=False),
                          rng.choice(anom, size=min(max_rows // 4, len(anom)), replace=False)]) if len(anom) else norm[:max_rows]
    lo, hi = np.percentile(X[:, keep_c], [1, 99], axis=0)
    Z = np.clip((X[np.ix_(sel, keep_c)] - lo) / np.where(hi - lo == 0, 1, hi - lo), -0.1, 1.1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, r in enumerate(sel):
        lab = y[r]
        ax.plot(range(len(keep_c)), Z[i], color="#d1495b" if lab else "#9aa5b1",
                alpha=0.55 if lab else 0.12, lw=1.1 if lab else 0.7, zorder=3 if lab else 1)
    ax.set_xticks(range(len(keep_c)))
    ax.set_xticklabels([cols[j] for j in keep_c], rotation=45, ha="right", fontsize=8)
    ax.plot([], [], color="#9aa5b1", label="normal"); ax.plot([], [], color="#d1495b", label="anomaly")
    ax.legend(loc="upper right"); ax.set_title(title); ax.set_ylabel("min-max (1-99 pct) scaled value"); ax.set_xlabel("feature")
    return _save(fig, path)


def embed_2d(X, method="pca", seed=0, max_points=2500, perplexity=30):
    """Returns (coords, index_used). t-SNE/UMAP on a subsample for speed."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), size=min(max_points, len(X)), replace=False))
    Xs = X[idx]
    if method == "pca":
        return PCA(2, random_state=seed).fit_transform(Xs), idx
    if method == "tsne":
        from sklearn.manifold import TSNE
        Xp = PCA(min(30, Xs.shape[1]), random_state=seed).fit_transform(Xs) if Xs.shape[1] > 30 else Xs
        return TSNE(2, perplexity=min(perplexity, len(Xs) - 1), random_state=seed,
                    init="pca", learning_rate="auto").fit_transform(Xp), idx
    if method == "umap":
        import umap  # optional dependency
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15).fit_transform(Xs), idx
    raise ValueError("method must be pca|tsne|umap")


def plot_embedding(X, y=None, types=None, scores=None, method="tsne", path=None, seed=0,
                   max_points=2500):
    """Panels: ground-truth label | anomaly type | anomaly score (those that are provided)."""
    emb, idx = embed_2d(X, method, seed, max_points)
    panels = [p for p in ("label", "type", "score") if {"label": y, "type": types, "score": scores}[p] is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        if p == "label":
            yy = np.asarray(y)[idx]
            for v, name, c in [(0, "normal", "#9aa5b1"), (1, "anomaly", "#d1495b")]:
                m = yy == v
                ax.scatter(emb[m, 0], emb[m, 1], s=7 if v == 0 else 14, c=c, alpha=0.5 if v == 0 else 0.9, label=name)
            ax.set_title("Ground-truth label")
        elif p == "type":
            tt = np.asarray(types)[idx]
            for t, c in TYPE_COLORS.items():
                m = tt == t
                if m.any():
                    ax.scatter(emb[m, 0], emb[m, 1], s=7 if t == "normal" else 14, c=c,
                               alpha=0.45 if t == "normal" else 0.9, label=t)
            ax.set_title("Anomaly type")
        else:
            sc = normalize_scores(np.asarray(scores))[idx]
            h = ax.scatter(emb[:, 0], emb[:, 1], s=9, c=sc, cmap="magma")
            plt.colorbar(h, ax=ax, label="rank-normalised anomaly score")
            ax.set_title("Anomaly score")
        ax.set_xlabel(f"{method.upper()} 1"); ax.set_ylabel(f"{method.upper()} 2")
        if p != "score":
            ax.legend(markerscale=1.5, fontsize=8)
    fig.suptitle(f"{method.upper()} projection", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------- evaluation views ---------------------------- #
def plot_roc_curves(y, scores: dict, path=None, title="ROC curves"):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, s in scores.items():
        fpr, tpr, _ = roc_points(y, s)
        from sklearn.metrics import auc
        ax.plot(fpr, tpr, color=PALETTE.get(name), label=f"{name} (AUC={auc(fpr, tpr):.3f})", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, path)


def plot_pr_curves(y, scores: dict, path=None, title="Precision-Recall curves"):
    from sklearn.metrics import average_precision_score
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for name, s in scores.items():
        p, r, _ = pr_points(y, s)
        ax.plot(r, p, color=PALETTE.get(name), lw=1.8, label=f"{name} (AP={average_precision_score(y, s):.3f})")
    ax.axhline(np.mean(y), color="k", ls="--", lw=0.8, label=f"prevalence={np.mean(y):.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    return _save(fig, path)


def plot_confusion_matrices(y, scores: dict, ratio, path=None):
    n = len(scores)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.2 * nrow), squeeze=False)
    from sklearn.metrics import confusion_matrix
    k = int(round(ratio * len(y)))
    for ax, (name, s) in zip(axes.ravel(), scores.items()):
        cm = confusion_matrix(y, threshold_top_k(s, k), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
        ax.set_title(name); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices (top-{k} flagged)", y=1.02)
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------- experiment views ---------------------------- #
def _alg_order(df):
    return [a for a in ALG_ORDER if a in set(df.algorithm)]


def plot_metric_bars(df, metric="roc_auc", by="anomaly_type", path=None, title=None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=df, x=by, y=metric, hue="algorithm", hue_order=_alg_order(df),
                palette=PALETTE, errorbar="sd", ax=ax, capsize=0.08)
    ax.set_ylabel(metric.replace("_", " ").upper()); ax.set_xlabel(by.replace("_", " "))
    ax.set_title(title or f"{metric.upper()} by {by.replace('_', ' ')} (mean ± sd over conditions/seeds)")
    ax.legend(title="algorithm", fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    return _save(fig, path)


def plot_algorithm_heatmap(df, columns="n_dims", metric="roc_auc", path=None, title=None):
    t = df.pivot_table(index="algorithm", columns=columns, values=metric, aggfunc="mean")
    t = t.reindex([a for a in ALG_ORDER if a in t.index])
    fig, ax = plt.subplots(figsize=(1.3 * max(4, t.shape[1]) + 2, 4.2))
    sns.heatmap(t, annot=True, fmt=".2f", cmap="viridis", ax=ax, cbar_kws={"label": metric.upper()})
    ax.set_title(title or f"{metric.upper()}: algorithm x {columns}")
    ax.set_xlabel(columns.replace("_", " ")); ax.set_ylabel("algorithm")
    return _save(fig, path), t


def plot_performance_vs(df, factor, metric="roc_auc", col=None, path=None, title=None):
    d = df.copy()
    g = sns.relplot(data=d, x=factor, y=metric, hue="algorithm", hue_order=_alg_order(d), palette=PALETTE,
                    kind="line", marker="o", errorbar="sd", col=col, height=3.8, aspect=1.15,
                    facet_kws={"sharey": True})
    g.set_axis_labels(factor.replace("_", " "), metric.replace("_", " ").upper())
    g.figure.suptitle(title or f"{metric.upper()} vs {factor.replace('_', ' ')}", y=1.04)
    if factor in ("n_dims",):
        for ax in g.axes.ravel():
            ax.set_xscale("log", base=2); ax.set_xticks(sorted(d[factor].unique()))
            ax.set_xticklabels([str(v) for v in sorted(d[factor].unique())])
    return _save(g.figure, path)
