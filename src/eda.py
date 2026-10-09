"""
eda.py : exploratory data analysis of the validated UCI table (BEFORE any injection).

    python -m src.eda            # real data  (data/processed/gas_sensor_drift_parsed.csv)
    python -m src.eda --standin  # code test on synthetic stand-in data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import visualization as viz
from .preprocessing import (feature_columns, high_correlation_pairs, missing_value_report,
                            variance_analysis)

ROOT = Path(__file__).resolve().parents[1]


def run_eda(df: pd.DataFrame, fig_dir: Path, tab_dir: Path) -> dict:
    fig_dir, tab_dir = Path(fig_dir), Path(tab_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(df)
    out: dict = {"shape": list(df.shape), "n_features": len(cols)}

    # --- tables -------------------------------------------------------------
    pd.DataFrame({"dtype": df.dtypes.astype(str)}).to_csv(tab_dir / "eda_dtypes.csv")
    df[cols].describe().T.to_csv(tab_dir / "eda_descriptive_statistics.csv")
    mv = missing_value_report(df, cols)
    out["missing_cells"], out["rows_with_missing"] = mv["missing_cells"], mv["rows_with_missing"]
    mv["per_column"].to_csv(tab_dir / "eda_missing_per_column.csv")
    out["duplicate_rows"] = int(df.duplicated().sum())
    df.groupby(["Gas"]).size().to_frame("n").to_csv(tab_dir / "eda_gas_counts.csv")
    pd.crosstab(df["Batch"], df["Gas"]).to_csv(tab_dir / "eda_batch_by_gas.csv")
    df["Concentration"].value_counts().sort_index().to_csv(tab_dir / "eda_concentration_counts.csv")
    var = variance_analysis(df, cols)
    var.to_csv(tab_dir / "eda_feature_variance.csv", index=False)
    out["near_zero_variance_features"] = int(var.near_zero_variance.sum())
    hc = high_correlation_pairs(df, cols, 0.95)
    hc.to_csv(tab_dir / "eda_high_correlation_pairs.csv", index=False)
    out["n_pairs_abs_corr_ge_0.95"] = int(len(hc))

    # --- figures ------------------------------------------------------------
    viz.plot_category_counts(df, "Gas", fig_dir / "eda_gas_distribution.png", "Gas class distribution")
    viz.plot_category_counts(df, "Concentration", fig_dir / "eda_concentration_distribution.png",
                             "Concentration level distribution (ppmv)")
    viz.plot_category_counts(df, "Batch", fig_dir / "eda_batch_distribution.png", "Observations per batch")
    viz.plot_feature_distributions(df, cols, fig_dir / "eda_feature_distributions.png", n=12)
    viz.plot_correlation_heatmap(df, cols, fig_dir / "eda_correlation_heatmap.png",
                                 "Correlation of the 128 sensor features")
    _, sensor_tab = viz.plot_sensor_stats(df, cols, fig_dir / "eda_sensor_statistics.png")
    sensor_tab.to_csv(tab_dir / "eda_sensor_statistics.csv", index=False)
    _, frac = viz.plot_prior_outliers(df, cols, fig_dir / "eda_natural_outliers_before_injection.png")
    out["max_pct_natural_outliers_in_a_feature"] = float(frac.max() * 100)

    # --- drift across batches (motivates batch as context) -------------------
    probe = [c for c in ("Sensor1_DR", "Sensor8_DR", "Sensor16_DR") if c in df]
    if probe:
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in probe:
            m = df.groupby("Batch")[c].median()
            ax.plot(m.index, m.values / abs(m.values[0]) if m.values[0] else m.values, marker="o", label=c)
        ax.set_xlabel("batch (time order)"); ax.set_ylabel("median relative to batch 1")
        ax.set_title("Sensor drift across batches (median feature value)"); ax.legend()
        fig.savefig(fig_dir / "eda_batch_drift.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    (tab_dir / "eda_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--standin", action="store_true")
    a = ap.parse_args(argv)
    if a.standin:
        from .synthetic import make_standin
        df, base = make_standin(5000), ROOT / "results_standin"
        print("!! STAND-IN data, not UCI")
    else:
        p = ROOT / "data" / "processed" / "gas_sensor_drift_parsed.csv"
        if not p.exists():
            raise SystemExit("Run `python -m src.data_loader` first.")
        df, base = pd.read_csv(p), ROOT / "results"
    print(json.dumps(run_eda(df, base / "figures", base / "tables"), indent=2))


if __name__ == "__main__":
    main()
