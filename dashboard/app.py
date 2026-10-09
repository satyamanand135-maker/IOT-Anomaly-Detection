"""
Streamlit dashboard: run any of the 7 detectors on uploaded (or built-in) sensor data.

    streamlit run dashboard/app.py

Data can be CSV/Excel with numeric sensor columns. An optional 0/1 column with known anomalies
enables evaluation (ROC-AUC, PR-AUC, confusion matrix). The tool flags UNUSUAL MULTIVARIATE
PATTERNS (faults, drift, unusual conditions); it does not identify gas leaks.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src import visualization as viz
from src.anomaly_generation import InjectionConfig, inject_into_dataframe
from src.evaluation import pr_points, roc_points
from src.pipeline import IMPUTERS, run_detection
from src.preprocessing import encode_context, feature_columns
from src.registry import ALGORITHMS

PARSED = ROOT / "data" / "processed" / "gas_sensor_drift_parsed.csv"
st.set_page_config(page_title="IoT Sensor Anomaly Detection", layout="wide")
st.title("Robust Outlier Detection in High-Dimensional IoT Sensor Data")
st.caption("Flags unusual multivariate sensor patterns (possible faults, drift, unusual operating "
           "conditions). It does not directly detect gas leaks.")


@st.cache_data(show_spinner=False)
def load_upload(name: str, data: bytes) -> pd.DataFrame:
    import io
    return pd.read_excel(io.BytesIO(data)) if name.lower().endswith(("xls", "xlsx")) else pd.read_csv(io.BytesIO(data))


@st.cache_data(show_spinner=False)
def load_uci() -> pd.DataFrame:
    return pd.read_csv(PARSED)


@st.cache_data(show_spinner=False)
def load_standin() -> pd.DataFrame:
    from src.synthetic import make_standin
    return make_standin(3000)


# ------------------------------ sidebar: data ------------------------------ #
st.sidebar.header("1. Data")
options = ["Upload CSV/Excel"] + (["Validated UCI table"] if PARSED.exists() else []) + ["Synthetic demo (NOT UCI)"]
src = st.sidebar.radio("Source", options)
df = None
if src == "Upload CSV/Excel":
    up = st.sidebar.file_uploader("File", type=["csv", "xlsx", "xls"])
    if up is not None:
        df = load_upload(up.name, up.getvalue())
elif src == "Validated UCI table":
    df = load_uci()
    n = st.sidebar.slider("Rows to analyse (random subset)", 500, len(df), min(3000, len(df)), 500)
    df = df.sample(n, random_state=0).sort_index()
else:
    df = load_standin()
    st.sidebar.warning("Synthetic stand-in data: for demonstration only.")

if df is None:
    st.info("Upload a CSV/Excel file with numeric sensor columns to begin.")
    st.stop()

num_cols = df.select_dtypes(include=np.number).columns.tolist()
default_cols = feature_columns(df) or num_cols
cols = st.sidebar.multiselect("Sensor columns", num_cols, default=default_cols[:128])
if len(cols) < 2:
    st.warning("Select at least two numeric sensor columns.")
    st.stop()

# optional injection (to create ground truth on clean data)
label_col = None
with st.sidebar.expander("Inject synthetic anomalies (for validation)"):
    if st.checkbox("Inject known anomalies into a copy"):
        at = st.selectbox("Type", ["point", "collective", "mixed", "contextual"])
        ratio = st.slider("Anomaly ratio", 0.01, 0.30, 0.05, 0.01)
        strength = st.slider("Strength (sigma)", 1.0, 6.0, 3.0, 0.5)
        pool = df.sample(min(len(df), 2000), random_state=1) if {"Class", "Concentration"} <= set(df.columns) else None
        try:
            df = inject_into_dataframe(df, cols, InjectionConfig(anomaly_type=at, ratio=ratio, strength=strength, seed=0), pool)
            label_col = "anomaly_label"
            st.success("Injected; ground truth in 'anomaly_label'.")
        except Exception as e:
            st.error(f"Injection failed: {e}")
if label_col is None:
    cands = ["(none)"] + [c for c in df.columns if df[c].nunique() <= 2 and c not in cols]
    pick = st.sidebar.selectbox("Optional 0/1 ground-truth column", cands)
    label_col = None if pick == "(none)" else pick

# --------------------------- sidebar: preprocessing ------------------------ #
st.sidebar.header("2. Preprocessing")
imputer = st.sidebar.selectbox("Missing values", IMPUTERS, index=IMPUTERS.index("median"))
scaler = st.sidebar.selectbox("Scaling", ["standard", "robust", "minmax", "none"])
use_pca = st.sidebar.checkbox("PCA before detection")
pca_k = st.sidebar.slider("PCA components", 2, max(2, min(64, len(cols))), min(16, len(cols))) if use_pca else None
corr_thr = st.sidebar.slider("Drop features with |corr| >", 0.80, 1.00, 1.00, 0.01)
use_ctx = False
if {"Class", "Concentration"} <= set(df.columns):
    use_ctx = st.sidebar.checkbox("Add context (gas class + concentration) - for contextual anomalies")

# ------------------------------ sidebar: algorithm ------------------------- #
st.sidebar.header("3. Algorithm")
alg = st.sidebar.selectbox("Detector", list(ALGORITHMS), index=list(ALGORITHMS).index("IsolationForest"))
p: dict = {}
if alg == "Mahalanobis":
    p["cov"] = st.sidebar.selectbox("Covariance", ["ledoit_wolf", "empirical", "pinv", "mcd"])
elif alg == "Grubbs":
    p["aggregation"] = st.sidebar.selectbox("Aggregation across features", ["max", "fisher", "count"])
    p["alpha"] = st.sidebar.select_slider("alpha", [0.001, 0.01, 0.05, 0.1], 0.05)
elif alg == "KNN":
    p["k"] = st.sidebar.slider("k", 1, 100, 10)
    p["method"] = st.sidebar.selectbox("Distance", ["largest", "mean", "median"])
elif alg == "LOF":
    p["k"] = st.sidebar.slider("n_neighbors", 2, 100, 20)
elif alg == "DBSCAN":
    p["min_samples"] = st.sidebar.slider("min_samples", 2, 50, 10)
    p["eps_strategy"] = st.sidebar.selectbox("eps selection", ["knee", "quantile", "manual"])
    if p["eps_strategy"] == "quantile":
        p["quantile"] = st.sidebar.slider("k-distance quantile", 0.5, 0.99, 0.9, 0.01)
    if p["eps_strategy"] == "manual":
        p["eps"] = st.sidebar.number_input("eps", 0.01, 1000.0, 3.0)
        p.pop("eps_strategy")
elif alg == "IsolationForest":
    p["n_estimators"] = st.sidebar.slider("trees", 50, 500, 200, 50)
    p["max_samples"] = st.sidebar.select_slider("max_samples", [64, 128, 256, 512], 256)
elif alg == "ExtendedIF":
    p["n_estimators"] = st.sidebar.slider("trees", 50, 300, 100, 50)
    p["max_samples"] = st.sidebar.select_slider("max_samples", [64, 128, 256, 512], 256)

st.sidebar.header("4. Decision rule")
mode = st.sidebar.radio("Score -> label", ["top_percent", "mad", "native"],
                        format_func={"top_percent": "Flag top x %", "mad": "Robust z (MAD) > 3.5",
                                     "native": "Algorithm's own rule"}.get)
cont = st.sidebar.slider("Expected anomaly %", 0.5, 30.0, 5.0, 0.5) / 100 if mode == "top_percent" else 0.05
run = st.sidebar.button("Run anomaly detection", type="primary")

st.subheader("Data preview")
st.write(f"{df.shape[0]:,} rows, {len(cols)} sensor columns selected, {int(df[cols].isna().sum().sum())} missing values.")
st.dataframe(df.head(10))

if run:
    ctx = encode_context(df["Class"].to_numpy(), df["Concentration"].to_numpy()) if use_ctx else None
    with st.spinner("Running..."):
        try:
            st.session_state["out"] = run_detection(df, cols, alg, p, imputer, scaler, pca_k,
                                                    corr_thr if corr_thr < 1 else None, mode, cont,
                                                    label_col, ctx)
            st.session_state["df"], st.session_state["alg"], st.session_state["cols"] = df, alg, cols
        except Exception as e:
            st.error(f"Detection failed: {type(e).__name__}: {e}")

out = st.session_state.get("out")
if out is not None:
    df_r, alg_r, cols_r = st.session_state["df"], st.session_state["alg"], st.session_state["out"]["columns_used"]
    res = out["result_df"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", alg_r)
    c2.metric("Anomalies flagged", f"{out['n_anomalies']:,}")
    c3.metric("Share flagged", f"{out['pct']:.2f} %")
    if out["extras"]:
        with st.expander("Algorithm diagnostics"):
            st.json({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in out["extras"].items()})

    tabs = st.tabs(["Scores", "Score distribution", "Feature distributions", "Correlation", "Projection",
                    "Parallel coordinates", "Evaluation", "Download"])
    with tabs[0]:
        show = pd.concat([df_r.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        st.dataframe(show.sort_values("anomaly_score", ascending=False).head(200))
    with tabs[1]:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.histplot(data=res, x="anomaly_score", hue="label", bins=60, ax=ax, palette={"normal": "#9aa5b1", "anomaly": "#d1495b"})
        ax.set_title(f"{alg_r}: anomaly score distribution"); ax.set_xlabel("anomaly score (higher = more anomalous)"); ax.set_ylabel("count")
        st.pyplot(fig)
    with tabs[2]:
        st.pyplot(viz.plot_feature_distributions(df_r, cols_r, None, n=12))
    with tabs[3]:
        sub = cols_r if len(cols_r) <= 128 else cols_r[:128]
        st.pyplot(viz.plot_correlation_heatmap(df_r, sub))
    with tabs[4]:
        method = st.selectbox("Method", ["pca", "tsne", "umap"])
        if st.button("Compute projection"):
            try:
                st.pyplot(viz.plot_embedding(out["X_processed"], out.get("y"), None, res["anomaly_score"].to_numpy(), method))
            except Exception as e:
                st.error(f"{method} failed ({e}). For UMAP: pip install umap-learn")
    with tabs[5]:
        y = out.get("y", res["prediction"].to_numpy())
        st.caption("Coloured by ground truth if provided, otherwise by predicted label.")
        st.pyplot(viz.plot_parallel_coordinates(out["X_processed"], y, cols_r))
    with tabs[6]:
        m = out["metrics"]
        if m is None:
            st.info("Provide a 0/1 ground-truth column (or inject anomalies) to see evaluation.")
        else:
            st.dataframe(pd.DataFrame([{k: round(float(v), 4) for k, v in m.items()}]).T.rename(columns={0: "value"}))
            sc = {alg_r: res["anomaly_score"].to_numpy()}
            a, b = st.columns(2)
            a.pyplot(viz.plot_roc_curves(out["y"], sc)); b.pyplot(viz.plot_pr_curves(out["y"], sc))
    with tabs[7]:
        dl = pd.concat([df_r.reset_index(drop=True), res.reset_index(drop=True)], axis=1)
        st.download_button("Download predictions (CSV)", dl.to_csv(index=False).encode(), "predictions.csv", "text/csv")
