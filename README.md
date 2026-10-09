# Robust Outlier Detection in High-Dimensional Sensor Data from IoT Networks

A systematic empirical comparison of **statistical** (Mahalanobis, Grubbs), **proximity** (KNN, LOF),
**density** (DBSCAN) and **tree-based** (Isolation Forest, Extended Isolation Forest) anomaly detectors under
controlled variation of **dimensionality, anomaly type, anomaly ratio, noise, sensor dropout, imputation,
irregular sampling and preprocessing**. The deliverable is a *measured* decision guide:
"which method works best under which data conditions?"

> **Status of results.** This repository contains code only. Every table/figure/number in `results/` and in the
> report is produced by running the code on the real UCI data. Nothing is pre-filled. `results_standin/`
> (if you create it) comes from synthetic stand-in data used only to test the code and must never be reported.

## Dataset and honest scope
UCI *Gas Sensor Array Drift Dataset at Different Concentrations* (13,910 measurements, 16 sensors x 8 derived
features = 128 features, 6 gases, 10 batches, ~36 months). It is a **processed multivariate gas-sensor table**,
not a raw temperature/humidity/pressure time series, and it has **no anomaly labels** (the gas class is *not* an
anomaly label). Anomalies are **injected** with known ground truth. Contextual and collective anomalies are
*constructed* (context = gas/concentration; "stream" = row order of the sampled subset); irregular sampling is
*simulated*. The system finds unusual multivariate sensor patterns; it does **not** detect gas leaks.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Google Colab: upload the project folder (or `git clone`), `!pip install -r requirements.txt`, run the same commands with `!`.

## Run order
```bash
python tests/test_data_loader.py        # parser logic (synthetic files)
python tests/test_core.py               # injection / preprocessing / detectors / pipeline
python -m src.data_loader               # 1. download + parse + VALIDATE (13,910 rows; batch counts) -> data/processed/
python -m src.eda                       # 2. EDA tables + figures -> results/
python -m src.experiments --quick       # 3a. fast development run
python -m src.experiments --full        # 3b. final experiments (long; resumable; use --n-jobs N)
python -m src.analysis                  # 4. tables, figures, RQ tests, method-selection guide
streamlit run dashboard/app.py          # 5. dashboard
```
Add `--standin` to steps 2-4 to smoke-test everything without the UCI data (outputs go to `results_standin/`).
If `python -m src.data_loader` reports a mismatch, **stop**: run `python -m src.data_loader --diagnose` and adapt the parser.

## Pipeline (what actually happens per experimental condition)
sample rows -> select dimensions (nested, sensor-aware) -> **inject anomalies** (ground truth) -> add noise ->
simulate sensor dropout -> simulate irregular sampling -> impute/interpolate -> scale -> (PCA / context block) ->
7 detectors -> score (higher = more anomalous) -> predictions (top-k oracle ratio **and** MAD rule) -> metrics.
Injection precedes dropout/resampling (the brief lists the reverse) so dropout can destroy anomalous readings;
this is a deliberate, documented deviation. Detectors never see labels; scalers/imputers are fitted without labels.

## Layout
```
src/data_loader.py          download, parse, name features, validate
src/eda.py                  EDA tables + plots
src/preprocessing.py        missing-value analysis, imputers, scalers, dropout, irregular sampling, dimension selection
src/anomaly_generation.py   point / contextual / collective / mixed injection + trace log
src/{statistical,proximity,density,tree}_methods.py, base.py, registry.py
src/evaluation.py           score standardisation, thresholds, metrics
src/visualization.py        all figures        src/experiments.py   experiment matrix + runner (QUICK/FULL)
src/analysis.py             aggregation, Friedman/Wilcoxon, guide     src/pipeline.py  dashboard backend
src/synthetic.py            STAND-IN data for code tests only
notebooks/01..10            walk-through notebooks (call src/)       dashboard/app.py  Streamlit app
report/project_report.md    report skeleton with placeholders         tests/              unit tests
```
## Extended Isolation Forest source
Backends tried in order: `isotree` (Cortes; pip install isotree), `eif` (Hariri's reference code), then the built-in
NumPy re-implementation of Hariri, Kind & Brunner (2021). The backend used is stored in every result row (`eif_backend`).

## Experiment design
`--quick`: factorial over a small grid, 800 rows, 1 seed. `--full`: one-factor-at-a-time around a baseline
(64-D, 5 % anomalies, no noise, no dropout) for all 3 anomaly types, 3000 rows, 5 seeds, plus imputation /
preprocessing / irregular-sampling / context / k- and eps-sensitivity sweeps. `--design factorial --max-conditions N`
runs a (sub-sampled) full factorial. Runs are appended to `results/experiment_results.csv` and are resumable.
