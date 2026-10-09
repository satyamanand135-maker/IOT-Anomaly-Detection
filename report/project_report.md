# Robust Outlier Detection in High-Dimensional Sensor Data from IoT Networks
### A systematic empirical comparison of anomaly detection methods under controlled data conditions

*Status: methodology complete; experiments **not yet executed**. Every `[[RESULT: ...]]` placeholder must be replaced with numbers taken from `results/tables/*.csv` or `results/results_summary.md` after running the code. Do not write any performance claim that is not produced by those files.*

---

## 1. Abstract
Anomaly detection in multivariate IoT sensor streams is difficult because the best method depends on dimensionality, anomaly type, noise, anomaly prevalence and data loss. Rather than proposing a single "best" detector, this work compares seven detectors from four families (statistical: Mahalanobis distance, Grubbs' test; proximity: k-nearest-neighbour distance, Local Outlier Factor; density: DBSCAN; tree-based: Isolation Forest, Extended Isolation Forest) on the UCI Gas Sensor Array Drift dataset (13,910 measurements, 16 sensors, 128 features). Because the dataset has no anomaly labels, point, contextual and collective anomalies are injected with known ground truth under controlled dimensionality (16-128), anomaly ratio (1-20 %), noise (0-20 % of feature std) and sensor dropout (0-30 %), with several imputation and interpolation strategies. Performance is measured with ROC-AUC, PR-AUC, F1, FPR and FNR and summarised as a condition-to-method decision guide. `[[RESULT: one-sentence headline finding, e.g. whether a consistent winner exists (Friedman p, win-rates)]]`.

## 2. Introduction
IoT deployments produce high-dimensional sensor data in which faults, drift and unusual operating conditions appear as anomalies (Cook et al., 2020; Ni et al., 2009). Many detectors exist (Chandola et al., 2009) but comparative evidence under *controlled, varying* data conditions is scarce, and high dimensionality is known to degrade distance-based reasoning (Beyer et al., 1999; Aggarwal et al., 2001; Zimek et al., 2012).

## 3. Problem Statement
Given multivariate sensor measurements without anomaly labels, which unsupervised detector gives the best ranking and decisions under which combination of dimensionality, anomaly type, anomaly ratio, noise level and missing-data condition?

## 4. Motivation
Practitioners must choose a detector before labels exist. A method-selection guide derived from controlled experiments, with uncertainty, is more actionable than a single leaderboard.

## 5. Objectives
1. Build a validated, reproducible pipeline on the official UCI data.
2. Define documented, reproducible point, contextual and collective anomaly injection.
3. Compare seven detectors across dimensionality, noise, ratio, dropout, imputation and preprocessing.
4. Answer RQ1-RQ7 from measured results and generate a practical method-selection guide.
5. Provide an interactive dashboard usable on new sensor data.

**Research questions.** RQ1 dimensionality; RQ2 noise robustness; RQ3 anomaly type; RQ4 anomaly ratio; RQ5 sensor dropout; RQ6 imputation; RQ7 is there a consistent best algorithm?

## 6. Dataset Description
UCI "Gas Sensor Array Drift Dataset at Different Concentrations" (Vergara et al., 2012; Rodriguez-Lujan et al., 2014): 13,910 measurements from 16 metal-oxide sensors over ~36 months in 10 batches (445, 1244, 1586, 161, 197, 2300, 3613, 294, 470, 3600), six gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, toluene) at several concentrations. Each sensor contributes 8 derived features (steady-state response, its absolute value, and EMA transient features with smoothing 0.001/0.01/0.1, increasing and decreasing) = 128 features (naming in `data/processed/feature_mapping.csv`; **verify order against the UCI variable table**).
**Caveats.** (i) processed multivariate table, not raw temperature/humidity/pressure time series; (ii) no anomaly labels - the gas class is *not* an anomaly label; (iii) no raw timestamps; (iv) the system flags unusual multivariate patterns, not gas leaks; (v) the real data contain natural outliers and batch drift, which add unlabeled noise to the injected ground truth.
`[[RESULT: validation output (shape, batch counts, NaN, duplicates) from notebook 01 / python -m src.data_loader]]` `[[RESULT: EDA summary from results/tables/eda_summary.json]]`

## 7. Data Preprocessing
Missing-value analysis; imputation (mean, median, KNN; time-aware ffill/linear/PCHIP/nearest for stream gaps); scaling (standard, robust, min-max); variance and correlation analysis; nested dimensionality subsets (16 = one derived feature per sensor, 32, 64, 128 - all 16 sensors retained; PCA and sensor-subset strategies available).
**Leakage control.** Detectors are unsupervised and never receive labels. Imputers and scalers are fitted without labels on the data to be scored (transductive, as available to a practitioner); the `Preprocessor` class supports train-only fitting for deployment (tested). *Limitation:* added derived features are highly correlated, so nominal dimensionality overstates intrinsic dimensionality (see `curse_of_dimensionality_diagnostics.csv`: PCA components for 95 % variance).

## 8. Anomaly Types
Point (single observation unusual), contextual (unusual relative to context such as gas/concentration), collective (a group jointly unusual) following Chandola et al. (2009). Here: context = (gas class, concentration); "stream order" = row order of the sampled subset and is *constructed*, not physical time.

## 9. Anomaly Injection Methodology
Original data are never modified; each experiment works on a copy; every injection is logged (`InjectedData.log`).
* **Point:** per anomalous row, k = 4 randomly chosen features are altered by one of three mechanisms: z-score shift (+/- s sigma), multiplicative spike (x(1+s); scale-dependent), or extreme replacement (mu +/- (s+U(0,1)) sigma); s = 3 by default (1-4 supported). A fixed k means the anomalous subspace shrinks relative to d, deliberately exposing dilution in high dimensions.
* **Contextual:** the feature vector of the selected row is replaced by a real measurement from a donor of a *different gas* (taken from rows outside the experimental subset) while keeping the row's own context. The vector is globally plausible but inconsistent with its context. Detectors receive context as appended attributes (one-hot gas + standardised log concentration; Chandola's contextual vs behavioural attributes); a context-blind variant is run for comparison.
* **Collective:** non-overlapping windows of 10 consecutive rows receive the same coordinated shift (2 sigma on k features); all window rows are labelled anomalous. Detectors score rows individually (no window-level scoring) - a stated limitation.
* **Noise** N(0, (level x sigma_j)^2) is added after injection; **dropout** (random values / sensor blocks / consecutive bursts) and **irregular sampling** (random or bursty row removal; *simulated*) follow, then imputation. *Order deviation:* the brief lists dropout/resampling before injection; we inject first so that dropout can destroy anomalous readings.

## 10. Algorithms
* **Mahalanobis** (Mahalanobis, 1936): squared distance with Ledoit-Wolf shrinkage covariance by default (Ledoit & Wolf, 2004); empirical+ridge, pseudo-inverse and MCD (Rousseeuw & Van Driessen, 1999) available because the sample covariance of 128 correlated features is ill-conditioned. Condition numbers are logged.
* **Grubbs** (Grubbs, 1969): univariate test; adapted by computing the Grubbs statistic per feature and aggregating (max / Fisher / count). Limitations: ignores feature dependence, masking by contaminated mean/sd, normality, single pass.
* **KNN** (Ramaswamy et al., 2000): distance to the k-th neighbour (k = 10). **LOF** (Breunig et al., 2000): k = 20; score = -negative_outlier_factor_.
* **DBSCAN** (Ester et al., 1996; Schubert et al., 2017): eps from the k-distance elbow, min_samples = 10; continuous score = distance to nearest core point / eps (>1 = noise); degeneracy (noise > 50 % or no cluster) logged.
* **Isolation Forest** (Liu et al., 2008; scikit-learn): score = -score_samples. **Extended Isolation Forest** (Hariri et al., 2021): random-hyperplane splits; backend isotree -> eif -> in-house NumPy re-implementation, recorded per run.
Default hyper-parameters are fixed (not tuned with labels); sensitivity to k and eps is reported separately.

## 11. Experimental Setup
Conditions (full design): anomaly type {point, contextual, collective}; n_dims {16, 32, 64, 128}; ratio {1, 5, 10, 20 %}; noise {0, 5, 10, 20 %}; dropout {0, 10, 20 %} (+30 % in the imputation sweep); imputers {mean, median, KNN, ffill, linear}; scalers {standard, robust, min-max}; irregular sampling {0, 20, 40 %} x {ffill, linear, PCHIP}. Design: one-factor-at-a-time around a baseline (64-D, 5 %, no noise, no dropout) with `n_samples = 3000`, 5 seeds (a sub-sampled full factorial is available). Seeds fix sampling, injection, noise, dropout and tree randomness. `[[RESULT: actual configuration used, number of conditions, runs, runtime, software versions]]`

## 12. Evaluation Metrics
ROC-AUC and PR-AUC (threshold-free, primary); accuracy, precision, recall, F1, FPR, FNR under (a) top-k flagging with the true ratio (oracle contamination; here FP = FN so precision = recall = F1) and (b) a data-driven MAD rule (robust z > 3.5). Accuracy is reported but not used for conclusions (class imbalance). Metrics are computed on observed (non-interpolated) rows. Paired Wilcoxon tests compare the top two methods per condition group; the Friedman test and mean ranks address RQ7.

## 13. Results
> Fill from `results/` only.
* Overall win-rates / mean ranks / Friedman: `[[RESULT: results/tables/rq7_win_rates_mean_ranks.csv, rq7_friedman.json]]`
* RQ1 dimensionality: `[[RESULT: perf_vs_dimensionality_*.png, heatmap_algorithm_x_dimensionality.csv, curse_of_dimensionality_diagnostics.csv, mahalanobis_condition_number.csv, dbscan_degeneracy_by_dims.csv]]`
* RQ2 noise: `[[RESULT: perf_vs_noise_*.png, robustness_noise.csv]]`
* RQ3 anomaly type: `[[RESULT: bar_*_by_anomaly_type.png, summary_by_anomaly_type.csv, contextual_context_mode_roc_auc.csv]]`
* RQ4 ratio: `[[RESULT: perf_vs_anomaly_ratio_*.png]]`
* RQ5 dropout: `[[RESULT: perf_vs_sensor_dropout_*.png, robustness_dropout.csv]]`
* RQ6 imputation: `[[RESULT: rq6_imputation_roc_auc.csv, rq6_imputation_effect.png, imputation_rmse.csv]]`
* Irregular sampling, scalers, k/eps sensitivity: `[[RESULT: irregular_sampling_roc_auc.csv, preprocessing_scaler_roc_auc.csv, sensitivity_k.csv, sensitivity_dbscan_eps.csv]]`
* ROC/PR/confusion/embeddings for representative conditions: `[[RESULT: roc_*.png, pr_*.png, confusion_*.png, pca/tsne/umap_*.png]]`

## 14. Discussion
`[[RESULT: interpret ONLY what the tables show. Address: do distances lose discrimination (relative contrast table)? do LOF/KNN degrade? is DBSCAN harder to tune (degeneracy rate, eps sensitivity)? is Mahalanobis unstable (condition numbers, shrinkage vs empirical)? are tree methods more robust? Do not assume the hypothesis in advance - if results contradict expectations, report that.]]`

## 15. Algorithm Comparison
`[[RESULT: mean +/- sd ROC-AUC / PR-AUC per algorithm and family; runtime table runtime_by_dims.csv]]`

## 16. Practical Method Selection Guide
`[[RESULT: paste results/results_summary.md - "Condition -> recommended methods -> measured evidence -> limitations"]]` Recommendations hold only for the injected-anomaly protocol of this study.

## 17. Limitations
Injected, not naturally occurring, anomalies; processed (not raw) sensor features; constructed stream order; simulated irregular sampling; fixed k anomalous features; nominal vs intrinsic dimensionality; default hyper-parameters; top-k oracle ratio in the primary F1; natural outliers/drift in the real data act as label noise; row-wise scoring for collective anomalies; EIF backend may differ slightly from the reference implementation; one dataset.

## 18. Future Work
Window-level scoring for collective anomalies; per-context conditional detectors; raw time-series IoT datasets with real timestamps and labelled faults; hyper-parameter selection without labels; additional detectors (autoencoders, HBOS, ECOD); drift-aware evaluation across batches.

## 19. Conclusion
`[[RESULT: after experiments - answer RQ1-RQ7 in a few sentences, strictly from the results.]]`

## 20. References
References marked **[verified]** were checked against an external source while preparing this project; the others were written from memory of the literature and **must be checked (title, volume, pages, DOI) before submission**.

1. Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. *IEEE ICDM*, 413-422. https://doi.org/10.1109/ICDM.2008.17 **[verified]**
2. Hariri, S., Carrasco Kind, M., Brunner, R. J. (2021). Extended Isolation Forest. *IEEE TKDE* 33(4), 1479-1489. https://doi.org/10.1109/TKDE.2019.2947676 **[verified]**
3. Cortes, D. (2021). Revisiting randomized choices in isolation forests. arXiv:2110.13402 (isotree library). https://doi.org/10.48550/arXiv.2110.13402 **[verified]**; Cortes, D. (2019). Imputing missing values with unsupervised random trees. arXiv:1911.06646 **[verified as cited by 2110.13402]**
4. Breunig, M. M., Kriegel, H.-P., Ng, R. T., Sander, J. (2000). LOF: identifying density-based local outliers. *ACM SIGMOD*, 93-104. https://doi.org/10.1145/335191.335388
5. Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 226-231.
6. Schubert, E., Sander, J., Ester, M., Kriegel, H. P., Xu, X. (2017). DBSCAN revisited, revisited. *ACM TODS* 42(3), 19. https://doi.org/10.1145/3068335
7. Ramaswamy, S., Rastogi, R., Shim, K. (2000). Efficient algorithms for mining outliers from large data sets. *ACM SIGMOD*, 427-438. https://doi.org/10.1145/342009.335437
8. Mahalanobis, P. C. (1936). On the generalised distance in statistics. *Proc. National Institute of Sciences of India* 2(1), 49-55.
9. Grubbs, F. E. (1969). Procedures for detecting outlying observations in samples. *Technometrics* 11(1), 1-21. https://doi.org/10.1080/00401706.1969.10490657
10. Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Analysis* 88(2), 365-411. https://doi.org/10.1016/S0047-259X(03)00096-4
11. Rousseeuw, P. J., Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics* 41(3), 212-223. https://doi.org/10.1080/00401706.1999.10485670
12. Chandola, V., Banerjee, A., Kumar, V. (2009). Anomaly detection: a survey. *ACM Computing Surveys* 41(3), 15. https://doi.org/10.1145/1541880.1541882
13. Beyer, K., Goldstein, J., Ramakrishnan, R., Shaft, U. (1999). When is "nearest neighbor" meaningful? *ICDT*, 217-235. https://doi.org/10.1007/3-540-49257-7_15
14. Aggarwal, C. C., Hinneburg, A., Keim, D. A. (2001). On the surprising behavior of distance metrics in high dimensional space. *ICDT*, 420-434. https://doi.org/10.1007/3-540-44503-X_27
15. Zimek, A., Schubert, E., Kriegel, H.-P. (2012). A survey on unsupervised outlier detection in high-dimensional numerical data. *Statistical Analysis and Data Mining* 5(5), 363-387. https://doi.org/10.1002/sam.11161
16. Campos, G. O., et al. (2016). On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. *Data Mining and Knowledge Discovery* 30, 891-927. https://doi.org/10.1007/s10618-015-0444-8
17. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L., Huerta, R. (2012). Chemical gas sensor drift compensation using classifier ensembles. *Sensors and Actuators B* 166-167, 320-329. https://doi.org/10.1016/j.snb.2012.01.074
18. Rodriguez-Lujan, I., Fonollosa, J., Vergara, A., Homer, M., Huerta, R. (2014). On the calibration of sensor arrays for pattern recognition using the minimal number of experiments. *Chemometrics and Intelligent Laboratory Systems* 130, 123-134. https://doi.org/10.1016/j.chemolab.2013.10.012
19. Cook, A. A., Misirli, G., Fan, Z. (2020). Anomaly detection for IoT time-series data: a survey. *IEEE Internet of Things Journal* 7(7), 6481-6494. https://doi.org/10.1109/JIOT.2019.2958185
20. Ni, K., et al. (2009). Sensor network data fault types. *ACM Transactions on Sensor Networks* 5(3), 25. https://doi.org/10.1145/1525856.1525863
21. Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics* 17(6), 520-525. https://doi.org/10.1093/bioinformatics/17.6.520 (KNN imputation)
22. Little, R. J. A., Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed. Wiley. https://doi.org/10.1002/9781119482260
23. Pedregosa, F., et al. (2011). Scikit-learn: machine learning in Python. *JMLR* 12, 2825-2830.
24. UCI Machine Learning Repository, Gas Sensor Array Drift Dataset at Different Concentrations (id 270). https://archive.ics.uci.edu/dataset/270
