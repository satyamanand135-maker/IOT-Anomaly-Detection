"""
statistical_methods.py : Mahalanobis distance and (adapted) Grubbs' test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, MinCovDet

from .base import BaseDetector


class MahalanobisDetector(BaseDetector):
    """Squared Mahalanobis distance d^2 = (x-m)^T S^-1 (x-m); higher = more anomalous.

    WHY COVARIANCE HANDLING MATTERS IN HIGH DIMENSIONS
    The sample covariance has d(d+1)/2 free parameters. With d=128 (and strongly
    correlated EMA/DR features) it is ill-conditioned or singular, so S^-1 amplifies
    noise in the smallest eigen-directions and distances become unstable. Options:
      ledoit_wolf : shrinkage towards a scaled identity (Ledoit & Wolf 2004)  [default]
      empirical   : plain covariance + tiny ridge; pinv fallback if singular
      pinv        : Moore-Penrose pseudo-inverse of the empirical covariance
      mcd         : Minimum Covariance Determinant (robust to the contamination itself;
                    Rousseeuw & Van Driessen 1999) - slower, needs n >> d.
    Native rule: d^2 > chi2_{0.975}(d), valid only under multivariate normality
    (not true here, so it is reported but not used for the main comparison).
    """
    name, family = "Mahalanobis", "statistical"

    def __init__(self, cov: str = "ledoit_wolf", ridge: float = 1e-6, random_state: int = 0):
        super().__init__()
        self.cov, self.ridge, self.random_state = cov, ridge, random_state

    def fit_score(self, X):
        X = self._check(X)
        n, d = X.shape
        used = self.cov
        try:
            if self.cov == "mcd":
                est = MinCovDet(random_state=self.random_state).fit(X)
                loc, prec, covm = est.location_, est.precision_, est.covariance_
            elif self.cov == "ledoit_wolf":
                est = LedoitWolf().fit(X)
                loc, prec, covm = est.location_, est.precision_, est.covariance_
                self.extras["shrinkage"] = float(est.shrinkage_)
            else:
                est = EmpiricalCovariance().fit(X)
                loc, covm = est.location_, est.covariance_
                if self.cov == "pinv":
                    prec = np.linalg.pinv(covm, hermitian=True)
                else:
                    reg = covm + self.ridge * np.trace(covm) / d * np.eye(d)
                    try:
                        prec = np.linalg.inv(reg)
                    except np.linalg.LinAlgError:
                        prec, used = np.linalg.pinv(covm, hermitian=True), "pinv(fallback)"
        except Exception as exc:                    # e.g. MCD failing when n is small
            est = LedoitWolf().fit(X)
            loc, prec, covm = est.location_, est.precision_, est.covariance_
            used = f"ledoit_wolf(fallback: {type(exc).__name__})"
        diff = X - loc
        d2 = np.einsum("ij,jk,ik->i", diff, prec, diff)
        self.extras.update({"cov_used": used, "cov_cond": float(np.linalg.cond(covm)),
                            "n": n, "d": d})
        self._native = d2 > stats.chi2.ppf(0.975, df=d)
        return d2


def grubbs_critical(n: int, alpha: float) -> float:
    """Two-sided Grubbs critical value G_crit(n, alpha)."""
    t = stats.t.isf(alpha / (2.0 * n), n - 2)
    return (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))


class GrubbsDetector(BaseDetector):
    """Feature-wise Grubbs statistic aggregated across features.

    Grubbs' test is UNIVARIATE: for one sample it tests whether the most extreme value is
    an outlier, G = max|x_i - mean| / s, assuming approximate normality. It does NOT
    natively handle a 128-D vector. ADAPTATION used here:
      1. compute G_ij = |x_ij - mean_j| / s_j for every row i and feature j (single pass);
      2. aggregate across features:
           max    : score_i = max_j G_ij                               [default]
           fisher : score_i = -2 sum_j log p_ij  (Fisher combination; features are
                    correlated, so p-values are NOT valid - ranking score only)
           count  : number of features whose Bonferroni-corrected Grubbs test fires
      3. native flag: any feature exceeds G_crit(n, alpha/d) (Bonferroni over d features).
    LIMITATIONS: ignores feature dependence/correlation structure (cannot see anomalies
    that are only unusual in combination), mean/std are inflated by the outliers
    themselves (masking), normality assumed, single pass (no iterative removal).
    """
    name, family = "Grubbs", "statistical"

    def __init__(self, alpha: float = 0.05, aggregation: str = "max"):
        super().__init__()
        self.alpha, self.aggregation = alpha, aggregation

    def fit_score(self, X):
        X = self._check(X)
        n, d = X.shape
        sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        G = np.abs(X - X.mean(axis=0)) / sd
        gcrit = grubbs_critical(n, self.alpha / d)
        self.extras.update({"g_crit": float(gcrit), "aggregation": self.aggregation})
        self._native = (G > gcrit).any(axis=1)
        if self.aggregation == "max":
            return G.max(axis=1)
        if self.aggregation == "count":
            return (G > gcrit).sum(axis=1).astype(float) + 1e-6 * G.max(axis=1)  # tiny tie-break
        if self.aggregation == "fisher":
            a = np.minimum(n * G ** 2 / (n - 1) ** 2, 1 - 1e-12)
            t2 = a * (n - 2) / (1 - a)
            p = np.clip(n * 2 * stats.t.sf(np.sqrt(t2), n - 2), 1e-300, 1.0)
            return -2.0 * np.log(p).sum(axis=1)
        raise ValueError(f"unknown aggregation {self.aggregation}")
