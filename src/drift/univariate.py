"""
Univariate temporal drift detection (robust z-score feature deviation count).

For every feature the historical mean / standard deviation are computed from PAST
data only (year < current_year).  The test year's value is converted to a z-score
against that baseline and a feature is flagged when |z| exceeds DRIFT_Z_FEATURE
(2.5 sigma).  The number of flagged features is what the detector uses for severity.

Works for single-row test inputs (one crop-year = one row), unlike KS / PSI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.settings import DRIFT_Z_FEATURE

logger = logging.getLogger(__name__)

_MIN_BASELINE_ROWS = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UnivariateDriftResult:
    """Drift test result for a single feature.

    Field names are kept so the API schema and dashboard keep working:
      ks_statistic = |z| of the test year vs the historical baseline
      ks_pvalue    = two-sided normal p-value of that z-score
      ks_drifted   = |z| > DRIFT_Z_FEATURE
      psi_score    = 0.0 (PSI is no longer computed)
      psi_severity = "LOW"
    """
    feature: str
    ks_statistic: float
    ks_pvalue: float
    ks_drifted: bool
    psi_score: float = 0.0
    psi_severity: str = "LOW"
    z_score: float = 0.0          # signed z-score
    hist_mean: float = 0.0
    hist_std: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_univariate_drift(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    train_years: np.ndarray | None = None,
    current_year: int | None = None,
) -> list[UnivariateDriftResult]:
    """
    Z-score every feature of the test year against its historical baseline.

    Parameters
    ----------
    X_train : (n_train, n_features) historical feature matrix.
    X_test  : (n_test, n_features) test matrix (a single row is fine; rows are averaged).
    feature_names : names for reporting.
    train_years, current_year : optional.  When both are given, only rows with
        year < current_year are used for the baseline (guards against leakage).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    if train_years is not None and current_year is not None:
        mask = np.asarray(train_years) < current_year
        X_train = X_train[mask]

    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    results: list[UnivariateDriftResult] = []
    for i in range(n_features):
        tr = X_train[:, i]
        tr = tr[np.isfinite(tr)]
        te = X_test[:, i]
        te = te[np.isfinite(te)]

        z = 0.0
        mu = float(tr.mean()) if len(tr) else 0.0
        sd = float(tr.std(ddof=1)) if len(tr) >= 2 else 0.0

        if len(tr) >= _MIN_BASELINE_ROWS and len(te) > 0 and sd > 1e-12:
            z = float((te.mean() - mu) / sd)

        abs_z = abs(z)
        results.append(UnivariateDriftResult(
            feature=feature_names[i],
            ks_statistic=abs_z,
            ks_pvalue=float(2 * (1 - norm.cdf(abs_z))),
            ks_drifted=abs_z > DRIFT_Z_FEATURE,
            psi_score=0.0,
            psi_severity="LOW",
            z_score=z,
            hist_mean=mu,
            hist_std=sd,
        ))

    logger.info(
        "Univariate drift: %d/%d features beyond %.1f sigma",
        count_drifted_features(results), n_features, DRIFT_Z_FEATURE,
    )
    return results


def count_drifted_features(results: list[UnivariateDriftResult]) -> int:
    """Number of features whose |z| exceeds DRIFT_Z_FEATURE."""
    return sum(1 for r in results if r.ks_drifted)


def summarize_univariate(results: list[UnivariateDriftResult]) -> dict:
    """Summarize univariate drift results into a dashboard-friendly dict."""
    drifted = sorted((r for r in results if r.ks_drifted),
                     key=lambda r: r.ks_statistic, reverse=True)
    return {
        "n_features_tested": len(results),
        "n_features_drifted": len(drifted),
        "z_threshold": DRIFT_Z_FEATURE,
        "drifted_features": [
            {
                "feature": r.feature,
                "ks_pvalue": round(r.ks_pvalue, 4),
                "psi_score": round(r.psi_score, 4),
                "psi_severity": r.psi_severity,
                "z_score": round(r.z_score, 3),
            }
            for r in drifted
        ],
        "max_psi": max((r.psi_score for r in results), default=0.0),
        "max_abs_z": max((r.ks_statistic for r in results), default=0.0),
    }
