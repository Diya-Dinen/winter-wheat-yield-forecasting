"""
Univariate temporal drift detection.

Tests whether individual weather features in a new test year are statistically
different from their training-period distribution, using:
  - Kolmogorov–Smirnov (KS) two-sample test
  - Population Stability Index (PSI)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    DRIFT_Z_FEATURE,
    KS_ALPHA,
    MIN_BATCH_FOR_PSI_KS,
    PSI_HIGH,
    PSI_MODERATE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UnivariateDriftResult:
    """Drift test results for a single feature."""
    feature: str
    ks_statistic: float
    ks_pvalue: float
    ks_drifted: bool
    psi_score: float
    psi_severity: str  # LOW / MODERATE / HIGH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_univariate_drift(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
) -> list[UnivariateDriftResult]:
    """
    Run KS test and PSI on each feature independently.

    Parameters
    ----------
    X_train : array-like, shape (n_train, n_features)
        Training feature matrix.
    X_test : array-like, shape (n_test, n_features)
        Test feature matrix (can be a single row for 1-year prediction).
    feature_names : list[str], optional
        Feature names for reporting.

    Returns
    -------
    list[UnivariateDriftResult]
        One result per feature.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # KS and PSI compare DISTRIBUTIONS and are meaningless for a single test year
    # (PSI from one observation is ~8-10 for ANY value).  For small test samples use a
    # per-feature z-score against the training distribution instead.  The result fields
    # are reused: ks_statistic = |z|, ks_pvalue = two-sided normal p-value of z,
    # ks_drifted = |z| > DRIFT_Z_FEATURE, PSI is not computed (0.0 / LOW).
    if X_test.shape[0] < MIN_BATCH_FOR_PSI_KS:
        from scipy.stats import norm
        results = []
        for i in range(n_features):
            tr = X_train[:, i]
            tr = tr[~np.isnan(tr)]
            te = X_test[:, i]
            te = te[~np.isnan(te)]
            if len(tr) < 3 or len(te) == 0 or np.std(tr, ddof=1) == 0:
                z, p = 0.0, 1.0
            else:
                z = abs(float((te.mean() - tr.mean()) / np.std(tr, ddof=1)))
                p = float(2 * (1 - norm.cdf(z)))
            results.append(UnivariateDriftResult(
                feature=feature_names[i], ks_statistic=z, ks_pvalue=p,
                ks_drifted=z > DRIFT_Z_FEATURE, psi_score=0.0, psi_severity="LOW",
            ))
        return results

    results = []
    for i in range(n_features):
        train_col = X_train[:, i]
        test_col = X_test[:, i]

        # Remove NaN from both
        train_clean = train_col[~np.isnan(train_col)]
        test_clean = test_col[~np.isnan(test_col)]

        # KS test
        if len(train_clean) > 0 and len(test_clean) > 0:
            ks_stat, ks_pval = stats.ks_2samp(train_clean, test_clean)
        else:
            ks_stat, ks_pval = 0.0, 1.0

        ks_drifted = ks_pval < KS_ALPHA

        # PSI
        psi_score = _compute_psi(train_clean, test_clean)
        psi_severity = _psi_severity(psi_score)

        results.append(UnivariateDriftResult(
            feature=feature_names[i],
            ks_statistic=float(ks_stat),
            ks_pvalue=float(ks_pval),
            ks_drifted=ks_drifted,
            psi_score=float(psi_score),
            psi_severity=psi_severity,
        ))

    n_drifted = sum(1 for r in results if r.ks_drifted)
    logger.info(
        "Univariate drift: %d/%d features flagged by KS test (α=%.2f)",
        n_drifted, n_features, KS_ALPHA,
    )

    return results


def summarize_univariate(
    results: list[UnivariateDriftResult],
) -> dict:
    """Summarize univariate drift results into a dashboard-friendly dict."""
    drifted = [r for r in results if r.ks_drifted or r.psi_severity != "LOW"]
    return {
        "n_features_tested": len(results),
        "n_features_drifted": len(drifted),
        "drifted_features": [
            {
                "feature": r.feature,
                "ks_pvalue": round(r.ks_pvalue, 4),
                "psi_score": round(r.psi_score, 4),
                "psi_severity": r.psi_severity,
            }
            for r in drifted
        ],
        "max_psi": max((r.psi_score for r in results), default=0.0),
    }


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def _compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Population Stability Index.

    PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

    Uses equal-frequency binning on the training (expected) distribution.
    """
    if len(expected) < n_bins or len(actual) == 0:
        return 0.0

    # Equal-frequency bin edges from training distribution
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicates

    if len(bin_edges) < 2:
        return 0.0

    # Compute bin proportions
    expected_counts = np.histogram(expected, bins=bin_edges)[0]
    actual_counts = np.histogram(actual, bins=bin_edges)[0]

    # Convert to proportions with smoothing (avoid div-by-zero / log-of-zero)
    eps = 1e-4
    expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * len(actual_counts))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _psi_severity(psi: float) -> str:
    """Map PSI score to severity label."""
    if psi >= PSI_HIGH:
        return "HIGH"
    elif psi >= PSI_MODERATE:
        return "MODERATE"
    else:
        return "LOW"
