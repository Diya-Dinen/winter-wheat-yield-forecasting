"""
Multivariate temporal drift detection.

Weather features are correlated (hot springs are often dry), so univariate
tests can miss joint distribution shifts.  This module provides:
  - Mahalanobis distance from training centroid
  - PCA reconstruction error
  - Maximum Mean Discrepancy (MMD) with RBF kernel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import mahalanobis as scipy_mahalanobis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sklearn.covariance import LedoitWolf

from config.settings import (
    DRIFT_PERCENTILE_HIGH,
    DRIFT_PERCENTILE_MODERATE,
    MIN_DRIFT_REF_YEARS,
    PCA_VARIANCE_RETAINED,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultivariateDriftResult:
    """Results from multivariate drift detection."""
    mahalanobis_distance: float
    mahalanobis_severity: str
    pca_reconstruction_error: float
    pca_reconstruction_percentile: float  # Where test error falls relative to training
    mmd_score: float | None  # None if not enough test samples
    mahalanobis_pvalue: float | None = None  # empirical p-value vs leave-one-out null
    mahalanobis_threshold_moderate: float | None = None  # dynamic 95th percentile of historical distances
    mahalanobis_threshold_high: float | None = None      # dynamic 99th percentile of historical distances
    pca_severity: str = "LOW"                            # from PCA percentile vs the same baseline
    extreme_anomaly: bool = False                        # Mahalanobis or PCA beyond the 99th percentile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def test_multivariate_drift(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> MultivariateDriftResult:
    """
    Run multivariate drift detection on the test sample(s).

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training feature matrix.
    X_test : np.ndarray, shape (n_test, n_features)
        Test feature matrix (can be a single row).

    Returns
    -------
    MultivariateDriftResult
        Combined multivariate drift scores.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)

    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    # --- Mahalanobis (dynamic percentile thresholds from historical distribution) ---
    maha_dist, maha_p, thr_mod, thr_high = _calibrated_mahalanobis(X_train, X_test)
    maha_sev = _distance_severity(maha_dist, thr_mod, thr_high)

    # --- PCA reconstruction (percentile vs leave-one-out historical errors) ---
    pca_error, pca_percentile = _pca_reconstruction_error(X_train, X_test)
    pca_sev = _percentile_severity(pca_percentile, maha_p is not None)

    # --- MMD ---
    mmd = _mmd_rbf(X_train, X_test) if X_test.shape[0] >= 2 else None

    result = MultivariateDriftResult(
        mahalanobis_distance=maha_dist,
        mahalanobis_severity=maha_sev,
        pca_reconstruction_error=pca_error,
        pca_reconstruction_percentile=pca_percentile,
        mmd_score=mmd,
        mahalanobis_pvalue=maha_p,
        mahalanobis_threshold_moderate=thr_mod,
        mahalanobis_threshold_high=thr_high,
        pca_severity=pca_sev,
        extreme_anomaly=(maha_sev == "HIGH" or pca_sev == "HIGH"),
    )

    logger.info(
        "Multivariate drift: Mahalanobis=%.2f (%s), PCA_error=%.4f (p%.0f), MMD=%s",
        maha_dist, maha_sev, pca_error, pca_percentile,
        f"{mmd:.4f}" if mmd is not None else "N/A",
    )

    return result


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------

def _mahalanobis_distance(X_train: np.ndarray, X_test: np.ndarray) -> float:
    """
    Mahalanobis distance of the test centroid from the training centroid.

    Robust version: standardises features, imputes NaN/inf, shrinks the
    covariance toward the identity and uses a pseudo-inverse, so it never
    fails when n_samples < n_features (singular covariance).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)
    if X_train.shape[0] < 2:
        return 0.0

    # Replace inf with NaN, then impute NaN with training column means
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    X_test = np.where(np.isfinite(X_test), X_test, np.nan)
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    X_train = np.where(np.isnan(X_train), col_means, X_train)
    X_test = np.where(np.isnan(X_test), col_means, X_test)

    # Standardise (constant columns get sd = 1)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z_train = (X_train - mu) / sd
    z_test = (X_test.mean(axis=0) - mu) / sd

    # Shrunk covariance -> always positive definite
    p = Z_train.shape[1]
    cov = np.atleast_2d(np.cov(Z_train, rowvar=False))
    shrink = 0.1
    cov = (1 - shrink) * cov + shrink * np.eye(p) + 1e-6 * np.eye(p)
    cov_inv = np.linalg.pinv(cov)

    d2 = float(z_test @ cov_inv @ z_test)
    if not np.isfinite(d2):
        logger.warning("Non-finite Mahalanobis value - using Euclidean fallback")
        return float(np.linalg.norm(z_test))
    return float(np.sqrt(max(d2, 0.0)))

def _shrunk_mahalanobis(ref: np.ndarray, point: np.ndarray) -> float:
    """Mahalanobis distance with a Ledoit-Wolf shrunk covariance on standardized data."""
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0) + 1e-9
    z_ref = (ref - mu) / sd
    z = (point - mu) / sd
    prec = LedoitWolf().fit(z_ref).precision_
    return float(np.sqrt(max(z @ prec @ z, 0.0)))


def _calibrated_mahalanobis(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[float, float | None, float | None, float | None]:
    """
    Shrinkage Mahalanobis distance of the test centroid plus DYNAMIC thresholds.

    The raw distance has no fixed scale with ~40 features and ~25-75 samples, so
    fixed cut-offs (3.0 / 5.0) are meaningless.  Instead each historical year's
    distance to the remaining historical years (leave-one-out) forms the null
    distribution; its 95th and 99th percentiles are the MODERATE / HIGH thresholds.

    Returns (distance, empirical_p_value, threshold_moderate, threshold_high).
    Thresholds / p-value are None if fewer than MIN_DRIFT_REF_YEARS reference rows.
    """
    train_means = np.nanmean(X_train, axis=0)
    train_means = np.where(np.isfinite(train_means), train_means, 0.0)
    X_tr = np.where(np.isfinite(X_train), X_train, train_means)
    X_te = np.where(np.isfinite(X_test), X_test, train_means)

    dist = _shrunk_mahalanobis(X_tr, X_te.mean(axis=0))
    n = X_tr.shape[0]
    if n < MIN_DRIFT_REF_YEARS:
        return dist, None, None, None

    null = np.array([
        _shrunk_mahalanobis(np.delete(X_tr, i, axis=0), X_tr[i]) for i in range(n)
    ])
    p_value = float((1 + np.sum(null >= dist)) / (n + 1))
    thr_mod = float(np.percentile(null, DRIFT_PERCENTILE_MODERATE))
    thr_high = float(np.percentile(null, DRIFT_PERCENTILE_HIGH))
    return dist, p_value, thr_mod, thr_high


def _distance_severity(
    dist: float, thr_mod: float | None, thr_high: float | None
) -> str:
    """Map a Mahalanobis distance to severity using the dynamic percentile thresholds."""
    if thr_mod is None or thr_high is None:
        return "LOW"  # not enough reference years to judge
    if dist >= thr_high:
        return "HIGH"
    if dist >= thr_mod:
        return "MODERATE"
    return "LOW"


def _percentile_severity(percentile: float, calibrated: bool) -> str:
    """Map the PCA reconstruction percentile (0-100) to severity (same 95 / 99 cut-offs)."""
    if not calibrated:
        return "LOW"
    if percentile >= DRIFT_PERCENTILE_HIGH:
        return "HIGH"
    if percentile >= DRIFT_PERCENTILE_MODERATE:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# PCA reconstruction error
# ---------------------------------------------------------------------------

def _pca_fit_errors(
    X_fit: np.ndarray, X_eval: np.ndarray
) -> np.ndarray:
    """Fit scaler + truncated PCA on X_fit; return per-row reconstruction MSE of X_eval."""
    scaler = StandardScaler()
    Xf = scaler.fit_transform(X_fit)
    Xe = scaler.transform(X_eval)

    n_components = min(Xf.shape[0], Xf.shape[1])
    full = PCA(n_components=n_components).fit(Xf)
    cumvar = np.cumsum(full.explained_variance_ratio_)
    n_keep = min(int(np.searchsorted(cumvar, PCA_VARIANCE_RETAINED) + 1), n_components)

    pca = PCA(n_components=n_keep).fit(Xf)
    recon = pca.inverse_transform(pca.transform(Xe))
    return np.mean((Xe - recon) ** 2, axis=1)


def _pca_reconstruction_error(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[float, float]:
    """
    PCA reconstruction error of the test sample and its percentile rank (0-100)
    within the historical baseline.

    The baseline is a LEAVE-ONE-OUT distribution (each historical year is
    reconstructed by a PCA fitted on the other years).  In-sample training errors
    are optimistically small, so the test error would always land at the 100th
    percentile.  The percentile is then compared with the same dynamic 95th / 99th
    cut-offs used for the Mahalanobis distance.
    """
    col_means = np.nanmean(X_train, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    X_tr = np.where(np.isfinite(X_train), X_train, col_means)
    X_te = np.where(np.isfinite(X_test), X_test, col_means)

    test_error = float(_pca_fit_errors(X_tr, X_te).mean())

    n = X_tr.shape[0]
    if n < MIN_DRIFT_REF_YEARS:
        return test_error, 0.0

    loo_errors = np.array([
        _pca_fit_errors(np.delete(X_tr, i, axis=0), X_tr[i:i + 1])[0] for i in range(n)
    ])
    percentile = float(np.mean(loo_errors <= test_error) * 100)
    return test_error, percentile


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) with RBF kernel
# ---------------------------------------------------------------------------

def _mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float | None = None,
) -> float:
    """
    Compute the squared MMD between two samples using an RBF kernel.

    MMD² = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    Parameters
    ----------
    X, Y : np.ndarray
        Sample matrices.
    gamma : float, optional
        RBF kernel bandwidth (1/2σ²). If None, uses the median heuristic.
    """
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    Y = np.where(np.isnan(Y), np.nanmean(X, axis=0), Y)

    if gamma is None:
        # Median heuristic for bandwidth
        combined = np.vstack([X, Y])
        dists = np.linalg.norm(
            combined[:, None, :] - combined[None, :, :], axis=2
        )
        median_dist = np.median(dists[dists > 0])
        gamma = 1.0 / (2 * median_dist ** 2 + 1e-10)

    def rbf_kernel(A, B):
        sq_dists = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=2)
        return np.exp(-gamma * sq_dists)

    K_XX = rbf_kernel(X, X)
    K_YY = rbf_kernel(Y, Y)
    K_XY = rbf_kernel(X, Y)

    n = X.shape[0]
    m = Y.shape[0]

    # Unbiased estimator (exclude diagonal for XX and YY)
    mmd_sq = (
        (K_XX.sum() - np.trace(K_XX)) / (n * (n - 1) + 1e-10)
        + (K_YY.sum() - np.trace(K_YY)) / (m * (m - 1) + 1e-10)
        - 2 * K_XY.mean()
    )

    return float(max(0.0, mmd_sq))
