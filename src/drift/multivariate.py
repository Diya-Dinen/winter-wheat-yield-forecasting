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
    DRIFT_P_HIGH,
    DRIFT_P_MODERATE,
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

    # --- Mahalanobis ---
    maha_dist, maha_p = _calibrated_mahalanobis(X_train, X_test)
    maha_sev = _pvalue_severity(maha_p)

    # --- PCA reconstruction ---
    pca_error, pca_percentile = _pca_reconstruction_error(X_train, X_test)

    # --- MMD ---
    mmd = _mmd_rbf(X_train, X_test) if X_test.shape[0] >= 2 else None

    result = MultivariateDriftResult(
        mahalanobis_distance=maha_dist,
        mahalanobis_severity=maha_sev,
        pca_reconstruction_error=pca_error,
        pca_reconstruction_percentile=pca_percentile,
        mmd_score=mmd,
        mahalanobis_pvalue=maha_p,
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
    Compute the Mahalanobis distance of the test centroid from the
    training distribution centroid, using the training covariance.
    """
    # Handle NaN by imputing column means
    train_means = np.nanmean(X_train, axis=0)
    X_train_clean = np.where(np.isnan(X_train), train_means, X_train)
    X_test_clean = np.where(np.isnan(X_test), train_means, X_test)

    mu = X_train_clean.mean(axis=0)
    cov = np.cov(X_train_clean, rowvar=False)

    # Regularize covariance to ensure invertibility
    cov += np.eye(cov.shape[0]) * 1e-6

    test_point = X_test_clean.mean(axis=0)  # centroid of test sample(s)

    try:
        cov_inv = np.linalg.inv(cov)
        dist = float(scipy_mahalanobis(test_point, mu, cov_inv))
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix not invertible — using Euclidean fallback")
        dist = float(np.linalg.norm(test_point - mu))

    return dist


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
) -> tuple[float, float | None]:
    """
    Shrinkage Mahalanobis distance of the test centroid, plus an EMPIRICAL p-value.

    The raw distance has no fixed scale with 40 features and ~25-75 samples (the old
    fixed thresholds 3/5 were always exceeded).  Instead, the distance is compared with
    a leave-one-year-out null: each training year's distance to the rest of the
    training years.  p = (1 + #{null >= d}) / (n + 1).  Returns p=None if there are
    fewer than MIN_DRIFT_REF_YEARS reference years (cannot calibrate).
    """
    train_means = np.nanmean(X_train, axis=0)
    X_tr = np.where(np.isnan(X_train), train_means, X_train)
    X_te = np.where(np.isnan(X_test), train_means, X_test)

    dist = _shrunk_mahalanobis(X_tr, X_te.mean(axis=0))
    n = X_tr.shape[0]
    if n < MIN_DRIFT_REF_YEARS:
        return dist, None

    null = np.array([
        _shrunk_mahalanobis(np.delete(X_tr, i, axis=0), X_tr[i]) for i in range(n)
    ])
    p_value = float((1 + np.sum(null >= dist)) / (n + 1))
    return dist, p_value


def _pvalue_severity(p_value: float | None) -> str:
    """Map an empirical drift p-value to a severity label."""
    if p_value is None:
        return "LOW"  # not enough reference years to judge
    if p_value <= DRIFT_P_HIGH:
        return "HIGH"
    if p_value <= DRIFT_P_MODERATE:
        return "MODERATE"
    return "LOW"


# ---------------------------------------------------------------------------
# PCA reconstruction error
# ---------------------------------------------------------------------------

def _pca_reconstruction_error(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[float, float]:
    """
    Fit PCA on training data, reconstruct test data, measure residual.

    Returns (mean_squared_error, percentile_rank).
    The percentile rank indicates where the test error falls relative
    to the distribution of training reconstruction errors.
    """
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(
        np.where(np.isnan(X_train), np.nanmean(X_train, axis=0), X_train)
    )
    X_test_scaled = scaler.transform(
        np.where(np.isnan(X_test), np.nanmean(X_train, axis=0), X_test)
    )

    # Fit PCA retaining specified variance
    n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)

    # Find number of components for target variance
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, PCA_VARIANCE_RETAINED) + 1)
    n_keep = min(n_keep, n_components)

    # Reconstruct using truncated PCA
    pca_trunc = PCA(n_components=n_keep)
    X_train_proj = pca_trunc.fit_transform(X_train_scaled)
    X_train_recon = pca_trunc.inverse_transform(X_train_proj)

    X_test_proj = pca_trunc.transform(X_test_scaled)
    X_test_recon = pca_trunc.inverse_transform(X_test_proj)

    # MSE for each sample
    train_errors = np.mean((X_train_scaled - X_train_recon) ** 2, axis=1)
    test_error = float(np.mean((X_test_scaled - X_test_recon) ** 2))

    # Percentile rank of test error in training error distribution
    percentile = float(np.mean(train_errors <= test_error) * 100)

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
