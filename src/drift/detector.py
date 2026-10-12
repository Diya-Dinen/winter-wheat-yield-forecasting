"""
Combined drift detector.

Merges univariate (KS + PSI) and multivariate (Mahalanobis + PCA) signals
into a single severity label (LOW / MODERATE / HIGH) with an explanation
of which variables drove the flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.drift.multivariate import MultivariateDriftResult, test_multivariate_drift
from src.drift.univariate import (
    UnivariateDriftResult,
    summarize_univariate,
    test_univariate_drift,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftResult:
    """Complete drift assessment for a test year."""
    severity: str                          # LOW / MODERATE / HIGH
    score: float                           # 0.0–1.0 composite score
    contributing_variables: list[str]      # Feature names that drove the flag
    univariate: list[UnivariateDriftResult] = field(default_factory=list)
    multivariate: MultivariateDriftResult | None = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for API / logging."""
        return {
            "severity": self.severity,
            "score": round(self.score, 4),
            "contributing_variables": self.contributing_variables,
            "univariate_summary": summarize_univariate(self.univariate),
            "multivariate": {
                "mahalanobis_distance": round(self.multivariate.mahalanobis_distance, 2),
                "mahalanobis_severity": self.multivariate.mahalanobis_severity,
                "pca_reconstruction_error": round(self.multivariate.pca_reconstruction_error, 4),
                "pca_reconstruction_percentile": round(self.multivariate.pca_reconstruction_percentile, 1),
            } if self.multivariate else None,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_drift(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
) -> DriftResult:
    """
    Run full drift detection pipeline: univariate + multivariate.

    Parameters
    ----------
    X_train : array-like, shape (n_train, n_features)
        Training feature matrix.
    X_test : array-like, shape (n_test, n_features)
        Test feature matrix (can be a single row).
    feature_names : list[str], optional
        Feature names for reporting.

    Returns
    -------
    DriftResult
        Combined drift assessment.
    """
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)

    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(1, -1)

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_arr.shape[1])]

    # --- Univariate tests ---
    uni_results = test_univariate_drift(X_train_arr, X_test_arr, feature_names)

    # --- Multivariate tests ---
    multi_result = test_multivariate_drift(X_train_arr, X_test_arr)

    # --- Combine into severity ---
    severity, score, contributors = _combine_scores(uni_results, multi_result)

    result = DriftResult(
        severity=severity,
        score=score,
        contributing_variables=contributors,
        univariate=uni_results,
        multivariate=multi_result,
    )

    logger.info(
        "Drift detection: severity=%s, score=%.3f, contributors=%s",
        severity, score, contributors,
    )

    return result


# ---------------------------------------------------------------------------
# Score combination logic
# ---------------------------------------------------------------------------

def _combine_scores(
    uni_results: list[UnivariateDriftResult],
    multi_result: MultivariateDriftResult,
) -> tuple[str, float, list[str]]:
    """
    Severity comes from the CALIBRATED multivariate p-value (empirical, leave-one-year-out).

    score = 1 - p_value (0..1; higher = more unusual).  Severity: p <= 0.025 HIGH,
    p <= 0.10 MODERATE, otherwise LOW.  The previous composite (PSI from one point,
    uncalibrated Mahalanobis, saturated PCA percentile) was always ~0.75 = HIGH.
    Contributing variables are the features with the largest |z| above the threshold.
    """
    p_value = multi_result.mahalanobis_pvalue
    severity = multi_result.mahalanobis_severity
    score = 0.0 if p_value is None else float(np.clip(1.0 - p_value, 0.0, 1.0))

    flagged = [r for r in uni_results if r.ks_drifted]
    flagged.sort(key=lambda r: r.ks_statistic, reverse=True)
    contributors = [r.feature for r in flagged][:5]

    return severity, score, contributors
