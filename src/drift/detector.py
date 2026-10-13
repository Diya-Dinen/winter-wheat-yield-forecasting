"""
Combined drift detector.

Merges univariate (z-score feature deviation count) and multivariate
(Mahalanobis + PCA, dynamic percentile thresholds) signals into a single
severity label (LOW / MODERATE / HIGH) with an explanation of which
variables drove the flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config.settings import DRIFT_FEATURES_HIGH, DRIFT_FEATURES_MODERATE
from src.drift.multivariate import MultivariateDriftResult, test_multivariate_drift
from src.drift.univariate import (
    UnivariateDriftResult,
    count_drifted_features,
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
    train_years: np.ndarray | None = None,
    current_year: int | None = None,
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
    train_years, current_year : optional
        When both are given, only rows with year < current_year form the baseline.

    Returns
    -------
    DriftResult
        Combined drift assessment.
    """
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)

    if X_test_arr.ndim == 1:
        X_test_arr = X_test_arr.reshape(1, -1)

    if train_years is not None and current_year is not None:
        X_train_arr = X_train_arr[np.asarray(train_years) < current_year]

    if feature_names is None:
        if isinstance(X_train, pd.DataFrame):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X_train_arr.shape[1])]

    # No usable history -> nothing to compare against
    if X_train_arr.shape[0] < 2:
        return DriftResult(severity="LOW", score=0.0, contributing_variables=[],
                           details={"reason": "insufficient historical data"})

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
    Map drift severity to the number of features beyond 2.5 sigma.

      LOW      : 0-2 features beyond 2.5 sigma AND normal multivariate distance
      MODERATE : 3-5 features beyond 2.5 sigma (or multivariate >= 95th percentile)
      HIGH     : >= 6 features beyond 2.5 sigma OR an extreme multivariate
                 anomaly (Mahalanobis / PCA >= 99th percentile of history)

    score (0..1) = max(n_features / 6, multivariate level) so it rises
    monotonically with severity.  Contributing variables are the features with
    the largest |z| above the threshold.
    """
    n_out = count_drifted_features(uni_results)

    multi_sev = "LOW"
    extreme = False
    if multi_result is not None:
        extreme = multi_result.extreme_anomaly
        sevs = {multi_result.mahalanobis_severity, multi_result.pca_severity}
        multi_sev = "HIGH" if extreme else ("MODERATE" if "MODERATE" in sevs else "LOW")

    if n_out >= DRIFT_FEATURES_HIGH or extreme:
        severity = "HIGH"
    elif n_out >= DRIFT_FEATURES_MODERATE or multi_sev == "MODERATE":
        severity = "MODERATE"
    else:
        severity = "LOW"

    multi_level = {"LOW": 0.0, "MODERATE": 0.5, "HIGH": 1.0}[multi_sev]
    score = float(np.clip(max(n_out / DRIFT_FEATURES_HIGH, multi_level), 0.0, 1.0))

    flagged = [r for r in uni_results if r.ks_drifted]
    flagged.sort(key=lambda r: r.ks_statistic, reverse=True)
    contributors = [r.feature for r in flagged][:5]

    return severity, score, contributors
