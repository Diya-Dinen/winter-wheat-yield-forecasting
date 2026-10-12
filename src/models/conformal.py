"""
Conformal prediction for yield forecasting (rolling split-conformal).

Why this replaces the MAPIE hold-out approach
---------------------------------------------
The previous version carved a calibration block out of the training years, so the
model was trained on only 60-80 % of the data, and with ~10 calibration points
the intervals were very wide.  Here the base model is trained on ALL available
years and the interval half-width comes from the model's own PAST out-of-sample
one-step-ahead errors (rolling-origin residuals).  No training data is spent on
calibration and no future information is used.

The guarantee is approximate: it assumes the recent residuals are roughly
exchangeable with the next one, which yields (trend/regime changes) can violate.
Empirical coverage is therefore always reported, never assumed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from config.settings import (
    CONFORMAL_ALPHA,
    CONFORMAL_MIN_RESIDUALS,
    CONFORMAL_WINDOW,
)
from src.models.baseline import BaseYieldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConformalResult:
    """Prediction with conformal interval."""
    point_prediction: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    alpha: float  # Nominal miscoverage rate


@dataclass
class CoverageStats:
    """Empirical coverage statistics, optionally stratified by drift."""
    n_total: int
    n_covered: int
    empirical_coverage: float
    nominal_coverage: float
    is_reliable: bool
    by_severity: dict[str, dict] | None = None  # severity -> {n, covered, coverage}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conformal_halfwidth(
    residuals: np.ndarray,
    alpha: float = CONFORMAL_ALPHA,
    window: int = CONFORMAL_WINDOW,
) -> float:
    """
    Finite-sample conformal quantile of |residuals| (most recent ``window`` values).

    q = the ceil((n+1)(1-alpha))-th smallest absolute residual; if that rank
    exceeds n the largest residual is used (interval is then conservative-ish
    but not guaranteed; callers check ``len(residuals)``).
    """
    r = np.abs(np.asarray(residuals, dtype=float))
    r = r[~np.isnan(r)][-window:]
    n = len(r)
    if n == 0:
        return float("nan")
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(k, n)
    return float(np.sort(r)[k - 1])


def make_interval(point: float, halfwidth: float, alpha: float) -> ConformalResult:
    """Symmetric interval around ``point``; yields cannot be negative."""
    lower = max(0.0, point - halfwidth)
    upper = point + halfwidth
    return ConformalResult(
        point_prediction=float(point),
        lower_bound=float(lower),
        upper_bound=float(upper),
        interval_width=float(upper - lower),
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ConformalPredictor:
    """
    Rolling split-conformal wrapper around a BaseYieldModel.

    Usage:
        1. ``fit(X_train, y_train)``        - trains the underlying model on ALL data
        2. ``set_residuals(signed_resid)``  - past out-of-sample errors (actual - predicted)
        3. ``predict(X_test)``              - ConformalResult(s) with interval
    """

    def __init__(self, model: BaseYieldModel, alpha: float = CONFORMAL_ALPHA):
        self.model = model
        self.alpha = alpha
        self.signed_residuals: np.ndarray = np.array([])
        self._is_calibrated = False

    # -- training / calibration -------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)
        logger.info("Conformal: underlying model trained on %d samples", len(y_train))

    def set_residuals(self, signed_residuals) -> None:
        """Store past out-of-sample residuals (actual - predicted), oldest first."""
        self.signed_residuals = np.asarray(signed_residuals, dtype=float)
        self._is_calibrated = len(self.signed_residuals) > 0

    @property
    def n_residuals(self) -> int:
        return int(len(self.signed_residuals))

    @property
    def has_enough_residuals(self) -> bool:
        return self.n_residuals >= CONFORMAL_MIN_RESIDUALS

    @property
    def halfwidth(self) -> float:
        return conformal_halfwidth(self.signed_residuals, self.alpha)

    # -- prediction -------------------------------------------------------
    def predict(self, X_test: np.ndarray) -> list[ConformalResult]:
        if not self._is_calibrated:
            raise RuntimeError(
                "ConformalPredictor has no residuals. Call set_residuals() first."
            )
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        hw = self.halfwidth
        return [make_interval(float(p), hw, self.alpha) for p in self.model.predict(X_test)]

    def predict_single(self, X_test: np.ndarray) -> ConformalResult:
        return self.predict(X_test)[0]


# ---------------------------------------------------------------------------
# Coverage tracking
# ---------------------------------------------------------------------------

def compute_coverage(
    y_true: np.ndarray,
    conformal_results: list[ConformalResult],
    drift_severities: list[str] | None = None,
    alpha: float = CONFORMAL_ALPHA,
) -> CoverageStats:
    """Empirical coverage of conformal intervals (optionally by drift severity)."""
    n = len(y_true)
    covered = sum(
        1 for yt, cr in zip(y_true, conformal_results)
        if cr.lower_bound <= yt <= cr.upper_bound
    )
    empirical = covered / n if n > 0 else 0.0
    nominal = 1 - alpha
    is_reliable = empirical >= 0.85 * nominal

    by_severity = None
    if drift_severities is not None:
        by_severity = {}
        for sev in ["LOW", "MODERATE", "HIGH"]:
            mask = [s == sev for s in drift_severities]
            n_sev = sum(mask)
            if n_sev == 0:
                continue
            covered_sev = sum(
                1 for m, yt, cr in zip(mask, y_true, conformal_results)
                if m and cr.lower_bound <= yt <= cr.upper_bound
            )
            by_severity[sev] = {
                "n": n_sev,
                "covered": covered_sev,
                "coverage": covered_sev / n_sev,
            }

    stats = CoverageStats(
        n_total=n, n_covered=covered, empirical_coverage=empirical,
        nominal_coverage=nominal, is_reliable=is_reliable, by_severity=by_severity,
    )
    logger.info(
        "Coverage: %d/%d = %.1f%% (nominal %.0f%%) -> %s",
        covered, n, empirical * 100, nominal * 100,
        "RELIABLE" if is_reliable else "UNRELIABLE",
    )
    return stats
