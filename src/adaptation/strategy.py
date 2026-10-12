"""
Adaptation strategy orchestrator.

Decides whether adaptation is needed (based on drift severity), selects
the best strategy (bias correction vs. windowed retrain vs. ensemble blend),
and logs every adaptation event.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    ADAPTATION_TRIGGER_SEVERITY,
    BIAS_CORRECTION_WINDOW,
    EXPERIMENTS_DIR,
)
from src.adaptation.bias_correction import apply_bias_correction, compute_bias_correction
from src.adaptation.windowed_retrain import find_best_window
from src.drift.detector import DriftResult
from src.models.baseline import BaseYieldModel
from src.models.conformal import ConformalResult

logger = logging.getLogger(__name__)

# Severity ordering for comparison
_SEVERITY_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdaptationEvent:
    """Record of a single adaptation decision."""
    year: int
    drift_severity: str
    drift_score: float
    strategy_used: str  # "none" | "bias_correction" | "windowed_retrain" | "ensemble"
    adaptation_applied: bool
    original_prediction: float
    adapted_prediction: float | None
    original_lower: float
    adapted_lower: float | None
    original_upper: float
    adapted_upper: float | None
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "year": self.year,
            "drift_severity": self.drift_severity,
            "drift_score": round(self.drift_score, 4),
            "strategy_used": self.strategy_used,
            "adaptation_applied": self.adaptation_applied,
            "original_prediction": round(self.original_prediction, 2),
            "adapted_prediction": round(self.adapted_prediction, 2) if self.adapted_prediction is not None else None,
            "original_interval": [round(self.original_lower, 2), round(self.original_upper, 2)],
            "adapted_interval": (
                [round(self.adapted_lower, 2), round(self.adapted_upper, 2)]
                if self.adapted_lower is not None else None
            ),
            "details": self.details,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Adaptation log
# ---------------------------------------------------------------------------

_adaptation_log: list[AdaptationEvent] = []


def get_adaptation_log() -> list[AdaptationEvent]:
    """Return the current in-memory adaptation log."""
    return _adaptation_log.copy()


def save_adaptation_log(path: Path | None = None) -> Path:
    """Save adaptation log to JSON."""
    if path is None:
        path = EXPERIMENTS_DIR / "adaptation_log.json"
    with open(path, "w") as f:
        json.dump([e.to_dict() for e in _adaptation_log], f, indent=2)
    logger.info("Saved adaptation log (%d events) to %s", len(_adaptation_log), path)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def should_adapt(drift_result: DriftResult) -> bool:
    """Check if drift severity meets the adaptation trigger threshold."""
    trigger_level = _SEVERITY_ORDER.get(ADAPTATION_TRIGGER_SEVERITY, 1)
    current_level = _SEVERITY_ORDER.get(drift_result.severity, 0)
    return current_level >= trigger_level


def adapt_prediction(
    model: BaseYieldModel,
    conformal_result: ConformalResult,
    drift_result: DriftResult,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    X_test: np.ndarray,
    force: bool = False,
    recent_residuals: np.ndarray | None = None,
    model_name: str = "",
) -> tuple[ConformalResult, AdaptationEvent]:
    """
    Apply adaptation if drift warrants it.

    Strategy selection:
      - If MODERATE drift: try bias correction first (lightweight)
      - If HIGH drift: try windowed retrain (heavyweight), fall back to ensemble
      - If ``force=True``: apply bias correction regardless of drift level

    Parameters
    ----------
    model : BaseYieldModel
        The currently-trained model.
    conformal_result : ConformalResult
        Original prediction + interval.
    drift_result : DriftResult
        Drift assessment for this test year.
    df : pd.DataFrame
        Full historical dataset.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column name.
    test_year : int
        Year being predicted.
    X_test : np.ndarray
        Test features.
    force : bool
        If True, apply bias correction even when drift is below the threshold.

    Returns
    -------
    (adapted_conformal_result, adaptation_event)
    """
    original = conformal_result

    if not force and not should_adapt(drift_result):
        # No adaptation needed
        event = AdaptationEvent(
            year=test_year,
            drift_severity=drift_result.severity,
            drift_score=drift_result.score,
            strategy_used="none",
            adaptation_applied=False,
            original_prediction=original.point_prediction,
            adapted_prediction=None,
            original_lower=original.lower_bound,
            adapted_lower=None,
            original_upper=original.upper_bound,
            adapted_upper=None,
        )
        event.model = model_name
        _adaptation_log.append(event)
        return original, event

    # --- Determine strategy based on severity (or force) ---
    if force or drift_result.severity == "MODERATE":
        strategy, result = _try_bias_correction(
            model, original, df, feature_cols, target_col, test_year, recent_residuals
        )
    else:  # HIGH
        strategy, result = _try_windowed_retrain(
            model, original, df, feature_cols, target_col, test_year, X_test
        )
        # Fallback to bias correction if retrain didn't help
        if result is None:
            strategy, result = _try_bias_correction(
                model, original, df, feature_cols, target_col, test_year, recent_residuals
            )

    if result is None:
        result = original
        strategy = "none"
        adapted = False
    else:
        adapted = True

    event = AdaptationEvent(
        year=test_year,
        drift_severity=drift_result.severity,
        drift_score=drift_result.score,
        strategy_used=strategy,
        adaptation_applied=adapted,
        original_prediction=original.point_prediction,
        adapted_prediction=result.point_prediction if adapted else None,
        original_lower=original.lower_bound,
        adapted_lower=result.lower_bound if adapted else None,
        original_upper=original.upper_bound,
        adapted_upper=result.upper_bound if adapted else None,
    )
    event.model = model_name
    _adaptation_log.append(event)

    if adapted:
        logger.info(
            "Adaptation (%s): %.1f → %.1f bu/acre for year %d",
            strategy, original.point_prediction, result.point_prediction, test_year,
        )

    return result if adapted else original, event


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _try_bias_correction(
    model: BaseYieldModel,
    original: ConformalResult,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    recent_residuals: np.ndarray | None = None,
) -> tuple[str, ConformalResult | None]:
    """
    Bias correction from recent OUT-OF-SAMPLE residuals (actual - predicted).

    The model is trained on all years < test_year, so re-predicting those years would
    give in-sample residuals (~0, meaningless).  Callers must pass the past
    out-of-sample residuals; without them no correction is applied.  The mean is
    shrunk toward 0 (n / (n + 2)) because a handful of residuals is a noisy estimate.
    """
    if recent_residuals is None or len(recent_residuals) == 0:
        return "bias_correction", None

    res = np.asarray(recent_residuals, dtype=float)[-BIAS_CORRECTION_WINDOW:]
    bias = float(np.mean(res)) * len(res) / (len(res) + 2.0)

    corrected_point, corrected_lower, corrected_upper = apply_bias_correction(
        original.point_prediction,
        original.lower_bound,
        original.upper_bound,
        bias,
    )

    return "bias_correction", ConformalResult(
        point_prediction=corrected_point,
        lower_bound=max(0.0, corrected_lower),
        upper_bound=corrected_upper,
        interval_width=corrected_upper - max(0.0, corrected_lower),
        alpha=original.alpha,
    )


def _try_windowed_retrain(
    model: BaseYieldModel,
    original: ConformalResult,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    X_test: np.ndarray,
) -> tuple[str, ConformalResult | None]:
    """Retrain on a recent window and re-predict."""
    try:
        best_size, retrained_model = find_best_window(
            model, df, feature_cols, target_col, test_year,
        )
    except Exception as exc:
        logger.warning("Windowed retrain failed: %s", exc)
        return "windowed_retrain", None

    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    new_pred = float(retrained_model.predict(X_test)[0])

    # Shift interval by prediction difference
    delta = new_pred - original.point_prediction
    new_lower = original.lower_bound + delta
    new_upper = original.upper_bound + delta

    return "windowed_retrain", ConformalResult(
        point_prediction=new_pred,
        lower_bound=new_lower,
        upper_bound=new_upper,
        interval_width=new_upper - new_lower,
        alpha=original.alpha,
    )
