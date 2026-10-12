"""
Drift-aware model ensemble.

Blends a full-history model with a recent-window model, weighting toward
the recent model when drift severity is high.
"""

from __future__ import annotations

import logging

import numpy as np

from src.models.baseline import BaseYieldModel

logger = logging.getLogger(__name__)


def ensemble_predict(
    full_model: BaseYieldModel,
    recent_model: BaseYieldModel,
    X_test: np.ndarray,
    drift_score: float,
    base_weight_recent: float = 0.3,
    drift_weight_scale: float = 0.35,
) -> np.ndarray:
    """
    Blend predictions from a full-history and a recent-window model.

    The weight of the recent model increases with drift severity:
        w_recent = min(1.0, base_weight + drift_weight_scale × drift_score)

    Parameters
    ----------
    full_model : BaseYieldModel
        Model trained on the full historical dataset.
    recent_model : BaseYieldModel
        Model trained on a recent window.
    X_test : np.ndarray
        Test features.
    drift_score : float
        Composite drift score (0–1).
    base_weight_recent : float
        Minimum weight for the recent model.
    drift_weight_scale : float
        How much drift_score increases the recent model's weight.

    Returns
    -------
    np.ndarray
        Blended predictions.
    """
    if X_test.ndim == 1:
        X_test = X_test.reshape(1, -1)

    pred_full = full_model.predict(X_test)
    pred_recent = recent_model.predict(X_test)

    w_recent = min(1.0, base_weight_recent + drift_weight_scale * drift_score)
    w_full = 1.0 - w_recent

    blended = w_full * pred_full + w_recent * pred_recent

    logger.info(
        "Ensemble: w_full=%.2f, w_recent=%.2f (drift_score=%.3f)",
        w_full, w_recent, drift_score,
    )

    return blended
