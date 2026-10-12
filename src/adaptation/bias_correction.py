"""
Bias correction adaptation strategy.

When drift is detected, computes the systematic bias from the most recent
K known years and applies an additive correction to both point forecast
and prediction interval.
"""

from __future__ import annotations

import logging

import numpy as np

from config.settings import BIAS_CORRECTION_WINDOW

logger = logging.getLogger(__name__)


def compute_bias_correction(
    recent_actuals: np.ndarray,
    recent_predictions: np.ndarray,
    window: int = BIAS_CORRECTION_WINDOW,
) -> float:
    """
    Compute the mean residual (actual − predicted) from the last K years.

    Parameters
    ----------
    recent_actuals : np.ndarray
        Actual yields for the most recent known years.
    recent_predictions : np.ndarray
        Model predictions for those same years.
    window : int
        Number of most-recent years to use.

    Returns
    -------
    float
        Additive bias correction (positive = model underpredicts).
    """
    n = min(window, len(recent_actuals))
    if n == 0:
        logger.warning("No recent data for bias correction — returning 0")
        return 0.0

    residuals = recent_actuals[-n:] - recent_predictions[-n:]
    bias = float(np.mean(residuals))

    logger.info(
        "Bias correction: mean residual over last %d years = %.2f bu/acre",
        n, bias,
    )
    return bias


def apply_bias_correction(
    point_prediction: float,
    lower_bound: float,
    upper_bound: float,
    bias: float,
) -> tuple[float, float, float]:
    """
    Apply additive bias correction to prediction + interval.

    Returns (corrected_point, corrected_lower, corrected_upper).
    """
    return (
        point_prediction + bias,
        lower_bound + bias,
        upper_bound + bias,
    )
