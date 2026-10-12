"""
Windowed retrain adaptation strategy.

When drift is detected, retrain the model on only the most recent W years
(sliding window), discarding older data that may no longer be representative.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.settings import RETRAIN_WINDOW_SIZES
from src.models.baseline import BaseYieldModel, create_model

logger = logging.getLogger(__name__)


def windowed_retrain(
    model: BaseYieldModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    window_size: int,
) -> BaseYieldModel:
    """
    Retrain a model using only the last ``window_size`` years before ``test_year``.

    Parameters
    ----------
    model : BaseYieldModel
        The original model (used for its name + hyperparameters).
    df : pd.DataFrame
        Full dataset with ``year``, features, and target.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column name.
    test_year : int
        The year being predicted (excluded from training).
    window_size : int
        Number of most-recent years to include in training.

    Returns
    -------
    BaseYieldModel
        A freshly-trained model on the windowed data.
    """
    # Select window: last W years before test_year
    cutoff_start = test_year - window_size
    mask = (df["year"] > cutoff_start) & (df["year"] < test_year)
    train_df = df[mask]

    if len(train_df) < 5:
        logger.warning(
            "Window too small (%d years) for window_size=%d — "
            "falling back to full training set",
            len(train_df), window_size,
        )
        mask = df["year"] < test_year
        train_df = df[mask]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    # Create a fresh model with same hyperparameters
    params = {k: v for k, v in model.get_params().items() if k != "model"}
    new_model = create_model(model.name, **params)
    new_model.fit(X_train, y_train)

    logger.info(
        "Windowed retrain: %s on years %d–%d (%d samples, window=%d)",
        model.name, train_df["year"].min(), train_df["year"].max(),
        len(train_df), window_size,
    )

    return new_model


def find_best_window(
    model: BaseYieldModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    test_year: int,
    window_sizes: list[int] | None = None,
) -> tuple[int, BaseYieldModel]:
    """
    Try multiple window sizes and pick the one with lowest recent-year MAE.

    Inner validation: for each window size, retrain on the window and evaluate
    on the last year of the window (pseudo-validation).

    Returns
    -------
    (best_window_size, best_model)
    """
    if window_sizes is None:
        window_sizes = RETRAIN_WINDOW_SIZES

    # Inner validation over the last N_INNER past years (past-only), not a single year.
    # A single-year choice is mostly noise.
    N_INNER = 8
    past_years = sorted(y for y in df["year"].unique() if y < test_year)
    inner_years = past_years[-N_INNER:]

    best_size = window_sizes[0]
    best_mae = float("inf")
    best_model = None

    for ws in window_sizes:
        errors = []
        for inner_year in inner_years:
            test_row = df[df["year"] == inner_year]
            if len(test_row) == 0:
                continue
            inner_model = windowed_retrain(
                model, df, feature_cols, target_col,
                test_year=int(inner_year), window_size=ws,
            )
            y_true = test_row[target_col].values[0]
            y_pred = inner_model.predict(test_row[feature_cols].values)[0]
            errors.append(abs(y_true - y_pred))
        if not errors:
            continue
        mae = float(np.mean(errors))
        logger.debug("  Window %d -> inner MAE=%.2f over %d years", ws, mae, len(errors))
        if mae < best_mae:
            best_mae = mae
            best_size = ws

    # Final retrain with best window size
    best_model = windowed_retrain(
        model, df, feature_cols, target_col,
        test_year=test_year, window_size=best_size,
    )

    logger.info("Best window size: %d (inner MAE=%.2f)", best_size, best_mae)
    return best_size, best_model
