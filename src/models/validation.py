"""
Chronological and expanding-window cross-validation for yield forecasting.

Key constraint: NO information leakage from future to past.
- Train on years ≤ cutoff, test on cutoff + 1.
- Expanding window: repeat across multiple cutoffs, growing training set each time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import CV_MIN_TRAIN_YEARS, CV_STEP_YEARS
from src.models.baseline import BaseYieldModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Result for a single chronological train/test split."""
    cutoff_year: int
    test_year: int
    train_size: int
    y_true: float
    y_pred: float
    mae: float
    rmse: float
    r2: float | None  # R² is undefined for a single test point
    mape: float


@dataclass
class CVResult:
    """Aggregated result across all expanding-window splits."""
    model_name: str
    splits: list[SplitResult] = field(default_factory=list)

    @property
    def overall_mae(self) -> float:
        return np.mean([s.mae for s in self.splits])

    @property
    def overall_rmse(self) -> float:
        """RMSE computed from all individual predictions, not averaged."""
        residuals = np.array([s.y_true - s.y_pred for s in self.splits])
        return float(np.sqrt(np.mean(residuals ** 2)))

    @property
    def overall_r2(self) -> float:
        """R² across all test-year predictions."""
        y_true = np.array([s.y_true for s in self.splits])
        y_pred = np.array([s.y_pred for s in self.splits])
        if len(y_true) < 2:
            return float("nan")
        return float(r2_score(y_true, y_pred))

    @property
    def overall_mape(self) -> float:
        return np.mean([s.mape for s in self.splits])

    def to_dataframe(self) -> pd.DataFrame:
        """Convert splits to a DataFrame for logging/analysis."""
        records = []
        for s in self.splits:
            records.append({
                "model": self.model_name,
                "cutoff_year": s.cutoff_year,
                "test_year": s.test_year,
                "train_size": s.train_size,
                "y_true": s.y_true,
                "y_pred": s.y_pred,
                "mae": s.mae,
                "rmse": s.rmse,
                "mape": s.mape,
            })
        return pd.DataFrame(records)

    def summary(self) -> dict:
        """Summary metrics dictionary."""
        return {
            "model": self.model_name,
            "n_splits": len(self.splits),
            "mae": round(self.overall_mae, 2),
            "rmse": round(self.overall_rmse, 2),
            "r2": round(self.overall_r2, 4),
            "mape": round(self.overall_mape, 2),
        }


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    cutoff_year: int,
    feature_cols: list[str],
    target_col: str = "yield_bu_per_acre",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data chronologically at ``cutoff_year``.

    Returns (X_train, y_train, X_test, y_test) where:
    - train = years ≤ cutoff_year
    - test  = year == cutoff_year + 1
    """
    train_mask = df["year"] <= cutoff_year
    test_mask = df["year"] == cutoff_year + 1

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_col].values

    return X_train, y_train, X_test, y_test


def expanding_window_cv(
    model: BaseYieldModel,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "yield_bu_per_acre",
    min_train_years: int = CV_MIN_TRAIN_YEARS,
    step: int = CV_STEP_YEARS,
) -> CVResult:
    """
    Expanding-window chronological cross-validation.

    Iterates cutoff years from ``min_year + min_train_years`` to ``max_year - 1``
    in steps of ``step``, training on all years ≤ cutoff and testing on
    cutoff + 1.

    Parameters
    ----------
    model : BaseYieldModel
        A fresh model instance (will be re-fitted for each split).
    df : pd.DataFrame
        Must contain ``year``, feature columns, and ``target_col``.
    feature_cols : list[str]
        Column names to use as features.
    target_col : str
        Target variable column.
    min_train_years : int
        Minimum number of years in the training set before starting CV.
    step : int
        Step size between cutoff years.

    Returns
    -------
    CVResult
        Aggregated cross-validation results.
    """
    df = df.sort_values("year").reset_index(drop=True)
    years = sorted(df["year"].unique())
    min_year = years[0]
    max_year = years[-1]

    # First possible cutoff: ensure at least min_train_years in training set
    first_cutoff = min_year + min_train_years - 1

    # Generate cutoff years
    cutoffs = list(range(first_cutoff, max_year, step))

    # Ensure we include the last possible cutoff (max_year - 1)
    if cutoffs and cutoffs[-1] < max_year - 1:
        cutoffs.append(max_year - 1)

    result = CVResult(model_name=model.name)

    for cutoff in cutoffs:
        test_year = cutoff + 1

        # Skip if test year doesn't exist in data
        if test_year not in years:
            continue

        X_train, y_train, X_test, y_test = chronological_split(
            df, cutoff, feature_cols, target_col
        )

        if len(X_test) == 0 or len(X_train) == 0:
            continue

        # Fresh model for each split (no state leak between splits)
        # get_params() includes "model" key (the name) — strip it before passing to constructor
        from src.models.baseline import create_model
        params = {k: v for k, v in model.get_params().items() if k != "model"}
        fresh_model = create_model(model.name, **params)

        fresh_model.fit(X_train, y_train)
        y_pred = fresh_model.predict(X_test)

        # Metrics (single test point per split)
        y_t = float(y_test[0])
        y_p = float(y_pred[0])
        mae = abs(y_t - y_p)
        rmse = mae  # RMSE = MAE for a single point
        mape = (mae / abs(y_t)) * 100 if y_t != 0 else float("inf")

        split = SplitResult(
            cutoff_year=cutoff,
            test_year=test_year,
            train_size=len(y_train),
            y_true=y_t,
            y_pred=y_p,
            mae=mae,
            rmse=rmse,
            r2=None,  # Undefined for single point
            mape=mape,
        )
        result.splits.append(split)

        logger.info(
            "  Split: train ≤ %d (%d yrs) → test %d | "
            "actual=%.1f predicted=%.1f MAE=%.1f",
            cutoff, len(y_train), test_year, y_t, y_p, mae,
        )

    logger.info(
        "%s expanding-window CV: %d splits, MAE=%.2f, RMSE=%.2f, R²=%.4f",
        model.name, len(result.splits),
        result.overall_mae, result.overall_rmse, result.overall_r2,
    )

    return result


def run_all_baselines(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "yield_bu_per_acre",
) -> dict[str, CVResult]:
    """
    Run expanding-window CV for all three baseline models.

    Returns a dict mapping model name → CVResult.
    """
    from src.models.baseline import ALL_MODELS, create_model

    results = {}
    for name in ALL_MODELS:
        if name == "detrended_ridge":
            continue  # needs a 'year' column; evaluated in experiments/run_all.py
        logger.info("=" * 60)
        logger.info("Running expanding-window CV for: %s", name)
        logger.info("=" * 60)
        model = create_model(name)
        results[name] = expanding_window_cv(model, df, feature_cols, target_col)

    return results
