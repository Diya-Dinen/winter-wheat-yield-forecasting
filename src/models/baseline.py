"""
Baseline yield forecasting models.

Provides a uniform interface for three model families:
  - Ridge Regression (linear baseline with regularization)
  - Random Forest (nonlinear, handles interactions)
  - XGBoost (gradient-boosted trees — typically strongest)

All models share a common API: ``.fit(X, y)``, ``.predict(X)``, and
``.get_params()`` for experiment logging.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseYieldModel(ABC):
    """Common interface for all yield forecasting models."""

    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return point predictions."""

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters for logging."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


# ---------------------------------------------------------------------------
# Ridge Regression
# ---------------------------------------------------------------------------

class RidgeModel(BaseYieldModel):
    """
    Ridge regression with StandardScaler.

    The scaler is critical because Ridge penalizes coefficient magnitudes,
    so features must be on comparable scales.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, random_state=RANDOM_STATE)),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        self._pipeline.fit(X, y)
        logger.info("Ridge trained on %d samples, alpha=%.2f", len(y), self.alpha)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._pipeline.predict(X)

    def get_params(self) -> dict[str, Any]:
        return {"model": self.name, "alpha": self.alpha}


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel(BaseYieldModel):
    """
    Random Forest regressor.

    No scaling needed — tree-based models are invariant to feature scale.
    """

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 3,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        self._model.fit(X, y)
        logger.info(
            "Random Forest trained on %d samples, %d trees",
            len(y), self.n_estimators,
        )

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        """Access underlying feature importances for diagnostics."""
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostModel(BaseYieldModel):
    """
    XGBoost gradient-boosted trees.

    The "main" model that the drift/adaptation layer wraps around,
    since it's typically the strongest baseline for tabular data.
    """

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        self._model.fit(X, y)
        logger.info(
            "XGBoost trained on %d samples, %d rounds, lr=%.3f",
            len(y), self.n_estimators, self.learning_rate,
        )

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        """Access underlying feature importances for diagnostics."""
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# Detrended Ridge (trend + weather-driven residual)
# ---------------------------------------------------------------------------

class DetrendedRidgeModel(BaseYieldModel):
    """
    Linear time trend + RidgeCV on the trend residuals.

    IMPORTANT: the LAST column of X must be the crop year.  Callers use
    ``feature_cols + ["year"]`` for this model.  Yields have a technology trend
    (corr(year, yield) ~ 0.6 here); modelling it explicitly lets the weather
    features explain only the remaining year-to-year variation.
    """

    name = "detrended_ridge"

    def __init__(self, alphas: tuple = (10.0, 100.0, 1000.0)):
        self.alphas = tuple(alphas)
        self._coef = (0.0, 0.0)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=self.alphas)),
        ])

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        years = X[:, -1]
        self._coef = tuple(np.polyfit(years, y, 1))
        residual = y - np.polyval(self._coef, years)
        self._pipeline.fit(X[:, :-1], residual)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        trend = np.polyval(self._coef, X[:, -1])
        return trend + self._pipeline.predict(X[:, :-1])

    def get_params(self) -> dict[str, Any]:
        return {"model": self.name, "alphas": self.alphas}


# ---------------------------------------------------------------------------
# Naive baselines (computed from the yield history only)
# ---------------------------------------------------------------------------

def naive_forecasts(years: np.ndarray, y: np.ndarray, test_year: int) -> dict[str, float]:
    """
    Naive one-step-ahead forecasts for ``test_year`` using ONLY past yields.

    Every model must be compared against these; a model that cannot beat
    them has no demonstrated skill.
    """
    years = np.asarray(years, dtype=float)
    y = np.asarray(y, dtype=float)
    return {
        "naive_expanding_mean": float(np.mean(y)),
        "naive_rolling5_mean": float(np.mean(y[-5:])),
        "naive_rolling10_mean": float(np.mean(y[-10:])),
        "naive_last_year": float(y[-1]),
        "naive_linear_trend": float(np.polyval(np.polyfit(years, y, 1), test_year)),
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ALL_MODELS: dict[str, type[BaseYieldModel]] = {
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "detrended_ridge": DetrendedRidgeModel,
}


def create_model(name: str, **kwargs) -> BaseYieldModel:
    """
    Factory function to create a model by name.

    Parameters
    ----------
    name : str
        One of ``'ridge'``, ``'random_forest'``, ``'xgboost'``.
    **kwargs
        Passed to the model constructor.
    """
    cls = ALL_MODELS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(ALL_MODELS.keys())}"
        )
    return cls(**kwargs)
