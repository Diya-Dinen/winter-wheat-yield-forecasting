"""
End-to-end inference pipeline.

Given a year (with live or historical weather), produces:
  - Point forecast + conformal interval
  - Drift assessment
  - Reliability signal
  - Adaptation decision
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.adaptation.strategy import adapt_prediction, AdaptationEvent
from src.data.feature_engineering import engineer_features
from src.data.fetch_noaa import fetch_historical_weather, fetch_live_weather
from src.data.preprocess import _assign_crop_year
from src.drift.detector import DriftResult, detect_drift
from src.models.baseline import BaseYieldModel
from src.models.conformal import ConformalPredictor, ConformalResult
from src.pipeline.train_pipeline import load_trained_artifacts
from config.settings import GROWING_SEASON_MONTHS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Complete inference output for a single year."""
    year: int
    predicted_yield: float
    lower_bound: float
    upper_bound: float
    drift_severity: str
    drift_score: float
    contributing_variables: list[str]
    reliability: str
    adaptation_applied: bool
    adaptation_strategy: str | None
    weather_source: str   # "observed" or "forecast" or "historical"
    lead_time_note: str

    def to_dict(self) -> dict:
        return {
            "year": self.year,
            "predicted_yield": round(self.predicted_yield, 2),
            "lower_bound": round(self.lower_bound, 2),
            "upper_bound": round(self.upper_bound, 2),
            "drift_severity": self.drift_severity,
            "drift_score": round(self.drift_score, 4),
            "contributing_variables": self.contributing_variables,
            "reliability": self.reliability,
            "adaptation_applied": self.adaptation_applied,
            "adaptation_strategy": self.adaptation_strategy,
            "weather_source": self.weather_source,
            "lead_time_note": self.lead_time_note,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(
    year: int,
    use_live_weather: bool = True,
    artifacts: dict | None = None,
    force_adapt: bool = False,
) -> InferenceResult:
    """
    Run the full inference pipeline for a given year.

    Parameters
    ----------
    year : int
        The crop year to forecast.
    use_live_weather : bool
        If True and the year is current/future, fetch live weather from NWS.
        If False, use historical data only.
    artifacts : dict, optional
        Pre-loaded training artifacts (model, conformal, df, feature_cols).
        If None, loads from disk.
    force_adapt : bool
        If True, force bias-correction adaptation regardless of drift level.

    Returns
    -------
    InferenceResult
    """
    # --- Load artifacts ---
    if artifacts is None:
        artifacts = load_trained_artifacts()

    model: BaseYieldModel = artifacts["model"]
    conformal: ConformalPredictor = artifacts["conformal"]
    feature_cols: list[str] = artifacts["feature_cols"]
    train_df: pd.DataFrame = artifacts["df"]
    target_col: str = artifacts["target_col"]

    # --- Get weather for the target year ---
    weather_source, lead_time_note, X_test = _prepare_test_features(
        year, feature_cols, use_live_weather
    )

    # --- Drift detection ---
    # Reference distribution = years BEFORE the requested year (never include the test year)
    ref = train_df[train_df["year"] < year]
    if len(ref) < 15:
        ref = train_df
    X_train = ref[feature_cols].values
    drift_result = detect_drift(X_train, X_test, feature_names=feature_cols)

    # --- Conformal prediction ---
    conformal_result = conformal.predict_single(X_test)

    # --- Adaptation ---
    adapted_result, adaptation_event = adapt_prediction(
        model=model,
        conformal_result=conformal_result,
        drift_result=drift_result,
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_year=year,
        X_test=X_test,
        force=force_adapt,
        recent_residuals=conformal.signed_residuals,
        model_name=getattr(model, "name", ""),
    )

    # --- Reliability ---
    reliability = _assess_reliability(drift_result, adapted_result)

    return InferenceResult(
        year=year,
        predicted_yield=adapted_result.point_prediction,
        lower_bound=adapted_result.lower_bound,
        upper_bound=adapted_result.upper_bound,
        drift_severity=drift_result.severity,
        drift_score=drift_result.score,
        contributing_variables=drift_result.contributing_variables,
        reliability=reliability,
        adaptation_applied=adaptation_event.adaptation_applied,
        adaptation_strategy=adaptation_event.strategy_used,
        weather_source=weather_source,
        lead_time_note=lead_time_note,
    )


def run_historical_inference(
    artifacts: dict | None = None,
) -> list[InferenceResult]:
    """
    Run inference for all historical years (for the dashboard history chart).
    """
    if artifacts is None:
        artifacts = load_trained_artifacts()

    train_df = artifacts["df"]
    years = sorted(train_df["year"].unique())

    results = []
    for year in years:
        try:
            result = run_inference(year, use_live_weather=False, artifacts=artifacts)
            results.append(result)
        except Exception as exc:
            logger.warning("Inference failed for year %d: %s", year, exc)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_test_features(
    year: int,
    feature_cols: list[str],
    use_live_weather: bool,
) -> tuple[str, str, np.ndarray]:
    """
    Prepare test features for a given year.

    Logic:
      - If a completed growing season exists in the engineered features file → use it
        (covers all historical years including the most recently completed season).
      - Else if use_live_weather=True → fetch live NWS observations and engineer on-the-fly
        (for the current in-progress season, or any year we lack stored features for).
      - Else → raise a clear error explaining why we can't produce a forecast.

    Returns (weather_source, lead_time_note, X_test).
    """
    import datetime

    current_year = datetime.date.today().year
    current_month = datetime.date.today().month

    # A crop-year N is "complete" once July of year N has passed
    # (growing season runs Oct year-1 through Jul year N)
    season_complete = (year < current_year) or (year == current_year and current_month > 7)

    # --- Path 1: check if we already have engineered features stored ---
    from config.settings import PROCESSED_DIR
    features_path = PROCESSED_DIR / "engineered_features.parquet"
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        row = features_df[features_df["year"] == year]
        if len(row) > 0:
            weather_source = "historical"
            if season_complete:
                lead_time_note = "Full growing season observed."
            else:
                lead_time_note = (
                    "Partial season — features based on available observed data. "
                    "Forecast will improve as more weather is recorded."
                )
            X_test = row[feature_cols].values
            return weather_source, lead_time_note, X_test

    # --- Path 2: stored features missing — try live weather ---
    if not season_complete and use_live_weather:
        weather_source = "forecast"

        if current_month <= 3:
            lead_time_note = (
                "Early season — mostly forecast data. "
                "Prediction uncertainty is HIGH due to limited observed weather."
            )
        elif current_month <= 5:
            lead_time_note = (
                "Mid season — partial observed data through spring. "
                "Moderate forecast uncertainty."
            )
        elif current_month <= 7:
            lead_time_note = (
                "Late season — most growing season weather observed. "
                "Prediction is relatively reliable."
            )
        else:
            lead_time_note = "Post-harvest — full season weather available."

        try:
            live_df = fetch_live_weather()
        except Exception as exc:
            raise ValueError(
                f"Live weather fetch failed for crop year {year}: {exc}. "
                "The NOAA NWS API may be temporarily unavailable. Try again later."
            ) from exc

        live_df = _assign_crop_year(live_df)
        live_df = live_df[live_df["year"] == year]
        live_df = live_df[live_df["date"].dt.month.isin(GROWING_SEASON_MONTHS)]

        if len(live_df) == 0:
            raise ValueError(
                f"No live weather observations found for crop year {year}. "
                f"The NWS station near Caldwell, KS has not yet reported data "
                f"for the Oct {year-1}–Jul {year} growing season."
            )

        features_df = engineer_features(live_df)
        row = features_df[features_df["year"] == year]
        if len(row) == 0:
            raise ValueError(
                f"Could not engineer features for crop year {year} from live weather. "
                "Not enough observations yet — try again later in the season."
            )
        X_test = row[feature_cols].values
        return weather_source, lead_time_note, X_test

    # --- Path 3: season complete but no stored features, and live weather disabled ---
    if season_complete:
        raise ValueError(
            f"No stored features found for crop year {year}. "
            f"The training data only covers years up to the last training run. "
            f"Re-run the training pipeline to include year {year}: "
            f"python -m src.pipeline.train_pipeline --force-refresh"
        )

    # --- Path 4: future year with live weather disabled ---
    raise ValueError(
        f"Year {year} is a future/in-progress growing season "
        f"(current date: {current_year}-{current_month:02d}). "
        f"Enable 'Use live weather' to forecast the current season, "
        f"or select a historical year (≤ {current_year - 1})."
    )


def _assess_reliability(
    drift_result: DriftResult,
    conformal_result: ConformalResult,
) -> str:
    """
    Assess prediction reliability based on drift + interval width.

    RELIABLE if:
      - Drift severity ≤ MODERATE AND
      - Interval width is reasonable (< 30% of typical yield)
    """
    TYPICAL_YIELD = 40.0  # approximate bu/acre for Sumner Co.
    width_ratio = conformal_result.interval_width / TYPICAL_YIELD

    if drift_result.severity == "HIGH":
        return "UNRELIABLE"
    elif width_ratio > 0.5:
        return "UNRELIABLE"
    elif drift_result.severity == "MODERATE" and width_ratio > 0.3:
        return "UNRELIABLE"
    else:
        return "RELIABLE"
