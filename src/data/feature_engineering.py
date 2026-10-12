"""
Feature engineering for winter wheat yield forecasting.

Computes agronomically-meaningful features from daily weather observations,
grouped by winter wheat growth stages:

  - **Fall** (Oct–Nov): planting + emergence
  - **Winter** (Dec–Feb): dormancy + vernalization
  - **Spring** (Mar–May): tillering + stem elongation + heading
  - **Summer** (Jun–Jul): grain fill + maturity

Feature groups:
  1. Temperature summaries (means, extremes)
  2. Growing Degree Days (GDD)
  3. Precipitation (cumulative, frequency)
  4. Extreme event counts (frost, heat, dry spells)
  5. Variability (CV of precip, temp range)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    DRY_SPELL_THRESHOLD_MM,
    FROST_THRESHOLD_C,
    GDD_BASE_TEMP_C,
    GROWTH_STAGES,
    HEAT_THRESHOLD_C,
    INTERIM_DIR,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def engineer_features(
    daily_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute the full engineered feature set from daily weather data.

    Parameters
    ----------
    daily_weather : pd.DataFrame, optional
        Daily weather with columns ``date``, ``year`` (crop-year),
        ``TMAX``, ``TMIN``, ``PRCP``, ``SNOW``.
        If None, loads from ``data/interim/daily_weather_with_crop_year.csv``.

    Returns
    -------
    pd.DataFrame
        One row per crop-year with all engineered features.
    """
    if daily_weather is None:
        path = INTERIM_DIR / "daily_weather_with_crop_year.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"No interim daily weather at {path}. "
                "Run preprocess.build_modeling_table() first."
            )
        daily_weather = pd.read_csv(path, parse_dates=["date"])

    df = daily_weather.copy()
    df["month"] = df["date"].dt.month
    df["tavg"] = (df["TMAX"] + df["TMIN"]) / 2.0

    # Compute all feature groups per crop-year
    features_list = []
    for year, year_df in df.groupby("year"):
        row = {"year": int(year)}
        row.update(_temperature_features(year_df))
        row.update(_gdd_features(year_df))
        row.update(_precipitation_features(year_df))
        row.update(_extreme_features(year_df))
        row.update(_variability_features(year_df))
        features_list.append(row)

    features_df = pd.DataFrame(features_list).sort_values("year").reset_index(drop=True)

    # Save
    out_path = PROCESSED_DIR / "engineered_features.parquet"
    features_df.to_parquet(out_path, index=False)
    logger.info(
        "Engineered %d features for %d crop-years → %s",
        len(features_df.columns) - 1,  # exclude 'year'
        len(features_df),
        out_path,
    )

    return features_df


def get_feature_names() -> list[str]:
    """Return the list of engineered feature column names (excluding 'year')."""
    # Generate a dummy row to discover column names
    dummy = pd.DataFrame({
        "date": pd.date_range("2020-10-01", "2021-07-31"),
        "TMAX": 20.0, "TMIN": 5.0, "PRCP": 2.0, "SNOW": 0.0,
    })
    dummy["month"] = dummy["date"].dt.month
    dummy["tavg"] = (dummy["TMAX"] + dummy["TMIN"]) / 2.0
    dummy["year"] = 2021

    row = {"year": 2021}
    row.update(_temperature_features(dummy))
    row.update(_gdd_features(dummy))
    row.update(_precipitation_features(dummy))
    row.update(_extreme_features(dummy))
    row.update(_variability_features(dummy))

    return [k for k in row.keys() if k != "year"]


# ---------------------------------------------------------------------------
# Feature computation — Temperature
# ---------------------------------------------------------------------------

def _temperature_features(df: pd.DataFrame) -> dict:
    """Mean and extreme temperature by growth stage + full season."""
    feats = {}

    # Full growing season
    feats["tavg_growing"] = df["tavg"].mean()
    feats["tmax_growing_max"] = df["TMAX"].max()
    feats["tmin_growing_min"] = df["TMIN"].min()

    # Per growth stage
    for stage, months in GROWTH_STAGES.items():
        stage_df = df[df["month"].isin(months)]
        feats[f"tavg_{stage}"] = stage_df["tavg"].mean() if len(stage_df) else np.nan
        feats[f"tmax_mean_{stage}"] = stage_df["TMAX"].mean() if len(stage_df) else np.nan
        feats[f"tmin_mean_{stage}"] = stage_df["TMIN"].mean() if len(stage_df) else np.nan

    return feats


# ---------------------------------------------------------------------------
# Feature computation — Growing Degree Days
# ---------------------------------------------------------------------------

def _gdd_features(df: pd.DataFrame) -> dict:
    """Growing Degree Days (base 0°C) by growth stage + total."""
    feats = {}

    # Daily GDD = max(0, Tavg - base)
    gdd_daily = np.maximum(0, df["tavg"] - GDD_BASE_TEMP_C)

    feats["gdd_total"] = gdd_daily.sum()

    for stage, months in GROWTH_STAGES.items():
        mask = df["month"].isin(months)
        feats[f"gdd_{stage}"] = gdd_daily[mask].sum()

    return feats


# ---------------------------------------------------------------------------
# Feature computation — Precipitation
# ---------------------------------------------------------------------------

def _precipitation_features(df: pd.DataFrame) -> dict:
    """Cumulative precipitation and rainy-day counts by stage."""
    feats = {}

    feats["prcp_total"] = df["PRCP"].sum()
    feats["prcp_days_total"] = (df["PRCP"] > 0).sum()

    for stage, months in GROWTH_STAGES.items():
        stage_df = df[df["month"].isin(months)]
        feats[f"prcp_{stage}"] = stage_df["PRCP"].sum() if len(stage_df) else 0.0
        feats[f"prcp_days_{stage}"] = (
            (stage_df["PRCP"] > 0).sum() if len(stage_df) else 0
        )

    # Snow
    feats["snow_total"] = df["SNOW"].sum()

    return feats


# ---------------------------------------------------------------------------
# Feature computation — Extremes
# ---------------------------------------------------------------------------

def _extreme_features(df: pd.DataFrame) -> dict:
    """Frost days, heat days, and maximum dry spell length."""
    feats = {}

    # Frost: days where TMIN ≤ 0°C
    feats["frost_days"] = (df["TMIN"] <= FROST_THRESHOLD_C).sum()

    # Spring frost (most damaging to wheat heading)
    spring_mask = df["month"].isin(GROWTH_STAGES["spring"])
    feats["frost_days_spring"] = (
        (df.loc[spring_mask, "TMIN"] <= FROST_THRESHOLD_C).sum()
    )

    # Heat: days where TMAX ≥ 35°C
    feats["heat_days_above_35"] = (df["TMAX"] >= HEAT_THRESHOLD_C).sum()

    # Summer heat (grain fill stress)
    summer_mask = df["month"].isin(GROWTH_STAGES["summer"])
    feats["heat_days_summer"] = (
        (df.loc[summer_mask, "TMAX"] >= HEAT_THRESHOLD_C).sum()
    )

    # Maximum consecutive dry days
    feats["dry_spell_max_days"] = _max_consecutive_dry(df["PRCP"])

    # Spring dry spell (critical period)
    spring_prcp = df.loc[spring_mask, "PRCP"]
    feats["dry_spell_max_spring"] = _max_consecutive_dry(spring_prcp)

    return feats


def _max_consecutive_dry(prcp_series: pd.Series) -> int:
    """Count the longest consecutive run of dry days (PRCP < threshold)."""
    if prcp_series.empty:
        return 0
    is_dry = (prcp_series < DRY_SPELL_THRESHOLD_MM).astype(int)
    # Group consecutive runs
    groups = is_dry.ne(is_dry.shift()).cumsum()
    dry_runs = is_dry.groupby(groups).sum()
    return int(dry_runs.max()) if not dry_runs.empty else 0


# ---------------------------------------------------------------------------
# Feature computation — Variability
# ---------------------------------------------------------------------------

def _variability_features(df: pd.DataFrame) -> dict:
    """Coefficient of variation and range metrics for key variables."""
    feats = {}

    # Spring precipitation CV (high variability → risk)
    spring_mask = df["month"].isin(GROWTH_STAGES["spring"])
    spring_prcp = df.loc[spring_mask, "PRCP"]
    if len(spring_prcp) > 1 and spring_prcp.mean() > 0:
        feats["prcp_cv_spring"] = spring_prcp.std() / spring_prcp.mean()
    else:
        feats["prcp_cv_spring"] = 0.0

    # Spring temperature range (max TMAX - min TMIN within spring)
    spring_tmax = df.loc[spring_mask, "TMAX"]
    spring_tmin = df.loc[spring_mask, "TMIN"]
    if len(spring_tmax) > 0 and len(spring_tmin) > 0:
        feats["temp_range_spring"] = spring_tmax.max() - spring_tmin.min()
    else:
        feats["temp_range_spring"] = np.nan

    # Growing-season temperature standard deviation
    feats["tavg_std_growing"] = df["tavg"].std() if len(df) > 1 else 0.0

    return feats
