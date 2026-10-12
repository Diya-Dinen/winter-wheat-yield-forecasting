"""
Data preprocessing — merge yield + weather, optional fertilizer, versioned output.

Handles the winter wheat crop-year convention (Oct year-1 → Jul year) so that
each row in the output table represents one crop-year with its corresponding
weather and yield.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.settings import (
    GROWING_SEASON_MONTHS,
    INTERIM_DIR,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.data.fetch_nass import fetch_yield_data
from src.data.fetch_noaa import fetch_historical_weather

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_modeling_table(
    start_year: int = 1948,
    end_year: int = _dt.date.today().year,
    include_fertilizer: bool = False,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Build the merged modeling table: one row per crop-year with weather + yield.

    Parameters
    ----------
    start_year : int
        First crop year (must be ≥ 1948 since weather data starts then).
    end_year : int
        Last crop year.
    include_fertilizer : bool
        If True and a fertilizer CSV exists in ``data/raw/``, merge it
        (forward-filled from 5-year intervals).
    force_refresh : bool
        Re-fetch raw data even if cached.

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, weather columns (daily, one row per crop-year-day),
        ``yield_bu_per_acre``.  Saved as versioned Parquet.
    """
    # --- 1. Fetch raw data ---
    yield_df = fetch_yield_data(start_year=start_year, end_year=end_year,
                                force_refresh=force_refresh)
    weather_df = fetch_historical_weather(force_refresh=force_refresh)

    # --- 2. Assign crop-year to each weather day ---
    weather_df = _assign_crop_year(weather_df)

    # --- 3. Filter to growing-season months only ---
    weather_df = weather_df[
        weather_df["date"].dt.month.isin(GROWING_SEASON_MONTHS)
    ].copy()

    # --- 4. Save interim daily weather with crop-year ---
    interim_path = INTERIM_DIR / "daily_weather_with_crop_year.csv"
    weather_df.to_csv(interim_path, index=False)
    logger.info("Saved interim daily weather to %s", interim_path)

    # --- 5. Aggregate daily → crop-year level (basic stats) ---
    yearly_weather = _aggregate_yearly(weather_df)

    # --- 6. Merge with yield ---
    merged = pd.merge(yield_df, yearly_weather, on="year", how="inner")
    logger.info(
        "Merged dataset: %d crop-years (yield ∩ weather)",
        len(merged),
    )

    # --- 7. Optional fertilizer ---
    if include_fertilizer:
        merged = _merge_fertilizer(merged)

    # --- 8. Final cleaning ---
    merged = merged.sort_values("year").reset_index(drop=True)

    # --- 9. Save versioned Parquet ---
    version_tag = _version_hash(merged)
    out_path = PROCESSED_DIR / f"modeling_table_v{version_tag}.parquet"
    merged.to_parquet(out_path, index=False)

    # Also save a "latest" symlink / copy for convenience
    latest_path = PROCESSED_DIR / "modeling_table_latest.parquet"
    merged.to_parquet(latest_path, index=False)

    logger.info("Saved modeling table to %s (%d rows, %d cols)",
                out_path, len(merged), len(merged.columns))

    return merged


def load_latest_table() -> pd.DataFrame:
    """Load the most recently saved modeling table."""
    path = PROCESSED_DIR / "modeling_table_latest.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No modeling table found at {path}.  Run build_modeling_table() first."
        )
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assign_crop_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a ``crop_year`` column.

    Convention: crop-year N covers Oct of year N-1 through Jul of year N.
    - Oct–Dec → crop_year = calendar_year + 1
    - Jan–Jul → crop_year = calendar_year
    """
    df = df.copy()
    month = df["date"].dt.month
    cal_year = df["date"].dt.year
    df["crop_year"] = cal_year.where(month <= 7, cal_year + 1)
    df = df.rename(columns={"crop_year": "year"})
    df["year"] = df["year"].astype(int)
    return df


def _aggregate_yearly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily weather to crop-year level.

    Produces basic seasonal summaries — the full feature engineering
    (GDD, extremes, variability) is handled by ``feature_engineering.py``
    which operates on the interim daily data.
    """
    agg = df.groupby("year").agg(
        tmax_mean=("TMAX", "mean"),
        tmax_max=("TMAX", "max"),
        tmin_mean=("TMIN", "mean"),
        tmin_min=("TMIN", "min"),
        prcp_total=("PRCP", "sum"),
        prcp_days=("PRCP", lambda s: (s > 0).sum()),
        snow_total=("SNOW", "sum"),
        obs_count=("TMAX", "count"),
    ).reset_index()

    return agg


def _merge_fertilizer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge optional fertilizer data, forward-filling 5-year intervals.

    Expects a CSV at ``data/raw/fertilizer_sumner_ks.csv`` with columns
    ``year`` and ``fertilizer_lbs_per_acre`` (or similar).
    """
    fert_path = RAW_DIR / "fertilizer_sumner_ks.csv"
    if not fert_path.exists():
        logger.warning(
            "Fertilizer CSV not found at %s — skipping fertilizer merge",
            fert_path,
        )
        return df

    fert = pd.read_csv(fert_path)
    # Normalize column names
    fert.columns = [c.strip().lower().replace(" ", "_") for c in fert.columns]
    if "year" not in fert.columns:
        logger.warning("Fertilizer CSV has no 'year' column — skipping")
        return df

    # Forward-fill to annual frequency
    full_years = pd.DataFrame({"year": range(df["year"].min(), df["year"].max() + 1)})
    fert = pd.merge(full_years, fert, on="year", how="left")
    fert = fert.ffill()  # forward-fill sparse intervals

    df = pd.merge(df, fert, on="year", how="left")
    logger.info("Merged fertilizer data (%d non-null rows)", fert.dropna().shape[0])
    return df


def _version_hash(df: pd.DataFrame) -> str:
    """Create a short hash from the dataframe content for versioning."""
    content = json.dumps({
        "shape": list(df.shape),
        "columns": list(df.columns),
        "years": [int(df["year"].min()), int(df["year"].max())],
        "timestamp": datetime.now().isoformat(),
    })
    return hashlib.md5(content.encode()).hexdigest()[:8]
