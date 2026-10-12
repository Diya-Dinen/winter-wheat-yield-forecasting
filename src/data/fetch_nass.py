"""
USDA NASS QuickStats API client.

Fetches historical winter wheat yield data for Sumner County, KS.
Supports both API-based retrieval (preferred) and fallback CSV loading.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

import datetime as _dt
from config.settings import (
    NASS_API_KEY,
    NASS_BASE_URL,
    NASS_PARAMS,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 2.0
_RAW_FILENAME = "nass_yield_sumner_ks.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_yield_data(
    start_year: int = 1926,
    end_year: int = _dt.date.today().year,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch winter wheat yield for Sumner County, KS.

    Tries the NASS QuickStats REST API first.  If no API key is configured,
    falls back to a previously-downloaded CSV in ``data/raw/``.

    Parameters
    ----------
    start_year : int
        First crop year to include (inclusive).
    end_year : int
        Last crop year to include (inclusive).
    force_refresh : bool
        If True, re-download even if a cached CSV already exists.

    Returns
    -------
    pd.DataFrame
        Columns: ``year`` (int), ``yield_bu_per_acre`` (float).
        Sorted by year ascending, NaN-yields dropped.
    """
    cache_path = RAW_DIR / _RAW_FILENAME

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached NASS yield data from %s", cache_path)
        return _load_and_clean(cache_path, start_year, end_year)

    if not NASS_API_KEY:
        if cache_path.exists():
            logger.warning(
                "NASS_API_KEY not set — falling back to cached CSV at %s",
                cache_path,
            )
            return _load_and_clean(cache_path, start_year, end_year)
        raise EnvironmentError(
            "NASS_API_KEY is not set and no cached CSV exists at "
            f"{cache_path}.  Set the key in your .env file or place a "
            "manually-downloaded CSV at the expected path."
        )

    df = _fetch_from_api(start_year, end_year)
    df.to_csv(cache_path, index=False)
    logger.info("Saved NASS yield data to %s (%d rows)", cache_path, len(df))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_from_api(start_year: int, end_year: int) -> pd.DataFrame:
    """Call the NASS QuickStats REST endpoint with retries."""
    params = {
        **NASS_PARAMS,
        "key": NASS_API_KEY,
        "year__GE": str(start_year),
        "year__LE": str(end_year),
        "format": "JSON",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            logger.info(
                "NASS API request (attempt %d/%d): years %d–%d",
                attempt, _MAX_RETRIES, start_year, end_year,
            )
            resp = requests.get(NASS_BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            logger.warning("NASS API attempt %d failed: %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"NASS API failed after {_MAX_RETRIES} attempts"
                ) from exc
            time.sleep(_RETRY_DELAY_SECONDS * attempt)

    rows = data.get("data", [])
    if not rows:
        raise ValueError(
            "NASS API returned no data for the given parameters. "
            "Check county/commodity filters in config/settings.py."
        )

    df = pd.DataFrame(rows)

    # The API returns many columns; keep only what we need
    df = df.rename(columns={"year": "year", "Value": "yield_bu_per_acre"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield_bu_per_acre"] = pd.to_numeric(
        df["yield_bu_per_acre"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    df = (
        df[["year", "yield_bu_per_acre"]]
        .dropna()
        .drop_duplicates(subset="year")
        .sort_values("year")
        .reset_index(drop=True)
    )

    logger.info("Fetched %d yield records from NASS API", len(df))
    return df


def _load_and_clean(
    path: Path, start_year: int, end_year: int
) -> pd.DataFrame:
    """Load a cached CSV and normalize column names."""
    df = pd.read_csv(path)

    # Handle both API-saved format and manually-downloaded NASS CSVs
    col_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in ("year",):
            col_map[col] = "year"
        elif lower in ("value", "yield_bu_per_acre", "yield"):
            col_map[col] = "yield_bu_per_acre"
    df = df.rename(columns=col_map)

    if "year" not in df.columns or "yield_bu_per_acre" not in df.columns:
        raise KeyError(
            f"Expected 'year' and 'yield_bu_per_acre' columns in {path}. "
            f"Found: {list(df.columns)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["yield_bu_per_acre"] = pd.to_numeric(
        df["yield_bu_per_acre"].astype(str).str.replace(",", ""),
        errors="coerce",
    )

    df = (
        df[["year", "yield_bu_per_acre"]]
        .dropna()
        .query("@start_year <= year <= @end_year")
        .drop_duplicates(subset="year")
        .sort_values("year")
        .reset_index(drop=True)
    )
    return df
