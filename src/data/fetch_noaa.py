"""
NOAA weather data clients.

Two modes:
  1. **Historical** — NOAA CDO (Climate Data Online) Web Services API for
     daily weather records from the Caldwell, KS station (1948–2023).
  2. **Live / forecast** — NOAA NWS (National Weather Service) API for
     current-season weather at inference time.

Both clients cache raw data to ``data/raw/`` to avoid redundant downloads.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    CALDWELL_LAT,
    CALDWELL_LON,
    NOAA_CDO_BASE_URL,
    NOAA_CDO_TOKEN,
    NOAA_DATA_END,
    NOAA_DATA_START,
    NOAA_DATATYPES,
    NOAA_NWS_BASE_URL,
    NOAA_STATION_ID,
    NOAA_STATION_ID_SECONDARY,
    NOAA_STATION_CUTOVER_YEAR,
    RAW_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0
_CDO_PAGE_LIMIT = 1000           # Max records per CDO request
_CDO_RATE_LIMIT_DELAY = 0.25     # 5 req/sec → 200 ms buffer
_RAW_HISTORICAL = "noaa_daily_caldwell_ks.csv"
_RAW_LIVE = "noaa_nws_live.csv"


# ═══════════════════════════════════════════════════════════════════════════
# 1. HISTORICAL — NOAA CDO API
# ═══════════════════════════════════════════════════════════════════════════

def fetch_historical_weather(
    start_date: str = NOAA_DATA_START,
    end_date: str = NOAA_DATA_END,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch daily weather observations from the Caldwell, KS GHCN-D station.

    Parameters
    ----------
    start_date, end_date : str
        ISO date strings (``YYYY-MM-DD``).
    force_refresh : bool
        If True, re-download even if a cached CSV exists.

    Returns
    -------
    pd.DataFrame
        Columns: ``date`` (datetime64), ``TMAX`` (°C), ``TMIN`` (°C),
        ``PRCP`` (mm), ``SNOW`` (mm).  One row per day.
    """
    cache_path = RAW_DIR / _RAW_HISTORICAL

    if cache_path.exists() and not force_refresh:
        logger.info("Loading cached NOAA weather from %s", cache_path)
        return _load_historical_csv(cache_path)

    if not NOAA_CDO_TOKEN:
        if cache_path.exists():
            logger.warning(
                "NOAA_CDO_TOKEN not set — falling back to cached CSV at %s",
                cache_path,
            )
            return _load_historical_csv(cache_path)
        raise EnvironmentError(
            "NOAA_CDO_TOKEN is not set and no cached CSV exists at "
            f"{cache_path}.  Set the token in your .env file or place a "
            "manually-downloaded CSV at the expected path."
        )

    df = _fetch_cdo_paginated(start_date, end_date)
    df.to_csv(cache_path, index=False)
    logger.info("Saved historical weather to %s (%d rows)", cache_path, len(df))
    return df


def _fetch_cdo_paginated(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch all daily records via CDO API, handling pagination and the
    1-year-max-per-request limitation.

    Uses two stations to cover the full date range:
      - NOAA_STATION_ID_SECONDARY for years before NOAA_STATION_CUTOVER_YEAR
      - NOAA_STATION_ID (primary) for years from NOAA_STATION_CUTOVER_YEAR onward
    """
    headers = {"token": NOAA_CDO_TOKEN}

    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date, "%Y-%m-%d").date()

    cutover = date(NOAA_STATION_CUTOVER_YEAR, 1, 1)

    all_records: list[dict] = []

    year_start = sd
    while year_start <= ed:
        year_end = min(date(year_start.year, 12, 31), ed)

        # Choose station based on year
        if year_start < cutover:
            station = NOAA_STATION_ID_SECONDARY
        else:
            station = NOAA_STATION_ID

        logger.info(
            "Fetching CDO data: station=%s  %s → %s",
            station, year_start, year_end,
        )

        offset = 1
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station,
                "datatypeid": ",".join(NOAA_DATATYPES),
                "startdate": year_start.isoformat(),
                "enddate": year_end.isoformat(),
                "units": "metric",
                "limit": _CDO_PAGE_LIMIT,
                "offset": offset,
            }
            data = _cdo_request(headers, params)
            results = data.get("results", [])
            if not results:
                break
            all_records.extend(results)
            offset += _CDO_PAGE_LIMIT
            if len(results) < _CDO_PAGE_LIMIT:
                break

        year_start = date(year_start.year + 1, 1, 1)

    if not all_records:
        raise ValueError(
            "NOAA CDO returned no weather records for either station. "
            "Check your NOAA_CDO_TOKEN and station IDs in config/settings.py."
        )

    df = pd.DataFrame(all_records)
    logger.info("CDO: fetched %d raw records total", len(df))
    return _pivot_cdo(df)


def _cdo_request(headers: dict, params: dict) -> dict:
    """Single CDO request with retry + rate-limit handling."""
    url = f"{NOAA_CDO_BASE_URL}/data"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            time.sleep(_CDO_RATE_LIMIT_DELAY)
            logger.debug("CDO request: %s (attempt %d)", params, attempt)
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", _RETRY_DELAY * attempt))
                logger.warning("CDO rate-limited; waiting %.1fs", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("CDO attempt %d failed: %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"CDO API failed after {_MAX_RETRIES} attempts"
                ) from exc
            time.sleep(_RETRY_DELAY * attempt)
    return {}  # unreachable but satisfies type checkers


def _pivot_cdo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the long-format CDO response into one row per day with columns
    TMAX, TMIN, PRCP, SNOW (all metric — °C and mm).
    """
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[["date", "datatype", "value"]]
    pivoted = df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="first"
    ).reset_index()
    pivoted.columns.name = None
    pivoted["date"] = pd.to_datetime(pivoted["date"])

    # Ensure all expected columns exist (fill missing with NaN)
    for col in NOAA_DATATYPES:
        if col not in pivoted.columns:
            pivoted[col] = float("nan")

    return pivoted[["date"] + NOAA_DATATYPES].sort_values("date").reset_index(drop=True)


def _load_historical_csv(path: Path) -> pd.DataFrame:
    """Load a previously-saved historical weather CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    for col in NOAA_DATATYPES:
        if col not in df.columns:
            df[col] = float("nan")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. LIVE — NOAA NWS API (no key required)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_live_weather() -> pd.DataFrame:
    """
    Fetch recent observed weather for the Caldwell, KS area from the
    NOAA National Weather Service API.

    Returns daily TMAX, TMIN, PRCP observations for the current growing
    season (Oct 1 of previous year through today).

    Returns
    -------
    pd.DataFrame
        Same schema as ``fetch_historical_weather`` (date, TMAX, TMIN, PRCP, SNOW).
    """
    # Step 1: resolve grid point for lat/lon
    point_url = f"{NOAA_NWS_BASE_URL}/points/{CALDWELL_LAT},{CALDWELL_LON}"
    point_data = _nws_request(point_url)
    station_url = point_data["properties"]["observationStations"]

    # Step 2: get nearest observation station
    stations_data = _nws_request(station_url)
    if not stations_data.get("features"):
        raise ValueError("No NWS observation stations found near Caldwell, KS")
    station_id = stations_data["features"][0]["properties"]["stationIdentifier"]

    # Step 3: fetch recent observations
    obs_url = (
        f"{NOAA_NWS_BASE_URL}/stations/{station_id}/observations"
    )
    obs_data = _nws_request(obs_url)
    observations = obs_data.get("features", [])

    if not observations:
        logger.warning("No live NWS observations returned")
        return pd.DataFrame(columns=["date", "TMAX", "TMIN", "PRCP", "SNOW"])

    records = []
    for obs in observations:
        props = obs["properties"]
        timestamp = props.get("timestamp")
        if not timestamp:
            continue
        dt = pd.to_datetime(timestamp).date()
        records.append({
            "date": dt,
            "TMAX": _safe_value(props, "maxTemperatureLast24Hours"),
            "TMIN": _safe_value(props, "minTemperatureLast24Hours"),
            "PRCP": _safe_value(props, "precipitationLast24Hours"),
            "SNOW": None,  # NWS observations don't consistently report snow
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate to daily (multiple observations per day possible)
    daily = df.groupby("date").agg({
        "TMAX": "max",
        "TMIN": "min",
        "PRCP": "sum",
        "SNOW": "sum",
    }).reset_index()

    # Cache it
    cache_path = RAW_DIR / _RAW_LIVE
    daily.to_csv(cache_path, index=False)
    logger.info("Saved live weather to %s (%d days)", cache_path, len(daily))

    return daily


def _nws_request(url: str) -> dict:
    """Single NWS API request with retries."""
    headers = {
        "User-Agent": "(agr1-wheat-forecast, contact@example.com)",
        "Accept": "application/geo+json",
    }
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            logger.warning("NWS attempt %d for %s failed: %s", attempt, url, exc)
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"NWS API failed after {_MAX_RETRIES} attempts: {url}"
                ) from exc
            time.sleep(_RETRY_DELAY * attempt)
    return {}


def _safe_value(props: dict, key: str) -> float | None:
    """Extract a numeric value from a NWS observation property, or None."""
    entry = props.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        val = entry.get("value")
        return float(val) if val is not None else None
    return float(entry) if entry is not None else None
