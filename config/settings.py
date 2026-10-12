"""
Centralized configuration for the Winter Wheat Yield Forecasting system.

All paths, API URLs, station IDs, thresholds, and feature definitions live here
so that no module needs to hardcode any of these values.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env (project root)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
NASS_API_KEY: str = os.getenv("NASS_API_KEY", "")
NOAA_CDO_TOKEN: str = os.getenv("NOAA_CDO_TOKEN", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments" / "results"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXPERIMENTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# USDA NASS Configuration
# ---------------------------------------------------------------------------
NASS_BASE_URL = "https://quickstats.nass.usda.gov/api/api_GET"
NASS_PARAMS = {
    "commodity_desc": "WHEAT",
    "util_practice_desc": "ALL UTILIZATION PRACTICES",
    "statisticcat_desc": "YIELD",
    "unit_desc": "BU / ACRE",
    "domain_desc": "TOTAL",
    "agg_level_desc": "COUNTY",
    "state_alpha": "KS",
    "county_name": "SUMNER",
}

# ---------------------------------------------------------------------------
# NOAA Configuration
# ---------------------------------------------------------------------------
NOAA_CDO_BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
# Primary station: Wichita Eisenhower Airport (1953–present, 100% coverage)
NOAA_STATION_ID = "GHCND:USW00003928"
# Secondary station: Wichita River (1897–1953, 97% coverage) — fills 1948–1952 gap
NOAA_STATION_ID_SECONDARY = "GHCND:USW00013998"
NOAA_STATION_CUTOVER_YEAR = 1953  # Use secondary station before this year
NOAA_DATATYPES = ["TMAX", "TMIN", "PRCP", "SNOW"]
NOAA_DATA_START = "1948-01-01"
# End date is set dynamically to yesterday to always pull the most recent data.
# Override this only if you want to pin to a specific date.
import datetime as _dt
NOAA_DATA_END: str = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()

# NOAA NWS API (no key required) — for live weather at inference time
NOAA_NWS_BASE_URL = "https://api.weather.gov"
CALDWELL_LAT = 37.0317
CALDWELL_LON = -97.6061

# ---------------------------------------------------------------------------
# Winter Wheat Growing Season (months, 1-indexed)
# Crop-year N means: Oct of year N-1 through Jul of year N
# ---------------------------------------------------------------------------
GROWTH_STAGES = {
    "fall": [10, 11],          # Planting + emergence
    "winter": [12, 1, 2],     # Dormancy + vernalization
    "spring": [3, 4, 5],      # Tillering + stem elongation + heading
    "summer": [6, 7],         # Grain fill + maturity
}
GROWING_SEASON_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7]

# GDD base temperature (°C) for winter wheat
GDD_BASE_TEMP_C = 0.0

# Extreme thresholds
FROST_THRESHOLD_C = 0.0        # Days with TMIN ≤ 0°C
HEAT_THRESHOLD_C = 35.0        # Days with TMAX ≥ 35°C
DRY_SPELL_THRESHOLD_MM = 1.0   # Day counts as "dry" if PRCP < this

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
CV_MIN_TRAIN_YEARS = 25         # Minimum training set size for expanding-window CV
CV_STEP_YEARS = 1               # Annual rolling-origin evaluation (one test year per cutoff)
CONFORMAL_ALPHA = 0.10          # 90% prediction interval (1 - alpha)
CONFORMAL_WINDOW = 20           # Most recent out-of-sample residuals used for calibration
CONFORMAL_MIN_RESIDUALS = 10    # Fewer residuals than this -> interval flagged as unreliable
MIN_SEASON_COVERAGE = 0.95      # Crop-years with less weather coverage are excluded
EXPECTED_SEASON_DAYS = 304      # Oct 1 - Jul 31 (non-leap)
MIN_DRIFT_REF_YEARS = 15        # Minimum reference years for drift detection

# ---------------------------------------------------------------------------
# Drift Detection Thresholds
# ---------------------------------------------------------------------------
KS_ALPHA = 0.05                 # p-value threshold for KS test
DRIFT_P_HIGH = 0.025            # Empirical p-value <= this -> HIGH drift
DRIFT_P_MODERATE = 0.10         # Empirical p-value <= this -> MODERATE drift
DRIFT_Z_FEATURE = 3.0           # |z| above this flags an individual feature
MIN_BATCH_FOR_PSI_KS = 8        # KS/PSI only meaningful with at least this many test rows
PSI_MODERATE = 0.10             # PSI ≥ this → MODERATE
PSI_HIGH = 0.25                 # PSI ≥ this → HIGH
MAHALANOBIS_MODERATE = 3.0      # Mahalanobis distance thresholds
MAHALANOBIS_HIGH = 5.0
PCA_VARIANCE_RETAINED = 0.95    # Retain 95% variance for PCA reconstruction

# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------
ADAPTATION_TRIGGER_SEVERITY = "MODERATE"  # Minimum severity to trigger adaptation
BIAS_CORRECTION_WINDOW = 5      # Number of recent years for bias correction
RETRAIN_WINDOW_SIZES = [20, 30, 40]  # Candidate window sizes for windowed retrain

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
DASHBOARD_PORT = 8501
