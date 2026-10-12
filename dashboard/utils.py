"""
API client helpers for the Streamlit dashboard.

All communication with the FastAPI backend goes through this module.
"""

from __future__ import annotations

import os

import httpx

from config.settings import API_PORT

# In Docker the dashboard container receives API_HOST as an env var ("api").
# Locally it defaults to localhost.
_API_HOST = os.getenv("API_HOST", "localhost")
_API_PORT = int(os.getenv("API_PORT", str(API_PORT)))
_BASE_URL = f"http://{_API_HOST}:{_API_PORT}"
_TIMEOUT = 30.0


def get_health() -> dict:
    """Check API health."""
    try:
        r = httpx.get(f"{_BASE_URL}/health", timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"status": "offline", "model_loaded": False, "data_years": 0}


def post_predict(year: int, use_live_weather: bool = True) -> dict:
    """POST /predict — get forecast for a year."""
    r = httpx.post(
        f"{_BASE_URL}/predict",
        json={"year": year, "use_live_weather": use_live_weather},
        timeout=_TIMEOUT,
    )
    if not r.is_success:
        # Extract detail message from JSON error body
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ValueError(detail)
    return r.json()


def get_drift(year: int) -> dict:
    """GET /drift/{year} — drift analysis."""
    r = httpx.get(f"{_BASE_URL}/drift/{year}", timeout=_TIMEOUT)
    if not r.is_success:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ValueError(detail)
    return r.json()


def get_reliability(year: int) -> dict:
    """GET /reliability/{year} — reliability assessment."""
    r = httpx.get(f"{_BASE_URL}/reliability/{year}", timeout=_TIMEOUT)
    if not r.is_success:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise ValueError(detail)
    return r.json()


def get_history() -> dict:
    """GET /history — all historical records."""
    r = httpx.get(f"{_BASE_URL}/history", timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def post_adapt(year: int) -> dict:
    """POST /adapt — trigger adaptation."""
    r = httpx.post(
        f"{_BASE_URL}/adapt",
        json={"year": year},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()
