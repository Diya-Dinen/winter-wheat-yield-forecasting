"""
Pydantic schemas for the FastAPI endpoints.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for POST /predict."""
    year: int = Field(..., description="Crop year to forecast", ge=1948, le=2100)
    use_live_weather: bool = Field(
        True,
        description="Pull live weather from NOAA NWS API if True (current/future years)",
    )


class AdaptRequest(BaseModel):
    """Request body for POST /adapt (admin only)."""
    year: int = Field(..., description="Year to run adaptation for")
    force: bool = Field(
        False,
        description="Force adaptation even if drift is below threshold",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    """Response for POST /predict."""
    year: int
    predicted_yield: float = Field(..., description="Point forecast (bu/acre)")
    lower_bound: float = Field(..., description="Lower bound of 90% prediction interval")
    upper_bound: float = Field(..., description="Upper bound of 90% prediction interval")
    drift_severity: str = Field(..., description="LOW / MODERATE / HIGH")
    drift_score: float = Field(..., description="Composite drift score (0–1)")
    contributing_variables: list[str] = Field(
        ..., description="Weather variables contributing to drift"
    )
    reliability: str = Field(..., description="RELIABLE / UNRELIABLE")
    adaptation_applied: bool
    adaptation_strategy: Optional[str]
    weather_source: str = Field(..., description="'observed', 'forecast', or 'historical'")
    lead_time_note: str = Field(..., description="Caveat about forecast lead time")


class DriftResponse(BaseModel):
    """Response for GET /drift/{year}."""
    year: int
    severity: str
    score: float
    contributing_variables: list[str]
    univariate_summary: dict
    multivariate: Optional[dict]


class ReliabilityResponse(BaseModel):
    """Response for GET /reliability/{year}."""
    year: int
    reliability: str
    drift_severity: str
    interval_width: float
    confidence_level: float = Field(0.90, description="Nominal coverage level")


class HistoryRecord(BaseModel):
    """Single record in the history response."""
    year: int
    actual_yield: Optional[float]
    predicted_yield: Optional[float]
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    drift_severity: Optional[str]


class HistoryResponse(BaseModel):
    """Response for GET /history."""
    records: list[HistoryRecord]
    total_years: int


class AdaptResponse(BaseModel):
    """Response for POST /adapt."""
    year: int
    adaptation_applied: bool
    strategy_used: str
    original_prediction: float
    adapted_prediction: Optional[float]
    message: str


class HealthResponse(BaseModel):
    """Response for GET /health."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    data_years: int
