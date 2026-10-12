"""GET /reliability/{year} — Reliability assessment."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_state, is_loaded
from api.schemas import ReliabilityResponse
from src.drift.detector import detect_drift
from src.models.conformal import ConformalPredictor

router = APIRouter()


@router.get("/reliability/{year}", response_model=ReliabilityResponse)
def get_reliability(year: int):
    """
    Return reliability assessment for a prediction year.

    Combines drift severity with interval width to produce a
    RELIABLE / UNRELIABLE label.
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    state = get_state()
    df = state["df"]
    feature_cols = state["feature_cols"]
    conformal: ConformalPredictor = state["conformal"]

    if year not in df["year"].values:
        raise HTTPException(status_code=404, detail=f"Year {year} not found.")

    # Drift
    train_mask = df["year"] < year
    test_mask = df["year"] == year
    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values

    if len(X_train) == 0:
        raise HTTPException(status_code=400, detail="No training data before this year.")

    drift_result = detect_drift(X_train, X_test, feature_names=feature_cols)

    # Conformal interval
    conformal_result = conformal.predict_single(X_test)
    interval_width = conformal_result.interval_width

    # Reliability logic
    TYPICAL_YIELD = 40.0
    width_ratio = interval_width / TYPICAL_YIELD

    if drift_result.severity == "HIGH":
        reliability = "UNRELIABLE"
    elif width_ratio > 0.5:
        reliability = "UNRELIABLE"
    elif drift_result.severity == "MODERATE" and width_ratio > 0.3:
        reliability = "UNRELIABLE"
    else:
        reliability = "RELIABLE"

    return ReliabilityResponse(
        year=year,
        reliability=reliability,
        drift_severity=drift_result.severity,
        interval_width=round(interval_width, 2),
    )
