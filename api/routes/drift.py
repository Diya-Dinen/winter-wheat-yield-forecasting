"""GET /drift/{year} — Detailed drift analysis for a specific year."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_state, is_loaded
from api.schemas import DriftResponse
from src.drift.detector import detect_drift

router = APIRouter()


@router.get("/drift/{year}", response_model=DriftResponse)
def get_drift(year: int):
    """
    Return detailed drift analysis for the specified year.

    Includes univariate (per-feature KS + PSI) and multivariate
    (Mahalanobis + PCA) drift scores.
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    state = get_state()
    df = state["df"]
    feature_cols = state["feature_cols"]

    # Check if year exists in data
    if year not in df["year"].values:
        raise HTTPException(
            status_code=404,
            detail=f"Year {year} not found in dataset.",
        )

    # Training data = all years before the target year
    train_mask = df["year"] < year
    test_mask = df["year"] == year

    X_train = df.loc[train_mask, feature_cols].values
    X_test = df.loc[test_mask, feature_cols].values

    if len(X_train) == 0:
        raise HTTPException(
            status_code=400,
            detail=f"No training data available before year {year}.",
        )

    drift_result = detect_drift(X_train, X_test, feature_names=feature_cols)
    drift_dict = drift_result.to_dict()

    return DriftResponse(
        year=year,
        severity=drift_dict["severity"],
        score=drift_dict["score"],
        contributing_variables=drift_dict["contributing_variables"],
        univariate_summary=drift_dict["univariate_summary"],
        multivariate=drift_dict["multivariate"],
    )
