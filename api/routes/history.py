"""GET /history — Past predictions vs. actuals for the dashboard trend chart."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_state, is_loaded
from api.schemas import HistoryRecord, HistoryResponse

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def get_history():
    """
    Return historical yield data (actual + predicted) for all available years.

    Used by the dashboard's trend chart to show actual vs. forecast overlay.
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    state = get_state()
    df = state["df"]
    feature_cols = state["feature_cols"]
    model = state["model"]
    conformal = state["conformal"]

    records = []
    years = sorted(df["year"].unique())

    for year in years:
        row = df[df["year"] == year]
        actual = float(row["yield_bu_per_acre"].values[0])

        # Predict (using the loaded model — note this is in-sample for training years)
        X = row[feature_cols].values
        try:
            cr = conformal.predict_single(X)
            predicted = cr.point_prediction
            lower = cr.lower_bound
            upper = cr.upper_bound
        except Exception:
            predicted = float(model.predict(X)[0])
            lower = None
            upper = None

        # Simple drift severity for historical display
        drift_severity = None
        try:
            from src.drift.detector import detect_drift
            train_mask = df["year"] < year
            X_train = df.loc[train_mask, feature_cols].values
            if len(X_train) > 10:
                drift_result = detect_drift(X_train, X, feature_names=feature_cols)
                drift_severity = drift_result.severity
        except Exception:
            pass

        records.append(HistoryRecord(
            year=year,
            actual_yield=round(actual, 2),
            predicted_yield=round(predicted, 2) if predicted else None,
            lower_bound=round(lower, 2) if lower is not None else None,
            upper_bound=round(upper, 2) if upper is not None else None,
            drift_severity=drift_severity,
        ))

    return HistoryResponse(records=records, total_years=len(records))
