"""POST /adapt — Admin-only manual adaptation trigger."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_state, is_loaded
from api.schemas import AdaptRequest, AdaptResponse
from src.pipeline.inference_pipeline import run_inference

router = APIRouter()


@router.post("/adapt", response_model=AdaptResponse)
def trigger_adaptation(request: AdaptRequest):
    """
    Manually trigger or inspect an adaptation event for a given year.

    Admin-only endpoint — use when you want to force adaptation
    regardless of the automatic drift threshold.

    Set ``force=true`` to override the drift threshold and always apply
    the bias-correction strategy.
    """
    if not is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        result = run_inference(
            year=request.year,
            use_live_weather=False,
            artifacts=get_state(),
            force_adapt=request.force,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return AdaptResponse(
        year=request.year,
        adaptation_applied=result.adaptation_applied,
        strategy_used=result.adaptation_strategy or "none",
        original_prediction=result.predicted_yield,
        adapted_prediction=result.predicted_yield if result.adaptation_applied else None,
        message=(
            f"Adaptation {'applied' if result.adaptation_applied else 'not needed'} "
            f"for year {request.year} (drift: {result.drift_severity})"
            + (" [forced]" if request.force and result.adaptation_applied else "")
        ),
    )
