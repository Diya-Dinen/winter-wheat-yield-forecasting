"""POST /predict — Yield forecast with drift + conformal interval."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.dependencies import get_state, is_loaded
from api.schemas import PredictRequest, PredictResponse
from src.pipeline.inference_pipeline import run_inference

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Generate a yield forecast for the specified crop year.

    Returns point prediction, 90% conformal interval, drift assessment,
    reliability flag, and adaptation status.
    """
    if not is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run the training pipeline first.",
        )

    try:
        result = run_inference(
            year=request.year,
            use_live_weather=request.use_live_weather,
            artifacts=get_state(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictResponse(**result.to_dict())
