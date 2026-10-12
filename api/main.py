"""
FastAPI application factory.

Serves the yield forecasting API with endpoints for prediction, drift
detection, reliability assessment, history, and manual adaptation.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import set_state, is_loaded
from api.routes import predict, drift, reliability, history, adapt
from api.schemas import HealthResponse
from config.settings import API_HOST, API_PORT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and data on startup."""
    logger.info("Loading trained artifacts...")
    try:
        from src.pipeline.train_pipeline import load_trained_artifacts
        artifacts = load_trained_artifacts()
        set_state(artifacts)
        logger.info("Artifacts loaded successfully")
    except FileNotFoundError:
        logger.warning(
            "No trained artifacts found. Run the training pipeline first: "
            "python -m src.pipeline.train_pipeline"
        )
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Winter Wheat Yield Forecasting API",
    description=(
        "Adaptive Temporal OOD Winter Wheat Yield Forecasting with "
        "Drift Detection, Conformal Prediction, and Adaptive Recalibration. "
        "Sumner County, KS."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Streamlit dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(predict.router, tags=["Prediction"])
app.include_router(drift.router, tags=["Drift Detection"])
app.include_router(reliability.router, tags=["Reliability"])
app.include_router(history.router, tags=["History"])
app.include_router(adapt.router, tags=["Adaptation"])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    from api.dependencies import get_state
    state = get_state()
    return HealthResponse(
        status="healthy" if is_loaded() else "no_model",
        model_loaded=is_loaded(),
        data_years=len(state.get("df", [])),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
