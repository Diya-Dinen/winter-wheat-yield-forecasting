"""
Shared dependencies for FastAPI routes.

Manages the global state: loaded model, conformal predictor, training data.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global state — populated on app startup
_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """Get the global application state."""
    return _state


def set_state(artifacts: dict[str, Any]) -> None:
    """Set the global application state from training artifacts."""
    _state.update(artifacts)
    logger.info("Application state loaded: model=%s, %d years of data",
                artifacts.get("model", "?"),
                len(artifacts.get("df", [])))


def is_loaded() -> bool:
    """Check if the model and data are loaded."""
    return "model" in _state and "df" in _state
