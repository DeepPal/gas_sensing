"""
src.api.dependencies
=====================
FastAPI dependency injection: provides a single shared ``RealTimePipeline``
instance loaded once at server startup.

Usage in route handlers
-----------------------
::

    from fastapi import Depends
    from src.api.dependencies import get_pipeline

    @router.post("/predict")
    async def predict(reading: SpectrumReading, pipeline=Depends(get_pipeline)):
        result = pipeline.process_spectrum(...)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request

log = logging.getLogger(__name__)


def get_pipeline(request: Request) -> Any:
    """Return the shared ``RealTimePipeline`` from application state.

    Raises
    ------
    HTTPException (503)
        If the pipeline has not been initialised (startup failed).
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialised. Server is still starting up.",
        )
    return pipeline


def get_model_registry(request: Request) -> Any:
    """Return the shared ``ModelRegistry`` from application state."""
    registry = getattr(request.app.state, "model_registry", None)
    return registry  # None is acceptable — models are optional


def get_config(request: Request) -> dict:
    """Return the loaded configuration dict from application state."""
    cfg = getattr(request.app.state, "config", {})
    return cfg
