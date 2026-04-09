"""
src.api.routes.health
======================
Health and status endpoints.

GET /health  → basic liveness check
GET /status  → pipeline statistics + model load status
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness check")
async def health(request: Request) -> dict:
    """Return ``{"status": "ok"}`` if the server is running."""
    registry = getattr(request.app.state, "model_registry", None)

    models_loaded: dict[str, bool] = {"cnn": False, "gpr": False}
    if registry is not None:
        models_loaded["cnn"] = getattr(registry, "cnn_loaded", False)
        models_loaded["gpr"] = getattr(registry, "gpr_loaded", False)

    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "pipeline_version": "3.0.0",
    }


@router.get("/status", summary="Pipeline statistics")
async def status(request: Request) -> dict:
    """Return accumulated pipeline statistics."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        return {"status": "not_ready"}

    stats: dict = pipeline.get_statistics()
    stats["status"] = "ready"
    return stats
