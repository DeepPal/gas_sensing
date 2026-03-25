"""
src.api.main
=============
FastAPI application factory for the Au-MIP LSPR inference server.

Quick start
-----------
::

    # From project root:
    python serve.py

    # Or directly with uvicorn:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

Interactive API docs available at:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc (ReDoc)

Architecture
------------
The ``lifespan`` context manager loads the ``RealTimePipeline`` and optional
``ModelRegistry`` **once** at startup and stores them in ``app.state``.  All
route handlers access these via FastAPI dependency injection (see
``src/api/dependencies.py``) — no global variables, no re-loading per request.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import health, predict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load pipeline and models at startup; release at shutdown."""
    log.info("Starting Au-MIP LSPR inference server...")

    # Load configuration
    try:
        from config.config_loader import load_config

        cfg = load_config()
        app.state.config = cfg
        log.info("Configuration loaded (%d top-level keys).", len(cfg))
    except Exception as exc:
        log.warning("Config load failed (%s) — using defaults.", exc)
        app.state.config = {}

    # Initialise pipeline
    try:
        from src.inference.realtime_pipeline import PipelineConfig, RealTimePipeline

        sensor_cfg = app.state.config.get("sensor", {})
        pipeline_config = PipelineConfig(
            target_wavelength=float(sensor_cfg.get("target_wavelength", 532.0)),
            calibration_slope=float(sensor_cfg.get("calibration_slope", 0.116)),
            calibration_intercept=float(sensor_cfg.get("calibration_intercept", 0.0)),
            reference_wavelength=float(sensor_cfg.get("reference_wavelength", 531.5)),
        )
        app.state.pipeline = RealTimePipeline(pipeline_config)
        log.info("RealTimePipeline initialised (v%s).", RealTimePipeline._VERSION)
    except Exception as exc:
        log.error("Pipeline initialisation failed: %s", exc)
        app.state.pipeline = None

    # Load ML models (optional — server runs without them)
    app.state.model_registry = None
    model_dir = Path(app.state.config.get("api", {}).get("model_dir", "models/registry"))
    if model_dir.exists():
        try:
            from src.models.registry import ModelRegistry

            registry = ModelRegistry()
            status = registry.load_all(str(model_dir))
            app.state.model_registry = registry
            log.info("ModelRegistry loaded from %s (status: %s).", model_dir, status)
        except Exception as exc:
            log.info("ModelRegistry not loaded (%s) — heuristic calibration only.", exc)

    yield  # Server runs here

    log.info("Shutting down inference server.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Au-MIP LSPR Gas Sensing API",
        description=(
            "Real-time spectral inference API for Au nanoparticle "
            "Molecularly Imprinted Polymer LSPR gas sensing. "
            "Primary signal: Δλ = λ_gas − λ_reference (nm)."
        ),
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow dashboard and local development
    app.add_middleware(
        CORSMiddleware,
        # Wildcard "*" is incompatible with allow_credentials=True (CORS spec §3.2.2)
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router)
    app.include_router(predict.router)

    # Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "message": "Au-MIP LSPR Gas Sensing API v3.0.0",
                "docs": "/docs",
                "health": "/health",
                "predict": "POST /predict",
            }
        )

    return app


# Module-level app instance (used by uvicorn)
app = create_app()
