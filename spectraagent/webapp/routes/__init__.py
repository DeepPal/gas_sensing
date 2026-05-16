"""FastAPI route modules for the SpectraAgent webapp."""
from .acquisition import router as acquisition_router
from .agents import router as agents_router
from .reports import router as reports_router
from .sessions import router as sessions_router

__all__ = [
    "acquisition_router",
    "agents_router",
    "reports_router",
    "sessions_router",
]
