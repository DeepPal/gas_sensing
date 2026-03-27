"""
spectraagent.webapp.server
==========================
FastAPI application — all HTTP routes and WebSocket endpoints.

``create_app(simulate)`` is a factory used both by the CLI (``spectraagent start``)
and by the test suite (``TestClient(create_app(simulate=True))``).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from collections import deque

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import spectraagent
from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)

_STATIC_DIST = Path(__file__).resolve().parent / "static" / "dist"


# ---------------------------------------------------------------------------
# Broadcaster
# ---------------------------------------------------------------------------


class Broadcaster:
    """Thread-safe WebSocket fan-out.

    Adapted from ``dashboard.live_server._Broadcaster``.
    All WebSocket ``send_text`` calls are awaited in the asyncio event loop.
    """

    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    def connect(self, ws: WebSocket) -> None:
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.discard(ws)

    async def broadcast(self, message: str) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AcquisitionConfig(BaseModel):
    integration_time_ms: float = 50.0
    gas_label: str = "unknown"
    target_concentration: float | None = None


class CalibrationPoint(BaseModel):
    concentration: float
    delta_lambda: float


class AskRequest(BaseModel):
    query: str


# Module-level AgentBus singleton (created once per process, shared across requests)
_agent_bus = AgentBus()

_ASK_MODEL = "claude-sonnet-4-6"


def _get_ask_client():
    """Return anthropic.AsyncAnthropic for the /api/agents/ask endpoint, or None.

    Module-level so tests can patch it:
        with patch("spectraagent.webapp.server._get_ask_client", return_value=None):
            ...
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=api_key)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(simulate: bool = False) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    simulate:
        If True, use SimulationDriver regardless of config.
        Hardware connection is NOT started here — that happens in the CLI.
    """
    app = FastAPI(
        title="SpectraAgent",
        version=spectraagent.__version__,
        docs_url="/api/docs",
        redoc_url=None,
    )

    # CORS — allow all origins so LAN clients work without configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state for route handlers
    app.state.simulate = simulate
    app.state.driver = None
    app.state.plugin = None
    app.state.reference = None
    app.state.cached_ref = None

    # Store AgentBus on app.state so tests and other modules can reach it
    app.state.agent_bus = _agent_bus
    # Bounded event log for /api/agents/ask context (last 200 events)
    app.state.agent_events_log = deque(maxlen=200)

    @app.on_event("startup")
    async def _startup() -> None:
        """Wire AgentBus to the running event loop once uvicorn starts."""
        _agent_bus.setup_loop(asyncio.get_running_loop())

        # Start background task that populates agent_events_log for ask context
        async def _log_events() -> None:
            q = _agent_bus.subscribe()
            try:
                while True:
                    event = await q.get()
                    app.state.agent_events_log.append(event.to_dict())
            except asyncio.CancelledError:
                pass
            finally:
                _agent_bus.unsubscribe(q)

        asyncio.ensure_future(_log_events())

    # ------------------------------------------------------------------
    # Health endpoint
    # ------------------------------------------------------------------

    @app.get("/api/health")
    async def health() -> JSONResponse:
        driver = app.state.driver
        return JSONResponse({
            "status": "ok",
            "version": spectraagent.__version__,
            "hardware": driver.name if driver is not None else "not_connected",
            "simulate": app.state.simulate,
        })

    # ------------------------------------------------------------------
    # WebSocket: /ws/spectrum and /ws/trend
    # ------------------------------------------------------------------
    _spectrum_bc = Broadcaster()
    _trend_bc = Broadcaster()

    @app.websocket("/ws/spectrum")
    async def ws_spectrum(websocket: WebSocket) -> None:
        await websocket.accept()
        _spectrum_bc.connect(websocket)
        try:
            while True:
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            pass
        finally:
            _spectrum_bc.disconnect(websocket)

    @app.websocket("/ws/trend")
    async def ws_trend(websocket: WebSocket) -> None:
        await websocket.accept()
        _trend_bc.connect(websocket)
        try:
            while True:
                await asyncio.sleep(60)
        except WebSocketDisconnect:
            pass
        finally:
            _trend_bc.disconnect(websocket)

    # Store broadcasters on app.state so acquisition loop can reach them
    app.state.spectrum_bc = _spectrum_bc
    app.state.trend_bc = _trend_bc

    # ------------------------------------------------------------------
    # WebSocket: /ws/agent-events — streams AgentEvent JSON to clients
    # ------------------------------------------------------------------

    @app.websocket("/ws/agent-events")
    async def ws_agent_events(websocket: WebSocket) -> None:
        await websocket.accept()
        q = _agent_bus.subscribe()
        try:
            while True:
                event = await q.get()
                await websocket.send_text(event.to_json())
        except WebSocketDisconnect:
            pass
        finally:
            _agent_bus.unsubscribe(q)

    # ------------------------------------------------------------------
    # Acquisition API
    # ------------------------------------------------------------------
    _acq_config: dict[str, Any] = {
        "integration_time_ms": 50.0,
        "gas_label": "unknown",
        "target_concentration": None,
    }
    _session_active: dict[str, Any] = {"running": False, "session_id": None}
    _latest_spectrum: dict[str, Any] = {"wl": None, "intensities": None}

    @app.post("/api/acquisition/config")
    async def acq_config(cfg: AcquisitionConfig) -> JSONResponse:
        _acq_config.update(cfg.model_dump())
        if app.state.driver is not None:
            app.state.driver.set_integration_time_ms(cfg.integration_time_ms)
        return JSONResponse(_acq_config)

    @app.post("/api/acquisition/start")
    async def acq_start() -> JSONResponse:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_active["running"] = True
        _session_active["session_id"] = session_id
        return JSONResponse({"status": "started", "session_id": session_id})

    @app.post("/api/acquisition/stop")
    async def acq_stop() -> JSONResponse:
        _session_active["running"] = False
        return JSONResponse({"status": "stopped",
                             "session_id": _session_active.get("session_id")})

    @app.post("/api/acquisition/reference")
    async def acq_reference() -> JSONResponse:
        intensities = _latest_spectrum.get("intensities")
        if intensities is None:
            return JSONResponse(
                {"error": "No spectrum available yet — wait for first frame"},
                status_code=400,
            )
        app.state.reference = intensities
        app.state.cached_ref = None
        return JSONResponse({"status": "reference_captured", "peak_wavelength": None})

    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    @app.post("/api/calibration/add-point")
    async def calibration_add_point(point: CalibrationPoint) -> JSONResponse:
        """Add a calibration data point; CalibrationAgent re-fits all models."""
        calib_agent = getattr(app.state, "calibration_agent", None)
        if calib_agent is not None:
            calib_agent.add_point(point.concentration, point.delta_lambda)
        return JSONResponse({
            "status": "added",
            "concentration": point.concentration,
            "delta_lambda": point.delta_lambda,
        })

    @app.post("/api/calibration/suggest")
    async def calibration_suggest() -> JSONResponse:
        """Return the next recommended concentration from ExperimentPlannerAgent."""
        planner = getattr(app.state, "planner_agent", None)
        if planner is None:
            return JSONResponse({"suggestion": None, "reason": "planner_not_initialized"})
        suggested = planner.suggest()
        if suggested is None:
            return JSONResponse({"suggestion": None, "reason": "no_gpr_fitted"})
        return JSONResponse({"suggestion": suggested})

    # ------------------------------------------------------------------
    # Claude API — free-text query with SSE streaming response
    # ------------------------------------------------------------------

    @app.post("/api/agents/ask")
    async def agents_ask(request: AskRequest) -> StreamingResponse:
        """Stream a Claude response to a free-text query about the current session.

        Returns Server-Sent Events (SSE) with content-type text/event-stream.
        Each chunk: ``data: {"text": "...", "done": false}\\n\\n``
        Final frame: ``data: {"done": true}\\n\\n``
        """
        query = request.query
        events_log = list(app.state.agent_events_log)
        context = {
            "query": query,
            "last_20_agent_events": events_log[-20:],
        }

        async def _generate():
            client = _get_ask_client()
            if client is None:
                yield (
                    f'data: {json.dumps({"text": "Claude unavailable: set ANTHROPIC_API_KEY", "done": False})}\n\n'
                )
                yield f'data: {json.dumps({"done": True})}\n\n'
                return

            prompt = (
                f"You are a spectroscopy AI assistant. "
                f"Context about the current session:\n"
                f"{json.dumps(context, indent=2, default=str)}\n\n"
                f"User question: {query}\n\n"
                f"Respond concisely and accurately. "
                f"Only reference data that appears in the context."
            )
            try:
                async with client.messages.stream(
                    model=_ASK_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for chunk in stream.text_stream:
                        yield f'data: {json.dumps({"text": chunk, "done": False})}\n\n'
            except Exception as exc:
                log.warning("agents_ask: Claude streaming error: %s", exc)
                yield (
                    f'data: {json.dumps({"text": f"Error: {exc}", "done": False})}\n\n'
                )
            finally:
                yield f'data: {json.dumps({"done": True})}\n\n'

        return StreamingResponse(_generate(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Static files (React SPA) — mounted last so API routes take priority
    # ------------------------------------------------------------------
    _has_real_content = (
        _STATIC_DIST.exists()
        and any(f for f in _STATIC_DIST.iterdir() if f.name != ".gitkeep")
    )
    if _has_real_content:
        app.mount("/", StaticFiles(directory=str(_STATIC_DIST), html=True), name="static")
    else:
        @app.get("/")
        async def index_placeholder() -> JSONResponse:
            return JSONResponse({
                "message": "React frontend not yet built. Run: cd spectraagent/webapp/frontend && npm run build"
            })

    return app
