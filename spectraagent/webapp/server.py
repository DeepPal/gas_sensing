"""
spectraagent.webapp.server
==========================
FastAPI application — all HTTP routes and WebSocket endpoints.

``create_app(simulate)`` is a factory used both by the CLI (``spectraagent start``)
and by the test suite (``TestClient(create_app(simulate=True))``).
"""
from __future__ import annotations

import asyncio
from collections import deque
import contextlib
from contextlib import asynccontextmanager
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import spectraagent
from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.session_writer import SessionWriter

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
    query: str = Field(..., max_length=2000)


class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class AgentSettings(BaseModel):
    auto_explain: bool


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
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manage per-app async resources for startup and shutdown."""
        loop = asyncio.get_running_loop()
        app.state.asyncio_loop = loop
        app.state.agent_bus.setup_loop(loop)

        async def _log_events() -> None:
            q = app.state.agent_bus.subscribe()
            try:
                while True:
                    event = await q.get()
                    event_dict = event.to_dict()
                    app.state.agent_events_log.append(event_dict)
                    sw = getattr(app.state, "session_writer", None)
                    if sw is not None:
                        sw.append_event(event_dict)
            except asyncio.CancelledError:
                pass
            finally:
                app.state.agent_bus.unsubscribe(q)

        app.state.log_events_task = asyncio.ensure_future(_log_events())
        try:
            for callback in list(app.state.startup_callbacks):
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            yield
        finally:
            for callback in reversed(list(app.state.shutdown_callbacks)):
                result = callback()
                if asyncio.iscoroutine(result):
                    await result

            task = getattr(app.state, "log_events_task", None)
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            claude_runner = getattr(app.state, "claude_runner", None)
            if claude_runner is not None:
                with contextlib.suppress(Exception):
                    claude_runner.stop()

            if app.state.session_running:
                sw = getattr(app.state, "session_writer", None)
                if sw is not None:
                    sw.stop_session(frame_count=int(app.state.session_frame_count))
                app.state.session_running = False

    app = FastAPI(
        title="SpectraAgent",
        version=spectraagent.__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
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
    app.state.latest_spectrum = None
    app.state.asyncio_loop = None
    app.state.session_running = False
    app.state.session_frame_count = 0
    app.state.startup_callbacks = []
    app.state.shutdown_callbacks = []

    # Keep AgentBus scoped to this app instance.
    agent_bus = AgentBus()
    app.state.agent_bus = agent_bus
    # Bounded event log for /api/agents/ask context (last 200 events)
    app.state.agent_events_log = deque(maxlen=200)
    app.state.session_writer = SessionWriter()

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
        q = agent_bus.subscribe()
        try:
            while True:
                event = await q.get()
                await websocket.send_text(event.to_json())
        except WebSocketDisconnect:
            pass
        finally:
            agent_bus.unsubscribe(q)

    # ------------------------------------------------------------------
    # Acquisition API
    # ------------------------------------------------------------------
    _acq_config: dict[str, Any] = {
        "integration_time_ms": 50.0,
        "gas_label": "unknown",
        "target_concentration": None,
    }
    _session_active: dict[str, Any] = {"running": False, "session_id": None}

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
        app.state.session_running = True
        app.state.session_frame_count = 0
        sw = getattr(app.state, "session_writer", None)
        if sw is not None:
            meta = {
                "gas_label": _acq_config.get("gas_label", "unknown"),
                "target_concentration": _acq_config.get("target_concentration"),
                "hardware": getattr(getattr(app.state, "driver", None), "name", "unknown"),
            }
            sw.start_session(session_id, meta)
        return JSONResponse({"status": "started", "session_id": session_id})

    @app.post("/api/acquisition/stop")
    async def acq_stop() -> JSONResponse:
        _session_active["running"] = False
        app.state.session_running = False
        frame_count = int(app.state.session_frame_count)
        sw = getattr(app.state, "session_writer", None)
        if sw is not None:
            sw.stop_session(frame_count=frame_count)

        # Auto-run SessionAnalyzer and emit results to the agent bus
        session_events = getattr(app.state, "session_events", [])
        try:
            from src.inference.session_analyzer import SessionAnalyzer
            analysis = SessionAnalyzer().analyze(session_events, frame_count)
            app.state.last_session_analysis = analysis
            bus = getattr(app.state, "agent_bus", None)
            if bus is not None:
                from spectraagent.webapp.agent_bus import AgentEvent
                bus.emit(AgentEvent(
                    source="SessionAnalyzer",
                    level="info",
                    type="session_complete",
                    data={
                        "lod_ppm": analysis.lod_ppm,
                        "loq_ppm": analysis.loq_ppm,
                        "calibration_r2": analysis.calibration_r2,
                        "mean_snr": analysis.mean_snr,
                        "drift_rate_nm_per_frame": analysis.drift_rate_nm_per_frame,
                        "frame_count": analysis.frame_count,
                        "summary": analysis.summary_text,
                    },
                    text=analysis.summary_text,
                ))
        except Exception as exc:
            log.warning("Post-session analysis failed: %s", exc)
        # Clear events for next session
        app.state.session_events = []

        return JSONResponse({"status": "stopped",
                             "session_id": _session_active.get("session_id")})

    @app.post("/api/acquisition/reference")
    async def acq_reference() -> JSONResponse:
        latest_spectrum = getattr(app.state, "latest_spectrum", None)
        intensities = None if latest_spectrum is None else latest_spectrum.get("intensities")
        if intensities is None:
            return JSONResponse(
                {"error": "No spectrum available yet — wait for first frame"},
                status_code=400,
            )
        app.state.reference = intensities
        app.state.cached_ref = None

        # Propagate to RealTimePipeline (feature extraction + calibration stages)
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "set_reference"):
            try:
                import numpy as _np
                pipeline.set_reference(_np.asarray(intensities))
            except Exception as exc:
                log.warning("Pipeline reference set failed: %s", exc)

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
    # Session API — list and retrieve saved sessions
    # ------------------------------------------------------------------

    @app.get("/api/sessions")
    def sessions_list() -> JSONResponse:
        """Return all session metadata dicts, newest first."""
        sw = getattr(app.state, "session_writer", None)
        if sw is None:
            return JSONResponse([])
        return JSONResponse(sw.list_sessions())

    @app.get("/api/sessions/{session_id}")
    def sessions_get(session_id: str) -> JSONResponse:
        """Return metadata + last 100 agent events for a session, or 404."""
        sw = getattr(app.state, "session_writer", None)
        if sw is None:
            raise HTTPException(status_code=404, detail="Session not found")
        data = sw.get_session(session_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return JSONResponse(data)

    # ------------------------------------------------------------------
    # Reports API — generate prose report for a completed session
    # ------------------------------------------------------------------

    @app.post("/api/reports/generate")
    async def reports_generate(request: ReportRequest) -> JSONResponse:
        """Call ReportWriter to generate a Methods+Results prose report.

        Returns ``{"report": "<text>", "session_id": "<id>"}`` on success.
        Returns 503 when ReportWriter is not available or Claude is unreachable.
        """
        writer = getattr(app.state, "report_writer", None)
        if writer is None:
            raise HTTPException(status_code=503, detail="ReportWriter not available")

        # Build context from session data if available
        sw = getattr(app.state, "session_writer", None)
        context: dict = {"session_id": request.session_id}
        if sw is not None:
            session_data = sw.get_session(request.session_id)
            if session_data is not None:
                context.update(session_data)

        text = await writer.write(context)
        if text is None:
            raise HTTPException(
                status_code=503, detail="Claude unavailable: set ANTHROPIC_API_KEY"
            )
        return JSONResponse({"report": text, "session_id": request.session_id})

    # ------------------------------------------------------------------
    # Agent settings — runtime toggle for auto-explain
    # ------------------------------------------------------------------

    @app.put("/api/agents/settings")
    def agents_settings(settings: AgentSettings) -> JSONResponse:
        """Toggle auto_explain for AnomalyExplainer and ExperimentNarrator.

        Graceful: if agents are not yet created (no API key configured),
        the request still returns 200 — there is nothing to toggle.
        """
        anomaly = getattr(app.state, "anomaly_explainer", None)
        narrator = getattr(app.state, "experiment_narrator", None)
        if anomaly is not None:
            anomaly.set_auto_explain(settings.auto_explain)
        if narrator is not None:
            narrator.set_auto_explain(settings.auto_explain)
        return JSONResponse({"auto_explain": settings.auto_explain})

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
                    f'data: {json.dumps({"text": "An error occurred while streaming the response.", "done": False})}\n\n'
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
