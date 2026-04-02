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
from datetime import datetime, timezone
import hashlib
import hmac
import html
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import spectraagent
from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.session_writer import SessionWriter

log = logging.getLogger(__name__)

_STATIC_DIST = Path(__file__).resolve().parent / "static" / "dist"


def _is_nan(v: Any) -> bool:
    """Return True if v is float NaN (safe for None and non-float types)."""
    try:
        import math
        return v is not None and math.isnan(float(v))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Rate limiter (stdlib sliding window — no extra dependencies)
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Sliding-window rate limiter keyed by (client_ip, endpoint).

    Designed for LAN research use: protects Claude API quota from accidental
    hammering and prevents runaway report generation loops.

    Parameters
    ----------
    max_calls:
        Maximum number of requests allowed within ``window_s`` seconds.
    window_s:
        Sliding window size in seconds.
    """

    def __init__(self, max_calls: int, window_s: float) -> None:
        self._max = max_calls
        self._window = window_s
        self._history: dict[str, deque] = {}

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        dq = self._history.setdefault(key, deque())
        # Evict timestamps outside the window
        while dq and now - dq[0] > self._window:
            dq.popleft()
        if len(dq) >= self._max:
            return False
        dq.append(now)
        return True


def _int_from_env(name: str, default: int, minimum: int) -> int:
    """Read an integer env var safely, clamped to a lower bound."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    with contextlib.suppress(ValueError):
        return max(int(raw), minimum)
    return default


def _float_from_env(name: str, default: float, minimum: float) -> float:
    """Read a float env var safely, clamped to a lower bound."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    with contextlib.suppress(ValueError):
        return max(float(raw), minimum)
    return default


# One limiter instance per policy; shared across all requests.
_CLAUDE_RATE_MAX = _int_from_env("SPECTRAAGENT_CLAUDE_RATE_MAX", default=10, minimum=1)
_CLAUDE_RATE_WINDOW_S = _int_from_env(
    "SPECTRAAGENT_CLAUDE_RATE_WINDOW_S", default=60, minimum=1
)
_REPORT_RATE_MAX = _int_from_env("SPECTRAAGENT_REPORT_RATE_MAX", default=3, minimum=1)
_REPORT_RATE_WINDOW_S = _int_from_env(
    "SPECTRAAGENT_REPORT_RATE_WINDOW_S", default=60, minimum=1
)

_claude_limiter = _RateLimiter(max_calls=_CLAUDE_RATE_MAX, window_s=_CLAUDE_RATE_WINDOW_S)
_report_limiter = _RateLimiter(max_calls=_REPORT_RATE_MAX, window_s=_REPORT_RATE_WINDOW_S)


def _rate_limit_claude(request: Request) -> None:
    """FastAPI dependency — raises 429 when Claude rate limit is exceeded."""
    client = request.client.host if request.client else "unknown"
    if not _claude_limiter.is_allowed(client):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {_CLAUDE_RATE_MAX} Claude calls "
                f"per {_CLAUDE_RATE_WINDOW_S} seconds."
            ),
        )


def _rate_limit_report(request: Request) -> None:
    """FastAPI dependency — raises 429 when report rate limit is exceeded."""
    client = request.client.host if request.client else "unknown"
    if not _report_limiter.is_allowed(client):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: max {_REPORT_RATE_MAX} reports "
                f"per {_REPORT_RATE_WINDOW_S} seconds."
            ),
        )


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
    temperature_c: float | None = None   # room temperature at session start (°C)
    humidity_pct: float | None = None    # relative humidity (%) — LSPR sensitivity ~0.02 nm/°C


class CalibrationPoint(BaseModel):
    concentration: float
    delta_lambda: float


class AskRequest(BaseModel):
    query: str = Field(..., max_length=2000)


class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class AgentSettings(BaseModel):
    auto_explain: bool


class QualitySettings(BaseModel):
    saturation_threshold: float | None = None
    snr_warn_threshold: float | None = None


class DriftSettings(BaseModel):
    drift_threshold_nm_per_min: float | None = None
    window_frames: int | None = None


_ASK_MODEL = "claude-sonnet-4-6"


def _build_research_flow_payload(app: FastAPI) -> dict[str, Any]:
    """Build a step-by-step flow state for researchers and commercialization.

    This endpoint is designed as an operational coach: it reports current
    progress, identifies blockers, and recommends the next highest-impact step.
    """
    driver = getattr(app.state, "driver", None)
    plugin = getattr(app.state, "plugin", None)
    session_running = bool(getattr(app.state, "session_running", False))
    reference_ready = getattr(app.state, "reference", None) is not None
    analysis = getattr(app.state, "last_session_analysis", None)
    calib_agent = getattr(app.state, "calibration_agent", None)

    cal_r2 = None if analysis is None else analysis.calibration_r2
    mean_snr = None if analysis is None else analysis.mean_snr

    n_cal_points = 0
    if calib_agent is not None and hasattr(calib_agent, "data"):
        concentrations, _ = calib_agent.data
        n_cal_points = len(concentrations)

    cal_ready = n_cal_points >= 5
    analysis_ready = analysis is not None
    r2_ok = bool(cal_r2 is not None and float(cal_r2) >= 0.95)
    snr_ok = bool(mean_snr is not None and float(mean_snr) >= 3.0)

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    kb_available = False
    with contextlib.suppress(Exception):
        from spectraagent.webapp.agents.claude_agents import knowledge_backend_status

        kb_available = bool(knowledge_backend_status().get("knowledge_base_available"))

    checkpoints: list[dict[str, Any]] = [
        {
            "id": "hardware_connected",
            "title": "Connect hardware driver",
            "done": driver is not None,
            "impact": "high",
        },
        {
            "id": "physics_loaded",
            "title": "Load physics plugin",
            "done": plugin is not None,
            "impact": "high",
        },
        {
            "id": "reference_captured",
            "title": "Capture reference spectrum",
            "done": reference_ready,
            "impact": "high",
        },
        {
            "id": "session_recorded",
            "title": "Record a full acquisition session",
            "done": analysis_ready,
            "impact": "high",
        },
        {
            "id": "calibration_points",
            "title": "Collect at least 5 calibration points",
            "done": cal_ready,
            "value": n_cal_points,
            "target": 5,
            "impact": "high",
        },
        {
            "id": "quality_r2",
            "title": "Reach calibration R² >= 0.95",
            "done": r2_ok,
            "value": cal_r2,
            "target": 0.95,
            "impact": "medium",
        },
        {
            "id": "quality_snr",
            "title": "Reach mean SNR >= 3",
            "done": snr_ok,
            "value": mean_snr,
            "target": 3.0,
            "impact": "medium",
        },
        {
            "id": "ai_ready",
            "title": "Enable Claude API",
            "done": has_api_key,
            "impact": "medium",
        },
        {
            "id": "knowledge_grounded",
            "title": "Use domain-grounded AI context",
            "done": kb_available,
            "impact": "medium",
        },
    ]

    done_count = sum(1 for c in checkpoints if c["done"])
    readiness_score = int(round((done_count / len(checkpoints)) * 100))

    next_steps: list[str] = []
    if driver is None:
        next_steps.append("Connect or initialize a spectrometer driver.")
    if plugin is None:
        next_steps.append("Load a sensor physics plugin before acquisition.")
    if not reference_ready:
        next_steps.append("Capture a reference spectrum before trusting concentration estimates.")
    if not analysis_ready and not session_running:
        next_steps.append("Run at least one full start/stop acquisition session.")
    if n_cal_points < 5:
        next_steps.append("Add calibration points across the operating range (minimum 5).")
    if analysis_ready and not r2_ok:
        next_steps.append("Improve calibration fit quality (target R² >= 0.95).")
    if analysis_ready and not snr_ok:
        next_steps.append("Improve optical SNR with exposure/averaging/hardware setup.")
    if not has_api_key:
        next_steps.append("Set ANTHROPIC_API_KEY to unlock explainability and report generation.")
    if has_api_key and not kb_available:
        next_steps.append("Install/restore knowledge modules to avoid generic AI fallback.")

    if not next_steps:
        next_steps.append(
            "System is commercialization-ready for pilot trials: run reproducibility and stress-test lanes."
        )

    return {
        "readiness_score": readiness_score,
        "session_running": session_running,
        "checkpoints": checkpoints,
        "next_steps": next_steps,
        "commercialization_signal": "strong" if readiness_score >= 85 else "developing",
    }


def _is_finite_number(v: Any) -> bool:
    with contextlib.suppress(TypeError, ValueError):
        return math.isfinite(float(v))
    return False


def _quality_thresholds() -> dict[str, float]:
    """Return qualification criteria (env-overridable for deployment profiles)."""
    return {
        "min_calibration_points": _float_from_env(
            "SPECTRAAGENT_QUAL_MIN_CAL_POINTS", default=5.0, minimum=3.0
        ),
        "min_r2": _float_from_env("SPECTRAAGENT_QUAL_MIN_R2", default=0.95, minimum=0.0),
        "min_snr": _float_from_env("SPECTRAAGENT_QUAL_MIN_SNR", default=3.0, minimum=0.0),
        "max_abs_drift_nm_per_frame": _float_from_env(
            "SPECTRAAGENT_QUAL_MAX_ABS_DRIFT_NM_PER_FRAME", default=0.005, minimum=0.0
        ),
    }


def _latest_session_complete_payload(session: dict[str, Any] | None) -> dict[str, Any] | None:
    """Extract latest SessionAnalyzer summary from a persisted session record."""
    if not session:
        return None
    events = session.get("events", [])
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if event.get("type") == "session_complete" and isinstance(event.get("data"), dict):
            return dict(event["data"])
    return None


def _build_qualification_dossier(
    app: FastAPI,
    session_id: str | None,
    active_session: dict[str, Any],
) -> dict[str, Any]:
    """Build supplier-facing qualification dossier with pass/fail criteria."""
    thresholds = _quality_thresholds()
    analysis = getattr(app.state, "last_session_analysis", None)
    resolved_session_id = session_id or active_session.get("session_id")

    metrics: dict[str, Any] | None = None
    source = "none"

    if analysis is not None and (session_id is None or session_id == active_session.get("session_id")):
        metrics = {
            "calibration_n_points": getattr(analysis, "calibration_n_points", None),
            "calibration_r2": getattr(analysis, "calibration_r2", None),
            "mean_snr": getattr(analysis, "mean_snr", None),
            "lod_ppm": getattr(analysis, "lod_ppm", None),
            "loq_ppm": getattr(analysis, "loq_ppm", None),
            "drift_rate_nm_per_frame": getattr(analysis, "drift_rate_nm_per_frame", None),
            "summary_text": getattr(analysis, "summary_text", ""),
        }
        source = "live_analysis"

    if metrics is None and resolved_session_id:
        sw = getattr(app.state, "session_writer", None)
        session = None if sw is None else sw.get_session(str(resolved_session_id))
        payload = _latest_session_complete_payload(session)
        if payload is not None:
            metrics = payload
            source = "session_log"

    if metrics is None:
        return {
            "status": "insufficient_data",
            "session_id": resolved_session_id,
            "overall_pass": False,
            "source": source,
            "criteria": thresholds,
            "checks": [],
            "next_actions": [
                "Run a full acquisition session and stop it to generate SessionAnalyzer outputs.",
                "Collect calibration points (>= 5) and capture a reference spectrum.",
            ],
        }

    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        title: str,
        value: Any,
        target: Any,
        passed: bool,
        critical: bool,
        recommendation: str,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "title": title,
                "value": value,
                "target": target,
                "pass": passed,
                "critical": critical,
                "recommendation": recommendation,
            }
        )

    n_points = metrics.get("calibration_n_points")
    r2 = metrics.get("calibration_r2")
    snr = metrics.get("mean_snr")
    lod = metrics.get("lod_ppm")
    loq = metrics.get("loq_ppm")
    drift = metrics.get("drift_rate_nm_per_frame")

    min_pts = int(thresholds["min_calibration_points"])
    add_check(
        "cal_points",
        "Calibration points",
        n_points,
        f">= {min_pts}",
        _is_finite_number(n_points) and int(float(n_points)) >= min_pts,
        True,
        "Acquire additional calibration concentrations across low/mid/high range.",
    )
    add_check(
        "cal_r2",
        "Calibration R²",
        r2,
        f">= {thresholds['min_r2']:.2f}",
        _is_finite_number(r2) and float(r2) >= thresholds["min_r2"],
        True,
        "Improve baseline correction and repeat calibration with stable reference capture.",
    )
    add_check(
        "mean_snr",
        "Mean SNR",
        snr,
        f">= {thresholds['min_snr']:.1f}",
        _is_finite_number(snr) and float(snr) >= thresholds["min_snr"],
        True,
        "Increase integration time, improve optical alignment, or reduce mechanical noise.",
    )
    add_check(
        "lod_present",
        "LOD computed",
        lod,
        "finite",
        _is_finite_number(lod),
        True,
        "Ensure calibration includes low-concentration points and blank/noise characterization.",
    )
    add_check(
        "loq_present",
        "LOQ computed",
        loq,
        "finite",
        _is_finite_number(loq),
        True,
        "Ensure calibration includes quantifiable response region and repeatability data.",
    )
    add_check(
        "drift",
        "Absolute drift rate (nm/frame)",
        drift,
        f"<= {thresholds['max_abs_drift_nm_per_frame']:.6f}",
        _is_finite_number(drift) and abs(float(drift)) <= thresholds["max_abs_drift_nm_per_frame"],
        False,
        "Stabilize temperature/humidity and allow longer warm-up before measurement.",
    )

    passed = sum(1 for c in checks if c["pass"])
    critical_failed = [c for c in checks if c["critical"] and not c["pass"]]
    overall_pass = len(critical_failed) == 0
    pass_rate = passed / len(checks)
    score = int(round(pass_rate * 100))

    if overall_pass and score >= 95:
        tier = "gold"
    elif overall_pass and score >= 80:
        tier = "silver"
    elif overall_pass:
        tier = "bronze"
    else:
        tier = "not_qualified"

    next_actions = [c["recommendation"] for c in checks if not c["pass"]]
    if not next_actions:
        next_actions = [
            "Qualification criteria passed; proceed to pilot deployment and external validation package generation."
        ]

    return {
        "status": "ok",
        "session_id": resolved_session_id,
        "source": source,
        "overall_pass": overall_pass,
        "qualification_tier": tier,
        "score": score,
        "criteria": thresholds,
        "checks": checks,
        "next_actions": next_actions,
        "summary": metrics.get("summary_text"),
    }


def _dossier_artifact_dir() -> Path:
        """Directory where qualification dossier exports are written."""
        return Path(os.environ.get("SPECTRAAGENT_DOSSIER_DIR", "output/qualification"))


def _dossier_signature(payload_json: str) -> dict[str, Any]:
        """Create integrity signature metadata for exported dossier payload."""
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        secret = os.environ.get("SPECTRAAGENT_DOSSIER_SIGNING_KEY", "")
        if not secret:
                return {
                        "algorithm": "sha256",
                        "payload_sha256": payload_hash,
                        "signed": False,
                        "message": "Set SPECTRAAGENT_DOSSIER_SIGNING_KEY for HMAC signatures.",
                }
        mac = hmac.new(secret.encode("utf-8"), payload_json.encode("utf-8"), hashlib.sha256)
        return {
                "algorithm": "hmac-sha256",
                "payload_sha256": payload_hash,
                "signature": mac.hexdigest(),
                "signed": True,
        }


def _render_dossier_html(dossier: dict[str, Any]) -> str:
        """Render a simple standalone HTML dossier for procurement/review workflows."""
        session = html.escape(str(dossier.get("session_id") or "unknown"))
        tier = html.escape(str(dossier.get("qualification_tier") or "not_qualified"))
        score = html.escape(str(dossier.get("score") or "0"))
        overall = "PASS" if dossier.get("overall_pass") else "FAIL"
        checks = dossier.get("checks", [])
        rows = []
        for c in checks:
                title = html.escape(str(c.get("title", "")))
                value = html.escape(str(c.get("value", "n/a")))
                target = html.escape(str(c.get("target", "n/a")))
                passed = "PASS" if c.get("pass") else "FAIL"
                rows.append(f"<tr><td>{title}</td><td>{value}</td><td>{target}</td><td>{passed}</td></tr>")

        row_html = "\n".join(rows) if rows else "<tr><td colspan='4'>No checks available</td></tr>"
        return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Qualification Dossier - {session}</title>
    <style>
        body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #122; }}
        h1 {{ margin-bottom: 0; }}
        .meta {{ margin: 8px 0 20px; color: #334; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #b9c6d1; padding: 8px; text-align: left; }}
        th {{ background: #eaf1f6; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; background: #edf7ed; }}
    </style>
</head>
<body>
    <h1>Qualification Dossier</h1>
    <div class=\"meta\">Session: <strong>{session}</strong> | Overall: <strong>{overall}</strong> | Tier: <strong>{tier}</strong> | Score: <strong>{score}</strong></div>
    <table>
        <thead><tr><th>Check</th><th>Value</th><th>Target</th><th>Status</th></tr></thead>
        <tbody>{row_html}</tbody>
    </table>
</body>
</html>
"""


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
    app.state.last_session_id = None
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
        plugin = app.state.plugin
        quality_agent = getattr(app.state, "quality_agent", None)
        drift_agent = getattr(app.state, "drift_agent", None)
        knowledge_status: dict[str, Any] = {
            "knowledge_base_available": False,
            "knowledge_context_mode": "unknown",
            "knowledge_status": "unknown",
        }
        with contextlib.suppress(Exception):
            from spectraagent.webapp.agents.claude_agents import knowledge_backend_status

            knowledge_status = knowledge_backend_status()

        claude_api_key_configured = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return JSONResponse({
            "status": "ok",
            "version": spectraagent.__version__,
            "hardware": driver.name if driver is not None else "not_connected",
            "simulate": app.state.simulate,
            "physics_plugin": plugin.name if plugin is not None else "none",
            "integration_time_ms": driver.integration_time_ms if driver is not None and hasattr(driver, "integration_time_ms") else None,
            "quality_settings": quality_agent.settings if quality_agent is not None else {},
            "drift_settings": drift_agent.settings if drift_agent is not None else {},
            "claude_api_key_configured": claude_api_key_configured,
            "rate_limits": {
                "claude": {
                    "max_calls": _CLAUDE_RATE_MAX,
                    "window_seconds": _CLAUDE_RATE_WINDOW_S,
                },
                "report": {
                    "max_calls": _REPORT_RATE_MAX,
                    "window_seconds": _REPORT_RATE_WINDOW_S,
                },
            },
            **knowledge_status,
        })

    @app.get("/api/research-flow")
    async def research_flow() -> JSONResponse:
        """Return guided next steps from lab workflow to commercialization readiness."""
        return JSONResponse(_build_research_flow_payload(app))

    @app.get("/api/qualification/dossier")
    async def qualification_dossier(session_id: str | None = None) -> JSONResponse:
        """Return pass/fail qualification dossier for supplier-facing evidence."""
        return JSONResponse(_build_qualification_dossier(app, session_id, _session_active))

    @app.post("/api/qualification/dossier/export")
    async def qualification_dossier_export(
        session_id: str | None = None,
        artifact: str = "both",
    ) -> JSONResponse:
        """Export qualification dossier to JSON/HTML with integrity signature metadata."""
        artifact = artifact.lower().strip()
        if artifact not in {"json", "html", "both"}:
            raise HTTPException(status_code=422, detail="artifact must be one of: json, html, both")

        dossier = _build_qualification_dossier(app, session_id, _session_active)
        resolved_session_id = str(
            dossier.get("session_id") or getattr(app.state, "last_session_id", None) or "unknown"
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _dossier_artifact_dir() / resolved_session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        payload_json = json.dumps(dossier, indent=2, sort_keys=True, default=str)
        signature = _dossier_signature(payload_json)
        signature_payload = {
            "session_id": resolved_session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": artifact,
            "signature": signature,
        }

        paths: dict[str, str] = {}
        if artifact in {"json", "both"}:
            json_path = out_dir / f"qualification_dossier_{stamp}.json"
            json_path.write_text(payload_json, encoding="utf-8")
            paths["json"] = str(json_path)

        if artifact in {"html", "both"}:
            html_path = out_dir / f"qualification_dossier_{stamp}.html"
            html_path.write_text(_render_dossier_html(dossier), encoding="utf-8")
            paths["html"] = str(html_path)

        sig_path = out_dir / f"qualification_dossier_{stamp}.sig.json"
        sig_path.write_text(json.dumps(signature_payload, indent=2), encoding="utf-8")
        paths["signature"] = str(sig_path)

        return JSONResponse(
            {
                "status": "exported",
                "session_id": resolved_session_id,
                "artifact": artifact,
                "paths": paths,
                "signature": signature,
            }
        )

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
    # Expose on app.state so __main__.py can build a get_analyte lambda
    # that always returns the current gas label at call time.
    app.state._acq_config = _acq_config
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
        app.state.last_session_id = session_id
        app.state.session_frame_count = 0
        # Reset per-session agent state so cross-session history doesn't bleed.
        drift_agent = getattr(app.state, "drift_agent", None)
        if drift_agent is not None:
            drift_agent.reset()
        planner_agent = getattr(app.state, "planner_agent", None)
        if planner_agent is not None:
            planner_agent.reset()
        app.state.session_events = []
        import time as _time
        app.state.session_start_monotonic = _time.monotonic()
        sw = getattr(app.state, "session_writer", None)
        if sw is not None:
            meta = {
                "gas_label": _acq_config.get("gas_label", "unknown"),
                "target_concentration": _acq_config.get("target_concentration"),
                "hardware": getattr(getattr(app.state, "driver", None), "name", "unknown"),
                "temperature_c": _acq_config.get("temperature_c"),
                "humidity_pct": _acq_config.get("humidity_pct"),
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
        session_id = _session_active.get("session_id") or "unknown"
        gas_label = _acq_config.get("gas_label", "unknown")
        analysis = None
        try:
            from src.inference.session_analyzer import SessionAnalyzer
            analysis = SessionAnalyzer().analyze(session_events, frame_count)
            app.state.last_session_analysis = analysis

            # Build the full session_complete event payload — includes all
            # calibration metrics needed by SensorHealthAgent and context builders.
            session_data: dict = {
                "session_id": session_id,
                "gas_label": gas_label,
                "lod_ppm": analysis.lod_ppm,
                "lob_ppm": getattr(analysis, "lob_ppm", None),
                "lod_ci_lower": getattr(analysis, "lod_ci_lower", None),
                "lod_ci_upper": getattr(analysis, "lod_ci_upper", None),
                "loq_ppm": analysis.loq_ppm,
                "calibration_r2": analysis.calibration_r2,
                "calibration_rmse_ppm": getattr(analysis, "calibration_rmse_ppm", None),
                "calibration_n_points": getattr(analysis, "calibration_n_points", 0),
                "mean_snr": analysis.mean_snr,
                "drift_rate_nm_per_frame": analysis.drift_rate_nm_per_frame,
                "frame_count": analysis.frame_count,
                "summary": analysis.summary_text,
                # Kinetics (B1)
                "tau_63_s": getattr(analysis, "tau_63_s", None),
                "tau_95_s": getattr(analysis, "tau_95_s", None),
                "k_on_per_s": getattr(analysis, "k_on_per_s", None),
                "kinetics_fit_r2": getattr(analysis, "kinetics_fit_r2", None),
                # Environmental metadata (B5)
                "temperature_c": _acq_config.get("temperature_c"),
                "humidity_pct": _acq_config.get("humidity_pct"),
            }
            bus = getattr(app.state, "agent_bus", None)
            if bus is not None:
                from spectraagent.webapp.agent_bus import AgentEvent
                bus.emit(AgentEvent(
                    source="SessionAnalyzer",
                    level="info",
                    type="session_complete",
                    data=session_data,
                    text=analysis.summary_text,
                ))
        except Exception as exc:
            log.warning("Post-session analysis failed: %s", exc)

        # Wire SensorMemory: record lightweight session log regardless of
        # whether full analysis succeeded.  Records calibration outcomes when
        # analysis is available so SensorHealthAgent has history to reason from.
        try:
            memory = getattr(app.state, "sensor_memory", None)
            if memory is not None:
                import datetime as _dt
                now_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
                memory.record_session(
                    session_id=session_id,
                    analyte=gas_label,
                    frame_count=frame_count,
                    stopped_at=now_utc,
                )
                if analysis is not None and analysis.calibration_r2 is not None:
                    from spectraagent.knowledge.sensor_memory import CalibrationObservation
                    # Pull best model + sensitivity from CalibrationAgent last state
                    cal_agent = getattr(app.state, "calibration_agent", None)
                    best_model = "unknown"
                    sensitivity: float | None = None
                    if cal_agent is not None:
                        best_model = getattr(cal_agent, "_last_best_model", "unknown") or "unknown"
                        sensitivity = getattr(cal_agent, "_last_sensitivity_nm_per_ppm", None)
                    memory.record_calibration(CalibrationObservation(
                        session_id=session_id,
                        timestamp_utc=now_utc,
                        analyte=gas_label,
                        sensitivity_nm_per_ppm=sensitivity,
                        lod_ppm=analysis.lod_ppm if not _is_nan(analysis.lod_ppm) else None,
                        loq_ppm=analysis.loq_ppm if not _is_nan(analysis.loq_ppm) else None,
                        r_squared=analysis.calibration_r2,
                        rmse_ppm=getattr(analysis, "calibration_rmse_ppm", None),
                        calibration_model=best_model,
                        n_calibration_points=getattr(analysis, "calibration_n_points", 0),
                        reference_peak_nm=getattr(app.state, "ref_peak_nm", None),
                        conformal_coverage=None,
                        tau_63_s=getattr(analysis, "tau_63_s", None),
                        reference_fwhm_nm=getattr(app.state, "ref_fwhm_nm", None),
                    ))
        except Exception as exc:
            log.warning("SensorMemory record failed: %s", exc)

        # Clear events for next session
        app.state.session_events = []

        active: dict[str, Any] = _session_active
        return JSONResponse({"status": "stopped",
                             "session_id": active.get("session_id")})

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

        import numpy as _np

        from src.features.lspr_features import detect_all_peaks, fit_lorentzian_peak
        wl_np = _np.asarray(latest_spectrum.get("wl", []))
        int_np = _np.asarray(intensities)
        plugin = getattr(app.state, "plugin", None)

        # ── Detect ALL spectral peaks (multi-peak sensor support) ──────────
        # Works for any sensor: single-peak, multi-peak, multi-analyte.
        # Peak positions are discovered at runtime — never hardcoded.
        ref_peak_wls: list[float] = []
        if len(wl_np) > 0:
            try:
                cfg_obj = getattr(app.state, "cfg", None)
                smin = cfg_obj.physics.search_min_nm if cfg_obj else float(wl_np[0])
                smax = cfg_obj.physics.search_max_nm if cfg_obj else float(wl_np[-1])
                ref_peak_wls = detect_all_peaks(wl_np, int_np, search_min=smin, search_max=smax)
            except Exception as exc:
                log.warning("Multi-peak detection failed: %s", exc)

        ref_peak: float | None = ref_peak_wls[0] if ref_peak_wls else None
        app.state.ref_peak_nm = ref_peak
        app.state.ref_peak_wls = ref_peak_wls

        # Update plugin so subsequent extract_features calls use discovered peaks
        if plugin is not None and ref_peak_wls and hasattr(plugin, "update_from_reference"):
            plugin.update_from_reference(ref_peak_wls)

        # Pre-compute cached Lorentzian fit for the primary peak (saves ~5 ms/frame)
        if plugin is not None and ref_peak is not None:
            try:
                app.state.cached_ref = plugin.compute_reference_cache(wl_np, int_np)
            except Exception as exc:
                log.warning("Reference cache build failed: %s", exc)
                app.state.cached_ref = None
        else:
            app.state.cached_ref = None

        # ── Extract FWHM of primary peak for sensor health tracking (B4) ──
        ref_fwhm: float | None = None
        if ref_peak is not None:
            try:
                lfit = fit_lorentzian_peak(wl_np, int_np, peak_wl_init=ref_peak)
                if lfit is not None:
                    ref_fwhm = lfit[1]
                    app.state.ref_fwhm_nm = ref_fwhm
            except Exception as exc:
                log.debug("Reference FWHM extraction failed: %s", exc)

        # Propagate to RealTimePipeline
        pipeline = getattr(app.state, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "set_reference"):
            try:
                pipeline.set_reference(int_np)
            except Exception as exc:
                log.warning("Pipeline reference set failed: %s", exc)

        return JSONResponse({
            "status": "reference_captured",
            "peak_wavelength": ref_peak,           # primary peak (backward compat)
            "peak_wavelengths": ref_peak_wls,      # all detected peaks
            "n_peaks": len(ref_peak_wls),
            "fwhm_nm": ref_fwhm,
        })

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
    # Multi-analyte calibration API (sensitivity matrix + mixture deconvolution)
    # ------------------------------------------------------------------

    class SensitivityFitRequest(BaseModel):
        """Fit sensitivity matrix from single-analyte calibration data."""
        analytes: list[str]
        n_peaks: int
        calibration_data: list[dict]
        # Each entry: {analyte, peak_idx, conc_ppm: [..], shifts_nm: [..]}

    class MixtureInferenceRequest(BaseModel):
        """Estimate analyte concentrations from observed peak shifts."""
        delta_lambda: list[float]          # observed peak shifts (nm), one per peak
        analytes: list[str]
        S_matrix: list[list[float]]        # [[S_00, S_01, ...], [S_10, ...]] (N×M)
        Kd_matrix: list[list[float]] | None = None   # K_d matrix (ppm), same shape; null = linear
        use_nonlinear: bool = False

    class SimGenerateRequest(BaseModel):
        """Generate a batch of synthetic spectra from the physics simulation."""
        peak_nm: float = 700.0
        fwhm_nm: float = 20.0
        wl_start: float = 500.0
        wl_end: float = 900.0
        analyte_name: str = "Gas"
        sensitivity_nm_per_ppm: float = -0.5
        tau_s: float = 30.0
        kd_ppm: float = 100.0
        concentrations: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
        n_sessions: int = 5
        random_seed: int = 42

    @app.post("/api/calibration/sensitivity-matrix/fit")
    async def calibration_sensitivity_fit(req: SensitivityFitRequest) -> JSONResponse:
        """Fit a sensitivity matrix from single-analyte calibration runs.

        Returns the fitted S matrix, condition number, R² per entry, and
        LOD estimates in mixture context.
        """
        try:
            from src.calibration.sensitivity_matrix import SensitivityMatrix
            sm = SensitivityMatrix(req.analytes, req.n_peaks)
            for entry in req.calibration_data:
                sm.fit_analyte(
                    analyte=entry["analyte"],
                    peak_idx=int(entry["peak_idx"]),
                    conc_ppm=entry["conc_ppm"],
                    shifts_nm=entry["shifts_nm"],
                )
            summary = sm.summary()
            lod = sm.compute_lod_mixture()
            return JSONResponse({
                "status": "fitted",
                "S_matrix": summary["S_matrix"],
                "condition_number": summary["condition_number"],
                "rank": summary["rank"],
                "r2_per_entry": [
                    {"analyte": e["analyte"], "peak": e["peak"], "r2": e["r_squared"]}
                    for e in summary["entries"]
                ],
                "lod_mixture_ppm": lod,
            })
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/inference/mixture")
    async def inference_mixture(req: MixtureInferenceRequest) -> JSONResponse:
        """Estimate all analyte concentrations from a peak-shift observation.

        Uses the linear pseudoinverse (fast) or non-linear Langmuir solver
        (accurate in saturation regime) as requested.
        """
        try:
            import numpy as np

            from src.calibration.mixture_deconvolution import deconvolve_mixture
            S = np.array(req.S_matrix, dtype=float)
            Kd = np.array(req.Kd_matrix, dtype=float) if req.Kd_matrix else None
            dl = np.array(req.delta_lambda, dtype=float)
            result = deconvolve_mixture(dl, req.analytes, S, Kd=Kd,
                                        use_nonlinear=req.use_nonlinear)
            return JSONResponse({
                "concentrations_ppm": result.concentrations,
                "residual_nm": result.residual_nm,
                "solver": result.solver,
                "success": result.success,
                "predicted_shifts_nm": result.predicted_shifts.tolist(),
            })
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/analytes")
    async def list_analytes() -> JSONResponse:
        """List analytes registered in the current sensor configuration.

        Returns analyte names, peak count, and S matrix if available from
        the active physics plugin.
        """
        plugin = app.state.plugin
        if plugin is None:
            return JSONResponse({"analytes": [], "n_peaks": 0, "S_matrix": None})
        sensor_cfg = getattr(plugin, "_cfg", None)
        if sensor_cfg is None:
            return JSONResponse({"analytes": [], "n_peaks": 0, "S_matrix": None})
        return JSONResponse({
            "analytes": [a.name for a in sensor_cfg.analytes],
            "n_peaks": len(sensor_cfg.peaks),
            "peak_wavelengths_nm": [p.center_nm for p in sensor_cfg.peaks],
            "S_matrix": sensor_cfg.sensitivity_matrix.tolist() if sensor_cfg.analytes else None,
        })

    @app.post("/api/simulation/generate")
    async def simulation_generate(req: SimGenerateRequest) -> JSONResponse:
        """Generate a synthetic calibration dataset from the physics simulation.

        Returns a summary of the generated dataset including mean peak shifts
        per concentration level, for use in sensitivity matrix fitting.
        """
        try:
            from src.simulation.dataset_generator import DatasetConfig, DatasetGenerator
            from src.simulation.gas_response import make_analyte, make_single_peak_sensor
            sensor = make_single_peak_sensor(req.peak_nm, req.fwhm_nm, req.wl_start, req.wl_end)
            sensor.analytes = [make_analyte(
                req.analyte_name, 1,
                req.sensitivity_nm_per_ppm,
                tau_s=req.tau_s,
                kd_ppm=req.kd_ppm,
            )]
            cfg = DatasetConfig(
                sensor_config=sensor,
                analyte_names=[req.analyte_name],
                concentration_levels=req.concentrations,
                n_sessions=req.n_sessions,
                random_seed=req.random_seed,
                domain_randomize=True,
            )
            df = DatasetGenerator(cfg).generate_calibration_dataset()
            # Aggregate: mean shift per concentration
            summary = (
                df.groupby("concentration_ppm")["peak_shift_0"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .rename(columns={"mean": "mean_shift_nm", "std": "std_shift_nm", "count": "n"})
                .to_dict(orient="records")
            )
            return JSONResponse({
                "status": "ok",
                "analyte": req.analyte_name,
                "n_sessions": req.n_sessions,
                "n_rows": len(df),
                "calibration_summary": summary,
            })
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

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
    async def reports_generate(
        request: ReportRequest,
        _rl: None = Depends(_rate_limit_report),
    ) -> JSONResponse:
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
                status_code=503,
                detail=(
                    "Claude unavailable — check ANTHROPIC_API_KEY is set and valid, "
                    "or see agent events for the specific error."
                ),
            )
        return JSONResponse({"report": text, "session_id": request.session_id})

    # ------------------------------------------------------------------
    # Agent settings — runtime toggle for auto-explain
    # ------------------------------------------------------------------

    @app.put("/api/agents/quality-settings")
    def agents_quality_settings(settings: QualitySettings) -> JSONResponse:
        """Update QualityAgent thresholds at runtime."""
        qa = getattr(app.state, "quality_agent", None)
        if qa is not None:
            qa.configure(
                saturation_threshold=settings.saturation_threshold,
                snr_warn_threshold=settings.snr_warn_threshold,
            )
            return JSONResponse({"status": "updated", **qa.settings})
        return JSONResponse({"status": "agent_not_ready"})

    @app.get("/api/agents/quality-settings")
    def agents_quality_settings_get() -> JSONResponse:
        qa = getattr(app.state, "quality_agent", None)
        if qa is None:
            return JSONResponse({})
        return JSONResponse(qa.settings)

    @app.put("/api/agents/drift-settings")
    def agents_drift_settings(settings: DriftSettings) -> JSONResponse:
        """Update DriftAgent thresholds at runtime."""
        da = getattr(app.state, "drift_agent", None)
        if da is not None:
            da.configure(
                drift_threshold_nm_per_min=settings.drift_threshold_nm_per_min,
                window_frames=settings.window_frames,
            )
            return JSONResponse({"status": "updated", **da.settings})
        return JSONResponse({"status": "agent_not_ready"})

    @app.get("/api/agents/drift-settings")
    def agents_drift_settings_get() -> JSONResponse:
        da = getattr(app.state, "drift_agent", None)
        if da is None:
            return JSONResponse({})
        return JSONResponse(da.settings)

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
    async def agents_ask(
        request: AskRequest,
        _rl: None = Depends(_rate_limit_claude),
    ) -> StreamingResponse:
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
