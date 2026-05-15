"""
spectraagent.webapp.server
==========================
FastAPI application factory.

``create_app(simulate)`` is used by both the CLI (``spectraagent start``)
and the test suite (``TestClient(create_app(simulate=True))``).

Route handlers live in ``spectraagent.webapp.routes.*``:
  - acquisition  — /api/acquisition/*, /api/calibration/*, /api/analytes, /api/simulation/*
  - sessions     — /api/sessions/*
  - reports      — /api/reports/*
  - agents       — /api/agents/*
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
from typing import Any, TypeGuard
import zipfile

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
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
# Pydantic models (kept here for backward-compat; also re-exported from routes._models)
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


# ---------------------------------------------------------------------------
# Helper functions used by route modules (imported via spectraagent.webapp.server)
# ---------------------------------------------------------------------------


def _is_finite_number(v: Any) -> bool:
    with contextlib.suppress(TypeError, ValueError):
        return math.isfinite(float(v))
    return False


def _is_float_convertible(v: Any) -> TypeGuard[float | int]:
    """Type guard: returns True only if v is a valid float and finite."""
    return _is_finite_number(v)


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


def _compute_rsd_pct(values: list[float]) -> float | None:
    """Return relative standard deviation in percent for finite samples."""
    if len(values) < 2:
        return None
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if abs(mean) < 1e-12:
        return None
    return float((std / abs(mean)) * 100.0)


def _build_reproducibility_overview(app: FastAPI, session_id: str | None) -> dict[str, Any]:
    """Summarize cross-session reproducibility from recent session_complete payloads."""
    sw = getattr(app.state, "session_writer", None)
    if sw is None:
        return {"available": False, "reason": "session_writer_unavailable"}

    sessions = sw.list_sessions()
    if not sessions:
        return {"available": False, "reason": "no_sessions_recorded"}

    window_n = _int_from_env("SPECTRAAGENT_REPRO_WINDOW", default=5, minimum=2)
    recent_ids: list[str] = []
    if session_id:
        recent_ids.append(str(session_id))
    for meta in sessions:
        sid = str(meta.get("session_id", "")).strip()
        if sid and sid not in recent_ids:
            recent_ids.append(sid)
        if len(recent_ids) >= window_n:
            break

    payloads: list[dict[str, Any]] = []
    for sid in recent_ids:
        payload = _latest_session_complete_payload(sw.get_session(sid))
        if payload is not None:
            payloads.append(payload)

    if len(payloads) < 2:
        return {
            "available": False,
            "reason": "insufficient_completed_sessions",
            "session_count": len(payloads),
            "required": 2,
        }

    lod_vals = [float(p["lod_ppm"]) for p in payloads if _is_finite_number(p.get("lod_ppm"))]
    loq_vals = [float(p["loq_ppm"]) for p in payloads if _is_finite_number(p.get("loq_ppm"))]
    r2_vals = [float(p["calibration_r2"]) for p in payloads if _is_finite_number(p.get("calibration_r2"))]

    lod_rsd = _compute_rsd_pct(lod_vals)
    loq_rsd = _compute_rsd_pct(loq_vals)
    r2_mean = float(np.mean(r2_vals)) if r2_vals else None
    r2_min = float(np.min(r2_vals)) if r2_vals else None

    reasons: list[str] = []
    if len(payloads) >= 3:
        if lod_rsd is not None and lod_rsd > 20.0:
            reasons.append(f"LOD RSD {lod_rsd:.1f}% exceeds 20%")
        if loq_rsd is not None and loq_rsd > 20.0:
            reasons.append(f"LOQ RSD {loq_rsd:.1f}% exceeds 20%")
        if r2_min is not None and r2_min < 0.95:
            reasons.append(f"Minimum calibration R² {r2_min:.4f} below 0.95")
        batch_ready = len(reasons) == 0
    else:
        batch_ready = None
        reasons.append("At least 3 completed sessions are recommended for reproducibility acceptance")

    return {
        "available": True,
        "window_sessions": len(recent_ids),
        "session_count": len(payloads),
        "lod_rsd_pct": lod_rsd,
        "loq_rsd_pct": loq_rsd,
        "r2_mean": r2_mean,
        "r2_min": r2_min,
        "batch_ready": batch_ready,
        "notes": reasons,
    }


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
            "lol_ppm": getattr(analysis, "lol_ppm", None),
            "kinetics_fit_r2": getattr(analysis, "kinetics_fit_r2", None),
            "tau_63_s": getattr(analysis, "tau_63_s", None),
            "interval_coverage": getattr(analysis, "interval_coverage", None),
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
    lol_ppm = metrics.get("lol_ppm")
    kinetics_fit_r2 = metrics.get("kinetics_fit_r2")
    tau_63_s = metrics.get("tau_63_s")
    interval_coverage = metrics.get("interval_coverage")

    min_pts = int(thresholds["min_calibration_points"])
    add_check(
        "cal_points",
        "Calibration points",
        n_points,
        f">= {min_pts}",
        _is_float_convertible(n_points) and int(float(n_points)) >= min_pts,
        True,
        "Acquire additional calibration concentrations across low/mid/high range.",
    )
    add_check(
        "cal_r2",
        "Calibration R²",
        r2,
        f">= {thresholds['min_r2']:.2f}",
        _is_float_convertible(r2) and float(r2) >= thresholds["min_r2"],
        True,
        "Improve baseline correction and repeat calibration with stable reference capture.",
    )
    add_check(
        "mean_snr",
        "Mean SNR",
        snr,
        f">= {thresholds['min_snr']:.1f}",
        _is_float_convertible(snr) and float(snr) >= thresholds["min_snr"],
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
        _is_float_convertible(drift) and abs(float(drift)) <= thresholds["max_abs_drift_nm_per_frame"],
        False,
        "Stabilize temperature/humidity and allow longer warm-up before measurement.",
    )
    add_check(
        "linearity_limit",
        "Limit of linearity (LOL) computed",
        lol_ppm,
        "finite",
        _is_finite_number(lol_ppm),
        False,
        "Collect more calibration points and verify Mandel linearity to establish linear operating range.",
    )
    add_check(
        "kinetics_fit",
        "Kinetics fit quality (R²)",
        kinetics_fit_r2,
        ">= 0.90 (when kinetics available)",
        (not _is_finite_number(kinetics_fit_r2)) or (_is_float_convertible(kinetics_fit_r2) and float(kinetics_fit_r2) >= 0.90),
        False,
        "Capture a cleaner step response and re-run kinetics fitting to support mechanism claims.",
    )
    add_check(
        "kinetics_tau63",
        "Response time constant τ63 (s)",
        tau_63_s,
        "finite (when kinetics available)",
        (not _is_finite_number(tau_63_s)) or (_is_float_convertible(tau_63_s) and float(tau_63_s) > 0),
        False,
        "Run a full association transient to characterize sensor response dynamics.",
    )
    add_check(
        "interval_coverage",
        "Predictive interval coverage",
        interval_coverage,
        ">= 0.90 (if ground truth available)",
        (not _is_finite_number(interval_coverage)) or (_is_float_convertible(interval_coverage) and float(interval_coverage) >= 0.90),
        False,
        "Evaluate predicted intervals against labeled concentrations to validate uncertainty calibration.",
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

    shipment_label = "QUALIFIED FOR EXTERNAL REVIEW" if overall_pass else "RESEARCH ONLY - NOT QUALIFIED"
    shipment_notice = (
        "Qualification gates passed. Artifact is suitable for external pilot review and supplier discussion."
        if overall_pass
        else "Critical qualification gates remain open. Do not use this artifact as evidence of supplier readiness."
    )
    reproducibility = _build_reproducibility_overview(app, resolved_session_id)

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
        "shipment_label": shipment_label,
        "shipment_notice": shipment_notice,
        "reproducibility": reproducibility,
    }


def _dossier_artifact_dir() -> Path:
    """Directory where qualification dossier exports are written."""
    return Path(os.environ.get("SPECTRAAGENT_DOSSIER_DIR", "output/qualification"))


def _resolve_downloadable_path(path_str: str) -> Path:
    """Resolve a requested artifact path and ensure it stays inside allowed roots."""
    requested = Path(path_str).expanduser().resolve()
    allowed_roots = [
        _dossier_artifact_dir().resolve(),
        Path("output/sessions").resolve(),
    ]
    if not any(root == requested or root in requested.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Requested path is outside allowed artifact roots")
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="Artifact file not found")
    return requested


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
    shipment_label = html.escape(str(dossier.get("shipment_label") or "RESEARCH ONLY"))
    shipment_notice = html.escape(str(dossier.get("shipment_notice") or ""))
    banner_class = "pass" if dossier.get("overall_pass") else "fail"
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
        .banner {{ margin: 16px 0; padding: 12px 14px; border-radius: 8px; font-weight: 700; }}
        .banner.pass {{ background: #edf7ed; color: #14532d; border: 1px solid #86efac; }}
        .banner.fail {{ background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #b9c6d1; padding: 8px; text-align: left; }}
        th {{ background: #eaf1f6; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; background: #edf7ed; }}
    </style>
</head>
<body>
    <h1>Qualification Dossier</h1>
    <div class=\"meta\">Session: <strong>{session}</strong> | Overall: <strong>{overall}</strong> | Tier: <strong>{tier}</strong> | Score: <strong>{score}</strong></div>
    <div class="banner {banner_class}">{shipment_label}: {shipment_notice}</div>
    <table>
        <thead><tr><th>Check</th><th>Value</th><th>Target</th><th>Status</th></tr></thead>
        <tbody>{row_html}</tbody>
    </table>
</body>
</html>
"""


def _write_dossier_artifacts(
    dossier: dict[str, Any],
    session_id: str,
    artifact: str,
) -> tuple[Path, str, dict[str, str], dict[str, Any]]:
    """Write dossier artifacts and return (out_dir, stamp, paths, signature)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _dossier_artifact_dir() / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_json = json.dumps(dossier, indent=2, sort_keys=True, default=str)
    signature = _dossier_signature(payload_json)
    signature_payload = {
        "session_id": session_id,
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

    return out_dir, stamp, paths, signature


def _build_package_status_readme(
    dossier: dict[str, Any],
    session_id: str,
    signature: dict[str, Any],
) -> str:
    """Create a plain-text package overview that explains shipment status."""
    overall = "PASS" if dossier.get("overall_pass") else "FAIL"
    shipment_label = str(dossier.get("shipment_label") or "RESEARCH ONLY")
    shipment_notice = str(dossier.get("shipment_notice") or "")
    tier = str(dossier.get("qualification_tier") or "not_qualified")
    score = str(dossier.get("score") or "0")
    next_actions = dossier.get("next_actions") or []
    action_lines = "\n".join(f"- {action}" for action in next_actions[:5]) or "- None"
    signing_mode = (
        str(signature.get("algorithm")) if signature.get("signed") else "unsigned-sha256"
    )
    return (
        "SPECTRAAGENT QUALIFICATION PACKAGE\n"
        "=================================\n\n"
        f"Session ID: {session_id}\n"
        f"Overall Qualification: {overall}\n"
        f"Tier: {tier}\n"
        f"Score: {score}\n"
        f"Shipment Label: {shipment_label}\n"
        f"Shipment Notice: {shipment_notice}\n"
        f"Signature Mode: {signing_mode}\n\n"
        "How to interpret this package\n"
        "-----------------------------\n"
        "- qualification/: exported dossier, HTML review copy, and signature metadata\n"
        "- session/: captured machine-readable session evidence when available\n"
        "- A RESEARCH ONLY label means the session did not satisfy supplier-facing qualification gates\n\n"
        "Immediate actions\n"
        "-----------------\n"
        f"{action_lines}\n"
    )


def _write_session_manifest(app: FastAPI, session_id: str) -> Path | None:
    """Write a reproducibility manifest beside a stored session when possible."""
    sw = getattr(app.state, "session_writer", None)
    session_base = getattr(sw, "_dir", None)
    if session_base is None:
        return None
    session_dir = Path(session_base) / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        return None
    try:
        from dashboard.reproducibility import create_session_manifest

        return create_session_manifest(session_id=session_id, output_dir=session_dir)
    except Exception as exc:
        log.warning("Session manifest creation failed for %s: %s", session_id, exc)
        return None


def _write_session_scientific_summary(
    app: FastAPI,
    session_id: str,
    context: dict[str, Any],
) -> dict[str, str] | None:
    """Write deterministic scientist-facing summary artifacts beside a session."""
    sw = getattr(app.state, "session_writer", None)
    session_base = getattr(sw, "_dir", None)
    if session_base is None:
        return None
    session_dir = Path(session_base) / session_id
    if not session_dir.exists() or not session_dir.is_dir():
        return None
    try:
        from src.reporting.scientific_summary import save_deterministic_scientific_summary

        return save_deterministic_scientific_summary(
            session_dir=session_dir,
            session_id=session_id,
            context=context,
        )
    except Exception as exc:
        log.warning("Scientific summary creation failed for %s: %s", session_id, exc)
        return None


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


def _build_research_flow_payload(app: FastAPI) -> dict[str, Any]:
    """Build a step-by-step flow state for researchers and commercialization."""
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
                    session_id = getattr(app.state, "last_session_id", None)
                    sw.stop_session(frame_count=int(app.state.session_frame_count))
                    if session_id:
                        _write_session_manifest(app, str(session_id))
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

    # Acquisition mutable state — stored on app.state so route modules can access them
    _acq_config: dict[str, Any] = {
        "integration_time_ms": 50.0,
        "gas_label": "unknown",
        "target_concentration": None,
    }
    # Expose on app.state so __main__.py can build a get_analyte lambda
    # that always returns the current gas label at call time.
    app.state._acq_config = _acq_config
    _session_active: dict[str, Any] = {"running": False, "session_id": None}
    app.state._session_active = _session_active

    # ------------------------------------------------------------------
    # Register route modules
    # ------------------------------------------------------------------
    from spectraagent.webapp.routes import (
        acquisition_router,
        agents_router,
        reports_router,
        sessions_router,
    )

    app.include_router(acquisition_router)
    app.include_router(sessions_router)
    app.include_router(reports_router)
    app.include_router(agents_router)

    # ------------------------------------------------------------------
    # Health endpoint (stays in server.py — uses app closure directly)
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
        _, _, paths, signature = _write_dossier_artifacts(dossier, resolved_session_id, artifact)

        return JSONResponse(
            {
                "status": "exported",
                "session_id": resolved_session_id,
                "artifact": artifact,
                "paths": paths,
                "signature": signature,
            }
        )

    @app.post("/api/qualification/package")
    async def qualification_package(session_id: str | None = None) -> JSONResponse:
        """Create a zipped research package containing dossier + session evidence files."""
        dossier = _build_qualification_dossier(app, session_id, _session_active)
        resolved_session_id = str(
            dossier.get("session_id") or getattr(app.state, "last_session_id", None) or "unknown"
        )
        out_dir, stamp, paths, signature = _write_dossier_artifacts(
            dossier, resolved_session_id, "both"
        )

        bundle_path = out_dir / f"research_package_{stamp}.zip"
        included: list[str] = []
        with zipfile.ZipFile(bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            status_readme = _build_package_status_readme(dossier, resolved_session_id, signature)
            zf.writestr("README_STATUS.txt", status_readme)
            included.append("README_STATUS.txt")

            for k in ("json", "html", "signature"):
                p = paths.get(k)
                if not p:
                    continue
                src = Path(p)
                arc = f"qualification/{src.name}"
                zf.write(src, arcname=arc)
                included.append(arc)

            sw = getattr(app.state, "session_writer", None)
            session_base = getattr(sw, "_dir", Path("output/sessions"))
            session_root = Path(session_base) / resolved_session_id
            for name in ("session_meta.json", "agent_events.jsonl", "pipeline_results.csv"):
                src = session_root / name
                if src.exists() and src.is_file():
                    arc = f"session/{name}"
                    zf.write(src, arcname=arc)
                    included.append(arc)

            for summary_path in sorted(session_root.glob("*_scientific_summary.*")):
                if summary_path.is_file():
                    arc = f"session/{summary_path.name}"
                    zf.write(summary_path, arcname=arc)
                    included.append(arc)

            for manifest_path in sorted(session_root.glob("*_manifest.json")):
                if manifest_path.is_file():
                    arc = f"session/{manifest_path.name}"
                    zf.write(manifest_path, arcname=arc)
                    included.append(arc)

        return JSONResponse(
            {
                "status": "packaged",
                "session_id": resolved_session_id,
                "package_path": str(bundle_path),
                "included": included,
                "signature": signature,
            }
        )

    @app.get("/api/artifacts/download")
    async def artifact_download(path: str) -> FileResponse:
        """Serve generated qualification/session artifacts from approved output roots."""
        resolved = _resolve_downloadable_path(path)
        return FileResponse(resolved, filename=resolved.name)

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
    # Static files (React SPA) — mounted at /app so FastAPI's /docs,
    # /openapi.json, and /redoc routes remain reachable.
    # ------------------------------------------------------------------
    _has_real_content = (
        _STATIC_DIST.exists()
        and any(f for f in _STATIC_DIST.iterdir() if f.name != ".gitkeep")
    )
    if _has_real_content:
        # Root redirect: / → /app/ so browser bookmarks still work
        @app.get("/")
        async def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/app/", status_code=302)

        app.mount("/app", StaticFiles(directory=str(_STATIC_DIST), html=True), name="static")
    else:
        @app.get("/")
        async def index_placeholder() -> JSONResponse:
            return JSONResponse({
                "message": "React frontend not yet built. Run: cd spectraagent/webapp/frontend && npm run build"
            })

    return app
