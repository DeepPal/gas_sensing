"""Agent settings and Claude ask endpoints: /api/agents/*"""
from __future__ import annotations

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from spectraagent.webapp.routes._models import AgentSettings, AskRequest, DriftSettings, QualitySettings

log = logging.getLogger(__name__)

_ASK_MODEL = "claude-sonnet-4-6"

router = APIRouter(tags=["agents"])


def _rate_limit_claude_dep(request: Request) -> None:
    """Thin wrapper that delegates to the module-level Claude rate limiter in server.py."""
    from spectraagent.webapp import server as _srv  # noqa: PLC0415

    _srv._rate_limit_claude(request)  # type: ignore[attr-defined]


@router.put("/api/agents/quality-settings")
def agents_quality_settings(settings: QualitySettings, request: Request) -> JSONResponse:
    """Update QualityAgent thresholds at runtime."""
    qa = getattr(request.app.state, "quality_agent", None)
    if qa is not None:
        qa.configure(
            saturation_threshold=settings.saturation_threshold,
            snr_warn_threshold=settings.snr_warn_threshold,
        )
        return JSONResponse({"status": "updated", **qa.settings})
    return JSONResponse({"status": "agent_not_ready"})


@router.get("/api/agents/quality-settings")
def agents_quality_settings_get(request: Request) -> JSONResponse:
    qa = getattr(request.app.state, "quality_agent", None)
    if qa is None:
        return JSONResponse({})
    return JSONResponse(qa.settings)


@router.put("/api/agents/drift-settings")
def agents_drift_settings(settings: DriftSettings, request: Request) -> JSONResponse:
    """Update DriftAgent thresholds at runtime."""
    da = getattr(request.app.state, "drift_agent", None)
    if da is not None:
        da.configure(
            drift_threshold_nm_per_min=settings.drift_threshold_nm_per_min,
            window_frames=settings.window_frames,
        )
        return JSONResponse({"status": "updated", **da.settings})
    return JSONResponse({"status": "agent_not_ready"})


@router.get("/api/agents/drift-settings")
def agents_drift_settings_get(request: Request) -> JSONResponse:
    da = getattr(request.app.state, "drift_agent", None)
    if da is None:
        return JSONResponse({})
    return JSONResponse(da.settings)


@router.put("/api/agents/settings")
def agents_settings(settings: AgentSettings, request: Request) -> JSONResponse:
    """Toggle auto_explain for AnomalyExplainer and ExperimentNarrator.

    Graceful: if agents are not yet created (no API key configured),
    the request still returns 200 — there is nothing to toggle.
    """
    anomaly = getattr(request.app.state, "anomaly_explainer", None)
    narrator = getattr(request.app.state, "experiment_narrator", None)
    if anomaly is not None:
        anomaly.set_auto_explain(settings.auto_explain)
    if narrator is not None:
        narrator.set_auto_explain(settings.auto_explain)
    return JSONResponse({"auto_explain": settings.auto_explain})


@router.post("/api/agents/ask")
async def agents_ask(
    request_body: AskRequest,
    request: Request,
    _rl: Annotated[None, Depends(_rate_limit_claude_dep)] = None,
) -> StreamingResponse:
    """Stream a Claude response to a free-text query about the current session.

    Returns Server-Sent Events (SSE) with content-type text/event-stream.
    Each chunk: ``data: {"text": "...", "done": false}\\n\\n``
    Final frame: ``data: {"done": true}\\n\\n``
    """
    query = request_body.query
    events_log = list(request.app.state.agent_events_log)
    context = {
        "query": query,
        "last_20_agent_events": events_log[-20:],
    }

    async def _generate():
        from spectraagent.webapp import server as _srv  # noqa: PLC0415

        client = _srv._get_ask_client()  # type: ignore[attr-defined]
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
