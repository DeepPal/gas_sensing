"""Report generation endpoints: /api/reports/*"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from spectraagent.webapp.routes._models import ReportRequest

log = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


def _rate_limit_report_dep(request: Request) -> None:
    """Thin wrapper that delegates to the module-level rate limiter in server.py.

    Defined here so the Depends() annotation stays clean.  The actual limiter
    instance lives in server.py so it is shared with any inline usages.
    """
    from spectraagent.webapp import server as _srv  # noqa: PLC0415

    _srv._rate_limit_report(request)  # type: ignore[attr-defined]


@router.post("/api/reports/generate")
async def reports_generate(
    request_body: ReportRequest,
    request: Request,
    _rl: Annotated[None, Depends(_rate_limit_report_dep)] = None,
) -> JSONResponse:
    """Generate a scientist-facing session report.

    Returns ``{"report": "<text>", "session_id": "<id>"}`` on success.
    Prefers ``ReportWriter`` when available, but falls back to a deterministic
    scientific summary so researchers still get a useful report offline.
    """
    from src.inference.session_analyzer import SessionAnalyzer
    from src.reporting.scientific_summary import (
        build_deterministic_scientific_report,
        session_analysis_to_dict,
    )

    app = request.app
    sw = getattr(app.state, "session_writer", None)
    context: dict = {"session_id": request_body.session_id}
    if sw is not None:
        session_data = sw.get_session(request_body.session_id)
        if session_data is not None:
            context.update(session_data)

    analysis = getattr(app.state, "last_session_analysis", None)
    analysis_session_id = getattr(app.state, "last_session_analysis_session_id", None)
    if analysis is not None and analysis_session_id == request_body.session_id:
        context["analysis"] = session_analysis_to_dict(analysis)
    elif context.get("events"):
        try:
            on_demand_analysis = SessionAnalyzer().analyze(
                context["events"],
                int(context.get("frame_count") or 0),
            )
            context["analysis"] = session_analysis_to_dict(on_demand_analysis)
        except Exception as exc:
            log.debug("On-demand report analysis failed: %s", exc)

    writer = getattr(app.state, "report_writer", None)
    if writer is None:
        return JSONResponse(
            {
                "report": build_deterministic_scientific_report(context),
                "session_id": request_body.session_id,
                "report_source": "deterministic",
            }
        )

    text = await writer.write(context)
    if text is None:
        return JSONResponse(
            {
                "report": build_deterministic_scientific_report(context),
                "session_id": request_body.session_id,
                "report_source": "deterministic",
                "report_notice": (
                    "Claude unavailable; returned deterministic scientific summary instead."
                ),
            }
        )
    return JSONResponse(
        {
            "report": text,
            "session_id": request_body.session_id,
            "report_source": "claude",
        }
    )
