"""Session management endpoints: /api/sessions/*"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["sessions"])


@router.get("/api/sessions")
def sessions_list(request: Request) -> JSONResponse:
    """Return all session metadata dicts, newest first."""
    sw = getattr(request.app.state, "session_writer", None)
    if sw is None:
        return JSONResponse([])
    return JSONResponse(sw.list_sessions())


@router.get("/api/sessions/{session_id}")
def sessions_get(session_id: str, request: Request) -> JSONResponse:
    """Return metadata + last 100 agent events for a session, or 404."""
    sw = getattr(request.app.state, "session_writer", None)
    if sw is None:
        raise HTTPException(status_code=404, detail="Session not found")
    data = sw.get_session(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return JSONResponse(data)
