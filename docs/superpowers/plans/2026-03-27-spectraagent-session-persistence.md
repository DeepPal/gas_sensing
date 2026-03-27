# SpectraAgent Plan 4 — Session Persistence & Reports

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist agent events to `output/sessions/{id}/` during active acquisitions and expose session listing/detail, report generation, and agent settings via 4 new HTTP routes.

**Architecture:** A `SessionWriter` class handles all disk I/O (session_meta.json, agent_events.jsonl) with no background threads — it is called synchronously from the existing `_log_events` asyncio task and from acquisition route handlers. The 4 new routes read from `SessionWriter` or delegate to existing `app.state` agents.

**Tech Stack:** Python stdlib (`json`, `pathlib`, `datetime`), FastAPI, Pydantic, `AgentEvent.to_dict()` (already defined in `agent_bus.py`)

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `spectraagent/webapp/session_writer.py` | Create | `SessionWriter` — all session disk I/O |
| `tests/spectraagent/webapp/test_session_writer.py` | Create | 10 unit tests for SessionWriter |
| `spectraagent/webapp/server.py` | Modify | Init SessionWriter on app.state; wire to acq routes + _log_events; add 4 routes |
| `tests/spectraagent/webapp/test_server.py` | Modify | 6 integration tests for new routes |

---

## Current server.py state (context for implementer)

Relevant module-level and `create_app()` state — read before editing:

- Line 88: `_agent_bus = AgentBus()` — module-level singleton
- Lines 250–256: `_acq_config`, `_session_active`, `_latest_spectrum` — closure dicts inside `create_app()`
- Lines 265–270: `acq_start()` — sets `_session_active["running"] = True`, returns `{"status": "started", "session_id": session_id}`
- Lines 272–276: `acq_stop()` — sets `_session_active["running"] = False`
- Lines 158–169: `_log_events()` nested coroutine inside `_startup` — appends `event.to_dict()` to `agent_events_log` deque; currently does NOT persist to disk
- Line 19: `from fastapi import FastAPI, WebSocket, WebSocketDisconnect` — HTTPException not yet imported

---

## Task 1: `SessionWriter` class

**Files:**
- Create: `spectraagent/webapp/session_writer.py`
- Create: `tests/spectraagent/webapp/test_session_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/spectraagent/webapp/test_session_writer.py`:

```python
"""
Tests for spectraagent.webapp.session_writer.SessionWriter

All tests use the tmp_path pytest fixture to avoid writing to the real filesystem.
"""
import json
from pathlib import Path

import pytest

from spectraagent.webapp.session_writer import SessionWriter


@pytest.fixture
def writer(tmp_path):
    return SessionWriter(sessions_dir=tmp_path / "sessions")


# ---------------------------------------------------------------------------
# start_session
# ---------------------------------------------------------------------------


def test_start_session_creates_directory(writer, tmp_path):
    """start_session creates the session directory and session_meta.json."""
    writer.start_session("20260327_120000", {"gas_label": "Ethanol", "target_concentration": 0.5})
    session_dir = tmp_path / "sessions" / "20260327_120000"
    assert session_dir.is_dir()
    assert (session_dir / "session_meta.json").exists()


def test_start_session_meta_has_required_fields(writer, tmp_path):
    """session_meta.json contains session_id, started_at=set, stopped_at=None, and passed meta."""
    writer.start_session("20260327_120001", {"gas_label": "CO2", "target_concentration": 1.0})
    meta_path = tmp_path / "sessions" / "20260327_120001" / "session_meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["session_id"] == "20260327_120001"
    assert meta["started_at"] is not None
    assert meta["stopped_at"] is None
    assert meta["gas_label"] == "CO2"
    assert meta["target_concentration"] == 1.0


# ---------------------------------------------------------------------------
# append_event
# ---------------------------------------------------------------------------


def test_append_event_writes_jsonl(writer, tmp_path):
    """append_event writes a JSON line to agent_events.jsonl."""
    writer.start_session("20260327_120002", {})
    writer.append_event({"source": "QualityAgent", "level": "ok", "type": "quality", "text": "ok"})
    events_path = tmp_path / "sessions" / "20260327_120002" / "agent_events.jsonl"
    lines = events_path.read_text().splitlines()
    assert len(lines) == 1
    evt = json.loads(lines[0])
    assert evt["source"] == "QualityAgent"


def test_append_event_noop_when_no_session(writer):
    """append_event is a no-op when no session is active — must not raise."""
    writer.append_event({"source": "X", "level": "ok", "type": "t", "text": "t"})  # no exception


# ---------------------------------------------------------------------------
# stop_session
# ---------------------------------------------------------------------------


def test_stop_session_updates_meta(writer, tmp_path):
    """stop_session writes stopped_at and frame_count to session_meta.json."""
    writer.start_session("20260327_120003", {})
    writer.stop_session(frame_count=42)
    meta = json.loads(
        (tmp_path / "sessions" / "20260327_120003" / "session_meta.json").read_text()
    )
    assert meta["stopped_at"] is not None
    assert meta["frame_count"] == 42


def test_stop_session_noop_when_no_session(writer):
    """stop_session is a no-op when no session is active — must not raise."""
    writer.stop_session()  # no exception


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


def test_list_sessions_empty_dir(writer):
    """list_sessions returns [] when the sessions directory does not exist."""
    assert writer.list_sessions() == []


def test_list_sessions_returns_metadata(writer):
    """list_sessions returns one entry after a completed session."""
    writer.start_session("20260327_120004", {"gas_label": "Ethanol"})
    writer.stop_session()
    sessions = writer.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "20260327_120004"


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


def test_get_session_returns_meta_and_events(writer):
    """get_session returns meta dict with an 'events' list."""
    writer.start_session("20260327_120005", {"gas_label": "Ethanol"})
    writer.append_event({"source": "QA", "level": "ok", "type": "quality", "text": "ok"})
    writer.stop_session()
    result = writer.get_session("20260327_120005")
    assert result is not None
    assert result["session_id"] == "20260327_120005"
    assert isinstance(result["events"], list)
    assert len(result["events"]) == 1
    assert result["events"][0]["source"] == "QA"


def test_get_session_nonexistent_returns_none(writer):
    """get_session returns None for an unknown session_id."""
    assert writer.get_session("nonexistent_session") is None
```

- [ ] **Step 2: Run to confirm tests fail**

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_session_writer.py -v --tb=short --import-mode=importlib 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'SessionWriter'`

- [ ] **Step 3: Create `spectraagent/webapp/session_writer.py`**

```python
"""
spectraagent.webapp.session_writer
=====================================
SessionWriter — persists session metadata and agent events to disk.

Each active session creates a directory under ``sessions_dir / session_id /``:
  session_meta.json    — metadata (session_id, gas_label, timestamps, frame_count)
  agent_events.jsonl   — one JSON line per AgentEvent (append-only)

``append_event()`` is a no-op when no session is active.
``stop_session()`` is a no-op when no session is active.
Both are safe to call unconditionally from the asyncio event loop.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional

log = logging.getLogger(__name__)


class SessionWriter:
    """Persist session data to ``sessions_dir/{session_id}/``.

    Parameters
    ----------
    sessions_dir:
        Base directory for all session subdirectories.
        Created on demand when ``start_session()`` is called.
        Defaults to ``output/sessions`` relative to the working directory.
    """

    def __init__(self, sessions_dir: Path = Path("output/sessions")) -> None:
        self._dir = sessions_dir
        self._active_dir: Optional[Path] = None
        self._events_file: Optional[IO[str]] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, meta: dict) -> Path:
        """Create session directory and write initial session_meta.json.

        Opens ``agent_events.jsonl`` for appending.
        If a previous session was not stopped cleanly, it is closed first.

        Parameters
        ----------
        session_id:
            Unique session identifier (e.g. ``"20260327_120000"``).
        meta:
            Additional metadata merged into session_meta.json (e.g.
            ``gas_label``, ``target_concentration``, ``hardware``).

        Returns
        -------
        Path
            The created session directory path.
        """
        if self._events_file is not None:
            # Previous session not stopped — close it cleanly.
            self.stop_session()

        session_dir = self._dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        full_meta: dict = {
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "stopped_at": None,
        }
        full_meta.update(meta)
        (session_dir / "session_meta.json").write_text(
            json.dumps(full_meta, indent=2), encoding="utf-8"
        )

        self._events_file = open(  # noqa: WPS515
            session_dir / "agent_events.jsonl", "a", encoding="utf-8"
        )
        self._active_dir = session_dir
        log.info("SessionWriter: started session %s", session_id)
        return session_dir

    def append_event(self, event_dict: dict) -> None:
        """Append an AgentEvent dict as a JSON line to ``agent_events.jsonl``.

        No-op when no session is active.
        Never raises — errors are logged at WARNING level.
        """
        if self._events_file is None:
            return
        try:
            self._events_file.write(json.dumps(event_dict) + "\n")
            self._events_file.flush()
        except Exception as exc:
            log.warning("SessionWriter.append_event failed: %s", exc)

    def stop_session(self, frame_count: int = 0) -> None:
        """Update session_meta.json with stopped_at + frame_count; close events file.

        No-op when no session is active.
        Never raises.

        Parameters
        ----------
        frame_count:
            Total frames acquired during the session (default 0).
        """
        if self._active_dir is None:
            return

        meta_path = self._active_dir / "session_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["stopped_at"] = datetime.now(timezone.utc).isoformat()
                meta["frame_count"] = frame_count
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception as exc:
                log.warning("SessionWriter.stop_session: failed to update meta: %s", exc)

        if self._events_file is not None:
            try:
                self._events_file.close()
            except Exception:
                pass
            self._events_file = None

        log.info("SessionWriter: stopped session at %s", self._active_dir)
        self._active_dir = None

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def list_sessions(self) -> list:
        """Return all session metadata dicts, sorted newest first.

        Returns an empty list if the sessions directory does not exist.
        Skips entries whose ``session_meta.json`` cannot be parsed.
        """
        if not self._dir.exists():
            return []
        sessions = []
        for meta_path in sorted(self._dir.glob("*/session_meta.json"), reverse=True):
            try:
                sessions.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("SessionWriter.list_sessions: skipping %s: %s", meta_path, exc)
        return sessions

    def get_session(self, session_id: str) -> Optional[dict]:
        """Return session metadata merged with the last 100 agent events.

        Parameters
        ----------
        session_id:
            The session identifier to look up.

        Returns
        -------
        dict or None
            ``{...meta fields..., "events": [...last 100 events...]}``
            or ``None`` if the session does not exist on disk.
        """
        meta_path = self._dir / session_id / "session_meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("SessionWriter.get_session: failed to read meta: %s", exc)
            return None

        events: list[dict] = []
        events_path = self._dir / session_id / "agent_events.jsonl"
        if events_path.exists():
            try:
                lines = events_path.read_text(encoding="utf-8").splitlines()
                for line in lines[-100:]:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            except OSError as exc:
                log.warning("SessionWriter.get_session: failed to read events: %s", exc)

        return {**meta, "events": events}
```

- [ ] **Step 4: Run to confirm 10 tests pass**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_session_writer.py -v --tb=short --import-mode=importlib
```

Expected: `10 passed`

- [ ] **Step 5: Run full suite — confirm no regressions**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/ -q --tb=short --import-mode=importlib
```

Expected: `114+ passed` (104 existing + 10 new)

- [ ] **Step 6: Commit**

```bash
git add spectraagent/webapp/session_writer.py tests/spectraagent/webapp/test_session_writer.py
git commit -m "feat: SessionWriter — persist session metadata and agent events to disk"
```

---

## Task 2: Wire SessionWriter to server.py + session routes

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to the END of `tests/spectraagent/webapp/test_server.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: Session routes
# ---------------------------------------------------------------------------


def test_sessions_list_returns_200(client):
    """GET /api/sessions returns 200 with a list (empty in test mode)."""
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_sessions_get_not_found_returns_404(client):
    """GET /api/sessions/{session_id} returns 404 for an unknown session."""
    resp = client.get("/api/sessions/nonexistent_session_id_xyz")
    assert resp.status_code == 404


def test_acquisition_start_stop_creates_session(client, tmp_path):
    """POST /api/acquisition/start then stop populates session list."""
    import spectraagent.webapp.server as _srv
    from spectraagent.webapp.session_writer import SessionWriter

    # Inject a temp-dir writer into the live app via the client's app
    sw = SessionWriter(sessions_dir=tmp_path / "sessions")
    client.app.state.session_writer = sw

    # Start then stop a session
    start_resp = client.post("/api/acquisition/start")
    assert start_resp.status_code == 200
    session_id = start_resp.json()["session_id"]

    stop_resp = client.post("/api/acquisition/stop")
    assert stop_resp.status_code == 200

    # The writer should have a stopped session on disk
    sessions = sw.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["stopped_at"] is not None
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short --import-mode=importlib -k "sessions"
```

Expected: `FAILED` — route not yet defined (404) or `AttributeError`.

- [ ] **Step 3: Apply changes to `server.py`**

**Change 1:** Add `HTTPException` to the fastapi import and add the SessionWriter import.

Find line 19:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
```
Replace with:
```python
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
```

After `from spectraagent.webapp.agent_bus import AgentBus, AgentEvent` (line 28), add:
```python
from spectraagent.webapp.session_writer import SessionWriter
```

**Change 2:** Initialize `SessionWriter` on `app.state` in `create_app()`.

Find the line (currently line 150):
```python
    app.state.agent_events_log = deque(maxlen=200)
```
Add immediately after it:
```python
    app.state.session_writer = SessionWriter()
```

**Change 3:** Modify `_log_events()` to also call `session_writer.append_event()`.

Find the current `_log_events` body (inside `_startup`):
```python
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
```
Replace with:
```python
        async def _log_events() -> None:
            q = _agent_bus.subscribe()
            try:
                while True:
                    event = await q.get()
                    event_dict = event.to_dict()
                    app.state.agent_events_log.append(event_dict)
                    # Also persist to active session (no-op when no session)
                    sw = getattr(app.state, "session_writer", None)
                    if sw is not None:
                        sw.append_event(event_dict)
            except asyncio.CancelledError:
                pass
            finally:
                _agent_bus.unsubscribe(q)
```

**Change 4:** Call `session_writer.start_session()` in `acq_start()`.

Find:
```python
    @app.post("/api/acquisition/start")
    async def acq_start() -> JSONResponse:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_active["running"] = True
        _session_active["session_id"] = session_id
        return JSONResponse({"status": "started", "session_id": session_id})
```
Replace with:
```python
    @app.post("/api/acquisition/start")
    async def acq_start() -> JSONResponse:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        _session_active["running"] = True
        _session_active["session_id"] = session_id
        sw = getattr(app.state, "session_writer", None)
        if sw is not None:
            meta = {
                "gas_label": _acq_config.get("gas_label", "unknown"),
                "target_concentration": _acq_config.get("target_concentration"),
                "hardware": getattr(getattr(app.state, "driver", None), "name", "unknown"),
            }
            sw.start_session(session_id, meta)
        return JSONResponse({"status": "started", "session_id": session_id})
```

**Change 5:** Call `session_writer.stop_session()` in `acq_stop()`.

Find:
```python
    @app.post("/api/acquisition/stop")
    async def acq_stop() -> JSONResponse:
        _session_active["running"] = False
        return JSONResponse({"status": "stopped",
                             "session_id": _session_active.get("session_id")})
```
Replace with:
```python
    @app.post("/api/acquisition/stop")
    async def acq_stop() -> JSONResponse:
        _session_active["running"] = False
        sw = getattr(app.state, "session_writer", None)
        if sw is not None:
            sw.stop_session()
        return JSONResponse({"status": "stopped",
                             "session_id": _session_active.get("session_id")})
```

**Change 6:** Add session routes. Add the following section after the `# Calibration API` section and before the `# Claude API` section:

```python
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
```

- [ ] **Step 4: Run the session tests**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short --import-mode=importlib -k "sessions"
```

Expected: `3 passed`

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/ -q --tb=short --import-mode=importlib
```

Expected: `117+ passed`, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add spectraagent/webapp/server.py tests/spectraagent/webapp/test_server.py
git commit -m "feat: wire SessionWriter to acq routes; add GET /api/sessions and /api/sessions/{id}"
```

---

## Task 3: Reports generate + agent settings routes

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

- [ ] **Step 1: Write the failing tests**

Append these tests to the END of `tests/spectraagent/webapp/test_server.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: /api/reports/generate and /api/agents/settings
# ---------------------------------------------------------------------------


def test_reports_generate_requires_session_id(client):
    """POST /api/reports/generate without session_id returns 422."""
    resp = client.post("/api/reports/generate", json={})
    assert resp.status_code == 422


def test_reports_generate_no_writer_returns_503(client):
    """POST /api/reports/generate returns 503 when ReportWriter is not on app.state."""
    # In test mode, report_writer is not set (only set in CLI start())
    client.app.state.report_writer = None  # explicitly clear to be safe
    resp = client.post("/api/reports/generate", json={"session_id": "20260327_120000"})
    assert resp.status_code == 503
    assert "ReportWriter" in resp.json()["detail"]


def test_reports_generate_claude_unavailable_returns_503(client):
    """POST /api/reports/generate returns 503 when Claude is unavailable."""
    from unittest.mock import AsyncMock, MagicMock

    # Create a mock ReportWriter that returns None (Claude unavailable)
    mock_writer = MagicMock()
    mock_writer.write = AsyncMock(return_value=None)
    client.app.state.report_writer = mock_writer

    resp = client.post("/api/reports/generate", json={"session_id": "20260327_120000"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]

    # Clean up
    client.app.state.report_writer = None


def test_agents_settings_returns_200(client):
    """PUT /api/agents/settings returns 200 with echoed auto_explain value."""
    resp = client.put("/api/agents/settings", json={"auto_explain": True})
    assert resp.status_code == 200
    assert resp.json()["auto_explain"] is True


def test_agents_settings_requires_auto_explain(client):
    """PUT /api/agents/settings without auto_explain returns 422."""
    resp = client.put("/api/agents/settings", json={})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run to confirm tests fail**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short --import-mode=importlib -k "reports or settings"
```

Expected: `FAILED` — routes not yet defined.

- [ ] **Step 3: Apply changes to `server.py`**

**Change 1:** Add two new Pydantic models. After the `AskRequest` model (around line 84), add:

```python

class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class AgentSettings(BaseModel):
    auto_explain: bool
```

**Change 2:** Add the routes. In `create_app()`, after the session routes section and before the `# Claude API` section, add:

```python
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

        Graceful: if agents are not yet created (e.g. no API key configured),
        the request still returns 200 — there is nothing to toggle.
        """
        anomaly = getattr(app.state, "anomaly_explainer", None)
        narrator = getattr(app.state, "experiment_narrator", None)
        if anomaly is not None:
            anomaly.set_auto_explain(settings.auto_explain)
        if narrator is not None:
            narrator.set_auto_explain(settings.auto_explain)
        return JSONResponse({"auto_explain": settings.auto_explain})
```

- [ ] **Step 4: Run the new tests**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short --import-mode=importlib -k "reports or settings"
```

Expected: `5 passed`

- [ ] **Step 5: Run full suite**

```
.venv/Scripts/python.exe -m pytest tests/spectraagent/ -q --tb=short --import-mode=importlib
```

Expected: `122+ passed` (114 after Task 1 + 3 session tests + 5 reports/settings tests), 0 failures.

- [ ] **Step 6: Commit**

```bash
git add spectraagent/webapp/server.py tests/spectraagent/webapp/test_server.py
git commit -m "feat: add POST /api/reports/generate and PUT /api/agents/settings routes"
```

---

## Import Consistency Reference

| Symbol | Canonical import path |
|---|---|
| `SessionWriter` | `from spectraagent.webapp.session_writer import SessionWriter` |
| `HTTPException` | `from fastapi import ..., HTTPException, ...` |
| `ReportRequest` | Pydantic model defined in `spectraagent/webapp/server.py` |
| `AgentSettings` | Pydantic model defined in `spectraagent/webapp/server.py` |
| `report_writer` | `app.state.report_writer` — set in `__main__.py start()`, may be `None` in tests |
| `session_writer` | `app.state.session_writer` — set in `create_app()`, always available |

---

## Self-Review Checklist

- [ ] `SessionWriter.__init__` defaults to `Path("output/sessions")` — relative to CWD at runtime
- [ ] `append_event` and `stop_session` are no-ops when `_active_dir is None` — safe to call unconditionally
- [ ] `_log_events` calls `sw.append_event()` — no-op until `start_session()` is called
- [ ] `acq_start()` calls `sw.start_session()` with gas_label + target_concentration from `_acq_config`
- [ ] `acq_stop()` calls `sw.stop_session()` — frame_count defaults to 0 (acquisition loop does not yet count frames via server.py)
- [ ] `GET /api/sessions` returns `[]` when session_writer is None or has no sessions — never 500
- [ ] `GET /api/sessions/{session_id}` returns 404 (not 500) for unknown sessions
- [ ] `POST /api/reports/generate` checks `report_writer is None` → 503 before awaiting
- [ ] `POST /api/reports/generate` checks `writer.write() is None` → 503 after awaiting
- [ ] `PUT /api/agents/settings` uses `getattr(..., None)` guards — graceful when agents absent
- [ ] All new Pydantic models (`ReportRequest`, `AgentSettings`) use `Field` where validation needed
- [ ] No new `__init__.py` in test directories (using `--import-mode=importlib`)
- [ ] Full test suite (122+) passes after all 3 tasks
