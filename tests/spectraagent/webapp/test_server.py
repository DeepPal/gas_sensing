import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import pytest

from spectraagent.webapp.server import Broadcaster, _RateLimiter, create_app
from spectraagent.webapp.session_writer import SessionWriter


@pytest.fixture(autouse=True)
def _clear_rate_limiters():
    """Reset module-level rate limiter state before and after every test.

    The limiters are module-level singletons — without this fixture their
    sliding-window history accumulates across tests sharing the same process,
    causing the 4th call to /api/reports/generate to return 429 even in tests
    that don't intend to trigger the limit.
    """
    import spectraagent.webapp.server as srv

    srv._claude_limiter._history.clear()
    srv._report_limiter._history.clear()
    yield
    srv._claude_limiter._history.clear()
    srv._report_limiter._history.clear()


@pytest.fixture
def client():
    app = create_app(simulate=True)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Task 9: Health endpoint
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_health_response_schema(client):
    resp = client.get("/api/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "hardware" in data
    assert "version" in data
    assert "claude_api_key_configured" in data
    assert "knowledge_base_available" in data
    assert "knowledge_context_mode" in data
    assert "rate_limits" in data
    assert "claude" in data["rate_limits"]
    assert "report" in data["rate_limits"]


def test_cors_header_present(client):
    resp = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in resp.headers


def test_unknown_route_returns_404(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404


def test_research_flow_endpoint_schema(client):
    """GET /api/research-flow returns guided workflow and readiness metadata."""
    resp = client.get("/api/research-flow")
    assert resp.status_code == 200
    data = resp.json()
    assert "readiness_score" in data
    assert isinstance(data["readiness_score"], int)
    assert "checkpoints" in data
    assert isinstance(data["checkpoints"], list)
    assert "next_steps" in data
    assert isinstance(data["next_steps"], list)
    assert "commercialization_signal" in data


def test_research_flow_recommends_reference_capture_when_missing(client):
    """Without a captured reference, guided steps should explicitly call it out."""
    resp = client.get("/api/research-flow")
    assert resp.status_code == 200
    data = resp.json()
    joined = " ".join(data.get("next_steps", [])).lower()
    assert "reference" in joined


def test_qualification_dossier_insufficient_data(client):
    """Qualification dossier reports insufficient data before a session analysis exists."""
    resp = client.get("/api/qualification/dossier")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "insufficient_data"
    assert data["overall_pass"] is False
    assert isinstance(data["next_actions"], list)


def test_qualification_dossier_passes_with_strong_metrics(client):
    """With strong session metrics, dossier should pass and produce a qualification tier."""
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=8,
        calibration_r2=0.985,
        mean_snr=4.2,
        lod_ppm=0.012,
        loq_ppm=0.040,
        drift_rate_nm_per_frame=0.001,
        summary_text="Qualification-ready session.",
    )
    resp = client.get("/api/qualification/dossier")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["overall_pass"] is True
    assert data["qualification_tier"] in {"bronze", "silver", "gold"}
    assert data["shipment_label"] == "QUALIFIED FOR EXTERNAL REVIEW"
    assert isinstance(data["checks"], list)


def test_qualification_dossier_marks_failed_sessions_as_research_only(client):
    """Failed qualification dossiers should be explicitly labeled as research-only."""
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=3,
        calibration_r2=0.81,
        mean_snr=2.1,
        lod_ppm=0.020,
        loq_ppm=0.070,
        drift_rate_nm_per_frame=0.010,
        summary_text="Not yet qualification-ready session.",
    )

    resp = client.get("/api/qualification/dossier")
    assert resp.status_code == 200
    data = resp.json()
    assert data["overall_pass"] is False
    assert data["shipment_label"] == "RESEARCH ONLY - NOT QUALIFIED"
    assert "supplier readiness" in data["shipment_notice"].lower()


def test_qualification_dossier_export_writes_artifacts(client, tmp_path, monkeypatch):
    """Export endpoint writes dossier artifacts and signature metadata."""
    monkeypatch.setenv("SPECTRAAGENT_DOSSIER_DIR", str(tmp_path))
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=8,
        calibration_r2=0.985,
        mean_snr=4.2,
        lod_ppm=0.012,
        loq_ppm=0.040,
        drift_rate_nm_per_frame=0.001,
        summary_text="Qualification-ready session.",
    )

    resp = client.post("/api/qualification/dossier/export?artifact=both")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "exported"
    assert "paths" in data
    assert os.path.exists(data["paths"]["json"])
    assert os.path.exists(data["paths"]["html"])
    assert os.path.exists(data["paths"]["signature"])
    html_payload = Path(data["paths"]["html"]).read_text(encoding="utf-8")
    assert "QUALIFIED FOR EXTERNAL REVIEW" in html_payload


def test_qualification_dossier_export_signs_with_hmac_key(client, tmp_path, monkeypatch):
    """When signing key is configured, export includes HMAC signature metadata."""
    monkeypatch.setenv("SPECTRAAGENT_DOSSIER_DIR", str(tmp_path))
    monkeypatch.setenv("SPECTRAAGENT_DOSSIER_SIGNING_KEY", "top-secret-test-key")
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=7,
        calibration_r2=0.97,
        mean_snr=3.5,
        lod_ppm=0.020,
        loq_ppm=0.060,
        drift_rate_nm_per_frame=0.002,
        summary_text="Signed dossier session.",
    )

    resp = client.post("/api/qualification/dossier/export?artifact=json")
    assert resp.status_code == 200
    sig = resp.json()["signature"]
    assert sig["signed"] is True
    assert sig["algorithm"] == "hmac-sha256"
    assert "signature" in sig


def test_qualification_package_creates_zip(client, tmp_path, monkeypatch):
    """Package endpoint should create a zip with dossier artifacts."""
    monkeypatch.setenv("SPECTRAAGENT_DOSSIER_DIR", str(tmp_path))
    client.app.state.session_writer = SessionWriter(tmp_path / "sessions")
    session_dir = (tmp_path / "sessions" / "unknown")
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session_meta.json").write_text('{"session_id": "unknown"}', encoding="utf-8")
    (session_dir / "agent_events.jsonl").write_text('{"type": "session_complete"}\n', encoding="utf-8")
    (session_dir / "pipeline_results.csv").write_text('frame,timestamp\n1,now\n', encoding="utf-8")
    (session_dir / "unknown_manifest.json").write_text('{"session_id": "unknown"}', encoding="utf-8")
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=7,
        calibration_r2=0.97,
        mean_snr=3.8,
        lod_ppm=0.015,
        loq_ppm=0.050,
        drift_rate_nm_per_frame=0.0015,
        summary_text="Packaging-ready session.",
    )

    resp = client.post("/api/qualification/package")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "packaged"
    package_path = Path(data["package_path"])
    assert package_path.exists()

    with zipfile.ZipFile(package_path, "r") as zf:
        names = zf.namelist()
        assert "README_STATUS.txt" in names
        assert any(name.startswith("qualification/") and name.endswith(".json") for name in names)
        assert any(name.startswith("qualification/") and name.endswith(".html") for name in names)
        assert any(name.startswith("qualification/") and name.endswith(".sig.json") for name in names)
        assert "session/unknown_manifest.json" in names
        readme = zf.read("README_STATUS.txt").decode("utf-8")
        assert "Shipment Label: QUALIFIED FOR EXTERNAL REVIEW" in readme


def test_artifact_download_serves_exported_file(client, tmp_path, monkeypatch):
    """Download route should serve a generated dossier artifact from allowed roots."""
    monkeypatch.setenv("SPECTRAAGENT_DOSSIER_DIR", str(tmp_path))
    client.app.state.last_session_analysis = SimpleNamespace(
        calibration_n_points=8,
        calibration_r2=0.985,
        mean_snr=4.2,
        lod_ppm=0.012,
        loq_ppm=0.040,
        drift_rate_nm_per_frame=0.001,
        summary_text="Downloadable session.",
    )

    export_resp = client.post("/api/qualification/dossier/export?artifact=json")
    path = export_resp.json()["paths"]["json"]
    resp = client.get("/api/artifacts/download", params={"path": path})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")


# ---------------------------------------------------------------------------
# Task 10: WebSocket / Broadcaster
# ---------------------------------------------------------------------------

def test_broadcaster_fan_out():
    """Messages sent to Broadcaster are received by all connected clients."""
    bc = Broadcaster()
    received: list[str] = []

    class _FakeWS:
        async def send_text(self, msg):
            received.append(msg)

    loop = asyncio.new_event_loop()

    async def run():
        ws = _FakeWS()
        bc.connect(ws)
        await bc.broadcast("hello")
        bc.disconnect(ws)

    loop.run_until_complete(run())
    loop.close()
    assert received == ["hello"]


def test_ws_spectrum_endpoint_connects(client):
    """WebSocket /ws/spectrum accepts connections without error."""
    with client.websocket_connect("/ws/spectrum"):
        pass  # connection accepted = success


# ---------------------------------------------------------------------------
# Task 11: Acquisition routes
# ---------------------------------------------------------------------------

def test_acquisition_config_post(client):
    resp = client.post("/api/acquisition/config", json={
        "integration_time_ms": 100.0,
        "gas_label": "Ethanol",
        "target_concentration": 0.1,
    })
    assert resp.status_code == 200
    assert resp.json()["integration_time_ms"] == 100.0


def test_acquisition_config_defaults_ok(client):
    resp = client.post("/api/acquisition/config", json={})
    assert resp.status_code == 200


def test_acquisition_start_returns_session_id(client):
    resp = client.post("/api/acquisition/start")
    assert resp.status_code == 200
    assert "session_id" in resp.json()


def test_acquisition_stop(client):
    client.post("/api/acquisition/start")
    resp = client.post("/api/acquisition/stop")
    assert resp.status_code == 200


def test_acquisition_reference_requires_running(client):
    resp = client.post("/api/acquisition/reference")
    assert resp.status_code in (200, 400)


# ---------------------------------------------------------------------------
# Task 2: /ws/agent-events WebSocket
# ---------------------------------------------------------------------------

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent


def test_ws_agent_events_connects(client):
    """WebSocket /ws/agent-events accepts connections without error."""
    with client.websocket_connect("/ws/agent-events"):
        pass  # connecting and cleanly disconnecting = success


def test_ws_agent_events_receives_emitted_event(client):
    """Event emitted to AgentBus is delivered to connected WS client."""
    import json

    app = client.app
    agent_bus: AgentBus = app.state.agent_bus
    assert agent_bus is not None, "AgentBus must be initialised by create_app()"

    with client.websocket_connect("/ws/agent-events") as ws:
        # Emit an event directly from test (bus.setup_loop already called by startup)
        agent_bus.emit(AgentEvent(
            source="Test",
            level="info",
            type="test_event",
            data={"x": 1},
            text="hello from test",
        ))
        msg = ws.receive_text()
        parsed = json.loads(msg)
        assert parsed["type"] == "test_event"
        assert parsed["source"] == "Test"


def test_calibration_add_point_returns_200(client):
    resp = client.post("/api/calibration/add-point", json={
        "concentration": 0.5,
        "delta_lambda": -2.5,
    })
    assert resp.status_code == 200
    assert resp.json()["concentration"] == 0.5


def test_calibration_suggest_returns_200(client):
    resp = client.post("/api/calibration/suggest")
    assert resp.status_code == 200
    # No GPR fitted yet → suggestion is null
    data = resp.json()
    assert "suggestion" in data


# ---------------------------------------------------------------------------
# Task 3: /api/agents/ask SSE endpoint
# ---------------------------------------------------------------------------


def test_agents_ask_endpoint_exists(client):
    """POST /api/agents/ask exists and returns 200."""
    with patch("spectraagent.webapp.server._get_ask_client", return_value=None):
        resp = client.post("/api/agents/ask", json={"query": "What is happening?"})
    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}"
    )


def test_agents_ask_no_api_key_returns_200_with_sse(client):
    """Without API key, /api/agents/ask returns 200 text/event-stream with unavailable message."""
    import os

    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with patch("spectraagent.webapp.server._get_ask_client", return_value=None):
            resp = client.post("/api/agents/ask", json={"query": "What is happening?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        # Response body must contain an SSE frame with unavailable indication
        body = resp.text
        assert "data:" in body
        assert "unavailable" in body.lower() or "ANTHROPIC" in body
    finally:
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key


def test_agents_ask_requires_query_field(client):
    """Missing 'query' field returns 422 Unprocessable Entity."""
    resp = client.post("/api/agents/ask", json={})
    assert resp.status_code == 422


def test_agents_ask_empty_query_returns_200(client):
    """Empty string query is technically valid — endpoint handles it."""
    with patch("spectraagent.webapp.server._get_ask_client", return_value=None):
        resp = client.post("/api/agents/ask", json={"query": ""})
    assert resp.status_code == 200


def test_agents_ask_sse_format_done_true(client):
    """Response must include a final SSE frame with done=true."""
    import json as _json

    with patch("spectraagent.webapp.server._get_ask_client", return_value=None):
        resp = client.post("/api/agents/ask", json={"query": "test"})

    # Parse SSE frames
    frames = [
        line[len("data: "):]
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(frames) >= 1, "Expected at least one SSE data frame"
    last_frame = _json.loads(frames[-1])
    assert last_frame.get("done") is True, f"Last frame must have done=true, got: {last_frame}"


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


def test_lifespan_startup_callback_runs():
    """App-scoped startup callbacks registered before TestClient entry are executed."""
    app = create_app(simulate=True)
    app.state.startup_marker = False

    def _mark_started() -> None:
        app.state.startup_marker = True

    app.state.startup_callbacks.append(_mark_started)

    with TestClient(app):
        assert app.state.startup_marker is True


def test_shutdown_finalizes_active_session(tmp_path):
    """Lifespan shutdown finalizes an active session with the last frame count."""
    app = create_app(simulate=True)
    writer = SessionWriter(sessions_dir=tmp_path / "sessions")
    app.state.session_writer = writer

    with TestClient(app) as client:
        start_resp = client.post("/api/acquisition/start")
        assert start_resp.status_code == 200
        app.state.session_frame_count = 7

    sessions = writer.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["frame_count"] == 7
    assert sessions[0]["stopped_at"] is not None


# ---------------------------------------------------------------------------
# Task 3: /api/reports/generate and /api/agents/settings
# ---------------------------------------------------------------------------


def test_reports_generate_requires_session_id(client):
    """POST /api/reports/generate without session_id returns 422."""
    resp = client.post("/api/reports/generate", json={})
    assert resp.status_code == 422


def test_reports_generate_no_writer_returns_503(client):
    """POST /api/reports/generate returns 503 when ReportWriter is not on app.state."""
    client.app.state.report_writer = None  # explicitly clear
    resp = client.post("/api/reports/generate", json={"session_id": "20260327_120000"})
    assert resp.status_code == 503
    assert "ReportWriter" in resp.json()["detail"]


def test_reports_generate_claude_unavailable_returns_503(client):
    """POST /api/reports/generate returns 503 when writer.write() returns None."""
    from unittest.mock import AsyncMock, MagicMock

    mock_writer = MagicMock()
    mock_writer.write = AsyncMock(return_value=None)
    client.app.state.report_writer = mock_writer

    resp = client.post("/api/reports/generate", json={"session_id": "20260327_120000"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]

    # Clean up
    client.app.state.report_writer = None


def test_reports_generate_success_returns_200(client):
    """POST /api/reports/generate returns 200 with {report, session_id} when write() succeeds."""
    from unittest.mock import AsyncMock, MagicMock

    mock_writer = MagicMock()
    mock_writer.write = AsyncMock(return_value="Methods: LSPR sensor...")
    client.app.state.report_writer = mock_writer

    resp = client.post("/api/reports/generate", json={"session_id": "20260327_120000"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "20260327_120000"
    assert "report" in data
    assert isinstance(data["report"], str)

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


# ---------------------------------------------------------------------------
# _RateLimiter unit tests
# ---------------------------------------------------------------------------


def test_rate_limiter_allows_up_to_max():
    rl = _RateLimiter(max_calls=3, window_s=60)
    assert rl.is_allowed("c") is True
    assert rl.is_allowed("c") is True
    assert rl.is_allowed("c") is True


def test_rate_limiter_blocks_over_max():
    rl = _RateLimiter(max_calls=3, window_s=60)
    for _ in range(3):
        rl.is_allowed("c")
    assert rl.is_allowed("c") is False


def test_rate_limiter_independent_keys():
    rl = _RateLimiter(max_calls=1, window_s=60)
    assert rl.is_allowed("a") is True
    assert rl.is_allowed("b") is True  # different key → not blocked


def test_rate_limiter_window_eviction():
    """Calls outside the window no longer count against the limit."""
    import time as _time

    rl = _RateLimiter(max_calls=1, window_s=0.05)
    assert rl.is_allowed("x") is True
    assert rl.is_allowed("x") is False  # blocked within window
    _time.sleep(0.1)
    assert rl.is_allowed("x") is True   # window expired → allowed again


# ---------------------------------------------------------------------------
# Rate limit endpoint integration: 429 after limit exceeded
# ---------------------------------------------------------------------------


def test_ask_rate_limit_returns_429(client):
    """POST /api/agents/ask returns 429 after exceeding the per-minute limit."""
    import spectraagent.webapp.server as srv

    tight = _RateLimiter(max_calls=1, window_s=60)
    with patch.object(srv, "_claude_limiter", tight), \
            patch("spectraagent.webapp.server._get_ask_client", return_value=None):
        resp1 = client.post("/api/agents/ask", json={"query": "first"})
        resp2 = client.post("/api/agents/ask", json={"query": "second"})

    assert resp1.status_code == 200
    assert resp2.status_code == 429
    assert "Rate limit" in resp2.json()["detail"]


def test_report_rate_limit_returns_429(client):
    """POST /api/reports/generate returns 429 after exceeding the per-minute limit."""
    from unittest.mock import AsyncMock, MagicMock

    import spectraagent.webapp.server as srv

    tight = _RateLimiter(max_calls=1, window_s=60)
    mock_writer = MagicMock()
    mock_writer.write = AsyncMock(return_value="some report text")
    client.app.state.report_writer = mock_writer

    try:
        with patch.object(srv, "_report_limiter", tight):
            resp1 = client.post("/api/reports/generate", json={"session_id": "20260101_120000"})
            resp2 = client.post("/api/reports/generate", json={"session_id": "20260101_120000"})

        assert resp1.status_code == 200
        assert resp2.status_code == 429
        assert "Rate limit" in resp2.json()["detail"]
    finally:
        client.app.state.report_writer = None


# ---------------------------------------------------------------------------
# Health endpoint: integration_time_ms field
# ---------------------------------------------------------------------------


def test_health_includes_integration_time_ms(client):
    """GET /api/health schema includes the integration_time_ms field.

    The value is None when no hardware driver is connected (test-client state).
    A numeric value is only present when a driver is attached at startup.
    """
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    # Field must be present in the schema; None is valid when driver is absent
    assert "integration_time_ms" in data
