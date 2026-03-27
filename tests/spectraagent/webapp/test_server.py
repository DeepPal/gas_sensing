import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from spectraagent.webapp.server import create_app, Broadcaster


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


def test_cors_header_present(client):
    resp = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in resp.headers


def test_unknown_route_returns_404(client):
    resp = client.get("/api/nonexistent")
    assert resp.status_code == 404


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
    with client.websocket_connect("/ws/spectrum") as ws:
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
    with client.websocket_connect("/ws/agent-events") as ws:
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
