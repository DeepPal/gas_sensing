import asyncio

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
