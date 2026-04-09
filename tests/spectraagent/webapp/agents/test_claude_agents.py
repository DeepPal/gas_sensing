"""
Tests for spectraagent.webapp.agents.claude_agents

All tests are synchronous functions that manage their own asyncio event loop,
following the established project pattern (see test_agent_bus.py).

Mock pattern: patch 'spectraagent.webapp.agents.claude_agents._get_client'
to return a pre-configured AsyncMock that simulates anthropic.AsyncAnthropic.
"""
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent
from spectraagent.webapp.agents.claude_agents import (
    AnomalyExplainer,
    ClaudeAgentRunner,
    DiagnosticsAgent,
    ExperimentNarrator,
    ReportWriter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bus():
    """Return (bus, loop) with setup_loop called."""
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    """Run one event-loop tick so call_soon_threadsafe callbacks execute."""
    loop.run_until_complete(asyncio.sleep(0))


def _drift_warn_event():
    return AgentEvent(
        source="DriftAgent",
        level="warn",
        type="drift_warn",
        data={
            "drift_rate_nm_per_min": 0.12,
            "window_frames": 60,
            "peak_wavelength": 720.0,
            "frame": 120,
        },
        text="Drift detected",
    )


def _model_selected_event(n_points=4):
    return AgentEvent(
        source="CalibrationAgent",
        level="info",
        type="model_selected",
        data={
            "n_points": n_points,
            "best_model": "langmuir",
            "best_aic": -12.5,
            "r_squared": 0.9981,
            "aic_table": [{"model": "langmuir", "aic": -12.5}],
        },
        text="Model selected",
    )


def _hardware_error_event(error_code="E001"):
    return AgentEvent(
        source="HardwareDriver",
        level="error",
        type="hardware_error",
        data={
            "error_code": error_code,
            "error_message": "USB device not responding",
            "hardware_model": "ThorlabsCCS200",
            "last_successful_frame_ago_s": 5.0,
        },
        text="Hardware error",
    )


def _mock_claude_client(response_text="Thermal drift detected. Check temperature."):
    """Return a mock that simulates anthropic.AsyncAnthropic with messages.stream.

    The underlying _BaseClaude._call now uses the streaming API:
        async with client.messages.stream(...) as stream:
            async for text in stream.text_stream: ...
    Each call to messages.stream() returns a fresh async context manager so
    that the text_stream generator is not exhausted on repeated calls.
    """

    def _make_stream_cm():
        async def _text_stream():
            yield response_text

        mock_stream = MagicMock()
        mock_stream.text_stream = _text_stream()

        class _StreamCM:
            async def __aenter__(self):
                return mock_stream

            async def __aexit__(self, *args):
                return False

        return _StreamCM()

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.stream = MagicMock(side_effect=lambda **_kw: _make_stream_cm())
    return mock_client


# ---------------------------------------------------------------------------
# Test 1: No API key emits claude_unavailable
# ---------------------------------------------------------------------------


def test_no_api_key_emits_claude_unavailable():
    """When ANTHROPIC_API_KEY is not set, on_event emits claude_unavailable."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=True, cooldown_s=0.0)

    saved_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=None):
            loop.run_until_complete(agent.on_event(_drift_warn_event()))
        _flush(loop)
        assert not q.empty(), "Expected claude_unavailable event in queue"
        event = q.get_nowait()
        assert event.type == "claude_unavailable"
        assert event.level == "info"
    finally:
        if saved_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved_key
    loop.close()


# ---------------------------------------------------------------------------
# Test 2: AnomalyExplainer disabled by default (auto_explain=False)
# ---------------------------------------------------------------------------


def test_anomaly_explainer_disabled_by_default():
    """auto_explain defaults to False — drift_warn must not trigger any emit."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = AnomalyExplainer(bus)  # default auto_explain=False

    mock_client = _mock_claude_client()
    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)

    assert q.empty(), "No event should be emitted when auto_explain=False"
    mock_client.messages.stream.assert_not_called()
    loop.close()


# ---------------------------------------------------------------------------
# Test 3: AnomalyExplainer fires when auto_explain=True
# ---------------------------------------------------------------------------


def test_anomaly_explainer_fires_when_enabled():
    """With auto_explain=True and cooldown_s=0, drift_warn triggers anomaly_explanation."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=True, cooldown_s=0.0)
    mock_client = _mock_claude_client("Thermal drift. Check temperature stability.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)

    assert not q.empty(), "Expected anomaly_explanation event"
    event = q.get_nowait()
    assert event.type == "anomaly_explanation"
    assert event.level == "claude"
    assert event.source == "AnomalyExplainer"
    assert "explanation" in event.data
    loop.close()


# ---------------------------------------------------------------------------
# Test 4: AnomalyExplainer respects cooldown
# ---------------------------------------------------------------------------


def test_anomaly_explainer_respects_cooldown():
    """Second drift_warn within cooldown window must not trigger a second call."""
    bus, loop = _make_bus()
    _q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=True, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)

    # Only one event should have been emitted
    assert mock_client.messages.stream.call_count == 1
    loop.close()


# ---------------------------------------------------------------------------
# Test 5: AnomalyExplainer ignores wrong event type
# ---------------------------------------------------------------------------


def test_anomaly_explainer_ignores_wrong_event_type():
    """AnomalyExplainer must only react to drift_warn, not quality or other types."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=True, cooldown_s=0.0)
    mock_client = _mock_claude_client()

    quality_event = AgentEvent(
        source="QualityAgent",
        level="warn",
        type="quality",
        data={"snr": 1.5},
        text="Low SNR",
    )

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(quality_event))
    _flush(loop)

    assert q.empty(), "AnomalyExplainer must not react to quality events"
    mock_client.messages.stream.assert_not_called()
    loop.close()


# ---------------------------------------------------------------------------
# Test 6: ExperimentNarrator fires once per new n_points level
# ---------------------------------------------------------------------------


def test_experiment_narrator_fires_once_per_point():
    """ExperimentNarrator fires on n_points=4, must NOT fire again for same n_points=4."""
    bus, loop = _make_bus()
    _q = bus.subscribe()

    agent = ExperimentNarrator(bus, auto_explain=True)
    mock_client = _mock_claude_client("Langmuir model selected. Good fit.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))  # same n
    _flush(loop)

    assert mock_client.messages.stream.call_count == 1
    loop.close()


def test_experiment_narrator_fires_again_for_higher_n_points():
    """ExperimentNarrator fires again when n_points increases (new calibration point added)."""
    bus, loop = _make_bus()

    agent = ExperimentNarrator(bus, auto_explain=True)
    mock_client = _mock_claude_client("More data improves fit.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=5)))  # new point
    _flush(loop)

    assert mock_client.messages.stream.call_count == 2
    loop.close()


# ---------------------------------------------------------------------------
# Test 7: DiagnosticsAgent fires on hardware_error (always, no auto_explain gate)
# ---------------------------------------------------------------------------


def test_diagnostics_fires_on_hardware_error():
    """DiagnosticsAgent always fires on hardware_error — no auto_explain gate."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    # Note: DiagnosticsAgent takes no auto_explain parameter
    agent = DiagnosticsAgent(bus, cooldown_s=0.0)
    mock_client = _mock_claude_client("Check USB cable. Reconnect device.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_hardware_error_event()))
    _flush(loop)

    assert not q.empty(), "Expected diagnostics event"
    event = q.get_nowait()
    assert event.type == "diagnostics"
    assert event.level == "claude"
    assert event.source == "DiagnosticsAgent"
    assert "diagnosis" in event.data
    loop.close()


# ---------------------------------------------------------------------------
# Test 8: DiagnosticsAgent respects per-error-code cooldown
# ---------------------------------------------------------------------------


def test_diagnostics_respects_per_code_cooldown():
    """Same error code within cooldown window triggers only one call."""
    bus, loop = _make_bus()
    _q = bus.subscribe()

    agent = DiagnosticsAgent(bus, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))  # same code
    _flush(loop)

    assert mock_client.messages.stream.call_count == 1
    loop.close()


def test_diagnostics_different_codes_fire_independently():
    """Different error codes are tracked independently by their own cooldown timers."""
    bus, loop = _make_bus()

    agent = DiagnosticsAgent(bus, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))
        loop.run_until_complete(agent.on_event(_hardware_error_event("E002")))  # different code
    _flush(loop)

    assert mock_client.messages.stream.call_count == 2
    loop.close()


# ---------------------------------------------------------------------------
# Test 9: DiagnosticsAgent ignores non-hardware-error events
# ---------------------------------------------------------------------------


def test_diagnostics_ignores_non_error_events():
    """DiagnosticsAgent must ignore events that are not level=error + type=hardware_error."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = DiagnosticsAgent(bus, cooldown_s=0.0)
    mock_client = _mock_claude_client()

    # level=warn type=hardware_error — wrong level
    wrong_level = AgentEvent(
        source="Driver",
        level="warn",
        type="hardware_error",
        data={"error_code": "E001", "error_message": "warning"},
        text="warning",
    )
    # level=error type=quality — wrong type
    wrong_type = AgentEvent(
        source="QualityAgent",
        level="error",
        type="quality",
        data={},
        text="error",
    )

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(wrong_level))
        loop.run_until_complete(agent.on_event(wrong_type))
    _flush(loop)

    assert q.empty()
    mock_client.messages.stream.assert_not_called()
    loop.close()


# ---------------------------------------------------------------------------
# Test 10: ClaudeAgentRunner dispatches drift_warn to AnomalyExplainer
# ---------------------------------------------------------------------------


def test_runner_dispatches_drift_warn_to_anomaly_explainer():
    """ClaudeAgentRunner must route drift_warn events to AnomalyExplainer."""
    bus, loop = _make_bus()
    _q = bus.subscribe()

    anomaly = AnomalyExplainer(bus, auto_explain=True, cooldown_s=0.0)
    narrator = ExperimentNarrator(bus, auto_explain=True)
    diagnostics = DiagnosticsAgent(bus, cooldown_s=0.0)
    runner = ClaudeAgentRunner(bus, anomaly, narrator, diagnostics)

    mock_client = _mock_claude_client("Likely thermal expansion.")

    async def run():
        runner.start()
        with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
            bus.emit(_drift_warn_event())
            # Give the chain: fanout tick → queue.get → on_event → _call → messages.create
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        runner.stop()

    loop.run_until_complete(run())
    _flush(loop)

    # AnomalyExplainer should have called Claude exactly once
    assert mock_client.messages.stream.call_count == 1
    loop.close()


# ---------------------------------------------------------------------------
# Test 11: ReportWriter.write() returns text from Claude
# ---------------------------------------------------------------------------


def test_report_writer_returns_text():
    """ReportWriter.write() calls Claude and returns the response text."""
    bus, loop = _make_bus()

    writer = ReportWriter(bus)
    mock_client = _mock_claude_client("Methods: LSPR sensor with nanoparticles...")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        result = loop.run_until_complete(writer.write({
            "session_id": "20260327_120000",
            "gas_label": "Ethanol",
            "concentration": 0.5,
        }))

    assert result
    loop.close()


# ---------------------------------------------------------------------------
# Test 12: set_auto_explain runtime toggle works
# ---------------------------------------------------------------------------


def test_set_auto_explain_enables_anomaly_explainer():
    """set_auto_explain(True) at runtime causes subsequent drift_warn events to fire."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=False, cooldown_s=0.0)
    mock_client = _mock_claude_client()

    # First call — disabled, no emit
    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)
    assert q.empty()

    # Enable at runtime
    agent.set_auto_explain(True)

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)
    assert not q.empty()
    loop.close()
