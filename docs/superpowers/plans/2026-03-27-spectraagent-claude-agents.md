# SpectraAgent Phase 7 — Claude API Agents: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build four async Claude API agents (`AnomalyExplainer`, `ExperimentNarrator`, `ReportWriter`, `DiagnosticsAgent`) and a `ClaudeAgentRunner` background dispatcher, wired to `AgentBus`; add a `POST /api/agents/ask` SSE streaming endpoint for free-text LLM queries.

**Architecture:** Claude agents are fully decoupled from the 20 Hz signal path. They subscribe to `AgentBus` via `ClaudeAgentRunner` (an async task started in the FastAPI startup event). Each agent is an async coroutine that calls `anthropic.AsyncAnthropic.messages.create` (non-streaming) or `.messages.stream` (streaming for `/api/agents/ask`). On missing API key or failure they emit a `claude_unavailable` info event and continue — no exceptions propagate. Event log for `/api/agents/ask` context is maintained as a `collections.deque(maxlen=200)` on `app.state.agent_events_log`, populated by a background subscriber task registered in startup.

**Tech Stack:** Python 3.9+, `anthropic>=0.25.0` (already installed as `0.86.0`), `asyncio`, `FastAPI`, `pytest`. The `anthropic` package is already in `pyproject.toml` dependencies. No new pip dependencies are required for the implementation — however `pytest-asyncio` is NOT currently installed; tests use the project's established pattern of `loop.run_until_complete()` sync wrappers instead.

**Key Constraints:**
- Never add `__init__.py` to any test subdirectory — `--import-mode=importlib` is set in pyproject.toml.
- All tests are synchronous functions that manage their own `asyncio.new_event_loop()` — this matches the established pattern in `test_agent_bus.py`, `test_quality.py`, `test_drift.py`, etc.
- `_get_client()` is a module-level function in `claude_agents.py` that returns `anthropic.AsyncAnthropic(api_key=...)` or `None` — it is patchable via `unittest.mock.patch`.
- `AgentBus.emit()` is called from `on_event()` coroutines which run inside the asyncio loop, so `emit()` calls `_loop.call_soon_threadsafe(self._fanout, event)` — this is correct (emit works from both sync and async contexts when the loop is set).
- The `ClaudeAgentRunner._run()` loop `await`s from `self._q.get()` — it must be started with `asyncio.ensure_future()` inside the FastAPI startup event (after the loop is running).

---

## File Map

### Created

| File | Purpose |
|---|---|
| `spectraagent/webapp/agents/claude_agents.py` | `_BaseClaude`, `AnomalyExplainer`, `ExperimentNarrator`, `ReportWriter`, `DiagnosticsAgent`, `ClaudeAgentRunner` |
| `tests/spectraagent/webapp/agents/test_claude_agents.py` | Full test suite — 9 test cases using mocked anthropic client |

### Modified

| File | What changes |
|---|---|
| `spectraagent/webapp/server.py` | Add `StreamingResponse` import; add `AskRequest` Pydantic model; add `POST /api/agents/ask` SSE endpoint; add startup task to populate `app.state.agent_events_log` deque |
| `spectraagent/__main__.py` | Create `AnomalyExplainer`, `ExperimentNarrator`, `DiagnosticsAgent`, `ReportWriter`, `ClaudeAgentRunner` after existing agents in `start()`; store on `app.state`; start runner in startup handler |
| `tests/spectraagent/webapp/test_server.py` | Add two tests for `/api/agents/ask` |

**Do NOT create `__init__.py` in any test subdirectory.**

---

## Task 1: `claude_agents.py` and `test_claude_agents.py`

**Files:**
- Create: `spectraagent/webapp/agents/claude_agents.py`
- Create: `tests/spectraagent/webapp/agents/test_claude_agents.py`

### Step 1: Write the failing tests first

Create `tests/spectraagent/webapp/agents/test_claude_agents.py`:

```python
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
    """Return a mock that simulates anthropic.AsyncAnthropic with messages.create."""
    mock_content = MagicMock()
    mock_content.text = response_text

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_client = MagicMock()
    mock_client.messages = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)
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
    mock_client.messages.create.assert_not_called()
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
    q = bus.subscribe()

    agent = AnomalyExplainer(bus, auto_explain=True, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
        loop.run_until_complete(agent.on_event(_drift_warn_event()))
    _flush(loop)

    # Only one event should have been emitted
    assert mock_client.messages.create.call_count == 1
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
    mock_client.messages.create.assert_not_called()
    loop.close()


# ---------------------------------------------------------------------------
# Test 6: ExperimentNarrator fires once per new n_points level
# ---------------------------------------------------------------------------


def test_experiment_narrator_fires_once_per_point():
    """ExperimentNarrator fires on n_points=4, must NOT fire again for same n_points=4."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = ExperimentNarrator(bus, auto_explain=True)
    mock_client = _mock_claude_client("Langmuir model selected. Good fit.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))  # same n
    _flush(loop)

    assert mock_client.messages.create.call_count == 1
    loop.close()


def test_experiment_narrator_fires_again_for_higher_n_points():
    """ExperimentNarrator fires again when n_points increases (new calibration point added)."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = ExperimentNarrator(bus, auto_explain=True)
    mock_client = _mock_claude_client("More data improves fit.")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=4)))
        loop.run_until_complete(agent.on_event(_model_selected_event(n_points=5)))  # new point
    _flush(loop)

    assert mock_client.messages.create.call_count == 2
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
    q = bus.subscribe()

    agent = DiagnosticsAgent(bus, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))  # same code
    _flush(loop)

    assert mock_client.messages.create.call_count == 1
    loop.close()


def test_diagnostics_different_codes_fire_independently():
    """Different error codes are tracked independently by their own cooldown timers."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    agent = DiagnosticsAgent(bus, cooldown_s=9999.0)
    mock_client = _mock_claude_client()

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        loop.run_until_complete(agent.on_event(_hardware_error_event("E001")))
        loop.run_until_complete(agent.on_event(_hardware_error_event("E002")))  # different code
    _flush(loop)

    assert mock_client.messages.create.call_count == 2
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
    mock_client.messages.create.assert_not_called()
    loop.close()


# ---------------------------------------------------------------------------
# Test 10: ClaudeAgentRunner dispatches drift_warn to AnomalyExplainer
# ---------------------------------------------------------------------------


def test_runner_dispatches_drift_warn_to_anomaly_explainer():
    """ClaudeAgentRunner must route drift_warn events to AnomalyExplainer."""
    bus, loop = _make_bus()
    q = bus.subscribe()

    anomaly = AnomalyExplainer(bus, auto_explain=True, cooldown_s=0.0)
    narrator = ExperimentNarrator(bus, auto_explain=True)
    diagnostics = DiagnosticsAgent(bus, cooldown_s=0.0)
    runner = ClaudeAgentRunner(bus, anomaly, narrator, diagnostics)

    mock_client = _mock_claude_client("Likely thermal expansion.")

    async def run():
        runner.start()
        with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
            bus.emit(_drift_warn_event())
            # Give the runner loop two ticks to process
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        runner.stop()

    loop.run_until_complete(run())
    _flush(loop)

    # AnomalyExplainer should have called Claude
    assert mock_client.messages.create.call_count >= 1
    loop.close()


# ---------------------------------------------------------------------------
# Test 11: ReportWriter.write() returns text from Claude
# ---------------------------------------------------------------------------


def test_report_writer_returns_text():
    """ReportWriter.write() calls Claude and returns the response text."""
    bus, loop = _make_bus()

    writer = ReportWriter(bus)
    mock_client = _mock_claude_client("Methods: LSPR sensor with gold nanoparticles...")

    with patch("spectraagent.webapp.agents.claude_agents._get_client", return_value=mock_client):
        result = loop.run_until_complete(writer.write({
            "session_id": "20260327_120000",
            "gas_label": "Ethanol",
            "concentration": 0.5,
        }))

    assert result is not None
    assert "LSPR" in result or len(result) > 0
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
```

### Step 2: Run the tests and confirm they fail (collection passes, execution fails)

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/webapp/agents/test_claude_agents.py -v --tb=short 2>&1 | head -30
```

Expected output: `ImportError: cannot import name 'AnomalyExplainer' from 'spectraagent.webapp.agents.claude_agents'` (module does not exist yet).

### Step 3: Implement `spectraagent/webapp/agents/claude_agents.py`

Create `spectraagent/webapp/agents/claude_agents.py` with the following complete content:

```python
"""
spectraagent.webapp.agents.claude_agents
=========================================
Layer 2 Claude API Agents. All async, never in the 20 Hz signal path.

Four agents:
    AnomalyExplainer   — reacts to drift_warn events (opt-in: auto_explain=True)
    ExperimentNarrator — reacts to model_selected events (opt-in: auto_explain=True)
    ReportWriter       — user-triggered only via .write(context)
    DiagnosticsAgent   — reacts to hardware_error events (always fires, no opt-in gate)

ClaudeAgentRunner subscribes to AgentBus and dispatches events to the three
event-driven agents. ReportWriter is called directly by route handlers.

_get_client() is the single point of anthropic import — patch this in tests.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT_S = 30.0


# ---------------------------------------------------------------------------
# Client factory — patchable in tests
# ---------------------------------------------------------------------------


def _get_client() -> Optional[Any]:
    """Return anthropic.AsyncAnthropic if ANTHROPIC_API_KEY is set, else None.

    Returns None (instead of raising) when:
    - ANTHROPIC_API_KEY env var is not set
    - anthropic package is not importable (should not happen — it is a
      declared dependency, but we guard defensively)

    This function is intentionally thin so tests can patch it:
        with patch("spectraagent.webapp.agents.claude_agents._get_client",
                   return_value=mock_client):
            ...
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.AsyncAnthropic(api_key=api_key)
    except ImportError:
        log.warning("claude_agents: anthropic package not importable")
        return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _BaseClaude:
    """Base for all Claude agents. Provides _call() with error handling."""

    source: str = "ClaudeAgent"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> None:
        from spectraagent.webapp.agent_bus import AgentEvent
        self._bus = bus
        self._model = model
        self._timeout_s = timeout_s
        self._AgentEvent = AgentEvent

    async def _call(self, prompt: str) -> Optional[str]:
        """Call Claude with the given prompt. Returns text or None.

        On failure (no key, timeout, API error) emits a claude_unavailable
        event (level="info") onto the bus and returns None.  Never raises.
        """
        client = _get_client()
        if client is None:
            self._bus.emit(self._AgentEvent(
                source=self.source,
                level="info",
                type="claude_unavailable",
                data={"reason": "ANTHROPIC_API_KEY not set or anthropic not importable"},
                text=f"{self.source}: Claude unavailable — set ANTHROPIC_API_KEY",
            ))
            return None
        try:
            msg = await asyncio.wait_for(
                client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout=self._timeout_s,
            )
            return msg.content[0].text
        except asyncio.TimeoutError:
            log.warning("%s: Claude API timed out after %.1fs", self.source, self._timeout_s)
            self._bus.emit(self._AgentEvent(
                source=self.source,
                level="warn",
                type="claude_unavailable",
                data={"reason": f"timeout after {self._timeout_s}s"},
                text=f"{self.source}: Claude API timed out after {self._timeout_s}s",
            ))
            return None
        except Exception as exc:
            log.warning("%s: Claude API error: %s", self.source, exc)
            self._bus.emit(self._AgentEvent(
                source=self.source,
                level="warn",
                type="claude_unavailable",
                data={"reason": str(exc)},
                text=f"{self.source}: Claude API error: {exc}",
            ))
            return None


# ---------------------------------------------------------------------------
# AnomalyExplainer
# ---------------------------------------------------------------------------


class AnomalyExplainer(_BaseClaude):
    """Explains drift_warn events using Claude.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    model:
        Claude model ID (default: claude-sonnet-4-6).
    timeout_s:
        API call timeout in seconds (default: 30.0).
    cooldown_s:
        Minimum seconds between successive calls (default: 300.0).
    auto_explain:
        Whether to auto-fire on drift_warn (default: False — opt-in).
    """

    source = "AnomalyExplainer"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cooldown_s: float = 300.0,
        auto_explain: bool = False,
    ) -> None:
        super().__init__(bus, model, timeout_s)
        self._cooldown_s = cooldown_s
        self._auto_explain = auto_explain
        self._last_called: float = 0.0

    def set_auto_explain(self, enabled: bool) -> None:
        """Toggle auto-explain at runtime (e.g. when user changes settings)."""
        self._auto_explain = enabled

    async def on_event(self, event: Any) -> None:
        """Handle an AgentBus event. Only reacts to type='drift_warn'."""
        if event.type != "drift_warn":
            return
        if not self._auto_explain:
            return
        now = time.monotonic()
        if now - self._last_called < self._cooldown_s:
            return
        self._last_called = now

        data = event.data
        prompt = (
            "You are a spectroscopy expert advising on an LSPR biosensor experiment.\n"
            f"The sensor shows wavelength drift.\n"
            f"  Drift rate: {data.get('drift_rate_nm_per_min', '?')} nm/min\n"
            f"  Peak wavelength: {data.get('peak_wavelength', '?')} nm\n"
            f"  Window: {data.get('window_frames', '?')} frames\n"
            "In exactly 2–3 sentences: (1) most likely cause of this drift rate "
            "and (2) recommended corrective action. Be specific and actionable."
        )
        text = await self._call(prompt)
        if text:
            from spectraagent.webapp.agent_bus import AgentEvent
            self._bus.emit(AgentEvent(
                source=self.source,
                level="claude",
                type="anomaly_explanation",
                data={"drift_data": data, "explanation": text},
                text=text,
            ))


# ---------------------------------------------------------------------------
# ExperimentNarrator
# ---------------------------------------------------------------------------


class ExperimentNarrator(_BaseClaude):
    """Narrates calibration model selection events using Claude.

    Fires at most once per unique n_points value — i.e. once per new
    calibration data point added, not on every re-emission of the same count.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    model:
        Claude model ID (default: claude-sonnet-4-6).
    timeout_s:
        API call timeout in seconds (default: 30.0).
    auto_explain:
        Whether to auto-fire on model_selected (default: False — opt-in).
    """

    source = "ExperimentNarrator"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        auto_explain: bool = False,
    ) -> None:
        super().__init__(bus, model, timeout_s)
        self._auto_explain = auto_explain
        self._last_n_points: int = 0

    def set_auto_explain(self, enabled: bool) -> None:
        """Toggle auto-explain at runtime."""
        self._auto_explain = enabled

    async def on_event(self, event: Any) -> None:
        """Handle an AgentBus event. Only reacts to type='model_selected'."""
        if event.type != "model_selected":
            return
        if not self._auto_explain:
            return
        data = event.data
        n = int(data.get("n_points", 0))
        if n <= self._last_n_points:
            return
        self._last_n_points = n

        best_model = data.get("best_model", "unknown")
        best_aic = data.get("best_aic", "?")
        r_squared = float(data.get("r_squared", 0.0))
        prompt = (
            "You are a scientific instrument advisor for an LSPR calibration experiment.\n"
            f"The calibration system just fitted {n} data points.\n"
            f"  Best model: {best_model}\n"
            f"  AICc: {best_aic}\n"
            f"  R²: {r_squared:.4f}\n"
            "In exactly 2–3 sentences: (1) what this model selection tells you about "
            "the sensor response mechanism and (2) what the experimenter should do next. "
            "Use scientific language appropriate for a journal methods section."
        )
        text = await self._call(prompt)
        if text:
            from spectraagent.webapp.agent_bus import AgentEvent
            self._bus.emit(AgentEvent(
                source=self.source,
                level="claude",
                type="experiment_narration",
                data={"calibration_data": data, "narration": text},
                text=text,
            ))


# ---------------------------------------------------------------------------
# ReportWriter
# ---------------------------------------------------------------------------


class ReportWriter(_BaseClaude):
    """Generates a Methods + Results prose report. User-triggered only.

    Never auto-fires. Called directly via route handler:
        text = await app.state.report_writer.write(context)

    Parameters
    ----------
    bus:
        AgentBus for emitting events (used only for claude_unavailable events).
    model:
        Claude model ID (default: claude-sonnet-4-6).
    timeout_s:
        API call timeout in seconds (default: 30.0).
    """

    source = "ReportWriter"

    async def write(self, context: Dict[str, Any]) -> Optional[str]:
        """Generate Methods + Results prose for the given session context.

        Parameters
        ----------
        context:
            A dict containing session metadata, calibration results, peak
            shift data, and any other relevant fields.  Only keys present
            in context are referenced — no numbers are invented.

        Returns
        -------
        str or None
            Two-paragraph report (Methods then Results) in journal style,
            or None if Claude is unavailable.
        """
        prompt = (
            "You are a scientific journal author specializing in biosensor research.\n"
            "Write exactly two paragraphs of Methods + Results prose.\n"
            f"Session context:\n{json.dumps(context, indent=2, default=str)}\n\n"
            "Paragraph 1 — Methods: describe the sensor setup, spectral acquisition "
            "parameters, and calibration procedure using only the information provided.\n"
            "Paragraph 2 — Results: summarize peak shift observations, best-fit "
            "calibration model, and any limit-of-detection estimates.\n"
            "Write in past tense, third person. Journal style. "
            "NEVER invent numbers not present in the context."
        )
        return await self._call(prompt)


# ---------------------------------------------------------------------------
# DiagnosticsAgent
# ---------------------------------------------------------------------------


class DiagnosticsAgent(_BaseClaude):
    """Diagnoses hardware errors using Claude. Always fires — no auto_explain gate.

    Uses per-error-code cooldown so the same error code does not spam Claude.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    model:
        Claude model ID (default: claude-sonnet-4-6).
    timeout_s:
        API call timeout in seconds (default: 30.0).
    cooldown_s:
        Minimum seconds between calls for the SAME error code (default: 60.0).
    """

    source = "DiagnosticsAgent"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cooldown_s: float = 60.0,
    ) -> None:
        super().__init__(bus, model, timeout_s)
        self._cooldown_s = cooldown_s
        self._last_called: Dict[str, float] = {}

    async def on_event(self, event: Any) -> None:
        """Handle an AgentBus event. Only reacts to level='error', type='hardware_error'."""
        if event.level != "error" or event.type != "hardware_error":
            return

        code = str(event.data.get("error_code", "unknown"))
        now = time.monotonic()
        if now - self._last_called.get(code, 0.0) < self._cooldown_s:
            return
        self._last_called[code] = now

        data = event.data
        prompt = (
            "You are an expert spectrometer hardware technician.\n"
            f"A hardware error occurred on the instrument.\n"
            f"  Error code:    {data.get('error_code', '?')}\n"
            f"  Error message: {data.get('error_message', '?')}\n"
            f"  Hardware:      {data.get('hardware_model', '?')}\n"
            f"  Last success:  {data.get('last_successful_frame_ago_s', '?')}s ago\n"
            "Respond in under 80 words:\n"
            "(1) Most likely root cause of this specific error code.\n"
            "(2) Step-by-step troubleshooting procedure (numbered list, max 3 steps).\n"
            "Be precise. Do not repeat the error message."
        )
        text = await self._call(prompt)
        if text:
            from spectraagent.webapp.agent_bus import AgentEvent
            self._bus.emit(AgentEvent(
                source=self.source,
                level="claude",
                type="diagnostics",
                data={"error_data": data, "diagnosis": text},
                text=text,
            ))


# ---------------------------------------------------------------------------
# ClaudeAgentRunner — background dispatch loop
# ---------------------------------------------------------------------------


class ClaudeAgentRunner:
    """Subscribes to AgentBus and dispatches events to Claude agents.

    Lifecycle (called from FastAPI startup event, after setup_loop() fires):
        runner = ClaudeAgentRunner(bus, anomaly, narrator, diagnostics)
        runner.start()   # subscribe + asyncio.ensure_future(_run())

    On FastAPI shutdown (or test teardown):
        runner.stop()    # unsubscribe + task.cancel()

    ReportWriter is NOT included here — it is called directly by route handlers
    on user request, never from the event loop.
    """

    def __init__(
        self,
        bus: Any,
        anomaly_explainer: AnomalyExplainer,
        experiment_narrator: ExperimentNarrator,
        diagnostics_agent: DiagnosticsAgent,
    ) -> None:
        self._bus = bus
        self._agents = [anomaly_explainer, experiment_narrator, diagnostics_agent]
        self._q: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Subscribe to the bus and start the dispatch coroutine.

        Must be called from within a running asyncio event loop
        (i.e. from an ``@app.on_event("startup")`` handler).
        """
        self._q = self._bus.subscribe()
        self._task = asyncio.ensure_future(self._run())
        log.info("ClaudeAgentRunner started")

    def stop(self) -> None:
        """Unsubscribe from bus and cancel the dispatch task."""
        if self._q is not None:
            self._bus.unsubscribe(self._q)
            self._q = None
        if self._task is not None:
            self._task.cancel()
            self._task = None
        log.info("ClaudeAgentRunner stopped")

    async def _run(self) -> None:
        """Main dispatch loop. Runs until cancelled."""
        while True:
            try:
                event = await self._q.get()
                for agent in self._agents:
                    try:
                        await agent.on_event(event)
                    except Exception as exc:
                        log.warning(
                            "ClaudeAgentRunner: %s.on_event() raised: %s",
                            agent.source,
                            exc,
                        )
            except asyncio.CancelledError:
                log.info("ClaudeAgentRunner: dispatch loop cancelled")
                break
            except Exception as exc:
                log.warning("ClaudeAgentRunner: unexpected dispatch error: %s", exc)
```

### Step 4: Run the tests and confirm they pass

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/webapp/agents/test_claude_agents.py -v --tb=short
```

Expected output:
```
tests/spectraagent/webapp/agents/test_claude_agents.py::test_no_api_key_emits_claude_unavailable PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_anomaly_explainer_disabled_by_default PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_anomaly_explainer_fires_when_enabled PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_anomaly_explainer_respects_cooldown PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_anomaly_explainer_ignores_wrong_event_type PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_experiment_narrator_fires_once_per_point PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_experiment_narrator_fires_again_for_higher_n_points PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_diagnostics_fires_on_hardware_error PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_diagnostics_respects_per_code_cooldown PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_diagnostics_different_codes_fire_independently PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_diagnostics_ignores_non_error_events PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_runner_dispatches_drift_warn_to_anomaly_explainer PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_report_writer_returns_text PASSED
tests/spectraagent/webapp/agents/test_claude_agents.py::test_set_auto_explain_enables_anomaly_explainer PASSED
14 passed in X.XXs
```

### Step 5: Confirm full test suite still passes

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/ -v --tb=short -q
```

Expected: All previously passing tests (755+) plus new 14 pass. Zero regressions.

---

## Task 2: Wire `ClaudeAgentRunner` into `__main__.py`

**Files:**
- Modify: `spectraagent/__main__.py`

### Step 1: Write the test first (in `test_server.py` — server startup test covers wiring)

The existing test `test_start_simulate_no_browser_serves_health` is not in the test files reviewed — the startup integration is verified by running the full test suite. Before modifying `__main__.py`, confirm the current baseline passes:

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/ -q --tb=short
```

Expected: 755+ tests pass.

### Step 2: Add Claude agent creation to `start()` in `__main__.py`

In `spectraagent/__main__.py`, locate the block labeled `# Step 5a: Create deterministic agents` (lines 189–203). Immediately after the line `typer.echo("Agents ready: Quality, Drift, Calibration, Planner")` (currently line 203), add the following block:

```python
    # Step 5b: Create Claude API agents
    from spectraagent.webapp.agents.claude_agents import (
        AnomalyExplainer,
        ClaudeAgentRunner,
        DiagnosticsAgent,
        ExperimentNarrator,
        ReportWriter,
    )

    app.state.anomaly_explainer = AnomalyExplainer(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        cooldown_s=cfg.agents.anomaly_explainer_cooldown_s,
        auto_explain=cfg.agents.auto_explain,
    )
    app.state.experiment_narrator = ExperimentNarrator(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        auto_explain=cfg.agents.auto_explain,
    )
    app.state.diagnostics_agent = DiagnosticsAgent(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        cooldown_s=cfg.agents.diagnostics_cooldown_s,
    )
    app.state.report_writer = ReportWriter(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
    )
    app.state.claude_runner = ClaudeAgentRunner(
        agent_bus,
        anomaly_explainer=app.state.anomaly_explainer,
        experiment_narrator=app.state.experiment_narrator,
        diagnostics_agent=app.state.diagnostics_agent,
    )
    typer.echo(
        "Claude agents ready: AnomalyExplainer, ExperimentNarrator, "
        "DiagnosticsAgent, ReportWriter"
    )
```

### Step 3: Update the startup handler in `start()` to start the runner

Locate the `@app.on_event("startup")` handler currently at lines 206–216. It contains only the acquisition thread start. Add the runner start **after** `acq_thread.start()`:

The existing handler:
```python
    @app.on_event("startup")
    async def _start_acquisition_loop() -> None:
        """Start acquisition AFTER setup_loop() fires so AgentBus is ready."""
        acq_thread = threading.Thread(
            target=_acquisition_loop,
            args=(driver, app),
            daemon=True,
            name="spectraagent-acquisition",
        )
        acq_thread.start()
        typer.echo("Acquisition loop started")
```

Replace it with:
```python
    @app.on_event("startup")
    async def _start_acquisition_loop() -> None:
        """Start acquisition and Claude runner AFTER setup_loop() fires."""
        acq_thread = threading.Thread(
            target=_acquisition_loop,
            args=(driver, app),
            daemon=True,
            name="spectraagent-acquisition",
        )
        acq_thread.start()
        typer.echo("Acquisition loop started")

        # Start Claude agent runner (requires live event loop)
        claude_runner = getattr(app.state, "claude_runner", None)
        if claude_runner is not None:
            claude_runner.start()
            typer.echo("Claude agent runner started")
```

### Step 4: Verify the full test suite still passes

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/ -q --tb=short
```

Expected: All tests pass. The `ClaudeAgentRunner` is only started in the CLI `start()` path, not in `create_app()` used by tests — so no test infrastructure is affected.

### Step 5: Smoke-test the CLI (manual check, no browser)

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m spectraagent start --simulate --no-browser --port 8766
```

Expected output includes:
```
WARNING: ANTHROPIC_API_KEY not set -- Claude agents disabled.
Hardware: SimulationDriver
Physics: LSPRPlugin
Agents ready: Quality, Drift, Calibration, Planner
Claude agents ready: AnomalyExplainer, ExperimentNarrator, DiagnosticsAgent, ReportWriter
Acquisition loop started
Claude agent runner started
Serving at http://127.0.0.1:8766
```

Press Ctrl+C to stop.

---

## Task 3: `POST /api/agents/ask` SSE endpoint in `server.py`

**Files:**
- Modify: `spectraagent/webapp/server.py`
- Modify: `tests/spectraagent/webapp/test_server.py`

### Step 1: Write the failing tests first

Open `tests/spectraagent/webapp/test_server.py` and append these tests at the end of the file (after the last existing test `test_calibration_suggest_returns_200`):

```python
# ---------------------------------------------------------------------------
# Task 3: /api/agents/ask SSE endpoint
# ---------------------------------------------------------------------------


def test_agents_ask_endpoint_exists(client):
    """POST /api/agents/ask exists (not 404/405)."""
    resp = client.post("/api/agents/ask", json={"query": "What is happening?"})
    assert resp.status_code not in (404, 405), (
        f"Expected endpoint to exist, got {resp.status_code}"
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
```

Note: `patch("spectraagent.webapp.server._get_ask_client", return_value=None)` — this requires `_get_ask_client` to be a module-level function in `server.py` (see implementation step below).

### Step 2: Run the failing tests to confirm they fail

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short -k "ask"
```

Expected: `ImportError` or `FAILED` with `AttributeError: module has no attribute '_get_ask_client'`.

### Step 3: Implement the changes in `server.py`

Make the following targeted changes to `spectraagent/webapp/server.py`:

**Change 1: Add `StreamingResponse` to the `fastapi.responses` import and add `collections.deque`.**

Find this line (currently line 21):
```python
from fastapi.responses import JSONResponse
```
Replace with:
```python
from collections import deque

from fastapi.responses import JSONResponse, StreamingResponse
```

**Change 2: Add `AskRequest` Pydantic model after the existing `CalibrationPoint` model.**

Find this block (currently around line 77–79):
```python
class CalibrationPoint(BaseModel):
    concentration: float
    delta_lambda: float
```
Immediately after it (before the `# Module-level AgentBus singleton` comment), add:
```python

class AskRequest(BaseModel):
    query: str
```

**Change 3: Add module-level `_get_ask_client` helper after the `_agent_bus` singleton line.**

Find this line (currently line 82):
```python
_agent_bus = AgentBus()
```
After it, add:
```python

_ASK_MODEL = "claude-sonnet-4-6"


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
```

**Change 4: Add `app.state.agent_events_log` initialization and background logging task in `create_app()`.**

Find the block in `create_app()` that initializes app state (around lines 115–123, currently):
```python
    # Store state for route handlers
    app.state.simulate = simulate
    app.state.driver = None
    app.state.plugin = None
    app.state.reference = None
    app.state.cached_ref = None

    # Store AgentBus on app.state so tests and other modules can reach it
    app.state.agent_bus = _agent_bus
```
After `app.state.agent_bus = _agent_bus`, add:
```python
    # Bounded event log for /api/agents/ask context (last 200 events)
    app.state.agent_events_log = deque(maxlen=200)
```

**Change 5: Add background event-logging task to the existing `_startup` handler.**

Find the current `_startup` handler (around lines 124–127):
```python
    @app.on_event("startup")
    async def _startup() -> None:
        """Wire AgentBus to the running event loop once uvicorn starts."""
        _agent_bus.setup_loop(asyncio.get_running_loop())
```
Replace with:
```python
    @app.on_event("startup")
    async def _startup() -> None:
        """Wire AgentBus to the running event loop once uvicorn starts."""
        _agent_bus.setup_loop(asyncio.get_running_loop())

        # Start background task that populates agent_events_log for ask context
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

        asyncio.ensure_future(_log_events())
```

**Change 6: Add `POST /api/agents/ask` endpoint.**

Locate the `# Calibration API` section (around line 240). After the `@app.post("/api/calibration/suggest")` handler and before the `# Static files` section, add the following new route section:

```python
    # ------------------------------------------------------------------
    # Claude API — free-text query with SSE streaming response
    # ------------------------------------------------------------------

    @app.post("/api/agents/ask")
    async def agents_ask(request: AskRequest) -> StreamingResponse:
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
                    f'data: {json.dumps({"text": f"Error: {exc}", "done": False})}\n\n'
                )
            finally:
                yield f'data: {json.dumps({"done": True})}\n\n'

        return StreamingResponse(_generate(), media_type="text/event-stream")
```

### Step 4: Run the new server tests

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/webapp/test_server.py -v --tb=short
```

Expected: All existing tests pass plus the 4 new ask tests pass.

Note: The `patch` import needed by the new tests — add `from unittest.mock import patch` at the top of `test_server.py`. The existing test file does not have this import. Add it after the existing imports:
```python
from unittest.mock import patch
```

### Step 5: Run the complete test suite

```
cd C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula
python -m pytest tests/spectraagent/ -v --tb=short -q
```

Expected: All tests pass. Count should be 755 (prior) + 14 (claude_agents) + 4 (server ask) + 1 (patch import enables test) = 769+ passing, 0 failures.

---

## Import Consistency Reference

The following import chain must be internally consistent across all new and modified files:

| Symbol | Canonical import path |
|---|---|
| `AgentBus` | `from spectraagent.webapp.agent_bus import AgentBus` |
| `AgentEvent` | `from spectraagent.webapp.agent_bus import AgentEvent` |
| `AnomalyExplainer` | `from spectraagent.webapp.agents.claude_agents import AnomalyExplainer` |
| `ExperimentNarrator` | `from spectraagent.webapp.agents.claude_agents import ExperimentNarrator` |
| `ReportWriter` | `from spectraagent.webapp.agents.claude_agents import ReportWriter` |
| `DiagnosticsAgent` | `from spectraagent.webapp.agents.claude_agents import DiagnosticsAgent` |
| `ClaudeAgentRunner` | `from spectraagent.webapp.agents.claude_agents import ClaudeAgentRunner` |
| `_get_client` | `spectraagent.webapp.agents.claude_agents._get_client` (patch target) |
| `_get_ask_client` | `spectraagent.webapp.server._get_ask_client` (patch target) |
| `_ASK_MODEL` | module-level constant in `spectraagent/webapp/server.py` |

---

## Self-Review Checklist

- [ ] `claude_agents.py` has no `__init__.py` in tests dir — confirmed, using importlib mode
- [ ] All agents import `AgentEvent` lazily inside methods (except `_BaseClaude.__init__` which imports it once) — avoids circular imports at module load time
- [ ] `_get_client()` in `claude_agents.py` and `_get_ask_client()` in `server.py` are both patchable module-level functions
- [ ] `DiagnosticsAgent` has no `auto_explain` parameter — it always fires
- [ ] `AnomalyExplainer` and `ExperimentNarrator` both have `set_auto_explain(bool)` for runtime toggling
- [ ] `ClaudeAgentRunner.start()` uses `asyncio.ensure_future()` — correct for calling inside a startup handler where the loop is already running
- [ ] `ClaudeAgentRunner.stop()` cancels the task — safe to call from shutdown handlers
- [ ] `app.state.agent_events_log` is a `deque(maxlen=200)` — bounded, no memory leak
- [ ] SSE `_generate()` generator always yields `done: true` in a `finally` block — even on exception the stream closes cleanly
- [ ] `stream.text_stream` is the correct async iterator on `anthropic>=0.25.0` `AsyncMessageStream` — confirmed by inspection of installed version `0.86.0`
- [ ] `asyncio.wait_for(..., timeout=self._timeout_s)` used in `_call()` — not a raw `asyncio.TimeoutError` catch without the wrapper
- [ ] Cooldown uses `time.monotonic()` not `time.time()` — correct for duration measurements
- [ ] Per-error-code cooldown in `DiagnosticsAgent` uses a `Dict[str, float]` keyed by `str(error_code)` — handles integer and string error codes
- [ ] The `from unittest.mock import patch` import is added to `test_server.py`
- [ ] No `__init__.py` created in `tests/spectraagent/webapp/agents/` — correctly absent
- [ ] Full test suite (755+ tests) passes after all three tasks complete

---

### Critical Files for Implementation

- `C:/Users/deepp/Desktop/Chula_Work/PRojects/Main_Research_Chula/spectraagent/webapp/agents/claude_agents.py`
- `C:/Users/deepp/Desktop/Chula_Work/PRojects/Main_Research_Chula/tests/spectraagent/webapp/agents/test_claude_agents.py`
- `C:/Users/deepp/Desktop/Chula_Work/PRojects/Main_Research_Chula/spectraagent/webapp/server.py`
- `C:/Users/deepp/Desktop/Chula_Work/PRojects/Main_Research_Chula/spectraagent/__main__.py`
- `C:/Users/deepp/Desktop/Chula_Work/PRojects/Main_Research_Chula/tests/spectraagent/webapp/test_server.py`

---

**Important note:** This is a read-only planning session — I cannot write files. You will need to save the above plan to:

`C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula\docs\superpowers\plans\2026-03-27-spectraagent-claude-agents.md`

The plan directory already exists at `C:\Users\deepp\Desktop\Chula_Work\PRojects\Main_Research_Chula\docs\superpowers\plans\` (confirmed — it contains the two prior plan files). You can copy the markdown above directly into that file.