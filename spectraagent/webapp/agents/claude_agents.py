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
        self._client: Optional[Any] = None

    async def _call(self, prompt: str) -> Optional[str]:
        """Call Claude with the given prompt. Returns text or None.

        On failure (no key, timeout, API error) emits a claude_unavailable
        event (level="info") onto the bus and returns None.  Never raises.
        """
        if self._client is None:
            self._client = _get_client()
        client = self._client
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
            self._bus.emit(self._AgentEvent(
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
            self._bus.emit(self._AgentEvent(
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
            self._bus.emit(self._AgentEvent(
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
        q = self._q  # local ref; immune to stop() clearing self._q
        while True:
            try:
                event = await q.get()
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
