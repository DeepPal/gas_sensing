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

Knowledge integration
---------------------
When ``spectraagent.knowledge`` is available, agents build their prompts using
context builders that assemble:
  - Sensor physics background (sensor-type-aware)
  - Ranked failure mode candidates (for anomaly events)
  - SensorMemory history (accumulated observed values, never hardcoded literature)
  - ICH Q2(R1) compliance readiness (for calibration events)
  - Analyte chemistry properties (sensor-independent facts)

This transforms generic "you are a spectroscopy expert" prompts into deeply
domain-informed reasoning grounded in what *this specific sensor* has actually
measured. If the knowledge package is unavailable, agents fall back gracefully
to the previous generic prompts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TIMEOUT_S = 30.0

# ---------------------------------------------------------------------------
# Optional knowledge base — graceful fallback when unavailable
# ---------------------------------------------------------------------------

try:
    from spectraagent.knowledge.context_builders import (
        build_anomaly_context as _build_anomaly_context,
    )
    from spectraagent.knowledge.context_builders import (
        build_calibration_narration_context as _build_calibration_narration_context,
    )
    from spectraagent.knowledge.context_builders import (
        build_hardware_diagnostics_context as _build_hardware_diagnostics_context,
    )
    from spectraagent.knowledge.context_builders import (
        build_report_context as _build_report_context,
    )
    _KB_AVAILABLE = True
except ImportError:
    _KB_AVAILABLE = False
    log.debug("claude_agents: spectraagent.knowledge not available — using generic prompts")


def knowledge_backend_status() -> dict[str, str | bool]:
    """Return whether domain knowledge context builders are available.

    This is used by health/status endpoints so researchers can see whether
    Claude responses are grounded in sensor-specific knowledge or generic mode.
    """
    if _KB_AVAILABLE:
        return {
            "knowledge_base_available": True,
            "knowledge_context_mode": "domain",
            "knowledge_status": "ok",
        }
    return {
        "knowledge_base_available": False,
        "knowledge_context_mode": "generic",
        "knowledge_status": "fallback",
    }


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
        memory: Optional[Any] = None,
    ) -> None:
        from spectraagent.webapp.agent_bus import AgentEvent
        self._bus = bus
        self._model = model
        self._timeout_s = timeout_s
        self._memory = memory   # SensorMemory instance (or None — no history yet)
        self._AgentEvent = AgentEvent
        self._client: Optional[Any] = None

    async def on_event(self, event: Any) -> None:  # noqa: B027 — intentional no-op base
        """Handle an AgentBus event. Subclasses override to react to specific types."""

    async def _call(self, prompt: str) -> Optional[str]:
        """Call Claude with the given prompt. Returns text or None.

        Uses the streaming API (messages.stream) so that the call works with
        local API proxies (e.g. the VS Code Claude extension) that only
        proxy SSE streaming, not the standard blocking messages.create endpoint.

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
            chunks: list[str] = []

            async def _stream_collect() -> str:
                async with client.messages.stream(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    async for text in stream.text_stream:
                        chunks.append(text)
                return "".join(chunks)

            result = await asyncio.wait_for(_stream_collect(), timeout=self._timeout_s)
            return result if result else None
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

    When ``spectraagent.knowledge`` is available, each prompt includes:
    - Sensor physics preamble (sensor-type-aware: lspr, spr, fluorescence, optical)
    - Top-3 failure mode candidates ranked against observed drift/SNR symptoms
    - SensorMemory history: past LOD, drift stats, recurrence alert if this
      failure type has appeared before
    - Analyte chemistry (interferents, vapour pressure, functional group)

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
    memory:
        SensorMemory instance for sensor history context (default: None).
    sensor_type:
        Sensor modality hint for physics preamble: "lspr", "spr",
        "fluorescence", or "optical" (generic, default).
    get_analyte:
        Callable returning current analyte name (or None).  Called at event
        time so it always reflects the active session's analyte.
    """

    source = "AnomalyExplainer"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cooldown_s: float = 300.0,
        auto_explain: bool = False,
        memory: Optional[Any] = None,
        sensor_type: str = "optical",
        get_analyte: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(bus, model, timeout_s, memory=memory)
        self._cooldown_s = cooldown_s
        self._auto_explain = auto_explain
        self._sensor_type = sensor_type
        self._get_analyte = get_analyte
        self._last_called: float = float("-inf")  # "never called" → first call always fires

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
        meas_lines = ""
        if data.get("concentration_ppm") is not None:
            meas_lines += f"  Concentration estimate: {data['concentration_ppm']:.4f} ppm\n"
        if data.get("ci_low") is not None and data.get("ci_high") is not None:
            meas_lines += f"  90% CI: [{data['ci_low']:.4f}, {data['ci_high']:.4f}] ppm\n"
        if data.get("peak_shift_nm") is not None:
            meas_lines += f"  Peak wavelength shift (Δλ): {data['peak_shift_nm']:.4f} nm\n"
        if data.get("snr") is not None:
            meas_lines += f"  Signal-to-noise ratio: {data['snr']:.1f}\n"

        if _KB_AVAILABLE:
            analyte = self._get_analyte() if self._get_analyte is not None else None
            ctx = _build_anomaly_context(data, analyte, self._memory, self._sensor_type)
            prompt = (
                ctx
                + "\n\n---\n"
                "**Current event measurements:**\n"
                f"  Drift rate: {data.get('drift_rate_nm_per_min', '?')} nm/min\n"
                f"  Peak wavelength: {data.get('peak_wavelength', '?')} nm\n"
                f"  Window: {data.get('window_frames', '?')} frames\n"
                + meas_lines
                + "\nIn exactly 2–3 sentences: (1) most likely cause of this drift rate, "
                "citing the failure mode evidence above, and (2) recommended corrective "
                "action. Be specific and actionable."
            )
        else:
            prompt = (
                "You are a spectroscopy expert advising on an optical sensor experiment.\n"
                f"The sensor shows wavelength drift.\n"
                f"  Drift rate: {data.get('drift_rate_nm_per_min', '?')} nm/min\n"
                f"  Peak wavelength: {data.get('peak_wavelength', '?')} nm\n"
                f"  Window: {data.get('window_frames', '?')} frames\n"
                + meas_lines
                + "In exactly 2–3 sentences: (1) most likely cause of this drift rate "
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

    When ``spectraagent.knowledge`` is available, each prompt includes:
    - Current calibration result summary (R², LOD, LOQ, RMSE, model, AICc)
    - Historical comparison from SensorMemory (LOD trend, sensitivity drift)
    - ICH Q2(R1) compliance readiness checklist
    - Analyte chemistry context

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
    memory:
        SensorMemory instance for historical comparison (default: None).
    get_analyte:
        Callable returning current analyte name (or None).
    """

    source = "ExperimentNarrator"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        auto_explain: bool = False,
        memory: Optional[Any] = None,
        get_analyte: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(bus, model, timeout_s, memory=memory)
        self._auto_explain = auto_explain
        self._get_analyte = get_analyte
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
        stat_lines = ""
        if data.get("lod_ppm") is not None:
            stat_lines += f"  LOD: {data['lod_ppm']:.4f} ppm\n"
        if data.get("loq_ppm") is not None:
            stat_lines += f"  LOQ: {data['loq_ppm']:.4f} ppm\n"
        if data.get("rmse_ppm") is not None:
            stat_lines += f"  Calibration RMSE: {data['rmse_ppm']:.4f} ppm\n"

        if _KB_AVAILABLE:
            analyte = self._get_analyte() if self._get_analyte is not None else None
            ctx = _build_calibration_narration_context(data, analyte, self._memory)
            prompt = (
                ctx
                + "\n\n---\n"
                "In exactly 2–3 sentences using scientific journal language: "
                "(1) what this model selection reveals about the sensor's response "
                "mechanism and the analyte–surface interaction, and (2) what "
                "measurement the experimenter should run next to advance toward "
                "ICH Q2(R1) compliance. Cite the R² and LOD values."
            )
        else:
            prompt = (
                "You are a scientific instrument advisor for a calibration experiment.\n"
                f"The calibration system just fitted {n} data points.\n"
                f"  Best model: {best_model}\n"
                f"  AICc: {best_aic}\n"
                f"  R²: {r_squared:.4f}\n"
                + stat_lines
                + "In exactly 2–3 sentences: (1) what this model selection tells you about "
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

    When ``spectraagent.knowledge`` is available, the prompt includes:
    - Full session data (JSON)
    - Sensor health summary from SensorMemory (total sessions, drift stats)
    - Historical performance comparison (is this session's LOD better or worse
      than the sensor's typical performance?)
    - Analyte chemistry properties for the Methods section
    - ICH Q2(R1) compliance readiness

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
    memory:
        SensorMemory instance for historical context (default: None).
    get_analyte:
        Callable returning current analyte name.  Falls back to
        ``context.get("gas_label")`` when not provided.
    """

    source = "ReportWriter"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        memory: Optional[Any] = None,
        get_analyte: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        super().__init__(bus, model, timeout_s, memory=memory)
        self._get_analyte = get_analyte

    async def write(self, context: dict[str, Any]) -> Optional[str]:
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
        # Analyte: prefer explicit context key, fall back to callable
        analyte: Optional[str] = context.get("gas_label")
        if not analyte and self._get_analyte is not None:
            analyte = self._get_analyte()

        if _KB_AVAILABLE:
            ctx_block = _build_report_context(context, analyte, self._memory)
            prompt = (
                "You are a scientific journal author specializing in optical chemical "
                "sensor research.\n"
                "Write exactly two paragraphs of Methods + Results prose.\n\n"
                + ctx_block
                + "\n\n"
                "Paragraph 1 — Methods: describe the sensor setup, spectral acquisition "
                "parameters, and calibration procedure using only the information provided.\n"
                "Paragraph 2 — Results: summarize peak shift observations, best-fit "
                "calibration model, LOD/LOQ estimates, and — where sensor memory history "
                "is provided — note whether this session's performance is better or worse "
                "than the sensor's typical baseline.\n"
                "Write in past tense, third person. Journal style. "
                "NEVER invent numbers not present in the context."
            )
        else:
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

    When ``spectraagent.knowledge`` is available, the prompt includes:
    - Known error code database with physical mechanism explanations
      (VISA VI_ERROR_TMO, device not powered, SCAN_PENDING, etc.)
    - General hardware diagnostic checklist (power, USB, driver, stale state)

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
        self._last_called: dict[str, float] = {}

    async def on_event(self, event: Any) -> None:
        """Handle an AgentBus event. Only reacts to level='error', type='hardware_error'."""
        if event.level != "error" or event.type != "hardware_error":
            return

        code = str(event.data.get("error_code", "unknown"))
        now = time.monotonic()
        if now - self._last_called.get(code, float("-inf")) < self._cooldown_s:
            return
        self._last_called[code] = now

        data = event.data
        meas_lines = ""
        if data.get("concentration_ppm") is not None:
            meas_lines += f"  Last concentration estimate: {data['concentration_ppm']:.4f} ppm\n"
        if data.get("snr") is not None:
            meas_lines += f"  Last SNR: {data['snr']:.1f}\n"
        if data.get("peak_shift_nm") is not None:
            meas_lines += f"  Last peak shift (Δλ): {data['peak_shift_nm']:.4f} nm\n"

        if _KB_AVAILABLE:
            hw_ctx = _build_hardware_diagnostics_context(data)
            prompt = (
                hw_ctx
                + "\n\n---\n"
                "**Current error instance:**\n"
                f"  Error code:    {data.get('error_code', '?')}\n"
                f"  Error message: {data.get('error_message', '?')}\n"
                f"  Hardware:      {data.get('hardware_model', '?')}\n"
                f"  Last success:  {data.get('last_successful_frame_ago_s', '?')}s ago\n"
                + meas_lines
                + "\nRespond in under 80 words:\n"
                "(1) Confirm the root cause, citing the known-code explanation above.\n"
                "(2) Step-by-step troubleshooting procedure (numbered list, max 3 steps).\n"
                "Do not repeat the error message."
            )
        else:
            prompt = (
                "You are an expert spectrometer hardware technician.\n"
                f"A hardware error occurred on the instrument.\n"
                f"  Error code:    {data.get('error_code', '?')}\n"
                f"  Error message: {data.get('error_message', '?')}\n"
                f"  Hardware:      {data.get('hardware_model', '?')}\n"
                f"  Last success:  {data.get('last_successful_frame_ago_s', '?')}s ago\n"
                + meas_lines
                + "Respond in under 80 words:\n"
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

    def add_agent(self, agent: _BaseClaude) -> None:
        """Register an additional agent to receive dispatched events.

        Must be called before :meth:`start` — agents added after start are
        silently ignored because the dispatch loop holds a reference to the
        list at the time start() was called.
        """
        self._agents.append(agent)

    def start(self) -> None:
        """Subscribe to the bus and start the dispatch coroutine.

        Must be called from within a running asyncio event loop
        (i.e. from an ``@app.on_event("startup")`` handler).
        """
        self._q = self._bus.subscribe()
        self._task = asyncio.ensure_future(self._run())
        log.info("ClaudeAgentRunner started (agents: %s)", [a.source for a in self._agents])

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
        assert q is not None, "_run() called before start()"
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
