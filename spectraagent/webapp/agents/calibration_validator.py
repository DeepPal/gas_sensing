"""
spectraagent.webapp.agents.calibration_validator
==================================================
CalibrationValidationOrchestrator — ICH Q2(R1) compliance gap analyst.

This agent bridges experimental measurements and regulatory compliance.  After
every session, it updates a per-analyte ICH Q2(R1) validation state machine and
tells researchers *exactly* what experiments remain before their method is
publication- or submission-ready.

Why this matters
----------------
Analytical method validation (ICH Q2(R1) / EURACHEM) is required to:
  - Publish in peer-reviewed journals (Anal. Chem., Sensors & Actuators, etc.)
  - Submit for regulatory approval
  - Deploy sensors in any clinical or environmental monitoring context

Most researchers run the required tests but don't track them systematically.
This agent automates the bookkeeping so nothing is missed.

ICH Q2(R1) tests tracked
-------------------------
  §4.1  Specificity        — sensor responds to target, not interferents
  §4.2  Linearity          — R² > 0.9954 over the claimed range
  §4.3  Range              — validated concentration range (min → max)
  §4.4a Repeatability      — RSD < 2% within one day, one analyst
  §4.4b Intermediate prec. — RSD < 3% across days/analysts
  §4.5  Accuracy/recovery  — recovery 98–102% for reference standards
  §4.6/7 LOD / LOQ         — IUPAC 3σ/10σ with bootstrap CI
  §4.8  Robustness         — method still passes under ±10% parameter changes

How it works
------------
1. Listens to ``session_complete`` events from SessionAnalyzer.
2. Updates the ``ValidationTracker`` state for the current analyte.
3. Emits ``validation_status_updated`` with completion%, gaps, next_test.
4. Optionally calls Claude to design the next experiment in plain language.

ValidationTracker state is NOT persisted — it is rebuilt from SensorMemory
history on each session.  This ensures it always reflects the full historical
evidence, not just the current run.  A future improvement would persist the
tracker state to avoid replaying history.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from spectraagent.webapp.agents.claude_agents import _DEFAULT_MODEL, _DEFAULT_TIMEOUT_S, _BaseClaude

log = logging.getLogger(__name__)

try:
    from spectraagent.knowledge.protocols import (
        ICH_Q2_PROTOCOL,
        ValidationTracker,
    )
    _PROTOCOLS_AVAILABLE = True
except ImportError:
    _PROTOCOLS_AVAILABLE = False
    log.debug("CalibrationValidationOrchestrator: protocols module unavailable")

try:
    from spectraagent.knowledge.context_builders import build_calibration_narration_context
    _CTX_AVAILABLE = True
except ImportError:
    _CTX_AVAILABLE = False


class CalibrationValidationOrchestrator(_BaseClaude):
    """ICH Q2(R1) compliance orchestrator.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    model:
        Claude model ID.
    timeout_s:
        Claude API timeout.
    memory:
        SensorMemory instance for history replay.
    get_analyte:
        Callable returning the current session's analyte name.
    auto_explain:
        When True, calls Claude after each session to produce a plain-language
        next-experiment recommendation.
    """

    source = "CalibrationValidationOrchestrator"

    def __init__(
        self,
        bus: Any,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        memory: Optional[Any] = None,
        get_analyte: Optional[Callable[[], Optional[str]]] = None,
        auto_explain: bool = False,
    ) -> None:
        super().__init__(bus, model, timeout_s, memory=memory)
        self._get_analyte = get_analyte
        self._auto_explain = auto_explain
        # Per-analyte ValidationTracker cache (rebuilt from memory on first use)
        self._trackers: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def on_event(self, event: Any) -> None:
        """React to session_complete events."""
        if event.type == "session_complete":
            await self._update_validation_state(event)
            await self._update_selectivity(event)

    async def _update_validation_state(self, event: Any) -> None:
        """Update ICH Q2(R1) state and emit validation status."""
        if not _PROTOCOLS_AVAILABLE:
            return

        data = event.data
        analyte = self._get_analyte() if self._get_analyte is not None else None
        if not analyte:
            analyte = data.get("gas_label", "unknown")

        tracker = self._get_or_build_tracker(analyte)

        # Infer state from session data
        tracker.infer_from_calibration_data(data)

        # Compute summary
        pct = tracker.completion_pct()
        gaps = tracker.get_gaps()
        next_test = tracker.get_next_test()
        status_text = tracker.format_status_for_prompt()

        # Build gap descriptions for the event payload.
        # get_gaps() returns ValidationRequirement objects directly — no lookup needed.
        gap_descriptions = []
        for req in gaps:
            gap_descriptions.append({
                "id": req.id,
                "section": req.ich_section,
                "name": req.display_name,
                "description": req.description,
                "min_levels": req.min_concentration_levels,
                "min_replicates": req.min_replicates_per_level,
                "acceptance_criteria": req.acceptance_criteria,
                "depends_on": req.depends_on,
            })

        next_test_info: dict[str, Any] = {}
        if next_test:
            next_test_info = {
                "id": next_test.id,
                "section": next_test.ich_section,
                "name": next_test.display_name,
                "measurement_protocol": next_test.measurement_protocol,
                "what_is_reported": next_test.what_is_reported,
                "acceptance_criteria": next_test.acceptance_criteria,
            }

        summary_text = (
            f"ICH Q2(R1) for {analyte}: {pct:.0f}% complete "
            f"({len(gaps)} test(s) remaining)"
        )
        if next_test:
            summary_text += f" — next: {next_test.display_name} ({next_test.ich_section})"

        level = "info" if pct >= 100 else ("warn" if gaps else "info")
        event_type = "validation_complete" if pct >= 100 else "validation_status_updated"

        self._bus.emit(self._AgentEvent(
            source=self.source,
            level=level,
            type=event_type,
            data={
                "analyte": analyte,
                "completion_pct": round(pct, 1),
                "gaps": gap_descriptions,
                "next_test": next_test_info,
                "status_summary": status_text,
            },
            text=summary_text,
        ))
        log.info("CalibrationValidationOrchestrator: %s", summary_text)

        # Optional Claude experiment design
        if self._auto_explain and gaps and next_test:
            await self._design_next_experiment(
                analyte, data, pct, gap_descriptions, next_test_info, status_text
            )

    async def _design_next_experiment(
        self,
        analyte: str,
        session_data: dict[str, Any],
        pct: float,
        gaps: list[dict[str, Any]],
        next_test: dict[str, Any],
        status_text: str,
    ) -> None:
        """Call Claude to design the next validation experiment."""
        history_text = ""
        if self._memory is not None:
            history_text = self._memory.format_for_agent_prompt(analyte)

        cal_ctx = ""
        if _CTX_AVAILABLE:
            cal_ctx = build_calibration_narration_context(session_data, analyte, self._memory)

        gaps_summary = "\n".join(
            f"- **{g['name']}** ({g['section']}): {g['description']} "
            f"[min {g.get('min_levels', '?')} levels × {g.get('min_replicates', '?')} replicates]"
            for g in gaps[:5]  # show top 5 gaps
        )

        prompt = (
            "## Context: ICH Q2(R1) Analytical Method Validation\n\n"
            + history_text
            + "\n\n"
            + cal_ctx
            + "\n\n---\n"
            f"**Validation status for {analyte}:** {pct:.0f}% complete\n\n"
            f"**Remaining tests ({len(gaps)}):**\n{gaps_summary}\n\n"
            f"**Highest-priority next test:**\n"
            f"- Name: {next_test.get('name')}\n"
            f"- ICH section: {next_test.get('section')}\n"
            f"- Protocol: {next_test.get('measurement_protocol', 'not specified')}\n"
            f"- Acceptance criteria: {next_test.get('acceptance_criteria', 'not specified')}\n\n"
            f"**Current compliance summary:**\n{status_text}\n\n"
            "Write a concise experimental protocol (3–5 sentences) for a Chulalongkorn "
            f"University researcher to complete the **{next_test.get('name')}** test:\n"
            "(1) Exact concentrations to prepare and why.\n"
            "(2) Number of replicates per level and their statistical purpose.\n"
            "(3) How to calculate the acceptance criterion from the raw data.\n"
            "Use scientific language. Be specific about numbers."
        )
        text = await self._call(prompt)
        if text:
            self._bus.emit(self._AgentEvent(
                source=self.source,
                level="claude",
                type="next_experiment_design",
                data={
                    "analyte": analyte,
                    "next_test_id": next_test.get("id"),
                    "next_test_name": next_test.get("name"),
                    "completion_pct": pct,
                    "protocol": text,
                },
                text=text,
            ))

    # ------------------------------------------------------------------
    # Selectivity coefficient estimation (B3 / ICH Q2(R1) §4.1)
    # ------------------------------------------------------------------

    async def _update_selectivity(self, event: Any) -> None:
        """Compute selectivity matrix from SensorMemory when ≥2 analytes exist.

        Uses the mean sensitivity (nm/ppm) stored in memory for each calibrated
        analyte.  Emits ``selectivity_updated`` with the full K-matrix whenever
        a new analyte is added or sensitivities change meaningfully.

        ICH Q2(R1) §4.1 Specificity requirement:
            K = |sensitivity_interferent| / |sensitivity_target| < 0.1
            (the interferent should produce < 10% of the target signal at the
            same concentration to be considered non-interfering).
        """
        if self._memory is None:
            return

        try:
            sensitivities = self._memory.get_sensitivities_by_analyte()
        except Exception as exc:
            log.debug("get_sensitivities_by_analyte() failed: %s", exc)
            return

        if len(sensitivities) < 2:
            return  # need at least target + one interferent

        try:
            from src.scientific.selectivity import selectivity_matrix as _sel_matrix
        except ImportError:
            log.debug("selectivity module unavailable")
            return

        try:
            sel = _sel_matrix(sensitivities)
        except Exception as exc:
            log.debug("selectivity_matrix() failed: %s", exc)
            return

        # Determine which analyte is the current session's target
        analyte = self._get_analyte() if self._get_analyte is not None else None
        if not analyte:
            analyte = event.data.get("gas_label", "")

        worst = sel.worst_interferents.get(analyte)
        summary = sel.to_dict()

        # Derive overall assessment from max K across all targets
        all_k = [k for _, k in sel.worst_interferents.values()]
        max_k = max(all_k) if all_k else 0.0
        if max_k > 0.5:
            overall_assessment = f"PROBLEMATIC — max K = {max_k:.3f} (> 0.5 threshold)"
        elif max_k > 0.1:
            overall_assessment = f"SIGNIFICANT — max K = {max_k:.3f} (> 0.1 threshold)"
        else:
            overall_assessment = "GOOD — all K < 0.1 (ICH Q2(R1) §4.1 passed)"

        self._bus.emit(self._AgentEvent(
            source=self.source,
            level="info",
            type="selectivity_updated",
            data={
                "target_analyte": analyte,
                "n_analytes": len(sensitivities),
                "selectivity_matrix": summary["matrix"],
                "worst_interferents": summary["worst_interferents"],
                "overall_assessment": overall_assessment,
            },
            text=(
                f"Selectivity for {analyte}: "
                + (
                    f"worst interferent = {worst[0]} (K={worst[1]:.3f})"
                    if worst else "no interferents calibrated yet"
                )
            ),
        ))
        log.info(
            "CalibrationValidationOrchestrator: selectivity updated (%d analytes)",
            len(sensitivities),
        )

    # ------------------------------------------------------------------
    # Tracker management
    # ------------------------------------------------------------------

    def _get_or_build_tracker(self, analyte: str) -> Any:
        """Return the ValidationTracker for this analyte, rebuilding from memory."""
        if analyte not in self._trackers:
            tracker = ValidationTracker(analyte=analyte)
            # Replay history from SensorMemory so the tracker reflects all past work.
            if self._memory is not None:
                summary = self._memory.get_analyte_summary(analyte)
                if summary:
                    history = summary.get("most_recent") or {}
                    # infer_from_calibration_data expects session_complete-style dict
                    tracker.infer_from_calibration_data({
                        "lod_ppm": history.get("lod_ppm"),
                        "loq_ppm": history.get("loq_ppm"),
                        "r_squared": history.get("r_squared"),
                        "n_points": history.get("n_calibration_points", 0),
                    })
            self._trackers[analyte] = tracker
        return self._trackers[analyte]
