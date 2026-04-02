"""
spectraagent.knowledge.context_builders
=========================================
Assembles rich, structured context for Claude agent prompts.

These functions are the bridge between the knowledge base (analytes,
failure_modes, protocols, sensor_memory) and the agent prompt strings.
They are called at agent invocation time — NOT at import time — so they
always reflect the most current sensor memory state.

Design principles
-----------------
1. **Data-first**: sensor_memory observed values take precedence over
   any hardcoded knowledge.  If memory is empty, agents get generic context
   and are told explicitly that no sensor history exists yet.

2. **Proportional depth**: the amount of injected context scales with
   available data.  An agent working with 10 sessions gets far richer
   context than one seeing the first measurement.

3. **Agent-aware**: each builder produces context shaped for its specific
   agent's reasoning task (anomaly explanation ≠ report writing ≠ calibration
   narration).

4. **Testable without disk**: builders accept SensorMemory as an argument so
   tests can pass in-memory instances without any file I/O.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from spectraagent.knowledge.analytes import format_analyte_chemistry_brief, lookup_analyte
from spectraagent.knowledge.failure_modes import (
    format_candidate_modes_for_prompt,
    match_failure_modes,
)

if TYPE_CHECKING:
    from spectraagent.knowledge.sensor_memory import SensorMemory


# ---------------------------------------------------------------------------
# Shared physics preamble
# ---------------------------------------------------------------------------

def build_sensor_physics_preamble(sensor_type: str = "optical") -> str:
    """Return a generic optical sensor physics preamble for all agents.

    Parameters
    ----------
    sensor_type : str
        Hint for the sensor modality: "lspr", "spr", "fluorescence",
        "raman", or "optical" (generic default).
    """
    generic = (
        "## Sensor physics background\n"
        "This platform operates a spectrometer-based optical chemical sensor. "
        "The primary analytical signal is **peak wavelength shift (Δλ)** — "
        "the change in the sensor's characteristic spectral peak position "
        "relative to a clean-air reference spectrum captured before analyte introduction.\n\n"
        "Key signal relationships:\n"
        "- **Δλ > 0 (redshift)**: effective refractive index at sensor surface increased "
        "(analyte binding raises local RI).\n"
        "- **Δλ < 0 (blueshift)**: effective RI decreased (analyte absorption compresses "
        "or deforms surface matrix, lowering effective RI, OR the sensor material has "
        "negative dn/d[analyte]).\n"
        "- **SNR** = peak_intensity / noise_floor — frames with SNR < 3 have unreliable "
        "peak wavelength estimates.\n"
        "- **Drift rate** (nm/min) reflects baseline instability — NOT analyte signal.\n\n"
        "The calibration model maps Δλ → concentration. The response is often nonlinear "
        "(Langmuir saturation) at higher concentrations — SpectraAgent automatically "
        "selects the best model (linear, Langmuir, Freundlich) using AIC.\n"
    )

    type_addenda: dict[str, str] = {
        "lspr": (
            "This sensor uses **LSPR (Localized Surface Plasmon Resonance)**: "
            "conduction electrons in Au (or Ag) nanostructures oscillate resonantly "
            "with incident light. The resonance wavelength is exquisitely sensitive "
            "to the local dielectric environment. A thin molecular coating or MIP "
            "(molecularly imprinted polymer) layer on the nanostructures selectively "
            "concentrates the target analyte near the plasmonic surface, amplifying "
            "the RI change signal by 10–1000× vs. bulk RI methods."
        ),
        "spr": (
            "This sensor uses **SPR (Surface Plasmon Resonance)**: evanescent "
            "plasmonic wave at a thin gold film interface. Mass loading on the "
            "sensor surface shifts the resonance angle/wavelength."
        ),
        "fluorescence": (
            "This sensor uses **fluorescence spectroscopy**: analyte binding changes "
            "the emission intensity or peak wavelength of a fluorescent probe. "
            "Signal is fluorescence intensity change (ΔI) or emission peak shift (Δλ_em)."
        ),
    }

    extra = type_addenda.get(sensor_type.lower(), "")
    return generic + ("\n" + extra if extra else "")


# ---------------------------------------------------------------------------
# Anomaly / drift explanation context
# ---------------------------------------------------------------------------

def build_anomaly_context(
    event_data: dict[str, Any],
    analyte_name: Optional[str],
    memory: Optional[SensorMemory],
    sensor_type: str = "optical",
) -> str:
    """Build comprehensive context for AnomalyExplainer agent.

    Assembles: sensor physics preamble + failure mode candidates ranked by
    the observed symptoms + sensor-specific history from SensorMemory +
    analyte chemistry properties.

    Parameters
    ----------
    event_data : dict
        Raw event data from the drift_warn or quality event.
    analyte_name : str or None
        Current target analyte (from session config).
    memory : SensorMemory or None
        Live sensor memory instance.  None → "no history" context.
    sensor_type : str
        Sensor modality hint for physics preamble.
    """
    sections: list[str] = []

    # 1. Physics preamble
    sections.append(build_sensor_physics_preamble(sensor_type))

    # 2. Failure mode matching
    drift_rate = event_data.get("drift_rate_nm_per_min")
    is_sudden = event_data.get("onset") == "sudden"
    snr_val = event_data.get("snr")
    snr_stable = snr_val is None or snr_val > 10.0

    candidates = match_failure_modes(
        drift_rate_nm_per_min=drift_rate,
        is_sudden=is_sudden,
        snr_stable=snr_stable,
        max_n=3,
    )
    sections.append(format_candidate_modes_for_prompt(candidates))

    # 3. Sensor memory context
    if memory is not None:
        sections.append(memory.format_for_agent_prompt(analyte=analyte_name))
        if analyte_name:
            trend_text = memory.get_calibration_trend_text(analyte_name)
            if trend_text:
                sections.append(f"**Calibration trend**: {trend_text}")

        # Recent failures — are we seeing a recurring pattern?
        recent = memory.get_recent_failures(5)
        if recent:
            recurring = [f for f in recent if f.get("event_type") in
                         {c[0] for c in candidates}]
            if recurring:
                sections.append(
                    f"\n**⚠ Recurrence alert**: This failure type has occurred "
                    f"{len(recurring)} time(s) in recent sessions — likely a "
                    f"systematic issue, not a one-off event."
                )
    else:
        sections.append(
            "## Sensor Memory\n"
            "No session history available yet for this sensor. "
            "Context is based on generic optical sensor physics only."
        )

    # 4. Analyte chemistry context
    if analyte_name:
        props = lookup_analyte(analyte_name)
        if props:
            sections.append(
                "## Analyte chemistry\n" + format_analyte_chemistry_brief(props)
            )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Calibration narration context
# ---------------------------------------------------------------------------

def build_calibration_narration_context(
    calibration_data: dict[str, Any],
    analyte_name: Optional[str],
    memory: Optional[SensorMemory],
) -> str:
    """Build context for ExperimentNarrator agent.

    Frames the calibration result against expected ranges from SensorMemory
    and ICH Q2(R1) requirements.
    """
    sections: list[str] = []

    # 1. Summarise observed calibration
    n = calibration_data.get("n_points", 0)
    r2 = calibration_data.get("r_squared")
    lod = calibration_data.get("lod_ppm")
    loq = calibration_data.get("loq_ppm")
    rmse = calibration_data.get("rmse_ppm")
    best_model = calibration_data.get("best_model", "unknown")
    aic = calibration_data.get("best_aic")

    lines = ["## Current calibration result\n"]
    lines.append(f"- Analyte: **{analyte_name or 'unknown'}**")
    lines.append(f"- Points: {n}")
    if r2 is not None:
        r2_status = "✅" if r2 >= 0.999 else "⚠" if r2 >= 0.99 else "❌"
        lines.append(f"- R²: {r2:.5f} {r2_status}")
    if lod is not None:
        lines.append(f"- LOD: {lod:.5f} ppm")
    if loq is not None:
        lines.append(f"- LOQ: {loq:.5f} ppm")
    if rmse is not None:
        lines.append(f"- RMSE: {rmse:.5f} ppm")
    lines.append(f"- Best model: {best_model} (AICc={aic})")
    sections.append("\n".join(lines))

    # 2. Historical comparison from SensorMemory
    if memory is not None and analyte_name:
        summary = memory.get_analyte_summary(analyte_name)
        if summary and summary.get("n_sessions", 0) > 0:
            hist_lod = summary.get("lod_ppm", {})
            hist_r2 = summary.get("r_squared", {})
            hist_sens = summary.get("sensitivity_nm_per_ppm", {})

            comparison = ["## Historical comparison (from sensor memory)\n"]
            if hist_lod.get("n", 0) >= 2:
                hist_mean = hist_lod["mean"]
                hist_std = hist_lod.get("std", 0)
                delta = ((lod or 0) - hist_mean) / hist_mean * 100 if hist_mean else 0
                direction = "worse" if delta > 10 else "better" if delta < -10 else "similar"
                comparison.append(
                    f"- LOD vs history: {lod:.5f} vs mean={hist_mean:.5f} ± {hist_std:.5f} ppm "
                    f"({direction}, {delta:+.1f}%) — trend: **{summary['trend']}**"
                )
            if hist_r2.get("n", 0) >= 2:
                comparison.append(
                    f"- R² vs history: {r2:.5f} vs mean={hist_r2['mean']:.5f}"
                )
            if hist_sens.get("n", 0) >= 2:
                comparison.append(
                    f"- Sensitivity vs history: mean={hist_sens['mean']:.3f} nm/ppm "
                    f"(this session: {calibration_data.get('sensitivity_nm_per_ppm', '?')})"
                )
            sections.append("\n".join(comparison))
        else:
            sections.append(
                "## Historical comparison\n"
                f"This is the first calibration session for {analyte_name} on this sensor. "
                "No historical baseline to compare against yet."
            )

    # 3. ICH Q2(R1) readiness assessment
    ich_lines = ["## ICH Q2(R1) compliance readiness\n"]
    issues = []
    if n < 5:
        issues.append(f"⬜ Linearity (§4.2) needs ≥5 points (have {n})")
    if r2 is not None and r2 < 0.999:
        issues.append(f"⚠ R²={r2:.5f} is below the 0.999 linearity threshold")
    if lod is None:
        issues.append("⬜ LOD (§4.6) not yet computed")
    if not issues:
        ich_lines.append("✅ This calibration dataset meets linearity and LOD/LOQ requirements.")
        ich_lines.append(
            "Next steps: run specificity (§4.1) and repeatability (§4.4.1) tests."
        )
    else:
        ich_lines.append("Remaining requirements:")
        ich_lines.extend(issues)
    sections.append("\n".join(ich_lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Report writer context
# ---------------------------------------------------------------------------

def build_report_context(
    session_context: dict[str, Any],
    analyte_name: Optional[str],
    memory: Optional[SensorMemory],
) -> str:
    """Build context for ReportWriter agent (Methods + Results).

    Enriches the raw session context with: historical performance baseline
    (so the agent can note whether this session's LOD is better or worse
    than typical), analyte chemistry, and ICH Q2(R1) status.
    """
    sections: list[str] = []
    sections.append("## Session data for report\n```json")
    import json
    sections.append(json.dumps(session_context, indent=2, default=str))
    sections.append("```")

    if memory is not None and analyte_name:
        health = memory.get_sensor_health_summary()
        sections.append(
            f"## Sensor context\n"
            f"Sensor has completed {health['total_sessions']} session(s) total. "
            f"This session's results should be compared with historical performance "
            f"where available.\n"
            + memory.format_for_agent_prompt(analyte=analyte_name)
        )

    if analyte_name:
        props = lookup_analyte(analyte_name)
        if props:
            sections.append(
                "## Analyte properties (for methods section)\n"
                + format_analyte_chemistry_brief(props)
            )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Hardware diagnostics context
# ---------------------------------------------------------------------------

def build_hardware_diagnostics_context(error_data: dict[str, Any]) -> str:
    """Build context for DiagnosticsAgent (hardware error events).

    Provides structured hardware troubleshooting context for common
    spectrometer error codes and patterns.
    """
    code = str(error_data.get("error_code", "unknown"))

    # Known error codes for common hardware — expandable
    known_codes: dict[str, str] = {
        "-1073807339": (
            "VISA error VI_ERROR_TMO: instrument timeout. Most common cause: "
            "device was not properly closed in a previous session (stale USB state). "
            "Fix: (1) unplug and replug USB; (2) wait 10 s; (3) restart SpectraAgent. "
            "If recurring: add warm-up scan to driver init to drain stale state."
        ),
        "-1073807343": (
            "USB device found but not powered or not ready. "
            "Device is connected via VISA but hardware is off or in reset state. "
            "Fix: (1) check power LED on instrument; (2) unplug/replug USB; "
            "(3) wait for device ready indicator."
        ),
        "-1074001152": (
            "TLCCS_ERROR_SCAN_PENDING: previous scan not yet complete. "
            "Fix: increase wait time after startScan (add 50–100 ms buffer). "
            "Check integration time setting — it may have been changed."
        ),
    }

    explanation = known_codes.get(code, "")
    lines = ["## Hardware error context\n"]
    if explanation:
        lines.append(f"**Known error code {code}**:\n{explanation}")
    else:
        lines.append(
            f"**Error code {code}** is not in the known-code database. "
            f"General troubleshooting: (1) check USB/GPIB connection; "
            f"(2) verify instrument is powered; (3) consult instrument manual."
        )

    lines.append(
        "\n**General diagnostic checklist**:\n"
        "1. Is the instrument powered on and the status LED green?\n"
        "2. Is the USB cable firmly connected at both ends?\n"
        "3. Was the previous session properly closed (not force-killed)?\n"
        "4. Are there any other software instances accessing this instrument?\n"
        "5. Is the driver (e.g. ThorLabs TLCCS) installed and up to date?"
    )
    return "\n".join(lines)
