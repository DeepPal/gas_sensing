"""
spectraagent.knowledge.failure_modes
======================================
Generic optical sensor failure mode taxonomy.

All entries describe failure patterns common to spectrometer-based chemical
sensors regardless of sensor type (LSPR, SPR, fluorescence, Raman, etc.).
Sensor-specific behaviour is learned from SensorMemory at runtime.

Design rationale
----------------
Each failure mode encodes THREE things agents need:
1. **Detection signature** — what the raw data looks like when this happens.
2. **Physical mechanism** — WHY it happens (causal chain, not just correlation).
3. **Resolution** — what to do, in priority order.

This structured representation lets the agent provide expert-level diagnosis
rather than generic "check your instrument" advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FailureMode:
    """One failure mode with detection rules and domain-expert knowledge.

    Attributes
    ----------
    id : str
        Short machine-readable identifier, e.g. "thermal_drift".
    display_name : str
        Human-readable name for display in UI and reports.
    category : str
        Category: "drift", "signal_quality", "calibration", "hardware".
    drift_signature : str or None
        Characterises the shape of peak-wavelength vs time:
        "monotonic_up", "monotonic_down", "oscillating", "step_change", "random".
    drift_rate_nm_per_min_range : tuple[float, float] or None
        Typical absolute drift rate range for this mode.
    snr_behaviour : str
        How SNR behaves: "stable", "degrading", "low_constant", "variable".
    onset : str
        How the failure appears: "gradual", "sudden", "periodic".
    reversible : bool
        Whether the sensor fully recovers after resolution steps.
    physical_mechanism : str
        Causal explanation of the underlying physics/chemistry.
        This is injected verbatim into agent prompts for expert reasoning.
    detection_rules : list[str]
        Programmatic detection hints (for future automated classification).
    corrective_actions : list[str]
        Ordered resolution steps (most likely first).
    prevention : str
        Best-practice advice to avoid the failure.
    typical_time_scale : str
        How quickly the failure manifests: "seconds", "minutes", "hours", "days".
    """

    id: str
    display_name: str
    category: str
    physical_mechanism: str
    corrective_actions: list[str]
    prevention: str
    drift_signature: Optional[str] = None
    drift_rate_nm_per_min_range: Optional[tuple[float, float]] = None
    snr_behaviour: str = "stable"
    onset: str = "gradual"
    reversible: bool = True
    detection_rules: list[str] = field(default_factory=list)
    typical_time_scale: str = "minutes"


# ---------------------------------------------------------------------------
# Failure taxonomy — generic optical sensor failures
# ---------------------------------------------------------------------------

FAILURE_TAXONOMY: dict[str, FailureMode] = {

    # --- Drift failures -------------------------------------------------------

    "thermal_drift": FailureMode(
        id="thermal_drift",
        display_name="Thermal drift",
        category="drift",
        drift_signature="monotonic_up",
        drift_rate_nm_per_min_range=(0.01, 0.30),
        snr_behaviour="stable",
        onset="gradual",
        reversible=True,
        physical_mechanism=(
            "The LSPR/SPR peak wavelength shifts with temperature because: "
            "(1) the refractive index of the surrounding medium is temperature-dependent "
            "(dn/dT is typically negative for organic solvents and polymers, meaning "
            "RI decreases as T increases, causing a blueshift); "
            "(2) the sensor surface material (Au, Ag nanostructures) undergoes "
            "thermal expansion, altering the plasmon resonance condition; "
            "(3) polymer sensor matrices (MIP, SAM, hydrogel) swell or contract "
            "with temperature, modifying the local dielectric environment. "
            "The combined effect is typically a monotonic drift of 0.05–0.2 nm/°C "
            "for Au-based optical sensors, but the exact sensitivity is sensor-specific "
            "and is best characterised from the sensor's own history in SensorMemory. "
            "Thermal drift is the MOST COMMON cause of slow, gradual peak drift."
        ),
        detection_rules=[
            "drift_rate between 0.01 and 0.30 nm/min",
            "SNR remains stable (not degrading)",
            "onset within first 5–15 minutes of session",
        ],
        corrective_actions=[
            "Wait for thermal equilibration: stop measurement and allow sensor+instrument "
            "to equilibrate for 15–20 min in measurement environment before re-capturing reference.",
            "Re-capture reference spectrum after equilibration.",
            "Check and stabilise room temperature (target: ±0.5°C).",
            "If repeated across sessions: add temperature monitoring to the setup "
            "and apply temperature compensation using the sensor's historical dn/dT.",
        ],
        prevention=(
            "Always allow 15–20 min warm-up before capturing reference spectrum. "
            "Avoid experiments during HVAC cycling or after opening lab doors. "
            "Track ambient temperature alongside sensor data."
        ),
        typical_time_scale="minutes",
    ),

    "surface_fouling": FailureMode(
        id="surface_fouling",
        display_name="Surface fouling / cavity saturation",
        category="drift",
        drift_signature="monotonic_down",   # sensitivity loss → apparent baseline shift
        drift_rate_nm_per_min_range=(0.0, 0.05),
        snr_behaviour="stable",
        onset="gradual",
        reversible=False,
        physical_mechanism=(
            "Irreversible accumulation of analyte molecules, interferents, or "
            "environmental contaminants on the active sensor surface or within "
            "polymer/MIP cavities. Unlike reversible analyte binding, fouling "
            "represents non-specific adsorption that: "
            "(1) permanently shifts the reference baseline to a new equilibrium; "
            "(2) progressively reduces sensitivity as active sites are blocked; "
            "(3) degrades calibration R² and increases LOD over successive sessions. "
            "Common causes: prolonged exposure to high analyte concentrations, "
            "exposure to interferent analytes not present during imprinting (for MIP), "
            "dust/aerosol deposition, or oxidation of the sensor surface. "
            "In SensorMemory, fouling appears as a DEGRADING trend in LOD and sensitivity "
            "over multiple sessions, and a systematic shift in the reference peak position."
        ),
        detection_rules=[
            "slow monotonic drift over >30 min with no temperature correlation",
            "calibration sensitivity decreasing across sessions (from SensorMemory)",
            "LOD showing degrading trend in SensorMemory",
        ],
        corrective_actions=[
            "Check SensorMemory: confirm LOD has degraded over multiple sessions.",
            "Attempt sensor regeneration per manufacturer protocol "
            "(typically: solvent rinse, UV treatment, or thermal desorption).",
            "Re-run full calibration after regeneration to verify recovery.",
            "If sensitivity does not recover to within 20% of original: "
            "sensor likely requires re-fabrication or replacement.",
        ],
        prevention=(
            "Avoid measuring at concentrations significantly above LOQ (near MIP saturation). "
            "Purge sensor with clean carrier gas between concentration steps. "
            "Track sensor age and calibration history in SensorMemory."
        ),
        typical_time_scale="hours to days",
    ),

    "stale_reference": FailureMode(
        id="stale_reference",
        display_name="Stale or invalid reference spectrum",
        category="drift",
        drift_signature="step_change",
        drift_rate_nm_per_min_range=None,
        snr_behaviour="stable",
        onset="sudden",
        reversible=True,
        physical_mechanism=(
            "The differential signal (Δλ or ΔI) is computed relative to a stored "
            "reference spectrum captured under specific conditions. If the reference "
            "was captured in a different state — different temperature, different "
            "analyte background, after partial surface fouling, or with stale "
            "optical alignment — all subsequent measurements carry a systematic error. "
            "This appears as a STEP CHANGE in the apparent Δλ baseline (the 'zero' "
            "suddenly jumps) rather than a gradual drift. "
            "Typical triggers: reference captured before thermal equilibration, "
            "reference captured while analyte was still present in the measurement cell, "
            "or reference loaded from a previous session when sensor state has changed."
        ),
        detection_rules=[
            "sudden step change in apparent Δλ baseline (>1 nm in <1 min)",
            "SNR stable before and after step",
            "step coincides with a reference capture or session start",
        ],
        corrective_actions=[
            "Purge measurement cell with clean carrier gas for ≥5 min.",
            "Wait for thermal equilibration (15 min).",
            "Re-capture a fresh reference spectrum.",
            "Discard measurements taken after the reference became invalid.",
        ],
        prevention=(
            "Always capture reference in the clean measurement environment immediately "
            "before analyte introduction. Never reuse references across sessions "
            "without verifying baseline stability."
        ),
        typical_time_scale="immediate",
    ),

    "humidity_interference": FailureMode(
        id="humidity_interference",
        display_name="Humidity / water vapour interference",
        category="drift",
        drift_signature="oscillating",
        drift_rate_nm_per_min_range=(0.01, 0.15),
        snr_behaviour="stable",
        onset="gradual",
        reversible=True,
        physical_mechanism=(
            "Water vapour (RH 30–90%) causes peak wavelength shifts in most optical "
            "chemical sensors because: "
            "(1) water has a high refractive index (n_liquid=1.333) and even vapour "
            "contributes measurably to local RI near the sensor surface; "
            "(2) polymer matrices (hydrogel, MIP with polar functional groups) absorb "
            "water, causing swelling that modifies the plasmonic/optical response; "
            "(3) water can compete for analyte binding sites in MIP, suppressing "
            "the analyte signal. "
            "The interference is reversible but creates oscillating baseline drift "
            "correlated with ambient humidity changes (breathing, HVAC, door openings). "
            "This is often overlooked because it is not associated with analyte introduction."
        ),
        detection_rules=[
            "oscillating drift correlated with humidity sensor readings",
            "drift present even at zero analyte concentration",
            "more pronounced in polymer-based sensors than bare metal surfaces",
        ],
        corrective_actions=[
            "Measure and record relative humidity alongside sensor data.",
            "Dry the carrier gas: pass through a molecular sieve or Drierite column.",
            "Control ambient humidity: keep measurement cell sealed during experiments.",
            "Apply humidity correction using a second reference channel if available.",
        ],
        prevention=(
            "Always dry carrier gas. Seal measurement cell. "
            "Record ambient humidity for post-hoc correction."
        ),
        typical_time_scale="minutes",
    ),

    # --- Signal quality failures -----------------------------------------------

    "optical_saturation": FailureMode(
        id="optical_saturation",
        display_name="Detector saturation",
        category="signal_quality",
        drift_signature=None,
        drift_rate_nm_per_min_range=None,
        snr_behaviour="variable",
        onset="sudden",
        reversible=True,
        physical_mechanism=(
            "CCD/CMOS detector pixels clip at their full-well capacity (typically "
            "60 000–65 535 counts for 16-bit detectors). When saturation occurs: "
            "(1) the spectral peak becomes flat-topped, shifting the apparent peak "
            "position (Lorentzian fit fails on a clipped peak); "
            "(2) the differential signal (Δλ) is unreliable; "
            "(3) charge bleeding from saturated pixels can elevate adjacent pixels. "
            "Causes: integration time too long, light source too bright, "
            "or sample reflectance/fluorescence unexpectedly high."
        ),
        detection_rules=[
            "max(intensities) > saturation_threshold (default 60 000 counts)",
        ],
        corrective_actions=[
            "Immediately reduce integration time (e.g. halve it).",
            "If light source power is adjustable: reduce it.",
            "Discard the saturated frame — do not attempt to fit the peak.",
            "Re-run the measurement with corrected settings.",
        ],
        prevention=(
            "Before each session: verify that the peak intensity at maximum analyte "
            "concentration is < 80% of saturation threshold. "
            "Use the auto-range feature if available."
        ),
        typical_time_scale="immediate",
    ),

    "low_snr": FailureMode(
        id="low_snr",
        display_name="Low signal-to-noise ratio",
        category="signal_quality",
        drift_signature=None,
        drift_rate_nm_per_min_range=None,
        snr_behaviour="low_constant",
        onset="gradual",
        reversible=True,
        physical_mechanism=(
            "SNR < 3 means the spectral peak is comparable in magnitude to the noise "
            "floor, making peak detection unreliable. Common causes: "
            "(1) integration time too short — shot noise dominates; "
            "(2) light source intensity low — bulb aging, misalignment, dirty optics; "
            "(3) sensor response degraded — surface contamination or oxidation; "
            "(4) detector dark current elevated — instrument temperature too high; "
            "(5) ambient light leak — cell not sealed. "
            "Low SNR frames are processed with a warning (not hard-blocked) but "
            "peak wavelength estimates carry large uncertainty and should not be "
            "used for calibration or quantitative analysis."
        ),
        detection_rules=[
            "SNR = max(intensities) / (mean(noise) + std(noise)) < 3.0",
        ],
        corrective_actions=[
            "Increase integration time (double it as first step).",
            "Clean optical fibre connectors and sensor surface.",
            "Check light source output — replace if aged.",
            "Seal measurement cell from ambient light.",
            "If multiple frames with low SNR: pause acquisition and inspect hardware.",
        ],
        prevention=(
            "Check SNR at session start with clean carrier gas. "
            "Target SNR > 50 for calibration-quality measurements."
        ),
        typical_time_scale="immediate to hours",
    ),

    "light_source_instability": FailureMode(
        id="light_source_instability",
        display_name="Light source instability",
        category="hardware",
        drift_signature="random",
        drift_rate_nm_per_min_range=None,
        snr_behaviour="variable",
        onset="gradual",
        reversible=True,
        physical_mechanism=(
            "Broadband or laser light sources exhibit intensity fluctuations due to: "
            "(1) thermal cycling of filaments or laser diodes (warm-up drift, "
            "typically 5–15 min for halogen/tungsten sources); "
            "(2) power supply noise (60/50 Hz ripple, transients); "
            "(3) aging — tungsten filaments and LED emitters lose output over time; "
            "(4) mode hopping in laser sources (discrete jumps). "
            "Intensity variation appears as peak intensity (ΔI) fluctuation but "
            "typically does NOT shift peak wavelength (Δλ) unless the source has "
            "spectrally-resolved instability. Use Δλ-based sensing to make the "
            "measurement immune to intensity drift."
        ),
        detection_rules=[
            "peak intensity (ΔI) shows high variance while SNR is borderline",
            "drift in ΔI but stable Δλ → light source instability confirmed",
        ],
        corrective_actions=[
            "Allow full warm-up: tungsten/halogen need 15–30 min, LEDs 5–10 min.",
            "If instability persists after warm-up: check power supply quality.",
            "Switch to Δλ-based calibration (peak wavelength shift) which is "
            "immune to intensity fluctuations — this is already the default mode.",
            "Replace light source if output has decayed below usable level.",
        ],
        prevention=(
            "Always allow full warm-up before reference capture. "
            "Use Δλ rather than ΔI as the primary analytical signal."
        ),
        typical_time_scale="minutes to hours",
    ),

    # --- Calibration failures --------------------------------------------------

    "calibration_nonlinearity": FailureMode(
        id="calibration_nonlinearity",
        display_name="Unexpected calibration nonlinearity",
        category="calibration",
        drift_signature=None,
        drift_rate_nm_per_min_range=None,
        snr_behaviour="stable",
        onset="gradual",
        reversible=True,
        physical_mechanism=(
            "Calibration curves that are expected to be linear (Henry's law regime, "
            "low concentration) but show nonlinearity may indicate: "
            "(1) concentration range extends beyond the linear dynamic range — "
            "analyte binding approaches saturation (switch to Langmuir model); "
            "(2) sensor surface is not fully regenerated — residual bound analyte "
            "compresses the effective linear range; "
            "(3) measurement cell has dead volumes causing concentration buildup "
            "(hysteresis between increasing/decreasing steps); "
            "(4) matrix effects — carrier gas composition changing between points. "
            "Mandel's F-test (built into SpectraAgent) detects this automatically "
            "and triggers a model switch from linear to Langmuir."
        ),
        detection_rules=[
            "Mandel's F-test p < 0.05 (implemented in src.scientific.lod)",
            "residuals show systematic curvature",
            "R² < 0.99 in expected linear range",
        ],
        corrective_actions=[
            "Check AIC: if Langmuir AIC < Linear AIC, switch to Langmuir model.",
            "Verify concentration points span only the linear dynamic range.",
            "Purge and regenerate sensor before repeating calibration.",
            "Check for hysteresis: measure increasing then decreasing concentrations.",
        ],
        prevention=(
            "Begin calibration with the highest expected linear range concentration "
            "to verify linearity before adding more points."
        ),
        typical_time_scale="per session",
    ),

    "cross_sensitivity": FailureMode(
        id="cross_sensitivity",
        display_name="Interferent / cross-sensitivity response",
        category="calibration",
        drift_signature="step_change",
        drift_rate_nm_per_min_range=None,
        snr_behaviour="stable",
        onset="sudden",
        reversible=True,
        physical_mechanism=(
            "The sensor responds to a molecule other than the target analyte, "
            "producing false positive signals or calibration curve offsets. "
            "Mechanisms: (1) structurally similar molecules fitting into sensor "
            "binding sites (MIP cavity cross-reactivity); "
            "(2) bulk refractive index change from a non-selective interferent; "
            "(3) competitive binding displacing the target analyte. "
            "Cross-sensitivity is quantified by the selectivity coefficient K_{B/A} "
            "(ratio of apparent analyte concentration from interferent B vs target A). "
            "K < 0.1 is acceptable for most applications. "
            "This is typically discovered during validation (ICH Q2(R1) §4.1 specificity) "
            "and should be characterised before deploying the sensor for quantitative use."
        ),
        detection_rules=[
            "signal present with known-zero target analyte",
            "step change in signal when interferent gas is introduced",
        ],
        corrective_actions=[
            "Identify the interferent by systematic exclusion testing.",
            "Measure cross-sensitivity coefficient K_{B/A} using the selectivity test.",
            "Apply mathematical correction if K is small (<0.3) and interferent "
            "concentration is known.",
            "If K > 0.5 and interferent cannot be controlled: report as a limitation.",
        ],
        prevention=(
            "Run the ICH Q2(R1) §4.1 specificity test with all expected "
            "environmental interferents before deploying for quantitative analysis."
        ),
        typical_time_scale="per experiment",
    ),
}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def match_failure_modes(
    drift_rate_nm_per_min: Optional[float] = None,
    is_sudden: bool = False,
    snr_stable: bool = True,
    max_n: int = 3,
) -> list[tuple[str, FailureMode]]:
    """Return ranked candidate failure modes given observed symptoms.

    Parameters
    ----------
    drift_rate_nm_per_min:
        Absolute drift rate observed.  None if drift is not the primary symptom.
    is_sudden:
        True if the anomaly appeared abruptly (step change), not gradually.
    snr_stable:
        True if SNR is within normal range, False if SNR is low/degraded.
    max_n:
        Maximum number of candidates to return.

    Returns
    -------
    List of (mode_id, FailureMode) tuples, best match first.
    """
    scored: list[tuple[float, str, FailureMode]] = []

    for mid, mode in FAILURE_TAXONOMY.items():
        score = 0.0

        # Match onset
        if is_sudden and mode.onset == "sudden":
            score += 2.0
        elif not is_sudden and mode.onset == "gradual":
            score += 1.0

        # Match SNR behaviour
        if not snr_stable and mode.snr_behaviour in ("degrading", "low_constant"):
            score += 2.0
        elif snr_stable and mode.snr_behaviour == "stable":
            score += 1.0

        # Match drift rate
        if drift_rate_nm_per_min is not None and mode.drift_rate_nm_per_min_range is not None:
            lo, hi = mode.drift_rate_nm_per_min_range
            if lo <= abs(drift_rate_nm_per_min) <= hi:
                score += 3.0
            elif abs(drift_rate_nm_per_min) < lo * 0.5 or abs(drift_rate_nm_per_min) > hi * 2:
                score -= 1.0

        scored.append((score, mid, mode))

    scored.sort(key=lambda t: -t[0])
    return [(mid, mode) for _, mid, mode in scored[:max_n] if _ >= 0]


def format_failure_mode_for_prompt(mode: FailureMode) -> str:
    """Format one FailureMode as an agent-prompt-ready block."""
    lines = [
        f"### {mode.display_name} (id: `{mode.id}`)",
        f"**Category**: {mode.category} | "
        f"**Onset**: {mode.onset} | "
        f"**Reversible**: {'yes' if mode.reversible else 'no'}",
        "",
        f"**Physical mechanism**: {mode.physical_mechanism}",
        "",
        "**Corrective actions (in order)**:",
    ]
    for i, action in enumerate(mode.corrective_actions, 1):
        lines.append(f"{i}. {action}")
    lines += ["", f"**Prevention**: {mode.prevention}"]
    return "\n".join(lines)


def format_candidate_modes_for_prompt(
    candidates: list[tuple[str, FailureMode]],
) -> str:
    """Format a ranked list of candidate modes for agent injection."""
    if not candidates:
        return "No closely matching failure modes found — this may be an unusual failure pattern."
    blocks = ["## Most likely failure modes (ranked):\n"]
    for rank, (mid, mode) in enumerate(candidates, 1):
        blocks.append(f"**Rank {rank}**: {format_failure_mode_for_prompt(mode)}\n")
    return "\n".join(blocks)
