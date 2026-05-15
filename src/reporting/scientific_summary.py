"""Deterministic researcher-facing session summaries.

These summaries are designed to be useful even when LLM-backed report generation
is unavailable. They convert session metrics into a publication-oriented markdown
report with explicit readiness checks and next experimental actions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

_REPORT_TITLE = "Deterministic Scientific Summary"


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fmt(value: Any, *, digits: int = 3, suffix: str = "") -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}g}{suffix}"


def _fmt_iso(value: Any) -> str:
    return str(value) if value else "n/a"


def session_analysis_to_dict(analysis: Any) -> dict[str, Any]:
    allan = getattr(analysis, "allan_deviation", None)
    allan_summary: dict[str, Any] | None = None
    if allan is not None:
        allan_summary = {
            "tau_opt_s": getattr(allan, "tau_opt_s", None),
            "sigma_min": getattr(allan, "sigma_min", None),
            "noise_type": getattr(allan, "noise_type", None),
            "drift_onset_tau_s": getattr(allan, "drift_onset_tau_s", None),
        }

    return {
        "frame_count": getattr(analysis, "frame_count", 0),
        "calibration_r2": getattr(analysis, "calibration_r2", None),
        "calibration_rmse_ppm": getattr(analysis, "calibration_rmse_ppm", None),
        "calibration_n_points": getattr(analysis, "calibration_n_points", 0),
        "lob_ppm": getattr(analysis, "lob_ppm", None),
        "lod_ppm": getattr(analysis, "lod_ppm", None),
        "lod_ci_lower": getattr(analysis, "lod_ci_lower", None),
        "lod_ci_upper": getattr(analysis, "lod_ci_upper", None),
        "loq_ppm": getattr(analysis, "loq_ppm", None),
        "loq_ci_lower": getattr(analysis, "loq_ci_lower", None),
        "loq_ci_upper": getattr(analysis, "loq_ci_upper", None),
        "lod_used_blanks": getattr(analysis, "lod_used_blanks", False),
        "mean_concentration_ppm": getattr(analysis, "mean_concentration_ppm", None),
        "std_concentration_ppm": getattr(analysis, "std_concentration_ppm", None),
        "mean_ci_width_ppm": getattr(analysis, "mean_ci_width_ppm", None),
        "mean_snr": getattr(analysis, "mean_snr", None),
        "drift_rate_nm_per_frame": getattr(analysis, "drift_rate_nm_per_frame", None),
        "total_drift_nm": getattr(analysis, "total_drift_nm", None),
        "lol_ppm": getattr(analysis, "lol_ppm", None),
        "response_time_t90_seconds": getattr(analysis, "response_time_t90_seconds", None),
        "response_time_t10_seconds": getattr(analysis, "response_time_t10_seconds", None),
        "tau_63_s": getattr(analysis, "tau_63_s", None),
        "tau_95_s": getattr(analysis, "tau_95_s", None),
        "k_on_per_s": getattr(analysis, "k_on_per_s", None),
        "kinetics_delta_lambda_eq_nm": getattr(analysis, "kinetics_delta_lambda_eq_nm", None),
        "kinetics_fit_r2": getattr(analysis, "kinetics_fit_r2", None),
        "interval_coverage": getattr(analysis, "interval_coverage", None),
        "summary_text": getattr(analysis, "summary_text", ""),
        "audit": getattr(analysis, "audit", {}),
        "allan_deviation": allan_summary,
    }


def build_deterministic_scientific_report(context: dict[str, Any]) -> str:
    analysis = context.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}

    session_id = context.get("session_id", "unknown")
    analyte = context.get("gas_label") or context.get("analyte") or "unknown"
    hardware = context.get("hardware", "unknown")
    started_at = _fmt_iso(context.get("started_at"))
    stopped_at = _fmt_iso(context.get("stopped_at"))
    target_conc = context.get("target_concentration")
    temperature_c = context.get("temperature_c")
    humidity_pct = context.get("humidity_pct")

    calibration_n = int(analysis.get("calibration_n_points") or 0)
    r2 = _as_float(analysis.get("calibration_r2"))
    lod = _as_float(analysis.get("lod_ppm"))
    loq = _as_float(analysis.get("loq_ppm"))
    lol = _as_float(analysis.get("lol_ppm"))
    snr = _as_float(analysis.get("mean_snr"))
    tau_63 = _as_float(analysis.get("tau_63_s"))
    kinetics_r2 = _as_float(analysis.get("kinetics_fit_r2"))
    drift = _as_float(analysis.get("drift_rate_nm_per_frame"))
    used_blanks = bool(analysis.get("lod_used_blanks"))
    interval_coverage = _as_float(analysis.get("interval_coverage"))

    readiness_checks: list[tuple[str, bool | None, str]] = [
        (
            "Calibration density",
            calibration_n >= 5 if calibration_n else False,
            f"{calibration_n} point(s); target >= 5 for defensible linearity.",
        ),
        (
            "Calibration fit quality",
            (r2 is not None and r2 >= 0.99),
            f"R^2 = {_fmt(r2, digits=4)}; target >= 0.99.",
        ),
        (
            "Detection limits reported",
            (lod is not None and loq is not None),
            f"LOD = {_fmt(lod, suffix=' ppm')}, LOQ = {_fmt(loq, suffix=' ppm')}.",
        ),
        (
            "Blank-backed LOD",
            bool(used_blanks),
            "Uses measured blank replicates." if used_blanks else "Currently based on residual noise rather than dedicated blank replicates.",
        ),
        (
            "Signal quality",
            (snr is not None and snr >= 3.0),
            f"Mean SNR = {_fmt(snr)}; target >= 3.",
        ),
        (
            "Environment metadata",
            (temperature_c is not None and humidity_pct is not None),
            f"Temperature = {temperature_c if temperature_c is not None else 'n/a'} C, humidity = {humidity_pct if humidity_pct is not None else 'n/a'} %.",
        ),
        (
            "Kinetic characterization",
            (tau_63 is not None and (kinetics_r2 is None or kinetics_r2 >= 0.90)),
            f"tau_63 = {_fmt(tau_63, suffix=' s')}, kinetics fit R^2 = {_fmt(kinetics_r2, digits=3)}.",
        ),
        (
            "Linearity ceiling reported",
            lol is not None,
            f"LOL = {_fmt(lol, suffix=' ppm')}.",
        ),
    ]

    recommendations: list[str] = []
    if calibration_n < 5:
        recommendations.append("Add more calibration concentrations or replicates before claiming full linearity.")
    if r2 is not None and r2 < 0.99:
        recommendations.append("Inspect residuals and consider narrowing the concentration range or using a nonlinear model.")
    if not used_blanks:
        recommendations.append("Run dedicated blank replicates so LOB/LOD are anchored to measured blank noise.")
    if snr is not None and snr < 3.0:
        recommendations.append("Increase integration time, replicate averaging, or optical alignment to raise SNR above 3.")
    if temperature_c is None or humidity_pct is None:
        recommendations.append("Record temperature and humidity for every run to defend cross-session comparisons.")
    if tau_63 is None:
        recommendations.append("Capture a step-response transient to report tau_63 / tau_95 kinetics.")
    if interval_coverage is not None and interval_coverage < 0.9:
        recommendations.append("Recalibrate uncertainty intervals; current coverage appears weaker than expected.")
    if not recommendations:
        recommendations.append("Dataset is analytically mature; next step is external replication or interferent testing.")

    findings: list[str] = []
    if lod is not None:
        lod_text = f"LOD {_fmt(lod, suffix=' ppm')}"
        if _as_float(analysis.get("lod_ci_lower")) is not None and _as_float(analysis.get("lod_ci_upper")) is not None:
            lod_text += (
                f" (95% CI {_fmt(analysis.get('lod_ci_lower'), suffix=' ppm')} to "
                f"{_fmt(analysis.get('lod_ci_upper'), suffix=' ppm')})"
            )
        findings.append(lod_text)
    if loq is not None:
        findings.append(f"LOQ {_fmt(loq, suffix=' ppm')}")
    if r2 is not None:
        findings.append(f"calibration R^2 {_fmt(r2, digits=4)}")
    if snr is not None:
        findings.append(f"mean SNR {_fmt(snr)}")
    if drift is not None:
        findings.append(f"drift {_fmt(drift, digits=3, suffix=' nm/frame')}")
    if tau_63 is not None:
        findings.append(f"tau_63 {_fmt(tau_63, suffix=' s')}")

    lines = [
        f"# {_REPORT_TITLE}",
        "",
        "This report was generated deterministically from recorded session metadata and computed analysis outputs. It is intended to remain available even when LLM-backed reporting is offline.",
        "",
        "## Session overview",
        "",
        f"- Session ID: {session_id}",
        f"- Analyte: {analyte}",
        f"- Hardware: {hardware}",
        f"- Started: {started_at}",
        f"- Stopped: {stopped_at}",
        f"- Target concentration: {target_conc if target_conc is not None else 'n/a'} ppm",
        f"- Frames acquired: {analysis.get('frame_count', context.get('frame_count', 'n/a'))}",
        "",
        "## Key quantitative findings",
        "",
    ]
    if findings:
        for finding in findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- Quantitative analysis was not available in the report context.")

    lines.extend([
        "",
        "## Publication-readiness checks",
        "",
    ])
    for label, passed, detail in readiness_checks:
        status = "PASS" if passed else "ACTION NEEDED"
        lines.append(f"- {label}: {status}. {detail}")

    lines.extend([
        "",
        "## Interpretation",
        "",
        (analysis.get("summary_text") or "No narrative summary was recorded by the analysis pipeline."),
        "",
        "## Recommended next actions",
        "",
    ])
    for rec in recommendations:
        lines.append(f"- {rec}")

    audit = analysis.get("audit") or {}
    if audit:
        lines.extend([
            "",
            "## Audit trail",
            "",
            f"- Method: {audit.get('method', 'n/a')}",
            f"- LOD formula: {audit.get('lod_formula', 'n/a')}",
            f"- Sigma source: {audit.get('sigma_source', 'n/a')}",
            f"- Bootstrap resamples: {audit.get('n_bootstrap', 'n/a')}",
            f"- Framework version: {audit.get('framework_version', 'n/a')}",
        ])

    allan = analysis.get("allan_deviation") or {}
    if allan:
        lines.extend([
            "",
            "## Noise-floor characterization",
            "",
            f"- tau_opt: {_fmt(allan.get('tau_opt_s'), suffix=' s')}",
            f"- sigma_min: {_fmt(allan.get('sigma_min'))}",
            f"- noise type: {allan.get('noise_type', 'n/a')}",
            f"- drift onset tau: {_fmt(allan.get('drift_onset_tau_s'), suffix=' s')}",
        ])

    return "\n".join(lines) + "\n"


def save_deterministic_scientific_summary(
    *,
    session_dir: str | Path,
    session_id: str,
    context: dict[str, Any],
) -> dict[str, str]:
    """Persist deterministic scientific summary artifacts beside a session.

    Writes both markdown and JSON so researchers can read the summary directly
    and downstream tooling can consume the structured content.
    """
    base = Path(session_dir)
    base.mkdir(parents=True, exist_ok=True)

    markdown_path = base / f"{session_id}_scientific_summary.md"
    json_path = base / f"{session_id}_scientific_summary.json"
    report_markdown = build_deterministic_scientific_report(context)

    json_payload = {
        "session_id": session_id,
        "context": context,
        "report_markdown": report_markdown,
    }

    markdown_path.write_text(report_markdown, encoding="utf-8")
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    return {
        "markdown": str(markdown_path),
        "json": str(json_path),
    }
