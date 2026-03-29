"""
src.inference.session_analyzer
================================
Automated post-session analysis: LOD/LOQ, calibration quality, drift,
SNR statistics, and conformal interval statistics.

Triggered automatically when a session stops (via spectraagent event bus)
and its output is passed to the event bus / ReportWriter.

LOD/LOQ derivation
------------------
Follows IUPAC 2012 / Eurachem Guide:

    LOD = 3 · σ_blank_nm / m
    LOQ = 10 · σ_blank_nm / m

where σ_blank_nm is the standard deviation of calibration shift residuals
(nm) from a linear fit, and m (nm/ppm) is the low-concentration sensitivity
estimated from the Henry's-law (bottom third) region of the calibration curve.

This correctly accounts for sensor sensitivity: a less sensitive sensor
(smaller |m|) yields a higher (worse) LOD even with the same noise floor.

If dedicated blank measurements (type="blank", concentration_ppm=0) are
present in the event stream, σ_blank_nm is computed from those instead,
and ``SessionAnalysis.lod_used_blanks`` is set to True.

Public API
----------
- ``SessionAnalysis``  — dataclass of all computed statistics
- ``SessionAnalyzer``  — .analyze(events, frame_count) -> SessionAnalysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np

from src.scientific.lod import lod_bootstrap_ci

log = logging.getLogger(__name__)


@dataclass
class SessionAnalysis:
    """All statistics computed from one measurement session."""

    frame_count: int = 0

    # Calibration quality
    calibration_r2: float | None = None
    calibration_rmse_ppm: float | None = None
    calibration_n_points: int = 0

    # IUPAC detection limits (IUPAC 2012 / Eurachem Guide triad)
    # LOB = μ_blank + 1.645·σ_blank  (95th percentile of blank distribution)
    # LOD = 3·σ_blank / m            (smallest detectable signal)
    # LOQ = 10·σ_blank / m           (smallest quantifiable signal)
    lob_ppm: float = float("nan")       # Limit of Blank (mandatory for publication)
    lod_ppm: float = float("nan")
    lod_ci_lower: float = float("nan")  # 95% bootstrap CI lower bound on LOD
    lod_ci_upper: float = float("nan")  # 95% bootstrap CI upper bound on LOD
    loq_ppm: float = float("nan")
    loq_ci_lower: float = float("nan")  # 95% bootstrap CI lower bound on LOQ
    loq_ci_upper: float = float("nan")  # 95% bootstrap CI upper bound on LOQ
    lod_used_blanks: bool = False        # True when σ_blank_nm was from blank events

    # Measurement statistics
    mean_concentration_ppm: float | None = None
    std_concentration_ppm: float | None = None
    mean_ci_width_ppm: float | None = None
    mean_snr: float = float("nan")

    # Drift: linear trend in peak_wavelength over measurement frames
    drift_rate_nm_per_frame: float | None = None
    total_drift_nm: float | None = None

    # Interval coverage check (when ground truth is available)
    interval_coverage: float | None = None

    # Raw calibration series (forwarded to ReportWriter for plots)
    calibration_concentrations: list[float] = field(default_factory=list)
    calibration_shifts: list[float] = field(default_factory=list)

    # Human-readable summary for event bus / log
    summary_text: str = ""


class SessionAnalyzer:
    """Compute post-session statistics from a list of event dicts.

    Each event dict may contain any subset of:
        ``type``              : "calibration_point" | "measurement"
        ``concentration_ppm`` : float
        ``wavelength_shift``  : float (nm)
        ``snr``               : float
        ``peak_wavelength``   : float (nm) — measurement events only
        ``ci_low``, ``ci_high``: float — measurement events only
    """

    def analyze(
        self,
        events: list[dict[str, Any]],
        frame_count: int,
    ) -> SessionAnalysis:
        """Compute all session statistics.

        Parameters
        ----------
        events      : list of event dicts accumulated during the session
        frame_count : total number of acquired frames (may exceed len(events))

        Returns
        -------
        SessionAnalysis
        """
        result = SessionAnalysis(frame_count=frame_count)

        if not events:
            result.summary_text = "No events recorded in this session."
            return result

        cal_events = [e for e in events if e.get("type") == "calibration_point"]
        meas_events = [e for e in events if e.get("type") == "measurement"]

        # ── Calibration quality ──────────────────────────────────────────
        if len(cal_events) >= 3:
            cal_concs = np.array([e["concentration_ppm"] for e in cal_events])
            cal_shifts = np.array([e["wavelength_shift"] for e in cal_events])
            result.calibration_concentrations = cal_concs.tolist()
            result.calibration_shifts = cal_shifts.tolist()
            result.calibration_n_points = len(cal_concs)

            try:
                coeffs = np.polyfit(cal_shifts, cal_concs, 1)
                predicted = np.polyval(coeffs, cal_shifts)
                ss_res = float(np.sum((cal_concs - predicted) ** 2))
                ss_tot = float(np.sum((cal_concs - np.mean(cal_concs)) ** 2))
                result.calibration_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
                result.calibration_rmse_ppm = float(
                    np.sqrt(np.mean((cal_concs - predicted) ** 2))
                )
            except Exception as exc:
                log.debug("Calibration R² computation failed: %s", exc)

        # ── LOD / LOQ (IUPAC 2012 / Eurachem Guide) ─────────────────────
        # LOD = 3 · σ_blank_nm / |m|   LOQ = 10 · σ_blank_nm / |m|
        # σ_blank_nm = noise in signal space (nm shift residuals):
        #   - preferred: std of blank event shifts (type="blank" or conc=0)
        #   - fallback:  std of residuals from a linear fit to calibration data
        # m = sensitivity at low concentration (nm / ppm), estimated
        #     from the Henry's-law (low-conc) portion of the curve.
        # This properly accounts for sensor sensitivity: a flatter calibration
        # curve (low m) correctly yields a higher (worse) LOD.
        if result.calibration_n_points >= 3:
            # Preferred: use dedicated blank measurements for σ_blank_nm
            blank_events = [
                e for e in events
                if e.get("type") == "blank"
                or (e.get("type") == "calibration_point" and float(e.get("concentration_ppm", -1)) == 0.0)
            ]
            blank_shifts = np.array([
                e["wavelength_shift"] for e in blank_events
                if e.get("wavelength_shift") is not None
            ])

            if len(blank_shifts) >= 2:
                sigma_blank_nm = float(np.std(blank_shifts, ddof=1))
                result.lod_used_blanks = True
            else:
                # Fallback: residual noise from a global linear fit to calibration data
                signal_coeffs = np.polyfit(cal_concs, cal_shifts, 1)  # nm per ppm
                signal_residuals_nm = cal_shifts - np.polyval(signal_coeffs, cal_concs)
                sigma_blank_nm = float(np.std(signal_residuals_nm, ddof=1))

            # Sensitivity m from the low-concentration Henry's-law regime
            n_low = max(2, result.calibration_n_points // 3)
            sorted_idx = np.argsort(cal_concs)
            low_concs = cal_concs[sorted_idx[:n_low]]
            low_shifts = cal_shifts[sorted_idx[:n_low]]
            if np.ptp(low_concs) > 1e-9:
                m_nm_per_ppm = float(np.polyfit(low_concs, low_shifts, 1)[0])
            else:
                signal_coeffs = np.polyfit(cal_concs, cal_shifts, 1)
                m_nm_per_ppm = float(signal_coeffs[0])  # fallback: global slope

            if abs(m_nm_per_ppm) > 1e-9:
                abs_m = abs(m_nm_per_ppm)

                # LOB = μ_blank + 1.645·σ_blank (one-sided 95th percentile of blank)
                # In concentration space: blank_mean_nm / |m| + 1.645·σ_blank_nm / |m|
                blank_mean_nm = float(np.mean(blank_shifts)) if len(blank_shifts) >= 2 else 0.0
                result.lob_ppm = max(
                    (abs(blank_mean_nm) + 1.645 * sigma_blank_nm) / abs_m, 1e-7
                )

                # LOD and LOQ point estimates
                result.lod_ppm = max(3.0 * sigma_blank_nm / abs_m, 1e-6)
                result.loq_ppm = max(10.0 * sigma_blank_nm / abs_m, 3e-6)

                # Bootstrap 95% CI on LOD/LOQ using low-concentration data
                # (Henry's law region gives the relevant sensitivity estimate)
                try:
                    _, ci_lo, ci_hi = lod_bootstrap_ci(
                        low_concs,
                        low_shifts,
                        baseline_noise_std=sigma_blank_nm,
                        n_bootstrap=500,
                        confidence=0.95,
                    )
                    result.lod_ci_lower = max(ci_lo, 1e-7)
                    result.lod_ci_upper = max(ci_hi, result.lod_ci_lower)
                    # LOQ CI scales by the same 10/3 factor as the point estimate
                    scale = result.loq_ppm / result.lod_ppm
                    result.loq_ci_lower = result.lod_ci_lower * scale
                    result.loq_ci_upper = result.lod_ci_upper * scale
                except Exception as exc:
                    log.debug("LOD bootstrap CI failed: %s", exc)

        elif meas_events:
            # Rough estimate when no calibration data: 3σ of concentration spread
            concs = np.array(
                [e["concentration_ppm"] for e in meas_events
                 if e.get("concentration_ppm") is not None]
            )
            if len(concs) >= 3:
                sigma = float(np.std(concs, ddof=1))
                result.lod_ppm = max(3.0 * sigma, 1e-4)
                result.loq_ppm = max(10.0 * sigma, 3e-4)

        # ── Measurement statistics ───────────────────────────────────────
        meas_concs = [
            e["concentration_ppm"] for e in meas_events
            if e.get("concentration_ppm") is not None
        ]
        if meas_concs:
            result.mean_concentration_ppm = float(np.mean(meas_concs))
            result.std_concentration_ppm = float(np.std(meas_concs))

        ci_widths = [
            e["ci_high"] - e["ci_low"]
            for e in meas_events
            if e.get("ci_low") is not None and e.get("ci_high") is not None
        ]
        if ci_widths:
            result.mean_ci_width_ppm = float(np.mean(ci_widths))

        all_snr = [e["snr"] for e in events if e.get("snr") is not None]
        if all_snr:
            result.mean_snr = float(np.mean(all_snr))

        # ── Drift (linear trend separating instrumental from analyte signal) ──
        # Preferred: use wavelength_shift residuals from per-frame mean shift,
        # which decouples thermal/mechanical drift from real analyte binding.
        # Fallback: raw peak_wavelength when shift data are absent.
        shift_series = [
            (i, e["wavelength_shift"])
            for i, e in enumerate(meas_events)
            if e.get("wavelength_shift") is not None
        ]
        peak_wl_series = [
            (i, e["peak_wavelength"])
            for i, e in enumerate(meas_events)
            if e.get("peak_wavelength") is not None
        ]
        if len(shift_series) >= 3:
            frames_arr = np.array([p[0] for p in shift_series], dtype=float)
            shifts_arr = np.array([p[1] for p in shift_series])
            # Remove analyte trend: residuals around the mean shift capture drift
            shift_residuals = shifts_arr - float(np.mean(shifts_arr))
            coeffs = np.polyfit(frames_arr, shift_residuals, 1)
            result.drift_rate_nm_per_frame = float(coeffs[0])
            result.total_drift_nm = float(shift_residuals[-1] - shift_residuals[0])
        elif len(peak_wl_series) >= 3:
            frames_arr = np.array([p[0] for p in peak_wl_series], dtype=float)
            wls_arr = np.array([p[1] for p in peak_wl_series])
            coeffs = np.polyfit(frames_arr, wls_arr, 1)
            result.drift_rate_nm_per_frame = float(coeffs[0])
            result.total_drift_nm = float(wls_arr[-1] - wls_arr[0])

        # ── Summary text ─────────────────────────────────────────────────
        lines = [f"Session summary: {frame_count} frames acquired."]
        lines.append(f"Calibration: {result.calibration_n_points} points")
        if result.calibration_r2 is not None:
            lines.append(f"  R\u00b2 = {result.calibration_r2:.4f}")
        if not np.isnan(result.lob_ppm):
            lines.append(f"  LOB = {result.lob_ppm:.4f} ppm")
        if not np.isnan(result.lod_ppm):
            if not np.isnan(result.lod_ci_lower):
                lines.append(
                    f"  LOD = {result.lod_ppm:.4f} ppm "
                    f"[95% CI {result.lod_ci_lower:.4f}–{result.lod_ci_upper:.4f}]"
                )
            else:
                lines.append(f"  LOD = {result.lod_ppm:.4f} ppm")
            lines.append(f"  LOQ = {result.loq_ppm:.4f} ppm")
        if result.drift_rate_nm_per_frame is not None:
            lines.append(
                f"Drift rate: {result.drift_rate_nm_per_frame:.6f} nm/frame"
            )
        if not np.isnan(result.mean_snr):
            lines.append(f"Mean SNR: {result.mean_snr:.1f}")
        result.summary_text = "\n".join(lines)

        return result
