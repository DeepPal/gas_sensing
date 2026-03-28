"""
src.inference.session_analyzer
================================
Automated post-session analysis: LOD/LOQ, calibration quality, drift,
SNR statistics, and conformal interval statistics.

Triggered automatically when a session stops (via spectraagent event bus)
and its output is passed to the event bus / ReportWriter.

LOD/LOQ derivation
------------------
Limit of Detection  (LOD) = 3σ  where σ = calibration RMSE (ppm)
Limit of Quantification (LOQ) = 10σ

These follow IUPAC recommendations (3σ/10σ criterion) using the calibration
residual as the proxy for measurement noise in the linear response region.

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

log = logging.getLogger(__name__)


@dataclass
class SessionAnalysis:
    """All statistics computed from one measurement session."""

    frame_count: int = 0

    # Calibration quality
    calibration_r2: float | None = None
    calibration_rmse_ppm: float | None = None
    calibration_n_points: int = 0

    # IUPAC detection limits (3σ / 10σ criterion)
    lod_ppm: float = float("nan")
    loq_ppm: float = float("nan")

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

        # ── LOD / LOQ (IUPAC 3σ / 10σ criterion) ────────────────────────
        if result.calibration_rmse_ppm is not None and result.calibration_rmse_ppm > 0:
            sigma = result.calibration_rmse_ppm
            result.lod_ppm = 3.0 * sigma
            result.loq_ppm = 10.0 * sigma
        elif meas_events:
            concs = np.array(
                [e["concentration_ppm"] for e in meas_events
                 if e.get("concentration_ppm") is not None]
            )
            if len(concs) >= 3:
                sigma = float(np.std(concs))
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

        # ── Drift (linear trend in peak wavelength over frames) ──────────
        peak_wls = [
            (i, e["peak_wavelength"])
            for i, e in enumerate(meas_events)
            if e.get("peak_wavelength") is not None
        ]
        if len(peak_wls) >= 3:
            frames_arr = np.array([p[0] for p in peak_wls], dtype=float)
            wls_arr = np.array([p[1] for p in peak_wls])
            coeffs = np.polyfit(frames_arr, wls_arr, 1)
            result.drift_rate_nm_per_frame = float(coeffs[0])
            result.total_drift_nm = float(wls_arr[-1] - wls_arr[0])

        # ── Summary text ─────────────────────────────────────────────────
        lines = [f"Session summary: {frame_count} frames acquired."]
        lines.append(f"Calibration: {result.calibration_n_points} points")
        if result.calibration_r2 is not None:
            lines.append(f"  R\u00b2 = {result.calibration_r2:.4f}")
        if not np.isnan(result.lod_ppm):
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
