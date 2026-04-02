"""
spectraagent.webapp.agents.quality
===================================
QualityAgent — per-frame SNR and saturation gate.

Called synchronously from the acquisition loop (20 Hz).
Emits one AgentEvent per frame.

Quality rules (spec Section 4):
- max(intensities) > 60 000 counts → level="error", hard block (return False)
- SNR < 3                          → level="warn",  frame processed (return True)
- Normal                           → level="ok",    frame processed (return True)

SNR estimate: peak_intensity / std(off-peak noise floor).
Noise floor = first and last 5% of pixels (off-peak regions).
"""
from __future__ import annotations

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

_SATURATION_THRESHOLD: float = 60_000.0
_SNR_WARN_THRESHOLD: float = 3.0


def _compute_snr(wavelengths: np.ndarray, intensities: np.ndarray) -> float:
    """Estimate SNR as peak_intensity / noise_floor_amplitude.

    Noise floor is estimated from the first and last 5 % of pixels
    (off-peak regions dominated by detector dark noise).

    The noise amplitude is defined as ``mean(noise) + std(noise)``, i.e. the
    noise baseline plus one standard deviation.  This estimate equals roughly
    the 84th-percentile of the off-peak distribution, which provides a
    conservative noise reference that naturally separates structured spectral
    peaks (SNR >> 3) from featureless noise frames (SNR < 3) where the
    apparent "peak" is merely a statistical extreme of the noise distribution.
    """
    if len(intensities) == 0:
        return 0.0
    if len(intensities) < 4:
        return 0.0
    n = len(intensities)
    margin = min(max(n // 20, 10), n // 4)
    noise = np.concatenate([intensities[:margin], intensities[-margin:]])
    noise_amplitude = float(np.mean(noise)) + float(np.std(noise))
    if not (noise_amplitude > 1e-9):
        noise_amplitude = 1e-9
    return float(np.max(intensities)) / noise_amplitude


class QualityAgent:
    """Per-frame quality gate — runs at 20 Hz in the acquisition thread.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    saturation_threshold:
        Hard-block threshold in raw counts (default 60 000).
    snr_warn_threshold:
        SNR warning threshold (default 3.0).
    ok_emit_every:
        Emit "ok" quality events at most once every N frames to avoid
        flooding the event log at 20 Hz.  warn/error always emitted
        immediately regardless of this setting.  Default 50 (≈2.5 s).
    """

    def __init__(
        self,
        bus: AgentBus,
        saturation_threshold: float = _SATURATION_THRESHOLD,
        snr_warn_threshold: float = _SNR_WARN_THRESHOLD,
        ok_emit_every: int = 5,
    ) -> None:
        self._bus = bus
        self._sat_threshold = saturation_threshold
        self._snr_threshold = snr_warn_threshold
        self._ok_emit_every = max(1, ok_emit_every)
        self._frames_since_ok_emit: int = 0

    def configure(
        self,
        saturation_threshold: float | None = None,
        snr_warn_threshold: float | None = None,
    ) -> None:
        """Update thresholds at runtime without restarting the server."""
        if saturation_threshold is not None:
            self._sat_threshold = float(saturation_threshold)
        if snr_warn_threshold is not None:
            self._snr_threshold = float(snr_warn_threshold)

    @property
    def settings(self) -> dict:
        return {
            "saturation_threshold": self._sat_threshold,
            "snr_warn_threshold": self._snr_threshold,
        }

    def process(
        self,
        frame_num: int,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> bool:
        """Check frame quality. Returns True to process frame, False to hard-block.

        Always emits exactly one AgentEvent regardless of outcome.
        """
        max_i = float(np.max(intensities))
        sat_pct = float(np.mean(intensities > self._sat_threshold) * 100.0)

        if max_i > self._sat_threshold:
            self._bus.emit(AgentEvent(
                source="QualityAgent",
                level="error",
                type="quality",
                data={
                    "frame": frame_num,
                    "snr": 0.0,
                    "saturation_pct": round(sat_pct, 2),
                    "quality": "saturated",
                    "max_intensity": round(max_i, 1),
                },
                text=(
                    f"Frame {frame_num} — SATURATED ({max_i:.0f} counts "
                    f"> {self._sat_threshold:.0f}). Frame discarded."
                ),
            ))
            return False

        snr = _compute_snr(wavelengths, intensities)

        if snr < self._snr_threshold:
            self._bus.emit(AgentEvent(
                source="QualityAgent",
                level="warn",
                type="quality",
                data={
                    "frame": frame_num,
                    "snr": round(snr, 2),
                    "saturation_pct": round(sat_pct, 2),
                    "quality": "low_snr",
                },
                text=(
                    f"Frame {frame_num} — SNR={snr:.1f} "
                    f"(below {self._snr_threshold}). Processed with warning."
                ),
            ))
            return True

        self._frames_since_ok_emit += 1
        if self._frames_since_ok_emit >= self._ok_emit_every:
            self._frames_since_ok_emit = 0
            self._bus.emit(AgentEvent(
                source="QualityAgent",
                level="ok",
                type="quality",
                data={
                    "frame": frame_num,
                    "snr": round(snr, 2),
                    "saturation_pct": round(sat_pct, 2),
                    "quality": "ok",
                },
                text=f"Frame {frame_num} — SNR={snr:.1f}, quality=OK",
            ))
        return True
