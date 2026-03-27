"""
spectraagent.webapp.agents.drift
==================================
DriftAgent — rolling CUSUM on LSPR peak wavelength shift.

Uses a 60-frame rolling window and linear regression slope to estimate
drift rate in nm/min.  Emits a ``drift_warn`` event when the absolute
drift rate exceeds the configurable threshold (default 0.05 nm/min).

Called from the acquisition loop after QualityAgent passes the frame.
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Optional

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

_WINDOW_FRAMES: int = 60
_DRIFT_THRESHOLD_NM_PER_MIN: float = 0.05


class DriftAgent:
    """Rolling drift monitor for LSPR peak wavelength.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    integration_time_ms:
        Frame acquisition time in ms; used to convert frames → minutes.
    window_frames:
        Rolling window size (default 60).
    drift_threshold_nm_per_min:
        Absolute drift rate that triggers ``drift_warn`` (default 0.05 nm/min).
    """

    def __init__(
        self,
        bus: AgentBus,
        integration_time_ms: float = 50.0,
        window_frames: int = _WINDOW_FRAMES,
        drift_threshold_nm_per_min: float = _DRIFT_THRESHOLD_NM_PER_MIN,
    ) -> None:
        self._bus = bus
        self._window = window_frames
        self._threshold = drift_threshold_nm_per_min
        # frames/min = 60 000 ms/min ÷ ms/frame
        self._frames_per_minute = 60_000.0 / integration_time_ms
        self._history: deque[float] = deque(maxlen=window_frames)
        # Pending undelivered emit handles; cancelled by reset() to suppress stale alerts.
        self._pending: list[asyncio.Handle] = []

    def update(self, frame_num: int, peak_wavelength: float) -> None:
        """Record a peak wavelength observation. Emits drift_warn if threshold exceeded.

        Requires a full window before emitting any events.

        Parameters
        ----------
        frame_num:
            Current frame counter (included in event data).
        peak_wavelength:
            Detected LSPR peak wavelength in nm for this frame.
        """
        self._history.append(peak_wavelength)
        if len(self._history) < self._window:
            return

        history = np.array(self._history)
        x = np.arange(len(history), dtype=float)
        slope_nm_per_frame = float(np.polyfit(x, history, 1)[0])
        drift_rate = slope_nm_per_frame * self._frames_per_minute

        if abs(drift_rate) > self._threshold:
            handle = self._bus.emit(AgentEvent(
                source="DriftAgent",
                level="warn",
                type="drift_warn",
                data={
                    "frame": frame_num,
                    "drift_rate_nm_per_min": round(drift_rate, 4),
                    "window_frames": self._window,
                    "peak_wavelength": round(peak_wavelength, 4),
                },
                text=(
                    f"Frame {frame_num} — drift rate {drift_rate:+.4f} nm/min "
                    f"exceeds ±{self._threshold} nm/min "
                    f"(over {self._window}-frame window)."
                ),
            ))
            if handle is not None:
                # Track so reset() can cancel delivery before the loop tick.
                self._pending.append(handle)

    def reset(self) -> None:
        """Clear rolling history and cancel any pending undelivered drift events.

        Call at session start or after a reference capture.  Any drift_warn
        events that were emitted but not yet delivered to subscriber queues
        are cancelled so stale alerts do not pollute the new session.
        """
        self._history.clear()
        for handle in self._pending:
            handle.cancel()
        self._pending.clear()
