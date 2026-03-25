"""
src.agents.drift
================
Drift Detection Agent for the Au-MIP LSPR gas sensing pipeline.

What is sensor drift?
---------------------
Over time, Au-MIP LSPR sensors exhibit baseline drift — the zero-gas
peak wavelength shifts slowly due to:

- Temperature fluctuations (thermoplasmonic effect)
- Humidity-induced swelling of the MIP layer
- Photo-degradation of the Au nanoparticle surface
- Fouling / partial saturation of binding sites

Drift manifests as a slow trend in the baseline peak wavelength that is
*independent* of analyte concentration.  If uncorrected, it biases every
concentration estimate upward or downward.

Detection strategy
------------------
The agent maintains a rolling window of recent peak wavelength readings.
It fits a linear trend to the window and flags drift when the slope
(nm/min) exceeds a configurable threshold.  A second check compares the
current baseline to the historical mean — if the absolute offset exceeds
``offset_threshold_nm``, it raises a ``DRIFT_OFFSET`` alert.

Public API
----------
- ``DriftDetectionAgent``   — push samples, check status, get alerts
- ``DriftAlert``            — typed alert dataclass
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------


class DriftAlertType(str, Enum):
    """Category of a drift event."""

    DRIFT_SLOPE = "DRIFT_SLOPE"  # Trend too steep (nm/min)
    DRIFT_OFFSET = "DRIFT_OFFSET"  # Absolute baseline shift too large
    DRIFT_CLEARED = "DRIFT_CLEARED"  # Previously drifting, now stable


@dataclass
class DriftAlert:
    """A single drift detection event."""

    alert_type: DriftAlertType
    timestamp: datetime
    peak_wavelength_nm: float
    drift_rate_nm_per_min: float
    baseline_offset_nm: float
    message: str
    severity: str = "warning"  # "info" | "warning" | "critical"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DriftDetectionAgent:
    """Monitors baseline peak-wavelength for slow drift.

    Parameters
    ----------
    window_size:
        Number of recent frames kept in the rolling window.
    drift_threshold_nm_per_min:
        Linear slope above which drift is declared (nm/min).
        Typical value: 0.5 nm/min corresponds to ~4 ppm EtOH / min baseline shift.
    offset_threshold_nm:
        Absolute peak-wavelength offset from the established baseline above
        which a DRIFT_OFFSET alert is raised.
    min_samples:
        Minimum number of samples required before drift analysis begins.
    """

    def __init__(
        self,
        window_size: int = 120,
        drift_threshold_nm_per_min: float = 0.5,
        offset_threshold_nm: float = 1.0,
        min_samples: int = 20,
    ) -> None:
        self.window_size = window_size
        self.drift_threshold = drift_threshold_nm_per_min
        self.offset_threshold = offset_threshold_nm
        self.min_samples = min_samples

        # Rolling buffers: (unix_timestamp_s, peak_wavelength_nm)
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._wavelengths: deque[float] = deque(maxlen=window_size)

        # Established baseline (set after first min_samples frames)
        self._baseline_wl: float | None = None

        # State
        self._is_drifting: bool = False
        self._alerts: list[DriftAlert] = []
        self._last_slope_nm_per_min: float = 0.0
        self._last_offset_nm: float = 0.0

    # ------------------------------------------------------------------
    # Feed
    # ------------------------------------------------------------------

    def push(
        self,
        peak_wavelength_nm: float,
        timestamp: datetime | None = None,
    ) -> DriftAlert | None:
        """Record a new peak-wavelength observation.

        Parameters
        ----------
        peak_wavelength_nm:
            Detected LSPR peak wavelength in nm.
        timestamp:
            UTC datetime of the measurement.  Defaults to ``now()``.

        Returns
        -------
        DriftAlert or None
            An alert if a new drift event was detected or cleared; otherwise None.
        """
        if not np.isfinite(peak_wavelength_nm):
            return None

        ts = (timestamp or datetime.now(timezone.utc)).timestamp()
        self._timestamps.append(ts)
        self._wavelengths.append(peak_wavelength_nm)

        if len(self._wavelengths) < self.min_samples:
            return None

        # Establish baseline from first min_samples frames
        if self._baseline_wl is None:
            self._baseline_wl = float(np.mean(list(self._wavelengths)[: self.min_samples]))
            log.info("DriftAgent: baseline established at %.3f nm", self._baseline_wl)

        return self._analyse()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analyse(self) -> DriftAlert | None:
        t = np.array(self._timestamps, dtype=float)
        wl = np.array(self._wavelengths, dtype=float)

        # Centre time to avoid numerical issues
        t_min = t[0]
        t_norm = (t - t_min) / 60.0  # convert to minutes

        # Linear trend over the window
        try:
            coeffs = np.polyfit(t_norm, wl, 1)
            slope_nm_per_min = float(coeffs[0])
        except Exception:
            slope_nm_per_min = 0.0

        offset_nm = float(wl[-1] - (self._baseline_wl or wl[0]))

        self._last_slope_nm_per_min = slope_nm_per_min
        self._last_offset_nm = offset_nm

        was_drifting = self._is_drifting
        slope_drift = abs(slope_nm_per_min) > self.drift_threshold
        offset_drift = abs(offset_nm) > self.offset_threshold
        now_drifting = slope_drift or offset_drift

        alert: DriftAlert | None = None
        now_dt = datetime.fromtimestamp(self._timestamps[-1], tz=timezone.utc)

        if now_drifting and not was_drifting:
            atype = DriftAlertType.DRIFT_SLOPE if slope_drift else DriftAlertType.DRIFT_OFFSET
            severity = "critical" if abs(slope_nm_per_min) > 2 * self.drift_threshold else "warning"
            alert = DriftAlert(
                alert_type=atype,
                timestamp=now_dt,
                peak_wavelength_nm=float(self._wavelengths[-1]),
                drift_rate_nm_per_min=slope_nm_per_min,
                baseline_offset_nm=offset_nm,
                message=(
                    f"Sensor drift detected: slope={slope_nm_per_min:+.3f} nm/min, "
                    f"offset={offset_nm:+.3f} nm from baseline."
                ),
                severity=severity,
            )
            self._is_drifting = True
            self._alerts.append(alert)
            log.warning("DriftAgent: %s", alert.message)

        elif was_drifting and not now_drifting:
            alert = DriftAlert(
                alert_type=DriftAlertType.DRIFT_CLEARED,
                timestamp=now_dt,
                peak_wavelength_nm=float(self._wavelengths[-1]),
                drift_rate_nm_per_min=slope_nm_per_min,
                baseline_offset_nm=offset_nm,
                message="Drift cleared — sensor appears stable.",
                severity="info",
            )
            self._is_drifting = False
            self._alerts.append(alert)
            log.info("DriftAgent: drift cleared")

        return alert

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def is_drifting(self) -> bool:
        """Return True if the sensor is currently in a drift state."""
        return self._is_drifting

    def get_status(self) -> dict[str, object]:
        """Return a JSON-serialisable status snapshot."""
        return {
            "is_drifting": self._is_drifting,
            "drift_rate_nm_per_min": round(self._last_slope_nm_per_min, 4),
            "baseline_offset_nm": round(self._last_offset_nm, 4),
            "baseline_wl_nm": round(self._baseline_wl, 4) if self._baseline_wl else None,
            "n_samples": len(self._wavelengths),
            "n_alerts": len(self._alerts),
        }

    def get_recent_alerts(self, n: int = 10) -> list[DriftAlert]:
        """Return the last *n* alerts."""
        return self._alerts[-n:]

    def reset_baseline(self) -> None:
        """Re-establish the baseline from the current window mean."""
        if self._wavelengths:
            self._baseline_wl = float(np.mean(self._wavelengths))
            self._is_drifting = False
            log.info("DriftAgent: baseline reset to %.3f nm", self._baseline_wl)

    def reset(self) -> None:
        """Full reset — clear all state."""
        self._timestamps.clear()
        self._wavelengths.clear()
        self._baseline_wl = None
        self._is_drifting = False
        self._alerts.clear()
        self._last_slope_nm_per_min = 0.0
        self._last_offset_nm = 0.0
