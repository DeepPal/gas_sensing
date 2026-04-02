"""
src.features.compensation
===========================
Temperature and humidity drift compensation for optical sensor measurements.

Problem
-------
All optical sensors exhibit a slow peak-position drift with temperature
(thermo-optic effect: Δλ_drift ≈ α × ΔT, typically 0.02–0.10 nm/°C) and
with humidity (refractive index of humid air: 0.01–0.05 nm/%RH).  If
uncompensated, this drift is indistinguishable from a low-concentration
analyte response.

Compensation strategies
-----------------------
1. **Model-based correction** (requires T, RH sensors):
   Δλ_corrected = Δλ_observed − (α_T × ΔT + α_RH × ΔRH)

   The sensitivity coefficients α_T, α_RH are fitted from the calibration
   dataset (using frames with known zero analyte concentration at varying T/RH).

2. **Reference-peak subtraction** (multi-peak sensors):
   Δλ_signal = Δλ_peak_analyte − Δλ_peak_reference

   A reference peak that is INSENSITIVE to the analyte shifts only with
   temperature/humidity. Subtracting it removes common-mode environmental drift.
   Powerful because it needs no external T/RH sensor.

3. **Polynomial detrending** (offline / post-processing):
   Fit a polynomial baseline to the time series of peak shifts during
   known-clean-air periods; subtract the interpolated trend from the
   analyte exposure period.

4. **Online adaptive drift correction** (real-time):
   Track the slow-moving mean peak position using an exponential moving
   average during clean-air gaps between exposures; subtract from subsequent
   frames.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-based compensation (requires T, RH sensors)
# ---------------------------------------------------------------------------


class EnvironmentalCompensator:
    """Model-based temperature and humidity compensation.

    Fits linear drift coefficients (α_T, α_RH) per spectral peak from
    calibration data at zero analyte concentration.  Applies the correction
    in real-time during measurement.

    Parameters
    ----------
    n_peaks:
        Number of spectral peaks to compensate.
    ref_temp_c:
        Reference temperature at calibration (°C). Drift is computed relative
        to this value.
    ref_humidity_pct:
        Reference humidity at calibration (%RH).
    """

    def __init__(
        self,
        n_peaks: int,
        ref_temp_c: float = 25.0,
        ref_humidity_pct: float = 50.0,
    ) -> None:
        self._n_peaks = n_peaks
        self._ref_temp = ref_temp_c
        self._ref_humidity = ref_humidity_pct
        # α[j, 0] = temp sensitivity of peak j (nm/°C)
        # α[j, 1] = humidity sensitivity of peak j (nm/%RH)
        self._alpha: np.ndarray = np.zeros((n_peaks, 2))
        self._intercepts: np.ndarray = np.zeros(n_peaks)
        self.is_fitted: bool = False

    def fit(
        self,
        temp_c: np.ndarray,
        humidity_pct: np.ndarray,
        peak_shifts: np.ndarray,
    ) -> dict[str, Any]:
        """Fit drift coefficients from zero-analyte calibration frames.

        Parameters
        ----------
        temp_c:
            Temperature at each frame (°C), shape (n_frames,).
        humidity_pct:
            Relative humidity at each frame (%RH), shape (n_frames,).
        peak_shifts:
            Observed peak shifts (nm) at each frame, shape (n_frames, n_peaks).
            These must be frames with ZERO analyte — pure environmental drift.

        Returns
        -------
        dict with fitted α values and R² per peak.
        """
        temp_c = np.asarray(temp_c, dtype=float)
        humidity_pct = np.asarray(humidity_pct, dtype=float)
        peak_shifts = np.asarray(peak_shifts, dtype=float)

        if peak_shifts.ndim == 1:
            peak_shifts = peak_shifts.reshape(-1, 1)

        n = len(temp_c)
        # Feature matrix: [ΔT, ΔRH, 1]
        A = np.column_stack([
            temp_c - self._ref_temp,
            humidity_pct - self._ref_humidity,
            np.ones(n),
        ])

        r2_per_peak: list[float] = []
        for j in range(self._n_peaks):
            y = peak_shifts[:, j] if peak_shifts.shape[1] > j else np.zeros(n)
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            self._alpha[j, 0] = coeffs[0]   # nm/°C
            self._alpha[j, 1] = coeffs[1]   # nm/%RH
            self._intercepts[j] = coeffs[2]

            y_hat = A @ coeffs
            ss_res = float(np.sum((y - y_hat) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
            r2_per_peak.append(r2)
            log.debug(
                "Peak %d: α_T=%.4f nm/°C, α_RH=%.4f nm/%%RH, R²=%.4f",
                j, coeffs[0], coeffs[1], r2,
            )

        self.is_fitted = True
        return {
            "alpha_temp_nm_per_c": self._alpha[:, 0].tolist(),
            "alpha_humidity_nm_per_pct": self._alpha[:, 1].tolist(),
            "r2_per_peak": r2_per_peak,
        }

    def compensate(
        self,
        peak_shifts_nm: np.ndarray,
        temp_c: float,
        humidity_pct: float,
    ) -> np.ndarray:
        """Remove environmental drift from observed peak shifts.

        Parameters
        ----------
        peak_shifts_nm:
            Observed peak shifts (nm) relative to reference, shape (n_peaks,).
        temp_c, humidity_pct:
            Current environmental conditions.

        Returns
        -------
        np.ndarray
            Compensated peak shifts with environmental drift removed.
        """
        if not self.is_fitted:
            log.warning("EnvironmentalCompensator not fitted — returning raw shifts.")
            return np.asarray(peak_shifts_nm, dtype=float)

        dT = temp_c - self._ref_temp
        dRH = humidity_pct - self._ref_humidity
        drift = self._alpha[:, 0] * dT + self._alpha[:, 1] * dRH + self._intercepts
        return np.asarray(peak_shifts_nm, dtype=float) - drift

    def compensate_batch(
        self,
        peak_shifts: np.ndarray,
        temp_c: np.ndarray,
        humidity_pct: np.ndarray,
    ) -> np.ndarray:
        """Batch compensation for a time series of frames.

        Parameters
        ----------
        peak_shifts:
            Shape (n_frames, n_peaks).
        temp_c, humidity_pct:
            Shape (n_frames,).

        Returns
        -------
        np.ndarray
            Compensated shifts, shape (n_frames, n_peaks).
        """
        shifts = np.asarray(peak_shifts, dtype=float)
        tc = np.asarray(temp_c, dtype=float)
        rh = np.asarray(humidity_pct, dtype=float)

        dT = tc - self._ref_temp           # (n_frames,)
        dRH = rh - self._ref_humidity

        # drift[n, j] = α_T[j] × dT[n] + α_RH[j] × dRH[n] + intercept[j]
        drift = (
            np.outer(dT, self._alpha[:, 0])
            + np.outer(dRH, self._alpha[:, 1])
            + self._intercepts[None, :]
        )
        return shifts - drift

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({
            "n_peaks": self._n_peaks,
            "ref_temp": self._ref_temp,
            "ref_humidity": self._ref_humidity,
            "alpha": self._alpha,
            "intercepts": self._intercepts,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path: str) -> "EnvironmentalCompensator":
        import joblib
        state = joblib.load(path)
        obj = cls(state["n_peaks"], state["ref_temp"], state["ref_humidity"])
        obj._alpha = state["alpha"]
        obj._intercepts = state["intercepts"]
        obj.is_fitted = state["is_fitted"]
        return obj


# ---------------------------------------------------------------------------
# Differential reference-peak compensation (no T/RH sensor needed)
# ---------------------------------------------------------------------------


def differential_peak_correction(
    peak_shifts_nm: np.ndarray,
    analyte_peak_indices: list[int],
    reference_peak_indices: list[int],
) -> np.ndarray:
    """Reference-peak subtraction for multi-peak sensors.

    Uses designated reference peaks (insensitive to analyte, sensitive only
    to environmental drift) to subtract common-mode drift from analyte peaks.

    The correction is:
        Δλ_corrected[j] = Δλ_observed[j] − mean(Δλ_reference_peaks)

    This completely removes common-mode temperature/humidity drift without
    requiring any external environmental sensor.

    Parameters
    ----------
    peak_shifts_nm:
        All peak shifts, shape (n_peaks,).
    analyte_peak_indices:
        Indices of peaks that respond to the analyte.
    reference_peak_indices:
        Indices of reference peaks (analyte-insensitive).

    Returns
    -------
    np.ndarray
        Corrected shifts for analyte peaks only, shape (n_analyte_peaks,).
    """
    shifts = np.asarray(peak_shifts_nm, dtype=float)
    ref_drift = shifts[reference_peak_indices].mean()
    return shifts[analyte_peak_indices] - ref_drift


# ---------------------------------------------------------------------------
# Online adaptive drift correction (no clean-air labels needed)
# ---------------------------------------------------------------------------


class AdaptiveDriftCorrector:
    """Exponential moving average drift tracker for real-time correction.

    Tracks a slow-moving baseline in the peak shift time series using an EMA.
    The EMA is updated only when the current shift is close to the current
    baseline (clean-air assumption: analyte exposures cause sharp upward steps;
    drift is slow and continuous).

    Parameters
    ----------
    n_peaks:
        Number of peaks to track.
    alpha_ema:
        EMA smoothing factor in [0, 1]. Smaller = slower response to drift
        (longer effective time constant τ ≈ 1/(2×f_s×α)).
    step_threshold_nm:
        If |Δλ - EMA| > threshold for any peak, treat as analyte exposure
        and freeze the EMA (don't update baseline with contaminated signal).
    """

    def __init__(
        self,
        n_peaks: int,
        alpha_ema: float = 0.01,
        step_threshold_nm: float = 0.1,
    ) -> None:
        self._n_peaks = n_peaks
        self._alpha = alpha_ema
        self._threshold = step_threshold_nm
        self._ema: np.ndarray | None = None    # current EMA baseline
        self._frozen: bool = False             # True during analyte exposure

    def update_and_correct(
        self,
        peak_shifts_nm: np.ndarray,
    ) -> np.ndarray:
        """Update the drift baseline and return corrected shifts.

        Parameters
        ----------
        peak_shifts_nm:
            Current frame peak shifts relative to reference, shape (n_peaks,).

        Returns
        -------
        np.ndarray
            Drift-corrected shifts (analyte signal only), shape (n_peaks,).
        """
        shifts = np.asarray(peak_shifts_nm, dtype=float)

        if self._ema is None:
            self._ema = shifts.copy()
            return np.zeros(self._n_peaks)

        # Detect analyte step: any peak deviating more than threshold from EMA
        deviation = np.abs(shifts - self._ema)
        if np.any(deviation > self._threshold):
            # Analyte exposure — freeze EMA to avoid corrupting baseline
            self._frozen = True
        else:
            self._frozen = False
            # Update EMA with current clean-air observation
            self._ema = (1.0 - self._alpha) * self._ema + self._alpha * shifts

        corrected = shifts - self._ema
        return corrected

    def reset(self) -> None:
        """Reset the drift tracker (call at start of new session)."""
        self._ema = None
        self._frozen = False

    @property
    def current_baseline_nm(self) -> np.ndarray | None:
        """Current estimated environmental drift baseline (nm per peak)."""
        return self._ema.copy() if self._ema is not None else None


# ---------------------------------------------------------------------------
# Polynomial baseline detrending (offline)
# ---------------------------------------------------------------------------


def polynomial_detrend(
    times_s: np.ndarray,
    peak_shifts: np.ndarray,
    clean_air_mask: np.ndarray,
    poly_degree: int = 2,
) -> np.ndarray:
    """Fit and subtract a polynomial drift baseline from a time series.

    Fits the polynomial to clean-air frames only (where analyte = 0),
    then evaluates and subtracts it across all frames.

    Parameters
    ----------
    times_s:
        Frame timestamps (s), shape (n_frames,).
    peak_shifts:
        Per-peak shifts (nm), shape (n_frames, n_peaks) or (n_frames,).
    clean_air_mask:
        Boolean mask of clean-air (zero-analyte) frames, shape (n_frames,).
    poly_degree:
        Degree of polynomial fit (2 = quadratic drift; 3 for more complex).

    Returns
    -------
    np.ndarray
        Detrended shifts, same shape as ``peak_shifts``.
    """
    t = np.asarray(times_s, dtype=float)
    shifts = np.asarray(peak_shifts, dtype=float)
    mask = np.asarray(clean_air_mask, dtype=bool)

    scalar_input = shifts.ndim == 1
    if scalar_input:
        shifts = shifts.reshape(-1, 1)

    n_peaks = shifts.shape[1]
    corrected = shifts.copy()

    for j in range(n_peaks):
        if mask.sum() < poly_degree + 1:
            log.warning("Insufficient clean-air frames for polynomial detrend (peak %d)", j)
            continue
        coeffs = np.polyfit(t[mask], shifts[mask, j], poly_degree)
        trend = np.polyval(coeffs, t)
        corrected[:, j] = shifts[:, j] - trend

    if scalar_input:
        return corrected[:, 0]
    return corrected
