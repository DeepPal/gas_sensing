"""
src.signal.peak
================
Sub-pixel peak detection for LSPR spectral data.

Background
----------
The Au-MIP LSPR peak sits at ~717.9 nm for the reference spectrum.  Gas
adsorption causes a wavelength *shift* of order −10 nm at 0.1 ppm Ethanol.
Accurate shift measurement requires sub-pixel peak localisation; the CCS200
pixel spacing is ~0.2 nm so a naive argmax gives ±0.1 nm error, comparable
to the signal itself at low concentrations.

Two complementary methods are provided:

* **Gaussian fit** (``gaussian_peak_center``) — fits a Gaussian to a local
  window around the apex.  Works well for smooth, isolated peaks.

* **Cross-correlation shift** (``estimate_shift_crosscorr``) — measures the
  global spectral shift between a reference and a sample spectrum via the
  peak of the normalised cross-correlation.  Robust when the peak shape
  changes slightly between reference and sample.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate


def gaussian_peak_center(
    x: np.ndarray,
    y: np.ndarray,
    idx_hint: int | None = None,
    half_width: int = 5,
) -> float:
    """Estimate sub-pixel peak centre using a Gaussian fit in a local window.

    Handles both maxima and minima by allowing a negative amplitude.
    Falls back to intensity-weighted centroid when curve fitting fails.

    Args:
        x: Wavelength array (1-D, monotonically increasing).
        y: Intensity (or signal) array of the same length as ``x``.
        idx_hint: Index near the expected peak apex.  If None, the index of
            the maximum absolute deviation from the median is used.
        half_width: Number of pixels on each side of ``idx_hint`` to include
            in the fitting window.

    Returns:
        Estimated peak centre wavelength in the same units as ``x``.
        Returns ``np.nan`` if ``x`` is empty.
    """
    if x.size < 3:
        if x.size == 0:
            return float(np.nan)
        return float(x[idx_hint] if idx_hint is not None else x[x.size // 2])

    if idx_hint is None:
        idx_hint = int(np.argmax(np.abs(y - np.median(y))))
    idx_hint = int(np.clip(idx_hint, 0, x.size - 1))

    half_width = max(1, min(int(half_width), x.size // 2))
    s = max(0, idx_hint - half_width)
    e = min(x.size - 1, idx_hint + half_width)
    xx = x[s : e + 1]
    yy = y[s : e + 1]

    baseline = float(np.median(yy))
    is_min = bool(yy.min() < (np.median(yy) - 0.25 * (yy.max() - yy.min())))
    if is_min:
        A0 = float(yy.min() - baseline)
        idx0 = int(np.argmin(yy))
    else:
        A0 = float(yy.max() - baseline)
        idx0 = int(np.argmax(yy))
    x0_0 = float(xx[idx0])
    sigma0 = max((xx[-1] - xx[0]) / 6.0, 1e-3)

    def _gauss(xv: np.ndarray, A: float, x0: float, sigma: float, C: float) -> np.ndarray:
        sigma = max(sigma, 1e-6)
        return np.asarray(C + A * np.exp(-0.5 * ((xv - x0) / sigma) ** 2))

    p0 = [A0, x0_0, sigma0, baseline]
    bounds = (
        [-np.inf, float(xx[0]) - 5.0, 1e-6, -np.inf],
        [np.inf, float(xx[-1]) + 5.0, (float(xx[-1]) - float(xx[0])) * 2.0, np.inf],
    )
    try:
        popt, _ = curve_fit(_gauss, xx, yy, p0=p0, bounds=bounds, maxfev=5000)
        x0 = float(popt[1])
        if not np.isfinite(x0):
            raise RuntimeError("non-finite centre")
        if x0 < float(xx[0]) - 1.0 or x0 > float(xx[-1]) + 1.0:
            raise RuntimeError("centre outside window")
        return x0
    except Exception:
        weights = np.abs(yy - baseline) + 1e-9
        return float(np.sum(xx * weights) / np.sum(weights))


def estimate_shift_crosscorr(
    ref_wl: np.ndarray,
    ref_signal: np.ndarray,
    target_signal: np.ndarray,
    upsample: int = 1,
) -> float:
    """Measure spectral shift between a reference and a target spectrum.

    Uses the peak of the normalised cross-correlation of mean-subtracted signals.
    Sign convention (matches ``gas_analysis.core.pipeline``): a negative return
    means the target peak is red-shifted relative to the reference; positive
    means blue-shifted.  This is the inverse of the physical Δλ convention —
    callers should negate the result when computing Δλ for calibration.

    Args:
        ref_wl: Wavelength axis shared by both spectra (1-D, uniform spacing
            assumed for accurate nm conversion).
        ref_signal: Reference spectrum intensities.
        target_signal: Target spectrum intensities (same length as ``ref_signal``).
        upsample: Integer upsampling factor applied before correlation (use 1
            for raw-pixel accuracy, 4-10 for sub-pixel interpolation).

    Returns:
        Estimated shift in nm (same units as ``ref_wl``).

    Raises:
        ValueError: If array lengths are inconsistent.
    """
    if len(ref_wl) != len(ref_signal) or len(ref_signal) != len(target_signal):
        raise ValueError(
            "Mismatch between reference wavelengths and signal lengths for cross-correlation"
        )

    if len(ref_wl) < 2:
        return 0.0

    ref_arr: np.ndarray
    tgt_arr: np.ndarray
    if upsample > 1:
        dense_wl = np.linspace(ref_wl[0], ref_wl[-1], len(ref_wl) * upsample)
        ref_arr = np.asarray(np.interp(dense_wl, ref_wl, ref_signal))
        tgt_arr = np.asarray(np.interp(dense_wl, ref_wl, target_signal))
        dw = float(np.mean(np.diff(dense_wl)))
    else:
        ref_arr = ref_signal.copy()
        tgt_arr = target_signal.copy()
        dw = float(np.mean(np.diff(ref_wl))) if len(ref_wl) > 1 else 0.0

    ref_arr = ref_arr - float(np.mean(ref_arr))
    tgt_arr = tgt_arr - float(np.mean(tgt_arr))
    corr = np.asarray(correlate(tgt_arr, ref_arr, mode="full"))
    lags = np.arange(-len(ref_arr) + 1, len(tgt_arr))
    best_lag = float(lags[np.argmax(corr)])
    return -best_lag * dw
