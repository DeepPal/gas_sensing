"""
src.preprocessing.denoising
============================
Spectral denoising: Savitzky-Golay smoothing and wavelet soft-thresholding.

All functions are **pure** — no side effects, no state, no project imports.
They accept and return numpy arrays with the same shape.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def _ensure_odd_window(window: int, length: int) -> int:
    """Return an odd window size that is valid for the given array length.

    Rules:
    - Minimum window is 3
    - Window must be odd
    - Window must be strictly less than ``length``
    """
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    return min(window, max(3, length - (1 - length % 2)))


def savgol_smooth(
    intensities: np.ndarray,
    window: int = 11,
    poly_order: int = 2,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing to a spectrum.

    Parameters
    ----------
    intensities:
        1-D array of measured intensities.
    window:
        Smoothing window size (will be coerced to a valid odd integer).
    poly_order:
        Polynomial order for the S-G filter.  Clamped to ``window - 1``.

    Returns
    -------
    np.ndarray
        Smoothed intensities, same shape as input.
    """
    if len(intensities) < 4:
        return intensities.copy()
    w = _ensure_odd_window(window, len(intensities))
    p = min(poly_order, w - 1)
    return np.asarray(savgol_filter(intensities, w, p))


def gaussian_smooth(intensities: np.ndarray, window: int = 11) -> np.ndarray:
    """Apply Gaussian smoothing with sigma derived from window size."""
    if len(intensities) < 4:
        return intensities.copy()
    sigma = max(1.0, window / 6.0)
    return np.asarray(gaussian_filter1d(intensities, sigma=sigma))


def moving_average_smooth(intensities: np.ndarray, window: int = 11) -> np.ndarray:
    """Apply uniform moving-average smoothing."""
    if len(intensities) < 4:
        return intensities.copy()
    w = _ensure_odd_window(window, len(intensities))
    kernel = np.ones(w) / w
    return np.asarray(np.convolve(intensities, kernel, mode="same"))


def wavelet_denoise(
    intensities: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> np.ndarray:
    """Wavelet soft-thresholding denoising (Donoho–Johnstone universal threshold).

    Parameters
    ----------
    intensities:
        1-D array of measured intensities.
    wavelet:
        PyWavelets wavelet name (default: ``'db4'``).
    level:
        Decomposition level.  ``None`` → automatic (floor(log2(N)) − 2).

    Returns
    -------
    np.ndarray
        Denoised intensities.  Falls back to Savitzky-Golay if pywt unavailable.
    """
    try:
        import pywt
    except ImportError:
        return savgol_smooth(intensities)

    if len(intensities) < 4:
        return intensities.copy()

    auto_level = max(1, int(np.floor(np.log2(max(2, len(intensities)))) - 2))
    lvl = level if level is not None else auto_level

    try:
        coeffs = pywt.wavedec(intensities, wavelet, level=lvl)
        # Universal threshold (Donoho-Johnstone)
        detail = coeffs[-1]
        if len(detail) == 0:
            return intensities.copy()
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(max(2, len(intensities))))

        thresholded = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
        reconstructed = pywt.waverec(thresholded, wavelet)

        # Match original length (IDWT can produce N±1 samples)
        if len(reconstructed) != len(intensities):
            reconstructed = np.interp(
                np.arange(len(intensities)),
                np.linspace(0, len(intensities) - 1, num=len(reconstructed)),
                reconstructed,
            )
        return reconstructed.astype(float)
    except Exception:
        return savgol_smooth(intensities)


def smooth_spectrum(
    intensities: np.ndarray,
    window: int = 11,
    poly_order: int = 2,
    method: str = "savgol",
) -> np.ndarray:
    """Dispatch to the requested smoothing method.

    Parameters
    ----------
    intensities:
        1-D intensity array.
    window:
        Smoothing window size.
    poly_order:
        S-G polynomial order (ignored for non-S-G methods).
    method:
        One of ``'savgol'``, ``'gaussian'``, ``'moving_average'``, ``'wavelet'``.

    Returns
    -------
    np.ndarray
        Smoothed intensities.
    """
    if len(intensities) == 0:
        return intensities

    if method == "savgol":
        return savgol_smooth(intensities, window, poly_order)
    if method == "gaussian":
        return gaussian_smooth(intensities, window)
    if method in ("moving_average", "boxcar"):
        return moving_average_smooth(intensities, window)
    if method == "wavelet":
        return wavelet_denoise(intensities)

    raise ValueError(
        f"Unknown smoothing method: {method!r}. "
        "Choose from: savgol, gaussian, moving_average, wavelet."
    )
