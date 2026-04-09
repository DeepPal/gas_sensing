"""
src.signal
==========
Spectral signal processing utilities for LSPR gas sensor analysis.

Sub-modules
-----------
- ``transforms`` — transmittance/absorbance conversions, frame smoothing
- ``peak``        — sub-pixel peak detection (Gaussian fit, cross-correlation)
- ``roi``         — ROI scanning: find wavelength regions with monotonic response

Public API
----------
- ``compute_transmittance(sample_df, ref_df)``
- ``append_absorbance_column(df)``
- ``smooth(y, window, poly)``
- ``ensure_odd_window(window)``
- ``gaussian_peak_center(x, y, ...)``
- ``estimate_shift_crosscorr(ref_wl, ref_signal, target_signal, ...)``
- ``compute_band_ratio_matrix(Y, half_width)``
- ``find_monotonic_wavelengths(canonical, min_wl, max_wl, ...)``
"""

from src.signal.peak import estimate_shift_crosscorr, gaussian_peak_center
from src.signal.roi import compute_band_ratio_matrix, find_monotonic_wavelengths
from src.signal.transforms import (
    append_absorbance_column,
    compute_transmittance,
    ensure_odd_window,
    smooth,
)

__all__ = [
    "compute_transmittance",
    "append_absorbance_column",
    "smooth",
    "ensure_odd_window",
    "gaussian_peak_center",
    "estimate_shift_crosscorr",
    "compute_band_ratio_matrix",
    "find_monotonic_wavelengths",
]
