"""
src.preprocessing.normalization
================================
Spectrum normalization methods (area, min-max, z-score, SNV).
"""

from __future__ import annotations

import numpy as np


def normalize_minmax(intensities: np.ndarray) -> np.ndarray:
    """Scale intensities to [0, 1] range."""
    lo: float = float(np.min(intensities))
    hi: float = float(np.max(intensities))
    span = hi - lo
    if span < 1e-10:
        return np.zeros_like(intensities)
    return np.asarray((intensities - lo) / span)


def normalize_area(
    intensities: np.ndarray,
    wavelengths: np.ndarray | None = None,
) -> np.ndarray:
    """Divide by the trapezoidal area under the spectrum.

    Parameters
    ----------
    intensities:
        1-D intensity array.
    wavelengths:
        Optional 1-D wavelength axis.  When provided, integration uses the
        actual wavelength spacing (correct for non-uniform pixel grids).
        When ``None``, unit pixel spacing is assumed (legacy behaviour).
    """
    if wavelengths is not None:
        area = float(np.trapezoid(np.abs(intensities), x=wavelengths))
    else:
        area = float(np.trapezoid(np.abs(intensities)))
    if area < 1e-10:
        return np.zeros_like(intensities)
    return np.asarray(intensities / area)


def normalize_zscore(intensities: np.ndarray) -> np.ndarray:
    """Z-score standardization (mean=0, std=1)."""
    mu: float = float(np.mean(intensities))
    sigma: float = float(np.std(intensities))
    if sigma < 1e-10:
        return np.zeros_like(intensities)
    return np.asarray((intensities - mu) / sigma)


def normalize_snv(intensities: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) — identical to z-score for 1-D arrays."""
    return normalize_zscore(intensities)


def normalize_peak(intensities: np.ndarray) -> np.ndarray:
    """Divide by the absolute maximum value."""
    peak: float = float(np.max(np.abs(intensities)))
    if peak < 1e-10:
        return np.zeros_like(intensities)
    return np.asarray(intensities / peak)


def normalize_spectrum(
    intensities: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """Dispatch to the requested normalization method.

    Parameters
    ----------
    intensities:
        1-D intensity array.
    method:
        One of ``'minmax'``, ``'area'``, ``'zscore'``, ``'snv'``, ``'peak'``.

    Returns
    -------
    np.ndarray
        Normalized intensities.
    """
    if len(intensities) == 0:
        return intensities

    dispatch = {
        "minmax": normalize_minmax,
        "area": normalize_area,
        "zscore": normalize_zscore,
        "standard": normalize_zscore,
        "snv": normalize_snv,
        "peak": normalize_peak,
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown normalization method: {method!r}. Choose from: {list(dispatch)}")
    return fn(intensities)  # type: ignore[operator]
