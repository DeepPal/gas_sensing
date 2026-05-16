"""
src.signal.transforms
======================
Spectral signal conversions and frame-level smoothing utilities.

These are pure functions operating on numpy arrays and pandas DataFrames.
They have no dependency on config files or I/O — safe to import anywhere.

Background
----------
The LSPR sensor workflow operates in three signal spaces:

* **Intensity** (raw photon counts from CCS200)
* **Transmittance** T = I_sample / I_ref  (range [0, 1])
* **Absorbance**   A = -log10(T)  (Beer-Lambert form, used for linearity tests)

The primary analytical signal is the LSPR peak wavelength *shift* (Δλ), not
absolute intensity — see ``src.features.lspr_features`` for that extraction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_transmittance(
    sample_df: pd.DataFrame,
    ref_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compute transmittance T = I_sample / I_ref on the sample wavelength grid.

    If ``ref_df`` is None or empty, or if the sample is not in intensity space,
    the sample DataFrame is returned unchanged.

    Args:
        sample_df: DataFrame with ``wavelength`` and ``intensity`` columns.
        ref_df: Reference DataFrame with ``wavelength`` and ``intensity`` columns.

    Returns:
        Copy of ``sample_df`` with an added ``transmittance`` column (clipped to [0, 1]).
    """
    if sample_df.empty:
        return sample_df
    if ref_df is None or ref_df.empty:
        return sample_df
    if "intensity" not in sample_df.columns:
        return sample_df

    ref_int = np.interp(
        sample_df["wavelength"].values,
        ref_df["wavelength"].values,
        ref_df["intensity"].values,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.clip(
            np.where(ref_int != 0, sample_df["intensity"].values / ref_int, 0.0),
            0.0,
            1.0,
        )
    out = sample_df.copy()
    out["transmittance"] = T
    return out


def append_absorbance_column(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Ensure an absorbance column A = -log10(T) exists when transmittance is available.

    Args:
        df: DataFrame that may contain a ``transmittance`` column.
        inplace: If True, modify ``df`` directly; otherwise return a copy.

    Returns:
        DataFrame with ``absorbance`` column added (or unchanged if no transmittance).
    """
    if "transmittance" not in df.columns:
        return df if inplace else df.copy()

    target = df if inplace else df.copy()
    trans = target["transmittance"].to_numpy(dtype=float, copy=True)
    trans = np.clip(trans, 1e-6, None)
    target["absorbance"] = -np.log10(trans)
    return target


def ensure_odd_window(window: int) -> int:
    """Return the smallest odd integer >= ``window`` (minimum 3).

    Savitzky-Golay and similar filters require odd window lengths.

    Args:
        window: Proposed window length (any positive integer).

    Returns:
        Odd integer, at least 3.
    """
    if window % 2 == 0:
        window += 1
    return max(window, 3)


def smooth(y: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing with automatic window clamping.

    Silently falls back to the original signal if fitting fails (e.g., if the
    array is shorter than the minimum window).

    Args:
        y: 1-D intensity array.
        window: Window length (will be forced odd, clamped to array length).
        poly: Polynomial order for the Savitzky-Golay filter.

    Returns:
        Smoothed 1-D array of the same shape as ``y``.
    """
    window = max(3, window if window % 2 == 1 else window + 1)
    window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    window = max(3, window)
    try:
        return np.asarray(
            savgol_filter(y, window_length=window, polyorder=min(poly, window - 1))
        )
    except Exception:
        return np.asarray(y)
