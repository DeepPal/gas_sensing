"""ROI scanning utilities for LSPR spectral calibration.

Implements wavelength-region discovery: scanning for spectral positions
where peak/valley wavelength shifts monotonically with analyte concentration.
These routines are the first step in automatic calibration curve construction.
"""
from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr


def compute_band_ratio_matrix(Y: np.ndarray, half_width: int) -> np.ndarray:
    """Compute the left/right mean-signal ratio for every sample and wavelength pixel.

    For each pixel *j* in each row of *Y*, the ratio is::

        ratio[i, j] = mean(Y[i, j-half_width : j+1]) / mean(Y[i, j : j+half_width+1])

    This captures local spectral asymmetry and is used as a feature for
    ROI candidate scoring.

    Args:
        Y: Spectra matrix of shape ``(n_samples, n_wavelengths)``.
        half_width: Number of pixels on each side for the left/right means.
            Clipped to at least 1.

    Returns:
        Ratio matrix of the same shape as *Y*, dtype ``float64``.

    Example:
        >>> import numpy as np
        >>> Y = np.array([[1.0, 2.0, 3.0, 2.0, 1.0]])
        >>> r = compute_band_ratio_matrix(Y, half_width=1)
        >>> r.shape
        (1, 5)
    """
    n_samples, n_wl = Y.shape
    if n_wl == 0:
        return np.empty_like(Y)
    half_width = max(1, int(half_width))
    ratios = np.empty_like(Y, dtype=float)
    eps = 1e-12
    for i in range(n_samples):
        row = Y[i]
        for j in range(n_wl):
            left_start = max(0, j - half_width)
            left_end = j + 1
            right_start = j
            right_end = min(n_wl, j + half_width + 1)

            left_segment = row[left_start:left_end]
            right_segment = row[right_start:right_end]

            left_mean = float(np.mean(left_segment)) if left_segment.size else float(row[j])
            right_mean = float(np.mean(right_segment)) if right_segment.size else float(row[j])

            denom = right_mean if abs(right_mean) > eps else (eps if right_mean >= 0 else -eps)
            ratios[i, j] = left_mean / denom
    return ratios


def find_monotonic_wavelengths(
    canonical: dict[float, pd.DataFrame],
    min_wl: float,
    max_wl: float,
    signal_col: str = "intensity",
    min_r2: float = 0.5,
    min_spearman: float = 0.7,
) -> dict[str, Any]:
    """Scan ROI windows to find spectral regions where peak/valley position shifts
    monotonically with analyte concentration.

    Scans *both* peaks and valleys because:

    - For transmittance spectra: valleys (absorption dips) are often more significant.
    - For absorbance spectra: peaks (absorption maxima) are more significant.

    Peak/valley position is estimated by parabolic sub-pixel interpolation, falling
    back to weighted centroid at window boundaries.

    Candidate regions are scored by a composite metric::

        composite = 0.70 * R² + 0.20 * |Spearman ρ| + 0.10 * normalised_range

    where *normalised_range* = min(peak_range_nm / 0.5, 1.0).

    Args:
        canonical: Mapping of concentration (ppm) → DataFrame containing
            ``"wavelength"`` and at least one signal column.  Requires ≥ 3 entries.
        min_wl: Lower bound of the wavelength search range (nm).
        max_wl: Upper bound of the wavelength search range (nm).
        signal_col: Preferred signal column name.  Falls back to
            ``"transmittance"`` → ``"absorbance"`` → ``"intensity"`` if absent.
        min_r2: Minimum linear R² for a candidate to pass.
        min_spearman: Minimum |Spearman ρ| for a candidate to pass.

    Returns:
        Dict with keys:

        - ``"best_wavelength"`` – float | None
        - ``"best_feature_type"`` – ``"peak"`` | ``"valley"`` | None
        - ``"candidates"`` – top-20 passing candidates (list of dicts)
        - ``"best_peak"`` / ``"best_valley"`` – best of each type, or None
        - ``"total_scanned"`` – total (wavelength, window) combinations scanned
        - ``"passing_count"`` – candidates that passed both thresholds
        - ``"peak_candidates"`` / ``"valley_candidates"`` – counts in top-10
        - ``"signal_type"`` – actual signal column used
        - ``"preferred_feature"`` – ``"peak"`` | ``"valley"``
    """
    items = sorted(canonical.items(), key=lambda kv: kv[0])
    if len(items) < 3:
        return {"best_wavelength": None, "candidates": []}

    concs: np.ndarray = np.array([c for c, _ in items])

    ref_df = items[0][1]
    all_wl: np.ndarray = ref_df["wavelength"].values
    wl_mask = (all_wl >= min_wl) & (all_wl <= max_wl)
    candidate_wl: np.ndarray = all_wl[wl_mask]

    if len(candidate_wl) == 0:
        return {"best_wavelength": None, "candidates": []}

    actual_signal_col = signal_col
    for _, df in items:
        if signal_col not in df.columns:
            if "transmittance" in df.columns:
                actual_signal_col = "transmittance"
            elif "absorbance" in df.columns:
                actual_signal_col = "absorbance"
            elif "intensity" in df.columns:
                actual_signal_col = "intensity"
            break

    results: list[dict[str, Any]] = []
    window_sizes = [5.0, 8.0, 10.0, 12.0, 15.0, 20.0]  # nm

    for feature_type in ["peak", "valley"]:
        for center_wl in candidate_wl[::4]:  # subsample for performance
            for window_nm in window_sizes:
                half_window = window_nm / 2.0
                wl_low = center_wl - half_window
                wl_high = center_wl + half_window

                feature_wavelengths: list[float] = []
                for _, df in items:
                    df_wl: np.ndarray = df["wavelength"].values
                    if actual_signal_col not in df.columns:
                        continue
                    df_sig: np.ndarray = df[actual_signal_col].values

                    mask = (df_wl >= wl_low) & (df_wl <= wl_high)
                    if np.sum(mask) < 3:
                        feature_wavelengths.append(float("nan"))
                        continue

                    wl_window: np.ndarray = df_wl[mask]
                    sig_window: np.ndarray = df_sig[mask]

                    sig_range = float(sig_window.max() - sig_window.min())
                    if sig_range < 1e-9:
                        feature_wavelengths.append(float(np.mean(wl_window)))
                        continue

                    extremum_idx = int(
                        np.argmin(sig_window) if feature_type == "valley" else np.argmax(sig_window)
                    )

                    # Parabolic sub-pixel interpolation
                    if 0 < extremum_idx < len(sig_window) - 1:
                        y0 = float(sig_window[extremum_idx - 1])
                        y1 = float(sig_window[extremum_idx])
                        y2 = float(sig_window[extremum_idx + 1])
                        denom = 2.0 * (2.0 * y1 - y0 - y2)
                        if abs(denom) > 1e-12:
                            delta = float(np.clip((y0 - y2) / denom, -0.5, 0.5))
                            wl_step = float(wl_window[extremum_idx] - wl_window[extremum_idx - 1])
                            feature_wl = float(wl_window[extremum_idx]) + delta * wl_step
                        else:
                            feature_wl = float(wl_window[extremum_idx])
                    else:
                        # Boundary fallback: weighted centroid
                        if feature_type == "valley":
                            weights = (sig_window.max() - sig_window) + 1e-9
                        else:
                            weights = (sig_window - sig_window.min()) + 1e-9
                        feature_wl = float(np.sum(wl_window * weights) / np.sum(weights))

                    feature_wavelengths.append(feature_wl)

                if len(feature_wavelengths) != len(concs) or any(
                    np.isnan(feature_wavelengths)
                ):
                    continue

                feat_arr: np.ndarray = np.array(feature_wavelengths)
                feature_range = float(feat_arr.max() - feat_arr.min())
                if feature_range < 0.01:  # < 0.01 nm variation is noise
                    continue

                try:
                    spearman_r, _ = spearmanr(concs, feat_arr)
                    reg = linregress(concs, feat_arr)
                    r2 = float(reg.rvalue ** 2)
                    slope = float(reg.slope)

                    if abs(float(spearman_r)) >= min_spearman and r2 >= min_r2:
                        results.append(
                            {
                                "wavelength": float(center_wl),
                                "window_nm": float(window_nm),
                                "feature_type": feature_type,
                                "r2": r2,
                                "spearman": float(spearman_r),
                                "slope": slope,
                                "peak_range_nm": feature_range,
                                "peak_wavelengths": feat_arr.tolist(),
                                "direction": "red_shift" if slope > 0 else "blue_shift",
                            }
                        )
                except Exception:
                    continue

    # Composite scoring: 70% R², 20% |Spearman|, 10% normalised range
    for r in results:
        range_score = min(float(r["peak_range_nm"]) / 0.5, 1.0)
        r["composite_score"] = (
            0.7 * float(r["r2"]) + 0.2 * abs(float(r["spearman"])) + 0.1 * range_score
        )

    results.sort(key=lambda x: float(x["composite_score"]), reverse=True)  # type: ignore[arg-type]

    prefer_valleys = actual_signal_col == "absorbance"
    peaks_only = [r for r in results if r["feature_type"] == "peak"]
    valleys_only = [r for r in results if r["feature_type"] == "valley"]
    overall_best_r2 = float(results[0]["r2"]) if results else 0.0

    best_result: dict[str, Any] | None
    if prefer_valleys and valleys_only:
        best_result = (
            valleys_only[0]
            if float(valleys_only[0]["r2"]) >= overall_best_r2 * 0.95
            else results[0]
        )
    elif not prefer_valleys and peaks_only:
        best_result = (
            peaks_only[0]
            if float(peaks_only[0]["r2"]) >= overall_best_r2 * 0.95
            else results[0]
        )
    elif results:
        best_result = results[0]
    else:
        best_result = None

    best_wl: float | None = float(best_result["wavelength"]) if best_result else None
    best_feature_type: str | None = (
        str(best_result["feature_type"]) if best_result else None
    )

    top_10 = results[:10]
    peak_count = sum(1 for r in top_10 if r["feature_type"] == "peak")
    valley_count = sum(1 for r in top_10 if r["feature_type"] == "valley")

    return {
        "best_wavelength": best_wl,
        "best_feature_type": best_feature_type,
        "candidates": results[:20],
        "best_peak": peaks_only[0] if peaks_only else None,
        "best_valley": valleys_only[0] if valleys_only else None,
        "total_scanned": len(candidate_wl[::4]) * len(window_sizes) * 2,
        "passing_count": len(results),
        "peak_candidates": peak_count,
        "valley_candidates": valley_count,
        "signal_type": actual_signal_col,
        "preferred_feature": "valley" if prefer_valleys else "peak",
    }
