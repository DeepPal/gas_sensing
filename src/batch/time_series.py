"""
src.batch.time_series
======================
Response time-series computation for spectrometer-based sensor experiments.

Extracted from ``gas_analysis/core/pipeline.py`` (Phase 7c strangler-fig).

The core function :func:`compute_response_time_series` takes pre-computed
spectral matrices (from :mod:`src.batch.preprocessing`) and returns a
per-frame DataFrame with:

- ``delta_lambda_nm``        — peak wavelength shift from baseline (nm)
- ``is_responsive``          — 1 if the frame is in the sensor response region
- ``segment_id``             — changepoint-detected segment label (optional)
- auxiliary columns          — ``mean_signal``, ``delta_mean``, ``peak_wavelength_nm``, …

Changepoint detection
---------------------
Two methods are supported via ``changepoint_cfg``:

``"pelt"`` (default, recommended)
    Pruned Exact Linear Time (PELT) — O(n) exact changepoint detection using
    a quadratic cost function (sum of squared deviations from segment mean).
    Reference: Killick, R. et al. (2012). *J. Am. Stat. Assoc.*, 107, 1590–1598.

``"threshold"``
    Hysteresis-based threshold detection — simpler, but requires manual
    tuning of ``threshold_scale`` and ``release_multiplier``.

Public API
----------
- :func:`compute_response_time_series`
- :func:`smooth_vector`
- :func:`pelt_changepoint_detection`
- :func:`detect_segments_above_threshold`
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.signal.transforms import ensure_odd_window

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------


def smooth_vector(signal: np.ndarray, window: int = 3) -> np.ndarray:
    """Savitzky-Golay smooth a 1-D signal array.

    Parameters
    ----------
    signal:
        1-D float array of sensor signal values.
    window:
        Smoothing window size (coerced to valid odd integer ≥ 3).

    Returns
    -------
    np.ndarray
        Smoothed signal, same shape as input.
    """
    from scipy.signal import savgol_filter

    if len(signal) < 4:
        return signal.copy()
    w = min(ensure_odd_window(window), len(signal) if len(signal) % 2 == 1 else len(signal) - 1)
    w = max(3, w)
    p = min(2, w - 1)
    return np.asarray(savgol_filter(signal.astype(float), w, p))


def pelt_changepoint_detection(
    signal: np.ndarray,
    penalty: float = 3.0,
    min_size: int = 8,
) -> list[int]:
    """Pruned Exact Linear Time (PELT) changepoint detection.

    Uses a quadratic cost function (sum of squared deviations from segment
    mean), which is optimal for Gaussian-distributed noise.

    O(n) complexity through pruning: once a partition point's cost
    exceeds the running minimum, it can never become optimal and is pruned.

    Parameters
    ----------
    signal:
        1-D float signal (e.g. |Δλ| time series).
    penalty:
        BIC-like penalty per changepoint.  Higher → fewer changepoints.
        Typical range: 1–10.  Default 3.0 ≈ log(n) for n~20.
    min_size:
        Minimum segment length between consecutive changepoints.

    Returns
    -------
    list[int]
        Sorted list of changepoint indices (exclusive upper bounds of segments).
        Empty list = no changepoints detected.
    """
    n = len(signal)
    if n < 2 * min_size:
        return []

    sig = np.asarray(signal, dtype=float)

    # Precompute prefix sums for O(1) segment cost evaluation
    # cost(i, j) = sum_sq(i..j-1) - (sum(i..j-1))^2 / (j-i)
    cumsum = np.concatenate(([0.0], np.cumsum(sig)))
    cumsum_sq = np.concatenate(([0.0], np.cumsum(sig ** 2)))

    def _seg_cost(start: int, end: int) -> float:
        """Quadratic (variance) cost of segment [start, end)."""
        n_seg = end - start
        if n_seg <= 0:
            return 0.0
        s = cumsum[end] - cumsum[start]
        sq = cumsum_sq[end] - cumsum_sq[start]
        return float(sq - s * s / n_seg)

    # F[t] = min cost of optimal partition of signal[0:t]
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    prev = [-1] * (n + 1)
    # Candidate set of last changepoint positions
    candidates: list[int] = [0]

    for t in range(min_size, n + 1):
        best_cost = np.inf
        best_cp = -1
        surviving: list[int] = []

        for cp in candidates:
            if t - cp < min_size:
                surviving.append(cp)
                continue
            cost = F[cp] + _seg_cost(cp, t) + penalty
            if cost < best_cost:
                best_cost = cost
                best_cp = cp

            # Pruning: if F[cp] + cost(cp, t) is already worse than best
            # and will only get worse as t grows, discard cp.
            if F[cp] + _seg_cost(cp, t) <= best_cost - penalty:
                surviving.append(cp)

        F[t] = best_cost
        prev[t] = best_cp
        surviving.append(t)
        candidates = surviving

    # Backtrack to recover changepoints
    changepoints: list[int] = []
    t = n
    while t > 0:
        cp = prev[t]
        if cp > 0:
            changepoints.append(cp)
        t = cp
    changepoints.sort()
    return changepoints


def detect_segments_above_threshold(
    signal: np.ndarray,
    high_threshold: float,
    low_threshold: float,
    min_length: int = 4,
    pad: int = 1,
) -> list[tuple[int, int]]:
    """Hysteresis-based segment detection.

    A segment starts when ``signal`` rises above ``high_threshold`` and ends
    when it falls below ``low_threshold``.  Segments shorter than
    ``min_length`` are discarded.  Each segment is expanded by ``pad``
    frames on each side.

    Parameters
    ----------
    signal:
        1-D float array.
    high_threshold:
        Onset threshold — segment begins when signal exceeds this.
    low_threshold:
        Release threshold — segment ends when signal drops below this.
        Should satisfy ``low_threshold ≤ high_threshold``.
    min_length:
        Minimum segment duration (frames).
    pad:
        Frames to add on each side of each detected segment.

    Returns
    -------
    list[tuple[int, int]]
        List of ``(start, end)`` index pairs (inclusive).
    """
    sig = np.asarray(signal, dtype=float)
    n = len(sig)
    segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0

    for i in range(n):
        if not in_segment:
            if np.isfinite(sig[i]) and sig[i] >= high_threshold:
                in_segment = True
                start = i
        else:
            if not np.isfinite(sig[i]) or sig[i] < low_threshold:
                end = i - 1
                if end - start + 1 >= min_length:
                    segments.append((max(0, start - pad), min(n - 1, end + pad)))
                in_segment = False

    if in_segment:
        end = n - 1
        if end - start + 1 >= min_length:
            segments.append((max(0, start - pad), min(n - 1, end + pad)))

    return segments


# ---------------------------------------------------------------------------
# Baseline computation (inner helper extracted from pipeline.py)
# ---------------------------------------------------------------------------


def _compute_baseline_outputs(
    absorb_matrix: np.ndarray,
    roi_matrix: np.ndarray,
    mean_absorb: np.ndarray,
    roi_wavelengths: np.ndarray,
    baseline_indices: list[int],
    n_frames: int,
) -> tuple[
    np.ndarray,  # baseline_ref (full spectrum)
    np.ndarray,  # roi_baseline_ref
    float,       # baseline_mean_val
    float,       # baseline_std_val
    np.ndarray,  # centered_matrix
    np.ndarray,  # delta_mean
    np.ndarray,  # roi_delta_matrix
    np.ndarray,  # peak_wavelengths per frame
    float,       # baseline_peak_nm
    np.ndarray,  # delta_lambda per frame
    np.ndarray,  # abs_delta_lambda per frame
    float,       # lambda_sigma
]:
    """Compute baseline statistics and per-frame Δλ from a baseline window.

    Parameters
    ----------
    absorb_matrix:
        (n_frames, n_wl) array of signal values.
    roi_matrix:
        (n_frames, n_roi_wl) array of signal values in the ROI.
    mean_absorb:
        (n_frames,) mean signal per frame.
    roi_wavelengths:
        (n_roi_wl,) wavelength axis of the ROI.
    baseline_indices:
        Frame indices used to compute the baseline reference.
    n_frames:
        Total number of frames.
    """
    valid = [idx for idx in baseline_indices if 0 <= idx < n_frames]
    if not valid:
        valid = [0]

    base_matrix = absorb_matrix[valid]
    base_roi_matrix = roi_matrix[valid]
    baseline_ref = np.nanmean(base_matrix, axis=0)
    roi_baseline_ref = np.nanmean(base_roi_matrix, axis=0)
    baseline_mean_val = float(np.nanmean(mean_absorb[valid]))
    baseline_std_val = float(np.nanstd(mean_absorb[valid], ddof=1)) if len(valid) > 1 else 0.0
    baseline_std_val = float(np.nan_to_num(baseline_std_val, nan=0.0))

    centered = absorb_matrix - baseline_ref
    delta_mean_local = mean_absorb - baseline_mean_val
    roi_delta_local = roi_matrix - roi_baseline_ref

    # Determine whether ROI extremum is a valley or peak
    extremum_mode = "valley"
    if np.any(np.isfinite(roi_baseline_ref)):
        roi_finite = roi_baseline_ref[np.isfinite(roi_baseline_ref)]
        if roi_finite.size:
            median_val = float(np.nanmedian(roi_finite))
            min_val = float(np.nanmin(roi_finite))
            max_val = float(np.nanmax(roi_finite))
            if (max_val - median_val) > (median_val - min_val):
                extremum_mode = "peak"

    if roi_wavelengths.size:
        if extremum_mode == "valley":
            baseline_peak_idx = int(
                np.nanargmin(
                    np.where(np.isfinite(roi_baseline_ref), roi_baseline_ref, np.inf)
                )
            )
        else:
            baseline_peak_idx = int(
                np.nanargmax(
                    np.where(np.isfinite(roi_baseline_ref), roi_baseline_ref, -np.inf)
                )
            )
    else:
        baseline_peak_idx = 0
    baseline_peak_idx = int(np.clip(baseline_peak_idx, 0, max(0, roi_wavelengths.size - 1)))

    wl_step = float(np.nanmedian(np.diff(roi_wavelengths))) if roi_wavelengths.size > 1 else 0.2
    wl_step = wl_step if np.isfinite(wl_step) and wl_step > 0 else 0.2
    search_radius = int(max(2, math.ceil(1.5 / wl_step)))

    peak_idx = np.full(n_frames, baseline_peak_idx, dtype=int)
    for frame_idx, row in enumerate(roi_matrix):
        if not np.any(np.isfinite(row)):
            continue
        start = max(0, baseline_peak_idx - search_radius)
        end = min(row.size, baseline_peak_idx + search_radius + 1)
        window = row[start:end].astype(float)
        if window.size == 0:
            continue
        if extremum_mode == "valley":
            window[~np.isfinite(window)] = np.inf
            local_idx = int(np.argmin(window))
        else:
            window[~np.isfinite(window)] = -np.inf
            local_idx = int(np.argmax(window))
        peak_idx[frame_idx] = start + local_idx

    peak_idx = np.clip(peak_idx, 0, max(0, roi_wavelengths.size - 1))
    peak_wls = (
        roi_wavelengths[peak_idx]
        if roi_wavelengths.size
        else np.full(n_frames, float("nan"))
    )

    baseline_peak_val = (
        float(peak_wls[valid].mean())
        if len(valid) > 0 and np.any(np.isfinite(peak_wls[valid]))
        else (
            float(roi_wavelengths[baseline_peak_idx]) if roi_wavelengths.size else float("nan")
        )
    )
    delta_lambda_local = peak_wls - baseline_peak_val
    abs_delta_lambda_local = np.abs(delta_lambda_local)
    baseline_delta_local = delta_lambda_local[valid]
    lambda_sigma_local = (
        float(np.nanstd(baseline_delta_local, ddof=1))
        if baseline_delta_local.size > 1
        else 0.0
    )
    lambda_sigma_local = float(np.nan_to_num(lambda_sigma_local, nan=0.0))

    return (
        baseline_ref,
        roi_baseline_ref,
        baseline_mean_val,
        baseline_std_val,
        centered,
        delta_mean_local,
        roi_delta_local,
        peak_wls,
        baseline_peak_val,
        delta_lambda_local,
        abs_delta_lambda_local,
        lambda_sigma_local,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_response_time_series(
    absorb_matrix: np.ndarray,
    roi_matrix: np.ndarray,
    mean_absorb: np.ndarray,
    roi_wavelengths: np.ndarray,
    roi_min_nm: float,
    roi_max_nm: float,
    n_frames: int,
    dataset_label: str | None = None,
    *,
    smooth_window: int = 5,
    baseline_frames: int = 12,
    activation_delta: float = 0.01,
    sigma_multiplier: float = 1.5,
    noise_floor: float = 1e-4,
    slope_sigma_multiplier: float = 1.0,
    min_response_slope: float = 0.0,
    min_activation_frames: int = 6,
    min_activation_fraction: float = 0.08,
    fallback_window: int = 4,
    monotonic_tolerance_nm: float = 0.05,
    changepoint_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Compute per-frame response metrics and identify responsive frames.

    Parameters
    ----------
    absorb_matrix:
        ``(n_frames, n_wavelengths)`` array of signal values (absorbance or
        differential intensity) after Savitzky-Golay smoothing.
    roi_matrix:
        ``(n_frames, n_roi_wavelengths)`` subset of *absorb_matrix* within
        the ROI wavelength window.
    mean_absorb:
        ``(n_frames,)`` mean signal per frame.
    roi_wavelengths:
        ``(n_roi_wavelengths,)`` wavelength axis for the ROI.
    roi_min_nm, roi_max_nm:
        Wavelength bounds of the ROI (used as metadata in the output).
    n_frames:
        Total number of frames (== ``absorb_matrix.shape[0]``).
    dataset_label:
        String identifier for this dataset (stored in output DataFrame).
    smooth_window:
        Savitzky-Golay smoothing window applied to input matrices.
    baseline_frames:
        Number of initial frames used to compute the baseline reference.
    activation_delta:
        Minimum |Δλ| (nm) to classify a frame as responsive.
    sigma_multiplier:
        Scales baseline Δλ noise (λ_sigma) to set the activation threshold.
        ``threshold = activation_delta + sigma_multiplier * max(λ_sigma, noise_floor)``
    noise_floor:
        Minimum noise floor for threshold computation (nm).
    slope_sigma_multiplier, min_response_slope:
        Parameters for the monotonic slope gate — filters out frames whose
        Δλ trend has the wrong sign or insufficient slope.
    min_activation_frames, min_activation_fraction:
        Minimum number / fraction of frames that must be responsive.
    fallback_window:
        If no responsive frames found, return first *fallback_window* frames.
    monotonic_tolerance_nm:
        Tolerance for monotonicity check (nm).  Small reversals within this
        tolerance are accepted.
    changepoint_cfg:
        Optional dict configuring changepoint-based segmentation.  Keys:
        ``"enabled"`` (bool), ``"method"`` (``"pelt"`` or ``"threshold"``),
        ``"signal"`` (``"abs_delta_lambda"`` | ``"delta_lambda"`` | ``"delta_mean"``),
        ``"smooth_window"`` (int), ``"penalty"`` (float, PELT only),
        ``"min_segment_size"`` (int), ``"threshold_scale"`` (float),
        ``"release_multiplier"`` (float), ``"min_length"`` (int), ``"pad"`` (int).

    Returns
    -------
    tuple of:
        - ``df_series``          — per-frame DataFrame
        - ``responsive_indices`` — frame indices in the sensor response region
        - ``roi_only_indices``   — same, before monotonicity/slope gating
    """
    cpd_cfg: dict[str, Any] = changepoint_cfg or {}

    # ── Apply smoothing to matrices if requested ──────────────────────────
    if smooth_window > 1:
        from scipy.signal import savgol_filter

        w = ensure_odd_window(smooth_window)
        absorb_matrix = savgol_filter(
            absorb_matrix,
            window_length=min(w, max(3, absorb_matrix.shape[1] - (absorb_matrix.shape[1] + 1) % 2)),
            polyorder=2,
            axis=1,
            mode="nearest",
        )
        if roi_matrix.shape[1] >= 4:
            w_roi = ensure_odd_window(smooth_window)
            roi_matrix = savgol_filter(
                roi_matrix,
                window_length=min(
                    w_roi,
                    max(3, roi_matrix.shape[1] - (roi_matrix.shape[1] + 1) % 2),
                ),
                polyorder=2,
                axis=1,
                mode="nearest",
            )

    # ── Baseline window ───────────────────────────────────────────────────
    baseline_target = max(1, min(baseline_frames, n_frames))
    baseline_indices = list(range(min(baseline_target, n_frames)))
    if not baseline_indices:
        baseline_indices = [0]

    (
        _baseline_reference,
        _roi_baseline_reference,
        baseline_mean_abs,
        baseline_std_abs,
        _centered_matrix,
        delta_mean,
        _roi_delta_matrix,
        peak_wavelengths,
        baseline_peak_nm,
        delta_lambda,
        abs_delta_lambda,
        lambda_sigma,
    ) = _compute_baseline_outputs(
        absorb_matrix, roi_matrix, mean_absorb, roi_wavelengths, baseline_indices, n_frames
    )

    # ── Threshold and direction ───────────────────────────────────────────
    threshold = activation_delta + sigma_multiplier * max(lambda_sigma, noise_floor)
    slope_threshold = max(
        min_response_slope, slope_sigma_multiplier * max(lambda_sigma, noise_floor)
    )
    direction = (
        float(np.sign(np.nanmedian(delta_lambda[np.isfinite(delta_lambda)])))
        if np.any(np.isfinite(delta_lambda))
        else 1.0
    )
    if not np.isfinite(direction) or direction == 0.0:
        direction = 1.0

    responsive_indices = [
        idx for idx, val in enumerate(abs_delta_lambda) if np.isfinite(val) and val >= threshold
    ]
    responsive_segments: list[tuple[int, int]] = []

    # ── Optional changepoint detection ────────────────────────────────────
    if cpd_cfg.get("enabled", False):
        cp_signal_mode = str(cpd_cfg.get("signal", "abs_delta_lambda")).lower()
        if cp_signal_mode == "delta_lambda":
            change_signal = np.copy(delta_lambda)
        elif cp_signal_mode == "delta_mean":
            change_signal = np.copy(delta_mean)
        else:
            change_signal = np.copy(abs_delta_lambda)

        cp_smooth_win = int(cpd_cfg.get("smooth_window", 3) or 3)
        if cp_smooth_win > 1:
            change_signal = smooth_vector(change_signal, cp_smooth_win)

        cp_method = str(cpd_cfg.get("method", "pelt")).lower()
        min_seg_size = int(cpd_cfg.get("min_segment_size", 8) or 8)

        if cp_method == "pelt":
            penalty = float(cpd_cfg.get("penalty", 3.0) or 3.0)
            cps = pelt_changepoint_detection(change_signal, penalty=penalty, min_size=min_seg_size)
            segments: list[tuple[int, int]] = []
            boundaries = [0] + cps + [len(change_signal)]
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if end - start >= min_seg_size:
                    seg_mean = float(np.nanmean(change_signal[start:end]))
                    if seg_mean >= threshold * 0.8:
                        segments.append((start, end - 1))
            responsive_segments = segments
        else:
            scale = float(cpd_cfg.get("threshold_scale", 1.0) or 1.0)
            release_mult = float(cpd_cfg.get("release_multiplier", 0.6) or 0.6)
            min_len = int(cpd_cfg.get("min_length", 4) or 4)
            pad_frames = int(cpd_cfg.get("pad", 1) or 1)
            cp_high = threshold * scale
            cp_low = cp_high * release_mult
            responsive_segments = detect_segments_above_threshold(
                change_signal, cp_high, cp_low, min_len, pad_frames
            )

        if responsive_segments:
            responsive_indices = sorted(
                {
                    idx
                    for start, end in responsive_segments
                    for idx in range(start, end + 1)
                    if 0 <= idx < n_frames
                }
            )

    roi_only_indices = list(responsive_indices)

    # ── Refine baseline once response onset is known ──────────────────────
    if responsive_indices:
        first_resp = min(responsive_indices)
        trimmed = [idx for idx in baseline_indices if idx < first_resp]
        if len(trimmed) >= max(1, baseline_target // 2):
            baseline_indices = trimmed
        elif first_resp > 0:
            start_bl = max(0, first_resp - baseline_target)
            baseline_indices = list(range(start_bl, first_resp)) or [max(0, first_resp - 1)]

        (
            _baseline_reference,
            _roi_baseline_reference,
            baseline_mean_abs,
            baseline_std_abs,
            _centered_matrix,
            delta_mean,
            _roi_delta_matrix,
            peak_wavelengths,
            baseline_peak_nm,
            delta_lambda,
            abs_delta_lambda,
            lambda_sigma,
        ) = _compute_baseline_outputs(
            absorb_matrix, roi_matrix, mean_absorb, roi_wavelengths, baseline_indices, n_frames
        )
        direction = (
            float(np.sign(np.nanmedian(delta_lambda[np.isfinite(delta_lambda)])))
            if np.any(np.isfinite(delta_lambda))
            else 1.0
        )
        if not np.isfinite(direction) or direction == 0.0:
            direction = 1.0
        threshold = activation_delta + sigma_multiplier * max(lambda_sigma, noise_floor)
        responsive_indices = [
            idx for idx, val in enumerate(abs_delta_lambda) if np.isfinite(val) and val >= threshold
        ]

    roi_only_indices = list(responsive_indices)

    # ── Minimum activation size guard ─────────────────────────────────────
    required = max(
        min_activation_frames, int(math.ceil(min_activation_fraction * n_frames))
    )
    if responsive_indices and len(responsive_indices) < required:
        top_idx = np.argsort(abs_delta_lambda)[::-1][:required]
        responsive_indices = sorted(set(list(responsive_indices) + top_idx.tolist()))

    if not responsive_indices and fallback_window > 0:
        responsive_indices = list(range(min(n_frames, fallback_window)))

    # ── Monotonic slope gate ──────────────────────────────────────────────
    slope_threshold = max(
        min_response_slope, slope_sigma_multiplier * max(lambda_sigma, noise_floor)
    )
    selected_slope = float("nan")

    if responsive_indices:
        sorted_resp = sorted(responsive_indices)
        monotonic_indices = list(sorted_resp)
        if len(sorted_resp) >= 2:
            signed_trace = delta_lambda[sorted_resp] * direction
            diffs = np.diff(signed_trace)
            negative_diffs = np.where(diffs < -monotonic_tolerance_nm)[0]
            if negative_diffs.size > 0:
                cutoff = negative_diffs[0] + 1
                trimmed_m = sorted_resp[:cutoff]
                if trimmed_m:
                    monotonic_indices = trimmed_m

        min_len_for_slope = max(3, min_activation_frames)
        candidate_indices = list(monotonic_indices)
        slope_pass = False
        while candidate_indices and len(candidate_indices) >= min_len_for_slope:
            candidate_lambda = delta_lambda[candidate_indices]
            slope_val = (
                linregress(candidate_indices, candidate_lambda).slope
                if len(candidate_indices) >= 2
                else 0.0
            )
            if not np.isfinite(slope_val):
                slope_val = 0.0
            slope_along_dir = slope_val * direction
            selected_slope = slope_along_dir
            if slope_along_dir >= slope_threshold:
                slope_pass = True
                break
            candidate_indices = candidate_indices[1:]
        responsive_indices = candidate_indices if slope_pass else []

    # ── Build output DataFrame ────────────────────────────────────────────
    segment_ids = np.full(n_frames, -1, dtype=int)
    if responsive_segments:
        for seg_id, (seg_start, seg_end) in enumerate(responsive_segments, start=1):
            seg_start = int(max(0, seg_start))
            seg_end = int(min(n_frames - 1, seg_end))
            segment_ids[seg_start : seg_end + 1] = seg_id

    records = []
    for idx, mean_val in enumerate(mean_absorb):
        records.append(
            {
                "frame_index": idx,
                "mean_signal": float(mean_val),
                "delta_mean": float(delta_mean[idx]),
                "delta_lambda_nm": float(delta_lambda[idx])
                if np.isfinite(delta_lambda[idx])
                else float("nan"),
                "delta_lambda_abs_nm": float(abs_delta_lambda[idx])
                if np.isfinite(abs_delta_lambda[idx])
                else float("nan"),
                "is_responsive": 1 if idx in responsive_indices else 0,
                "peak_wavelength_nm": float(peak_wavelengths[idx])
                if np.isfinite(peak_wavelengths[idx])
                else float("nan"),
                "segment_id": int(segment_ids[idx]) if segment_ids.size else -1,
            }
        )

    df_series = pd.DataFrame(records)
    df_series["dataset_label"] = dataset_label
    df_series["threshold_nm"] = threshold
    df_series["baseline_mean"] = baseline_mean_abs
    df_series["baseline_std"] = baseline_std_abs
    df_series["baseline_std_abs"] = baseline_std_abs
    df_series["baseline_std_delta_lambda_nm"] = lambda_sigma
    df_series["baseline_peak_nm"] = baseline_peak_nm
    df_series["slope_threshold"] = slope_threshold
    df_series["responsive_slope"] = selected_slope
    df_series["baseline_frames"] = len(baseline_indices)
    df_series["activation_delta_nm"] = activation_delta
    df_series["roi_min_nm"] = float(roi_min_nm)
    df_series["roi_max_nm"] = float(roi_max_nm)
    df_series["response_direction"] = direction
    df_series["baseline_indices"] = [baseline_indices] * len(df_series)
    if responsive_segments:
        df_series["responsive_segments"] = [responsive_segments] * len(df_series)

    log.info(
        "Response time series: %d/%d frames responsive, baseline_peak=%.3f nm, "
        "threshold=%.4f nm, slope=%.4f",
        len(responsive_indices),
        n_frames,
        baseline_peak_nm,
        threshold,
        selected_slope if np.isfinite(selected_slope) else 0.0,
    )

    return df_series, list(responsive_indices), list(roi_only_indices)
