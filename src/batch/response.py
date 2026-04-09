"""Batch response-analysis utilities — delta summarisation and aggregation.

All functions are CONFIG-free: every parameter is passed explicitly.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.reporting.metrics import select_signal_column


def _safe_float(val: object) -> float:
    try:
        fval = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return fval if np.isfinite(fval) else float("nan")


def scale_reference_to_baseline(
    ref_df: Optional[pd.DataFrame],
    baseline_frames: Sequence[pd.DataFrame],
    percentile: float = 95.0,
) -> tuple[Optional[pd.DataFrame], float]:
    """Scale a reference spectrum so that it matches a trial's baseline intensity."""
    if ref_df is None or not baseline_frames:
        return ref_df, 1.0

    if "wavelength" not in ref_df.columns or "intensity" not in ref_df.columns:
        return ref_df, 1.0

    ref_wl = ref_df["wavelength"].to_numpy(dtype=float)
    ref_int = ref_df["intensity"].to_numpy(dtype=float)
    if ref_wl.size == 0 or ref_int.size == 0:
        return ref_df, 1.0

    baseline_vals: list[float] = []
    for frame in baseline_frames:
        if frame is None or frame.empty:
            continue
        if "wavelength" not in frame.columns or "intensity" not in frame.columns:
            continue
        frame_wl = frame["wavelength"].to_numpy(dtype=float)
        frame_int = frame["intensity"].to_numpy(dtype=float)
        if frame_wl.size == 0 or frame_int.size == 0:
            continue
        interp = np.interp(ref_wl, frame_wl, frame_int)
        baseline_vals.append(float(np.percentile(interp, percentile)))

    if not baseline_vals:
        return ref_df, 1.0

    baseline_target = float(np.nanmean(baseline_vals))
    ref_percentile = float(np.percentile(ref_int, percentile))
    if ref_percentile <= 0 or not np.isfinite(baseline_target):
        return ref_df, 1.0

    scale_factor = baseline_target / ref_percentile
    if scale_factor <= 0 or not np.isfinite(scale_factor):
        return ref_df, 1.0

    scaled = ref_df.copy(deep=True)
    scaled["intensity"] = scaled["intensity"].astype(float) * scale_factor
    return scaled, float(scale_factor)


def score_trial_quality(
    df: pd.DataFrame,
    *,
    roi_bounds: tuple[Optional[float], Optional[float]],
    expected_center: Optional[float],
) -> tuple[float, dict[str, float]]:
    """Return a 0–1 quality score using SNR, contrast, and peak alignment."""
    details: dict[str, float] = {}
    if df is None or df.empty or "wavelength" not in df.columns:
        return 0.0, details

    wl = df["wavelength"].to_numpy(dtype=float)
    signal_col = select_signal_column(df)
    signal = df[signal_col].to_numpy(dtype=float)

    min_wl, max_wl = roi_bounds
    if min_wl is None:
        min_wl = float(np.nanmin(wl))
    if max_wl is None:
        max_wl = float(np.nanmax(wl))
    mask = (wl >= min_wl) & (wl <= max_wl)
    if not np.any(mask):
        mask = np.ones_like(wl, dtype=bool)

    roi_signal = signal[mask]
    roi_wl = wl[mask]
    if roi_signal.size == 0:
        return 0.0, details

    baseline_window = min(len(signal), 200)
    baseline_noise = float(np.nanstd(signal[:baseline_window], ddof=1)) if baseline_window else 0.0
    if not np.isfinite(baseline_noise) or baseline_noise <= 0:
        baseline_noise = 1e-3

    peak_idx = int(np.nanargmax(roi_signal)) if np.any(np.isfinite(roi_signal)) else 0
    peak_val = float(roi_signal[peak_idx]) if roi_signal.size else 0.0
    snr = peak_val / baseline_noise
    details["snr"] = float(snr)

    snr_score = 0.0
    if snr >= 5:
        snr_score = 0.5
    elif snr >= 3:
        snr_score = 0.35
    elif snr >= 2:
        snr_score = 0.2
    elif snr >= 1:
        snr_score = 0.1

    contrast = float(np.nanmax(roi_signal) - np.nanmin(roi_signal)) if roi_signal.size else 0.0
    details["contrast"] = contrast
    contrast_score = 0.0
    if contrast >= 0.5:
        contrast_score = 0.3
    elif contrast >= 0.25:
        contrast_score = 0.2
    elif contrast >= 0.1:
        contrast_score = 0.1

    if expected_center is None:
        expected_center = float((min_wl + max_wl) / 2.0)
    peak_wl = float(roi_wl[peak_idx]) if roi_wl.size else expected_center
    shift = abs(peak_wl - expected_center)
    details["peak_wavelength"] = peak_wl
    details["expected_center"] = expected_center
    details["shift_nm"] = shift
    shift_score = max(0.0, 0.2 - min(shift, 1.0) * 0.2)

    total_score = min(1.0, snr_score + contrast_score + shift_score)
    details["total_score"] = total_score
    return total_score, details


def summarize_responsive_delta(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}

    try:
        delta_series = pd.to_numeric(df.get("delta_lambda_nm"), errors="coerce")
    except Exception:
        delta_series = pd.Series(dtype=float)

    if "is_responsive" in df.columns:
        responsive_mask = (
            pd.to_numeric(df["is_responsive"], errors="coerce").fillna(0).astype(int) == 1
        )
    else:
        responsive_mask = pd.Series([False] * len(df))

    segment_ids = None
    try:
        if "segment_id" in df.columns:
            segment_ids = pd.to_numeric(df["segment_id"], errors="coerce").fillna(-1).astype(int)
    except Exception:
        segment_ids = None

    segment_definitions: Optional[list[tuple[int, int]]] = None
    if "responsive_segments" in df.columns and len(df.index) > 0:
        try:
            raw_segments = df["responsive_segments"].iloc[0]
            if isinstance(raw_segments, list):
                segment_definitions = []
                for entry in raw_segments:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        try:
                            start = int(entry[0])
                            end = int(entry[1])
                        except (TypeError, ValueError):
                            continue
                        if start <= end:
                            segment_definitions.append((start, end))
        except Exception:
            segment_definitions = None

    delta_values = (
        delta_series.to_numpy(dtype=float) if delta_series.size else np.array([], dtype=float)
    )
    responsive_values = (
        delta_series[responsive_mask].to_numpy(dtype=float)
        if delta_series.size
        else np.array([], dtype=float)
    )
    responsive_finite = responsive_values[np.isfinite(responsive_values)]
    all_finite = delta_values[np.isfinite(delta_values)]

    # Derive segments if definitions missing but segment ids present
    if segment_definitions is None and segment_ids is not None and segment_ids.size:
        segment_definitions = []
        current_id = None
        start_idx = None
        for idx, seg_id in enumerate(segment_ids):
            if seg_id > 0:
                if current_id != seg_id:
                    if current_id is not None and start_idx is not None:
                        segment_definitions.append((start_idx, idx - 1))
                    current_id = seg_id
                    start_idx = idx
            else:
                if current_id is not None and start_idx is not None:
                    segment_definitions.append((start_idx, idx - 1))
                current_id = None
                start_idx = None
        if current_id is not None and start_idx is not None:
            segment_definitions.append((start_idx, len(segment_ids) - 1))
        if not segment_definitions:
            segment_definitions = None

    baseline_peak_nm = float("nan")
    if "baseline_peak_nm" in df.columns:
        try:
            baseline_vals = pd.to_numeric(df["baseline_peak_nm"], errors="coerce").dropna()
            if not baseline_vals.empty:
                baseline_peak_nm = float(baseline_vals.iloc[0])
        except Exception:
            baseline_peak_nm = float("nan")

    responsive_frame_count = int(responsive_mask.sum()) if hasattr(responsive_mask, "sum") else 0
    total_frame_count = int(len(df))
    responsive_fraction = float(responsive_frame_count / max(1, total_frame_count))

    median_delta = float("nan")
    mean_delta = float("nan")
    std_delta = float("nan")
    max_abs_delta = float("nan")
    signed_consistency = float("nan")
    if responsive_finite.size:
        median_delta = float(np.nanmedian(responsive_finite))
        mean_delta = float(np.nanmean(responsive_finite))
        std_delta = (
            float(np.nanstd(responsive_finite, ddof=1)) if responsive_finite.size > 1 else 0.0
        )
        idx_max = int(np.abs(responsive_finite).argmax())
        max_abs_delta = float(responsive_finite[idx_max])
        total = responsive_finite.size
        if total > 0:
            same_sign = float(np.sum(np.sign(responsive_finite) == np.sign(median_delta)))
            signed_consistency = same_sign / total if total else float("nan")
    elif all_finite.size:
        mean_delta = float(np.nanmean(all_finite))
        std_delta = float(np.nanstd(all_finite, ddof=1)) if all_finite.size > 1 else 0.0
        idx_max = int(np.abs(all_finite).argmax())
        max_abs_delta = float(all_finite[idx_max])

    fallback_delta = float("nan")
    if all_finite.size:
        idx_fb = int(np.abs(all_finite).argmax())
        fallback_delta = float(all_finite[idx_fb])

    selected_delta = median_delta if np.isfinite(median_delta) else fallback_delta
    direction = (
        float(np.sign(selected_delta))
        if np.isfinite(selected_delta) and selected_delta != 0
        else float("nan")
    )

    median_peak_nm = (
        baseline_peak_nm + median_delta
        if np.isfinite(baseline_peak_nm) and np.isfinite(median_delta)
        else float("nan")
    )
    mean_peak_nm = (
        baseline_peak_nm + mean_delta
        if np.isfinite(baseline_peak_nm) and np.isfinite(mean_delta)
        else float("nan")
    )
    selected_peak_nm = (
        baseline_peak_nm + selected_delta
        if np.isfinite(baseline_peak_nm) and np.isfinite(selected_delta)
        else float("nan")
    )

    segment_count = 0
    segment_lengths: list[int] = []
    segment_coverage = float("nan")
    if segment_definitions:
        segment_count = len(segment_definitions)
        segment_lengths = [max(0, int(end) - int(start) + 1) for start, end in segment_definitions]
        total_len = sum(segment_lengths)
        if total_frame_count > 0:
            segment_coverage = total_len / float(total_frame_count)

    return {
        "responsive_frame_count": responsive_frame_count,
        "total_frame_count": total_frame_count,
        "responsive_fraction": responsive_fraction,
        "responsive_finite_count": int(responsive_finite.size),
        "median_delta_nm": median_delta if np.isfinite(median_delta) else float("nan"),
        "mean_delta_nm": mean_delta if np.isfinite(mean_delta) else float("nan"),
        "selected_delta_nm": selected_delta if np.isfinite(selected_delta) else float("nan"),
        "std_delta_nm": std_delta if np.isfinite(std_delta) else float("nan"),
        "max_abs_delta_nm": max_abs_delta if np.isfinite(max_abs_delta) else float("nan"),
        "fallback_delta_nm": fallback_delta if np.isfinite(fallback_delta) else float("nan"),
        "median_peak_nm": median_peak_nm if np.isfinite(median_peak_nm) else float("nan"),
        "mean_peak_nm": mean_peak_nm if np.isfinite(mean_peak_nm) else float("nan"),
        "selected_peak_nm": selected_peak_nm if np.isfinite(selected_peak_nm) else float("nan"),
        "baseline_peak_nm": baseline_peak_nm if np.isfinite(baseline_peak_nm) else float("nan"),
        "direction": direction if np.isfinite(direction) else float("nan"),
        "signed_consistency": signed_consistency
        if np.isfinite(signed_consistency)
        else float("nan"),
        "responsive_segment_count": float(segment_count),
        "responsive_segment_lengths": segment_lengths,
        "responsive_segment_coverage": segment_coverage
        if np.isfinite(segment_coverage)
        else float("nan"),
    }


def aggregate_responsive_delta_maps(
    responsive_delta_by_conc: dict[float, dict[str, dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, dict[str, float]]]:
    if not responsive_delta_by_conc:
        empty_df = pd.DataFrame()
        return empty_df, empty_df.copy(), {}

    rows_trial: list[dict[str, object]] = []
    rows_conc: list[dict[str, object]] = []
    summary_by_conc: dict[float, dict[str, Any]] = {}

    for conc, trial_map in sorted(responsive_delta_by_conc.items(), key=lambda kv: kv[0]):
        if not isinstance(trial_map, dict) or not trial_map:
            continue

        trial_count = 0
        responsive_trial_count = 0
        total_responsive_frames = 0
        total_frames = 0

        selected_delta_vals: list[float] = []
        median_delta_vals: list[float] = []
        mean_delta_vals: list[float] = []
        selected_peak_vals: list[float] = []
        baseline_peak_vals: list[float] = []
        responsive_fraction_vals: list[float] = []
        direction_vals: list[float] = []
        segment_count_vals: list[float] = []
        segment_coverage_vals: list[float] = []
        segment_length_mean_vals: list[float] = []

        for trial, summary in sorted(trial_map.items()):
            if not isinstance(summary, dict) or not summary:
                continue
            trial_count += 1

            responsive_frames = int(summary.get("responsive_frame_count", 0) or 0)  # type: ignore[call-overload]
            frame_total = int(summary.get("total_frame_count", 0) or 0)  # type: ignore[call-overload]
            total_responsive_frames += responsive_frames
            total_frames += frame_total

            selected_delta = _safe_float(summary.get("selected_delta_nm"))
            median_delta = _safe_float(summary.get("median_delta_nm"))
            mean_delta = _safe_float(summary.get("mean_delta_nm"))
            selected_peak = _safe_float(summary.get("selected_peak_nm"))
            baseline_peak = _safe_float(summary.get("baseline_peak_nm"))
            responsive_fraction = _safe_float(summary.get("responsive_fraction"))
            direction = _safe_float(summary.get("direction"))

            seg_count = _safe_float(summary.get("responsive_segment_count"))
            seg_coverage = _safe_float(summary.get("responsive_segment_coverage"))
            seg_lengths = (
                summary.get("responsive_segment_lengths")
                if isinstance(summary.get("responsive_segment_lengths"), list)
                else None
            )
            mean_seg_len = float("nan")
            if isinstance(seg_lengths, list) and seg_lengths:
                try:
                    mean_seg_len = float(np.nanmean([float(v) for v in seg_lengths]))
                except Exception:
                    mean_seg_len = float("nan")

            if np.isfinite(selected_delta):
                selected_delta_vals.append(selected_delta)
            if np.isfinite(median_delta):
                median_delta_vals.append(median_delta)
            if np.isfinite(mean_delta):
                mean_delta_vals.append(mean_delta)
            if np.isfinite(selected_peak):
                selected_peak_vals.append(selected_peak)
            if np.isfinite(baseline_peak):
                baseline_peak_vals.append(baseline_peak)
            if np.isfinite(responsive_fraction):
                responsive_fraction_vals.append(responsive_fraction)
            if np.isfinite(direction) and direction != 0.0:
                direction_vals.append(float(np.sign(direction)))
            if np.isfinite(seg_count):
                segment_count_vals.append(seg_count)
            if np.isfinite(seg_coverage):
                segment_coverage_vals.append(seg_coverage)
            if np.isfinite(mean_seg_len):
                segment_length_mean_vals.append(mean_seg_len)

            if responsive_frames > 0:
                responsive_trial_count += 1

            row = {
                "concentration": float(conc),
                "trial": trial,
                "selected_delta_nm": selected_delta
                if np.isfinite(selected_delta)
                else float("nan"),
                "median_delta_nm": median_delta if np.isfinite(median_delta) else float("nan"),
                "mean_delta_nm": mean_delta if np.isfinite(mean_delta) else float("nan"),
                "selected_peak_nm": selected_peak if np.isfinite(selected_peak) else float("nan"),
                "median_peak_nm": _safe_float(summary.get("median_peak_nm")),
                "mean_peak_nm": _safe_float(summary.get("mean_peak_nm")),
                "baseline_peak_nm": baseline_peak if np.isfinite(baseline_peak) else float("nan"),
                "responsive_frame_count": responsive_frames,
                "total_frame_count": frame_total,
                "responsive_fraction": responsive_fraction
                if np.isfinite(responsive_fraction)
                else float("nan"),
                "std_delta_nm": _safe_float(summary.get("std_delta_nm")),
                "max_abs_delta_nm": _safe_float(summary.get("max_abs_delta_nm")),
                "fallback_delta_nm": _safe_float(summary.get("fallback_delta_nm")),
                "direction": direction if np.isfinite(direction) else float("nan"),
                "responsive_segment_count": seg_count if np.isfinite(seg_count) else float("nan"),
                "responsive_segment_coverage": seg_coverage
                if np.isfinite(seg_coverage)
                else float("nan"),
                "responsive_segment_mean_length": mean_seg_len
                if np.isfinite(mean_seg_len)
                else float("nan"),
            }
            rows_trial.append(row)

        if trial_count == 0:
            continue

        selected_arr = (
            np.array(selected_delta_vals, dtype=float) if selected_delta_vals else np.array([])
        )
        median_arr = np.array(median_delta_vals, dtype=float) if median_delta_vals else np.array([])
        mean_arr = np.array(mean_delta_vals, dtype=float) if mean_delta_vals else np.array([])
        selected_peak_arr = (
            np.array(selected_peak_vals, dtype=float) if selected_peak_vals else np.array([])
        )
        baseline_peak_arr = (
            np.array(baseline_peak_vals, dtype=float) if baseline_peak_vals else np.array([])
        )
        responsive_fraction_arr = (
            np.array(responsive_fraction_vals, dtype=float)
            if responsive_fraction_vals
            else np.array([])
        )
        segment_count_arr = (
            np.array(segment_count_vals, dtype=float) if segment_count_vals else np.array([])
        )
        segment_cov_arr = (
            np.array(segment_coverage_vals, dtype=float) if segment_coverage_vals else np.array([])
        )
        segment_len_arr = (
            np.array(segment_length_mean_vals, dtype=float)
            if segment_length_mean_vals
            else np.array([])
        )

        def _nan_stat(arr: np.ndarray, func: str) -> float:
            if arr.size == 0 or not np.isfinite(arr).any():
                return float("nan")
            if func == "mean":
                return float(np.nanmean(arr))
            if func == "median":
                return float(np.nanmedian(arr))
            if func == "std":
                return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            if func == "min":
                return float(np.nanmin(arr))
            if func == "max":
                return float(np.nanmax(arr))
            return float("nan")

        dominant_direction = float("nan")
        if direction_vals:
            pos = sum(1 for d in direction_vals if d > 0)
            neg = sum(1 for d in direction_vals if d < 0)
            if pos or neg:
                dominant_direction = 1.0 if pos >= neg else -1.0

        summary = {
            "trial_count": float(trial_count),
            "responsive_trial_count": float(responsive_trial_count),
            "total_responsive_frames": float(total_responsive_frames),
            "total_frames": float(total_frames),
            "selected_delta_nm_median": _nan_stat(selected_arr, "median"),
            "selected_delta_nm_mean": _nan_stat(selected_arr, "mean"),
            "selected_delta_nm_std": _nan_stat(selected_arr, "std"),
            "selected_delta_nm_min": _nan_stat(selected_arr, "min"),
            "selected_delta_nm_max": _nan_stat(selected_arr, "max"),
            "median_delta_nm_median": _nan_stat(median_arr, "median"),
            "median_delta_nm_mean": _nan_stat(median_arr, "mean"),
            "mean_delta_nm_mean": _nan_stat(mean_arr, "mean"),
            "selected_peak_nm_median": _nan_stat(selected_peak_arr, "median"),
            "selected_peak_nm_mean": _nan_stat(selected_peak_arr, "mean"),
            "baseline_peak_nm_median": _nan_stat(baseline_peak_arr, "median"),
            "responsive_fraction_mean": _nan_stat(responsive_fraction_arr, "mean"),
            "responsive_fraction_median": _nan_stat(responsive_fraction_arr, "median"),
            "dominant_direction": dominant_direction,
            "responsive_segment_count_mean": _nan_stat(segment_count_arr, "mean"),
            "responsive_segment_count_median": _nan_stat(segment_count_arr, "median"),
            "responsive_segment_coverage_mean": _nan_stat(segment_cov_arr, "mean"),
            "responsive_segment_coverage_median": _nan_stat(segment_cov_arr, "median"),
            "responsive_segment_mean_length": _nan_stat(segment_len_arr, "mean"),
        }

        rows_conc.append({"concentration": float(conc), **summary})
        summary_by_conc[float(conc)] = summary

    per_trial_df = pd.DataFrame(rows_trial)
    per_conc_df = pd.DataFrame(rows_conc)
    return per_trial_df, per_conc_df, summary_by_conc
