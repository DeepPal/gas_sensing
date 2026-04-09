import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress

try:
    from config.config_loader import CONFIG
except ImportError:
    CONFIG = {}

import contextlib

from ..signal_proc import smooth_spectrum


def _write_json(path: Path, payload: dict[str, object]) -> bool:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        return True
    except Exception:
        return False


def _signal_column(df: pd.DataFrame) -> str:
    if "absorbance" in df.columns:
        return "absorbance"
    if "transmittance" in df.columns:
        return "transmittance"
    return "intensity"


def _apply_wavelength_limits(
    x: np.ndarray,
    y: np.ndarray,
    min_wl: Optional[float] = None,
    max_wl: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if min_wl is not None:
        mask &= x >= min_wl
    if max_wl is not None:
        mask &= x <= max_wl
    if mask.sum() == 0:
        return x, y
    return x[mask], y[mask]


def _apply_feature_prep_single(y: np.ndarray, x: np.ndarray, prep_spec: str) -> np.ndarray:
    if y.size == 0:
        return y
    prep = (prep_spec or "").lower().strip()
    if not prep or prep == "raw":
        return y
    steps = [token.strip() for token in prep.replace(",", "+").split("+") if token.strip()]
    arr = y.astype(float, copy=True)
    for token in steps:
        if token in ("derivative", "first_derivative"):
            try:
                arr = np.gradient(arr, x)
            except Exception:
                arr = np.gradient(arr)
        elif token == "snv":
            mu = float(np.mean(arr))
            sd = float(np.std(arr))
            if not np.isfinite(sd) or sd < 1e-9:
                sd = 1.0
            arr = (arr - mu) / sd
        elif token in ("mean_center", "center"):
            arr = arr - float(np.mean(arr))
        else:
            # ignore unknown tokens to keep pipeline robust
            continue
    return arr


def _prepare_calibration_signal(
    df: pd.DataFrame, centroid_cfg: Optional[dict[str, object]] = None
) -> tuple[np.ndarray, np.ndarray]:
    centroid_cfg = centroid_cfg or {}
    ycol = _signal_column(df)
    if ycol not in df.columns:
        # Fallback: use first non-wavelength column if present
        candidates = [c for c in df.columns if c != "wavelength"]
        if not candidates:
            raise KeyError("No signal column available for calibration")
        ycol = candidates[0]
    x = df["wavelength"].values
    y = smooth_spectrum(df[ycol].values)
    prep_spec = centroid_cfg.get("feature_prep")
    if prep_spec:
        y = _apply_feature_prep_single(y, x, str(prep_spec))
    min_wl = centroid_cfg.get("min_wavelength")
    max_wl = centroid_cfg.get("max_wavelength")
    if min_wl is not None or max_wl is not None:
        x, y = _apply_wavelength_limits(x, y, min_wl, max_wl)
    return x, y


def _find_peak_wavelength(
    df: pd.DataFrame, centroid_cfg: Optional[dict[str, object]] = None
) -> float:
    """Find peak/valley wavelength using robust sub-pixel interpolation.

    Uses parabolic interpolation around the extremum for sub-pixel accuracy,
    which is more stable than weighted centroid for noisy data.
    """
    centroid_cfg = centroid_cfg or {}
    x, y = _prepare_calibration_signal(df, centroid_cfg)

    if x.size < 3:
        return float(x[0]) if x.size else float("nan")

    # Decide if peak is max or min using overall skew
    is_min_peak = y.min() < (np.median(y) - 0.25 * (y.max() - y.min()))
    centroid_hint = centroid_cfg.get("centroid_hint")

    # Find initial index
    if centroid_hint is not None and np.isfinite(centroid_hint):
        idx = int(np.argmin(np.abs(x - centroid_hint)))
        # Search for local extremum near hint
        half_width = int(centroid_cfg.get("centroid_half_width", 5))
        search_start = max(0, idx - half_width)
        search_end = min(len(x), idx + half_width + 1)
        local_y = y[search_start:search_end]
        if is_min_peak:
            local_idx = int(np.argmin(local_y))
        else:
            local_idx = int(np.argmax(local_y))
        idx = search_start + local_idx
    else:
        idx = int(np.argmin(y) if is_min_peak else np.argmax(y))

    # Ensure idx is valid for parabolic fit (needs neighbors)
    idx = int(np.clip(idx, 1, len(x) - 2))

    # Parabolic interpolation for sub-pixel accuracy
    # Fit parabola through 3 points: (x[idx-1], y[idx-1]), (x[idx], y[idx]), (x[idx+1], y[idx+1])
    x0, x1, x2 = x[idx - 1], x[idx], x[idx + 1]
    y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]

    # Parabolic vertex formula
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-12:
        return float(x[idx])

    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    B = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom

    if abs(A) < 1e-12:
        return float(x[idx])

    # Vertex at x = -B / (2A)
    x_vertex = -B / (2.0 * A)

    # Validate: vertex should be within the 3-point window
    if x_vertex < x0 - 0.5 * (x1 - x0) or x_vertex > x2 + 0.5 * (x2 - x1):
        # Fallback to weighted centroid if parabolic fit is unreliable
        half_width = int(centroid_cfg.get("centroid_half_width", 5))
        half_width = max(1, min(half_width, len(x) // 2))
        s = max(0, idx - half_width)
        e = min(len(x) - 1, idx + half_width)
        xx = x[s : e + 1]
        yy = y[s : e + 1]

        if is_min_peak:
            weights = (yy.max() - yy) + 1e-9
        else:
            weights = (yy - yy.min()) + 1e-9

        return float(np.sum(xx * weights) / np.sum(weights))

    return float(x_vertex)


def _measure_peak_within_window(
    df: pd.DataFrame,
    center_nm: float,
    window_nm: float,
    centroid_cfg: Optional[dict[str, object]] = None,
) -> float:
    df_sorted = df.sort_values("wavelength").reset_index(drop=True)
    if df_sorted.empty:
        return float("nan")

    half = max(window_nm / 2.0, 0.1)
    min_wl = center_nm - half
    max_wl = center_nm + half
    mask = (df_sorted["wavelength"] >= min_wl) & (df_sorted["wavelength"] <= max_wl)
    subset = df_sorted.loc[mask].copy()

    expand_step = 0.5
    expand_limit = max(window_nm, 6.0)
    expand = expand_step
    while subset.empty and expand <= expand_limit:
        lower = min_wl - expand
        upper = max_wl + expand
        mask = (df_sorted["wavelength"] >= lower) & (df_sorted["wavelength"] <= upper)
        subset = df_sorted.loc[mask].copy()
        expand += expand_step

    if subset.empty:
        return float("nan")

    cfg = dict(centroid_cfg or {})
    cfg["min_wavelength"] = subset["wavelength"].min()
    cfg["max_wavelength"] = subset["wavelength"].max()
    cfg["centroid_hint"] = center_nm
    cfg.setdefault("centroid_half_width", max(3, int(np.ceil(subset.shape[0] / 6.0))))

    try:
        return _find_peak_wavelength(subset, centroid_cfg=cfg)
    except Exception:
        try:
            signal_col = _signal_column(subset)
            wl = subset["wavelength"].to_numpy(dtype=float)
            signal = subset[signal_col].to_numpy(dtype=float)
            weights = np.abs(signal - np.median(signal)) + 1e-9
            return float(np.sum(wl * weights) / np.sum(weights))
        except Exception:
            wl = subset["wavelength"].to_numpy(dtype=float)
            return float(np.nanmean(wl)) if wl.size else float("nan")


def _resolve_roi_bounds(
    dataset_label: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    min_global = roi_cfg.get("min_wavelength")
    max_global = roi_cfg.get("max_wavelength")
    overrides = roi_cfg.get("per_gas_overrides", {}) if isinstance(roi_cfg, dict) else {}
    if dataset_label and isinstance(overrides, dict):
        entry = overrides.get(dataset_label, {})
        if isinstance(entry, dict):
            # Check for direct min/max_wavelength in the entry (preferred)
            min_override = entry.get("min_wavelength")
            max_override = entry.get("max_wavelength")
            if min_override is not None or max_override is not None:
                return (
                    min_override if min_override is not None else min_global,
                    max_override if max_override is not None else max_global,
                )
            # Fallback to nested 'range' dict for backwards compatibility
            rng = entry.get("range", {})
            if isinstance(rng, dict):
                min_override = rng.get("min_wavelength", min_global)
                max_override = rng.get("max_wavelength", max_global)
                return min_override, max_override
    return min_global, max_global


def _resolve_expected_center(
    dataset_label: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    overrides = roi_cfg.get("per_gas_overrides", {}) if isinstance(roi_cfg, dict) else {}
    if dataset_label and isinstance(overrides, dict):
        entry = overrides.get(dataset_label, {})
        if isinstance(entry, dict):
            validation = entry.get("validation", {})
            if isinstance(validation, dict):
                return validation.get("expected_center"), validation.get("tolerance")
    validation_cfg = roi_cfg.get("validation", {}) if isinstance(roi_cfg, dict) else {}
    if isinstance(validation_cfg, dict):
        return validation_cfg.get("expected_center"), validation_cfg.get("tolerance")
    return None, None


def _merge_discovered_bounds(
    min_wl: float,
    max_wl: float,
    expected_center: Optional[float],
    discovered_roi: Optional[dict[str, object]],
) -> tuple[float, float, Optional[float], Optional[dict[str, object]]]:
    """Merge discovered ROI bounds with existing bounds."""
    info: Optional[dict[str, object]] = None
    if isinstance(discovered_roi, dict):
        selected = discovered_roi.get("selected", {})
        if isinstance(selected, dict) and bool(selected.get("quality_ok", False)):
            sel_min = selected.get("min_wavelength_nm")
            sel_max = selected.get("max_wavelength_nm")
            sel_center = selected.get("center_nm")
            try:
                sel_min = float(sel_min)
                sel_max = float(sel_max)
                sel_center = (
                    float(sel_center) if sel_center is not None else (sel_min + sel_max) * 0.5
                )
            except (TypeError, ValueError):
                sel_min = sel_max = sel_center = None

            # Only merge if discovered center is within existing bounds
            if (
                sel_min is not None
                and sel_max is not None
                and sel_center is not None
                and np.isfinite(sel_min)
                and np.isfinite(sel_max)
                and np.isfinite(sel_center)
                and sel_min < sel_max
                and min_wl <= sel_center <= max_wl
            ):
                new_min = max(min_wl, sel_min)
                new_max = min(max_wl, sel_max)

                # Only apply if resulting bounds are valid
                if new_min < new_max:
                    min_wl = new_min
                    max_wl = new_max
                    info = {
                        "min_wavelength_nm": float(min_wl),
                        "max_wavelength_nm": float(max_wl),
                        "center_nm": float(sel_center),
                        "window_nm": float(selected.get("window_nm", sel_max - sel_min)),
                        "score": selected.get("score"),
                    }
                    if expected_center is None:
                        expected_center = sel_center
    return min_wl, max_wl, expected_center, info


def _evaluate_roi_candidate(
    canonical_items: list[tuple[float, pd.DataFrame]],
    center_nm: float,
    window_nm: float,
    centroid_cfg: Optional[dict[str, object]],
    gates: dict[str, float],
    prior_center: Optional[float],
    prior_weight: float,
    weights: dict[str, float],
) -> dict[str, object]:
    concs = np.array([c for c, _ in canonical_items], dtype=float)
    peaks: list[float] = []
    deltas: list[float] = []
    baseline_peak = float("nan")

    def _infer_feature_kind(sig_window: np.ndarray) -> str:
        if sig_window.size < 3 or not np.any(np.isfinite(sig_window)):
            return "unknown"
        med = float(np.nanmedian(sig_window))
        mn = float(np.nanmin(sig_window))
        mx = float(np.nanmax(sig_window))
        is_valley = mn < (med - 0.25 * (mx - mn))
        return "valley" if is_valley else "peak"

    window_half = max(float(window_nm) / 2.0, 0.1)
    w_min = float(center_nm) - window_half
    w_max = float(center_nm) + window_half

    baseline_wl_win: Optional[np.ndarray] = None
    baseline_sig_win: Optional[np.ndarray] = None
    baseline_kind: Optional[str] = None
    shape_corrs: list[float] = []
    feature_kinds: list[str] = []

    for _, df in canonical_items:
        peak = _measure_peak_within_window(df, center_nm, window_nm, centroid_cfg)
        peaks.append(float(peak))
        if not np.isfinite(baseline_peak) and np.isfinite(peak):
            baseline_peak = float(peak)
        if np.isfinite(peak) and np.isfinite(baseline_peak):
            deltas.append(float(peak) - float(baseline_peak))
        else:
            deltas.append(float("nan"))

        try:
            signal_col = _signal_column(df)
            wl = df["wavelength"].to_numpy(dtype=float)
            sig = df[signal_col].to_numpy(dtype=float)
            mask = (wl >= w_min) & (wl <= w_max)
            wl_win = wl[mask]
            sig_win = sig[mask]

            if wl_win.size < 3 or not np.any(np.isfinite(sig_win)):
                feature_kinds.append("unknown")
                shape_corrs.append(float("nan"))
                continue

            kind = _infer_feature_kind(sig_win)
            feature_kinds.append(kind)
            if baseline_wl_win is None or baseline_sig_win is None:
                baseline_wl_win = wl_win
                baseline_sig_win = sig_win
                baseline_kind = kind
                shape_corrs.append(1.0)
                continue

            base_wl = baseline_wl_win
            base_sig = baseline_sig_win
            if base_wl.size < 3 or not np.any(np.isfinite(base_sig)):
                shape_corrs.append(float("nan"))
                continue

            sig_cmp = (
                np.interp(base_wl, wl_win, sig_win)
                if (wl_win.size != base_wl.size or not np.allclose(wl_win, base_wl))
                else sig_win
            )

            a = base_sig.astype(float)
            b = sig_cmp.astype(float)
            finite = np.isfinite(a) & np.isfinite(b)
            if np.count_nonzero(finite) < 3:
                shape_corrs.append(float("nan"))
                continue

            a = a[finite] - float(np.mean(a[finite]))
            b = b[finite] - float(np.mean(b[finite]))
            sa = float(np.std(a))
            sb = float(np.std(b))
            if sa < 1e-12 or sb < 1e-12:
                shape_corrs.append(float("nan"))
                continue
            corr = float(np.mean((a / sa) * (b / sb)))
            shape_corrs.append(corr)
        except Exception:
            feature_kinds.append("unknown")
            shape_corrs.append(float("nan"))

    peaks_arr = np.array(peaks, dtype=float)
    deltas_arr = np.array(deltas, dtype=float)
    valid_mask = np.isfinite(peaks_arr) & np.isfinite(deltas_arr) & np.isfinite(concs)
    concs_valid = concs[valid_mask]
    deltas_valid = deltas_arr[valid_mask]

    max_adjacent_jump = float("nan")
    try:
        finite_peaks = peaks_arr[np.isfinite(peaks_arr)]
        if finite_peaks.size >= 2:
            max_adjacent_jump = float(np.nanmax(np.abs(np.diff(finite_peaks))))
    except Exception:
        max_adjacent_jump = float("nan")

    shape_corr_median = float("nan")
    try:
        corr_arr = np.array(shape_corrs, dtype=float)
        corr_fin = corr_arr[np.isfinite(corr_arr)]
        if corr_fin.size:
            shape_corr_median = float(np.nanmedian(corr_fin))
    except Exception:
        shape_corr_median = float("nan")

    feature_type_consistency = float("nan")
    feature_type_mode = None
    try:
        if baseline_kind is not None:
            baseline_kind_str = str(baseline_kind)
            kinds = [k for k in feature_kinds if k in ("peak", "valley")]
            if kinds:
                same = sum(1 for k in kinds if k == baseline_kind_str)
                feature_type_consistency = float(same / len(kinds))
                feature_type_mode = baseline_kind_str
    except Exception:
        feature_type_consistency = float("nan")
        feature_type_mode = None

    slope = float("nan")
    intercept = float("nan")
    r2 = float("nan")
    rmse = float("nan")
    residuals = np.full_like(deltas_valid, float("nan"))
    if concs_valid.size >= 2:
        try:
            slope_lin, intercept_lin, r_val_lin, _, _ = linregress(concs_valid, deltas_valid)
            preds = intercept_lin + slope_lin * concs_valid
            residuals = deltas_valid - preds
            ss_tot = np.sum((deltas_valid - np.nanmean(deltas_valid)) ** 2)
            ss_res = np.sum(residuals**2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            slope = float(slope_lin)
            intercept = float(intercept_lin)
            rmse = float(np.sqrt(ss_res / residuals.size)) if residuals.size else float("nan")
        except Exception:
            slope = float("nan")
            intercept = float("nan")
            r2 = float("nan")
            rmse = float("nan")

    consistency = float("nan")
    if concs_valid.size >= 2:
        finite_mask = np.isfinite(deltas_valid)
        if np.count_nonzero(finite_mask) > 0:
            finite_deltas = deltas_valid[finite_mask]
            sign_ref = np.sign(np.nanmedian(finite_deltas))
            if sign_ref == 0:
                sign_ref = np.sign(slope)
            if sign_ref == 0:
                sign_ref = 1.0
            same = np.sum(np.sign(finite_deltas) == sign_ref)
            consistency = float(same / finite_deltas.size)

    snr = float("nan")
    if np.isfinite(slope) and np.isfinite(rmse) and rmse > 0:
        snr = float(abs(slope) / rmse)

    gate_min_r2 = float(gates.get("min_r2", 0.7))
    gate_min_consistency = float(gates.get("min_consistency", 0.8))
    gate_min_snr = float(gates.get("min_snr", 3.0))
    gate_min_abs_slope = float(gates.get("min_abs_slope", 0.02))
    gate_min_count = int(gates.get("min_conc_count", 3))
    gate_min_shape_corr = float(gates.get("min_shape_corr", 0.6))
    gate_max_adjacent_jump = float(gates.get("max_adjacent_jump_nm", 2.5))
    gate_min_feature_consistency = float(gates.get("min_feature_type_consistency", 0.7))

    valid_conc_count = int(np.unique(np.round(concs_valid, decimals=6)).size)
    slope_abs = abs(slope) if np.isfinite(slope) else float("nan")

    quality_flags = {
        "min_conc_count": bool(valid_conc_count >= gate_min_count),
        "min_r2": bool(np.isfinite(r2) and r2 >= gate_min_r2),
        "min_consistency": bool(np.isfinite(consistency) and consistency >= gate_min_consistency),
        "min_snr": bool(np.isfinite(snr) and snr >= gate_min_snr),
        "min_abs_slope": bool(np.isfinite(slope_abs) and slope_abs >= gate_min_abs_slope),
        "min_shape_corr": bool(
            (not np.isfinite(shape_corr_median)) or (shape_corr_median >= gate_min_shape_corr)
        ),
        "max_adjacent_jump_nm": bool(
            (not np.isfinite(max_adjacent_jump)) or (max_adjacent_jump <= gate_max_adjacent_jump)
        ),
        "min_feature_type_consistency": bool(
            (not np.isfinite(feature_type_consistency))
            or (feature_type_consistency >= gate_min_feature_consistency)
        ),
    }
    quality_ok = all(quality_flags.values())

    w_r2 = float(weights.get("r2", 1.5))
    w_slope = float(weights.get("slope", 0.6))
    w_snr = float(weights.get("snr", 0.4))
    penalty = (
        prior_weight * abs(center_nm - prior_center)
        if (prior_center is not None and np.isfinite(prior_center))
        else 0.0
    )
    score_components = {
        "r2": float(r2),
        "abs_slope": float(abs(slope)) if np.isfinite(slope) else float("nan"),
        "snr": float(snr) if np.isfinite(snr) else float("nan"),
    }
    score = 0.0
    if np.isfinite(r2):
        score += w_r2 * r2
    if np.isfinite(slope):
        score += w_slope * abs(slope)
    if np.isfinite(snr):
        score += w_snr * snr
    score -= penalty

    candidate = {
        "center_nm": float(center_nm),
        "window_nm": float(window_nm),
        "min_wavelength_nm": float(center_nm - window_nm / 2.0),
        "max_wavelength_nm": float(center_nm + window_nm / 2.0),
        "concentrations_ppm": concs.tolist(),
        "peak_wavelengths_nm": peaks_arr.astype(float).tolist(),
        "deltas_nm": deltas_arr.astype(float).tolist(),
        "concentrations_valid_ppm": concs_valid.astype(float).tolist(),
        "deltas_valid_nm": deltas_valid.astype(float).tolist(),
        "baseline_peak_nm": float(baseline_peak) if np.isfinite(baseline_peak) else float("nan"),
        "slope_nm_per_ppm": float(slope) if np.isfinite(slope) else float("nan"),
        "intercept_nm": float(intercept) if np.isfinite(intercept) else float("nan"),
        "r2": float(r2) if np.isfinite(r2) else float("nan"),
        "rmse_nm": float(rmse) if np.isfinite(rmse) else float("nan"),
        "snr": float(snr) if np.isfinite(snr) else float("nan"),
        "consistency": float(consistency) if np.isfinite(consistency) else float("nan"),
        "shape_corr_median": float(shape_corr_median)
        if np.isfinite(shape_corr_median)
        else float("nan"),
        "max_adjacent_jump_nm": float(max_adjacent_jump)
        if np.isfinite(max_adjacent_jump)
        else float("nan"),
        "feature_type_consistency": float(feature_type_consistency)
        if np.isfinite(feature_type_consistency)
        else float("nan"),
        "feature_type_mode": feature_type_mode,
        "quality_ok": quality_ok,
        "quality_flags": quality_flags,
        "score": float(score),
        "score_components": score_components,
    }
    return candidate


def _discover_roi_in_band(
    canonical: dict[float, pd.DataFrame],
    dataset_label: Optional[str] = None,
    out_root: Optional[str] = None,
) -> dict[str, object]:
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    discovery_cfg = (
        roi_cfg.get("discovery", {}) if isinstance(roi_cfg.get("discovery", {}), dict) else {}
    )
    if not discovery_cfg.get("enabled", False):
        return {}

    # Use per-gas ROI bounds if available, otherwise fall back to discovery band
    per_gas_min, per_gas_max = _resolve_roi_bounds(dataset_label)
    if per_gas_min is not None and per_gas_max is not None:
        band_min = float(per_gas_min)
        band_max = float(per_gas_max)
    else:
        band = discovery_cfg.get("band", [600.0, 700.0])
        band_min = float(band[0]) if isinstance(band, (list, tuple)) and len(band) >= 2 else 600.0
        band_max = float(band[1]) if isinstance(band, (list, tuple)) and len(band) >= 2 else 700.0
    window_nm = float(discovery_cfg.get("window_nm", 12.0))
    step_nm_cfg = discovery_cfg.get("step_nm", 0.2)
    try:
        step_nm = float(step_nm_cfg)
    except Exception:
        step_nm = 0.2
    if step_nm <= 0:
        step_nm = 0.2
    expected_center = discovery_cfg.get("expected_center", None)
    prior_center = float(expected_center) if expected_center is not None else None
    prior_weight = float(discovery_cfg.get("prior_weight", 0.03))
    gates = (
        discovery_cfg.get("gates", {}) if isinstance(discovery_cfg.get("gates", {}), dict) else {}
    )
    weights = (
        discovery_cfg.get("weights", {})
        if isinstance(discovery_cfg.get("weights", {}), dict)
        else {}
    )

    calib_cfg = CONFIG.get("calibration", {}) if isinstance(CONFIG, dict) else {}
    centroid_cfg = calib_cfg.copy() if isinstance(calib_cfg, dict) else {}

    canonical_items = sorted(canonical.items(), key=lambda kv: kv[0])
    if not canonical_items:
        return {}

    # Build an instrument-aware set of candidate centers from the actual wavelength grid.
    try:
        wl_all: list[float] = []
        for _, df in canonical_items:
            if "wavelength" in df.columns:
                wl_vals = df["wavelength"].to_numpy(dtype=float)
                wl_all.append(wl_vals)
        if wl_all:
            wl_concat = np.unique(np.concatenate(wl_all))
            mask_band = (wl_concat >= band_min) & (wl_concat <= band_max)
            wl_band = wl_concat[mask_band]
        else:
            wl_band = np.array([], dtype=float)
    except Exception:
        wl_band = np.array([], dtype=float)

    if wl_band.size == 0:
        # Fallback to uniform centers if wavelength grid could not be resolved.
        centers = np.arange(band_min, band_max + step_nm / 2.0, step_nm)
    else:
        # Use step_nm as an approximate stride in units of the native wavelength spacing.
        if wl_band.size > 1:
            diffs = np.diff(wl_band)
            median_spacing = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else step_nm
            if median_spacing <= 0:
                median_spacing = step_nm
            stride = max(1, int(round(step_nm / median_spacing)))
        else:
            stride = 1
        centers = wl_band[::stride]
    candidates: list[dict[str, object]] = []
    for center_nm in centers:
        candidate = _evaluate_roi_candidate(
            canonical_items,
            float(center_nm),
            window_nm,
            centroid_cfg,
            gates,
            prior_center,
            prior_weight,
            weights,
        )
        candidates.append(candidate)

    if not candidates:
        return {}

    candidates_sorted = sorted(
        candidates, key=lambda c: c.get("score", float("-inf")), reverse=True
    )
    top_candidate = next(
        (c for c in candidates_sorted if c.get("quality_ok")), candidates_sorted[0]
    )

    discovery = {
        "selected": top_candidate,
        "candidates": candidates_sorted[: int(discovery_cfg.get("retain_top", 10))],
        "band": [band_min, band_max],
        "window_nm": window_nm,
        "step_nm": step_nm,
        "prior_center": prior_center,
        "prior_weight": prior_weight,
        "gates": gates,
        "weights": weights,
        "dataset_label": dataset_label,
    }

    if out_root:
        try:
            metrics_dir = Path(out_root) / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            _write_json(metrics_dir / "roi_discovery.json", discovery)
        except Exception:
            pass

    return discovery


def find_roi_and_calibration(
    canonical: dict[float, pd.DataFrame],
    dataset_label: Optional[str] = None,
    responsive_delta: Optional[dict[str, object]] = None,
    discovered_roi: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """Compute wavelength shift vs concentration and fit linear calibration.

    Returns dict with keys: 'concentrations', 'peak_wavelengths', 'slope', 'intercept',
    'r2', 'rmse', 'slope_se', 'slope_ci_low', 'slope_ci_high', 'lod', 'loq', 'roi_center'.
    If ``responsive_delta`` is provided, responsive-frame Δλ statistics are used to
    override the canonical calibration metrics while preserving canonical details in
    the ``canonical_model`` field for auditability.
    """
    if not canonical:
        raise ValueError("No canonical spectra provided")

    # Sort by concentration
    items = sorted(canonical.items(), key=lambda kv: kv[0])
    concs = np.array([kv[0] for kv in items], dtype=float)
    calib_cfg = CONFIG.get("calibration", {}) or {}
    str(calib_cfg.get("shift_method", "centroid")).lower()
    int(max(1, calib_cfg.get("xcorr_upsample", 1)))

    # Apply ROI wavelength limits to calibration
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = 500.0
    if max_wl_roi is None:
        max_wl_roi = 900.0

    expected_center, expected_tol = _resolve_expected_center(dataset_label)
    expected_min = None
    expected_max = None
    if expected_center is not None and expected_tol is not None and expected_tol > 0:
        expected_min = expected_center - expected_tol
        expected_max = expected_center + expected_tol
        if expected_min is not None:
            min_wl_roi = max(min_wl_roi, expected_min)
        if expected_max is not None:
            max_wl_roi = min(max_wl_roi, expected_max)
        if min_wl_roi >= max_wl_roi:
            min_wl_roi = expected_min if expected_min is not None else min_wl_roi
            max_wl_roi = expected_max if expected_max is not None else max_wl_roi

    min_wl_roi, max_wl_roi, expected_center, discovery_applied = _merge_discovered_bounds(
        min_wl_roi,
        max_wl_roi,
        expected_center,
        discovered_roi,
    )

    # Trim canonical spectra to ROI bounds (with minor expansion fallback)
    roi_center_guess = (
        expected_center if expected_center is not None else 0.5 * (min_wl_roi + max_wl_roi)
    )
    trimmed_canonical: dict[float, pd.DataFrame] = {}
    for conc, df in items:
        df_sorted = df.sort_values("wavelength").reset_index(drop=True)
        mask = (df_sorted["wavelength"] >= min_wl_roi) & (df_sorted["wavelength"] <= max_wl_roi)
        df_roi = df_sorted.loc[mask].copy()
        expand_step = 0.5
        expand_limit = 3.0
        current_expand = expand_step
        while df_roi.empty and current_expand <= expand_limit:
            lower = min_wl_roi - current_expand
            upper = max_wl_roi + current_expand
            mask = (df_sorted["wavelength"] >= lower) & (df_sorted["wavelength"] <= upper)
            df_roi = df_sorted.loc[mask].copy()
            current_expand += expand_step
        if df_roi.empty and not df_sorted.empty:
            wl = df_sorted["wavelength"].to_numpy()
            nearest_indices = np.argsort(np.abs(wl - roi_center_guess))[: min(25, wl.size)]
            df_roi = df_sorted.iloc[np.sort(nearest_indices)].copy()
        trimmed_canonical[conc] = df_roi if not df_roi.empty else df_sorted.copy()
        if df_roi.empty and df_sorted.empty:
            trimmed_canonical[conc] = df.copy()

    # Apply Savitzky-Golay smoothing to ROI spectra for noise reduction
    if isinstance(CONFIG, dict):
        smoothing_cfg = CONFIG.get("preprocessing_roi", {}).get("smoothing", {})
        if not isinstance(smoothing_cfg, dict) or not smoothing_cfg:
            smoothing_cfg = CONFIG.get("preprocessing", {}).get("smoothing", {})
    else:
        smoothing_cfg = {}
    smoothing_enabled = bool(smoothing_cfg.get("enabled", False))  # Default to False
    if smoothing_enabled:
        try:
            window_length = int(smoothing_cfg.get("window", 11))
            poly_order = int(smoothing_cfg.get("poly_order", 2))

            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1

            # Simplified output printing
            # print(f"[SMOOTHING] Applying Savitzky-Golay filter (window={window_length}, poly={poly_order})")

            smoothed_canonical = {}
            for conc, df in trimmed_canonical.items():
                if df is None or df.empty:
                    smoothed_canonical[conc] = df
                    continue

                df_smooth = df.copy()
                signal_cols = [c for c in df.columns if c != "wavelength"]

                for col in signal_cols:
                    if col in df_smooth.columns and len(df_smooth) >= window_length:
                        with contextlib.suppress(Exception):
                            df_smooth[col] = savgol_filter(
                                df_smooth[col], window_length, poly_order
                            )
                smoothed_canonical[conc] = df_smooth
            trimmed_canonical = smoothed_canonical
        except Exception:
            pass

    # Measure peaks
    peaks = []
    baseline_peak = float("nan")
    for conc, df in sorted(trimmed_canonical.items(), key=lambda x: x[0]):
        # Calculate window width
        wl_vals = df["wavelength"].values
        if wl_vals.size > 0:
            window_nm = float(wl_vals.max() - wl_vals.min())
            center_nm = float(wl_vals.min() + window_nm / 2.0)
        else:
            window_nm = max_wl_roi - min_wl_roi
            center_nm = (min_wl_roi + max_wl_roi) / 2.0

        peak = _measure_peak_within_window(df, center_nm, window_nm, calib_cfg)
        peaks.append(peak)
        if np.isnan(baseline_peak) and np.isfinite(peak):
            baseline_peak = peak

    # Perform Regression
    concentrations = concs
    peak_wavelengths = np.array(peaks)

    # Calculate deltas properly
    wavelength_shifts = []
    if np.isfinite(baseline_peak):
        wavelength_shifts = peak_wavelengths - baseline_peak
    else:
        wavelength_shifts = np.full_like(peak_wavelengths, np.nan)

    mask = np.isfinite(concentrations) & np.isfinite(wavelength_shifts)

    slope = np.nan
    intercept = np.nan
    r2 = np.nan
    rmse = np.nan

    if np.sum(mask) >= 2:
        res = linregress(concentrations[mask], wavelength_shifts[mask])
        slope = res.slope
        intercept = res.intercept
        r2 = res.rvalue**2
        rmse = np.sqrt(
            np.mean((wavelength_shifts[mask] - (slope * concentrations[mask] + intercept)) ** 2)
        )

    return {
        "concentrations": concentrations.tolist(),
        "peak_wavelengths": peak_wavelengths.tolist(),
        "wavelength_shifts": wavelength_shifts.tolist(),
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "rmse": rmse,
        "roi_center": (min_wl_roi + max_wl_roi) / 2.0,
    }


def perform_absorbance_amplitude_calibration(
    canonical: dict[float, pd.DataFrame],
    out_root: str,
    dataset_label: Optional[str] = None,
    wl_min: float = 400.0,
    wl_max: float = 800.0,
) -> Optional[dict[str, object]]:
    """Perform ENHANCED calibration using absorbance amplitude (ΔA) method."""
    if not canonical or len(canonical) < 2:
        return None

    # Sort by concentration
    items = sorted(canonical.items(), key=lambda x: x[0])
    concentrations = np.array([c for c, _ in items])

    # Get reference (lowest concentration)
    ref_conc, ref_df = items[0]

    # Determine signal column (prefer absorbance)
    signal_col = "absorbance"
    if signal_col not in ref_df.columns:
        signal_col = "transmittance" if "transmittance" in ref_df.columns else "intensity"

    # Get wavelength array
    wl = ref_df["wavelength"].values
    mask = (wl >= wl_min) & (wl <= wl_max)
    wl_roi = wl[mask]

    if len(wl_roi) < 10:
        return None

    # Build signal matrix: rows = wavelengths, cols = concentrations
    signal_matrix = []
    for _conc, df in items:
        if signal_col in df.columns:
            sig_arr = df[signal_col].values[mask]
            signal_matrix.append(sig_arr)
    signal_matrix = np.array(signal_matrix).T  # Shape: (n_wavelengths, n_concentrations)

    # Simple single wavelength calibration for now
    # Find wavelength with max correlation to concentration
    best_r2 = -1.0
    best_wl = np.nan
    best_res = None

    for i, w in enumerate(wl_roi):
        sigs = signal_matrix[i, :]
        if np.std(sigs) < 1e-9:
            continue
        res = linregress(concentrations, sigs)
        if res.rvalue**2 > best_r2:
            best_r2 = res.rvalue**2
            best_wl = w
            best_res = res

    if best_res is None:
        return None

    return {
        "best_wavelength": best_wl,
        "r2": best_r2,
        "slope": best_res.slope,
        "intercept": best_res.intercept,
    }
