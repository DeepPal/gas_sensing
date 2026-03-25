"""ROI discovery via per-wavelength linear regression scanning.

All functions are CONFIG-free — every threshold and parameter is passed
explicitly through :class:`RoiScanConfig` so results are fully reproducible
without a global CONFIG object.

Typical usage::

    cfg = RoiScanConfig(selection_metric="hybrid", r2_weight=0.7,
                        expected_trend="decreasing")
    response, avg_by_conc = compute_concentration_response(stable_by_conc, cfg)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.reporting.metrics import select_signal_column
from src.signal.roi import compute_band_ratio_matrix


# ---------------------------------------------------------------------------
# Trial stacking helper
# ---------------------------------------------------------------------------


def stack_trials_for_response(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """Stack all trials from *stable_by_conc* into a design matrix.

    Concentrations are processed in ascending order.  When a trial's
    wavelength grid differs from the first grid encountered, signal values are
    linearly interpolated onto the common grid.

    Args:
        stable_by_conc: Nested mapping ``{concentration_ppm: {trial_name: DataFrame}}``.
            Each DataFrame must have a ``"wavelength"`` column and at least one
            signal column (auto-detected via :func:`~src.reporting.metrics.select_signal_column`).

    Returns:
        Tuple ``(wl, Y, concs, avg_by_conc)`` where:

        - ``wl``: common wavelength grid, shape ``(W,)``
        - ``Y``: stacked signal matrix, shape ``(N_total, W)``
        - ``concs``: concentration label per row, shape ``(N_total,)``
        - ``avg_by_conc``: per-concentration mean spectrum ``{conc_ppm: ndarray(W,)}``

    Raises:
        ValueError: If *stable_by_conc* is empty or contains no data.

    Example:
        >>> import pandas as pd, numpy as np
        >>> wl = np.linspace(680, 750, 50)
        >>> agg = {0.5: {"t0": pd.DataFrame({"wavelength": wl, "transmittance": np.ones(50)*0.9})}}
        >>> base_wl, Y, concs, avg = stack_trials_for_response(agg)
        >>> Y.shape[1] == 50
        True
    """
    base_wl: Optional[np.ndarray] = None
    stacked: list[np.ndarray] = []
    conc_labels: list[float] = []
    avg_by_conc: dict[float, np.ndarray] = {}

    for conc, trials in sorted(stable_by_conc.items(), key=lambda kv: kv[0]):
        trial_arrays: list[np.ndarray] = []
        for df in trials.values():
            col = select_signal_column(df)
            wl = df["wavelength"].values
            y = df[col].values
            if base_wl is None:
                base_wl = wl
            elif not np.array_equal(base_wl, wl):
                y = np.interp(base_wl, wl, y)
            stacked.append(y)
            conc_labels.append(conc)
            trial_arrays.append(y)
        if trial_arrays:
            avg_by_conc[conc] = np.mean(trial_arrays, axis=0)

    if base_wl is None:
        raise ValueError("No spectra available for concentration response analysis")

    return (
        base_wl,
        np.vstack(stacked) if stacked else np.zeros((0, len(base_wl))),
        np.array(conc_labels, dtype=float),
        avg_by_conc,
    )


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class RoiScanConfig:
    """All tunable parameters for :func:`compute_concentration_response`.

    Defaults match the legacy ``CONFIG["roi"]`` section so the original
    behaviour is preserved when constructing with no arguments.

    Example:
        >>> cfg = RoiScanConfig(selection_metric="hybrid", r2_weight=0.7)
        >>> cfg.selection_metric
        'hybrid'
        >>> cfg.r2_weight
        0.7
    """

    # ── ROI selection metric ──────────────────────────────────────────────
    selection_metric: str = "r2"
    """Scoring metric: ``"r2"`` | ``"slope"`` | ``"hybrid"`` | ``"poly_r2"``."""

    min_r2: float = 0.0
    """Minimum R² required for a wavelength to be eligible for selection."""

    r2_weight: float = 1.0
    """Weight of the R² component in ``"hybrid"`` score (clipped to [0, 1])."""

    # ── Trend filter ─────────────────────────────────────────────────────
    expected_trend: str = "any"
    """Single-mode filter: ``"increasing"`` | ``"decreasing"`` | ``"valley"`` | ``"peak"`` | ``"any"``."""

    trend_modes: Optional[list[str]] = None
    """Multi-mode override.  Each mode is tried independently; best score wins."""

    min_corr: float = 0.0
    """Minimum absolute Pearson |r| required (applied per *trend_modes* entry)."""

    # ── Wavelength range ──────────────────────────────────────────────────
    min_wavelength: Optional[float] = None
    """Lower wavelength bound (nm).  ``None`` → no clipping."""

    max_wavelength: Optional[float] = None
    """Upper wavelength bound (nm).  ``None`` → no clipping."""

    # ── Band / ROI width ──────────────────────────────────────────────────
    band_half_width: Optional[int] = None
    """Half-width in wavelength indices.  ``None`` → auto: ``max(3, len(wl) // 40)``."""

    band_window: int = 0
    """Moving-average smoothing window applied to per-wavelength scores (0 = off)."""

    # ── Feature augmentation weights ──────────────────────────────────────
    derivative_weight: float = 0.0
    """Fraction of score from the first-derivative channel (clipped to [0, 1])."""

    ratio_weight: float = 0.0
    """Fraction of score from the band-ratio channel (clipped to [0, 1])."""

    ratio_half_width: int = 5
    """Half-width passed to :func:`~src.signal.roi.compute_band_ratio_matrix`."""

    # ── Slope-to-noise filter ─────────────────────────────────────────────
    slope_noise_weight: float = 0.0
    """Weight of slope-to-noise ratio in scoring (clipped to [0, 1])."""

    min_slope_to_noise: float = 0.0
    """Minimum slope-to-noise ratio; wavelengths below this are zeroed out."""

    global_std: float = 0.0
    """Global noise floor σ from repeatability analysis.

    Replaces the legacy ``CONFIG["_last_repeatability"]`` side-channel read.
    Pass :pycode:`repeatability["global"]["std_transmittance"]` here.
    """

    min_abs_slope: float = 0.0
    """Minimum absolute regression slope; wavelengths below this are excluded."""

    # ── Polynomial (alternative) model ────────────────────────────────────
    alt_models_enabled: bool = False
    """Enable polynomial fit alongside the linear model."""

    poly_degree: int = 2
    """Polynomial degree for the alternative model (used only when *alt_models_enabled*)."""

    # ── Adaptive band width ───────────────────────────────────────────────
    adaptive_band_enabled: bool = False
    """Expand ROI until slope drops below *slope_fraction* × peak slope."""

    slope_fraction: float = 0.6
    """Target slope fraction for adaptive band expansion (clipped to [0, 1])."""

    adaptive_max_half_width: int = 20
    """Maximum half-width (indices) for the adaptive expansion."""

    # ── Validation ────────────────────────────────────────────────────────
    expected_center: Optional[float] = None
    """Expected ROI centre wavelength (nm) for physics-based validation."""

    center_tolerance: float = 0.0
    """Allowed deviation from *expected_center* (nm).  0 disables the check."""

    validation_notes: str = ""
    """Free-text notes stored verbatim in the validation result dict."""


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def compute_concentration_response(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    cfg: Optional[RoiScanConfig] = None,
    top_k_candidates: int = 0,
    debug_out_root: Optional[str] = None,
) -> tuple[dict[str, object], dict[float, np.ndarray]]:
    """Scan every wavelength for the best ROI via per-wavelength linear regression.

    For each wavelength column in the stacked trial matrix a linear regression
    of signal vs. concentration is computed.  The wavelength that maximises
    the configured scoring metric becomes the ROI centre.

    Args:
        stable_by_conc: Nested ``{concentration_ppm: {trial_name: DataFrame}}``.
            Each DataFrame must have ``"wavelength"`` and at least one signal column.
        cfg: Scan parameters.  ``None`` uses :class:`RoiScanConfig` defaults,
            which reproduce the original legacy behaviour.
        top_k_candidates: If > 0, populate ``response["candidates"]`` with the
            top-*k* ROI candidates ranked by selection score.
        debug_out_root: Optional path.  If provided, writes a per-wavelength
            regression CSV to ``{debug_out_root}/metrics/debug_all_wavelength_regressions.csv``.

    Returns:
        ``(response, avg_by_conc)`` where *response* is a dict containing:

        - ``"wavelengths"``, ``"slopes"``, ``"r_squared"``, ``"correlations"``
        - ``"max_slope_wavelength"``, ``"max_r_squared"``, ``"max_slope"``
        - ``"roi_start_wavelength"``, ``"roi_end_wavelength"``, ``"roi_start_index"``, ``"roi_end_index"``
        - ``"roi_selection_metric"``, ``"roi_score"``
        - ``"validation"``, ``"candidates"``

        *avg_by_conc* maps ``concentration_ppm → mean_spectrum_array``.

    Raises:
        ValueError: If *stable_by_conc* is empty or the wavelength filter
            removes all points.
    """
    if cfg is None:
        cfg = RoiScanConfig()

    wl, Y, concs, avg_by_conc = stack_trials_for_response(stable_by_conc)
    if Y.size == 0:
        raise ValueError("No spectra available for concentration response analysis")

    slopes: list[float] = []
    intercepts: list[float] = []
    r2_vals: list[float] = []
    poly_r2_vals: list[float] = []
    corr_vals: list[float] = []
    residual_stds: list[float] = []

    # Unpack config with safety clips
    selection_metric = str(cfg.selection_metric).lower()
    min_r2 = float(cfg.min_r2)
    r2_weight = float(np.clip(cfg.r2_weight, 0.0, 1.0))
    band_half_width_cfg = cfg.band_half_width
    band_window = int(max(1, cfg.band_window)) if cfg.band_window else 0
    expected_trend = str(cfg.expected_trend).lower()
    min_corr = float(np.clip(cfg.min_corr, 0.0, 1.0))
    derivative_weight = float(np.clip(cfg.derivative_weight, 0.0, 1.0))
    ratio_weight = float(np.clip(cfg.ratio_weight, 0.0, 1.0))
    ratio_half_width = int(max(1, cfg.ratio_half_width))
    slope_noise_weight = float(np.clip(cfg.slope_noise_weight, 0.0, 1.0))
    min_slope_to_noise = float(max(0.0, cfg.min_slope_to_noise))
    min_abs_slope_cfg = float(max(0.0, cfg.min_abs_slope))
    global_std = float(cfg.global_std)

    allowed_modes = {"increasing", "decreasing", "any", "valley", "peak", "dip"}
    if isinstance(cfg.trend_modes, (list, tuple, set)):
        trend_modes = [str(m).lower() for m in cfg.trend_modes if str(m).lower() in allowed_modes]
    else:
        trend_modes = []
    if not trend_modes:
        trend_modes = [expected_trend]
    if expected_trend == "any" and "any" not in trend_modes:
        trend_modes.append("any")
    trend_modes = list(dict.fromkeys(trend_modes))

    min_wavelength = cfg.min_wavelength
    max_wavelength = cfg.max_wavelength

    mask_wl = np.ones_like(wl, dtype=bool)
    if min_wavelength is not None:
        mask_wl &= wl >= float(min_wavelength)
    if max_wavelength is not None:
        mask_wl &= wl <= float(max_wavelength)

    if not np.any(mask_wl):
        raise ValueError("No wavelengths remain after applying ROI wavelength constraints")

    wl = wl[mask_wl]
    Y = Y[:, mask_wl]
    for conc_key in list(avg_by_conc):
        avg_by_conc[conc_key] = avg_by_conc[conc_key][mask_wl]

    # Feature matrices
    derivative_matrix = None
    ratio_matrix = None
    if derivative_weight > 0:
        derivative_matrix = np.gradient(Y, wl, axis=1)
    if ratio_weight > 0:
        ratio_matrix = compute_band_ratio_matrix(Y, ratio_half_width)

    alt_enabled = bool(cfg.alt_models_enabled)
    poly_degree = int(max(1, cfg.poly_degree)) if alt_enabled else 1
    adaptive_enabled = bool(cfg.adaptive_band_enabled)
    slope_fraction = float(np.clip(cfg.slope_fraction, 0.0, 1.0)) if adaptive_enabled else 0.0
    adaptive_max_half = int(cfg.adaptive_max_half_width) if adaptive_enabled else 0

    # Per-wavelength regression
    for j in range(Y.shape[1]):
        column = Y[:, j]
        res = linregress(concs, column)
        preds_lin = res.intercept + res.slope * concs
        residual_std = float(np.std(column - preds_lin, ddof=1)) if column.size > 1 else 0.0
        slopes.append(float(res.slope))
        intercepts.append(float(res.intercept))
        r_val = float(res.rvalue) if not np.isnan(res.rvalue) else np.nan
        r2_vals.append(float(r_val ** 2) if not np.isnan(r_val) else np.nan)
        corr_vals.append(r_val)
        residual_stds.append(residual_std)

        if alt_enabled and len(concs) > poly_degree:
            try:
                poly_c = np.polyfit(concs, column, poly_degree)
                preds_p = np.polyval(poly_c, concs)
                ss_res = float(np.sum((column - preds_p) ** 2))
                ss_tot = float(np.sum((column - np.mean(column)) ** 2))
                poly_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except np.linalg.LinAlgError:
                poly_r2 = np.nan
        else:
            poly_r2 = np.nan
        poly_r2_vals.append(poly_r2)

    slopes_arr = np.array(slopes)
    abs_slopes = np.abs(slopes_arr)
    r2_arr = np.array(r2_vals)
    poly_r2_arr = np.array(poly_r2_vals)
    corr_arr = np.array(corr_vals)
    residual_std_arr = np.array(residual_stds)

    # Optional debug CSV (I/O only, gated on caller intent)
    if debug_out_root is not None:
        debug_df = pd.DataFrame({
            "wavelength": wl,
            "slope": slopes_arr,
            "abs_slope": abs_slopes,
            "intercept": intercepts,
            "r2": r2_arr,
            "correlation": corr_arr,
            "residual_std": residual_std_arr,
        })
        try:
            metrics_dir = os.path.join(debug_out_root, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            debug_df.to_csv(os.path.join(metrics_dir, "debug_all_wavelength_regressions.csv"), index=False)
        except OSError:
            pass

    # Slope-to-noise ratio (requires non-zero global_std)
    noise_per_band: Optional[np.ndarray] = None
    if (slope_noise_weight > 0 or min_slope_to_noise > 0) and global_std > 0:
        noise_per_band = abs_slopes / global_std

    # Smoothed score arrays
    slope_scores = abs_slopes.copy()
    r2_scores = r2_arr.copy()
    poly_r2_scores = poly_r2_arr.copy()
    corr_scores = corr_arr.copy()

    if derivative_matrix is not None and derivative_weight > 0:
        deriv_slopes = [float(linregress(concs, derivative_matrix[:, j]).slope)
                        for j in range(derivative_matrix.shape[1])]
        deriv_arr = np.abs(np.array(deriv_slopes))
        slope_scores = (1 - derivative_weight) * slope_scores + derivative_weight * deriv_arr

    if ratio_matrix is not None and ratio_weight > 0:
        ratio_slopes = [float(linregress(concs, ratio_matrix[:, j]).slope)
                        for j in range(ratio_matrix.shape[1])]
        ratio_arr = np.abs(np.array(ratio_slopes))
        slope_scores = (1 - ratio_weight) * slope_scores + ratio_weight * ratio_arr

    if band_window > 1:
        kernel = np.ones(band_window) / band_window
        slope_scores = np.convolve(slope_scores, kernel, mode="same")
        r2_scores = np.convolve(r2_scores, kernel, mode="same")
        if not np.all(np.isnan(poly_r2_scores)):
            poly_r2_scores = np.convolve(poly_r2_scores, kernel, mode="same")
        if not np.all(np.isnan(corr_scores)):
            corr_scores = np.convolve(corr_scores, kernel, mode="same")

    max_abs_slope: float = float(np.nanmax(abs_slopes))
    norm_slopes = (
        abs_slopes / max_abs_slope
        if max_abs_slope and not np.isnan(max_abs_slope)
        else np.zeros_like(abs_slopes)
    )
    r2_clean = np.clip(r2_scores, 0.0, 1.0)
    poly_r2_clean = np.clip(poly_r2_scores, 0.0, 1.0)

    # Build composite score array
    if selection_metric in ("poly", "poly_r2") and alt_enabled:
        score = poly_r2_clean.copy()
        if min_r2 > 0:
            score[poly_r2_clean < min_r2] = 0.0
    elif selection_metric == "slope":
        score = norm_slopes * (
            np.minimum(1.0, r2_clean / max(min_r2, 1e-6)) if min_r2 > 0 else 1.0
        )
    elif selection_metric == "hybrid":
        r2_comp = (
            poly_r2_clean
            if alt_enabled and not np.all(np.isnan(poly_r2_clean))
            else r2_clean
        )
        score = r2_weight * r2_comp + (1.0 - r2_weight) * norm_slopes
        if min_r2 > 0:
            score[r2_clean < min_r2] = 0.0
    else:  # default: r2
        score = poly_r2_clean.copy() if (alt_enabled and selection_metric == "poly_r2") else r2_clean.copy()
        if min_r2 > 0:
            score[r2_clean < min_r2] = 0.0

    if noise_per_band is not None and slope_noise_weight > 0:
        noise_scaled = np.nan_to_num(noise_per_band, nan=0.0, posinf=0.0, neginf=0.0)
        score = (1.0 - slope_noise_weight) * score + slope_noise_weight * noise_scaled

    if noise_per_band is not None and min_slope_to_noise > 0:
        score[(noise_per_band < min_slope_to_noise) | np.isnan(noise_per_band)] = 0.0

    if min_abs_slope_cfg > 0:
        score[abs_slopes < min_abs_slope_cfg] = 0.0

    # Trend-filtered best-index selection
    idx_all = np.arange(len(score))
    best_idx: Optional[int] = None
    best_score_val = -np.inf

    def _apply_trend_filter(trend: str, corr_array: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(corr_array)
        if trend in ("decreasing", "valley", "dip"):
            thresh = -min_corr if min_corr > 0 else 0.0
            mask &= corr_array <= thresh
            if min_corr > 0 and not mask.any():
                mask = (~np.isnan(corr_array)) & (corr_array < 0.0)
        elif trend in ("increasing", "peak"):
            thresh = min_corr if min_corr > 0 else 0.0
            mask &= corr_array >= thresh
            if min_corr > 0 and not mask.any():
                mask = (~np.isnan(corr_array)) & (corr_array > 0.0)
        else:  # "any"
            if min_corr > 0:
                mask &= np.abs(corr_array) >= min_corr
        return mask

    for mode in trend_modes:
        eligible_mask = _apply_trend_filter(mode, corr_arr)
        if not eligible_mask.any():
            continue
        mode_score = score.copy()
        mode_score[~eligible_mask] = 0.0
        valid_idx = idx_all[eligible_mask & (~np.isnan(mode_score))]
        if valid_idx.size == 0:
            continue
        mode_best = int(valid_idx[np.nanargmax(mode_score[valid_idx])])
        mode_val = float(mode_score[mode_best])
        if mode_val > best_score_val:
            best_score_val = mode_val
            best_idx = mode_best
        elif mode_val == best_score_val and best_idx is not None:
            if abs_slopes[mode_best] > abs_slopes[best_idx]:
                best_idx = mode_best

    if best_idx is None:
        eligible = ~np.isnan(score)
        valid = idx_all[eligible & (~np.isnan(score))]
        if valid.size > 0:
            best_idx = int(valid[np.nanargmax(score[valid])])
        else:
            fallback = idx_all[eligible & (~np.isnan(r2_scores))]
            best_idx = int(fallback[np.nanargmax(r2_scores[fallback])]) if fallback.size > 0 else int(np.nanargmax(norm_slopes))

    assert best_idx is not None

    # ROI window
    default_half = max(3, min(25, len(wl) // 40))
    half_width = int(band_half_width_cfg) if band_half_width_cfg is not None else default_half
    half_width = max(1, min(half_width, len(wl) // 2))
    roi_start_idx = max(0, best_idx - half_width)
    roi_end_idx = min(len(wl) - 1, best_idx + half_width)

    if adaptive_enabled and max_abs_slope > 0:
        target = slope_fraction * abs_slopes[best_idx]
        max_hw = max(1, min(adaptive_max_half, len(wl) // 2))
        left = best_idx
        while left > 0 and abs_slopes[left - 1] >= target and (best_idx - (left - 1)) <= max_hw:
            left -= 1
        right = best_idx
        while right < len(wl) - 1 and abs_slopes[right + 1] >= target and ((right + 1) - best_idx) <= max_hw:
            right += 1
        roi_start_idx = min(roi_start_idx, left)
        roi_end_idx = max(roi_end_idx, right)

    # Validation
    validation_result: dict[str, object] = {}
    if cfg.expected_center is not None:
        observed = float(wl[best_idx])
        deviation = observed - float(cfg.expected_center)
        validation_result = {
            "expected_center": float(cfg.expected_center),
            "observed_center": observed,
            "tolerance": float(cfg.center_tolerance),
            "deviation": float(deviation),
            "within_tolerance": bool(abs(deviation) <= abs(cfg.center_tolerance)),
            "notes": cfg.validation_notes,
        }

    # Top-K candidates
    def _candidate_allowed(wavelength: float) -> bool:
        if cfg.expected_center is not None and cfg.center_tolerance:
            ec, tol = float(cfg.expected_center), abs(cfg.center_tolerance)
            return (ec - tol) <= wavelength <= (ec + tol)
        if min_wavelength is not None and wavelength < float(min_wavelength):
            return False
        return not (max_wavelength is not None and wavelength > float(max_wavelength))

    candidates: list[dict[str, float]] = []
    if top_k_candidates > 0:
        valid = idx_all[~np.isnan(score)]
        if valid.size > 0:
            order = valid[np.argsort(score[valid])[::-1]]
            filtered = [int(i) for i in order if _candidate_allowed(float(wl[int(i)]))]
            for i in filtered[: int(top_k_candidates)]:
                s_i = max(0, i - half_width)
                e_i = min(len(wl) - 1, i + half_width)
                stn = float(noise_per_band[i]) if noise_per_band is not None and 0 <= i < noise_per_band.size else float("nan")
                candidates.append({
                    "wavelength": float(wl[i]),
                    "r2": float(r2_arr[i]) if not np.isnan(r2_arr[i]) else float("nan"),
                    "slope": float(slopes_arr[i]),
                    "slope_to_noise": stn,
                    "corr": float(corr_arr[i]) if not np.isnan(corr_arr[i]) else float("nan"),
                    "score": float(score[i]) if not np.isnan(score[i]) else float("nan"),
                    "roi_start_wavelength": float(wl[s_i]),
                    "roi_end_wavelength": float(wl[e_i]),
                })

    response: dict[str, object] = {
        "wavelengths": wl.tolist(),
        "slopes": slopes_arr.tolist(),
        "intercepts": intercepts,
        "r_squared": r2_vals,
        "poly_r_squared": poly_r2_vals if alt_enabled else None,
        "correlations": corr_arr.tolist(),
        "max_correlation": float(corr_arr[best_idx]) if not np.isnan(corr_arr[best_idx]) else float("nan"),
        "max_slope": float(slopes_arr[best_idx]),
        "max_slope_wavelength": float(wl[best_idx]),
        "max_r_squared": float(r2_arr[best_idx]) if not np.isnan(r2_arr[best_idx]) else float("nan"),
        "max_poly_r_squared": float(poly_r2_arr[best_idx]) if alt_enabled and not np.isnan(poly_r2_arr[best_idx]) else None,
        "roi_selection_metric": selection_metric,
        "roi_score": float(score[best_idx]) if not np.isnan(score[best_idx]) else float("nan"),
        "roi_start_index": int(roi_start_idx),
        "roi_end_index": int(roi_end_idx),
        "roi_start_wavelength": float(wl[roi_start_idx]),
        "roi_end_wavelength": float(wl[roi_end_idx]),
        "validation": validation_result,
        "candidates": candidates,
    }

    return response, avg_by_conc
