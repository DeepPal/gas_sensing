"""Pure metric aggregation utilities for LSPR calibration reporting.

All functions are CONFIG-free — every threshold and parameter is passed
explicitly so results are reproducible without a global CONFIG object.

Typical pipeline flow::

    noise_map = compute_noise_metrics_map(aggregated)
    repeatability = compute_roi_repeatability(stable_by_conc, response)
    performance = compute_roi_performance(repeatability)
    dynamics = summarize_dynamics_metrics(dynamics_df)
"""
from __future__ import annotations
from typing import Any

from collections.abc import Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.preprocessing.quality import estimate_noise_metrics


# ---------------------------------------------------------------------------
# Signal column helpers
# ---------------------------------------------------------------------------


def select_signal_column(df: pd.DataFrame) -> str:
    """Return the best available signal column in *df*.

    Priority: ``"absorbance"`` → ``"transmittance"`` → ``"intensity"``.

    Args:
        df: Spectrum DataFrame; must contain at least one of the priority columns.

    Returns:
        Column name string.  Falls back to ``"intensity"`` if none of the
        preferred columns exist (callers should validate *df* has this column).

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"wavelength": [700.0], "transmittance": [0.95]})
        >>> select_signal_column(df)
        'transmittance'
    """
    if "absorbance" in df.columns:
        return "absorbance"
    if "transmittance" in df.columns:
        return "transmittance"
    return "intensity"


def select_common_signal(
    frames: Sequence[pd.DataFrame],
    priority: Sequence[str] = ("transmittance", "intensity", "absorbance"),
) -> str | None:
    """Return the first column in *priority* that is present in *all* frames.

    Args:
        frames: Non-empty sequence of spectrum DataFrames.
        priority: Ordered preference list of column names.

    Returns:
        Column name, or ``None`` if *frames* is empty or no priority column is
        shared across all frames.

    Example:
        >>> import pandas as pd
        >>> dfs = [pd.DataFrame({"wavelength": [700.], "transmittance": [0.9]})] * 3
        >>> select_common_signal(dfs)
        'transmittance'
    """
    if not frames:
        return None
    for col in priority:
        if all(col in df.columns for df in frames):
            return col
    return None


def common_signal_columns(frames: Sequence[pd.DataFrame]) -> list[str]:
    """Return all non-wavelength columns that appear in *every* frame, sorted.

    Args:
        frames: Sequence of spectrum DataFrames.

    Returns:
        Sorted list of common column names, excluding ``"wavelength"``.
        Returns an empty list if *frames* is empty.

    Example:
        >>> import pandas as pd
        >>> dfs = [pd.DataFrame({"wavelength": [700.], "intensity": [0.5], "extra": [1.]})
        ...        for _ in range(2)]
        >>> common_signal_columns(dfs)
        ['extra', 'intensity']
    """
    if not frames:
        return []
    common = set(frames[0].columns) - {"wavelength"}
    for df in frames[1:]:
        common &= set(df.columns) - {"wavelength"}
    return sorted(common)


# ---------------------------------------------------------------------------
# Noise / quality metrics
# ---------------------------------------------------------------------------


def compute_noise_metrics_map(
    aggregated: dict[float, dict[str, pd.DataFrame]],
) -> dict[float, dict[str, Any]]:
    """Compute noise metrics for every (concentration, trial) in *aggregated*.

    Wraps :func:`~src.preprocessing.quality.estimate_noise_metrics` to
    produce a nested dict keyed by ``(concentration_ppm, trial_name)``.

    Args:
        aggregated: Nested mapping ``{concentration_ppm: {trial_name: DataFrame}}``.
            Each DataFrame must have a ``"wavelength"`` column and at least one
            signal column (detected automatically via :func:`select_signal_column`).

    Returns:
        Nested dict ``{concentration_ppm: {trial_name: noise_metrics_dict}}``
        where each leaf is the :func:`dataclasses.asdict` of a
        :class:`~src.preprocessing.quality.NoiseMetrics` instance.

    Example:
        >>> import pandas as pd, numpy as np
        >>> wl = np.linspace(600, 800, 100)
        >>> sig = np.random.default_rng(0).normal(0.5, 0.01, 100)
        >>> aggregated = {0.5: {"t1": pd.DataFrame({"wavelength": wl, "intensity": sig})}}
        >>> m = compute_noise_metrics_map(aggregated)
        >>> all("snr" in v for v in m[0.5].values())
        True
    """
    metrics: dict[float, dict[str, Any]] = {}
    for conc, trials in aggregated.items():
        metrics[conc] = {}
        for trial, df in trials.items():
            col = select_signal_column(df)
            nm = estimate_noise_metrics(df["wavelength"].values, df[col].values)
            metrics[conc][trial] = asdict(nm)
    return metrics


# ---------------------------------------------------------------------------
# ROI repeatability and performance
# ---------------------------------------------------------------------------


def compute_roi_repeatability(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Compute trial-to-trial repeatability statistics within each ROI window.

    For each concentration level the mean and CV of the average signal within
    ``[roi_start_wavelength, roi_end_wavelength]`` is computed across trials.

    Args:
        stable_by_conc: Nested mapping ``{concentration_ppm: {trial_name: DataFrame}}``.
        response: Dict containing ``"roi_start_wavelength"`` and
            ``"roi_end_wavelength"`` (floats, nm).  Falls back to interpolating
            at the midpoint when the wavelength range has no data.

    Returns:
        Dict with keys:

        - ``"per_concentration"`` – ``{str(conc): {"mean_transmittance", "std_transmittance", "cv_transmittance", "trial_count"}}``
        - ``"global"`` – ``{"mean_transmittance", "std_transmittance", "cv_transmittance", "count"}``
        - ``"indices"`` – ``[roi_start_nm, roi_end_nm]``
        - ``"roi_width"`` – span in nm
    """
    start = float(response["roi_start_wavelength"])
    end = float(response["roi_end_wavelength"])
    center = 0.5 * (start + end)
    roi_span = end - start

    repeatability: dict[str, Any] = {
        "indices": [],
        "per_concentration": {},
        "global": {},
    }

    global_vals: list[float] = []
    for conc, trials in stable_by_conc.items():
        trial_means: list[float] = []
        for _, df in trials.items():
            col = select_signal_column(df)
            wl: np.ndarray = df["wavelength"].values
            y: np.ndarray = df[col].values
            mask = (wl >= start) & (wl <= end)
            if not mask.any():
                trial_means.append(float(np.interp(center, wl, y)))
                continue
            trial_means.append(float(np.nanmean(y[mask])))

        if not trial_means:
            continue
        arr = np.array(trial_means, dtype=float)
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        cv = float(std_val / mean_val) if mean_val else float("inf")
        repeatability["per_concentration"][str(conc)] = {  # type: ignore[index]
            "mean_transmittance": mean_val,
            "std_transmittance": std_val,
            "cv_transmittance": cv,
            "trial_count": int(len(trial_means)),
        }
        global_vals.extend(trial_means)

    if global_vals:
        gv = np.array(global_vals, dtype=float)
        g_std = float(np.nanstd(gv, ddof=1)) if gv.size > 1 else 0.0
        g_mean = float(np.nanmean(gv))
        repeatability["global"] = {
            "mean_transmittance": g_mean,
            "std_transmittance": g_std,
            "cv_transmittance": float(g_std / g_mean) if g_mean else float("inf"),
            "count": int(gv.size),
        }

    repeatability["indices"] = [start, end]
    repeatability["roi_width"] = roi_span
    return repeatability


def compute_roi_performance(
    repeatability: dict[str, Any],
    lod_sigma: float = 3.0,
    loq_sigma: float = 10.0,
) -> dict[str, Any]:
    """Derive LOD, LOQ, sensitivity, and regression quality from repeatability data.

    Args:
        repeatability: Dict returned by :func:`compute_roi_repeatability`.
        lod_sigma: Multiplier for LOD estimate: ``LOD = lod_sigma * σ_global / |slope|``.
            Default 3 (ICH Q2 / 3σ convention).
        loq_sigma: Multiplier for LOQ estimate.  Default 10 (ICH Q2 convention).

    Returns:
        Performance dict with keys:

        - ``"regression_slope"`` / ``"regression_intercept"`` / ``"regression_r2"`` / ``"regression_rmse"``
        - ``"dynamic_range"`` – absolute signal range across concentrations
        - ``"dynamic_range_per_ppm"`` – sensitivity (signal range / ppm span)
        - ``"mean_cv"`` / ``"max_cv"`` / ``"min_cv"`` – CV statistics across concentrations
        - ``"lod_ppm"`` / ``"loq_ppm"`` – limit of detection/quantitation (ppm)
        - ``"ppm_span"`` – concentration range covered
        - Per-concentration arrays: ``"concentrations"``, ``"mean_transmittance_per_concentration"``, etc.

        Returns empty dict if fewer than 2 concentrations are available.
    """
    per_conc = repeatability.get("per_concentration", {})
    if not per_conc:
        return {}

    concs: list[float] = []
    means: list[float] = []
    stds: list[float] = []
    cvs: list[float] = []

    for conc_str, stats in per_conc.items():
        try:
            conc_val = float(conc_str)
        except ValueError:
            continue
        concs.append(conc_val)
        means.append(float(stats.get("mean_transmittance", float("nan"))))  # type: ignore[union-attr]
        stds.append(float(stats.get("std_transmittance", float("nan"))))  # type: ignore[union-attr]
        cvs.append(float(stats.get("cv_transmittance", float("nan"))))  # type: ignore[union-attr]

    if len(concs) < 2:
        return {}

    order = np.argsort(concs)
    concs_arr: np.ndarray = np.array(concs)[order]
    means_arr: np.ndarray = np.array(means)[order]
    stds_arr: np.ndarray = np.array(stds)[order]
    cvs_arr: np.ndarray = np.array(cvs)[order]

    reg = linregress(concs_arr, means_arr)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)
    rmse = float(np.sqrt(np.mean((means_arr - (intercept + slope * concs_arr)) ** 2)))

    dynamic_range = float(np.nanmax(means_arr) - np.nanmin(means_arr))
    ppm_span = float(np.nanmax(concs_arr) - np.nanmin(concs_arr)) or 1.0
    sensitivity = dynamic_range / ppm_span

    global_stats = repeatability.get("global", {})
    global_std = float(global_stats.get("std_transmittance", float("nan")) or 0.0)  # type: ignore[union-attr]
    if slope == 0.0:
        lod = float("inf")
        loq = float("inf")
    else:
        lod = float(lod_sigma * global_std / abs(slope))
        loq = float(loq_sigma * global_std / abs(slope))

    return {
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r2": r2,
        "regression_rmse": rmse,
        "dynamic_range": dynamic_range,
        "dynamic_range_per_ppm": sensitivity,
        "mean_cv": float(np.nanmean(cvs_arr)),
        "max_cv": float(np.nanmax(cvs_arr)),
        "min_cv": float(np.nanmin(cvs_arr)),
        "lod_ppm": lod,
        "loq_ppm": loq,
        "ppm_span": ppm_span,
        "concentrations": concs_arr.tolist(),
        "mean_transmittance_per_concentration": means_arr.tolist(),
        "std_transmittance_per_concentration": stds_arr.tolist(),
        "cv_transmittance_per_concentration": cvs_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Dynamics and comparison summaries
# ---------------------------------------------------------------------------


def summarize_dynamics_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute T90/T10 response-time statistics grouped by concentration.

    Args:
        df: DataFrame with columns ``"concentration"``, ``"response_time_T90"``
            (time to reach 90 % of steady-state), and ``"recovery_time_T10"``
            (time to fall back to 10 % above baseline).  Non-finite values are
            silently excluded from statistics.

    Returns:
        Dict with:

        - ``"per_concentration"`` – ``{str(conc): {"mean_T90", "std_T90", "mean_T10", "std_T10", "count"}}``
        - ``"overall"`` – same statistics across all rows

        Returns empty dict if *df* is empty.
    """
    if df.empty:
        return {}

    df = df.replace([np.inf, -np.inf], np.nan)
    metrics: dict[str, Any] = {"per_concentration": {}}

    for conc, group in df.groupby("concentration"):
        conc_key = str(float(conc))
        metrics["per_concentration"][conc_key] = {  # type: ignore[index]
            "mean_T90": float(group["response_time_T90"].mean(skipna=True)),
            "std_T90": float(group["response_time_T90"].std(skipna=True) or 0.0),
            "mean_T10": float(group["recovery_time_T10"].mean(skipna=True)),
            "std_T10": float(group["recovery_time_T10"].std(skipna=True) or 0.0),
            "count": int(group.shape[0]),
        }

    metrics["overall"] = {
        "mean_T90": float(df["response_time_T90"].mean(skipna=True)),
        "std_T90": float(df["response_time_T90"].std(skipna=True) or 0.0),
        "mean_T10": float(df["recovery_time_T10"].mean(skipna=True)),
        "std_T10": float(df["recovery_time_T10"].std(skipna=True) or 0.0),
        "count": int(df.shape[0]),
    }

    return metrics


def summarize_quality_control(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    noise_metrics: dict[float, dict[str, Any]],
    *,
    min_snr: float = 10.0,
    max_rsd: float = 5.0,
) -> dict[str, Any]:
    """Summarise dataset-level QC: SNR distribution and trial-to-trial RSD.

    Args:
        stable_by_conc: Nested ``{concentration_ppm: {trial_name: DataFrame}}``.
        noise_metrics: Dict as returned by :func:`compute_noise_metrics_map`.
        min_snr: Minimum acceptable signal-to-noise ratio (default 10).
        max_rsd: Maximum acceptable trial-to-trial RSD in % (default 5).

    Returns:
        Dict with keys ``"min_snr"``, ``"median_snr"``, ``"max_rsd_percent"``,
        ``"snr_threshold"``, ``"rsd_threshold_percent"``, ``"snr_pass"``,
        ``"rsd_pass"``, ``"overall_pass"``, ``"rsd_by_concentration"``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> df = pd.DataFrame({"wavelength": np.linspace(680, 750, 50),
        ...                    "intensity": np.ones(50) * 0.5})
        >>> nm = {0.5: {"t0": {"snr": 15.0, "rms": 0.01}}}
        >>> qc = summarize_quality_control({0.5: {"t0": df}}, nm, min_snr=10.0)
        >>> qc["snr_pass"]
        True
    """
    snr_values: list[float] = []
    for _conc, trials in (noise_metrics or {}).items():
        try:
            for tinfo in trials.values():
                snr = float(tinfo.get("snr", float("nan")))  # type: ignore[union-attr]
                if np.isfinite(snr):
                    snr_values.append(snr)
        except Exception:
            continue

    rsd_by_conc: dict[float, float] = {}
    for conc, trials in (stable_by_conc or {}).items():
        vals: list[float] = []
        for df in trials.values():
            col = select_signal_column(df)
            arr = df[col].to_numpy(dtype=float)
            if arr.size:
                vals.append(float(np.mean(arr)))
        if len(vals) >= 2:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            rsd = std_v / mean_v * 100.0 if abs(mean_v) > 1e-12 else float("inf")
            rsd_by_conc[float(conc)] = rsd

    max_rsd_obs = float(np.nanmax(list(rsd_by_conc.values()))) if rsd_by_conc else float("nan")

    qc: dict[str, Any] = {
        "min_snr": float(np.nanmin(snr_values)) if snr_values else float("nan"),
        "median_snr": float(np.nanmedian(snr_values)) if snr_values else float("nan"),
        "max_rsd_percent": max_rsd_obs,
        "snr_threshold": float(min_snr),
        "rsd_threshold_percent": float(max_rsd),
    }
    qc["snr_pass"] = bool(np.isfinite(qc["min_snr"]) and float(qc["min_snr"]) >= min_snr)
    qc["rsd_pass"] = bool(
        np.isfinite(qc["max_rsd_percent"]) and float(qc["max_rsd_percent"]) <= max_rsd
    )
    qc["overall_pass"] = bool(qc["snr_pass"]) and bool(qc["rsd_pass"])
    qc["rsd_by_concentration"] = {str(k): float(v) for k, v in rsd_by_conc.items()}
    return qc


def summarize_top_comparison(
    results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten a multi-mode analysis result dict into a comparison table.

    Args:
        results: Mapping of ``{mode_name: payload_dict}`` where each payload
            contains ``"performance"``, ``"response_stats"``, ``"canonical_count"``,
            ``"metrics_path"``, and ``"plot_path"`` keys (all optional).

    Returns:
        List of dicts with keys: ``"mode"``, ``"canonical_count"``,
        ``"roi_max_r2"``, ``"roi_max_slope"``, ``"roi_center"``,
        ``"lod"``, ``"loq"``, ``"metrics_path"``, ``"plot_path"``.
        One entry per mode, in iteration order of *results*.
    """
    summary: list[dict[str, Any]] = []
    for mode, payload in results.items():
        perf = payload.get("performance", {}) or {}
        roi_perf = perf.get("roi_performance", {}) if isinstance(perf, dict) else {}
        summary.append(
            {
                "mode": mode,
                "canonical_count": payload.get("canonical_count"),
                "roi_max_r2": payload.get("response_stats", {}).get("max_r_squared"),
                "roi_max_slope": payload.get("response_stats", {}).get("max_slope"),
                "roi_center": payload.get("response_stats", {}).get("max_slope_wavelength"),
                "lod": roi_perf.get("lod"),
                "loq": roi_perf.get("loq"),
                "metrics_path": payload.get("metrics_path"),
                "plot_path": payload.get("plot_path"),
            }
        )
    return summary
