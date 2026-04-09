from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, nnls
from scipy.signal import correlate, savgol_filter
from scipy.stats import linregress, probplot
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy as sp
import sklearn as sk
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor
from sklearn.metrics import mean_squared_error, r2_score

from config.config_loader import load_config

CONFIG = load_config()
REPO_ROOT = Path(__file__).resolve().parents[2]

import contextlib

from src.batch.aggregation import (
    average_stable_block as _average_stable_block_src,
)
from src.batch.aggregation import (
    average_top_frames as _average_top_frames_src,
)
from src.batch.aggregation import (
    find_stable_block as _find_stable_block_src,
)
from src.batch.aggregation import (
    select_canonical_per_concentration as _select_canonical_src,
)
from src.batch.preprocessing import sort_frame_paths as _sort_frame_paths_src
from src.batch.response import (
    aggregate_responsive_delta_maps as _aggregate_responsive_delta_maps_src,
)
from src.batch.response import (
    scale_reference_to_baseline as _scale_reference_to_baseline_src,
)
from src.batch.response import (
    score_trial_quality as _score_trial_quality_src,
)
from src.batch.response import (
    summarize_responsive_delta as _summarize_responsive_delta_src,
)
from src.calibration.multi_roi import (
    fit_multi_roi_fusion as _fit_multi_roi_fusion,
)
from src.calibration.multi_roi import (
    select_multi_roi_candidates as _select_multi_roi_candidates,
)
from src.calibration.roi_scan import (
    RoiScanConfig as _RoiScanConfig,
)
from src.calibration.roi_scan import (
    compute_concentration_response as _compute_concentration_response_pure,
)
from src.calibration.roi_scan import (
    stack_trials_for_response as _stack_trials_for_response,
)
from src.calibration.transforms import transform_concentrations as _transform_concentrations
from src.reporting.environment import (
    compute_environment_coefficients as _compute_environment_coefficients_pure,
)
from src.reporting.environment import (
    compute_environment_summary as _compute_environment_summary_pure,
)
from src.reporting.io import (
    save_aggregated_spectra as _save_aggregated_spectra_io,
)
from src.reporting.io import (
    save_aggregated_summary as _save_aggregated_summary_io,
)
from src.reporting.io import (
    save_canonical_spectra as _save_canonical_spectra_io,
)
from src.reporting.io import (
    save_concentration_response_metrics as _save_concentration_response_metrics_io,
)
from src.reporting.io import (
    save_dynamics_error as _save_dynamics_error_io,
)
from src.reporting.io import (
    save_dynamics_summary as _save_dynamics_summary_io,
)
from src.reporting.io import (
    save_environment_compensation_summary as _save_env_compensation_summary_io,
)
from src.reporting.io import (
    save_noise_metrics as _save_noise_metrics_io,
)
from src.reporting.io import (
    save_quality_summary as _save_quality_summary_io,
)
from src.reporting.io import (
    save_roi_performance_metrics as _save_roi_performance_metrics_io,
)
from src.reporting.metrics import (
    common_signal_columns as _common_signal_columns,
)
from src.reporting.metrics import (
    compute_noise_metrics_map,
    compute_roi_performance,
    compute_roi_repeatability,
    summarize_dynamics_metrics,
    summarize_top_comparison,
)
from src.reporting.metrics import (
    select_common_signal as _select_common_signal,
)
from src.reporting.metrics import (
    summarize_quality_control as _summarize_quality_control_pure,
)
from src.reporting.plots import (
    save_aggregated_plots as _save_aggregated_plots_src,
)
from src.reporting.plots import (
    save_calibration_outputs as _save_calibration_outputs_src,
)
from src.reporting.plots import (
    save_canonical_overlay as _save_canonical_overlay_src,
)
from src.reporting.plots import (
    save_concentration_response_plot as _save_concentration_response_plot_pure,
)
from src.reporting.plots import (
    save_research_grade_calibration_plot as _save_research_grade_calibration_plot_src,
)
from src.reporting.plots import (
    save_roi_discovery_plot as _save_roi_discovery_plot_src,
)
from src.reporting.plots import (
    save_roi_repeatability_plot as _save_roi_repeatability_plot_src,
)
from src.reporting.plots import (
    save_spectral_response_diagnostic as _save_spectral_response_diagnostic_pure,
)
from src.reporting.plots import (
    save_wavelength_shift_visualization as _save_wavelength_shift_visualization_src,
)
from src.scientific.regression import (
    ransac as _ransac,
)
from src.scientific.regression import (
    theil_sen as _theil_sen,
)
from src.scientific.regression import (
    weighted_linear as _weighted_linear,
)
from src.signal.peak import (
    estimate_shift_crosscorr as _estimate_shift_crosscorr,
)
from src.signal.peak import (
    gaussian_peak_center as _gaussian_peak_center,
)
from src.signal.roi import (
    compute_band_ratio_matrix as _compute_band_ratio_matrix,
)
from src.signal.roi import (
    find_monotonic_wavelengths as _find_monotonic_wavelengths,
)

# ---------------------------------------------------------------------------
# Migrated to src/ — imported here as aliases so all internal callers work
# unchanged. The duplicate function bodies below have been removed.
# ---------------------------------------------------------------------------
from src.signal.transforms import (
    append_absorbance_column as _append_absorbance_column,
)
from src.signal.transforms import (
    compute_transmittance,
)
from src.signal.transforms import (
    ensure_odd_window as _ensure_odd_window,
)
from src.signal.transforms import (
    smooth as _smooth,
)

from ..advanced.mcr_als import (
    fit_mcrals_from_canonical,
    save_mcrals_outputs,
)
from .calibration.methods import (
    _discover_roi_in_band,
    _prepare_calibration_signal,
    _resolve_roi_bounds,
    _signal_column,
    find_roi_and_calibration,
    perform_absorbance_amplitude_calibration,
)
from .calibration.plsr import (
    build_feature_matrix_from_canonical as _build_feature_matrix_from_canonical,
)
from .calibration.plsr import (
    fit_plsr_calibration as _fit_plsr_calibration,
)
from .preprocessing import (
    baseline_correction,
    detect_outliers,
    downsample_spectrum,
    estimate_noise_metrics,
    normalize_spectrum,
    smooth_spectrum,
)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _autosave_env_coefficients_to_config(est: dict[str, object]) -> bool:
    """Persist estimated environment coefficients into config/config.yaml if enabled.

    Returns True if write succeeded, else False.
    """
    try:
        cfg_path = REPO_ROOT / "config" / "config.yaml"
        if not cfg_path.exists():
            return False
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        env = cfg.get("environment", {}) if isinstance(cfg.get("environment", {}), dict) else {}
        if not env.get("autosave_coefficients", False):
            return False
        coeffs = (
            env.get("coefficients", {}) if isinstance(env.get("coefficients", {}), dict) else {}
        )
        t = est.get("temperature")
        h = est.get("humidity")
        if t is not None:
            coeffs["temperature"] = float(t)
        if h is not None:
            coeffs["humidity"] = float(h)
        env["coefficients"] = coeffs
        cfg["environment"] = env
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return True
    except Exception:
        return False


# ----------------------
def _preprocess_dataframe(df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    settings = CONFIG.get("preprocessing", {})
    if not settings.get("enabled", False):
        return df

    stage_norm = stage.lower()
    if stage_norm == "frame" and not settings.get("apply_to_frames", False):
        return df
    if stage_norm == "transmittance" and not settings.get("apply_to_transmittance", True):
        return df

    if "wavelength" not in df.columns:
        return df

    preferred_col = None
    if stage_norm == "transmittance" and "transmittance" in df.columns:
        preferred_col = "transmittance"
    elif "intensity" in df.columns:
        preferred_col = "intensity"
    else:
        other_cols = [c for c in df.columns if c != "wavelength"]
        if other_cols:
            preferred_col = other_cols[0]
        else:
            return df

    is_transmittance = preferred_col == "transmittance"

    wl = df["wavelength"].to_numpy(copy=True)
    signal = df[preferred_col].to_numpy(copy=True)

    smooth_cfg = settings.get("smooth", {})
    if smooth_cfg.get("enabled", False):
        window = smooth_cfg.get("window", 21)
        poly_order = smooth_cfg.get("poly_order", 3)
        method = smooth_cfg.get("method", "savgol")
        signal = smooth_spectrum(signal, window=window, poly_order=poly_order, method=method)
        if smooth_cfg.get("extra_pass", False):
            signal = smooth_spectrum(signal, window=window, poly_order=poly_order, method=method)

    if not is_transmittance:
        baseline_cfg = settings.get("baseline", {})
        if baseline_cfg.get("enabled", False):
            method = baseline_cfg.get("method", "polynomial")
            order = baseline_cfg.get("order", 2)
            signal = baseline_correction(wl, signal, method=method, poly_order=order)

    norm_cfg = settings.get("normalization", {})
    if norm_cfg.get("enabled", False):
        signal = normalize_spectrum(signal, method=norm_cfg.get("method", "minmax"))

    # Optional environment compensation (additive offset in measurement units)
    try:
        env_cfg = CONFIG.get("environment", {})
        if env_cfg.get("enabled", False):
            apply_frames = bool(env_cfg.get("apply_to_frames", False))
            apply_trans = bool(env_cfg.get("apply_to_transmittance", True))
            if (stage_norm == "frame" and apply_frames) or (
                stage_norm == "transmittance" and apply_trans
            ):
                ref = (
                    env_cfg.get("reference", {})
                    if isinstance(env_cfg.get("reference", {}), dict)
                    else {}
                )
                coeffs = (
                    env_cfg.get("coefficients", {})
                    if isinstance(env_cfg.get("coefficients", {}), dict)
                    else {}
                )
                override = (
                    env_cfg.get("override", {})
                    if isinstance(env_cfg.get("override", {}), dict)
                    else {}
                )

                # Determine environment values from DataFrame columns or overrides
                T_ref = float(ref.get("temperature", 25.0))
                H_ref = float(ref.get("humidity", 50.0))
                T_val = None
                H_val = None
                if "temperature" in df.columns:
                    try:
                        T_val = float(
                            pd.to_numeric(df["temperature"], errors="coerce").dropna().mean()
                        )
                    except Exception:
                        T_val = None
                if "humidity" in df.columns:
                    try:
                        H_val = float(
                            pd.to_numeric(df["humidity"], errors="coerce").dropna().mean()
                        )
                    except Exception:
                        H_val = None
                if T_val is None and override.get("temperature") is not None:
                    T_val = float(override.get("temperature"))
                if H_val is None and override.get("humidity") is not None:
                    H_val = float(override.get("humidity"))

                # Apply additive offset using provided coefficients
                offset = 0.0
                cT = coeffs.get("temperature", None)
                cH = coeffs.get("humidity", None)
                if cT is not None and T_val is not None:
                    offset += float(cT) * (T_val - T_ref)
                if cH is not None and H_val is not None:
                    offset += float(cH) * (H_val - H_ref)
                if offset != 0.0 and np.isfinite(offset):
                    signal = signal - float(offset)
    except Exception:
        pass

    ds_cfg = settings.get("downsample", {})
    new_wl = wl
    if ds_cfg.get("enabled", False):
        new_wl, signal = downsample_spectrum(
            wl,
            signal,
            factor=ds_cfg.get("factor"),
            target_points=ds_cfg.get("target_points"),
            method="average",
        )

    out_df = df.copy()
    if len(new_wl) != len(wl):
        out_df = pd.DataFrame({"wavelength": new_wl})
        for col in df.columns:
            if col == "wavelength":
                continue
            source = df[col].to_numpy(copy=True)
            if len(source) != len(wl):
                continue
            out_df[col] = np.interp(new_wl, wl, source)
    out_df[preferred_col] = signal

    return out_df


def _record_outlier(metadata: dict[str, object], conc: float, trial: str):
    metadata.setdefault("dropped_outliers", []).append({"concentration": conc, "trial": trial})


def _save_run_metadata(out_root: str, metadata: dict[str, object]) -> Path:
    metrics_dir = Path(out_root) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metrics_dir / "run_metadata.json"
    _write_json(meta_path, metadata)
    return meta_path


def _archive_run(out_root: str, metadata: dict[str, object]):
    archive_cfg = CONFIG.get("archiving", {})
    if not archive_cfg.get("enabled", False):
        return None

    out_path = Path(out_root)
    archive_dir = out_path / "archives" / metadata["run_timestamp"]
    archive_dir.mkdir(parents=True, exist_ok=True)

    for rel in [
        "metrics",
        "plots",
        "dynamics",
        "reports",
        "aggregated",
        "stable_selected",
    ]:
        src = out_path / rel
        if src.exists():
            dst = archive_dir / rel
            _copy_tree(src, dst)

    for extra_file in ["run_metadata.json"]:
        src = out_path / "metrics" / extra_file
        if src.exists():
            shutil.copy2(src, archive_dir / extra_file)

    return archive_dir


def _invoke_report_generation(out_root: str, metadata: dict[str, object]) -> dict[str, object]:
    reporting = CONFIG.get("reporting", {})
    results: dict[str, object] = {}
    if not reporting:
        return results

    run_dir = Path(out_root)
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_script = REPO_ROOT / "scripts" / "generate_report.py"
    notebook_path = reports_dir / "analysis_report.ipynb"

    if reporting.get("generate_notebook", True) and report_script.exists():
        cmd = [sys.executable, str(report_script), "--run", str(run_dir)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results["notebook"] = str(notebook_path)
        except subprocess.CalledProcessError as exc:
            results["notebook_error"] = exc.stderr.decode("utf-8", errors="ignore")
        else:
            try:
                summary_md = reports_dir / "summary.md"
                md_text = (
                    summary_md.read_text(encoding="utf-8")
                    if summary_md.exists()
                    else "# Gas Analysis Report\n\nSummary not found."
                )
                cells = []
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [
                            "# Gas Analysis Report\n",
                            "\n",
                            "This notebook aggregates key figures and metrics for peer review.\n",
                        ],
                    }
                )
                cells.append(
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": md_text.splitlines(True),
                    }
                )
                # Embed key figures if they exist
                try:
                    plots_dir = run_dir / "plots"

                    def _rel(p):
                        try:
                            return os.path.relpath(str(p), start=str(reports_dir))
                        except Exception:
                            return str(p)

                    figs = [
                        (
                            "## Concentration Response",
                            plots_dir / "concentration_response.png",
                        ),
                        (
                            "## ROI Repeatability",
                            plots_dir / "roi_repeatability.png",
                        ),
                        (
                            "## Multivariate CV R² Comparison",
                            plots_dir / "multivariate_cv_r2.png",
                        ),
                        (
                            "## Selected Model: Predicted vs Actual",
                            plots_dir / "selected_pred_vs_actual.png",
                        ),
                        (
                            "## MCR-ALS Components",
                            plots_dir / "mcr_als_components.png",
                        ),
                        (
                            "## MCR-ALS Pred vs Actual",
                            plots_dir / "mcr_als_pred_vs_actual.png",
                        ),
                    ]
                    for title, path in figs:
                        if path.exists():
                            cells.append(
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": [title + "\n"],
                                }
                            )
                            cells.append(
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": [f"![]({_rel(path)})\n"],
                                }
                            )
                except Exception:
                    pass
                nb = {
                    "cells": cells,
                    "metadata": {"language_info": {"name": "python"}},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
                with notebook_path.open("w", encoding="utf-8") as f:
                    json.dump(nb, f, indent=2)
                results["notebook"] = str(notebook_path)
            except Exception as exc:
                results["notebook_error"] = str(exc)

    if reporting.get("export_pdf", False) and notebook_path.exists():
        nbconvert_exe = reporting.get("nbconvert_executable", "jupyter")
        cmd = [nbconvert_exe, "nbconvert", "--to", "pdf", str(notebook_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results["pdf"] = str(notebook_path.with_suffix(".pdf"))
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            results["pdf_error"] = str(exc)

    return results


def _gather_trend_records(run_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    archives_root = run_root / "archives"
    if archives_root.exists():
        for arch in sorted(archives_root.iterdir()):
            perf_path = arch / "metrics" / "roi_performance.json"
            meta_path = arch / "run_metadata.json"
            if perf_path.exists():
                try:
                    performance = json.loads(perf_path.read_text())
                    timestamp = arch.name
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        timestamp = meta.get("run_timestamp", timestamp)
                    records.append({"timestamp": timestamp, "performance": performance})
                except json.JSONDecodeError:
                    continue

    current_perf_path = run_root / "metrics" / "roi_performance.json"
    current_meta_path = run_root / "metrics" / "run_metadata.json"
    if current_perf_path.exists():
        try:
            performance = json.loads(current_perf_path.read_text())
            timestamp = _timestamp()
            if current_meta_path.exists():
                meta = json.loads(current_meta_path.read_text())
                timestamp = meta.get("run_timestamp", timestamp)
            records.append({"timestamp": timestamp, "performance": performance})
        except json.JSONDecodeError:
            pass

    unique = {rec["timestamp"]: rec for rec in records}
    return [unique[k] for k in sorted(unique.keys())]


def generate_trend_plots(out_root: str) -> dict[str, str]:
    reporting = CONFIG.get("reporting", {})
    if not reporting.get("trend_plots", True):
        return {}

    run_root = Path(out_root)
    records = _gather_trend_records(run_root)
    if len(records) < 2:
        return {}

    timestamps = []
    slopes = []
    r2_vals = []
    lod_vals = []

    for rec in records:
        perf = rec["performance"]
        timestamps.append(rec["timestamp"])
        slopes.append(perf.get("regression_slope"))
        r2_vals.append(perf.get("regression_r2"))
        lod_vals.append(perf.get("lod_ppm"))

    plots_dir = run_root / "plots" / "trends"
    plots_dir.mkdir(parents=True, exist_ok=True)
    trend_path = plots_dir / "roi_performance_trend.png"

    plt.figure(figsize=(10, 5))
    x = range(len(timestamps))
    plt.plot(x, slopes, marker="o", label="Slope (dT/ppm)")
    plt.plot(x, r2_vals, marker="s", label="R²")
    plt.plot(x, lod_vals, marker="^", label="LOD (ppm)")
    plt.xticks(x, timestamps, rotation=45, ha="right")
    plt.ylabel("Metric Value")
    plt.title("ROI Performance Trend Across Runs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(trend_path, dpi=300)
    plt.close()

    return {"roi_performance": str(trend_path)}


# IO and utilities
# ----------------------


def _read_csv_spectrum(path: str) -> pd.DataFrame:
    """Read a spectrum CSV and normalize columns to wavelength,intensity.
    Accepts headerless files as two columns.
    """
    try:
        df = pd.read_csv(path)
        if "wavelength" not in df.columns or "intensity" not in df.columns:
            df = pd.read_csv(path, header=None, names=["wavelength", "intensity"])
        df = df[["wavelength", "intensity"]].copy()
        df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df = df.dropna()
        return df.sort_values("wavelength").reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read spectrum {path}: {e}")


def _sort_frame_paths(paths: Sequence[str]) -> list[str]:
    return _sort_frame_paths_src(paths)


def _ensure_response_signal(df: pd.DataFrame, ref_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    updated = df
    has_trans = "transmittance" in df.columns
    if not has_trans and ref_df is not None and "intensity" in df.columns:
        updated = compute_transmittance(df, ref_df)
    if "absorbance" not in updated.columns:
        updated = _append_absorbance_column(updated)
    return updated


def _compute_response_time_series(
    frames: Sequence[pd.DataFrame],
    ref_df: Optional[pd.DataFrame],
    *,
    dataset_label: Optional[str],
    response_cfg: dict[str, object],
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Thin adapter — preprocesses DataFrames and delegates to src.batch.time_series."""
    from src.batch.time_series import compute_response_time_series as _rts

    if not frames:
        return None, [], []

    processed = [_ensure_response_signal(df, ref_df) for df in frames]
    absorb_col = (
        "absorbance" if "absorbance" in processed[0].columns else _signal_column(processed[0])
    )

    base_wl = processed[0]["wavelength"].to_numpy()
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = float(base_wl.min())
    if max_wl_roi is None:
        max_wl_roi = float(base_wl.max())
    roi_mask = (base_wl >= min_wl_roi) & (base_wl <= max_wl_roi)
    if not np.any(roi_mask):
        roi_mask = np.ones_like(base_wl, dtype=bool)
    roi_wavelengths = base_wl[roi_mask]

    absorb_rows: list[np.ndarray] = []
    mean_absorb_list: list[float] = []
    roi_rows: list[np.ndarray] = []
    for df in processed:
        wl = df["wavelength"].to_numpy()
        signal = df[absorb_col].to_numpy(dtype=float)
        if not np.array_equal(wl, base_wl):
            signal = np.interp(base_wl, wl, signal)
        absorb_rows.append(signal)
        mean_absorb_list.append(float(np.nanmean(signal)))
        roi_rows.append(signal[roi_mask])
    absorb_matrix = np.vstack(absorb_rows)
    mean_absorb = np.array(mean_absorb_list, dtype=float)
    roi_matrix = np.vstack(roi_rows)

    cp_raw = response_cfg.get("changepoint", {})
    changepoint_cfg: dict[str, Any] = cp_raw if isinstance(cp_raw, dict) else {}

    return _rts(
        absorb_matrix,
        roi_matrix,
        mean_absorb,
        roi_wavelengths,
        float(min_wl_roi),
        float(max_wl_roi),
        len(frames),
        dataset_label,
        smooth_window=int(response_cfg.get("smooth_window", 5) or 5),
        baseline_frames=int(response_cfg.get("baseline_frames", 12) or 12),
        activation_delta=float(response_cfg.get("activation_delta", 0.01) or 0.01),
        sigma_multiplier=float(response_cfg.get("activation_sigma_multiplier", 1.5) or 1.5),
        noise_floor=float(response_cfg.get("noise_floor", 1e-4) or 1e-4),
        slope_sigma_multiplier=float(response_cfg.get("slope_sigma_multiplier", 1.0) or 1.0),
        min_response_slope=float(response_cfg.get("min_response_slope", 0.0) or 0.0),
        min_activation_frames=int(response_cfg.get("min_activation_frames", 6) or 6),
        min_activation_fraction=float(response_cfg.get("min_activation_fraction", 0.08) or 0.08),
        fallback_window=int(response_cfg.get("fallback_window", 4) or 4),
        monotonic_tolerance_nm=float(response_cfg.get("monotonic_tolerance_nm", 0.05) or 0.0),
        changepoint_cfg=changepoint_cfg,
    )


def _scale_reference_to_baseline(
    ref_df: Optional[pd.DataFrame],
    baseline_frames: Sequence[pd.DataFrame],
    percentile: float = 95.0,
) -> tuple[Optional[pd.DataFrame], float]:
    return _scale_reference_to_baseline_src(
        ref_df, baseline_frames, percentile=percentile
    )


def _score_trial_quality(
    df: pd.DataFrame,
    *,
    roi_bounds: tuple[Optional[float], Optional[float]],
    expected_center: Optional[float],
) -> tuple[float, dict[str, float]]:
    return _score_trial_quality_src(
        df, roi_bounds=roi_bounds, expected_center=expected_center
    )


def _simple_response_selection(
    frames: Sequence[pd.DataFrame],
    ref_df: Optional[pd.DataFrame],
    *,
    dataset_label: Optional[str],
    response_cfg: dict[str, object],
) -> tuple[pd.DataFrame, list[int], list[int]]:
    """Lightweight frame selection that ranks frames by ROI absorbance energy."""
    if not frames:
        return None, [], []

    processed = [_ensure_response_signal(df, ref_df) for df in frames]
    base_wl = processed[0]["wavelength"].to_numpy()
    absorb_col = (
        "absorbance" if "absorbance" in processed[0].columns else _signal_column(processed[0])
    )

    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = float(base_wl.min())
    if max_wl_roi is None:
        max_wl_roi = float(base_wl.max())
    roi_mask = (base_wl >= min_wl_roi) & (base_wl <= max_wl_roi)
    if not np.any(roi_mask):
        roi_mask = np.ones_like(base_wl, dtype=bool)

    response_metrics: list[float] = []
    for df in processed:
        wl = df["wavelength"].to_numpy(dtype=float)
        signal = df[absorb_col].to_numpy(dtype=float)
        if not np.array_equal(wl, base_wl):
            signal = np.interp(base_wl, wl, signal)
        roi_signal = signal[roi_mask]
        metric = float(np.nansum(np.clip(roi_signal, 0.0, None)))
        response_metrics.append(metric)

    response_arr = np.array(response_metrics, dtype=float)
    smooth_window = int(response_cfg.get("simple_smooth_window", 5) or 5)
    if smooth_window > 1 and response_arr.size >= smooth_window:
        kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
        smoothed = np.convolve(response_arr, kernel, mode="same")
    else:
        smoothed = response_arr.copy()

    top_n = int(response_cfg.get("top_n_frames", 20) or 20)
    top_n = max(1, min(top_n, len(frames)))
    if np.all(~np.isfinite(smoothed)):
        smoothed = np.nan_to_num(smoothed, nan=0.0)

    finite_mask = np.isfinite(smoothed)
    if not np.any(finite_mask):
        smoothed = np.zeros_like(smoothed)
        finite_mask = np.ones_like(smoothed, dtype=bool)

    rank_indices = np.argsort(smoothed[finite_mask])
    finite_indices = np.where(finite_mask)[0][rank_indices]
    if finite_indices.size >= top_n:
        top_indices = finite_indices[-top_n:]
    else:
        fallback = np.arange(len(frames))[-top_n:]
        top_indices = np.unique(np.concatenate([finite_indices, fallback]))[-top_n:]

    top_indices = sorted(int(idx) for idx in top_indices)

    time_series_df = pd.DataFrame(
        {
            "frame_index": np.arange(len(frames)),
            "response_metric": response_arr,
            "smoothed_response": smoothed,
            "selected": [idx in top_indices for idx in range(len(frames))],
        }
    )

    return time_series_df, top_indices, top_indices


def _save_response_series(
    df: pd.DataFrame,
    out_root: str,
    concentration: float,
    trial: str,
    dataset_label: Optional[str],
) -> tuple[Path, Path]:
    series_dir = Path(out_root) / "time_series"
    series_dir.mkdir(parents=True, exist_ok=True)
    safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
    prefix = f"{dataset_label or 'dataset'}_{concentration:g}_{safe_trial}"
    csv_path = series_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)

    plot_path = series_dir / f"{prefix}.png"
    try:
        # Build multi-panel visualization
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        indices = df["frame_index"].to_numpy(dtype=float)
        delta_peak = df.get("delta_peak", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        delta_peak_raw = df.get("delta_peak_raw", pd.Series([np.nan] * len(df))).to_numpy(
            dtype=float
        )
        delta_mean = df.get("delta_mean", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        threshold = df.get("threshold", pd.Series([np.nan])).iloc[0]
        responsive_mask = df["is_responsive"].to_numpy(dtype=int) == 1

        ax0 = axes[0]
        ax0.plot(indices, delta_peak_raw, marker="o", label="Δ peak (raw)")
        ax0.plot(indices, delta_peak, marker="^", label="Δ peak (directed)")
        if np.any(np.isfinite(delta_mean)):
            ax0.plot(indices, delta_mean, marker="s", label="Δ mean")
        if np.isfinite(threshold):
            ax0.axhline(
                threshold,
                color="r",
                linestyle="--",
                linewidth=1.0,
                label="activation threshold",
            )
        if responsive_mask.any():
            ax0.scatter(
                indices[responsive_mask],
                delta_peak_raw[responsive_mask],
                color="red",
                zorder=3,
                label="responsive frames",
            )
            # Robust range calculation for fill_between
            valid_raw = delta_peak_raw[np.isfinite(delta_peak_raw)]
            if valid_raw.size > 0:
                valid_responsive = delta_peak_raw[responsive_mask]
                valid_responsive = valid_responsive[np.isfinite(valid_responsive)]
                if valid_responsive.size > 0:
                    ymin = float(np.min(valid_responsive))
                    ymax = float(np.max(valid_responsive))
                else:
                    ymin = float(np.min(valid_raw))
                    ymax = float(np.max(valid_raw))
            else:
                ymin, ymax = 0.0, 1.0

            ax0.fill_between(
                indices,
                ymin,
                ymax,
                where=responsive_mask,
                color="red",
                alpha=0.08,
                step="mid",
            )
        ax0.set_ylabel("Δ absorbance")
        ax0.legend(loc="best")
        ax0.grid(alpha=0.25)

        ax1 = axes[1]
        peak_wl = df.get("peak_wavelength_nm", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        if np.isfinite(peak_wl).any():
            ax1.plot(indices, peak_wl, color="tab:blue", marker=".", label="peak λ")
        mean_signal = df.get("mean_signal", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        ax1b = ax1.twinx()
        ax1b.plot(indices, mean_signal, color="tab:orange", alpha=0.6, label="mean absorbance")
        if responsive_mask.any():
            ax1.fill_between(
                indices,
                np.nanmin(peak_wl) if np.isfinite(peak_wl).any() else 0,
                np.nanmax(peak_wl) if np.isfinite(peak_wl).any() else 1,
                where=responsive_mask,
                color="red",
                alpha=0.1,
                step="mid",
                label="responsive window",
            )
        ax1.set_ylabel("Peak λ (nm)")
        ax1b.set_ylabel("Mean absorbance (a.u.)")
        ax1.grid(alpha=0.25)

        # Combine legends for bottom axis
        handles, labels = ax1.get_legend_handles_labels()
        handles_b, labels_b = ax1b.get_legend_handles_labels()
        if handles or handles_b:
            ax1.legend(handles + handles_b, labels + labels_b, loc="best")

        axes[-1].set_xlabel("Frame index (acquisition order)")

        title_label = dataset_label or "dataset"
        fig.suptitle(
            f"Response diagnostics: {title_label} {concentration:g} ppm ({trial})",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception:
        with contextlib.suppress(Exception):
            plt.close(fig)
    return csv_path, plot_path


def _summarize_responsive_delta(df: pd.DataFrame) -> dict[str, object]:
    return _summarize_responsive_delta_src(df)


def _aggregate_responsive_delta_maps(
    responsive_delta_by_conc: dict[float, dict[str, dict[str, object]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[float, dict[str, float]]]:
    return _aggregate_responsive_delta_maps_src(responsive_delta_by_conc)


def _safe_float(val: object) -> float:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return fval if np.isfinite(fval) else float("nan")


def _compute_responsive_trend_fallback(summary: dict[str, object]) -> dict[str, float]:
    if not summary:
        return {}

    def _get(name: str) -> float:
        try:
            return float(summary.get(name))
        except (TypeError, ValueError):
            return float("nan")

    responsive_fraction = _get("responsive_fraction")
    signed_consistency = _get("signed_consistency")
    std_delta = _get("std_delta_nm")
    median_delta = _get("median_delta_nm")
    selected_delta = _get("selected_delta_nm")
    fallback_delta = _get("fallback_delta_nm")
    median_peak = _get("median_peak_nm")
    selected_peak = _get("selected_peak_nm")
    baseline_peak = _get("baseline_peak_nm")

    config = CONFIG.get("responsive_trend", {}) if isinstance(CONFIG, dict) else {}
    min_fraction = float(config.get("min_fraction", 0.1))
    min_consistency = float(config.get("min_consistency", 0.6))
    max_noise = float(config.get("max_std_nm", 5.0))

    usable_delta = selected_delta if np.isfinite(selected_delta) else median_delta
    usable_peak = selected_peak if np.isfinite(selected_peak) else median_peak
    fallback_peak = (
        baseline_peak + fallback_delta
        if np.isfinite(baseline_peak) and np.isfinite(fallback_delta)
        else usable_peak
    )

    quality_flags = [
        bool(np.isfinite(responsive_fraction) and responsive_fraction >= min_fraction),
        bool(np.isfinite(signed_consistency) and signed_consistency >= min_consistency),
        bool(np.isfinite(std_delta) and std_delta <= max_noise),
    ]

    quality_ok = all(quality_flags)

    return {
        "responsive_fraction": responsive_fraction,
        "signed_consistency": signed_consistency,
        "std_delta_nm": std_delta,
        "usable_delta_nm": usable_delta if np.isfinite(usable_delta) else float("nan"),
        "usable_peak_nm": usable_peak if np.isfinite(usable_peak) else float("nan"),
        "fallback_delta_nm": fallback_delta if np.isfinite(fallback_delta) else float("nan"),
        "fallback_peak_nm": fallback_peak if np.isfinite(fallback_peak) else float("nan"),
        "quality_ok": bool(quality_ok),
        "quality_flags": quality_flags,
    }


def _first_finite(values: Sequence[object]) -> float:
    for val in values:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            return fval
    return float("nan")


# _weighted_linear, _theil_sen → migrated to src.scientific.regression (imported above)


# _ransac migrated to src.scientific.regression (imported above)
# Dead code that followed has been removed.


def load_reference_csv(ref_path: str) -> pd.DataFrame:
    """Load the reference spectrum (wavelength,intensity)."""
    ref = _read_csv_spectrum(ref_path)
    if ref.empty:
        raise ValueError("Reference file is empty or invalid")
    return ref


def scan_experiment_root(root_dir: str) -> dict[float, dict[str, list[str]]]:
    """Scan the experiment root to build mapping: concentration -> trial -> frame csv paths.

    Expected structure:
      root/
        <concA>/
          <trial1>/
            frame_001.csv
            ...
          <trial2>/
        <concB>/
          ...
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    def _extract_conc(name: str) -> float:
        pats = [
            r"(\d+(?:\.\d+)?)\s*ppm",
            r"(\d+(?:\.\d+)?)\s*ppb",
            r"(\d+(?:\.\d+)?)\s*%",
            r"conc_?(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)",
        ]
        for p in pats:
            m = re.search(p, name.lower())
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, TypeError, AttributeError):
                    continue
        return 0.0

    mapping: dict[float, dict[str, list[str]]] = {}
    for conc_name in os.listdir(root_dir):
        conc_path = os.path.join(root_dir, conc_name)
        if not os.path.isdir(conc_path):
            continue
        conc_val = _extract_conc(conc_name)
        conc_trials = mapping.setdefault(conc_val, {})

        # Case 1: CSV frames directly under the concentration directory
        direct_frames = [
            os.path.join(conc_path, f) for f in os.listdir(conc_path) if f.lower().endswith(".csv")
        ]
        if direct_frames:
            trial_key = conc_name
            if trial_key in conc_trials:
                suffix = 1
                while f"{trial_key}_{suffix}" in conc_trials:
                    suffix += 1
                trial_key = f"{trial_key}_{suffix}"
            conc_trials[trial_key] = direct_frames

        # Case 2: Trial subfolders
        for trial_name in os.listdir(conc_path):
            trial_path = os.path.join(conc_path, trial_name)
            if not os.path.isdir(trial_path):
                continue
            frames = [
                os.path.join(trial_path, f)
                for f in os.listdir(trial_path)
                if f.lower().endswith(".csv")
            ]
            if frames:
                trial_key = f"{conc_name}/{trial_name}"
                if trial_key in conc_trials:
                    suffix = 1
                    while f"{trial_key}_{suffix}" in conc_trials:
                        suffix += 1
                    trial_key = f"{trial_key}_{suffix}"
                conc_trials[trial_key] = frames

    if not mapping:
        raise ValueError(f"No trials found under {root_dir}")
    return mapping


# ----------------------
# Batch inference with selected multivariate model
# ----------------------


def _load_canonical_from_saved_dir(out_root: str) -> dict[float, pd.DataFrame]:
    saved_dir = Path(out_root) / "stable_selected"
    canonical: dict[float, pd.DataFrame] = {}
    if not saved_dir.exists():
        return canonical
    for f in sorted(saved_dir.glob("*.csv")):
        name = f.stem  # e.g., "0.5_stable"
        try:
            conc_str = name.split("_")[0]
            conc_val = float(conc_str)
        except Exception:
            continue
        try:
            df = pd.read_csv(f)
            if "wavelength" in df.columns and (
                "intensity" in df.columns or "transmittance" in df.columns
            ):
                canonical[conc_val] = df
        except Exception:
            continue
    return canonical


def _collect_csv_files(root: str) -> list[str]:
    files: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _prepare_sample_vector(
    df: pd.DataFrame, target_wl: np.ndarray, ref_df: Optional[pd.DataFrame]
) -> np.ndarray:
    # Optionally compute transmittance and apply preprocessing similar to training
    calib_settings = CONFIG.get("calibration", {})
    use_trans = bool(calib_settings.get("use_transmittance", True))
    if ref_df is not None and use_trans:
        df = compute_transmittance(df, ref_df)
    df = _preprocess_dataframe(df, stage="transmittance")
    col = _signal_column(df)
    wl = df["wavelength"].to_numpy()
    sig = df[col].to_numpy()
    # Interpolate to target grid
    vec = np.interp(target_wl, wl, sig)
    return vec


def _apply_feature_prep_matrix(X: np.ndarray, wl: np.ndarray, cfg: dict[str, object]) -> np.ndarray:
    prep = str(cfg.get("feature_prep", "raw")).lower()
    Xp = X.copy()
    if prep in ("derivative", "first_derivative") or "derivative" in prep:
        Xp = np.gradient(Xp, wl, axis=1)
    if prep == "snv" or "snv" in prep:
        mu = Xp.mean(axis=1, keepdims=True)
        sd = Xp.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        Xp = (Xp - mu) / sd
    elif prep == "mean_center":
        mu = Xp.mean(axis=0, keepdims=True)
        Xp = Xp - mu
    return Xp


def predict_batch_with_selected_model(
    predict_dir: str, ref_path: Optional[str], out_root: str
) -> Optional[str]:
    """Predict concentrations for CSV spectra in predict_dir using the selected multivariate model.

    Saves metrics/predictions_batch.csv and returns its path, or None on failure.
    """
    try:
        files = _collect_csv_files(predict_dir)
        if not files:
            return None
        # Determine selected method
        selected = None
        sel_json_path = Path(out_root) / "metrics" / "multivariate_selection.json"
        calib_json_path = Path(out_root) / "metrics" / "calibration_metrics.json"
        if sel_json_path.exists():
            try:
                with sel_json_path.open("r", encoding="utf-8") as f:
                    sel_obj = json.load(f)
                bm = sel_obj.get("best_method")
                if bm in {"plsr", "ica", "mcr_als"}:
                    selected = bm
            except Exception:
                pass
        if selected is None and calib_json_path.exists():
            try:
                with calib_json_path.open("r", encoding="utf-8") as f:
                    cm = json.load(f)
                sm = str(cm.get("selected_model", ""))
                if sm.startswith("plsr"):
                    selected = "plsr"
                elif sm.startswith("ica"):
                    selected = "ica"
                elif sm.startswith("mcr"):
                    selected = "mcr_als"
            except Exception:
                pass
        if selected is None:
            return None

        ref_df = load_reference_csv(ref_path) if ref_path else None
        rows = []

        if selected == "ica":
            # Load ICA metrics
            path = Path(out_root) / "metrics" / "deconvolution_ica.json"
            if not path.exists():
                return None
            obj = json.loads(path.read_text())
            wl = np.array(obj.get("wavelengths", []), dtype=float)
            comps = np.array(obj.get("components", []), dtype=float)
            bi = int(obj.get("best_component", 0))
            k = float(obj.get("best_linear_k", float("nan")))
            b = float(obj.get("best_linear_b", float("nan")))
            if not comps.size or not wl.size or bi >= comps.shape[0] or not np.isfinite(k):
                return None
            basis = comps[bi]
            denom = float(np.dot(basis, basis)) if basis.size else 0.0
            if denom <= 0:
                return None
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    s_amp = float(np.dot(vec, basis) / denom)
                    y_pred = b + k * s_amp
                    rows.append(
                        {
                            "file": fp,
                            "predicted": float(y_pred),
                            "method": "ica",
                            "component": bi,
                            "amplitude": s_amp,
                        }
                    )
                except Exception:
                    continue

        elif selected == "mcr_als":
            # Load MCR metrics
            path = Path(out_root) / "metrics" / "deconvolution_mcr_als.json"
            if not path.exists():
                return None
            obj = json.loads(path.read_text())
            wl = np.array(obj.get("wavelengths", []), dtype=float)
            comps = np.array(obj.get("components", []), dtype=float)  # comps x wl
            bi = int(obj.get("best_component", 0))
            k = float(obj.get("best_linear_k", float("nan")))
            b = float(obj.get("best_linear_b", float("nan")))
            if not comps.size or not wl.size or bi >= comps.shape[0] or not np.isfinite(k):
                return None
            S = comps  # basis matrix
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    # NNLS to get contributions
                    c, _ = nnls(S.T, vec)
                    amp = float(c[bi]) if bi < len(c) else float("nan")
                    y_pred = b + k * amp if np.isfinite(amp) else float("nan")
                    rows.append(
                        {
                            "file": fp,
                            "predicted": float(y_pred),
                            "method": "mcr_als",
                            "component": bi,
                            "amplitude": amp,
                        }
                    )
                except Exception:
                    continue

        else:  # PLSR
            # Rebuild PLSR on saved canonical and predict
            canonical = _load_canonical_from_saved_dir(out_root)
            if not canonical:
                return None
            mv_cfg = CONFIG.get("calibration", {}).get("multivariate", {})
            pm = _fit_plsr_calibration(canonical, mv_cfg)
            if not pm:
                return None
            wl = np.array(pm.get("wavelengths", []), dtype=float)
            if wl.size == 0:
                return None
            # Build X_train on wl subset and fit final model
            X_train, y_train, wl_base = _build_feature_matrix_from_canonical(canonical)
            # Restrict to wl subset
            # Interpolate X_train rows to selected wl grid
            Xw = []
            for _conc, df in sorted(canonical.items(), key=lambda kv: kv[0]):
                v = _prepare_sample_vector(
                    df, wl, ref_df=None
                )  # already stable canonical; do not recompute trans
                Xw.append(v)
            Xw = np.vstack(Xw)
            Xw = _apply_feature_prep_matrix(Xw, wl, mv_cfg)
            n_comp = int(pm.get("n_components", 1))
            pls = PLSRegression(n_components=n_comp, scale=bool(mv_cfg.get("scale", True)))
            pls.fit(Xw, y_train)
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    vec2 = _apply_feature_prep_matrix(vec.reshape(1, -1), wl, mv_cfg)
                    y_pred = float(pls.predict(vec2).ravel()[0])
                    rows.append({"file": fp, "predicted": y_pred, "method": "plsr"})
                except Exception:
                    continue

        if not rows:
            return None
        out_csv = Path(out_root) / "metrics" / "predictions_batch.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return str(out_csv)
    except Exception:
        return None


# ----------------------
# Transmittance
# ----------------------


# compute_transmittance migrated to src.signal.transforms (imported above)



# _append_absorbance_column migrated to src.signal.transforms (imported above)



# _ensure_odd_window migrated to src.signal.transforms (imported above)



def _apply_signal_strategy(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    strategies = (
        CONFIG.get("analysis", {}).get("signal_strategies", {}) if isinstance(CONFIG, dict) else {}
    )
    strat = strategies.get(signal, {}) if isinstance(strategies, dict) else {}
    if not strat:
        return df

    out = df.copy(deep=True)
    wl = out["wavelength"].to_numpy(dtype=float, copy=True)
    y = out[signal].to_numpy(dtype=float, copy=True)

    smooth_cfg = strat.get("smooth", {}) if isinstance(strat, dict) else {}
    if smooth_cfg.get("enabled", False):
        method = smooth_cfg.get("method", "savgol")
        window = _ensure_odd_window(smooth_cfg.get("window", 11))
        poly = int(max(1, smooth_cfg.get("poly_order", 3)))
        y = smooth_spectrum(y, window=window, poly_order=poly, method=method)

    baseline_cfg = strat.get("baseline", {}) if isinstance(strat, dict) else {}
    if baseline_cfg.get("enabled", False):
        method = baseline_cfg.get("method", "als")
        order = int(baseline_cfg.get("order", 2))
        y = baseline_correction(wl, y, method=method, poly_order=order)

    normalize_cfg = strat.get("normalize", {}) if isinstance(strat, dict) else {}
    if normalize_cfg.get("enabled", False):
        method = normalize_cfg.get("method", "minmax")
        y = normalize_spectrum(y, method=method)

    clip_cfg = strat.get("clip", {}) if isinstance(strat, dict) else {}
    if clip_cfg.get("enabled", False):
        vmin = clip_cfg.get("min", None)
        vmax = clip_cfg.get("max", None)
        y = np.clip(y, vmin if vmin is not None else y, vmax if vmax is not None else y)

    center_cfg = strat.get("center", {}) if isinstance(strat, dict) else {}
    if center_cfg.get("enabled", False):
        mode = str(center_cfg.get("mode", "mean")).lower()
        if mode == "median":
            y = y - float(np.nanmedian(y))
        else:
            y = y - float(np.nanmean(y))

    out[signal] = y
    return out


def _build_signal_views(
    processed: dict[float, dict[str, pd.DataFrame]],
    raw: dict[float, dict[str, pd.DataFrame]],
) -> dict[str, dict[float, dict[str, pd.DataFrame]]]:
    signal_views: dict[str, dict[float, dict[str, pd.DataFrame]]] = {}
    for signal in ("intensity", "transmittance", "absorbance"):
        view_map: dict[float, dict[str, pd.DataFrame]] = {}
        source = raw if (signal == "intensity" and raw) else processed
        for conc, trials in source.items():
            trial_views: dict[str, pd.DataFrame] = {}
            for name, df in trials.items():
                if signal not in df.columns:
                    continue
                cols = ["wavelength", signal]
                extracted = df[cols].copy(deep=True)
                prepared = _apply_signal_strategy(extracted, signal)
                trial_views[name] = prepared
            if trial_views:
                view_map[conc] = trial_views
        if view_map:
            signal_views[signal] = view_map
    return signal_views


def _resolve_primary_signal(
    signal_views: dict[str, dict[float, dict[str, pd.DataFrame]]],
) -> str:
    analysis_cfg = CONFIG.get("analysis", {}) if isinstance(CONFIG, dict) else {}
    preferred = str(analysis_cfg.get("primary_signal", "") or "").lower()
    candidates = [preferred] if preferred else []
    if analysis_cfg.get("enable_absorbance", False):
        candidates.append("absorbance")
    candidates.extend(["transmittance", "intensity"])
    for candidate in candidates:
        if candidate and candidate in signal_views:
            return candidate
    return next(iter(signal_views.keys())) if signal_views else "intensity"


def compute_transmittance_on_frames(
    frames: list[pd.DataFrame], ref_df: pd.DataFrame
) -> list[pd.DataFrame]:
    return [compute_transmittance(df, ref_df) for df in frames]


# ----------------------
# Stability on multi-frame spectral trials
# ----------------------


def _align_on_grid(
    frames: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (wl, Y, signal_col) using a signal present across all frames."""
    if not frames:
        raise ValueError("No frames provided for alignment")

    base = frames[0]
    wl = base["wavelength"].values
    signal_col = _select_common_signal(frames)
    if signal_col is None:
        signal_col = _signal_column(base)

    Y = []
    for df in frames:
        col = signal_col if signal_col in df.columns else _signal_column(df)
        vec = df[col].values
        if not np.array_equal(df["wavelength"].values, wl):
            vec = np.interp(wl, df["wavelength"].values, vec)
        Y.append(vec)

    return (
        wl,
        np.vstack(Y),
        signal_col if signal_col in base.columns else _signal_column(base),
    )


def find_stable_block(
    frames: list[pd.DataFrame],
    diff_threshold: float = 0.01,
    weight_mode: str = "uniform",
    top_k: Optional[int] = None,
    min_block: Optional[int] = None,
    **_unused: object,
) -> tuple[int, int, np.ndarray]:
    return _find_stable_block_src(
        frames, diff_threshold=diff_threshold, weight_mode=weight_mode,
        min_block=min_block
    )


def average_stable_block(
    frames: list[pd.DataFrame],
    start_idx: int,
    end_idx: int,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    return _average_stable_block_src(frames, start_idx, end_idx, weights=weights)


def average_top_frames(frames: list[pd.DataFrame], top_k: int = 5) -> pd.DataFrame:
    return _average_top_frames_src(
        frames, top_n=top_n, signal_col=signal_col, ascending=ascending
    )


def compute_roi_linearity(
    df: pd.DataFrame,
    concentrations: list[float],
    response_metric: str = "transmittance",
) -> dict[str, float]:
    """Compute linearity metrics for a single averaged spectrum across concentrations."""
    if response_metric not in df.columns:
        raise ValueError(f"Column '{response_metric}' not found in dataframe")
    wl = df["wavelength"].values
    signal = df[response_metric].values
    roi_cfg = CONFIG.get("roi", {})
    min_wl = roi_cfg.get("min_wavelength", wl.min())
    max_wl = roi_cfg.get("max_wavelength", wl.max())
    mask = (wl >= min_wl) & (wl <= max_wl)
    wl_roi = wl[mask]
    signal_roi = signal[mask]
    slope, intercept, r_value, _, _ = linregress(wl_roi, signal_roi)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value**2),
        "start_wavelength": float(wl_roi[0]) if wl_roi.size else float("nan"),
        "end_wavelength": float(wl_roi[-1]) if wl_roi.size else float("nan"),
    }


# ----------------------
# Canonical selection per concentration
# ----------------------


def select_canonical_per_concentration(
    stable_results: dict[float, dict[str, pd.DataFrame]],
) -> dict[float, pd.DataFrame]:
    return _select_canonical_src(stable_results)


def _baseline_correct_canonical(
    canonical: dict[float, pd.DataFrame],
) -> dict[float, pd.DataFrame]:
    # Return canonical spectra without subtracting the lowest concentration.
    # Previous behavior used the minimum concentration (e.g., 1 ppm) as a
    # surrogate “zero” baseline, which inverted the calibration trend when the
    # true baseline should be an external air reference. Until we wire in the
    # dedicated zero-gas reference, keep spectra in absolute wavelength space.
    return canonical


# ----------------------
# ROI and calibration (wavelength shift)
# ----------------------


# _smooth migrated to src.signal.transforms (imported above)



def _refine_centroid_with_derivative(
    df: pd.DataFrame,
    centroid_cfg: Optional[dict[str, object]],
    expected_center: Optional[float],
    span_nm: float = 2.0,
    smooth_window: int = 7,
) -> float:
    if expected_center is None or not np.isfinite(expected_center):
        return float("nan")

    centroid_cfg = centroid_cfg or {}
    try:
        span_nm = float(centroid_cfg.get("derivative_span_nm", span_nm) or span_nm)
    except Exception:
        span_nm = span_nm
    try:
        smooth_window = int(
            centroid_cfg.get("derivative_smooth_window", smooth_window) or smooth_window
        )
    except Exception:
        smooth_window = smooth_window

    if span_nm <= 0:
        return float("nan")

    x, y = _prepare_calibration_signal(df, centroid_cfg)
    if x.size < 3:
        return float("nan")

    mask = (x >= expected_center - span_nm) & (x <= expected_center + span_nm)
    if np.count_nonzero(mask) < 3:
        idx_closest = int(np.argmin(np.abs(x - expected_center)))
        return float(x[idx_closest])

    xx = x[mask]
    yy = y[mask]

    # Ensure odd window length within range
    smooth_window = max(5, smooth_window)
    smooth_window = _ensure_odd_window(smooth_window)
    if smooth_window >= xx.size:
        smooth_window = xx.size - 1 if xx.size % 2 == 0 else xx.size
    smooth_window = max(3, smooth_window)
    if smooth_window >= xx.size:
        # if still too large, fall back to gradient
        smooth_window = 0

    delta = float(np.mean(np.diff(xx))) if xx.size > 1 else 1.0
    try:
        if smooth_window >= 3:
            deriv = savgol_filter(
                yy,
                window_length=smooth_window,
                polyorder=min(3, smooth_window - 1),
                deriv=1,
                delta=delta,
                mode="interp",
            )
        else:
            raise RuntimeError("window_too_small")
    except Exception:
        deriv = np.gradient(yy, xx)

    idx_peak = int(np.argmax(np.abs(deriv)))
    idx_peak = int(np.clip(idx_peak, 0, xx.size - 1))
    return float(xx[idx_peak])


# _estimate_shift_crosscorr migrated to src.signal.peak (imported above)



# _gaussian_peak_center migrated to src.signal.peak (imported above)



# _compute_band_ratio_matrix — migrated to src.signal.roi (Phase 3)


# _find_monotonic_wavelengths — migrated to src.signal.roi (Phase 3)



# _transform_concentrations — migrated to src.calibration.transforms (Phase 3; unused in pipeline.py)


# _select_multi_roi_candidates — migrated to src.calibration.multi_roi (Phase 3)


def _compute_multi_roi_fusion_calibration(
    discovered_roi: Optional[dict[str, object]],
    calib: dict[str, object],
    out_root: str,
    dataset_label: Optional[str],
    max_features: int = 4,
) -> Optional[dict[str, object]]:
    if not discovered_roi or not isinstance(calib, dict):
        return None

    concentrations = calib.get("concentrations") or []
    if not isinstance(concentrations, list) or len(concentrations) < 3:
        return None

    # Pure computation delegated to src.calibration.multi_roi (Phase 3)
    metrics = _fit_multi_roi_fusion(discovered_roi, concentrations, max_features=max_features)
    if metrics is None:
        return None

    metrics["dataset"] = dataset_label

    metrics_dir = os.path.join(out_root, "metrics")
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    fusion_metrics_path = os.path.join(metrics_dir, "multi_roi_fusion_metrics.json")
    with open(fusion_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Reconstruct arrays for plotting from metrics dict
    y = np.array(metrics["actual_concentrations_ppm"])
    y_pred = np.array(metrics["predicted_concentrations_ppm"])
    cv_preds_list = metrics.get("cv_predictions_ppm")
    cv_preds = np.array(cv_preds_list) if cv_preds_list is not None else None
    r2 = metrics["r2"]
    rmse = metrics["rmse_ppm"]
    r2_cv = metrics["r2_cv"]
    lod_ppm = metrics["lod_ppm"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y, y_pred, color="#1f77b4", label="Train fit")
    if isinstance(cv_preds, np.ndarray) and np.all(np.isfinite(cv_preds)):
        ax.scatter(y, cv_preds, color="#ff7f0e", marker="s", label="LOOCV")
    min_c = min(np.min(y), np.min(y_pred))
    max_c = max(np.max(y), np.max(y_pred))
    ax.plot([min_c, max_c], [min_c, max_c], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Actual concentration (ppm)")
    ax.set_ylabel("Predicted concentration (ppm)")
    ax.set_title("Multi-ROI Fusion Calibration")
    ax.grid(True, alpha=0.3)
    text_lines = [f"R² = {r2:.3f}", f"RMSE = {rmse:.3f} ppm"]
    if np.isfinite(r2_cv):
        text_lines.append(f"R²_LOOCV = {r2_cv:.3f}")
    if np.isfinite(lod_ppm):
        text_lines.append(f"LOD ≈ {lod_ppm:.2f} ppm")
    ax.text(
        0.05, 0.05, "\n".join(text_lines), transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend()
    fig.tight_layout()
    fusion_plot_path = os.path.join(plots_dir, "calibration_multi_roi_fusion.png")
    fig.savefig(fusion_plot_path, dpi=200)
    plt.close(fig)

    metrics["metrics_path"] = fusion_metrics_path
    metrics["plot_path"] = fusion_plot_path
    return metrics


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_canonical_spectra(canonical: dict[float, pd.DataFrame], out_root: str) -> list[str]:
    return _save_canonical_spectra_io(canonical, out_root)


def save_aggregated_spectra(
    aggregated: dict[float, dict[str, pd.DataFrame]], out_root: str
) -> dict[float, dict[str, str]]:
    return _save_aggregated_spectra_io(aggregated, out_root)


# _select_common_signal — migrated to src.reporting.metrics (Phase 4)


# _common_signal_columns — migrated to src.reporting.metrics (Phase 4)


# compute_noise_metrics_map — migrated to src.reporting.metrics (Phase 4)


def save_noise_metrics(metrics: dict[float, dict[str, object]], out_root: str) -> str:
    return _save_noise_metrics_io(metrics, out_root)


def compute_environment_summary(
    stable_by_conc,
) -> dict:
    """Compute environment compensation summary (CONFIG-coupled wrapper)."""
    env_cfg = CONFIG.get("environment", {}) if isinstance(CONFIG, dict) else {}
    if not env_cfg:
        return {}
    ref = env_cfg.get("reference", {}) or {}
    coeffs = env_cfg.get("coefficients", {}) or {}
    override = env_cfg.get("override", {}) or {}
    return _compute_environment_summary_pure(
        stable_by_conc,
        T_ref=float(ref.get("temperature", 25.0)),
        H_ref=float(ref.get("humidity", 50.0)),
        cT=coeffs.get("temperature", None),
        cH=coeffs.get("humidity", None),
        env_enabled=bool(env_cfg.get("enabled", False)),
        apply_to_frames=bool(env_cfg.get("apply_to_frames", False)),
        apply_to_transmittance=bool(env_cfg.get("apply_to_transmittance", True)),
        override_temp=override.get("temperature", None),
        override_humid=override.get("humidity", None),
    )

def compute_environment_coefficients(
    stable_by_conc, calib
) -> dict:
    """Estimate environment coefficients (CONFIG-coupled wrapper)."""
    env_cfg = CONFIG.get("environment", {}) if isinstance(CONFIG, dict) else {}
    ref = env_cfg.get("reference", {}) or {}
    return _compute_environment_coefficients_pure(
        stable_by_conc,
        calib,
        T_ref=float(ref.get("temperature", 25.0)),
        H_ref=float(ref.get("humidity", 50.0)),
    )

def summarize_quality_control(
    stable_by_conc,
    noise_metrics,
) -> dict:
    """Summarise QC metrics (CONFIG-coupled wrapper)."""
    qcfg = CONFIG.get("quality", {}) if isinstance(CONFIG, dict) else {}
    return _summarize_quality_control_pure(
        stable_by_conc,
        noise_metrics,
        min_snr=float(qcfg.get("min_snr", 10.0)),
        max_rsd=float(qcfg.get("max_rsd", 5.0)),
    )

def save_quality_summary(qc: dict[str, object], out_root: str) -> str:
    return _save_quality_summary_io(qc, out_root)


def save_aggregated_summary(
    aggregated: dict[float, dict[str, pd.DataFrame]],
    noise_metrics: dict[float, dict[str, object]],
    out_root: str,
) -> str:
    return _save_aggregated_summary_io(aggregated, noise_metrics, out_root)


def compute_concentration_response(
    stable_by_conc,
    override_min_wavelength=None,
    override_max_wavelength=None,
    top_k_candidates: int = 0,
    debug_out_root=None,
):
    """Scan every wavelength for the best ROI (CONFIG-coupled wrapper)."""
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    validation_cfg = roi_cfg.get("validation", {}) or {}
    alt_cfg = roi_cfg.get("alternative_models", {}) or {}
    adp_cfg = roi_cfg.get("adaptive_band", {}) or {}
    bhw = roi_cfg.get("band_half_width", None)
    repeatability = CONFIG.get("_last_repeatability", {}) or {}
    g_std = float((repeatability.get("global", {}) or {}).get("std_transmittance", 0.0) or 0.0)
    cfg = _RoiScanConfig(
        selection_metric=str(roi_cfg.get("selection_metric", "r2")).lower(),
        min_r2=float(roi_cfg.get("min_r2", 0.0)),
        r2_weight=float(roi_cfg.get("r2_weight", 1.0)),
        expected_trend=str(roi_cfg.get("expected_trend", "any")).lower(),
        trend_modes=roi_cfg.get("trend_modes", None),
        min_corr=float(roi_cfg.get("min_corr", 0.0)),
        min_wavelength=override_min_wavelength if override_min_wavelength is not None else roi_cfg.get("min_wavelength", None),
        max_wavelength=override_max_wavelength if override_max_wavelength is not None else roi_cfg.get("max_wavelength", None),
        band_half_width=int(bhw) if bhw is not None else None,
        band_window=int(roi_cfg.get("band_window", 0)),
        derivative_weight=float(roi_cfg.get("derivative_weight", 0.0)),
        ratio_weight=float(roi_cfg.get("ratio_weight", 0.0)),
        ratio_half_width=int(max(1, roi_cfg.get("ratio_half_width", 5))),
        slope_noise_weight=float(roi_cfg.get("slope_noise_weight", 0.0)),
        min_slope_to_noise=float(roi_cfg.get("min_slope_to_noise", 0.0)),
        global_std=g_std,
        min_abs_slope=float(roi_cfg.get("min_abs_slope", 0.0)),
        alt_models_enabled=bool(alt_cfg.get("enabled", False)),
        poly_degree=int(max(1, alt_cfg.get("polynomial_degree", 2))),
        adaptive_band_enabled=bool(adp_cfg.get("enabled", False)),
        slope_fraction=float(adp_cfg.get("slope_fraction", 0.6)),
        adaptive_max_half_width=int(adp_cfg.get("max_half_width", bhw if bhw is not None else 20)),
        expected_center=validation_cfg.get("expected_center"),
        center_tolerance=float(validation_cfg.get("tolerance", 0.0)),
        validation_notes=str(validation_cfg.get("notes", "")),
    )
    return _compute_concentration_response_pure(
        stable_by_conc,
        cfg=cfg,
        top_k_candidates=top_k_candidates,
        debug_out_root=debug_out_root,
    )

def save_roi_performance_metrics(performance: dict[str, object], out_root: str) -> Optional[str]:
    return _save_roi_performance_metrics_io(performance, out_root)


def save_roi_discovery_plot(discovery: dict[str, object], out_root: str) -> Optional[str]:
    return _save_roi_discovery_plot_src(discovery, out_root)


def summarize_roi_performance(performance: dict[str, object]) -> Optional[str]:
    if not performance:
        return None
    sensitivity = performance.get("regression_slope")
    r2 = performance.get("regression_r2")
    lod = performance.get("lod_ppm")
    loq = performance.get("loq_ppm")
    return (
        f"slope={sensitivity:.6f} dT/ppm, R²={r2:.3f}, LOD={lod:.3f} ppm, LOQ={loq:.3f} ppm"
        if all(v is not None for v in (sensitivity, r2, lod, loq))
        else None
    )


# summarize_dynamics_metrics — migrated to src.reporting.metrics (Phase 4; unused in pipeline.py)


def save_dynamics_summary(summary: dict[str, object], out_root: str) -> str:
    return _save_dynamics_summary_io(summary, out_root)


def save_dynamics_error(message: str, out_root: str) -> str:
    return _save_dynamics_error_io(message, out_root)


def save_concentration_response_metrics(
    response: dict[str, object],
    repeatability: dict[str, object],
    out_root: str,
    name: str = "concentration_response",
) -> str:
    return _save_concentration_response_metrics_io(
        response, repeatability, out_root, name=name
    )


def save_concentration_response_plot(
    response: dict[str, object],
    avg_by_conc: dict[float, np.ndarray],
    out_root: str,
    name: str = "concentration_response",
    clamp_to_roi: bool = True,
) -> Optional[str]:
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    x_min = roi_cfg.get("min_wavelength")
    x_max = roi_cfg.get("max_wavelength")
    return _save_concentration_response_plot_pure(
        response, avg_by_conc, out_root, name=name, clamp_to_roi=clamp_to_roi,
        x_min=float(x_min) if x_min is not None else None,
        x_max=float(x_max) if x_max is not None else None,
    )


def save_wavelength_shift_visualization(
    canonical: dict[float, pd.DataFrame],
    calib_result: dict[str, object],
    out_root: str,
    dataset_label: Optional[str] = None,
) -> Optional[str]:
    return _save_wavelength_shift_visualization_src(
        canonical, calib_result, out_root, dataset_label=dataset_label
    )


def save_research_grade_calibration_plot(
    canonical: dict[float, pd.DataFrame],
    calib_result: dict[str, object],
    out_root: str,
    dataset_label: Optional[str] = None,
) -> Optional[str]:
    return _save_research_grade_calibration_plot_src(
        canonical, calib_result, out_root, dataset_label=dataset_label
    )


def save_spectral_response_diagnostic(
    canonical: dict[float, pd.DataFrame],
    out_root: str,
    dataset_label: Optional[str] = None,
    wl_min: float = 400.0,
    wl_max: float = 800.0,
) -> Optional[str]:
    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    shift_cfg = roi_cfg.get("shift", {}) if isinstance(roi_cfg.get("shift", {}), dict) else {}
    step_nm = float(shift_cfg.get("step_nm", 2.0) or 2.0)
    if not np.isfinite(step_nm) or step_nm <= 0:
        step_nm = 2.0
    window_nm = shift_cfg.get("window_nm", 10.0)
    return _save_spectral_response_diagnostic_pure(
        canonical, out_root, dataset_label=dataset_label,
        wl_min=wl_min, wl_max=wl_max,
        step_nm=step_nm, window_nm=window_nm,
    )


def generate_method_comparison_report(
    canonical: dict[float, pd.DataFrame],
    wavelength_shift_result: dict[str, object],
    absorbance_amp_result: dict[str, object],
    out_root: str,
    dataset_label: Optional[str] = None,
) -> Optional[str]:
    """Generate a comprehensive comparison report between Δλ and ΔA methods.

    Includes:
    - Side-by-side calibration curves
    - Leave-one-out cross-validation for both methods
    - Statistical comparison
    - Recommendation based on metrics
    """
    if not wavelength_shift_result and not absorbance_amp_result:
        return None

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "method_comparison.png")

    # Extract data
    concs_wl = (
        np.array(wavelength_shift_result.get("concentrations", []))
        if wavelength_shift_result
        else np.array([])
    )
    wl_values = (
        np.array(wavelength_shift_result.get("peak_wavelengths", []))
        if wavelength_shift_result
        else np.array([])
    )

    concs_abs = (
        np.array(absorbance_amp_result.get("concentrations", []))
        if absorbance_amp_result
        else np.array([])
    )
    abs_values = (
        np.array(absorbance_amp_result.get("absorbance_values", []))
        if absorbance_amp_result
        else np.array([])
    )

    # Perform Leave-One-Out Cross-Validation for both methods
    def loocv_r2(x, y):
        """Calculate LOOCV R² score."""
        if len(x) < 3:
            return np.nan, np.nan

        n = len(x)
        predictions = np.zeros(n)

        for i in range(n):
            # Leave one out
            x_train = np.delete(x, i)
            y_train = np.delete(y, i)

            # Fit on training data
            try:
                reg = linregress(x_train, y_train)
                predictions[i] = reg.slope * x[i] + reg.intercept
            except Exception:
                predictions[i] = np.nan

        # Calculate R² on predictions
        if np.any(np.isnan(predictions)):
            return np.nan, np.nan

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_cv = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse_cv = np.sqrt(np.mean((y - predictions) ** 2))

        return r2_cv, rmse_cv

    # Calculate LOOCV for wavelength shift
    wl_r2_cv, wl_rmse_cv = loocv_r2(concs_wl, wl_values) if len(concs_wl) > 0 else (np.nan, np.nan)

    # Calculate LOOCV for absorbance amplitude
    abs_r2_cv, abs_rmse_cv = (
        loocv_r2(concs_abs, abs_values) if len(concs_abs) > 0 else (np.nan, np.nan)
    )

    # Get metrics from results
    wl_r2 = wavelength_shift_result.get("r2", 0) if wavelength_shift_result else 0
    wl_slope = wavelength_shift_result.get("slope", 0) if wavelength_shift_result else 0
    wl_lod = wavelength_shift_result.get("lod", np.nan) if wavelength_shift_result else np.nan
    wl_center = wavelength_shift_result.get("roi_center", 0) if wavelength_shift_result else 0

    abs_r2 = absorbance_amp_result.get("r2", 0) if absorbance_amp_result else 0
    abs_slope = absorbance_amp_result.get("slope_au_per_ppm", 0) if absorbance_amp_result else 0
    abs_lod = absorbance_amp_result.get("lod_ppm", np.nan) if absorbance_amp_result else np.nan
    abs_wl = absorbance_amp_result.get("best_wavelength_nm", 0) if absorbance_amp_result else 0

    # Create comparison figure
    fig = plt.figure(figsize=(18, 14))

    # Plot 1: Wavelength Shift Calibration (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    if len(concs_wl) > 0 and len(wl_values) > 0:
        # Convert to delta from baseline
        wl_delta = (wl_values - wl_values[0]) * 1000  # pm
        ax1.scatter(
            concs_wl,
            wl_delta,
            s=100,
            c="#E74C3C",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )

        # Fit line
        if len(concs_wl) >= 2:
            z = np.polyfit(concs_wl, wl_delta, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(0, max(concs_wl) * 1.1, 100)
            ax1.plot(x_fit, p(x_fit), "--", color="#E74C3C", linewidth=2, alpha=0.7)

        ax1.set_xlabel("Concentration (ppm)", fontsize=11)
        ax1.set_ylabel("Δλ from baseline (pm)", fontsize=11)
        ax1.set_title(
            f"Wavelength Shift Method\nR² = {wl_r2:.4f}, R²_CV = {wl_r2_cv:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    else:
        ax1.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax1.set_title("Wavelength Shift Method\n(Not Available)")

    # Plot 2: Absorbance Amplitude Calibration (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    if len(concs_abs) > 0 and len(abs_values) > 0:
        # Convert to delta from baseline
        abs_delta = (abs_values - abs_values[0]) * 1000  # mAU
        ax2.scatter(
            concs_abs,
            abs_delta,
            s=100,
            c="#3498DB",
            edgecolor="black",
            linewidth=1.5,
            zorder=5,
        )

        # Fit line
        if len(concs_abs) >= 2:
            z = np.polyfit(concs_abs, abs_delta, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(0, max(concs_abs) * 1.1, 100)
            ax2.plot(x_fit, p(x_fit), "--", color="#3498DB", linewidth=2, alpha=0.7)

        ax2.set_xlabel("Concentration (ppm)", fontsize=11)
        ax2.set_ylabel("ΔA from baseline (mAU)", fontsize=11)
        ax2.set_title(
            f"Absorbance Amplitude Method\nR² = {abs_r2:.4f}, R²_CV = {abs_r2_cv:.4f}",
            fontsize=12,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    else:
        ax2.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
        ax2.set_title("Absorbance Amplitude Method\n(Not Available)")

    # Plot 3: R² Comparison Bar Chart (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    methods = ["Δλ Method", "ΔA Method"]
    r2_values = [wl_r2, abs_r2]
    r2_cv_values = [
        wl_r2_cv if not np.isnan(wl_r2_cv) else 0,
        abs_r2_cv if not np.isnan(abs_r2_cv) else 0,
    ]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        r2_values,
        width,
        label="R² (fit)",
        color=["#E74C3C", "#3498DB"],
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x + width / 2,
        r2_cv_values,
        width,
        label="R²_CV (LOOCV)",
        color=["#E74C3C", "#3498DB"],
        alpha=0.4,
        hatch="//",
    )

    ax3.set_ylabel("R² Value", fontsize=11)
    ax3.set_title("Model Quality Comparison", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.legend()
    ax3.set_ylim(0, 1.1)
    ax3.axhline(0.9, color="green", linestyle="--", alpha=0.5, label="R²=0.9 threshold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax3.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Plot 4: Sensitivity Comparison (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    sens_labels = ["Δλ Sensitivity\n(pm/ppm)", "ΔA Sensitivity\n(mAU/ppm)"]
    sens_values = [abs(wl_slope) * 1000, abs(abs_slope) * 1000]
    colors = ["#E74C3C", "#3498DB"]

    bars = ax4.bar(sens_labels, sens_values, color=colors, edgecolor="black", linewidth=1.5)
    ax4.set_ylabel("Sensitivity (absolute)", fontsize=11)
    ax4.set_title("Sensitivity Comparison", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, sens_values):
        ax4.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 5: LOD Comparison (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    lod_labels = ["Δλ Method", "ΔA Method"]
    lod_values = [
        wl_lod if not np.isnan(wl_lod) else 0,
        abs_lod if not np.isnan(abs_lod) else 0,
    ]

    bars = ax5.bar(lod_labels, lod_values, color=colors, edgecolor="black", linewidth=1.5)
    ax5.set_ylabel("LOD (ppm)", fontsize=11)
    ax5.set_title(
        "Limit of Detection Comparison\n(Lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, lod_values):
        if val > 0:
            ax5.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Plot 6: Summary and Recommendation (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    # Determine best method
    wl_score = 0
    abs_score = 0

    if wl_r2 > abs_r2:
        wl_score += 1
    else:
        abs_score += 1

    if not np.isnan(wl_r2_cv) and not np.isnan(abs_r2_cv):
        if wl_r2_cv > abs_r2_cv:
            wl_score += 1
        else:
            abs_score += 1

    if not np.isnan(wl_lod) and not np.isnan(abs_lod):
        if wl_lod < abs_lod:
            wl_score += 1
        else:
            abs_score += 1

    recommended = "Wavelength Shift (Δλ)" if wl_score > abs_score else "Absorbance Amplitude (ΔA)"

    summary_text = f"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              METHOD COMPARISON SUMMARY                        ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║  WAVELENGTH SHIFT (Δλ)           ABSORBANCE AMPLITUDE (ΔA)    ║
    ║  ─────────────────────           ─────────────────────────    ║
    ║  Center: {wl_center:>8.1f} nm            Wavelength: {abs_wl:>8.1f} nm     ║
    ║  R²:     {wl_r2:>8.4f}                  R²:     {abs_r2:>8.4f}             ║
    ║  R²_CV:  {wl_r2_cv:>8.4f}                  R²_CV:  {abs_r2_cv:>8.4f}             ║
    ║  Sens:   {abs(wl_slope) * 1000:>6.2f} pm/ppm           Sens:   {abs(abs_slope) * 1000:>6.2f} mAU/ppm       ║
    ║  LOD:    {wl_lod:>8.2f} ppm             LOD:    {abs_lod:>8.2f} ppm           ║
    ║                                                               ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  RECOMMENDATION:                                              ║
    ║  {recommended:^59}  ║
    ║                                                               ║
    ║  Score: Δλ={wl_score}/3, ΔA={abs_score}/3 (R², R²_CV, LOD)                      ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    ax6.text(
        0.5,
        0.5,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    # Main title
    fig.suptitle(
        f"{dataset_label or 'Gas'} - Calibration Method Comparison\n"
        f"Wavelength Shift (Δλ) vs Absorbance Amplitude (ΔA)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    tmp_path = out_path + ".tmp"
    try:
        fig.savefig(tmp_path, dpi=300, format="png", facecolor="white")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)

    # Save comparison metrics JSON
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    comparison_metrics = {
        "dataset": dataset_label,
        "wavelength_shift": {
            "center_nm": wl_center,
            "r2": wl_r2,
            "r2_cv": float(wl_r2_cv) if not np.isnan(wl_r2_cv) else None,
            "sensitivity_pm_per_ppm": abs(wl_slope) * 1000,
            "lod_ppm": float(wl_lod) if not np.isnan(wl_lod) else None,
        },
        "absorbance_amplitude": {
            "wavelength_nm": abs_wl,
            "r2": abs_r2,
            "r2_cv": float(abs_r2_cv) if not np.isnan(abs_r2_cv) else None,
            "sensitivity_mau_per_ppm": abs(abs_slope) * 1000,
            "lod_ppm": float(abs_lod) if not np.isnan(abs_lod) else None,
        },
        "recommendation": recommended,
        "scores": {"wavelength_shift": wl_score, "absorbance_amplitude": abs_score},
    }

    metrics_path = os.path.join(metrics_dir, "method_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(comparison_metrics, f, indent=2)

    return out_path


def save_roi_repeatability_plot(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    response: dict[str, object],
    out_root: str,
) -> Optional[str]:
    return _save_roi_repeatability_plot_src(stable_by_conc, response, out_root)


def save_aggregated_plots(
    aggregated: dict[float, dict[str, pd.DataFrame]], out_root: str
) -> dict[float, dict[str, str]]:
    return _save_aggregated_plots_src(aggregated, out_root)


def save_canonical_overlay(canonical: dict[float, pd.DataFrame], out_root: str) -> Optional[str]:
    return _save_canonical_overlay_src(canonical, out_root)


def save_environment_compensation_summary(info: dict[str, object], out_root: str) -> str:
    return _save_env_compensation_summary_io(info, out_root)


def write_run_summary(
    calib: dict[str, object],
    aggregated_paths: dict[float, dict[str, str]],
    noise_metrics_path: str,
    summary_csv_path: str,
    canonical_plot_path: Optional[str],
    response_metrics_path: str,
    response_plot_path: Optional[str],
    repeatability_plot_path: Optional[str],
    performance_metrics_path: Optional[str],
    dynamics_summary_path: Optional[str],
    dynamics_plot_path: Optional[str],
    metadata_path: Optional[str],
    archive_path: Optional[str],
    qc_summary_path: Optional[str],
    report_artifacts: dict[str, object],
    trend_plots: dict[str, str],
    performance: Optional[dict[str, object]],
    dynamics_summary: Optional[dict[str, object]],
    out_root: str,
) -> str:
    reports_dir = os.path.join(out_root, "reports")
    _ensure_dir(reports_dir)
    out_path = os.path.join(reports_dir, "summary.md")

    lines = [
        "# Gas Analysis Run Summary",
        "",
        "## Calibration Results",
        "",
    ]
    try:
        sel_model = str(calib.get("selected_model", ""))
        if sel_model.endswith("_cv"):
            lines.append(f"- **Selected model**: {sel_model}")
    except Exception:
        pass
    try:
        slope_v = float(calib.get("slope", float("nan")))
        if np.isfinite(slope_v):
            lines.append(f"- **Slope**: {slope_v:.4f} nm/ppm")
    except Exception:
        pass
    try:
        intercept_v = float(calib.get("intercept", float("nan")))
        if np.isfinite(intercept_v):
            lines.append(f"- **Intercept**: {intercept_v:.4f} nm")
    except Exception:
        pass
    try:
        r2_v = float(calib.get("r2", float("nan")))
        if np.isfinite(r2_v):
            lines.append(f"- **R²**: {r2_v:.4f}")
    except Exception:
        pass
    try:
        rmse_v = float(calib.get("rmse", float("nan")))
        if np.isfinite(rmse_v):
            try:
                units = "ppm" if str(calib.get("selected_model", "")).endswith("_cv") else "nm"
            except Exception:
                units = "nm"
            lines.append(f"- **RMSE**: {rmse_v:.4f} {units}")
    except Exception:
        pass
    try:
        lod_v = float(calib.get("lod", float("nan")))
        if np.isfinite(lod_v):
            lines.append(f"- **LOD**: {lod_v:.4f} ppm")
    except Exception:
        pass
    try:
        loq_v = float(calib.get("loq", float("nan")))
        if np.isfinite(loq_v):
            lines.append(f"- **LOQ**: {loq_v:.4f} ppm")
    except Exception:
        pass
    try:
        roi_v = float(calib.get("roi_center", float("nan")))
        if np.isfinite(roi_v):
            lines.append(f"- **ROI Center**: {roi_v:.4f} nm")
    except Exception:
        pass
    lines.append("")

    # Add CV metrics if available
    try:
        unc = calib.get("uncertainty", {}) if isinstance(calib, dict) else {}
        r2_cv = unc.get("r2_cv", None)
        rmse_cv = unc.get("rmse_cv", None)
        if r2_cv is not None and np.isfinite(r2_cv):
            lines.insert(9, f"- **R² (LOOCV)**: {float(r2_cv):.4f}")
        if rmse_cv is not None and np.isfinite(rmse_cv):
            lines.insert(10, f"- **RMSE (LOOCV)**: {float(rmse_cv):.4f} nm")
    except Exception:
        pass

    lines.extend(
        [
            "## Aggregated Spectra",
            "",
            f"- **Noise metrics**: `{noise_metrics_path}`",
            f"- **Aggregated summary CSV**: `{summary_csv_path}`",
            f"- **Concentration response metrics**: `{response_metrics_path}`",
        ]
    )
    # Optional: band-wise regressions CSV and per-trial plots
    try:
        dbg_csv = os.path.join(out_root, "metrics", "debug_all_wavelength_regressions.csv")
        if os.path.exists(dbg_csv):
            lines.append(f"- **Band-wise regressions CSV**: `{dbg_csv}`")
    except Exception:
        pass
    try:
        agg_dir = os.path.join(out_root, "plots", "aggregated")
        if os.path.isdir(agg_dir):
            lines.append(f"- **Per-trial aggregated plots folder**: `{agg_dir}`")
    except Exception:
        pass
    # ROI selection details and plots
    try:
        with open(response_metrics_path) as f:
            resp_json = json.load(f)
        lines.extend(
            [
                "",
                "### ROI Selection Details (Full-Scan Analysis)",
                "",
                "*Note: Final calibration uses monotonic feature detection which may select a different optimal wavelength.*",
            ]
        )
        rsel = resp_json
        # Pre-compute useful arrays and stats for summary/profile
        wl_arr = np.array(rsel.get("wavelengths", []), dtype=float)
        slopes_arr_r = np.array(rsel.get("slopes", []), dtype=float)
        r2_arr_r = np.array(rsel.get("r_squared", []), dtype=float)
        best_r2_wl_val = None
        if wl_arr.size and r2_arr_r.size and wl_arr.size == r2_arr_r.size:
            try:
                best_idx_tmp = int(np.nanargmax(r2_arr_r))
                best_r2_wl_val = float(wl_arr[best_idx_tmp])
            except Exception:
                best_r2_wl_val = None
        max_slope_wl_val = rsel.get("max_slope_wavelength", None)
        roi_start_val = rsel.get("roi_start_wavelength", None)
        roi_end_val = rsel.get("roi_end_wavelength", None)
        # Slope-to-noise from global repeatability if present
        global_std_val = None
        try:
            rep = (
                rsel.get("roi_repeatability", {})
                if isinstance(rsel.get("roi_repeatability", {}), dict)
                else {}
            )
            global_std_val = rep.get("global", {}).get("std_transmittance", None)
        except Exception:
            global_std_val = None
        stn_profile = None
        if (
            (global_std_val is not None)
            and (float(global_std_val) > 0)
            and slopes_arr_r.size
            and (wl_arr.size == slopes_arr_r.size)
        ):
            try:
                stn_profile = np.abs(slopes_arr_r) / float(global_std_val)
            except Exception:
                stn_profile = None
        sel_metric = rsel.get("roi_selection_metric")
        if sel_metric:
            lines.append(f"- Selection metric: {sel_metric}")
        try:
            roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
            if roi_cfg:
                r2w = roi_cfg.get("r2_weight", None)
                bhw = roi_cfg.get("band_half_width", None)
                bw = roi_cfg.get("band_window", None)
                mincorr = roi_cfg.get("min_corr", None)
                minslope = roi_cfg.get("min_abs_slope", None)
                adapt = roi_cfg.get("adaptive_band", {}) or {}
                lines.append(f"- r2_weight: {r2w}")
                lines.append(f"- band_half_width: {bhw}")
                lines.append(f"- band_window: {bw}")
                lines.append(f"- min_corr: {mincorr}")
                lines.append(f"- min_abs_slope: {minslope}")
                if adapt:
                    lines.append(
                        f"- adaptive_band: enabled={bool(adapt.get('enabled', False))}, slope_fraction={adapt.get('slope_fraction')}, max_half_width={adapt.get('max_half_width')}"
                    )
        except Exception:
            pass
        try:
            rs = rsel
            lines.append(
                f"- ROI: {rs.get('roi_start_wavelength'):.2f}–{rs.get('roi_end_wavelength'):.2f} nm"
            )
            lines.append(f"- Max R²: {float(rs.get('max_r_squared', float('nan'))):.4f}")
            lines.append(
                f"- Max slope @ λ: {float(rs.get('max_slope_wavelength', float('nan'))):.2f} nm"
            )
        except Exception:
            pass
        # Top candidates (limit 5)
        try:
            cands = rsel.get("candidates", [])
            if isinstance(cands, list) and cands:
                lines.append("- Top candidates:")
                for c in cands[:5]:
                    try:
                        lines.append(
                            f"  - wl={float(c.get('wavelength')):.2f} nm, R2={float(c.get('r2')):.4f}, score={float(c.get('score')):.4f}"
                        )
                    except Exception:
                        continue
        except Exception:
            pass
        # Link plots if exist
        if response_plot_path:
            lines.append(f"- Response plot: `{response_plot_path}`")
            with contextlib.suppress(Exception):
                lines.extend(
                    [
                        "",
                        "#### Concentration Response",
                        f"![]({response_plot_path})",
                    ]
                )
        # Generate slope-to-noise profile plot within ROI if available
        try:
            if stn_profile is not None and wl_arr.size:
                plots_dir = os.path.join(out_root, "plots")
                _ensure_dir(plots_dir)
                sn_path = os.path.join(plots_dir, "slope_to_noise_profile.png")
                figsn, axsn = plt.subplots(figsize=(10, 3))
                axsn.plot(wl_arr, stn_profile, color="purple", linewidth=1.2)
                if (roi_start_val is not None) and (roi_end_val is not None):
                    axsn.axvspan(
                        float(roi_start_val),
                        float(roi_end_val),
                        color="orange",
                        alpha=0.2,
                        label="ROI",
                    )
                if best_r2_wl_val is not None:
                    axsn.axvline(
                        best_r2_wl_val,
                        color="red",
                        linestyle="--",
                        linewidth=1.0,
                        label="Best R² λ",
                    )
                if max_slope_wl_val is not None:
                    axsn.axvline(
                        float(max_slope_wl_val),
                        color="blue",
                        linestyle="-.",
                        linewidth=1.0,
                        label="Max |slope| λ",
                    )
                axsn.set_xlabel("Wavelength (nm)")
                axsn.set_ylabel("Slope-to-noise (|slope|/σ)")
                axsn.set_title("Slope-to-Noise Profile (within ROI)")
                axsn.grid(True, alpha=0.3)
                axsn.legend(loc="upper right")
                figsn.tight_layout()
                figsn.savefig(sn_path, dpi=200)
                plt.close(figsn)
                lines.append(f"- Slope-to-noise profile: `{sn_path}`")
                try:
                    relsn = os.path.relpath(sn_path, start=reports_dir)
                    lines.append(f"![Slope-to-noise profile]({relsn})")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            fs_metrics = os.path.join(out_root, "metrics", "fullscan_concentration_response.json")
            fs_plot = os.path.join(out_root, "plots", "fullscan_concentration_response.png")
            if os.path.exists(fs_metrics):
                lines.append(f"- Full-scan response metrics: `{fs_metrics}`")
            if os.path.exists(fs_plot):
                lines.append(f"- Full-scan response plot: `{fs_plot}`")
                with contextlib.suppress(Exception):
                    lines.extend(
                        [
                            "",
                            "#### Full-scan Concentration Response",
                            f"![]({fs_plot})",
                        ]
                    )
        except Exception:
            pass
        # Per-Gas Summary table
        try:
            lines.extend(["", "## Per-Gas Summary", ""])
            header = "| ROI | Observed center (nm) | Best R² λ (nm) | Max |slope| λ (nm) | STN@BestR² | STN@Max|slope| | Selected | CV R² | RMSE | LOD | LOQ |"
            sep = "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"
            lines.extend([header, sep])
            # Observed center from validation
            obs_center = None
            try:
                vj = (
                    rsel.get("validation", {})
                    if isinstance(rsel.get("validation", {}), dict)
                    else {}
                )
                oc = vj.get("observed_center", None)
                if oc is not None:
                    obs_center = float(oc)
            except Exception:
                obs_center = None
            roi_label = "NA"
            try:
                if (roi_start_val is not None) and (roi_end_val is not None):
                    roi_label = f"{float(roi_start_val):.2f}–{float(roi_end_val):.2f}"
            except Exception:
                pass

            # Compute STN at specific λ via interpolation
            def _interp_stn(x):
                try:
                    if (
                        stn_profile is not None
                        and wl_arr.size
                        and stn_profile.size == wl_arr.size
                        and (x is not None)
                    ):
                        return float(np.interp(float(x), wl_arr, stn_profile))
                except Exception:
                    return float("nan")
                return float("nan")

            stn_best = _interp_stn(best_r2_wl_val)
            stn_maxs = _interp_stn(max_slope_wl_val)
            # Selected model and CV metrics
            sel_name = None
            sel_r2cv = float("nan")
            sel_rmsecv = float("nan")
            try:
                # Prefer explicit selection file if available
                sel_path = os.path.join(out_root, "metrics", "multivariate_selection.json")
                if os.path.exists(sel_path):
                    with open(sel_path) as f:
                        sjson = json.load(f)
                    # If calib carries selected_model, use that key for metrics
                    sm = str(calib.get("selected_model", "")).strip().lower()
                    if not sm:
                        sm = str(sjson.get("best_method", "")).strip().lower()
                    if sm:
                        if sm.startswith("plsr"):
                            sel_name = "PLSR"
                            key = "plsr"
                        elif sm.startswith("ica"):
                            sel_name = "ICA"
                            key = "ica"
                        elif sm.startswith("mcr"):
                            sel_name = "MCR-ALS"
                            key = "mcr_als"
                        else:
                            key = None
                        if key is not None:
                            scores = (
                                sjson.get("scores", {})
                                if isinstance(sjson.get("scores", {}), dict)
                                else {}
                            )
                            sc = (
                                scores.get(key, {}) if isinstance(scores.get(key, {}), dict) else {}
                            )
                            r2c = sc.get("r2_cv", None)
                            rmsec = sc.get("rmse_cv", None)
                            if r2c is not None:
                                sel_r2cv = float(r2c)
                            if rmsec is not None:
                                sel_rmsecv = float(rmsec)
                # Fallback to PLSR model metrics
                if sel_name is None and isinstance(calib.get("plsr_model"), dict):
                    pm = calib["plsr_model"]
                    sel_name = "PLSR"
                    if pm.get("r2_cv", None) is not None:
                        sel_r2cv = float(pm.get("r2_cv"))
                    if pm.get("rmse_cv", None) is not None:
                        sel_rmsecv = float(pm.get("rmse_cv"))
            except Exception:
                pass
            # LOD/LOQ from calibration
            lod_v = calib.get("lod")
            loq_v = calib.get("loq")
            try:
                lod_v = float(lod_v) if lod_v is not None else float("nan")
            except Exception:
                lod_v = float("nan")
            try:
                loq_v = float(loq_v) if loq_v is not None else float("nan")
            except Exception:
                loq_v = float("nan")
            # Compose row
            row = (
                f"| {roi_label} | "
                f"{(f'{obs_center:.2f}' if obs_center is not None else 'NA')} | "
                f"{(f'{best_r2_wl_val:.2f}' if best_r2_wl_val is not None else 'NA')} | "
                f"{(f'{float(max_slope_wl_val):.2f}' if max_slope_wl_val is not None else 'NA')} | "
                f"{(f'{stn_best:.3f}' if np.isfinite(stn_best) else 'NA')} | "
                f"{(f'{stn_maxs:.3f}' if np.isfinite(stn_maxs) else 'NA')} | "
                f"{(sel_name if sel_name is not None else 'NA')} | "
                f"{(f'{sel_r2cv:.4f}' if np.isfinite(sel_r2cv) else 'NA')} | "
                f"{(f'{sel_rmsecv:.4f}' if np.isfinite(sel_rmsecv) else 'NA')} | "
                f"{(f'{lod_v:.4f}' if np.isfinite(lod_v) else 'NA')} | "
                f"{(f'{loq_v:.4f}' if np.isfinite(loq_v) else 'NA')} |"
            )
            lines.append(row)
        except Exception:
            pass
    except Exception:
        pass

    # Link QC summary if present
    try:
        qc_path = os.path.join(out_root, "metrics", "qc_summary.json")
        if os.path.exists(qc_path):
            lines.append(f"- **Quality control summary**: `{qc_path}`")
    except Exception:
        pass

    # Link environment compensation summary if present
    try:
        env_path = os.path.join(out_root, "metrics", "environment_compensation.json")
        if os.path.exists(env_path):
            lines.append(f"- **Environment compensation**: `{env_path}`")
            try:
                with open(env_path) as f:
                    env_json = json.load(f)
                r2c = env_json.get("r2_conc_only", None)
                r2f = env_json.get("r2_full", None)
                dr2 = env_json.get("delta_r2", None)
                rmsec = env_json.get("rmse_conc_only", None)
                rmsef = env_json.get("rmse_full", None)
                drmse = env_json.get("delta_rmse", None)
                coef = (
                    env_json.get("estimated_coefficients", {})
                    if isinstance(env_json.get("estimated_coefficients", {}), dict)
                    else {}
                )
                ct = coef.get("temperature", None)
                ch = coef.get("humidity", None)
                # Append compact summary
                lines.extend(
                    [
                        "",
                        "### Environment Compensation Summary",
                    ]
                )
                if any(v is not None for v in [r2c, r2f, dr2, rmsec, rmsef, drmse]):
                    if r2c is not None:
                        lines.append(f"- R² (conc-only): {float(r2c):.4f}")
                    if r2f is not None:
                        lines.append(f"- R² (with env): {float(r2f):.4f}")
                    if dr2 is not None:
                        lines.append(f"- ΔR²: {float(dr2):.4f}")
                    if rmsec is not None:
                        lines.append(f"- RMSE (conc-only): {float(rmsec):.4f} nm")
                    if rmsef is not None:
                        lines.append(f"- RMSE (with env): {float(rmsef):.4f} nm")
                    if drmse is not None:
                        lines.append(f"- ΔRMSE: {float(drmse):.4f} nm")
                if any(v is not None for v in [ct, ch]):
                    lines.append(f"- Estimated cT: {ct if ct is not None else 'n/a'}")
                    lines.append(f"- Estimated cH: {ch if ch is not None else 'n/a'}")
            except Exception:
                pass
    except Exception:
        pass

    if canonical_plot_path:
        lines.append(f"- **Canonical overlay plot**: `{canonical_plot_path}`")
        try:
            rel = os.path.relpath(canonical_plot_path, start=reports_dir)
            lines.append(f"![Canonical overlay]({rel})")
        except Exception:
            pass
    # Link deconvolution artifacts if present
    try:
        ica_metrics = os.path.join(out_root, "metrics", "deconvolution_ica.json")
        if os.path.exists(ica_metrics):
            lines.append(f"- **Deconvolution (ICA)**: `{ica_metrics}`")
            comp_plot = os.path.join(out_root, "plots", "ica_components.png")
            pv_plot = os.path.join(out_root, "plots", "ica_pred_vs_actual.png")
            for title, pth in [
                ("ICA components", comp_plot),
                ("ICA predicted vs actual", pv_plot),
            ]:
                if os.path.exists(pth):
                    try:
                        rel = os.path.relpath(pth, start=reports_dir)
                        lines.append(f"![{title}]({rel})")
                    except Exception:
                        pass
                    # Optional bootstrap CIs for selected model
                    try:
                        bc = mv_cfg.get("bootstrap_ci", {}) if isinstance(mv_cfg, dict) else {}
                        if bc.get("enabled", False):
                            y_true_b = np.array(calib.get("selected_actual", []), dtype=float)
                            y_pred_b = np.array(calib.get("selected_predictions", []), dtype=float)
                            if y_true_b.size and y_true_b.size == y_pred_b.size:
                                iters = int(bc.get("iterations", 1000))
                                rng = np.random.default_rng(0)
                                n = y_true_b.size
                                r2_s = []
                                rmse_s = []
                                for _ in range(max(1, iters)):
                                    idx = rng.integers(0, n, size=n)
                                    yt = y_true_b[idx]
                                    yp = y_pred_b[idx]
                                    if np.isfinite(np.var(yt)) and np.var(yt) > 0:
                                        r2_s.append(float(r2_score(yt, yp)))
                                    else:
                                        r2_s.append(float("nan"))
                                    rmse_s.append(float(np.sqrt(mean_squared_error(yt, yp))))

                                def _nan_ci(arr, lo=2.5, hi=97.5):
                                    a = np.array(arr, dtype=float)
                                    return float(np.nanpercentile(a, lo)), float(
                                        np.nanpercentile(a, hi)
                                    )

                                r2_lo, r2_hi = _nan_ci(r2_s)
                                rmse_lo, rmse_hi = _nan_ci(rmse_s)
                                unc = (
                                    calib.get("uncertainty", {})
                                    if isinstance(calib.get("uncertainty", {}), dict)
                                    else {}
                                )
                                unc["r2_cv_ci"] = [r2_lo, r2_hi]
                                unc["rmse_cv_ci"] = [rmse_lo, rmse_hi]
                                calib["uncertainty"] = unc
                    except Exception:
                        pass
        mcr_metrics = os.path.join(out_root, "metrics", "deconvolution_mcr_als.json")
        if os.path.exists(mcr_metrics):
            lines.append(f"- **Deconvolution (MCR-ALS)**: `{mcr_metrics}`")
            comp_plot = os.path.join(out_root, "plots", "mcr_als_components.png")
            pv_plot = os.path.join(out_root, "plots", "mcr_als_pred_vs_actual.png")
            for title, pth in [
                ("MCR-ALS components", comp_plot),
                ("MCR-ALS predicted vs actual", pv_plot),
            ]:
                if os.path.exists(pth):
                    try:
                        rel = os.path.relpath(pth, start=reports_dir)
                        lines.append(f"![{title}]({rel})")
                    except Exception:
                        pass
        # Link PLSR predicted vs actual and coefficient plots if present
        try:
            plsr_pv = os.path.join(out_root, "plots", "plsr_pred_vs_actual.png")
            plsr_coef = os.path.join(out_root, "plots", "plsr_coefficients.png")
            plsr_cv = os.path.join(out_root, "plots", "plsr_cv_curves.png")
            plsr_resid = os.path.join(out_root, "plots", "plsr_residuals.png")
            if os.path.exists(plsr_pv):
                rel = os.path.relpath(plsr_pv, start=reports_dir)
                lines.append(f"- **PLSR predicted vs actual**: `{plsr_pv}`")
                lines.append(f"![PLSR predicted vs actual]({rel})")
            if os.path.exists(plsr_coef):
                relc = os.path.relpath(plsr_coef, start=reports_dir)
                lines.append(f"- **PLSR coefficients**: `{plsr_coef}`")
                lines.append(f"![PLSR coefficients]({relc})")
            if os.path.exists(plsr_cv):
                relcv = os.path.relpath(plsr_cv, start=reports_dir)
                lines.append(f"- **PLSR CV curves**: `{plsr_cv}`")
                lines.append(f"![PLSR CV curves]({relcv})")
            if os.path.exists(plsr_resid):
                relrd = os.path.relpath(plsr_resid, start=reports_dir)
                lines.append(f"- **PLSR residual diagnostics**: `{plsr_resid}`")
                lines.append(f"![PLSR residual diagnostics]({relrd})")
        except Exception:
            pass
    except Exception:
        pass
    if response_plot_path:
        lines.append(f"- **Concentration response plot**: `{response_plot_path}`")
        try:
            rel = os.path.relpath(response_plot_path, start=reports_dir)
            lines.append(f"![Concentration response]({rel})")
        except Exception:
            pass
    if repeatability_plot_path:
        lines.append(f"- **ROI repeatability plot**: `{repeatability_plot_path}`")
        try:
            rel = os.path.relpath(repeatability_plot_path, start=reports_dir)
            lines.append(f"![ROI repeatability]({rel})")
        except Exception:
            pass
    # Link multivariate selection if present
    try:
        sel_path = os.path.join(out_root, "metrics", "multivariate_selection.json")
        if os.path.exists(sel_path):
            lines.append(f"- **Multivariate model selection**: `{sel_path}`")
            try:
                with open(sel_path) as f:
                    sel_json = json.load(f)
                bm = sel_json.get("best_method")
                br2 = sel_json.get("best_r2_cv")
                lines.extend(
                    [
                        "",
                        "### Multivariate Selection",
                    ]
                )
                if bm is not None:
                    if br2 is not None and np.isfinite(br2):
                        lines.append(f"- Best by CV R²: {bm} ({float(br2):.4f})")
                    else:
                        lines.append(f"- Best by CV R²: {bm}")
                # Candidate table
                scores = (
                    sel_json.get("scores", {})
                    if isinstance(sel_json.get("scores", {}), dict)
                    else {}
                )
                if scores:
                    lines.extend(
                        [
                            "",
                            "| Model | CV R² | RMSE | Selected |",
                            "|---|---:|---:|:---:|",
                        ]
                    )

                    def _fmt(v):
                        try:
                            return f"{float(v):.4f}"
                        except Exception:
                            return "NA"

                    # Determine policy-selected model if available
                    sel_model = None
                    try:
                        if isinstance(calib, dict):
                            sm = str(calib.get("selected_model", "")).strip().lower()
                            if sm:
                                if sm.startswith("plsr"):
                                    sel_model = "plsr"
                                elif sm.startswith("ica"):
                                    sel_model = "ica"
                                elif sm.startswith("mcr"):
                                    sel_model = "mcr_als"
                    except Exception:
                        sel_model = None
                    for key, label in [
                        ("plsr", "PLSR"),
                        ("ica", "ICA"),
                        ("mcr_als", "MCR-ALS"),
                    ]:
                        sc = scores.get(key, {}) if isinstance(scores.get(key, {}), dict) else {}
                        r2v = _fmt(sc.get("r2_cv"))
                        rmsev = _fmt(sc.get("rmse_cv"))
                        selmark = "✔" if (sel_model == key) else ""
                        lines.append(f"| {label} | {r2v} | {rmsev} | {selmark} |")
                    # Explicitly state selection
                    if sel_model:
                        with contextlib.suppress(Exception):
                            lines.append(f"- Selected by policy: {sel_model}")
                # Selection policy summary (if configured)
                mv_cfg = (
                    CONFIG.get("calibration", {}).get("multivariate", {})
                    if isinstance(CONFIG, dict)
                    else {}
                )
                if mv_cfg:
                    try:
                        lines.append("")
                        lines.append(
                            "- Selection policy: "
                            f"min_r2_cv={mv_cfg.get('min_r2_cv', 'NA')}, "
                            f"improve_margin={mv_cfg.get('improve_margin', 'NA')}, "
                            f"prefer_plsr_on_tie={mv_cfg.get('prefer_plsr_on_tie', 'NA')}"
                        )
                    except Exception:
                        pass
                # Embed CV R² comparison plot if exists
                try:
                    cv_plot = os.path.join(out_root, "plots", "multivariate_cv_r2.png")
                    if os.path.exists(cv_plot):
                        lines.append(f"- CV R² comparison plot: `{cv_plot}`")
                        try:
                            rel = os.path.relpath(cv_plot, start=reports_dir)
                            lines.append(f"![Multivariate CV R² Comparison]({rel})")
                        except Exception:
                            pass
                    # Embed selected model predicted vs actual if exists
                    sel_plot = os.path.join(out_root, "plots", "selected_pred_vs_actual.png")
                    if os.path.exists(sel_plot):
                        lines.append(f"- Selected model predicted vs actual: `{sel_plot}`")
                        try:
                            rel2 = os.path.relpath(sel_plot, start=reports_dir)
                            lines.append(f"![Selected model predicted vs actual]({rel2})")
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    if performance_metrics_path:
        lines.append(f"- **ROI performance metrics**: `{performance_metrics_path}`")
    if dynamics_summary_path:
        lines.append(f"- **Dynamics summary**: `{dynamics_summary_path}`")
    if dynamics_plot_path:
        lines.append(f"- **Dynamics plot**: `{dynamics_plot_path}`")
        try:
            rel = os.path.relpath(dynamics_plot_path, start=reports_dir)
            lines.append(f"![Dynamics]({rel})")
        except Exception:
            pass
    if metadata_path:
        lines.append(f"- **Run metadata**: `{metadata_path}`")
    if archive_path:
        lines.append(f"- **Archive copy**: `{archive_path}`")

    perf_summary = summarize_roi_performance(performance or {})
    if perf_summary:
        lines.extend(["", "### ROI Performance Snapshot", "", f"- {perf_summary}"])

    if report_artifacts:
        lines.extend(["", "### Report Artifacts", ""])
        for key, value in report_artifacts.items():
            lines.append(f"- {key}: `{value}`")

    if trend_plots:
        lines.extend(["", "### Trend Plots", ""])
        for key, value in trend_plots.items():
            lines.append(f"- {key}: `{value}`")

    if dynamics_summary:
        lines.extend(["", "### Dynamics Overview", ""])
        # Handle both old format (overall.mean_T90) and new format (T90_mean_s)
        overall = dynamics_summary.get("overall", None)
        if overall and isinstance(overall, dict) and "mean_T90" in overall:
            # Old format with nested 'overall' dict
            t90_mean = overall.get("mean_T90")
            t90_std = overall.get("std_T90", 0.0)
            t10_mean = overall.get("mean_T10")
            t10_std = overall.get("std_T10", 0.0)
        else:
            # New format with flat structure (T90_mean_s, T10_mean_s)
            t90_mean = dynamics_summary.get("T90_mean_s")
            t90_std = dynamics_summary.get("T90_std_s", 0.0)
            t10_mean = dynamics_summary.get("T10_mean_s")
            t10_std = dynamics_summary.get("T10_std_s", 0.0)

        if t90_mean is not None or t10_mean is not None:
            t90_str = f"{t90_mean:.2f}" if t90_mean is not None else "N/A"
            t90_std_str = f"{t90_std:.2f}" if t90_std is not None else "0.00"
            t10_str = f"{t10_mean:.2f}" if t10_mean is not None else "N/A"
            t10_std_str = f"{t10_std:.2f}" if t10_std is not None else "0.00"
            lines.append(
                f"- Overall: T90={t90_str}s +/- {t90_std_str}s, T10={t10_str}s +/- {t10_std_str}s"
            )

        per_conc = dynamics_summary.get("per_concentration", {})
        for conc_key, stats in sorted(per_conc.items(), key=lambda kv: float(kv[0])):
            lines.append(
                f"- {conc_key} ppm: "
                f"T90={stats.get('mean_T90', float('nan')):.2f}s +/- {stats.get('std_T90', 0.0):.2f}s, "
                f"T10={stats.get('mean_T10', float('nan')):.2f}s +/- {stats.get('std_T10', 0.0):.2f}s"
            )

    # Recommendations section
    try:
        lines.extend(["", "## Recommendations", ""])
        # ROI recommendation
        roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
        roi_min = roi_cfg.get("min_wavelength", None)
        roi_max = roi_cfg.get("max_wavelength", None)
        if roi_min is not None and roi_max is not None:
            lines.append(
                f"- Keep analysis within ROI [{float(roi_min):.0f}, {float(roi_max):.0f}] nm; treat outside as contextual only."
            )
        # Validation recommendation (align expected center)
        try:
            with open(response_metrics_path) as f:
                rsj = json.load(f)
            v = rsj.get("validation", {}) if isinstance(rsj.get("validation", {}), dict) else {}
            expc = v.get("expected_center", None)
            obsc = v.get("observed_center", None)
            within = v.get("within_tolerance", None)
            tol = v.get("tolerance", None)
            if expc is not None and obsc is not None and within is False:
                lines.append(
                    f"- Align prior: set expected_center≈{float(obsc):.1f} nm (current {float(expc):.1f}±{float(tol) if tol is not None else 'NA'})."
                )
        except Exception:
            pass
        # Smoothing recommendation
        try:
            sm = (
                CONFIG.get("preprocessing", {}).get("smooth", {})
                if isinstance(CONFIG, dict)
                else {}
            )
            win = sm.get("window", None)
            if win and int(win) > 11:
                lines.append(
                    "- Reduce smoothing window (e.g., 11) to avoid shifting band apex visually."
                )
        except Exception:
            pass
        # Gating recommendation
        try:
            mv_cfg = (
                CONFIG.get("calibration", {}).get("multivariate", {})
                if isinstance(CONFIG, dict)
                else {}
            )
            minr = mv_cfg.get("min_r2_cv", None)
            im = mv_cfg.get("improve_margin", None)
            if minr is not None and im is not None:
                lines.append(
                    f"- Selection gating: min_r2_cv={float(minr):.2f}, improve_margin={float(im):.2f}. Relax slightly if no model is selected, or keep for conservatism."
                )
        except Exception:
            pass
        # Data quality recommendation
        lines.append(
            "- If CV R² is modest, add more concentration levels/replicates to stabilize LOOCV."
        )
        # Debug recommendation
        dbg_csv = os.path.join(out_root, "metrics", "debug_all_wavelength_regressions.csv")
        if os.path.exists(dbg_csv):
            lines.append(
                f"- Use Top-5 R² in `{dbg_csv}` to finalize per-gas ROI centers (expect ±5–10 nm stability)."
            )
    except Exception:
        pass

    lines.extend(["", "### Files per Concentration", ""])

    for conc, trials in sorted(aggregated_paths.items(), key=lambda kv: kv[0]):
        lines.append(f"- **{conc:g}**")
        for trial, path in trials.items():
            lines.append(f"  - `{trial}`: `{path}`")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def save_calibration_outputs(calib: dict[str, object], out_root: str, name_suffix: str = ""):
    _save_calibration_outputs_src(calib, out_root, name_suffix)


# High-level run helper (optional for CLI)
# ----------------------


def _resolve_gas_benchmarks(label: Optional[str]) -> dict[str, float]:
    benchmarks = {
        "Acetone": {
            "slope_nm_per_ppm": 0.116,
            "r2": 0.95,
            "lod_ppm": 3.26,
            "response_s": 26.0,
            "recovery_s": 32.0,
        },
        "Methanol": {"slope_nm_per_ppm": 0.081, "r2": 0.88},
        "Ethanol": {"r2": 0.27},
        "Isopropanol": {"r2": 0.67},
        "Toluene": {"r2": 0.31},
        "Xylene": {"r2": 0.65},
    }
    return benchmarks.get(str(label), {})


def _apply_response_overrides(
    base_cfg: dict[str, object], label: Optional[str]
) -> dict[str, object]:
    cfg = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    overrides = cfg.get("overrides", {}) if isinstance(cfg.get("overrides", {}), dict) else {}
    override = overrides.get(str(label), {}) if label is not None else {}
    # Merge override onto base without mutating original
    merged = {k: v for k, v in cfg.items() if k != "overrides"}
    if isinstance(override, dict):
        merged.update(override)
    merged["overrides"] = overrides
    merged["_applied_override"] = override
    return merged


def run_full_pipeline(
    root_dir: str,
    ref_path: str,
    out_root: str,
    diff_threshold: float = 0.01,
    avg_top_n: Optional[int] = None,
    scan_full: bool = False,
    top_k_candidates: int = 5,
    dataset_label: Optional[str] = None,
) -> dict[str, object]:
    """Run: scan → average frames per trial → preprocessing → calibration → persistence."""

    run_timestamp = _timestamp()
    metadata: dict[str, object] = {
        "run_timestamp": run_timestamp,
        "data_dir": os.path.abspath(root_dir),
        "ref_path": os.path.abspath(ref_path) if ref_path else None,
        "out_root": os.path.abspath(out_root),
        "diff_threshold": diff_threshold,
        "avg_top_n": avg_top_n,
        "scan_full": scan_full,
        "top_k_candidates": top_k_candidates,
        "config_snapshot": CONFIG,
        "dataset_label": dataset_label,
        "preprocessing": CONFIG.get("preprocessing", {}),
        "archiving": CONFIG.get("archiving", {}),
        "reporting": CONFIG.get("reporting", {}),
        "trials": {},
    }

    roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
    shift_cfg = roi_cfg.get("shift", {}) if isinstance(roi_cfg.get("shift", {}), dict) else {}
    minimal_outputs = bool(shift_cfg.get("minimal_outputs", False))
    metadata["minimal_outputs"] = minimal_outputs

    # Record resolved shift-scan configuration for auditability.
    try:
        roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
        shift_cfg = roi_cfg.get("shift", {}) if isinstance(roi_cfg.get("shift", {}), dict) else {}
        win_cfg = shift_cfg.get("window_nm", None)
        if isinstance(win_cfg, (list, tuple)):
            win_list = []
            for w in win_cfg:
                try:
                    wv = float(w)
                except Exception:
                    continue
                if np.isfinite(wv) and wv > 0:
                    win_list.append(wv)
            win_resolved = win_list
        else:
            try:
                win_resolved = [float(win_cfg)] if win_cfg is not None else []
            except Exception:
                win_resolved = []
        step_val = float(shift_cfg.get("step_nm", float("nan")))
        upsample_val = shift_cfg.get("upsample", None)
        metadata["roi_shift_scan_resolved"] = {
            "window_nm": win_resolved,
            "step_nm": step_val,
            "upsample": upsample_val,
            "source": "roi.shift",
        }
    except Exception:
        pass

    mapping = scan_experiment_root(root_dir)
    metadata["concentrations_detected"] = [float(c) for c in sorted(mapping.keys())]
    ref_df = load_reference_csv(ref_path) if ref_path else None

    preproc_settings = CONFIG.get("preprocessing", {})
    calib_settings = CONFIG.get("calibration", {})
    bool(calib_settings.get("use_transmittance", True))
    outlier_cfg = preproc_settings.get("outlier_rejection", {})
    apply_frames = preproc_settings.get("enabled", False) and preproc_settings.get(
        "apply_to_frames", False
    )
    apply_trans = preproc_settings.get("enabled", False) and preproc_settings.get(
        "apply_to_transmittance", True
    )
    dynamics_cfg = CONFIG.get("dynamics", {})
    dynamics_enabled = dynamics_cfg.get("enabled", True)
    baseline_n = (
        int(dynamics_cfg.get("baseline_frames", 20) or 20) if isinstance(dynamics_cfg, dict) else 20
    )
    baseline_n = max(1, baseline_n)
    response_cfg = _apply_response_overrides(
        CONFIG.get("response_series", {})
        if isinstance(CONFIG.get("response_series", {}), dict)
        else {},
        dataset_label,
    )
    (response_cfg.pop("_applied_override", {}) if isinstance(response_cfg, dict) else {})
    stability_cfg = (
        CONFIG.get("stability", {}) if isinstance(CONFIG.get("stability", {}), dict) else {}
    )
    (int(stability_cfg.get("min_block", 0)) if stability_cfg.get("min_block") else None)

    stable_by_conc: dict[float, dict[str, pd.DataFrame]] = {}
    stable_raw_by_conc: dict[float, dict[str, pd.DataFrame]] = {}
    top_path: dict[str, dict[float, dict[str, pd.DataFrame]]] = {}
    time_series_outputs: dict[str, dict[str, dict[str, object]]] = {}
    responsive_delta_by_conc: dict[float, dict[str, dict[str, object]]] = {}
    responsive_trend_fallback: dict[float, dict[str, float]] = {}
    for conc, trials in mapping.items():
        conc_key = float(conc)
        conc_entry: dict[str, object] = {
            "raw_trial_count": len(trials),
            "retained_trial_count": 0,
        }
        metadata["trials"][str(conc_key)] = conc_entry
        trial_debug = metadata.setdefault("trial_debug", {}).setdefault(str(conc_key), {})

        processed_trials: dict[str, pd.DataFrame] = {}
        raw_trials: dict[str, pd.DataFrame] = {}
        averaged_intensity_trials: dict[str, pd.DataFrame] = {}
        averaged_trans_trials: dict[str, pd.DataFrame] = {}
        averaged_abs_trials: dict[str, pd.DataFrame] = {}
        spectral_arrays: list[np.ndarray] = []
        trial_names: list[str] = []
        base_wavelengths: Optional[np.ndarray] = None
        wavelengths_consistent = True

        trial_quality_scores: dict[str, float] = {}

        for trial, frames in trials.items():
            frames_sorted = _sort_frame_paths(frames)
            dfs = [_read_csv_spectrum(p) for p in frames_sorted]
            dfs = [df for df in dfs if not df.empty]
            info_entry: dict[str, object] = {
                "frame_count": len(frames_sorted),
                "valid_frame_count": len(dfs),
            }
            trial_debug[trial] = info_entry
            if not dfs:
                info_entry["status"] = "no_valid_frames"
                continue

            if apply_frames:
                dfs = [_preprocess_dataframe(df, stage="frame") for df in dfs]

            # Per-trial reference scaling and simple responsive frame selection
            responsive_indices: list[int] = []
            trial_ref_df = ref_df
            if ref_df is not None:
                baseline_frames = dfs[:baseline_n]
                trial_ref_df, _ = _scale_reference_to_baseline(
                    ref_df, baseline_frames, percentile=95.0
                )

            if response_cfg.get("enabled", False):
                simple_series_df, simple_indices, _ = _simple_response_selection(
                    dfs,
                    trial_ref_df,
                    dataset_label=dataset_label,
                    response_cfg=response_cfg,
                )
                responsive_indices = list(simple_indices)
                info_entry["response_series"] = {
                    "responsive_frame_count": len(responsive_indices),
                }

                # Build full Δλ time-series per trial for downstream dynamics and reporting.
                try:
                    response_series_df, _, _ = _compute_response_time_series(
                        dfs,
                        trial_ref_df,
                        dataset_label=dataset_label,
                        response_cfg=response_cfg,
                    )
                except Exception as exc:
                    print(
                        f"[WARNING] Failed to compute response time-series for conc={conc_key}, trial={trial}: {exc}"
                    )
                    response_series_df = None

                if response_series_df is not None:
                    try:
                        csv_path, series_plot = _save_response_series(
                            response_series_df,
                            out_root,
                            conc_key,
                            trial,
                            dataset_label,
                        )
                        conc_key_str = str(conc_key)
                        ts_conc = time_series_outputs.setdefault(conc_key_str, {})
                        ts_conc[trial] = {
                            "csv": str(csv_path),
                            "plot": str(series_plot),
                        }
                        info_entry["response_series"].update(
                            {
                                "csv": str(csv_path),
                                "plot": str(series_plot),
                            }
                        )
                    except Exception as exc:
                        info_entry.setdefault("response_series_error", str(exc))

            if response_cfg.get("restrict_to_responsive", False) and responsive_indices:
                frames_for_stability = [dfs[i] for i in responsive_indices if 0 <= i < len(dfs)]
                if not frames_for_stability:
                    frames_for_stability = dfs
            else:
                frames_for_stability = dfs

            weight_mode = (
                stability_cfg.get("weight_mode", "uniform")
                if isinstance(stability_cfg, dict)
                else "uniform"
            )
            top_k = stability_cfg.get("top_k", 0) if isinstance(stability_cfg, dict) else 0
            s, e, weights = find_stable_block(
                frames_for_stability,
                diff_threshold=diff_threshold,
                weight_mode=weight_mode,
                top_k=int(top_k) if top_k else None,
                min_block=int(stability_cfg.get("min_block", 0) or 0)
                if isinstance(stability_cfg, dict)
                else None,
            )

            # Responsive frame selection is handled by _simple_response_selection;
            # no additional ROI-only tightening is applied here.
            avg_df = average_stable_block(frames_for_stability, s, e, weights=weights)
            avg_df_trans = (
                compute_transmittance(avg_df, trial_ref_df)
                if trial_ref_df is not None
                else avg_df.copy(deep=True)
            )
            avg_df_with_abs = _append_absorbance_column(avg_df_trans)
            raw_trials[trial] = avg_df_with_abs.copy(deep=True)

            if apply_trans:
                avg_df_proc = _preprocess_dataframe(avg_df_with_abs, stage="transmittance")
            else:
                avg_df_proc = avg_df_with_abs.copy(deep=True)

            avg_df_proc = _append_absorbance_column(avg_df_proc, inplace=True)

            processed_trials[trial] = avg_df_proc

            # Trial quality scoring for canonical weighting
            try:
                roi_bounds = _resolve_roi_bounds(dataset_label)
            except Exception:
                roi_bounds = (None, None)
            expected_center = None
            try:
                roi_cfg = CONFIG.get("roi", {}) if isinstance(CONFIG, dict) else {}
                per_gas = (
                    roi_cfg.get("per_gas_overrides", {})
                    if isinstance(roi_cfg.get("per_gas_overrides", {}), dict)
                    else roi_cfg.get("per_gas_overrides", {})
                )
                gas_cfg = (
                    per_gas.get(dataset_label, {})
                    if dataset_label is not None and isinstance(per_gas, dict)
                    else {}
                )
                val_cfg = (
                    gas_cfg.get("validation", {})
                    if isinstance(gas_cfg.get("validation", {}), dict)
                    else gas_cfg.get("validation", {})
                )
                expected_center = val_cfg.get("expected_center", None)
            except Exception:
                expected_center = None
            q_score, _ = _score_trial_quality(
                avg_df_proc, roi_bounds=roi_bounds, expected_center=expected_center
            )
            trial_quality_scores[trial] = float(q_score)
            info_entry["quality_score"] = float(q_score)

            if avg_top_n:
                # Restrict to stable block for top-N averaging to avoid transient artifacts
                stable_subset = (
                    frames_for_stability[s : e + 1]
                    if (s <= e and e < len(frames_for_stability))
                    else frames_for_stability
                )
                if not stable_subset:
                    stable_subset = frames_for_stability
                top_avg_int = average_top_frames(stable_subset, top_k=avg_top_n)
                averaged_intensity_trials[trial] = top_avg_int
                if trial_ref_df is not None:
                    top_avg_trans = compute_transmittance(top_avg_int, trial_ref_df)
                else:
                    top_avg_trans = top_avg_int.copy()
                averaged_trans_trials[trial] = top_avg_trans
                averaged_abs_trials[trial] = _append_absorbance_column(top_avg_trans)

            if outlier_cfg.get("enabled", False):
                col = _signal_column(avg_df)
                arr = avg_df[col].to_numpy()
                wl = avg_df["wavelength"].to_numpy()
                if base_wavelengths is None:
                    base_wavelengths = wl
                elif len(wl) != len(base_wavelengths) or not np.allclose(wl, base_wavelengths):
                    wavelengths_consistent = False
                spectral_arrays.append(arr)
                trial_names.append(trial)

        trial_debug["__processed__"] = sorted(processed_trials.keys())
        trial_debug["__raw__"] = sorted(raw_trials.keys())
        conc_entry["processed_trial_count"] = len(processed_trials)
        conc_entry["raw_trial_count_post"] = len(raw_trials)

        flagged_trials: set = set()
        if (
            outlier_cfg.get("enabled", False)
            and wavelengths_consistent
            and len(spectral_arrays) >= 2
        ):
            threshold = outlier_cfg.get("threshold", 3.0)
            flags = detect_outliers(spectral_arrays, threshold=threshold)
            for trial_name, flag in zip(trial_names, flags):
                if flag:
                    flagged_trials.add(trial_name)
                    _record_outlier(metadata, conc_key, trial_name)
            if flags and all(bool(f) for f in flags):
                flagged_trials.clear()
                conc_entry["outlier_relaxed"] = True

        final_trials = {
            trial: df for trial, df in processed_trials.items() if trial not in flagged_trials
        }
        final_raw_trials = {
            trial: raw_trials[trial]
            for trial in processed_trials
            if trial not in flagged_trials and trial in raw_trials
        }
        final_intensity_top = {
            trial: averaged_intensity_trials[trial]
            for trial in processed_trials
            if trial not in flagged_trials and trial in averaged_intensity_trials
        }
        final_trans_top = {
            trial: averaged_trans_trials[trial]
            for trial in processed_trials
            if trial not in flagged_trials and trial in averaged_trans_trials
        }
        final_abs_top = {
            trial: averaged_abs_trials[trial]
            for trial in processed_trials
            if trial not in flagged_trials and trial in averaged_abs_trials
        }

        if not final_trials and processed_trials:
            final_trials = dict(processed_trials)
            final_raw_trials = {
                trial: raw_trials.get(trial, df) for trial, df in processed_trials.items()
            }
            conc_entry["filter_relaxed_all"] = True
        if not final_trials and raw_trials:
            final_trials = dict(raw_trials)
            final_raw_trials = dict(raw_trials)
            conc_entry["raw_fallback"] = True

        if final_trials:
            chosen_source = "retained"
        elif processed_trials:
            chosen_source = "restored_from_processed"
            final_trials = dict(processed_trials)
            final_raw_trials = {
                trial: raw_trials.get(trial, df) for trial, df in processed_trials.items()
            }
        elif raw_trials:
            chosen_source = "restored_from_raw"
            final_trials = dict(raw_trials)
            final_raw_trials = dict(raw_trials)
        else:
            chosen_source = "dropped"
            final_trials = {}
            final_raw_trials = {}

        conc_entry["retained_trial_count"] = len(final_trials)
        conc_entry["restoration_status"] = chosen_source

        # Build quality weights for canonical selection (per concentration)
        final_quality_scores: dict[str, float] = {}
        for trial_name in final_trials:
            w = trial_quality_scores.get(trial_name, 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            final_quality_scores[trial_name] = w
        metadata.setdefault("trial_quality", {})[str(conc_key)] = final_quality_scores

        if final_trials:
            stable_by_conc[conc] = final_trials
            stable_raw_by_conc[conc] = final_raw_trials
            trial_debug["__final__"] = sorted(final_trials.keys())
            trial_debug["__final_raw__"] = sorted(final_raw_trials.keys())
            trial_debug["__final_intensity__"] = (
                sorted(final_intensity_top.keys()) if final_intensity_top else []
            )
            trial_debug["__final_trans__"] = (
                sorted(final_trans_top.keys()) if final_trans_top else []
            )
            trial_debug["__final_absorbance__"] = (
                sorted(final_abs_top.keys()) if final_abs_top else []
            )
            trial_debug["__status__"] = chosen_source
            if avg_top_n:
                if final_intensity_top:
                    top_path.setdefault("intensity", {})[conc] = final_intensity_top
                if final_trans_top:
                    top_path.setdefault("transmittance", {})[conc] = final_trans_top
                if final_abs_top:
                    top_path.setdefault("absorbance", {})[conc] = final_abs_top
        else:
            trial_debug["__final__"] = []
            trial_debug["__final_raw__"] = []
            trial_debug["__final_intensity__"] = []
            trial_debug["__final_trans__"] = []
            trial_debug["__final_absorbance__"] = []
            trial_debug["__status__"] = "dropped"

    # Expose per-trial quality weights for canonical aggregation
    try:
        trial_quality_global: dict[float, dict[str, float]] = {}
        for conc_str, per_trial in metadata.get("trial_quality", {}).items():
            try:
                conc_val = float(conc_str)
            except Exception:
                continue
            if isinstance(per_trial, dict):
                q_map: dict[str, float] = {}
                for t_name, w in per_trial.items():
                    try:
                        w_val = float(w)
                    except Exception:
                        w_val = 1.0
                    if not np.isfinite(w_val) or w_val <= 0.0:
                        w_val = 1.0
                    q_map[str(t_name)] = w_val
                trial_quality_global[conc_val] = q_map
        globals()["TRIAL_WEIGHTS_FOR_CANONICAL"] = trial_quality_global
    except Exception:
        globals()["TRIAL_WEIGHTS_FOR_CANONICAL"] = {}

    if not stable_by_conc:
        raise RuntimeError("No stable blocks found across trials")

    raw_to_save = stable_raw_by_conc if stable_raw_by_conc else stable_by_conc
    aggregated_paths = save_aggregated_spectra(raw_to_save, out_root)
    metadata["stable_concentrations"] = sorted(float(k) for k in stable_by_conc)
    metadata["stable_counts"] = {str(float(k)): len(v) for k, v in stable_by_conc.items()}
    metadata["aggregated_paths"] = aggregated_paths

    signal_views = _build_signal_views(stable_by_conc, stable_raw_by_conc)
    if not signal_views:
        raise RuntimeError("No signal representations available after preprocessing")

    delta_per_trial_df, delta_per_conc_df, responsive_delta_summary = (
        _aggregate_responsive_delta_maps(responsive_delta_by_conc)
    )
    metadata["responsive_delta"] = {
        "per_trial": delta_per_trial_df.to_dict(orient="records")
        if not delta_per_trial_df.empty
        else [],
        "per_concentration": delta_per_conc_df.to_dict(orient="records")
        if not delta_per_conc_df.empty
        else [],
        "summary_by_concentration": responsive_delta_summary,
    }

    multivariate_cfg = (
        calib_settings.get("multivariate", {}) if isinstance(calib_settings, dict) else {}
    )
    multivariate_enabled = bool(multivariate_cfg.get("enabled", False))

    signal_results: dict[str, dict[str, object]] = {}
    for signal_name, view_map in signal_views.items():
        if not view_map:
            continue
        try:
            signal_root = Path(out_root) / "signals" / signal_name
            signal_root.mkdir(parents=True, exist_ok=True)
            canonical_sig = select_canonical_per_concentration(view_map)
            canonical_paths_sig = save_canonical_spectra(canonical_sig, str(signal_root))
            canonical_plot_sig = save_canonical_overlay(canonical_sig, str(signal_root))

            matrix_cache_sig: Optional[tuple[np.ndarray, np.ndarray, np.ndarray]] = None
            if multivariate_enabled and canonical_sig:
                try:
                    matrix_cache_sig = _build_feature_matrix_from_canonical(canonical_sig)
                except ValueError:
                    matrix_cache_sig = None

            # Get per-gas ROI bounds for proper calibration
            min_wl_roi_sig, max_wl_roi_sig = _resolve_roi_bounds(dataset_label)
            response_stats_sig, avg_by_conc_sig = compute_concentration_response(
                view_map,
                override_min_wavelength=min_wl_roi_sig,
                override_max_wavelength=max_wl_roi_sig,
                top_k_candidates=top_k_candidates,
                debug_out_root=str(signal_root),
            )
            repeatability_sig = compute_roi_repeatability(view_map, response_stats_sig)
            performance_sig = compute_roi_performance(repeatability_sig)

            baseline_std = float("nan")
            try:
                baseline_std = float(
                    repeatability_sig.get("global", {}).get("std_transmittance", float("nan"))
                )
            except Exception:
                baseline_std = float("nan")
            baseline_slope = (
                performance_sig.get("regression_slope")
                if isinstance(performance_sig, dict)
                else None
            )
            baseline_slope_to_noise = float("nan")
            if (
                isinstance(baseline_slope, (int, float))
                and np.isfinite(baseline_slope)
                and np.isfinite(baseline_std)
                and baseline_std not in (0.0,)
            ):
                baseline_slope_to_noise = abs(float(baseline_slope)) / float(baseline_std)

            mv_result_sig: Optional[dict[str, object]] = None
            if multivariate_enabled:
                try:
                    mv_source = canonical_sig if canonical_sig else {}
                    mv_result = (
                        _fit_plsr_calibration(mv_source, multivariate_cfg, matrix_cache_sig)
                        if mv_source
                        else None
                    )
                    if mv_result is not None and isinstance(mv_result, dict):
                        try:
                            y_true = matrix_cache_sig[1] if matrix_cache_sig else None
                            baseline_den = (
                                baseline_std
                                if np.isfinite(baseline_std) and baseline_std not in (0.0,)
                                else float("nan")
                            )
                            preds_cv = mv_result.get("predictions_cv")
                            preds_in = mv_result.get("predictions_in")
                            preds_arr = None
                            if isinstance(preds_cv, list):
                                preds_arr = np.array(preds_cv, dtype=float)
                            elif isinstance(preds_in, list):
                                preds_arr = np.array(preds_in, dtype=float)
                            if (
                                preds_arr is not None
                                and preds_arr.size
                                and y_true is not None
                                and np.isfinite(baseline_den)
                                and baseline_den not in (0.0,)
                            ):
                                slope_cov = np.polyfit(
                                    np.asarray(y_true, dtype=float), preds_arr, 1
                                )[0]
                                mv_result["slope_to_noise"] = abs(float(slope_cov)) / baseline_den
                            mv_result["baseline_slope_to_noise"] = baseline_slope_to_noise
                        except Exception:
                            mv_result.setdefault("slope_to_noise", float("nan"))
                        mv_result_sig = mv_result
                except Exception as exc_mv:  # noqa: BLE001
                    mv_result_sig = {"error": str(exc_mv)}

            metrics_path_sig = save_concentration_response_metrics(
                response_stats_sig,
                repeatability_sig,
                str(signal_root),
                name=f"concentration_response_{signal_name}",
            )
            plot_path_sig = save_concentration_response_plot(
                response_stats_sig,
                avg_by_conc_sig,
                str(signal_root),
                name=f"concentration_response_{signal_name}",
                clamp_to_roi=True,
            )
            repeatability_plot_sig = save_roi_repeatability_plot(
                view_map, response_stats_sig, str(signal_root)
            )
            performance_metrics_sig = save_roi_performance_metrics(
                performance_sig, str(signal_root)
            )

            signal_results[signal_name] = {
                "canonical": canonical_sig,
                "canonical_paths": canonical_paths_sig,
                "canonical_plot": canonical_plot_sig,
                "response": response_stats_sig,
                "repeatability": repeatability_sig,
                "performance": performance_sig,
                "multivariate": mv_result_sig,
                "baseline": {
                    "std_transmittance": baseline_std,
                    "slope_to_noise": baseline_slope_to_noise,
                    "regression_r2": performance_sig.get("regression_r2")
                    if isinstance(performance_sig, dict)
                    else float("nan"),
                    "lod_ppm": performance_sig.get("lod_ppm")
                    if isinstance(performance_sig, dict)
                    else float("nan"),
                },
                "metrics_path": metrics_path_sig,
                "plot_path": plot_path_sig,
                "repeatability_plot": repeatability_plot_sig,
                "performance_metrics_path": performance_metrics_sig,
            }
        except Exception as exc:  # noqa: BLE001
            signal_results[signal_name] = {
                "error": str(exc),
            }

    def _compute_signal_score(
        entry: dict[str, object],
    ) -> tuple[float, dict[str, float]]:
        components: dict[str, float] = {}
        perf = entry.get("performance") if isinstance(entry, dict) else None
        if not isinstance(perf, dict):
            components["r2"] = float("nan")
            components["lod_ppm"] = float("nan")
            components["sensitivity"] = float("nan")
        else:
            r2 = perf.get("regression_r2")
            lod = perf.get("lod_ppm")
            slope = perf.get("regression_slope")
            components["r2"] = (
                float(r2) if isinstance(r2, (int, float)) and np.isfinite(r2) else float("nan")
            )
            components["lod_ppm"] = (
                float(lod) if isinstance(lod, (int, float)) and np.isfinite(lod) else float("nan")
            )
            components["sensitivity"] = (
                float(slope)
                if isinstance(slope, (int, float)) and np.isfinite(slope)
                else float("nan")
            )

        baseline_info = entry.get("baseline") if isinstance(entry, dict) else {}
        baseline_slope_to_noise = float("nan")
        baseline_r2 = float("nan")
        if isinstance(baseline_info, dict):
            baseline_slope_to_noise = float(baseline_info.get("slope_to_noise", float("nan")))
            baseline_r2 = float(baseline_info.get("regression_r2", float("nan")))

        score = 0.0
        if np.isfinite(components.get("r2", float("nan"))):
            score += components["r2"]
        if np.isfinite(components.get("lod_ppm", float("nan"))) and components["lod_ppm"] > 0:
            score += 1.0 / components["lod_ppm"]
        if np.isfinite(components.get("sensitivity", float("nan"))):
            score += abs(components["sensitivity"])

        mv_entry = entry.get("multivariate") if isinstance(entry, dict) else None
        mv_r2 = float("nan")
        mv_rmse = float("nan")
        mv_slope_to_noise = float("nan")
        if isinstance(mv_entry, dict) and "error" not in mv_entry:
            mv_r2 = mv_entry.get("r2_cv", float("nan"))
            mv_rmse = mv_entry.get("rmse_cv", float("nan"))
            mv_slope_to_noise = mv_entry.get("slope_to_noise", float("nan"))
        components["plsr_r2_cv"] = (
            float(mv_r2) if isinstance(mv_r2, (int, float)) and np.isfinite(mv_r2) else float("nan")
        )
        components["plsr_rmse_cv"] = (
            float(mv_rmse)
            if isinstance(mv_rmse, (int, float)) and np.isfinite(mv_rmse)
            else float("nan")
        )
        components["plsr_slope_to_noise"] = (
            float(mv_slope_to_noise)
            if isinstance(mv_slope_to_noise, (int, float)) and np.isfinite(mv_slope_to_noise)
            else float("nan")
        )
        components["baseline_r2"] = baseline_r2
        components["baseline_slope_to_noise"] = baseline_slope_to_noise
        if np.isfinite(components["plsr_r2_cv"]):
            score += 2.0 * components["plsr_r2_cv"]
        if np.isfinite(components["plsr_rmse_cv"]) and components["plsr_rmse_cv"] > 0:
            score += 1.0 / components["plsr_rmse_cv"]
        if np.isfinite(components["plsr_slope_to_noise"]):
            score += components["plsr_slope_to_noise"]

        return score, components

    for _sig_name, result in signal_results.items():
        score, components = _compute_signal_score(result)
        result["score"] = score
        result["score_components"] = components

    ranked_signals = sorted(
        signal_results.items(),
        key=lambda kv: kv[1].get("score", float("-inf")),
        reverse=True,
    )

    primary_signal = _resolve_primary_signal(signal_views)
    if ranked_signals:
        best_signal = ranked_signals[0][0]
        if best_signal in signal_views:
            primary_signal = best_signal

    primary_map = signal_views[primary_signal]
    metadata["signal_analysis"] = {
        "available_signals": sorted(signal_views.keys()),
        "primary_signal": primary_signal,
        "ranked_signals": [name for name, _ in ranked_signals],
        "per_signal_results": {
            name: {
                "performance": result.get("performance"),
                "multivariate": result.get("multivariate"),
                "metrics_path": result.get("metrics_path"),
                "plot_path": result.get("plot_path"),
                "repeatability_plot": result.get("repeatability_plot"),
                "performance_metrics_path": result.get("performance_metrics_path"),
                "error": result.get("error"),
                "canonical_plot": result.get("canonical_plot"),
                "score": result.get("score"),
                "score_components": result.get("score_components"),
            }
            for name, result in signal_results.items()
        },
    }

    top_results: dict[str, dict[str, object]] = {}
    if (not minimal_outputs) and avg_top_n and top_path:
        compare_dir = Path(out_root) / "top_avg_comparison"
        compare_dir.mkdir(parents=True, exist_ok=True)
        for metric_type, data_map in top_path.items():
            if not data_map:
                continue
            subset_dir = compare_dir / metric_type
            save_aggregated_spectra(data_map, str(subset_dir))
            canonical_subset = select_canonical_per_concentration(data_map)
            # Get per-gas ROI bounds for proper calibration
            min_wl_roi_subset, max_wl_roi_subset = _resolve_roi_bounds(dataset_label)
            response_stats_subset, avg_by_conc_subset = compute_concentration_response(
                data_map,
                override_min_wavelength=min_wl_roi_subset,
                override_max_wavelength=max_wl_roi_subset,
                top_k_candidates=top_k_candidates,
                debug_out_root=str(subset_dir),
            )
            repeatability_subset = compute_roi_repeatability(data_map, response_stats_subset)
            performance_subset = compute_roi_performance(repeatability_subset)
            metrics_path_subset = save_concentration_response_metrics(
                response_stats_subset,
                repeatability_subset,
                str(subset_dir),
                name=f"concentration_response_{metric_type}",
            )
            plot_path_subset = save_concentration_response_plot(
                response_stats_subset,
                avg_by_conc_subset,
                str(subset_dir),
                name=f"concentration_response_{metric_type}",
                clamp_to_roi=True,
            )
            full_metrics_path_subset = None
            full_plot_path_subset = None
            if scan_full:
                full_min = float(CONFIG.get("roi", {}).get("min_wavelength", 500.0))
                full_max = float(CONFIG.get("roi", {}).get("max_wavelength", 900.0))
                fs_stats_subset, fs_avg_subset = compute_concentration_response(
                    data_map,
                    override_min_wavelength=full_min,
                    override_max_wavelength=full_max,
                    top_k_candidates=top_k_candidates,
                    debug_out_root=str(subset_dir),
                )
                full_metrics_path_subset = save_concentration_response_metrics(
                    fs_stats_subset,
                    repeatability_subset,
                    str(subset_dir),
                    name=f"fullscan_concentration_response_{metric_type}",
                )
                full_plot_path_subset = save_concentration_response_plot(
                    fs_stats_subset,
                    fs_avg_subset,
                    str(subset_dir),
                    name=f"fullscan_concentration_response_{metric_type}",
                    clamp_to_roi=False,
                )
            top_results[metric_type] = {
                "canonical_count": len(canonical_subset),
                "response_stats": response_stats_subset,
                "repeatability": repeatability_subset,
                "performance": performance_subset,
                "metrics_path": metrics_path_subset,
                "plot_path": plot_path_subset,
                "fullscan_metrics_path": full_metrics_path_subset,
                "fullscan_plot_path": full_plot_path_subset,
                "canonical_raw": canonical_subset,
                "canonical": _baseline_correct_canonical(canonical_subset),
            }

    canonical_raw = select_canonical_per_concentration(stable_by_conc)
    canonical = _baseline_correct_canonical(canonical_raw)
    save_canonical_spectra(canonical, out_root)

    discovered_roi = _discover_roi_in_band(
        canonical, dataset_label=dataset_label, out_root=out_root
    )
    if discovered_roi:
        metadata["discovered_roi"] = discovered_roi
        if not minimal_outputs:
            try:
                roi_plot_path = save_roi_discovery_plot(discovered_roi, out_root)
                if roi_plot_path:
                    metadata["discovered_roi_plot"] = roi_plot_path
            except Exception:
                pass

    # Optional QC-filtered calibration using existing trial quality weights
    qc_calib: Optional[dict[str, object]] = None
    if not minimal_outputs:
        try:
            qc_cfg = calib_settings.get("qc", {}) if isinstance(calib_settings, dict) else {}
            min_q = float(qc_cfg.get("min_quality", 0.0))
            min_trials = int(qc_cfg.get("min_trials", 1))
            if min_trials < 1:
                min_trials = 1
            trial_weights_global = globals().get("TRIAL_WEIGHTS_FOR_CANONICAL", {})
            stable_qc: dict[float, dict[str, pd.DataFrame]] = {}
            if isinstance(trial_weights_global, dict):
                for conc_val, trials in stable_by_conc.items():
                    weights_map = (
                        trial_weights_global.get(conc_val, {})
                        if isinstance(trial_weights_global.get(conc_val, {}), dict)
                        else {}
                    )
                    filtered: dict[str, pd.DataFrame] = {}
                    for t_name, df in trials.items():
                        w = weights_map.get(t_name, 1.0)
                        try:
                            w_val = float(w)
                        except Exception:
                            w_val = 1.0
                        if not np.isfinite(w_val):
                            w_val = 1.0
                        if w_val >= min_q:
                            filtered[t_name] = df
                    if len(filtered) >= min_trials:
                        stable_qc[conc_val] = filtered
            if stable_qc:
                canonical_qc_raw = select_canonical_per_concentration(stable_qc)
                canonical_qc = _baseline_correct_canonical(canonical_qc_raw)
                qc_calib = find_roi_and_calibration(
                    canonical_qc,
                    dataset_label=dataset_label,
                    responsive_delta=responsive_calib,
                    discovered_roi=discovered_roi,
                )
                if isinstance(qc_calib, dict):
                    qc_calib.setdefault("variant", "qc_filtered")
                    save_calibration_outputs(qc_calib, out_root, name_suffix="_qc")
                    metadata["calibration_qc"] = qc_calib
        except Exception:
            qc_calib = None

    noise_metrics = compute_noise_metrics_map(primary_map)
    noise_metrics_path = save_noise_metrics(noise_metrics, out_root)
    summary_csv_path = save_aggregated_summary(primary_map, noise_metrics, out_root)
    # Quality control summary
    try:
        qc_summary = summarize_quality_control(primary_map, noise_metrics)
        qc_summary_path = save_quality_summary(qc_summary, out_root)
    except Exception:
        qc_summary_path = None
    if not minimal_outputs:
        aggregated_plot_paths = save_aggregated_plots(primary_map, out_root)
        canonical_plot_path = save_canonical_overlay(canonical, out_root)
    else:
        aggregated_plot_paths = {}
        canonical_plot_path = None

    ica_artifacts: dict[str, str] = {}
    mcr_artifacts: dict[str, str] = {}
    ica_result: Optional[dict[str, object]] = None
    mcr_result: Optional[dict[str, object]] = None
    if not minimal_outputs:
        try:
            ica_cfg = CONFIG.get("advanced", {}).get("deconvolution", {}).get("ica", {})
            if ica_cfg and bool(ica_cfg.get("enabled", False)):
                ica_res = fit_ica_from_canonical(canonical, ica_cfg)
                if ica_res:
                    ica_result = ica_res
                    ica_artifacts = save_ica_outputs(ica_res, out_root)
        except Exception:
            ica_artifacts = {}
        try:
            mcr_cfg = CONFIG.get("advanced", {}).get("deconvolution", {}).get("mcr_als", {})
            if mcr_cfg and bool(mcr_cfg.get("enabled", False)):
                mcr_res = fit_mcrals_from_canonical(canonical, mcr_cfg)
                if mcr_res:
                    mcr_result = mcr_res
                    mcr_artifacts = save_mcrals_outputs(mcr_res, out_root)
        except Exception:
            mcr_artifacts = {}

    # Get per-gas ROI bounds for proper calibration
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    responsive_calib = None
    if responsive_delta_summary:
        try:
            responsive_calib = perform_responsive_delta_calibration(
                responsive_delta_summary,
                out_root,
                dataset_label=dataset_label,
                trend_fallbacks=responsive_trend_fallback,
            )
            metadata["responsive_delta_calibration"] = responsive_calib
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Responsive d-lambda calibration failed: {exc}")
            responsive_calib = None

    response_stats = {}
    avg_by_conc = {}
    repeatability = {}
    performance = {}
    response_metrics_path = None
    response_plot_path = None
    repeatability_plot_path = None
    performance_metrics_path = None
    if not minimal_outputs:
        response_stats, avg_by_conc = compute_concentration_response(
            stable_by_conc,
            override_min_wavelength=min_wl_roi,
            override_max_wavelength=max_wl_roi,
            top_k_candidates=top_k_candidates,
            debug_out_root=out_root,
        )
        repeatability = compute_roi_repeatability(stable_by_conc, response_stats)
        performance = compute_roi_performance(repeatability)
        response_metrics_path = save_concentration_response_metrics(
            response_stats, repeatability, out_root, name="concentration_response"
        )
        response_plot_path = save_concentration_response_plot(
            response_stats,
            avg_by_conc,
            out_root,
            name="concentration_response",
            clamp_to_roi=True,
        )
        repeatability_plot_path = save_roi_repeatability_plot(
            stable_by_conc, response_stats, out_root
        )
        performance_metrics_path = save_roi_performance_metrics(performance, out_root)

    if not minimal_outputs:
        try:
            diagnostic_path = save_spectral_response_diagnostic(
                canonical,
                out_root,
                dataset_label=dataset_label,
                wl_min=min_wl_roi if min_wl_roi else 400.0,
                wl_max=max_wl_roi if max_wl_roi else 800.0,
            )
            if diagnostic_path:
                print(
                    f"  [PLOT] Spectral response diagnostic saved: {os.path.basename(diagnostic_path)}"
                )
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed to generate spectral diagnostic: {e}")
            sys.stdout.flush()

    abs_amp_result = None
    if not minimal_outputs:
        try:
            abs_amp_result = perform_absorbance_amplitude_calibration(
                canonical,
                out_root,
                dataset_label=dataset_label,
                wl_min=min_wl_roi if min_wl_roi else 400.0,
                wl_max=max_wl_roi if max_wl_roi else 800.0,
            )
            if abs_amp_result:
                method_name = abs_amp_result.get("enhancement_method", "raw")
                print(
                    f"  [CALIB] Enhanced dA calibration: R2={abs_amp_result['r2']:.4f} @ {abs_amp_result['best_wavelength_nm']:.1f} nm ({method_name})"
                )
                print(
                    "  [PLOT] Absorbance amplitude plot saved: absorbance_amplitude_calibration.png"
                )
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed absorbance amplitude calibration: {e}")
            sys.stdout.flush()

    env_summary_path = None
    env_info: dict[str, object] = {}
    try:
        env_info = compute_environment_summary(stable_by_conc)
        if env_info and (
            env_info.get("enabled")
            or env_info.get("offset_count", 0) > 0
            or env_info.get("temperature_mean") is not None
            or env_info.get("humidity_mean") is not None
        ):
            env_summary_path = save_environment_compensation_summary(env_info, out_root)
    except Exception:
        env_summary_path = None

    if (not minimal_outputs) and scan_full:
        full_min = float(CONFIG.get("roi", {}).get("min_wavelength", 500.0))
        full_max = float(CONFIG.get("roi", {}).get("max_wavelength", 900.0))
        fs_stats, fs_avg = compute_concentration_response(
            stable_by_conc,
            override_min_wavelength=full_min,
            override_max_wavelength=full_max,
            top_k_candidates=top_k_candidates,
            debug_out_root=out_root,
        )
        save_concentration_response_metrics(
            fs_stats, repeatability, out_root, name="fullscan_concentration_response"
        )
        save_concentration_response_plot(
            fs_stats,
            fs_avg,
            out_root,
            name="fullscan_concentration_response",
            clamp_to_roi=False,
        )

    dynamics_summary_path: Optional[str] = None
    dynamics_plot_path: Optional[str] = None
    dynamics_summary: dict[str, object] = {}
    if dynamics_enabled:
        try:
            from .dynamics import (
                compute_response_recovery_times,  # local import to avoid circular dependency
            )

            dynamics_result = compute_response_recovery_times(
                root_dir, out_root, signal_column="intensity"
            )
            # Use the summary from dynamics.py directly (it has correct T90/T10 in seconds)
            dynamics_summary = dynamics_result.get("summary", {})
            dynamics_summary_path = dynamics_result.get("json_path")
            dynamics_plot_path = dynamics_result.get("plot_path")
        except Exception as exc:
            dynamics_summary_path = save_dynamics_error(str(exc), out_root)

    calib = find_roi_and_calibration(
        canonical,
        dataset_label=dataset_label,
        responsive_delta=responsive_calib,
        discovered_roi=discovered_roi,
    )
    if responsive_calib:
        metadata.setdefault("responsive_delta", {})["calibration"] = responsive_calib

    if not minimal_outputs:
        try:
            shift_viz_path = save_wavelength_shift_visualization(
                canonical, calib, out_root, dataset_label=dataset_label
            )
            if shift_viz_path:
                print(
                    f"  [PLOT] Wavelength shift visualization saved: {os.path.basename(shift_viz_path)}"
                )
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed to generate wavelength shift visualization: {e}")
            sys.stdout.flush()

    if not minimal_outputs:
        try:
            research_plot_path = save_research_grade_calibration_plot(
                canonical, calib, out_root, dataset_label=dataset_label
            )
            if research_plot_path:
                print(
                    f"  [PLOT] Research-grade calibration plot saved: {os.path.basename(research_plot_path)}"
                )
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed to generate research-grade plot: {e}")
            sys.stdout.flush()

    if not minimal_outputs:
        try:
            if calib is not None:
                from gas_analysis.core.research_report import (
                    generate_analysis_json,
                    generate_methodology_markdown,
                )

                md_path = generate_methodology_markdown(out_root, dataset_label, calib)
                generate_analysis_json(out_root, dataset_label, calib, canonical)
                print(f"  [REPORT] Methodology report saved: {os.path.basename(md_path)}")
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed to generate methodology report: {e}")
            sys.stdout.flush()

    if not minimal_outputs:
        try:
            comparison_path = generate_method_comparison_report(
                canonical, calib, abs_amp_result, out_root, dataset_label=dataset_label
            )
            if comparison_path:
                print("  [PLOT] Method comparison report saved: method_comparison.png")
                sys.stdout.flush()
        except Exception as e:
            print(f"  [WARNING] Failed to generate method comparison: {e}")
            sys.stdout.flush()

    if not minimal_outputs:
        try:
            mv_cfg = (
                CONFIG.get("calibration", {}).get("multivariate", {})
                if isinstance(CONFIG, dict)
                else {}
            )
            sel_mode = str(mv_cfg.get("select_mode", "report_only")).lower()
            if not (mv_cfg.get("enabled", False) and sel_mode == "auto"):
                raise RuntimeError("multivariate_auto_selection_disabled")

            min_r2_cv = float(mv_cfg.get("min_r2_cv", 0.0))
            improve_margin = float(mv_cfg.get("improve_margin", 0.02))
            prefer_plsr_on_tie = bool(mv_cfg.get("prefer_plsr_on_tie", True))

            candidates = []  # list of (name, r2_cv, rmse_cv)
            plsr_r2 = float("-inf")
            if isinstance(calib, dict) and isinstance(calib.get("plsr_model", None), dict):
                pm = calib["plsr_model"]
                r2cv = pm.get("r2_cv", None)
                rmsecv = pm.get("rmse_cv", None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(
                        (
                            "plsr_cv",
                            float(r2cv),
                            float(rmsecv) if rmsecv is not None else float("nan"),
                        )
                    )
                    plsr_r2 = float(r2cv)
            if isinstance(ica_result, dict):
                r2cv = ica_result.get("r2_cv", None)
                rmsecv = ica_result.get("rmse_cv", None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(
                        (
                            "ica_cv",
                            float(r2cv),
                            float(rmsecv) if rmsecv is not None else float("nan"),
                        )
                    )
            if isinstance(mcr_result, dict):
                r2cv = mcr_result.get("r2_cv", None)
                rmsecv = mcr_result.get("rmse_cv", None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(
                        (
                            "mcr_cv",
                            float(r2cv),
                            float(rmsecv) if rmsecv is not None else float("nan"),
                        )
                    )

            if candidates:
                candidates.sort(key=lambda t: t[1], reverse=True)
                best_method, best_r2, best_rmse = candidates[0]

                if not np.isfinite(best_r2) or best_r2 < min_r2_cv:
                    best_method = None

                if best_method is not None and prefer_plsr_on_tie and np.isfinite(plsr_r2):
                    if best_method != "plsr_cv" and (best_r2 - plsr_r2) < improve_margin:
                        best_method = "plsr_cv"
                        for name, r2v, rmsev in candidates:
                            if name == "plsr_cv":
                                best_r2, best_rmse = r2v, rmsev
                                break

                if best_method is not None:
                    calib["selected_model"] = best_method
                    calib["slope"] = float("nan")
                    calib["intercept"] = float("nan")
                    calib["r2"] = float(best_r2)
                    calib["rmse"] = float(best_rmse)
                    unc = (
                        calib.get("uncertainty", {})
                        if isinstance(calib.get("uncertainty", {}), dict)
                        else {}
                    )
                    unc["r2_cv"] = float(best_r2)
                    unc["rmse_cv"] = float(best_rmse)
                    calib["uncertainty"] = unc
        except Exception:
            pass

    save_calibration_outputs(calib, out_root)

    fusion_cfg = (
        CONFIG.get("calibration", {}).get("multi_roi_fusion", {})
        if isinstance(CONFIG, dict)
        else {}
    )
    if (not minimal_outputs) and bool(fusion_cfg.get("enabled", False)):
        max_feats = int(fusion_cfg.get("max_features", 4) or 4)
        fusion_result = _compute_multi_roi_fusion_calibration(
            discovered_roi,
            calib,
            out_root,
            dataset_label=dataset_label,
            max_features=max_feats,
        )
        if fusion_result:
            metadata["multi_roi_fusion"] = fusion_result

    # Compute T90/T10 response and recovery times from time-series data
    try:
        from .dynamics import compute_t90_t10_from_timeseries

        time_series_dir = os.path.join(out_root, "time_series")
        if os.path.isdir(time_series_dir):
            dynamics_cfg = CONFIG.get("dynamics", {}) if isinstance(CONFIG, dict) else {}
            baseline_frames = int(dynamics_cfg.get("baseline_frames", 20))
            frame_rate_cfg = dynamics_cfg.get("frame_rate", None)
            min_amp_cfg = dynamics_cfg.get("min_response_amplitude_nm", 0.0)
            smooth_cfg = dynamics_cfg.get("timeseries_smoothing_window", 1)
            try:
                frame_rate_val = float(frame_rate_cfg) if frame_rate_cfg is not None else None
                if frame_rate_val is not None and frame_rate_val <= 0:
                    frame_rate_val = None
            except Exception:
                frame_rate_val = None
            try:
                min_amp_val = float(min_amp_cfg) if min_amp_cfg is not None else 0.0
            except Exception:
                min_amp_val = 0.0
            try:
                smooth_val = int(smooth_cfg) if smooth_cfg is not None else 1
            except Exception:
                smooth_val = 1
            if smooth_val < 1:
                smooth_val = 1

            dynamics_result = compute_t90_t10_from_timeseries(
                time_series_dir=time_series_dir,
                out_root=out_root,
                baseline_frames=baseline_frames,
                steady_state_frames=20,
                frame_rate=frame_rate_val,
                min_response_amplitude_nm=min_amp_val,
                smooth_window=smooth_val,
            )
            if dynamics_result and dynamics_result.get("summary"):
                # Update dynamics_summary with the correct data from timeseries analysis
                dynamics_summary = dynamics_result.get("summary", {})
                dynamics_summary_path = dynamics_result.get("json_path")
                print("[INFO] T90/T10 dynamics computed successfully")
        else:
            print(f"[WARNING] Time-series directory not found: {time_series_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to compute T90/T10 dynamics: {e}")

    if not minimal_outputs:
        try:
            env_coeffs = compute_environment_coefficients(stable_by_conc, calib)
            if env_coeffs:
                env_info.update(env_coeffs)
                env_summary_path = save_environment_compensation_summary(env_info, out_root)
                try:
                    est = (
                        env_coeffs.get("estimated_coefficients", {})
                        if isinstance(env_coeffs.get("estimated_coefficients", {}), dict)
                        else {}
                    )
                    if est and isinstance(CONFIG, dict):
                        env_cfg = (
                            CONFIG.get("environment", {})
                            if isinstance(CONFIG.get("environment", {}), dict)
                            else {}
                        )
                        if env_cfg.get("autosave_coefficients", False):
                            _autosave_env_coefficients_to_config(est)
                except Exception:
                    pass
        except Exception:
            pass

    if time_series_outputs:
        metadata["time_series_outputs"] = time_series_outputs

    if not minimal_outputs:
        report_results = _invoke_report_generation(out_root, metadata)
        trend_plots = generate_trend_plots(out_root)
    else:
        report_results = {}
        trend_plots = {}

    archive_dir = Path(out_root) / "archives" / run_timestamp

    multi_select_path = None
    multi_cv_plot_path = None
    if not minimal_outputs:
        try:
            plsr_r2cv = None
            plsr_rmsecv = None
            if isinstance(calib, dict) and isinstance(calib.get("plsr_model", None), dict):
                pm = calib["plsr_model"]
                plsr_r2cv = float(pm.get("r2_cv", float("nan")))
                plsr_rmsecv = float(pm.get("rmse_cv", float("nan")))

            ica_r2cv = (
                float(ica_result.get("r2_cv"))
                if isinstance(ica_result, dict) and ica_result.get("r2_cv") is not None
                else None
            )
            ica_rmsecv = (
                float(ica_result.get("rmse_cv"))
                if isinstance(ica_result, dict) and ica_result.get("rmse_cv") is not None
                else None
            )
            mcr_r2cv = (
                float(mcr_result.get("r2_cv"))
                if isinstance(mcr_result, dict) and mcr_result.get("r2_cv") is not None
                else None
            )
            mcr_rmsecv = (
                float(mcr_result.get("rmse_cv"))
                if isinstance(mcr_result, dict) and mcr_result.get("rmse_cv") is not None
                else None
            )

            scores = {
                "plsr": {
                    "r2_cv": plsr_r2cv,
                    "rmse_cv": plsr_rmsecv,
                    "metrics": os.path.join(out_root, "metrics", "calibration_metrics.json"),
                },
                "ica": {
                    "r2_cv": ica_r2cv,
                    "rmse_cv": ica_rmsecv,
                    "metrics": ica_artifacts.get("metrics")
                    if isinstance(ica_artifacts, dict)
                    else None,
                },
                "mcr_als": {
                    "r2_cv": mcr_r2cv,
                    "rmse_cv": mcr_rmsecv,
                    "metrics": mcr_artifacts.get("metrics")
                    if isinstance(mcr_artifacts, dict)
                    else None,
                },
            }
            best_method = None
            best_r2 = float("-inf")
            for method, sc in scores.items():
                r2v = sc.get("r2_cv")
                if r2v is not None and np.isfinite(r2v) and r2v > best_r2:
                    best_r2 = float(r2v)
                    best_method = method
            multivar = {
                "scores": scores,
                "best_method": best_method,
                "best_r2_cv": best_r2,
            }
            metrics_dir = os.path.join(out_root, "metrics")
            _ensure_dir(metrics_dir)
            multi_select_path = os.path.join(metrics_dir, "multivariate_selection.json")
            with open(multi_select_path, "w") as f:
                json.dump(multivar, f, indent=2)

            try:
                labels = []
                values = []
                for lbl in ["plsr", "ica", "mcr_als"]:
                    r2v = scores.get(lbl, {}).get("r2_cv", None)
                    if r2v is not None and np.isfinite(r2v):
                        labels.append(lbl.upper())
                        values.append(float(r2v))
                if labels:
                    plots_dir = os.path.join(out_root, "plots")
                    _ensure_dir(plots_dir)
                    plt.figure(figsize=(5, 3))
                    xpos = np.arange(len(labels))
                    plt.bar(
                        xpos,
                        values,
                        color=[
                            "#4F81BD" if l != (best_method or "").upper() else "#9BBB59"
                            for l in labels
                        ],
                    )
                    plt.xticks(xpos, labels)
                    plt.ylabel("CV R²")
                    plt.ylim(0.0, 1.0)
                    plt.title("Multivariate CV R² Comparison")
                    plt.tight_layout()
                    multi_cv_plot_path = os.path.join(plots_dir, "multivariate_cv_r2.png")
                    plt.savefig(multi_cv_plot_path, dpi=200)
                    plt.close()
            except Exception:
                multi_cv_plot_path = None
        except Exception:
            multi_select_path = None
            multi_cv_plot_path = None

    # Reproducibility metadata
    try:
        versions = {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": sp.__version__,
            "sklearn": sk.__version__,
            "matplotlib": matplotlib.__version__,
            "pyyaml": yaml.__version__,
        }
    except Exception:
        versions = {"python": sys.version}
    git_commit = None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=True,
        )
        git_commit = proc.stdout.decode("utf-8", errors="ignore").strip() or None
    except Exception:
        git_commit = None

    metadata.update(
        {
            "calibration": calib,
            "roi_response": response_stats,
            "roi_repeatability": repeatability,
            "archiving": CONFIG.get("archiving", {}),
            "reporting": CONFIG.get("reporting", {}),
            "dynamics_config": dynamics_cfg,
            "roi_config": CONFIG.get("roi", {}),
            "outputs": {
                "noise_metrics": noise_metrics_path,
                "aggregated_summary": summary_csv_path,
                "quality_control": qc_summary_path,
                "environment_compensation": env_summary_path,
                "deconvolution_ica": ica_artifacts.get("metrics")
                if isinstance(ica_artifacts, dict)
                else None,
                "deconvolution_mcr_als": mcr_artifacts.get("metrics")
                if isinstance(mcr_artifacts, dict)
                else None,
                "multivariate_selection": multi_select_path,
                "multivariate_cv_plot": multi_cv_plot_path,
                "selected_predictions_csv": os.path.join(
                    out_root, "metrics", "selected_predictions.csv"
                ),
                "selected_pred_vs_actual_plot": os.path.join(
                    out_root, "plots", "selected_pred_vs_actual.png"
                ),
                "concentration_response_metrics": response_metrics_path,
                "roi_performance_metrics": performance_metrics_path,
                "dynamics_summary": dynamics_summary_path,
                "canonical_plot": canonical_plot_path,
                "concentration_response_plot": response_plot_path,
                "roi_repeatability_plot": repeatability_plot_path,
                "dynamics_plot": dynamics_plot_path,
            },
            "reports": report_results,
            "trend_plots": trend_plots,
            "archive_path": str(archive_dir),
            "aggregated_plot_paths": aggregated_plot_paths,
            "versions": versions,
            "git_commit": git_commit,
        }
    )

    metadata_path: Optional[Path] = None
    try:
        metadata_path = _save_run_metadata(out_root, metadata)
    except Exception as exc:  # noqa: BLE001
        metadata.setdefault("errors", []).append({"stage": "save_run_metadata", "error": str(exc)})

    archive_path = None
    try:
        archive_path = _archive_run(out_root, metadata)
    except Exception as exc:  # noqa: BLE001
        metadata.setdefault("errors", []).append({"stage": "archive_run", "error": str(exc)})
        archive_path = None
    if archive_path is not None:
        metadata["archive_created"] = str(archive_path)
        try:
            metadata_path = _save_run_metadata(out_root, metadata)
        except Exception as exc:  # noqa: BLE001
            metadata.setdefault("errors", []).append(
                {"stage": "save_run_metadata_post_archive", "error": str(exc)}
            )

    if minimal_outputs:
        return {
            "mapping": mapping,
            "aggregated_files": aggregated_paths,
            "canonical_count": len(canonical),
            "calibration": calib,
            "run_metadata": metadata_path,
            "archive_path": str(archive_path) if archive_path else None,
        }

    report_path = None
    try:
        report_path = write_run_summary(
            calib,
            aggregated_paths,
            noise_metrics_path,
            summary_csv_path,
            canonical_plot_path,
            response_metrics_path,
            response_plot_path,
            repeatability_plot_path,
            performance_metrics_path,
            dynamics_summary_path,
            dynamics_plot_path,
            metadata_path,
            str(archive_path) if archive_path else None,
            qc_summary_path,
            report_results,
            trend_plots,
            performance,
            dynamics_summary,
            out_root,
        )
    except Exception as exc:  # noqa: BLE001
        metadata.setdefault("errors", []).append({"stage": "write_run_summary", "error": str(exc)})
        with contextlib.suppress(Exception):
            metadata_path = _save_run_metadata(out_root, metadata)

    return {
        "mapping": mapping,
        "aggregated_files": aggregated_paths,
        "aggregated_plots": aggregated_plot_paths,
        "canonical_count": len(canonical),
        "calibration": calib,
        "noise_metrics_path": noise_metrics_path,
        "aggregated_summary_csv": summary_csv_path,
        "canonical_plot": canonical_plot_path,
        "concentration_response_metrics": response_metrics_path,
        "concentration_response_plot": response_plot_path,
        "roi_repeatability_plot": repeatability_plot_path,
        "roi_performance_metrics": performance_metrics_path,
        "dynamics_summary": dynamics_summary_path,
        "dynamics_plot": dynamics_plot_path,
        "environment_compensation": env_summary_path,
        "deconvolution_ica": ica_artifacts,
        "deconvolution_mcr_als": mcr_artifacts,
        "multivariate_selection": multi_select_path,
        "run_metadata": metadata_path,
        "archive_path": str(archive_path) if archive_path else None,
        "report_artifacts": report_results,
        "top_avg_results": top_results,
        "top_avg_summary": summarize_top_comparison(top_results) if top_results else [],
        "trend_plots": trend_plots,
        "report": report_path,
    }
