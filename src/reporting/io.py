"""Reporting I/O — pure JSON/CSV serialisers for pipeline analysis outputs.

All functions are CONFIG-free: every parameter is passed explicitly.
They take a structured ``out_root`` directory and write one or more
files beneath it, returning the path(s) for downstream use.

Typical directory layout written by this module::

    out_root/
      metrics/
        noise_metrics.json
        qc_summary.json
        aggregated_summary.csv
        roi_performance.json
        dynamics_summary.json
        concentration_response.json
        environment_compensation.json
      stable_selected/
        0.5_stable.csv
        1.0_stable.csv
      aggregated/
        0.5/
          trial_0.csv
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.reporting.metrics import select_signal_column

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _json_safe(obj: object) -> object:
    """Recursively convert numpy scalars / arrays to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    return obj


def _write_json(obj: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(obj), fh, indent=2)


# ---------------------------------------------------------------------------
# Public savers
# ---------------------------------------------------------------------------


def save_canonical_spectra(
    canonical: dict[float, pd.DataFrame],
    out_root: str,
) -> list[str]:
    """Save the canonical (stable-block representative) spectrum for each concentration.

    Args:
        canonical: Mapping ``{concentration_ppm: DataFrame}`` of representative spectra.
        out_root: Root output directory.

    Returns:
        List of file paths written (one CSV per concentration).
    """
    out_dir = os.path.join(out_root, "stable_selected")
    _ensure_dir(out_dir)
    paths: list[str] = []
    for conc, df in canonical.items():
        fname = f"{conc:g}_stable.csv"
        fpath = os.path.join(out_dir, fname)
        df.to_csv(fpath, index=False)
        paths.append(fpath)
    return paths


def save_aggregated_spectra(
    aggregated: dict[float, dict[str, pd.DataFrame]],
    out_root: str,
) -> dict[float, dict[str, str]]:
    """Save every trial spectrum from the aggregated dataset.

    Each concentration gets its own sub-directory under ``aggregated/``.
    Stale CSVs in each sub-directory are removed before writing.

    Args:
        aggregated: Mapping ``{concentration: {trial_name: DataFrame}}``.
        out_root: Root output directory.

    Returns:
        Nested dict mirroring *aggregated* with file paths as leaf values.
    """
    base_dir = os.path.join(out_root, "aggregated")
    _ensure_dir(base_dir)
    saved: dict[float, dict[str, str]] = {}
    for conc, trials in aggregated.items():
        conc_dir = os.path.join(base_dir, f"{conc:g}")
        _ensure_dir(conc_dir)
        for old in os.listdir(conc_dir):
            if old.lower().endswith(".csv"):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(conc_dir, old))
        saved[conc] = {}
        for trial, df in trials.items():
            safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
            fname = f"{safe_trial or 'trial'}.csv"
            fpath = os.path.join(conc_dir, fname)
            df.to_csv(fpath, index=False)
            saved[conc][trial] = fpath
    return saved


def save_noise_metrics(
    metrics: dict[float, dict[str, Any]],
    out_root: str,
) -> str:
    """Serialise per-trial noise metrics to JSON.

    Args:
        metrics: Mapping ``{concentration: {trial_name: noise_dict}}``.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file.
    """
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    serializable = {str(conc): trials for conc, trials in metrics.items()}
    out_path = os.path.join(metrics_dir, "noise_metrics.json")
    _write_json(serializable, out_path)
    return out_path


def save_quality_summary(
    qc: dict[str, Any],
    out_root: str,
) -> str:
    """Serialise the QC summary dict to JSON.

    Args:
        qc: Output from :func:`src.reporting.metrics.summarize_quality_control`.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file.
    """
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, "qc_summary.json")
    _write_json(qc, out_path)
    return out_path


def save_aggregated_summary(
    aggregated: dict[float, dict[str, pd.DataFrame]],
    noise_metrics: dict[float, dict[str, Any]],
    out_root: str,
) -> str:
    """Build and save a per-trial summary CSV with signal statistics and noise metrics.

    Args:
        aggregated: Mapping ``{concentration: {trial_name: DataFrame}}``.
        noise_metrics: Output from :func:`src.reporting.metrics.compute_noise_metrics_map`.
        out_root: Root output directory.

    Returns:
        Path to the written CSV file.
    """
    rows = []
    for conc, trials in aggregated.items():
        for trial, df in trials.items():
            col = select_signal_column(df)
            arr = df[col].values
            nm = noise_metrics.get(conc, {}).get(trial, {})
            rows.append(
                {
                    "concentration": conc,
                    "trial": trial,
                    "signal_column": col,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "rms": nm.get("rms", np.nan),
                    "mad": nm.get("mad", np.nan),
                    "spectral_entropy": nm.get("spectral_entropy", np.nan),
                    "snr": nm.get("snr", np.nan),
                }
            )
    summary = pd.DataFrame(rows)
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, "aggregated_summary.csv")
    summary.to_csv(out_path, index=False)
    return out_path


def save_roi_performance_metrics(
    performance: dict[str, Any],
    out_root: str,
) -> Optional[str]:
    """Serialise ROI performance metrics (LOD, LOQ, regression) to JSON.

    Returns ``None`` when *performance* is empty.

    Args:
        performance: Output from :func:`src.reporting.metrics.compute_roi_performance`.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file, or ``None`` if *performance* is empty.
    """
    if not performance:
        return None
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, "roi_performance.json")
    _write_json(performance, out_path)
    return out_path


def save_dynamics_summary(
    summary: dict[str, Any],
    out_root: str,
) -> str:
    """Serialise response/recovery dynamics summary to JSON.

    Args:
        summary: Output from :func:`src.reporting.metrics.summarize_dynamics_metrics`.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file.
    """
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, "dynamics_summary.json")
    _write_json(summary, out_path)
    return out_path


def save_dynamics_error(message: str, out_root: str) -> str:
    """Save a dynamics error placeholder when the dynamics step fails.

    Args:
        message: Error description.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file.
    """
    return save_dynamics_summary({"error": message}, out_root)


def save_concentration_response_metrics(
    response: dict[str, Any],
    repeatability: dict[str, Any],
    out_root: str,
    name: str = "concentration_response",
) -> str:
    """Serialise the concentration-response dict (with embedded repeatability) to JSON.

    Args:
        response: Output from :func:`src.calibration.roi_scan.compute_concentration_response`.
        repeatability: Output from :func:`src.reporting.metrics.compute_roi_repeatability`.
        out_root: Root output directory.
        name: Base filename (without extension).

    Returns:
        Path to the written JSON file.
    """
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, f"{name}.json")
    payload = dict(response)
    payload["roi_repeatability"] = repeatability
    _write_json(payload, out_path)
    return out_path


def save_environment_compensation_summary(
    info: dict[str, Any],
    out_root: str,
) -> str:
    """Serialise the environment compensation summary to JSON.

    Args:
        info: Output from :func:`src.reporting.environment.compute_environment_summary`.
        out_root: Root output directory.

    Returns:
        Path to the written JSON file.
    """
    metrics_dir = os.path.join(out_root, "metrics")
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, "environment_compensation.json")
    _write_json(info, out_path)
    return out_path
