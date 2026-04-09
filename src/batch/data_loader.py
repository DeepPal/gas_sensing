"""
src.batch.data_loader
=====================
Utilities for discovering and loading raw spectrum CSV files from the Joy_Data
directory layout used by the the sensor LSPR experiment.

Expected directory layout
--------------------------
::

    data/raw/<GasType>/
        <concentration_label>/        e.g. "0.5 ppm EtOH IPA MeOH-1"
            frame_001.csv             headerless: col0=wavelength, col1=intensity
            ...

This module is intentionally dependency-light — only stdlib + numpy + pandas —
so it can be imported without torch, FastAPI, or MLflow.

Public API
----------
- ``read_spectrum_csv(path)``     → pd.DataFrame with wavelength/intensity columns
- ``sort_frame_paths(paths)``     → chronologically sorted list of paths
- ``scan_experiment_root(root)``  → concentration → trial → list[path] mapping
- ``ExperimentScan``              → typed result dataclass
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re

import pandas as pd

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ExperimentScan:
    """Result of scanning an experiment root directory.

    ``trials`` maps ``concentration_ppm → trial_label → list_of_csv_paths``.
    Paths within each trial are chronologically sorted.
    """

    root: Path
    gas_type: str
    trials: dict[float, dict[str, list[str]]] = field(default_factory=dict)

    @property
    def concentrations(self) -> list[float]:
        """Sorted list of concentration values found."""
        return sorted(self.trials.keys())

    @property
    def total_frames(self) -> int:
        """Total number of CSV files across all concentrations and trials."""
        return sum(
            len(paths) for conc_trials in self.trials.values() for paths in conc_trials.values()
        )

    def frames_for(self, concentration: float) -> list[str]:
        """All frame paths for *concentration*, sorted, across all trials."""
        paths: list[str] = []
        for trial_paths in self.trials.get(concentration, {}).values():
            paths.extend(trial_paths)
        return sort_frame_paths(paths)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

_COLUMN_ALIASES: dict[str, str] = {
    "wl": "wavelength",
    "wavelength_nm": "wavelength",
    "lambda": "wavelength",
    "nm": "wavelength",
    "intensity": "intensity",
    "counts": "intensity",
    "signal": "intensity",
    "int": "intensity",
}


def read_spectrum_csv(path: str | Path) -> pd.DataFrame:
    """Read a single spectrum CSV file.

    Handles both headered and headerless (two-column) files.  Returns a
    DataFrame with at minimum ``wavelength`` and ``intensity`` columns,
    sorted by wavelength ascending.

    Raises
    ------
    RuntimeError
        If the file cannot be read or does not contain usable numeric columns.
    """
    path = str(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Cannot read CSV {path}: {exc}") from exc

    # Normalise column names via aliases
    rename: dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    if rename:
        df = df.rename(columns=rename)

    # Fall back: treat two-column headerless file
    if "wavelength" not in df.columns or "intensity" not in df.columns:
        try:
            df = pd.read_csv(path, header=None, names=["wavelength", "intensity"])
        except Exception as exc:
            raise RuntimeError(f"Cannot parse headerless CSV {path}: {exc}") from exc

    required = {"wavelength", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV {path} missing columns {missing}. Found: {list(df.columns)}")

    df = df[list(required | (set(df.columns) - required))].copy()
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna(subset=["wavelength", "intensity"])

    if df.empty:
        raise RuntimeError(f"No valid numeric rows in {path}")

    return df.sort_values("wavelength").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Path sorting
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{8})_(?P<hour>\d{1,2})h(?P<minute>\d{1,2})m"
    r"(?P<second>\d{1,2})s(?P<msec>\d{1,3})ms"
)


def _sort_key(path: str) -> tuple[int, float, float]:
    """Return a (date_int, time_float, mtime) sort key for a frame path."""
    name = os.path.basename(path)

    # Prefer explicit timestamp: _20250605_10h26m29s408ms
    m = _TIMESTAMP_RE.search(name)
    if m:
        try:
            date_int = int(m.group("date"))
            time_secs = (
                int(m.group("hour")) * 3600 + int(m.group("minute")) * 60 + int(m.group("second"))
            )
            time_ms = int(m.group("msec"))
            time_float = time_secs * 1000.0 + time_ms
            mtime = os.path.getmtime(path)
            return date_int, time_float, mtime
        except (ValueError, OSError):
            pass

    # Fallback: extract trailing numeric token
    digits = re.findall(r"(\d+)", name)
    idx: float = float(digits[-1]) if digits else math.inf
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return int(idx) if math.isfinite(idx) else int(1e9), idx, mtime


def sort_frame_paths(paths: Sequence[str]) -> list[str]:
    """Return *paths* sorted chronologically using embedded timestamps."""
    return sorted(paths, key=_sort_key)


# ---------------------------------------------------------------------------
# Experiment root scanner
# ---------------------------------------------------------------------------

_CONC_PATTERNS: list[str] = [
    r"(\d+(?:\.\d+)?)\s*ppm",
    r"(\d+(?:\.\d+)?)\s*ppb",
    r"(\d+(?:\.\d+)?)\s*%",
    r"conc[_-]?(\d+(?:\.\d+)?)",
    r"(\d+(?:\.\d+)?)",
]


def _extract_concentration(name: str) -> float | None:
    """Return the first parseable concentration value from a directory name."""
    lower = name.lower()
    for pattern in _CONC_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                continue
    return None


def _unique_trial_key(existing: dict[str, list[str]], base: str) -> str:
    """Return *base* if not in *existing*, else *base_2*, *base_3*, etc."""
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def scan_experiment_root(
    root_dir: str | Path,
    gas_type: str = "unknown",
) -> ExperimentScan:
    """Scan an experiment root directory and return an :class:`ExperimentScan`.

    Supports two layouts:

    1. **Flat** — CSV files directly inside the concentration directory::

         root/0.5 ppm Ethanol/frame_001.csv

    2. **Nested** — trial subdirectories::

         root/0.5 ppm Ethanol/trial-1/frame_001.csv

    Parameters
    ----------
    root_dir:
        Path to the experiment root (the directory containing concentration
        subdirectories).
    gas_type:
        Label for the gas analyte being scanned; stored in the returned
        :class:`ExperimentScan` but not used for filtering.

    Returns
    -------
    ExperimentScan
        A typed container mapping ``concentration → trial → sorted frame paths``.

    Raises
    ------
    ValueError
        If *root_dir* does not exist or contains no CSV files.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root_dir}")

    trials: dict[float, dict[str, list[str]]] = {}

    for conc_entry in sorted(root.iterdir()):
        if not conc_entry.is_dir():
            continue

        conc_val = _extract_concentration(conc_entry.name)
        if conc_val is None:
            conc_val = 0.0

        conc_trials = trials.setdefault(conc_val, {})

        # Layout 1: CSV files directly in concentration directory
        direct_csvs = [str(p) for p in sorted(conc_entry.iterdir()) if p.suffix.lower() == ".csv"]
        if direct_csvs:
            key = _unique_trial_key(conc_trials, conc_entry.name)
            conc_trials[key] = sort_frame_paths(direct_csvs)

        # Layout 2: trial subdirectories
        for trial_entry in sorted(conc_entry.iterdir()):
            if not trial_entry.is_dir():
                continue
            csvs = [str(p) for p in sorted(trial_entry.iterdir()) if p.suffix.lower() == ".csv"]
            if csvs:
                key = _unique_trial_key(conc_trials, f"{conc_entry.name}/{trial_entry.name}")
                conc_trials[key] = sort_frame_paths(csvs)

    if not any(trials.values()):
        raise ValueError(f"No CSV files found under {root_dir}")

    return ExperimentScan(root=root, gas_type=gas_type, trials=trials)


# ---------------------------------------------------------------------------
# Helpers for batch loaders
# ---------------------------------------------------------------------------


def load_frames(
    paths: Sequence[str],
    *,
    max_frames: int | None = None,
) -> list[pd.DataFrame]:
    """Load a list of CSV paths into DataFrames, skipping unreadable files.

    Parameters
    ----------
    paths:
        Chronologically sorted list of CSV file paths.
    max_frames:
        If given, only load the last *max_frames* files (temporal tail gating).

    Returns
    -------
    List[pd.DataFrame]
        Successfully loaded frames; empty list if none could be read.
    """
    if max_frames is not None and max_frames > 0:
        paths = list(paths)[-max_frames:]

    import contextlib

    frames: list[pd.DataFrame] = []
    for p in paths:
        with contextlib.suppress(RuntimeError):
            frames.append(read_spectrum_csv(p))
    return frames


def load_last_n_frames(paths: Sequence[str], n: int = 10) -> list[pd.DataFrame]:
    """Convenience wrapper: load the last *n* frames (plateau / steady-state)."""
    return load_frames(paths, max_frames=n)
