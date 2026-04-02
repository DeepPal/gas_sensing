"""Universal spectral dataset loader.

Loads ANY tabular spectral dataset — regardless of sensor type, wavelength range,
signal type, or laboratory format — into a standardised ``SpectralDataset``.

Supported input formats
-----------------------
- **Long-format spectrum CSV**: one row per wavelength point, columns include a
  wavelength column and one or more signal columns (intensity, transmittance,
  absorbance, reflectance, ...).  Shape: (N_pixels, ≥2).
  Example: CCS200 output ``[wavelength, intensity, transmittance, absorbance]``.

- **Wide-format spectrum CSV**: one row per measurement frame, columns are
  wavelength values (numeric headers).  Shape: (N_frames, N_pixels).

- **Time-series feature CSV**: one row per frame, columns are computed features
  (delta_lambda_nm, peak_wavelength_nm, ...).  Detected by frame_index column.

- **HDF5**: dataset named ``spectra`` + optional ``wavelengths`` + ``metadata``.

Directory loading
-----------------
Pass a directory path to load all matching CSVs and stack them into a single
dataset matrix.  Metadata (analyte, concentration) are inferred from the
directory structure and filenames automatically.

Usage
-----
::

    from src.io.universal_loader import load_dataset, SpectralDataset

    # Single file
    ds = load_dataset("output/batch/Ethanol/stable_selected/1_stable.csv")

    # Whole directory (all concentrations)
    ds = load_dataset("output/batch/Ethanol/stable_selected",
                      normalisation="snv")

    # From a nested directory: analyte → concentration → csv files
    ds = load_dataset("output/batch/Ethanol",
                      signal_type="intensity",
                      normalisation="area")

    print(ds.spectra.shape)      # (N_samples, N_wavelengths)
    print(ds.wavelengths[:5])    # nm axis
    print(ds.labels)             # concentration values (float) or class strings
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

SignalType = Literal["intensity", "transmittance", "absorbance", "reflectance", "auto"]
Normalisation = Literal["snv", "msc", "area", "minmax", "none"]

_SIGNAL_PRIORITY = ["intensity", "transmittance", "absorbance", "reflectance"]


@dataclass
class SpectralDataset:
    """Standardised container for a multi-sample spectral dataset.

    Attributes
    ----------
    wavelengths : ndarray, shape (N_wl,)
        Wavelength axis in nm.
    spectra : ndarray, shape (N_samples, N_wl)
        Signal matrix after optional normalisation.  Rows = samples.
    signal_type : str
        Which signal column was extracted (e.g. 'intensity').
    normalisation : str
        Normalisation applied ('snv', 'area', 'none', …).
    labels : ndarray or None, shape (N_samples,)
        Numeric concentration (ppm) or string class label per sample.
    label_unit : str
        Unit of the numeric label (default 'ppm').
    analyte : str or None
        Analyte name inferred from directory structure.
    config_id : str or None
        Sensor configuration identifier (inferred from path or set manually).
    metadata : list[dict]
        Per-sample metadata dict (source_file, analyte, concentration, …).
    source_paths : list[str]
        Absolute file paths of all loaded files.
    """

    wavelengths: np.ndarray
    spectra: np.ndarray
    signal_type: str
    normalisation: str
    labels: np.ndarray | None = None
    label_unit: str = "ppm"
    analyte: str | None = None
    config_id: str | None = None
    metadata: list[dict] = field(default_factory=list)
    source_paths: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        return int(self.spectra.shape[0])

    @property
    def n_wavelengths(self) -> int:
        return int(self.spectra.shape[1])

    @property
    def wl_range(self) -> tuple[float, float]:
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    def subset(self, mask: np.ndarray) -> "SpectralDataset":
        """Return a new dataset containing only rows where mask is True."""
        idx = np.where(mask)[0]
        return SpectralDataset(
            wavelengths=self.wavelengths,
            spectra=self.spectra[idx],
            signal_type=self.signal_type,
            normalisation=self.normalisation,
            labels=self.labels[idx] if self.labels is not None else None,
            label_unit=self.label_unit,
            analyte=self.analyte,
            config_id=self.config_id,
            metadata=[self.metadata[i] for i in idx] if self.metadata else [],
            source_paths=[self.source_paths[i] for i in idx] if self.source_paths else [],
        )

    def split(self, test_fraction: float = 0.2, seed: int = 42
              ) -> tuple["SpectralDataset", "SpectralDataset"]:
        """Random train/test split."""
        rng = np.random.default_rng(seed)
        n = self.n_samples
        idx = rng.permutation(n)
        n_test = max(1, int(n * test_fraction))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_mask = np.zeros(n, dtype=bool)
        train_mask[train_idx] = True
        return self.subset(train_mask), self.subset(~train_mask)

    def __repr__(self) -> str:
        return (
            f"SpectralDataset(n_samples={self.n_samples}, "
            f"n_wl={self.n_wavelengths}, "
            f"wl=[{self.wl_range[0]:.1f}–{self.wl_range[1]:.1f} nm], "
            f"signal='{self.signal_type}', "
            f"norm='{self.normalisation}', "
            f"analyte={self.analyte!r})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_dataset(
    path: str | Path,
    signal_type: SignalType = "auto",
    normalisation: Normalisation = "none",
    pattern: str = "*.csv",
    reference_wavelengths: np.ndarray | None = None,
    analyte: str | None = None,
    config_id: str | None = None,
    recursive: bool = True,
) -> SpectralDataset:
    """Load a spectral dataset from a file or directory.

    Parameters
    ----------
    path :
        Path to a CSV/HDF5 file, or a directory containing CSV files.
    signal_type :
        Which signal column to extract.  ``'auto'`` picks the first available
        from: intensity → transmittance → absorbance → reflectance.
    normalisation :
        Preprocessing to apply after loading:
        - ``'snv'``: Standard Normal Variate (subtract mean, divide by std)
        - ``'msc'``: Multiplicative Scatter Correction (correct for scatter)
        - ``'area'``: Divide each spectrum by its area (trapezoid integral)
        - ``'minmax'``: Scale each spectrum to [0, 1]
        - ``'none'``: No preprocessing
    pattern :
        Glob pattern when loading a directory (default ``'*.csv'``).
    reference_wavelengths :
        If provided, all spectra are interpolated to this common grid.
        Otherwise the wavelength axis from the first loaded file is used.
    analyte :
        Override the analyte name (otherwise inferred from path).
    config_id :
        Optional sensor configuration identifier.
    recursive :
        If True, search subdirectories recursively for the pattern.

    Returns
    -------
    SpectralDataset
    """
    path = Path(path)

    if path.is_file():
        return _load_single_file(
            path, signal_type, normalisation,
            reference_wavelengths, analyte, config_id,
        )

    if path.is_dir():
        return _load_directory(
            path, signal_type, normalisation, pattern,
            reference_wavelengths, analyte, config_id, recursive,
        )

    raise FileNotFoundError(f"Path does not exist: {path}")


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------

def _load_directory(
    directory: Path,
    signal_type: SignalType,
    normalisation: Normalisation,
    pattern: str,
    reference_wavelengths: np.ndarray | None,
    analyte: str | None,
    config_id: str | None,
    recursive: bool,
) -> SpectralDataset:
    """Load all matching CSVs in a directory tree, infer labels from paths."""
    glob_fn = directory.rglob if recursive else directory.glob
    csv_files = sorted(glob_fn(pattern))

    # Filter out non-spectrum files (skip time-series feature CSVs)
    spectrum_files = [f for f in csv_files if _is_spectrum_file(f)]

    if not spectrum_files:
        raise ValueError(
            f"No spectrum CSV files found in {directory} "
            f"(pattern={pattern!r}). "
            "If all files are time-series feature files, use "
            "load_timeseries_features() instead."
        )

    # Infer analyte from directory name if not provided
    if analyte is None:
        analyte = _infer_analyte(directory)

    # Load all files and build common wavelength grid
    raw_spectra: list[np.ndarray] = []
    raw_wavelengths: list[np.ndarray] = []
    labels: list[float | str | None] = []
    metadata: list[dict] = []
    source_paths: list[str] = []
    signal_types_seen: list[str] = []

    for fpath in spectrum_files:
        try:
            wl, sig, stype, meta = _read_spectrum_csv(fpath, signal_type)
        except Exception as exc:
            warnings.warn(f"Skipping {fpath.name}: {exc}")
            continue

        conc = _infer_concentration(fpath)
        raw_spectra.append(sig)
        raw_wavelengths.append(wl)
        labels.append(conc)
        meta.update({"source_file": fpath.name, "analyte": analyte,
                     "concentration_ppm": conc})
        metadata.append(meta)
        source_paths.append(str(fpath))
        signal_types_seen.append(stype)

    if not raw_spectra:
        raise ValueError(f"Failed to load any spectra from {directory}")

    resolved_signal_type = signal_types_seen[0]

    # Build common wavelength grid
    if reference_wavelengths is not None:
        common_wl = reference_wavelengths
    else:
        # Use the wavelength axis from the file with the most points
        common_wl = raw_wavelengths[int(np.argmax([len(w) for w in raw_wavelengths]))]

    # Interpolate all spectra onto common grid
    spectra_matrix = np.stack([
        np.interp(common_wl, wl, sig)
        for wl, sig in zip(raw_wavelengths, raw_spectra)
    ])  # shape: (N_samples, N_wl)

    # Build label array
    label_array = _build_label_array(labels)

    # Normalise
    spectra_matrix = _normalise(spectra_matrix, common_wl, normalisation)

    return SpectralDataset(
        wavelengths=common_wl,
        spectra=spectra_matrix,
        signal_type=resolved_signal_type,
        normalisation=normalisation,
        labels=label_array,
        label_unit="ppm",
        analyte=analyte,
        config_id=config_id,
        metadata=metadata,
        source_paths=source_paths,
    )


def _load_single_file(
    path: Path,
    signal_type: SignalType,
    normalisation: Normalisation,
    reference_wavelengths: np.ndarray | None,
    analyte: str | None,
    config_id: str | None,
) -> SpectralDataset:
    """Load a single CSV or HDF5 file."""
    if path.suffix.lower() in (".h5", ".hdf5"):
        return _load_hdf5(path, signal_type, normalisation, analyte, config_id)

    if _is_timeseries_feature_file(path):
        raise ValueError(
            f"{path.name} appears to be a time-series feature file "
            "(has frame_index column). Use load_timeseries_features() instead."
        )

    wl, sig, stype, meta = _read_spectrum_csv(path, signal_type)
    conc = _infer_concentration(path)
    if analyte is None:
        analyte = _infer_analyte(path.parent)

    if reference_wavelengths is not None:
        sig = np.interp(reference_wavelengths, wl, sig)
        wl = reference_wavelengths

    spectra = sig[np.newaxis, :]  # (1, N_wl)
    spectra = _normalise(spectra, wl, normalisation)

    meta.update({"source_file": path.name, "analyte": analyte,
                 "concentration_ppm": conc})
    label_array = _build_label_array([conc])

    return SpectralDataset(
        wavelengths=wl,
        spectra=spectra,
        signal_type=stype,
        normalisation=normalisation,
        labels=label_array,
        label_unit="ppm",
        analyte=analyte,
        config_id=config_id,
        metadata=[meta],
        source_paths=[str(path)],
    )


# ---------------------------------------------------------------------------
# Time-series feature loading (separate public API)
# ---------------------------------------------------------------------------

def load_timeseries_features(path: str | Path) -> pd.DataFrame:
    """Load a time-series feature CSV (delta_lambda_nm, peak_wavelength_nm, …).

    Returns a pandas DataFrame with all columns preserved.
    This is the appropriate loader for files with frame_index columns.
    """
    path = Path(path)
    if path.is_dir():
        dfs = []
        for f in sorted(path.glob("*.csv")):
            if _is_timeseries_feature_file(f):
                df = pd.read_csv(f)
                df["source_file"] = f.name
                dfs.append(df)
        if not dfs:
            raise ValueError(f"No time-series feature CSVs found in {path}")
        return pd.concat(dfs, ignore_index=True)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------------

def _load_hdf5(
    path: Path,
    signal_type: SignalType,
    normalisation: Normalisation,
    analyte: str | None,
    config_id: str | None,
) -> SpectralDataset:
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py required for HDF5 loading: pip install h5py") from e

    with h5py.File(path, "r") as f:
        spectra = np.array(f["spectra"])
        wavelengths = (
            np.array(f["wavelengths"]) if "wavelengths" in f
            else np.arange(spectra.shape[1], dtype=float)
        )
        labels_raw = np.array(f["labels"]) if "labels" in f else None

    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]

    spectra = _normalise(spectra, wavelengths, normalisation)
    label_array = labels_raw

    return SpectralDataset(
        wavelengths=wavelengths,
        spectra=spectra,
        signal_type=signal_type if signal_type != "auto" else "intensity",
        normalisation=normalisation,
        labels=label_array,
        label_unit="ppm",
        analyte=analyte or _infer_analyte(path.parent),
        config_id=config_id,
        metadata=[{"source_file": path.name}] * len(spectra),
        source_paths=[str(path)],
    )


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def _read_spectrum_csv(
    path: Path, signal_type: SignalType
) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """Read a single long-format spectrum CSV.

    Returns
    -------
    wavelengths, signal, resolved_signal_type, metadata_dict
    """
    df = pd.read_csv(path)

    # Detect wide format (all numeric column headers = wavelength values)
    if _is_wide_format(df):
        wl = np.array([float(c) for c in df.columns], dtype=np.float64)
        # Wide format: multiple rows = multiple frames; take mean
        sig = df.values.astype(np.float64).mean(axis=0)
        return wl, sig, "intensity", {}

    # Long format: find wavelength column
    wl_col = _find_wavelength_column(df)
    if wl_col is None:
        raise ValueError(f"Cannot find wavelength column in {path.name}. "
                         f"Columns: {list(df.columns)}")

    wl = df[wl_col].values.astype(np.float64)

    # Find signal column
    sig_col, stype = _find_signal_column(df, wl_col, signal_type)
    if sig_col is None:
        raise ValueError(
            f"Cannot find signal column '{signal_type}' in {path.name}. "
            f"Available: {[c for c in df.columns if c != wl_col]}"
        )

    sig = df[sig_col].values.astype(np.float64)
    meta = {"n_pixels": len(wl), "signal_column": sig_col}
    return wl, sig, stype, meta


def _find_wavelength_column(df: pd.DataFrame) -> str | None:
    """Find the column that contains wavelength values (nm)."""
    candidates = ["wavelength", "wavelength_nm", "wl", "lambda", "nm",
                  "Wavelength", "WL", "Lambda"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back: first numeric column with values suggesting nm range (100–2500)
    for c in df.select_dtypes(include=[np.number]).columns:
        vals = df[c].dropna().values
        if len(vals) > 10 and 100 <= vals.min() and vals.max() <= 2500:
            return str(c)
    return None


def _find_signal_column(
    df: pd.DataFrame, wl_col: str, signal_type: SignalType
) -> tuple[str | None, str]:
    """Find the appropriate signal column."""
    non_wl = [c for c in df.columns if c != wl_col]

    if signal_type != "auto":
        # Exact or partial match
        for c in non_wl:
            if signal_type.lower() in c.lower():
                return c, signal_type
        return None, signal_type

    # Auto: pick by priority
    for stype in _SIGNAL_PRIORITY:
        for c in non_wl:
            if stype.lower() in c.lower():
                return c, stype

    # Last resort: first non-wavelength numeric column
    for c in non_wl:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c, "unknown"

    return None, "unknown"


def _is_wide_format(df: pd.DataFrame) -> bool:
    """Check if this is a wide-format spectrum (column headers are wavelengths)."""
    if len(df.columns) < 50:
        return False
    try:
        vals = [float(c) for c in df.columns]
        return 100 <= min(vals) and max(vals) <= 2500
    except (ValueError, TypeError):
        return False


def _is_timeseries_feature_file(path: Path) -> bool:
    """Check if this CSV contains time-series features rather than a spectrum."""
    try:
        header = pd.read_csv(path, nrows=0)
        cols = set(c.lower() for c in header.columns)
        ts_indicators = {"frame_index", "delta_lambda_nm", "delta_lambda",
                         "peak_wavelength_nm", "frame"}
        return bool(ts_indicators & cols)
    except Exception:
        return False


def _is_spectrum_file(path: Path) -> bool:
    """Return True if this CSV appears to be a spectrum file (not features)."""
    if not path.suffix.lower() == ".csv":
        return False
    return not _is_timeseries_feature_file(path)


# ---------------------------------------------------------------------------
# Metadata inference from paths
# ---------------------------------------------------------------------------

_ANALYTE_PATTERN = re.compile(
    r"(ethanol|ipa|isopropanol|methanol|acetone|toluene|benzene|ammonia|"
    r"co2|co|no2|so2|h2s|acetaldehyde|ethylene|propanol)",
    re.IGNORECASE,
)
_CONC_PATTERN = re.compile(
    r"(?:^|[-_/\\])(\d+(?:\.\d+)?)\s*(?:ppm|ppb|%)?(?:[-_]|$)",
)


def _infer_analyte(path: Path) -> str | None:
    """Infer analyte name from directory/file path."""
    for part in reversed(path.parts):
        m = _ANALYTE_PATTERN.search(part)
        if m:
            return m.group(0).capitalize()
    return None


def _infer_concentration(path: Path) -> float | None:
    """Infer concentration (ppm) from filename or parent directory name."""
    # Check filename first, then parent dir names
    candidates = [path.stem, path.parent.name, path.parent.parent.name]
    for text in candidates:
        m = _CONC_PATTERN.search(text)
        if m:
            return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalise(
    spectra: np.ndarray, wavelengths: np.ndarray, method: Normalisation
) -> np.ndarray:
    """Apply normalisation to a (N_samples, N_wl) matrix."""
    if method == "none":
        return spectra.copy()

    X = spectra.astype(np.float64).copy()

    if method == "snv":
        return _snv(X)

    if method == "msc":
        return _msc(X)

    if method == "area":
        areas = np.trapz(X, wavelengths, axis=1)[:, np.newaxis]
        areas = np.where(np.abs(areas) < 1e-12, 1.0, areas)
        return X / areas

    if method == "minmax":
        mn = X.min(axis=1, keepdims=True)
        mx = X.max(axis=1, keepdims=True)
        rng = np.where(mx - mn < 1e-12, 1.0, mx - mn)
        return (X - mn) / rng

    raise ValueError(f"Unknown normalisation method: {method!r}. "
                     "Choose from 'snv', 'msc', 'area', 'minmax', 'none'.")


def _snv(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate: subtract row mean, divide by row std."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


def _msc(X: np.ndarray) -> np.ndarray:
    """Multiplicative Scatter Correction against the mean spectrum."""
    reference = X.mean(axis=0)
    corrected = np.empty_like(X)
    for i, spec in enumerate(X):
        coeffs = np.polyfit(reference, spec, 1)
        corrected[i] = (spec - coeffs[1]) / coeffs[0]
    return corrected


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _build_label_array(labels: list) -> np.ndarray | None:
    """Convert list of concentration/label values to numpy array."""
    if all(v is None for v in labels):
        return None
    # Mixed None → nan for floats
    try:
        arr = np.array([float(v) if v is not None else np.nan for v in labels])
        return arr
    except (ValueError, TypeError):
        return np.array([str(v) for v in labels])


# ---------------------------------------------------------------------------
# SpectraAgent session loader  (C1 bridge)
# ---------------------------------------------------------------------------

#: Columns from pipeline_results.csv that become the feature matrix.
#: Order defines the synthetic "wavelength" axis (feature index 0, 1, …).
_SESSION_FEATURE_COLS = [
    "wavelength_shift",
    "peak_wavelength",
    "snr",
    "confidence_score",
]


def load_session_csv(
    session_dir: str | Path,
    normalisation: Normalisation = "none",
    min_frames: int = 1,
) -> SpectralDataset:
    """Load a SpectraAgent session directory into a :class:`SpectralDataset`.

    Reads ``pipeline_results.csv`` (processed per-frame LSPR features) and
    ``session_meta.json`` (gas label, session ID) from *session_dir* and
    returns a :class:`SpectralDataset` in **feature mode**:

    - ``spectra``    — shape ``(N_frames, N_features)``.  Features are
      ``[wavelength_shift, peak_wavelength, snr, confidence_score]``.
    - ``wavelengths``— synthetic integer index ``[0, 1, 2, 3]`` (feature index).
    - ``labels``     — ``concentration_ppm`` column (``NaN`` when absent).
    - ``analyte``    — ``gas_label`` from ``session_meta.json``.
    - ``signal_type``— ``"session_features"`` (not a raw spectrum).

    .. note::
        Because the feature matrix is not a raw spectrum, the spectral
        autoencoder sub-tab in the science dashboard will display a warning.
        Model Training and Cross-Dataset Analysis work normally.

    Parameters
    ----------
    session_dir :
        Path to an ``output/sessions/{YYYYMMDD_HHMMSS}/`` directory.
    normalisation :
        Normalisation to apply to the feature matrix (default ``'none'``).
    min_frames :
        Raise ``ValueError`` if fewer than this many frames are present after
        filtering NaN rows.

    Returns
    -------
    SpectralDataset
        Feature-mode dataset ready for model training and benchmarking.
    """
    import json

    session_dir = Path(session_dir)
    results_path = session_dir / "pipeline_results.csv"
    meta_path = session_dir / "session_meta.json"

    if not results_path.exists():
        raise FileNotFoundError(
            f"pipeline_results.csv not found in {session_dir}. "
            "Ensure the session ran long enough to record at least one frame."
        )

    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(
            f"pipeline_results.csv in {session_dir} is empty (0 rows). "
            "The session may not have captured any frames."
        )

    # Read session metadata
    analyte: str | None = None
    session_id = session_dir.name
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta_json = json.load(fh)
            gas_label = meta_json.get("gas_label", "")
            if gas_label and gas_label not in ("unknown", ""):
                analyte = str(gas_label)
            session_id = meta_json.get("session_id", session_dir.name)
        except Exception:
            pass  # Proceed without metadata — session_id from directory name

    # Extract available feature columns (gracefully skip missing ones)
    available = [c for c in _SESSION_FEATURE_COLS if c in df.columns]
    if not available:
        raise ValueError(
            f"pipeline_results.csv has none of the expected feature columns "
            f"({_SESSION_FEATURE_COLS}).  Columns found: {list(df.columns)}"
        )

    feature_matrix = df[available].values.astype(np.float64)  # (N, F)

    # Remove rows that are entirely NaN
    valid_rows = ~np.all(np.isnan(feature_matrix), axis=1)
    feature_matrix = feature_matrix[valid_rows]
    df_valid = df[valid_rows].reset_index(drop=True)

    if len(feature_matrix) < min_frames:
        raise ValueError(
            f"Session {session_dir.name} has only {len(feature_matrix)} valid "
            f"frame(s) after filtering NaN rows (minimum required: {min_frames})."
        )

    # Labels: concentration_ppm when available, else NaN
    if "concentration_ppm" in df_valid.columns:
        labels_raw = pd.to_numeric(df_valid["concentration_ppm"],
                                   errors="coerce").values.astype(np.float64)
    else:
        labels_raw = np.full(len(feature_matrix), np.nan)

    # Synthetic wavelength axis = feature indices
    synthetic_wl = np.arange(len(available), dtype=np.float64)

    # Apply normalisation
    feature_matrix = _normalise(feature_matrix, synthetic_wl, normalisation)

    # Per-row metadata
    meta_cols = [c for c in ("frame", "timestamp", "gas_type") if c in df_valid.columns]
    per_row_meta = [
        dict(row[meta_cols], feature_names=available, session_id=session_id)
        for _, row in df_valid[meta_cols].iterrows()
    ] if meta_cols else [{"feature_names": available, "session_id": session_id}] * len(feature_matrix)

    label_array = _build_label_array(list(labels_raw))

    return SpectralDataset(
        wavelengths=synthetic_wl,
        spectra=feature_matrix,
        signal_type="session_features",
        normalisation=normalisation,
        labels=label_array,
        label_unit="ppm",
        analyte=analyte,
        config_id=session_id,
        metadata=per_row_meta,
        source_paths=[str(results_path)],
    )


def list_sessions(sessions_root: str | Path = "output/sessions") -> list[Path]:
    """Return session directories sorted newest-first.

    Parameters
    ----------
    sessions_root :
        Parent directory containing ``YYYYMMDD_HHMMSS/`` sub-directories.
        Relative paths are resolved from the current working directory.

    Returns
    -------
    list[Path]
        Session directories in reverse-chronological order.
    """
    root = Path(sessions_root)
    if not root.is_dir():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and (d / "pipeline_results.csv").exists()],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Multi-dataset merge
# ---------------------------------------------------------------------------

def merge_datasets(*datasets: SpectralDataset,
                   reference_wavelengths: np.ndarray | None = None,
                   ) -> SpectralDataset:
    """Merge multiple SpectralDatasets onto a common wavelength grid.

    All datasets are interpolated to ``reference_wavelengths`` (or to the
    wavelength axis of the first dataset if not provided).  The merged dataset
    has ``config_id`` and ``analyte`` set to ``'merged'`` if they differ.
    """
    if not datasets:
        raise ValueError("At least one dataset required.")

    if reference_wavelengths is None:
        reference_wavelengths = datasets[0].wavelengths

    all_spectra, all_labels, all_meta, all_paths = [], [], [], []

    for ds in datasets:
        if np.allclose(ds.wavelengths, reference_wavelengths, atol=0.01):
            spectra = ds.spectra
        else:
            spectra = np.stack([
                np.interp(reference_wavelengths, ds.wavelengths, row)
                for row in ds.spectra
            ])
        all_spectra.append(spectra)
        all_labels.append(ds.labels)
        all_meta.extend(ds.metadata)
        all_paths.extend(ds.source_paths)

    merged_spectra = np.concatenate(all_spectra, axis=0)

    # Merge labels
    if all(lbl is None for lbl in all_labels):
        merged_labels = None
    else:
        parts = []
        for lbl, ds in zip(all_labels, datasets):
            if lbl is None:
                parts.append(np.full(ds.n_samples, np.nan))
            else:
                parts.append(lbl.astype(float))
        merged_labels = np.concatenate(parts)

    unique_analytes = {ds.analyte for ds in datasets}
    analytes = list(unique_analytes)
    unique_configs = {ds.config_id for ds in datasets}
    configs = list(unique_configs)

    return SpectralDataset(
        wavelengths=reference_wavelengths,
        spectra=merged_spectra,
        signal_type=datasets[0].signal_type,
        normalisation=datasets[0].normalisation,
        labels=merged_labels,
        label_unit=datasets[0].label_unit,
        analyte=analytes[0] if len(analytes) == 1 else "merged",
        config_id=configs[0] if len(unique_configs) == 1 else "merged",
        metadata=all_meta,
        source_paths=all_paths,
    )
