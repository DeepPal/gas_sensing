"""
src.io.hdf5
===========
HDF5 session archiving for the SpectraAgent — Spectrometer-Based Sensing Platform.

Writes complete measurement sessions to a self-describing HDF5 file
following a versioned schema.  The file is readable by:

- **Python / h5py** (``open_archive_reader``)
- **MATLAB** (``h5read``)
- **Origin** (File → Import → HDF5)
- **Thorlabs ThorSpectra** (spectrum import)

File layout
-----------
::

    session.h5
    ├── /acquisition
    │   ├── /frames
    │   │   ├── /f000000          (one group per SpectralFrame)
    │   │   │   ├── wavelengths   float64 (n_pixels,)
    │   │   │   ├── intensities   float64 (n_pixels,)
    │   │   │   └── attrs: timestamp_utc, integration_time_s,
    │   │   │              accumulations, dark_corrected,
    │   │   │              nonlinearity_corrected, serial_number,
    │   │   │              model_name, concentration_ppm (if known)
    │   │   └── ...
    │   ├── dark         float64 (n_pixels,) — mean dark spectrum
    │   └── reference    float64 (n_pixels,) — mean reference spectrum
    ├── /calibration
    │   ├── concentrations  float64 (n_cal,)
    │   ├── responses       float64 (n_cal,)
    │   └── attrs: sensitivity, intercept, r_squared,
    │              lod_ppm, loq_ppm, lol_ppm, lob_ppm
    └── /results
        ├── timestamps          str      (n,)
        ├── concentrations_ppm  float64  (n,)
        └── uncertainties_ppm   float64  (n,)  — NaN when unavailable

Root-level attributes
---------------------
``schema_version``, ``created_utc``, ``gas_name``, ``instrument_model``,
``instrument_serial``, ``n_frames``, ``pipeline_version``.

Usage
-----
::

    from src.io.hdf5 import open_archive_writer, open_archive_reader
    from src.spectrometer import SpectrometerRegistry

    # Write a session
    with open_archive_writer("session.h5", gas_name="Ethanol") as aw:
        for frame in frames:
            aw.add_frame(frame, concentration_ppm=conc)
        aw.set_dark(dark_frame)
        aw.set_reference(ref_frame)
        aw.set_calibration(concentrations, responses, fit_result)

    # Read it back
    with open_archive_reader("session.h5") as ar:
        frames = ar.read_frames()
        cal = ar.read_calibration()
        results = ar.read_results()

Dependencies
------------
``h5py`` is an optional dependency — a clear ``ImportError`` is raised with
install instructions if it is not present.

    pip install h5py
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Lazy h5py import — optional dependency
# ---------------------------------------------------------------------------


def _require_h5py() -> Any:
    try:
        import h5py  # type: ignore
        return h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 archiving.  "
            "Install with:  pip install h5py"
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _frame_key(index: int) -> str:
    return f"f{index:06d}"


def _write_str_dataset(group: Any, name: str, data: list[str]) -> None:
    """Write a list of strings as a fixed-length UTF-8 dataset."""
    h5py = _require_h5py()
    encoded = [s.encode("utf-8") for s in data]
    dt = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=encoded, dtype=dt)


def _read_str_dataset(dataset: Any) -> list[str]:
    """Read a string dataset back to a list of Python str."""
    raw = dataset[()]
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    result: list[str] = []
    for item in raw:
        if isinstance(item, bytes):
            result.append(item.decode("utf-8"))
        else:
            result.append(str(item))
    return result


# ---------------------------------------------------------------------------
# ArchiveWriter
# ---------------------------------------------------------------------------


class ArchiveWriter:
    """Write a measurement session to an HDF5 archive.

    This class should be used as a context manager (``open_archive_writer``).
    It is not thread-safe; use one writer per file.

    Parameters
    ----------
    path :
        Output file path.  Existing files are overwritten.
    gas_name :
        Analyte name stored in root-level metadata.
    instrument_model :
        Spectrometer model string (e.g. ``"CCS200/M"``).
    instrument_serial :
        Spectrometer serial number.
    pipeline_version :
        Software version string.  Defaults to ``src.__version__`` if
        available.
    """

    def __init__(
        self,
        path: str | Path,
        gas_name: str = "",
        instrument_model: str = "",
        instrument_serial: str = "",
        pipeline_version: str = "",
    ) -> None:
        h5py = _require_h5py()
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not pipeline_version:
            try:
                from src import __version__
                pipeline_version = str(__version__)
            except Exception:
                pipeline_version = "unknown"

        self._file = h5py.File(str(self._path), "w")
        self._frame_count = 0
        self._result_rows: list[dict[str, Any]] = []

        # Root attributes — measurement identity
        self._file.attrs["schema_version"] = _SCHEMA_VERSION
        self._file.attrs["created_utc"] = _utc_now_iso()
        self._file.attrs["gas_name"] = gas_name
        self._file.attrs["instrument_model"] = instrument_model
        self._file.attrs["instrument_serial"] = instrument_serial
        self._file.attrs["pipeline_version"] = pipeline_version

        # C8: Environment metadata — required to reproduce any numerical result
        import sys
        import platform as _platform
        from importlib.metadata import version as _pkg_version
        self._file.attrs["python_version"] = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )
        self._file.attrs["os_platform"] = _platform.platform()
        for _pkg in ("numpy", "scipy", "scikit-learn", "torch", "anthropic"):
            try:
                self._file.attrs[f"pkg_{_pkg.replace('-', '_')}"] = _pkg_version(_pkg)
            except Exception:
                self._file.attrs[f"pkg_{_pkg.replace('-', '_')}"] = "not_installed"

        # Create group skeleton
        self._acq = self._file.require_group("acquisition")
        self._frames_grp = self._acq.require_group("frames")
        self._cal_grp = self._file.require_group("calibration")
        self._res_grp = self._file.require_group("results")

        log.debug("ArchiveWriter: opened %s", self._path)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def add_frame(
        self,
        frame: Any,
        concentration_ppm: float | None = None,
    ) -> None:
        """Append one :class:`~src.spectrometer.SpectralFrame` to the archive.

        Parameters
        ----------
        frame :
            A :class:`~src.spectrometer.SpectralFrame` (or any object with
            ``wavelengths``, ``intensities``, ``timestamp`` attributes).
        concentration_ppm :
            Known analyte concentration for this frame.  Use ``None`` when
            unknown.
        """
        key = _frame_key(self._frame_count)
        grp = self._frames_grp.create_group(key)

        wavelengths = np.asarray(frame.wavelengths, dtype=np.float64)
        intensities = np.asarray(frame.intensities, dtype=np.float64)

        grp.create_dataset("wavelengths", data=wavelengths,
                           compression="gzip", compression_opts=4)
        grp.create_dataset("intensities", data=intensities,
                           compression="gzip", compression_opts=4)

        # Provenance attributes
        ts = frame.timestamp
        if isinstance(ts, datetime.datetime):
            grp.attrs["timestamp_utc"] = ts.isoformat()
        else:
            grp.attrs["timestamp_utc"] = str(ts)

        grp.attrs["integration_time_s"] = float(
            getattr(frame, "integration_time_s", float("nan"))
        )
        grp.attrs["accumulations"] = int(
            getattr(frame, "accumulations", 1)
        )
        grp.attrs["dark_corrected"] = bool(
            getattr(frame, "dark_corrected", False)
        )
        grp.attrs["nonlinearity_corrected"] = bool(
            getattr(frame, "nonlinearity_corrected", False)
        )
        grp.attrs["serial_number"] = str(
            getattr(frame, "serial_number", "")
        )
        grp.attrs["model_name"] = str(
            getattr(frame, "model_name", "")
        )
        if concentration_ppm is not None:
            grp.attrs["concentration_ppm"] = float(concentration_ppm)

        # Extra metadata dict
        meta = getattr(frame, "metadata", {}) or {}
        if meta:
            with contextlib.suppress(TypeError, ValueError):
                grp.attrs["metadata_json"] = json.dumps(meta)

        self._frame_count += 1
        log.debug("ArchiveWriter: added frame %s (total=%d)", key, self._frame_count)

    # ------------------------------------------------------------------
    # Dark / reference spectra
    # ------------------------------------------------------------------

    def set_dark(self, dark_frame: Any) -> None:
        """Store the mean dark spectrum (used for dark correction).

        Parameters
        ----------
        dark_frame :
            A :class:`~src.spectrometer.SpectralFrame` acquired with the
            shutter closed.
        """
        if "dark" in self._acq:
            del self._acq["dark"]
        self._acq.create_dataset(
            "dark",
            data=np.asarray(dark_frame.intensities, dtype=np.float64),
            compression="gzip", compression_opts=4,
        )
        self._acq["dark"].attrs["timestamp_utc"] = (
            dark_frame.timestamp.isoformat()
            if isinstance(dark_frame.timestamp, datetime.datetime)
            else str(dark_frame.timestamp)
        )
        log.debug("ArchiveWriter: stored dark spectrum")

    def set_reference(self, ref_frame: Any) -> None:
        """Store the reference (blank / white-light) spectrum.

        Parameters
        ----------
        ref_frame :
            A :class:`~src.spectrometer.SpectralFrame` acquired without analyte.
        """
        if "reference" in self._acq:
            del self._acq["reference"]
        self._acq.create_dataset(
            "reference",
            data=np.asarray(ref_frame.intensities, dtype=np.float64),
            compression="gzip", compression_opts=4,
        )
        wl = np.asarray(ref_frame.wavelengths, dtype=np.float64)
        if "wavelengths" not in self._acq:
            self._acq.create_dataset(
                "wavelengths", data=wl,
                compression="gzip", compression_opts=4,
            )
        log.debug("ArchiveWriter: stored reference spectrum")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def set_calibration(
        self,
        concentrations: np.ndarray,
        responses: np.ndarray,
        fit_result: dict[str, Any] | None = None,
    ) -> None:
        """Store calibration concentrations, responses, and model metrics.

        Parameters
        ----------
        concentrations :
            Calibration concentrations, shape ``(n,)``.
        responses :
            Corresponding sensor responses (Δλ or intensity), shape ``(n,)``.
        fit_result :
            Dict from :func:`~src.reporting.metrics.compute_comprehensive_sensor_characterization`.
            All scalar entries are stored as HDF5 attributes; the full dict
            is also JSON-serialised for lossless round-trip.
        """
        grp = self._cal_grp

        if "concentrations" in grp:
            del grp["concentrations"]
        if "responses" in grp:
            del grp["responses"]

        grp.create_dataset(
            "concentrations",
            data=np.asarray(concentrations, dtype=np.float64),
        )
        grp.create_dataset(
            "responses",
            data=np.asarray(responses, dtype=np.float64),
        )

        if fit_result:
            scalar_keys = [
                "sensitivity", "sensitivity_se", "intercept",
                "r_squared", "rmse", "noise_std",
                "lob_ppm", "lod_ppm", "loq_ppm", "lol_ppm",
            ]
            for k in scalar_keys:
                v = fit_result.get(k)
                if v is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        grp.attrs[k] = float(v)

            # Full dict — JSON for lossless round-trip
            try:
                grp.attrs["fit_result_json"] = json.dumps(
                    fit_result, default=_json_default
                )
            except (TypeError, ValueError) as exc:
                log.warning("ArchiveWriter: could not serialise fit_result — %s", exc)

        log.debug("ArchiveWriter: stored calibration (%d points)", len(concentrations))

    # ------------------------------------------------------------------
    # Pipeline results
    # ------------------------------------------------------------------

    def add_result(
        self,
        timestamp: str | datetime.datetime,
        concentration_ppm: float,
        uncertainty_ppm: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Append one pipeline result row.

        Parameters
        ----------
        timestamp :
            UTC timestamp (ISO string or datetime).
        concentration_ppm :
            Predicted analyte concentration.
        uncertainty_ppm :
            GPR uncertainty (1-σ).  Use ``None`` if not available.
        extra :
            Additional key-value pairs stored in the row's JSON blob.
        """
        ts_str = (
            timestamp.isoformat()
            if isinstance(timestamp, datetime.datetime)
            else str(timestamp)
        )
        row: dict[str, Any] = {
            "timestamp": ts_str,
            "concentration_ppm": float(concentration_ppm),
            "uncertainty_ppm": float(uncertainty_ppm) if uncertainty_ppm is not None else float("nan"),
        }
        if extra:
            row["extra_json"] = json.dumps(extra, default=_json_default)
        self._result_rows.append(row)

    # ------------------------------------------------------------------
    # Flush results on close
    # ------------------------------------------------------------------

    def _flush_results(self) -> None:
        if not self._result_rows:
            return
        grp = self._res_grp
        ts_list = [r["timestamp"] for r in self._result_rows]
        concs = np.array([r["concentration_ppm"] for r in self._result_rows], dtype=np.float64)
        uncerts = np.array([r["uncertainty_ppm"] for r in self._result_rows], dtype=np.float64)

        _write_str_dataset(grp, "timestamps", ts_list)
        grp.create_dataset("concentrations_ppm", data=concs)
        grp.create_dataset("uncertainties_ppm", data=uncerts)

        log.debug("ArchiveWriter: flushed %d result rows", len(self._result_rows))

    # ------------------------------------------------------------------
    # Close / context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        self._flush_results()
        self._file.attrs["n_frames"] = self._frame_count
        self._file.attrs["n_results"] = len(self._result_rows)
        self._file.flush()
        self._file.close()
        log.info("ArchiveWriter: closed %s (%d frames)", self._path, self._frame_count)

    def __enter__(self) -> ArchiveWriter:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ArchiveReader
# ---------------------------------------------------------------------------


class ArchiveReader:
    """Read a measurement session from an HDF5 archive.

    This class should be used as a context manager (``open_archive_reader``).

    Parameters
    ----------
    path :
        Path to the HDF5 archive file.
    """

    def __init__(self, path: str | Path) -> None:
        h5py = _require_h5py()
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"Archive not found: {self._path}")
        self._file = h5py.File(str(self._path), "r")
        log.debug("ArchiveReader: opened %s", self._path)

    # ------------------------------------------------------------------
    # Root metadata
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        """Return root-level attributes as a plain dict."""
        return dict(self._file.attrs)

    @property
    def schema_version(self) -> str:
        return str(self._file.attrs.get("schema_version", "unknown"))

    @property
    def gas_name(self) -> str:
        return str(self._file.attrs.get("gas_name", ""))

    @property
    def created_utc(self) -> str:
        return str(self._file.attrs.get("created_utc", ""))

    @property
    def n_frames(self) -> int:
        return int(self._file.attrs.get("n_frames", 0))

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------

    def read_frames(self) -> list[dict[str, Any]]:
        """Read all stored frames as a list of dicts.

        Each dict has keys:
        ``wavelengths``, ``intensities``, ``timestamp_utc``,
        ``integration_time_s``, ``accumulations``, ``dark_corrected``,
        ``nonlinearity_corrected``, ``serial_number``, ``model_name``,
        ``concentration_ppm`` (may be absent).

        Returns
        -------
        list[dict]
            Frames in acquisition order.
        """
        frames_grp = self._file.get("acquisition/frames")
        if frames_grp is None:
            return []

        frames: list[dict[str, Any]] = []
        for key in sorted(frames_grp.keys()):
            grp = frames_grp[key]
            frame: dict[str, Any] = {
                "wavelengths": grp["wavelengths"][()].copy(),
                "intensities": grp["intensities"][()].copy(),
            }
            for attr in (
                "timestamp_utc", "integration_time_s", "accumulations",
                "dark_corrected", "nonlinearity_corrected",
                "serial_number", "model_name", "concentration_ppm",
            ):
                if attr in grp.attrs:
                    frame[attr] = grp.attrs[attr]

            # Deserialise extra metadata
            if "metadata_json" in grp.attrs:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    frame["metadata"] = json.loads(grp.attrs["metadata_json"])

            frames.append(frame)

        return frames

    def read_frame(self, index: int) -> dict[str, Any]:
        """Read a single frame by zero-based index."""
        frames = self.read_frames()
        if index < 0 or index >= len(frames):
            raise IndexError(
                f"Frame index {index} out of range (n_frames={len(frames)})"
            )
        return frames[index]

    def read_dark(self) -> np.ndarray | None:
        """Return the stored dark spectrum or ``None`` if absent."""
        ds = self._file.get("acquisition/dark")
        return ds[()].copy() if ds is not None else None

    def read_reference(self) -> np.ndarray | None:
        """Return the stored reference spectrum or ``None`` if absent."""
        ds = self._file.get("acquisition/reference")
        return ds[()].copy() if ds is not None else None

    def read_wavelengths(self) -> np.ndarray | None:
        """Return the shared wavelength axis or ``None`` if absent."""
        ds = self._file.get("acquisition/wavelengths")
        return ds[()].copy() if ds is not None else None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def read_calibration(self) -> dict[str, Any]:
        """Return calibration data and stored metrics.

        Returns
        -------
        dict
            Keys: ``concentrations``, ``responses``, plus any scalar
            metric attributes (``sensitivity``, ``lod_ppm``, …) and
            optionally ``fit_result`` (full dict from JSON).
        """
        grp = self._file.get("calibration")
        if grp is None:
            return {}

        result: dict[str, Any] = {}

        if "concentrations" in grp:
            result["concentrations"] = grp["concentrations"][()].copy()
        if "responses" in grp:
            result["responses"] = grp["responses"][()].copy()

        for attr in (
            "sensitivity", "sensitivity_se", "intercept",
            "r_squared", "rmse", "noise_std",
            "lob_ppm", "lod_ppm", "loq_ppm", "lol_ppm",
        ):
            if attr in grp.attrs:
                result[attr] = float(grp.attrs[attr])

        if "fit_result_json" in grp.attrs:
            with contextlib.suppress(json.JSONDecodeError, TypeError):
                result["fit_result"] = json.loads(grp.attrs["fit_result_json"])

        return result

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def read_results(self) -> dict[str, Any]:
        """Return pipeline result time-series.

        Returns
        -------
        dict
            Keys: ``timestamps`` (list[str]), ``concentrations_ppm``
            (ndarray), ``uncertainties_ppm`` (ndarray).
        """
        grp = self._file.get("results")
        if grp is None:
            return {}

        result: dict[str, Any] = {}
        if "timestamps" in grp:
            result["timestamps"] = _read_str_dataset(grp["timestamps"])
        if "concentrations_ppm" in grp:
            result["concentrations_ppm"] = grp["concentrations_ppm"][()].copy()
        if "uncertainties_ppm" in grp:
            result["uncertainties_ppm"] = grp["uncertainties_ppm"][()].copy()

        return result

    # ------------------------------------------------------------------
    # Close / context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the HDF5 file handle."""
        self._file.close()
        log.debug("ArchiveReader: closed %s", self._path)

    def __enter__(self) -> ArchiveReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ArchiveReader(path={self._path!r}, "
            f"gas={self.gas_name!r}, n_frames={self.n_frames})"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def open_archive_writer(
    path: str | Path,
    gas_name: str = "",
    instrument_model: str = "",
    instrument_serial: str = "",
    pipeline_version: str = "",
) -> ArchiveWriter:
    """Create and return an :class:`ArchiveWriter` for *path*.

    Use as a context manager::

        with open_archive_writer("session.h5", gas_name="Ethanol") as aw:
            aw.add_frame(frame)
    """
    return ArchiveWriter(
        path=path,
        gas_name=gas_name,
        instrument_model=instrument_model,
        instrument_serial=instrument_serial,
        pipeline_version=pipeline_version,
    )


def open_archive_reader(path: str | Path) -> ArchiveReader:
    """Open *path* as an :class:`ArchiveReader`.

    Use as a context manager::

        with open_archive_reader("session.h5") as ar:
            frames = ar.read_frames()
    """
    return ArchiveReader(path=path)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """Handle non-serialisable types when writing fit_result JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")
