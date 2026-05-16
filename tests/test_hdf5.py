"""
Tests for src.io.hdf5 — ArchiveWriter and ArchiveReader.

Skipped automatically when h5py is not installed.
"""

from __future__ import annotations

import datetime
from pathlib import Path
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")

from src.io.hdf5 import ArchiveReader, ArchiveWriter, open_archive_reader, open_archive_writer
from src.spectrometer.base import SpectralFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(n: int = 512, conc: float | None = None, seed: int = 0) -> SpectralFrame:
    rng = np.random.default_rng(seed)
    wl = np.linspace(500.0, 1000.0, n)
    intensities = rng.uniform(100.0, 8000.0, n)
    meta: dict = {"modality": "lspr"}
    if conc is not None:
        meta["concentration_ppm"] = conc
    return SpectralFrame(
        wavelengths=wl,
        intensities=intensities,
        timestamp=datetime.datetime(2025, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        integration_time_s=0.05,
        accumulations=3,
        dark_corrected=False,
        nonlinearity_corrected=True,
        serial_number="TEST-001",
        model_name="TestSpec",
        metadata=meta,
    )


@pytest.fixture
def tmp_h5(tmp_path: Path) -> Path:
    return tmp_path / "test_session.h5"


# ===========================================================================
# ArchiveWriter
# ===========================================================================

class TestArchiveWriter:
    def test_creates_file(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5, gas_name="Ethanol"):
            pass
        assert tmp_h5.exists()

    def test_root_metadata_stored(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5, gas_name="Ethanol",
                                  instrument_model="CCS200", instrument_serial="SN-42"):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            assert f.attrs["gas_name"] == "Ethanol"
            assert f.attrs["instrument_model"] == "CCS200"
            assert f.attrs["instrument_serial"] == "SN-42"
            assert f.attrs["schema_version"] == "1.0"

    def test_n_frames_attribute_correct(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5) as aw:
            aw.add_frame(_make_frame(seed=0))
            aw.add_frame(_make_frame(seed=1))
            aw.add_frame(_make_frame(seed=2))
        with h5py.File(str(tmp_h5), "r") as f:
            assert int(f.attrs["n_frames"]) == 3

    def test_add_frame_stores_wavelengths_and_intensities(self, tmp_h5: Path) -> None:
        frame = _make_frame(n=256, seed=7)
        with open_archive_writer(tmp_h5) as aw:
            aw.add_frame(frame)
        with h5py.File(str(tmp_h5), "r") as f:
            wl = f["acquisition/frames/f000000/wavelengths"][()]
            it = f["acquisition/frames/f000000/intensities"][()]
        np.testing.assert_array_almost_equal(wl, frame.wavelengths)
        np.testing.assert_array_almost_equal(it, frame.intensities)

    def test_add_frame_stores_concentration(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5) as aw:
            aw.add_frame(_make_frame(), concentration_ppm=42.5)
        with h5py.File(str(tmp_h5), "r") as f:
            assert float(f["acquisition/frames/f000000"].attrs["concentration_ppm"]) == pytest.approx(42.5)

    def test_add_frame_stores_provenance_attrs(self, tmp_h5: Path) -> None:
        frame = _make_frame()
        with open_archive_writer(tmp_h5) as aw:
            aw.add_frame(frame)
        with h5py.File(str(tmp_h5), "r") as f:
            grp = f["acquisition/frames/f000000"]
            assert grp.attrs["serial_number"] == "TEST-001"
            assert grp.attrs["model_name"] == "TestSpec"
            assert float(grp.attrs["integration_time_s"]) == pytest.approx(0.05)
            assert int(grp.attrs["accumulations"]) == 3

    def test_set_dark_stored(self, tmp_h5: Path) -> None:
        dark = _make_frame(seed=99)
        with open_archive_writer(tmp_h5) as aw:
            aw.set_dark(dark)
        with h5py.File(str(tmp_h5), "r") as f:
            assert "acquisition/dark" in f

    def test_set_reference_stored(self, tmp_h5: Path) -> None:
        ref = _make_frame(seed=88)
        with open_archive_writer(tmp_h5) as aw:
            aw.set_reference(ref)
        with h5py.File(str(tmp_h5), "r") as f:
            assert "acquisition/reference" in f

    def test_set_calibration_stored(self, tmp_h5: Path) -> None:
        concs = np.array([1.0, 5.0, 10.0, 50.0])
        resps = np.array([-0.05, -0.25, -0.50, -2.50])
        fit = {"sensitivity": -0.05, "r_squared": 0.998, "lod_ppm": 0.3}
        with open_archive_writer(tmp_h5) as aw:
            aw.set_calibration(concs, resps, fit_result=fit)
        with h5py.File(str(tmp_h5), "r") as f:
            np.testing.assert_array_almost_equal(f["calibration/concentrations"][()], concs)
            assert float(f["calibration"].attrs["sensitivity"]) == pytest.approx(-0.05)
            assert float(f["calibration"].attrs["r_squared"]) == pytest.approx(0.998)

    def test_add_result_stored(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5) as aw:
            aw.add_result("2025-06-01T12:00:00+00:00", 42.1, uncertainty_ppm=1.5)
            aw.add_result("2025-06-01T12:00:01+00:00", 43.2, uncertainty_ppm=1.6)
        with h5py.File(str(tmp_h5), "r") as f:
            concs = f["results/concentrations_ppm"][()]
            assert len(concs) == 2
            assert float(concs[0]) == pytest.approx(42.1)

    def test_context_manager_closes_file(self, tmp_h5: Path) -> None:
        writer = open_archive_writer(tmp_h5)
        writer.__enter__()
        writer.add_frame(_make_frame())
        writer.__exit__(None, None, None)
        # File should be readable after close
        with h5py.File(str(tmp_h5), "r") as f:
            assert int(f.attrs["n_frames"]) == 1


# ===========================================================================
# ArchiveReader
# ===========================================================================

class TestArchiveReader:
    def _write_session(self, path: Path, n_frames: int = 3) -> None:
        with open_archive_writer(path, gas_name="Acetone") as aw:
            for i in range(n_frames):
                aw.add_frame(_make_frame(seed=i), concentration_ppm=float(i * 10))
            aw.set_dark(_make_frame(seed=100))
            aw.set_reference(_make_frame(seed=101))
            aw.set_calibration(
                np.array([0.0, 10.0, 20.0]),
                np.array([0.0, -0.5, -1.0]),
                fit_result={"sensitivity": -0.05, "r_squared": 0.999, "lod_ppm": 0.15},
            )
            aw.add_result("2025-01-01T00:00:00+00:00", 10.0, uncertainty_ppm=0.5)

    def test_read_metadata(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            assert ar.gas_name == "Acetone"
            assert ar.schema_version == "1.0"
            assert ar.n_frames == 3

    def test_read_frames_count(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            frames = ar.read_frames()
        assert len(frames) == 3

    def test_read_frames_content(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            frames = ar.read_frames()
        f0 = frames[0]
        assert "wavelengths" in f0
        assert "intensities" in f0
        assert len(f0["wavelengths"]) == 512

    def test_read_frame_concentration(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            frames = ar.read_frames()
        assert float(frames[1]["concentration_ppm"]) == pytest.approx(10.0)

    def test_read_dark(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            dark = ar.read_dark()
        assert dark is not None
        assert len(dark) == 512

    def test_read_reference(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            ref = ar.read_reference()
        assert ref is not None
        assert len(ref) == 512

    def test_read_calibration(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            cal = ar.read_calibration()
        assert "concentrations" in cal
        assert "responses" in cal
        assert float(cal["r_squared"]) == pytest.approx(0.999)

    def test_read_results(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            results = ar.read_results()
        assert "concentrations_ppm" in results
        assert float(results["concentrations_ppm"][0]) == pytest.approx(10.0)

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            open_archive_reader(tmp_path / "nonexistent.h5")

    def test_repr_contains_gas_name(self, tmp_h5: Path) -> None:
        self._write_session(tmp_h5)
        with open_archive_reader(tmp_h5) as ar:
            r = repr(ar)
        assert "Acetone" in r

    def test_roundtrip_wavelengths(self, tmp_h5: Path) -> None:
        """Wavelengths must survive the write→read cycle exactly."""
        frame = _make_frame(n=256, seed=5)
        with open_archive_writer(tmp_h5) as aw:
            aw.add_frame(frame)
        with open_archive_reader(tmp_h5) as ar:
            frames = ar.read_frames()
        np.testing.assert_array_almost_equal(
            frames[0]["wavelengths"], frame.wavelengths
        )

    def test_empty_archive_returns_empty_lists(self, tmp_h5: Path) -> None:
        with open_archive_writer(tmp_h5):
            pass
        with open_archive_reader(tmp_h5) as ar:
            assert ar.read_frames() == []
            assert ar.read_dark() is None
            assert ar.read_reference() is None

    # -------------------------------------------------------------------
    # C8: Environment metadata in HDF5 archives
    # -------------------------------------------------------------------

    def test_python_version_stored(self, tmp_h5: Path) -> None:
        """C8: HDF5 archive must record the Python version used to create it."""
        import sys
        with open_archive_writer(tmp_h5):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            pv = f.attrs["python_version"]
        expected = (
            f"{sys.version_info.major}.{sys.version_info.minor}"
            f".{sys.version_info.micro}"
        )
        assert pv == expected

    def test_os_platform_stored(self, tmp_h5: Path) -> None:
        """C8: HDF5 archive must record the OS platform."""
        import platform
        with open_archive_writer(tmp_h5):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            assert "os_platform" in f.attrs
            assert len(str(f.attrs["os_platform"])) > 0

    def test_numpy_version_stored(self, tmp_h5: Path) -> None:
        """C8: numpy package version must be recorded for reproducibility."""
        with open_archive_writer(tmp_h5):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            assert "pkg_numpy" in f.attrs
            assert str(f.attrs["pkg_numpy"]) != ""

    def test_scipy_version_stored(self, tmp_h5: Path) -> None:
        """C8: scipy package version must be recorded."""
        with open_archive_writer(tmp_h5):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            assert "pkg_scipy" in f.attrs

    def test_env_metadata_does_not_overwrite_schema_version(self, tmp_h5: Path) -> None:
        """C8: adding env metadata must not clobber the existing schema_version attr."""
        with open_archive_writer(tmp_h5, gas_name="Test"):
            pass
        with h5py.File(str(tmp_h5), "r") as f:
            assert f.attrs["schema_version"] == "1.0"
            assert "python_version" in f.attrs  # both must coexist
