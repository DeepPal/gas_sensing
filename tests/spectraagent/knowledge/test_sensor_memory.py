"""
Unit tests for spectraagent.knowledge.sensor_memory.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from spectraagent.knowledge.sensor_memory import (
    CalibrationObservation,
    FailureEvent,
    SensorMemory,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_dir(tmp_path: Path) -> Path:
    return tmp_path / "memory"


@pytest.fixture
def mem(mem_dir: Path) -> SensorMemory:
    return SensorMemory(memory_dir=mem_dir, sensor_id="test-sensor-001")


def _cal_obs(
    session_id: str = "s1",
    analyte: str = "Ethanol",
    lod: float = 0.05,
    r2: float = 0.995,
    sens: float = -0.116,
    tau_63: float | None = None,
    ref_fwhm: float | None = None,
) -> CalibrationObservation:
    return CalibrationObservation(
        session_id=session_id,
        timestamp_utc="2026-01-01T00:00:00+00:00",
        analyte=analyte,
        sensitivity_nm_per_ppm=sens,
        lod_ppm=lod,
        loq_ppm=lod * 10 / 3,
        r_squared=r2,
        rmse_ppm=0.01,
        calibration_model="linear",
        n_calibration_points=6,
        reference_peak_nm=717.9,
        conformal_coverage=0.94,
        tau_63_s=tau_63,
        reference_fwhm_nm=ref_fwhm,
    )


# ---------------------------------------------------------------------------
# CalibrationObservation dataclass
# ---------------------------------------------------------------------------

class TestCalibrationObservation:
    def test_new_fields_default_none(self):
        obs = CalibrationObservation(
            session_id="x", timestamp_utc="t", analyte="A",
            sensitivity_nm_per_ppm=None, lod_ppm=None, loq_ppm=None,
            r_squared=None, rmse_ppm=None, calibration_model="linear",
            n_calibration_points=0, reference_peak_nm=None, conformal_coverage=None,
        )
        assert obs.tau_63_s is None
        assert obs.reference_fwhm_nm is None

    def test_tau_63_stored(self):
        obs = _cal_obs(tau_63=18.5)
        assert obs.tau_63_s == pytest.approx(18.5)

    def test_ref_fwhm_stored(self):
        obs = _cal_obs(ref_fwhm=12.3)
        assert obs.reference_fwhm_nm == pytest.approx(12.3)


# ---------------------------------------------------------------------------
# SensorMemory.record_session / record_calibration
# ---------------------------------------------------------------------------

class TestSensorMemoryRecord:
    def test_record_session_increments_count(self, mem: SensorMemory) -> None:
        mem.record_session("s1", "Ethanol", 100, "2026-01-01T00:00:00+00:00")
        summary = mem.get_analyte_summary("Ethanol")
        assert summary is not None
        assert summary["n_sessions"] >= 1

    def test_record_calibration_populates_lod(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs("s1", "Ethanol", lod=0.05))
        summary = mem.get_analyte_summary("Ethanol")
        assert summary is not None
        # lod_ppm is a stats-dict; compare the mean value
        assert summary["lod_ppm"]["mean"] == pytest.approx(0.05)

    def test_multiple_sessions_averaged(self, mem: SensorMemory) -> None:
        for i, lod in enumerate([0.05, 0.06, 0.04]):
            mem.record_calibration(_cal_obs(f"s{i}", "Ethanol", lod=lod))
        summary = mem.get_analyte_summary("Ethanol")
        # mean of [0.05, 0.06, 0.04] = 0.05
        assert summary is not None
        assert summary["lod_ppm"]["mean"] == pytest.approx(0.05, abs=0.01)

    def test_tau_63_stored_in_history(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs("s1", "Ethanol", tau_63=22.0))
        # Verify it's persisted in the underlying data
        analyte_key = "ethanol"
        history = mem._data["analytes"][analyte_key]["calibration_history"]
        assert history[0]["tau_63_s"] == pytest.approx(22.0)


# ---------------------------------------------------------------------------
# SensorMemory.get_analyte_summary
# ---------------------------------------------------------------------------

class TestSensorMemoryGetSummary:
    def test_missing_analyte_returns_none(self, mem: SensorMemory) -> None:
        assert mem.get_analyte_summary("UnknownGas") is None

    def test_summary_keys_present(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs())
        summary = mem.get_analyte_summary("Ethanol")
        assert summary is not None
        for key in ("n_sessions", "lod_ppm", "dominant_model", "trend", "most_recent"):
            assert key in summary

    def test_trend_is_valid_string(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs("s1", lod=0.05))
        mem.record_calibration(_cal_obs("s2", lod=0.04))
        summary = mem.get_analyte_summary("Ethanol")
        assert summary is not None
        assert summary["trend"] in ("improving", "stable", "degrading", "insufficient_data")


# ---------------------------------------------------------------------------
# SensorMemory.get_all_analytes / get_sensitivities_by_analyte
# ---------------------------------------------------------------------------

class TestSensorMemoryMultiAnalyte:
    def test_get_all_analytes_empty(self, mem: SensorMemory) -> None:
        assert mem.get_all_analytes() == []

    def test_get_all_analytes_populated(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs(analyte="Ethanol"))
        mem.record_calibration(_cal_obs(analyte="IPA", session_id="s2"))
        analytes = mem.get_all_analytes()
        assert len(analytes) == 2

    def test_get_sensitivities_returns_dict(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs(analyte="Ethanol", sens=-0.116))
        result = mem.get_sensitivities_by_analyte()
        assert isinstance(result, dict)
        assert "Ethanol" in result

    def test_get_sensitivities_mean_is_correct(self, mem: SensorMemory) -> None:
        mem.record_calibration(_cal_obs("s1", analyte="Ethanol", sens=-0.10))
        mem.record_calibration(_cal_obs("s2", analyte="Ethanol", sens=-0.12))
        result = mem.get_sensitivities_by_analyte()
        assert result["Ethanol"] == pytest.approx(-0.11, abs=0.005)

    def test_multi_analyte_selectivity_ready(self, mem: SensorMemory) -> None:
        """With 2 analytes calibrated, get_sensitivities returns both."""
        mem.record_calibration(_cal_obs(analyte="Ethanol", sens=-0.116))
        mem.record_calibration(_cal_obs(analyte="IPA", session_id="s2", sens=-0.090))
        result = mem.get_sensitivities_by_analyte()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# SensorMemory persistence
# ---------------------------------------------------------------------------

class TestSensorMemoryPersistence:
    def test_data_persisted_across_instances(self, mem_dir: Path) -> None:
        m1 = SensorMemory(memory_dir=mem_dir, sensor_id="sensor-A")
        m1.record_calibration(_cal_obs("s1", "Ethanol", lod=0.05))

        m2 = SensorMemory(memory_dir=mem_dir, sensor_id="sensor-A")
        summary = m2.get_analyte_summary("Ethanol")
        assert summary is not None
        assert summary["lod_ppm"]["mean"] == pytest.approx(0.05)

    def test_different_sensor_ids_isolated(self, mem_dir: Path) -> None:
        m1 = SensorMemory(memory_dir=mem_dir, sensor_id="sensor-A")
        m2 = SensorMemory(memory_dir=mem_dir, sensor_id="sensor-B")
        m1.record_calibration(_cal_obs("s1", "Ethanol", lod=0.05))
        # sensor-B should not see sensor-A's data
        assert m2.get_analyte_summary("Ethanol") is None
