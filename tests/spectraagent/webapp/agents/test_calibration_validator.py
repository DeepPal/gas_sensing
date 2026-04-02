"""
Unit tests for CalibrationValidationOrchestrator.

Tests focus on:
- _get_or_build_tracker: correct key mapping from SensorMemory → ValidationTracker
- _update_selectivity: K-matrix derived overall_assessment
- on_event routing
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spectraagent.knowledge.protocols import ValidationStatus
from spectraagent.knowledge.sensor_memory import (
    CalibrationObservation,
    SensorMemory,
)
from spectraagent.webapp.agents.calibration_validator import (
    CalibrationValidationOrchestrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(memory=None, auto_explain=False) -> CalibrationValidationOrchestrator:
    bus = MagicMock()
    return CalibrationValidationOrchestrator(
        bus=bus,
        memory=memory,
        get_analyte=lambda: "Ethanol",
        auto_explain=auto_explain,
    )


def _make_memory(tmp_path: Path, with_calibration: bool = False) -> SensorMemory:
    mem = SensorMemory(memory_dir=tmp_path, sensor_id="test")
    if with_calibration:
        mem.record_calibration(CalibrationObservation(
            session_id="s1",
            timestamp_utc="2026-01-01T00:00:00+00:00",
            analyte="Ethanol",
            sensitivity_nm_per_ppm=-0.116,
            lod_ppm=0.015,
            loq_ppm=0.050,
            r_squared=0.9993,
            rmse_ppm=0.005,
            calibration_model="linear",
            n_calibration_points=7,
            reference_peak_nm=717.9,
            conformal_coverage=0.95,
        ))
    return mem


# ---------------------------------------------------------------------------
# _get_or_build_tracker — key mapping
# ---------------------------------------------------------------------------

class TestGetOrBuildTracker:
    def test_no_memory_returns_empty_tracker(self):
        orch = _make_orchestrator(memory=None)
        tracker = orch._get_or_build_tracker("Ethanol")
        assert tracker is not None
        assert tracker.completion_pct() == pytest.approx(0.0)

    def test_returns_same_tracker_on_second_call(self):
        orch = _make_orchestrator(memory=None)
        t1 = orch._get_or_build_tracker("Ethanol")
        t2 = orch._get_or_build_tracker("Ethanol")
        assert t1 is t2

    def test_memory_with_good_r2_advances_linearity(self, tmp_path: Path):
        """r_squared and n_calibration_points from memory must map to
        the 'r_squared' and 'n_points' keys expected by infer_from_calibration_data."""
        mem = _make_memory(tmp_path, with_calibration=True)
        orch = _make_orchestrator(memory=mem)
        tracker = orch._get_or_build_tracker("Ethanol")
        # r_squared=0.9993 and n_calibration_points=7 → linearity should advance
        assert tracker._status.get("linearity") == ValidationStatus.COMPLETE

    def test_memory_with_lod_advances_lod_loq(self, tmp_path: Path):
        mem = _make_memory(tmp_path, with_calibration=True)
        orch = _make_orchestrator(memory=mem)
        tracker = orch._get_or_build_tracker("Ethanol")
        assert tracker._status.get("lod_loq") == ValidationStatus.COMPLETE

    def test_different_analytes_have_independent_trackers(self, tmp_path: Path):
        mem = _make_memory(tmp_path, with_calibration=True)
        orch = _make_orchestrator(memory=mem)
        t_ethanol = orch._get_or_build_tracker("Ethanol")
        t_ipa = orch._get_or_build_tracker("IPA")
        assert t_ethanol is not t_ipa


# ---------------------------------------------------------------------------
# _update_selectivity — overall_assessment derivation
# ---------------------------------------------------------------------------

class TestUpdateSelectivity:
    @pytest.mark.asyncio
    async def test_no_memory_skips_selectivity(self):
        orch = _make_orchestrator(memory=None)
        event = MagicMock()
        event.data = {"gas_label": "Ethanol"}
        # Should not raise even with no memory
        await orch._update_selectivity(event)
        orch._bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_analyte_skips_selectivity(self, tmp_path: Path):
        mem = _make_memory(tmp_path, with_calibration=True)
        orch = _make_orchestrator(memory=mem)
        event = MagicMock()
        event.data = {"gas_label": "Ethanol"}
        # Only one analyte → not enough for selectivity matrix
        await orch._update_selectivity(event)
        orch._bus.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_analytes_emits_selectivity_updated(self, tmp_path: Path):
        mem = SensorMemory(memory_dir=tmp_path, sensor_id="test")
        # Record two analytes with different sensitivities
        for analyte, sens in [("Ethanol", -0.116), ("IPA", -0.090)]:
            mem.record_calibration(CalibrationObservation(
                session_id=f"s_{analyte}",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                analyte=analyte,
                sensitivity_nm_per_ppm=sens,
                lod_ppm=0.02,
                loq_ppm=0.07,
                r_squared=0.999,
                rmse_ppm=0.005,
                calibration_model="linear",
                n_calibration_points=6,
                reference_peak_nm=717.9,
                conformal_coverage=0.95,
            ))
        orch = _make_orchestrator(memory=mem)
        event = MagicMock()
        event.data = {"gas_label": "Ethanol"}
        await orch._update_selectivity(event)
        # Should have emitted selectivity_updated
        emitted_types = [call.args[0].type for call in orch._bus.emit.call_args_list]
        assert "selectivity_updated" in emitted_types

    @pytest.mark.asyncio
    async def test_selectivity_event_has_overall_assessment(self, tmp_path: Path):
        mem = SensorMemory(memory_dir=tmp_path, sensor_id="test")
        for analyte, sens in [("Ethanol", -0.116), ("IPA", -0.090)]:
            mem.record_calibration(CalibrationObservation(
                session_id=f"s_{analyte}",
                timestamp_utc="2026-01-01T00:00:00+00:00",
                analyte=analyte,
                sensitivity_nm_per_ppm=sens,
                lod_ppm=0.02, loq_ppm=0.07, r_squared=0.999, rmse_ppm=0.005,
                calibration_model="linear", n_calibration_points=6,
                reference_peak_nm=717.9, conformal_coverage=0.95,
            ))
        orch = _make_orchestrator(memory=mem)
        event = MagicMock()
        event.data = {"gas_label": "Ethanol"}
        await orch._update_selectivity(event)
        # Find the selectivity_updated event
        for call in orch._bus.emit.call_args_list:
            ev = call.args[0]
            if ev.type == "selectivity_updated":
                assert "overall_assessment" in ev.data
                assert isinstance(ev.data["overall_assessment"], str)
                return
        pytest.fail("selectivity_updated event not emitted")
