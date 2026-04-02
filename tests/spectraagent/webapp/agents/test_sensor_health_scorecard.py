"""
Unit tests for SensorHealthAgent._compute_scorecard logic.

Tests are isolated from the agent bus and Claude API — we construct the
agent with a stub bus and call _compute_scorecard directly.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spectraagent.webapp.agents.sensor_health import SensorHealthAgent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(memory=None) -> SensorHealthAgent:
    bus = MagicMock()
    return SensorHealthAgent(bus=bus, memory=memory)


# ---------------------------------------------------------------------------
# Scorecard — no history (cold start)
# ---------------------------------------------------------------------------

class TestScorecardColdStart:
    def test_returns_dict_with_required_keys(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=0.02, r2=0.999, sensitivity=-0.12,
            drift=0.0, snr=25.0, analyte="Ethanol",
        )
        for key in ("lod_score", "sensitivity_score", "r2_score",
                    "drift_score", "snr_score", "overall_health"):
            assert key in sc, f"missing key '{key}'"

    def test_all_none_inputs_returns_defaults(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=None, sensitivity=None,
            drift=None, snr=None, analyte=None,
        )
        # Should not raise; all defaults to partial score
        assert 0 <= sc["overall_health"] <= 100

    def test_scores_in_range_0_100(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=0.05, r2=0.995, sensitivity=-0.1,
            drift=0.0, snr=20.0, analyte="Ethanol",
        )
        for key in ("lod_score", "sensitivity_score", "r2_score",
                    "drift_score", "snr_score", "overall_health"):
            assert 0.0 <= sc[key] <= 100.0, f"{key} = {sc[key]} out of range"

    def test_perfect_r2_gives_max_r2_score(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=1.0, sensitivity=None,
            drift=None, snr=None, analyte=None,
        )
        assert sc["r2_score"] == pytest.approx(100.0)

    def test_poor_r2_gives_low_r2_score(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=0.90, sensitivity=None,
            drift=None, snr=None, analyte=None,
        )
        assert sc["r2_score"] == pytest.approx(0.0)

    def test_mid_r2_gives_mid_score(self):
        """R² = 0.95 maps to (0.95 - 0.90) / 0.10 * 100 = 50."""
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=0.95, sensitivity=None,
            drift=None, snr=None, analyte=None,
        )
        assert sc["r2_score"] == pytest.approx(50.0)

    def test_zero_drift_gives_full_drift_score(self):
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=None, sensitivity=None,
            drift=0.0, snr=None, analyte=None,
        )
        assert sc["drift_score"] == pytest.approx(100.0)

    def test_snr_at_minimum_gives_zero_snr_score(self):
        """SNR = 3.0 is the absolute minimum — score should be 0."""
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=None, sensitivity=None,
            drift=None, snr=3.0, analyte=None,
        )
        assert sc["snr_score"] == pytest.approx(0.0)

    def test_high_snr_gives_100_snr_score(self):
        """SNR = 30.0 maps to score = 100."""
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=None, sensitivity=None,
            drift=None, snr=30.0, analyte=None,
        )
        assert sc["snr_score"] == pytest.approx(100.0)

    def test_has_history_false_without_memory(self):
        agent = _make_agent(memory=None)
        sc = agent._compute_scorecard(
            lod=0.05, r2=0.999, sensitivity=-0.12,
            drift=0.0, snr=20.0, analyte="Ethanol",
        )
        assert sc["has_history"] is False

    def test_overall_is_weighted_sum(self):
        """Verify overall = weighted sum of dimension scores."""
        agent = _make_agent()
        sc = agent._compute_scorecard(
            lod=None, r2=1.0, sensitivity=None,
            drift=0.0, snr=30.0, analyte=None,
        )
        # Only r2, drift, snr are computable (no lod/sensitivity → default 50)
        expected = (
            50.0 * 0.30  # lod
            + 50.0 * 0.30  # sensitivity
            + 100.0 * 0.25  # r2
            + 100.0 * 0.10  # drift
            + 100.0 * 0.05  # snr
        )
        assert sc["overall_health"] == pytest.approx(expected, abs=0.5)


# ---------------------------------------------------------------------------
# Scorecard — with SensorMemory history
# ---------------------------------------------------------------------------

class TestScorecardWithHistory:
    def test_lod_better_than_best_gives_100(self, tmp_path: Path):
        from spectraagent.knowledge.sensor_memory import CalibrationObservation, SensorMemory
        mem = SensorMemory(memory_dir=tmp_path, sensor_id="test")
        mem.record_calibration(CalibrationObservation(
            session_id="s1", timestamp_utc="2026-01-01T00:00:00+00:00",
            analyte="Ethanol", sensitivity_nm_per_ppm=-0.12,
            lod_ppm=0.05, loq_ppm=0.167, r_squared=0.999, rmse_ppm=0.01,
            calibration_model="linear", n_calibration_points=6,
            reference_peak_nm=717.9, conformal_coverage=0.95,
        ))
        agent = _make_agent(memory=mem)
        # Current LOD = 0.03 < best recorded 0.05 → score = 0.05/0.03 * 100 > 100 → capped at 100
        sc = agent._compute_scorecard(
            lod=0.03, r2=None, sensitivity=None,
            drift=None, snr=None, analyte="Ethanol",
        )
        assert sc["lod_score"] == pytest.approx(100.0)
        assert sc["has_history"] is True

    def test_lod_worse_than_best_gives_sub_100(self, tmp_path: Path):
        from spectraagent.knowledge.sensor_memory import CalibrationObservation, SensorMemory
        mem = SensorMemory(memory_dir=tmp_path, sensor_id="test")
        mem.record_calibration(CalibrationObservation(
            session_id="s1", timestamp_utc="2026-01-01T00:00:00+00:00",
            analyte="Ethanol", sensitivity_nm_per_ppm=-0.12,
            lod_ppm=0.03, loq_ppm=0.10, r_squared=0.999, rmse_ppm=0.01,
            calibration_model="linear", n_calibration_points=6,
            reference_peak_nm=717.9, conformal_coverage=0.95,
        ))
        agent = _make_agent(memory=mem)
        # Current LOD = 0.06 > best 0.03 → score = 0.03/0.06 * 100 = 50
        sc = agent._compute_scorecard(
            lod=0.06, r2=None, sensitivity=None,
            drift=None, snr=None, analyte="Ethanol",
        )
        assert sc["lod_score"] == pytest.approx(50.0, abs=1.0)
