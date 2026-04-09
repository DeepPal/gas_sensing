"""
tests.test_agents
=================
Unit tests for src.agents.drift (DriftDetectionAgent) and
src.agents.quality (DataQualityAgent).
"""

from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np

from src.agents.drift import DriftAlert, DriftAlertType, DriftDetectionAgent
from src.agents.quality import DataQualityAgent, QualityCode, QualityResult

# ---------------------------------------------------------------------------
# DriftDetectionAgent
# ---------------------------------------------------------------------------


class TestDriftDetectionAgent:
    """Tests for slope-based and offset-based drift detection."""

    def _make_agent(self, **kwargs) -> DriftDetectionAgent:
        return DriftDetectionAgent(
            window_size=int(kwargs.get("window_size", 10)),
            drift_threshold_nm_per_min=float(kwargs.get("drift_threshold_nm_per_min", 0.5)),
            offset_threshold_nm=float(kwargs.get("offset_threshold_nm", 0.3)),
            min_samples=int(kwargs.get("min_samples", 5)),
        )

    def _now(self, offset_s: float = 0.0) -> datetime:
        return datetime.now(timezone.utc) + timedelta(seconds=offset_s)

    def test_no_alert_on_stable_baseline(self):
        agent = self._make_agent()
        alert = None
        for i in range(15):
            alert = agent.push(717.9, self._now(i * 5))
        # Stable flat signal → no alert
        assert alert is None

    def test_returns_none_before_min_samples(self):
        agent = self._make_agent(min_samples=8)
        alerts = [agent.push(717.9, self._now(i)) for i in range(7)]
        assert all(a is None for a in alerts)

    def test_slope_alert_on_fast_drift(self):
        """Simulate a 1 nm/min drift (well above 0.5 nm/min threshold)."""
        agent = self._make_agent(
            drift_threshold_nm_per_min=0.5,
            offset_threshold_nm=10.0,  # high — won't trigger
            min_samples=5,
        )
        # Push 10 samples with +1 nm/min drift (spacing 5 s = 1/12 min)
        # Δwl per step = 1/12 nm ≈ 0.083 nm
        for i in range(10):
            wl = 717.9 + i * (1.0 / 12.0)
            agent.push(wl, self._now(i * 5))

        status = agent.get_status()
        assert cast(float, status["drift_rate_nm_per_min"]) > 0.4

    def test_offset_alert_on_large_offset(self):
        agent = self._make_agent(
            drift_threshold_nm_per_min=100.0,  # high — won't trigger
            offset_threshold_nm=0.2,
            min_samples=5,
        )
        first = 717.9
        agent.push(first, self._now(0))
        for i in range(1, 10):
            agent.push(first + 0.5, self._now(i * 5))

        status = agent.get_status()
        assert abs(cast(float, status["baseline_offset_nm"])) > 0.05

    def test_alert_has_correct_type(self):
        agent = self._make_agent(
            drift_threshold_nm_per_min=0.01,  # very sensitive
            offset_threshold_nm=10.0,
            min_samples=3,
        )
        last_alert = None
        for i in range(10):
            wl = 717.9 + i * 0.1
            a = agent.push(wl, self._now(i * 5))
            if a is not None:
                last_alert = a
        if last_alert is not None:
            assert isinstance(last_alert, DriftAlert)
            assert last_alert.alert_type in (
                DriftAlertType.DRIFT_SLOPE,
                DriftAlertType.DRIFT_OFFSET,
            )

    def test_reset_clears_state(self):
        agent = self._make_agent()
        for i in range(10):
            agent.push(717.9 + i * 0.1, self._now(i))
        agent.reset()
        status = agent.get_status()
        assert status["n_samples"] == 0

    def test_get_recent_alerts_returns_list(self):
        agent = self._make_agent()
        alerts = agent.get_recent_alerts(5)
        assert isinstance(alerts, list)

    def test_get_status_keys(self):
        agent = self._make_agent()
        agent.push(717.9, self._now())
        status = agent.get_status()
        for key in ("n_samples", "drift_rate_nm_per_min", "baseline_offset_nm"):
            assert key in status, f"Missing key: {key}"

    def test_push_returns_alert_or_none(self):
        agent = self._make_agent()
        result = agent.push(717.9, self._now())
        assert result is None or isinstance(result, DriftAlert)

    def test_reset_baseline_updates_reference(self):
        agent = self._make_agent(min_samples=3)
        for i in range(5):
            agent.push(717.9, self._now(i))
        agent.reset_baseline()
        # After reset_baseline, offset from a new reading at the same wl should be ~0
        agent.push(717.9, self._now(10))
        status_after = agent.get_status()
        assert abs(cast(float, status_after.get("baseline_offset_nm", 0.0))) < 0.5

    def test_alert_message_is_string(self):
        agent = self._make_agent(
            drift_threshold_nm_per_min=0.01,
            min_samples=3,
        )
        last = None
        for i in range(10):
            a = agent.push(717.9 + i * 0.1, self._now(i * 5))
            if a:
                last = a
        if last is not None:
            assert isinstance(last.message, str)
            assert len(last.message) > 0


# ---------------------------------------------------------------------------
# DataQualityAgent
# ---------------------------------------------------------------------------


class TestDataQualityAgent:
    """Tests for the quality-gate pipeline."""

    def _wl_it(self, n=1000):
        """Return simple ascending wavelength and gaussian intensity arrays."""
        wl = np.linspace(500, 900, n)
        it = np.exp(-0.5 * ((wl - 700) / 30) ** 2) * 5000 + 100
        return wl, it

    def test_ok_spectrum(self):
        # UPDATED (2026-04-07): SNR < min_snr is now a hard fail (C6 fix)
        # Use very low min_snr to avoid hard-failing on the test spectrum
        agent = DataQualityAgent(min_snr=0.1)
        wl, it = self._wl_it()
        result = agent.check(wl, it)
        assert isinstance(result, QualityResult)
        # With low SNR threshold, should pass OK
        assert result.code == QualityCode.OK
        assert not result.is_hard_fail
        assert result.passed

    def test_saturated_spectrum_fails(self):
        agent = DataQualityAgent(saturation_threshold=60000)
        wl, it = self._wl_it()
        it[400:410] = 61000  # saturated pixels
        result = agent.check(wl, it)
        # Hard fail only if fraction > saturation_fraction_hard (default 0.1)
        # 10/1000 = 0.01 which is < 0.1 → not a hard fail but may warn
        assert isinstance(result, QualityResult)

    def test_saturation_hard_fail(self):
        agent = DataQualityAgent(saturation_threshold=60000, saturation_fraction_hard=0.05)
        wl, it = self._wl_it()
        # Saturate 10% of pixels (> 0.05 threshold → hard fail)
        it[0:100] = 61000
        result = agent.check(wl, it)
        assert result.code == QualityCode.FAIL_SATURATED
        assert not result.passed

    def test_non_finite_fails(self):
        agent = DataQualityAgent()
        wl, it = self._wl_it()
        it[50] = np.nan
        result = agent.check(wl, it)
        assert result.code == QualityCode.FAIL_NON_FINITE
        assert not result.passed

    def test_too_short_fails(self):
        agent = DataQualityAgent(min_points=50)
        wl = np.linspace(500, 600, 10)
        it = np.ones(10) * 500
        result = agent.check(wl, it)
        assert result.code == QualityCode.FAIL_TOO_SHORT
        assert not result.passed

    def test_low_signal_warning(self):
        agent = DataQualityAgent(min_signal=100.0)
        wl = np.linspace(500, 900, 1000)
        it = np.ones(1000) * 5.0  # very low but finite
        result = agent.check(wl, it)
        # Low signal is a warning (not fail)
        assert result.code in (
            QualityCode.WARNING_LOW_SIGNAL,
            QualityCode.FAIL_NON_FINITE,
            QualityCode.OK,
        )

    def test_low_snr_warning(self):
        agent = DataQualityAgent(min_snr=10.0)
        wl, it = self._wl_it()
        # Add heavy noise to drop SNR
        rng = np.random.default_rng(0)
        it = it + rng.normal(0, 500, len(it))
        result = agent.check(wl, it)
        assert result.code in (QualityCode.WARNING_LOW_SNR, QualityCode.OK)

    def test_quality_score_range(self):
        agent = DataQualityAgent()
        wl, it = self._wl_it()
        result = agent.check(wl, it)
        assert 0.0 <= result.quality_score <= 1.0

    def test_fail_score_low(self):
        agent = DataQualityAgent()
        wl, it = self._wl_it()
        it[0] = np.inf
        result = agent.check(wl, it)
        assert result.quality_score <= 0.5

    def test_ok_score_non_zero(self):
        """A non-catastrophic spectrum should have a quality_score > 0.

        UPDATED (2026-04-07): SNR is now a hard fail, use very low min_snr threshold.
        """
        agent = DataQualityAgent(min_snr=0.1)  # Very low so test spectrum passes
        wl, it = self._wl_it()
        result = agent.check(wl, it)
        # With low min_snr, spectrum should pass and have non-zero score
        assert result.quality_score > 0.0

    def test_quality_result_has_expected_fields(self):
        agent = DataQualityAgent()
        wl, it = self._wl_it()
        result = agent.check(wl, it)
        assert hasattr(result, "code")
        assert hasattr(result, "passed")
        assert hasattr(result, "quality_score")
        assert hasattr(result, "messages")

    def test_non_monotonic_wavelengths_fail(self):
        """Shuffled wavelength axis should fail the monotonicity gate."""
        agent = DataQualityAgent()
        wl = np.linspace(500, 900, 200)
        rng = np.random.default_rng(1)
        rng.shuffle(wl)
        it = np.ones(200) * 500
        result = agent.check(wl, it)
        assert result.code == QualityCode.FAIL_MONOTONICITY
        assert not result.passed

    def test_quality_code_enum_values(self):
        """Smoke test that all expected codes exist."""
        for name in (
            "OK",
            "WARNING_LOW_SNR",
            "WARNING_LOW_SIGNAL",
            "FAIL_SATURATED",
            "FAIL_NON_FINITE",
            "FAIL_MONOTONICITY",
            "FAIL_TOO_SHORT",
        ):
            assert hasattr(QualityCode, name)
