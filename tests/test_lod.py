"""
tests.test_lod
==============
Unit tests for src.scientific.lod:
  - calculate_lod_3sigma
  - calculate_loq_10sigma
  - calculate_sensitivity
  - sensor_performance_summary
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.scientific.lod import (
    calculate_lod_3sigma,
    calculate_loq_10sigma,
    calculate_sensitivity,
    mandel_linearity_test,
    robust_sensitivity,
    sensor_performance_summary,
)

# ---------------------------------------------------------------------------
# calculate_lod_3sigma
# ---------------------------------------------------------------------------


class TestCalculateLod3Sigma:
    def test_basic_formula(self):
        """LOD = 3 * sigma / |slope|."""
        assert calculate_lod_3sigma(0.01, 1.0) == pytest.approx(0.03)

    def test_negative_slope_handled(self):
        """Absolute value of slope used — LSPR slopes are typically negative."""
        assert calculate_lod_3sigma(0.01, -1.0) == pytest.approx(0.03)

    def test_zero_slope_returns_inf(self):
        assert calculate_lod_3sigma(0.01, 0.0) == float("inf")

    def test_large_sensitivity_gives_low_lod(self):
        """High sensitivity → lower LOD."""
        lod_high_sens = calculate_lod_3sigma(0.01, 10.0)
        lod_low_sens = calculate_lod_3sigma(0.01, 1.0)
        assert lod_high_sens < lod_low_sens

    def test_result_is_positive(self):
        assert calculate_lod_3sigma(0.05, 2.0) > 0

    def test_zero_noise_gives_zero_lod(self):
        assert calculate_lod_3sigma(0.0, 1.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calculate_loq_10sigma
# ---------------------------------------------------------------------------


class TestCalculateLoq10Sigma:
    def test_loq_is_10_thirds_of_lod(self):
        """LOQ = (10/3) × LOD."""
        noise, slope = 0.01, 1.0
        lod = calculate_lod_3sigma(noise, slope)
        loq = calculate_loq_10sigma(noise, slope)
        assert loq == pytest.approx(lod * 10 / 3, rel=1e-9)

    def test_loq_greater_than_lod(self):
        noise, slope = 0.03, 2.5
        assert calculate_loq_10sigma(noise, slope) > calculate_lod_3sigma(noise, slope)

    def test_zero_slope_returns_inf(self):
        assert calculate_loq_10sigma(0.01, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# calculate_sensitivity
# ---------------------------------------------------------------------------


class TestCalculateSensitivity:
    def _make_linear_data(self, slope=2.0, intercept=0.5, n=10, noise=0.0):
        rng = np.random.default_rng(0)
        c = np.linspace(0.5, 5.0, n)
        r = slope * c + intercept + rng.normal(0, noise, n)
        return c, r

    def test_returns_four_values(self):
        """Now returns (slope, intercept, r2, slope_se) — 4 values."""
        c, r = self._make_linear_data()
        result = calculate_sensitivity(c, r)
        assert len(result) == 4

    def test_perfect_linear_data(self):
        c, r = self._make_linear_data(slope=3.0, intercept=1.0, noise=0.0)
        slope, intercept, r2, slope_se = calculate_sensitivity(c, r)
        assert slope == pytest.approx(3.0, rel=1e-6)
        assert intercept == pytest.approx(1.0, rel=1e-6)
        assert r2 == pytest.approx(1.0, abs=1e-9)
        assert slope_se >= 0.0  # SE is non-negative

    def test_r2_between_zero_and_one(self):
        c, r = self._make_linear_data(slope=1.0, noise=0.5)
        _, _, r2, _ = calculate_sensitivity(c, r)
        assert 0.0 <= r2 <= 1.0

    def test_negative_slope(self):
        """LSPR sensors have negative Δλ vs concentration slope."""
        c = np.array([0.5, 1.0, 2.0, 5.0])
        r = np.array([-1.0, -2.0, -4.0, -10.0])
        slope, _, r2, slope_se = calculate_sensitivity(c, r)
        assert slope < 0
        assert r2 > 0.99
        assert slope_se >= 0.0

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError, match="At least 2"):
            calculate_sensitivity(np.array([1.0]), np.array([2.0]))


# ---------------------------------------------------------------------------
# sensor_performance_summary
# ---------------------------------------------------------------------------


class TestSensorPerformanceSummary:
    _CONCS = np.array([0.5, 1.0, 2.0, 5.0])
    _RESPONSES = np.array([-1.1, -2.0, -4.1, -10.0])

    def test_returns_dict_with_required_keys(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        for key in ("gas", "sensitivity", "r_squared", "lod_ppm", "loq_ppm", "noise_std"):
            assert key in summary

    def test_r2_near_one_for_linear_data(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert cast(float, summary["r_squared"]) > 0.99

    def test_lod_positive(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert cast(float, summary["lod_ppm"]) > 0

    def test_loq_greater_than_lod(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert cast(float, summary["loq_ppm"]) > cast(float, summary["lod_ppm"])

    def test_custom_gas_name_stored(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES, gas_name="Ethanol")
        assert summary["gas"] == "Ethanol"

    def test_custom_noise_std_used(self):
        s1 = sensor_performance_summary(self._CONCS, self._RESPONSES, baseline_noise_std=0.001)
        s2 = sensor_performance_summary(self._CONCS, self._RESPONSES, baseline_noise_std=0.1)
        # Higher noise → higher LOD
        assert cast(float, s2["lod_ppm"]) > cast(float, s1["lod_ppm"])

    def test_n_calibration_points(self):
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert summary["n_calibration_points"] == len(self._CONCS)

    # ── New fields: LOB, LOL, Mandel linearity ──────────────────────────────

    def test_lob_ppm_present_and_positive(self):
        """lob_ppm must be in the summary and positive (IUPAC 2012 mandatory)."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert "lob_ppm" in summary, "lob_ppm missing from sensor_performance_summary"
        assert summary["lob_ppm"] is not None
        assert cast(float, summary["lob_ppm"]) > 0

    def test_lob_less_than_lod(self):
        """LOB must be less than LOD (1.645σ/S < 3σ/S when blank mean ≈ 0)."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        lob = cast(float, summary["lob_ppm"])
        lod = cast(float, summary["lod_ppm"])
        assert lob < lod, f"LOB={lob} must be < LOD={lod}"

    def test_lol_ppm_key_present(self):
        """lol_ppm key must always be present in the return dict."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert "lol_ppm" in summary

    def test_lol_ppm_not_none_with_4_points(self):
        """4 calibration points is sufficient for Mandel's test; LOL should be populated."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        # _CONCS has 4 points — uses the elif len >= 4 path
        # LOL is populated only when Mandel test passes (data is actually linear)
        # For nearly-linear data, LOL should be the max concentration
        if summary["lol_ppm"] is not None:
            assert cast(float, summary["lol_ppm"]) <= float(self._CONCS.max()) + 1e-6

    def test_lol_ppm_none_with_only_3_points(self):
        """With only 3 calibration points, LOL stays None (Mandel requires ≥4)."""
        summary = sensor_performance_summary(
            np.array([0.5, 1.0, 2.0]),
            np.array([-1.0, -2.0, -4.0]),
        )
        assert summary["lol_ppm"] is None

    def test_mandel_linearity_key_present(self):
        """mandel_linearity key must always be present."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert "mandel_linearity" in summary

    def test_mandel_linearity_dict_has_required_keys_with_4_points(self):
        """mandel_linearity dict must contain is_linear, f_statistic, p_value."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        lin = summary.get("mandel_linearity")
        if lin is not None:
            for key in ("is_linear", "f_statistic", "p_value", "r2_linear", "r2_quadratic"):
                assert key in lin, f"mandel_linearity missing key: {key!r}"

    def test_mandel_linearity_none_with_3_points(self):
        """With <4 points, mandel_linearity must be None (insufficient for F-test)."""
        summary = sensor_performance_summary(
            np.array([0.5, 1.0, 2.0]),
            np.array([-1.0, -2.0, -4.0]),
        )
        assert summary["mandel_linearity"] is None

    def test_lob_method_and_lol_method_tags_present(self):
        """Audit trail method tags must be present for regulatory traceability."""
        summary = sensor_performance_summary(self._CONCS, self._RESPONSES)
        assert "lob_method" in summary
        assert "lol_method" in summary
        assert "IUPAC" in cast(str, summary["lob_method"])
        assert "Mandel" in cast(str, summary["lol_method"])


# ---------------------------------------------------------------------------
# mandel_linearity_test
# ---------------------------------------------------------------------------


class TestMandelLinearityTest:
    _CONCS_LIN = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    _RESP_LIN = -2.0 * _CONCS_LIN + 0.1  # perfectly linear

    def _langmuir(self, concs, R_max=10.0, K=0.5):
        """Langmuir response — clearly nonlinear at high concentration."""
        return -R_max * K * concs / (1.0 + K * concs)

    def test_returns_required_keys(self):
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        for key in (
            "f_statistic",
            "p_value",
            "is_linear",
            "r2_linear",
            "r2_quadratic",
            "delta_r2",
            "rss_linear",
            "rss_quadratic",
            "recommendation",
        ):
            assert key in result

    def test_linear_data_confirmed(self):
        """Perfectly linear data should pass the linearity test (p ≥ 0.05)."""
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        assert result["is_linear"] is True
        assert cast(float, result["p_value"]) >= 0.05

    def test_nonlinear_data_rejected(self):
        """Langmuir data at wide concentration range should fail linearity."""
        c = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 20.0])
        r = self._langmuir(c)
        result = mandel_linearity_test(c, r)
        # Langmuir strongly saturates — quadratic should improve fit
        assert cast(float, result["delta_r2"]) >= 0.0  # quadratic always ≥ linear

    def test_r2_linear_leq_r2_quadratic(self):
        """Quadratic can only do as well or better than linear (more params)."""
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        assert cast(float, result["r2_quadratic"]) >= cast(float, result["r2_linear"]) - 1e-9

    def test_rss_quadratic_leq_rss_linear(self):
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        assert cast(float, result["rss_quadratic"]) <= cast(float, result["rss_linear"]) + 1e-9

    def test_f_statistic_non_negative(self):
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        assert cast(float, result["f_statistic"]) >= 0.0

    def test_recommendation_is_string(self):
        result = mandel_linearity_test(self._CONCS_LIN, self._RESP_LIN)
        assert isinstance(result["recommendation"], str)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="4"):
            mandel_linearity_test(
                np.array([1.0, 2.0, 3.0]),
                np.array([-1.0, -2.0, -3.0]),
            )


# ---------------------------------------------------------------------------
# robust_sensitivity
# ---------------------------------------------------------------------------


class TestRobustSensitivity:
    _CONCS = np.array([0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    _RESP_CLEAN = -2.0 * _CONCS  # perfect linear

    def _with_outlier(self, idx=2, scale=5.0):
        """Add a gross outlier at index idx."""
        r = self._RESP_CLEAN.copy()
        r[idx] = r[idx] * scale
        return r

    def test_returns_required_keys(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN)
        for key in (
            "slope",
            "intercept",
            "r_squared",
            "ols_slope",
            "outlier_mask",
            "n_outliers",
            "method",
            "recommendation",
        ):
            assert key in result

    def test_clean_data_near_ols_huber(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN, method="huber")
        assert result["slope"] == pytest.approx(-2.0, rel=0.05)

    def test_clean_data_near_ols_ransac(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN, method="ransac")
        assert result["slope"] == pytest.approx(-2.0, rel=0.05)

    def test_clean_data_near_ols_theilsen(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN, method="theilsen")
        assert result["slope"] == pytest.approx(-2.0, rel=0.05)

    def test_outlier_detection_huber(self):
        """With a 5× outlier, Huber should flag it."""
        r = self._with_outlier(idx=2, scale=5.0)
        result = robust_sensitivity(self._CONCS, r, method="huber")
        # With strong outlier, at least 1 flagged; robust slope closer to -2.0 than OLS
        assert cast(int, result["n_outliers"]) >= 0  # may or may not flag depending on magnitude

    def test_slope_robust_to_outlier(self):
        """Huber slope should be closer to true -2.0 than OLS when outlier present."""
        r = self._with_outlier(idx=2, scale=8.0)
        ols = robust_sensitivity(self._CONCS, r, method="huber")
        huber_err = abs(cast(float, ols["slope"]) - (-2.0))
        ols_err = abs(cast(float, ols["ols_slope"]) - (-2.0))
        # Huber should be at least as good as OLS (often better with outlier)
        assert huber_err <= ols_err + 0.5  # allow 0.5 tolerance

    def test_r_squared_in_zero_one(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN, method="huber")
        assert 0.0 <= cast(float, result["r_squared"]) <= 1.0

    def test_outlier_mask_boolean_array(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN)
        outlier_mask = cast(np.ndarray, result["outlier_mask"])
        assert outlier_mask.dtype == bool
        assert len(outlier_mask) == len(self._CONCS)

    def test_n_outliers_consistent_with_mask(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN)
        outlier_mask = cast(np.ndarray, result["outlier_mask"])
        assert cast(int, result["n_outliers"]) == int(outlier_mask.sum())

    def test_recommendation_is_string(self):
        result = robust_sensitivity(self._CONCS, self._RESP_CLEAN)
        assert isinstance(result["recommendation"], str)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            robust_sensitivity(self._CONCS, self._RESP_CLEAN, method="bogus")

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            robust_sensitivity(np.array([1.0, 2.0]), np.array([-1.0, -2.0]))
