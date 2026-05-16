"""
Unit tests for src.scientific.cross_session.
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from src.scientific.cross_session import (
    CrossSessionComparison,
    SessionData,
    compare_lod_series,
    compare_sessions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(
    shifts: list[float],
    concs: list[float] | None = None,
    lod: float | None = None,
    sens: float | None = None,
    sid: str = "s1",
) -> SessionData:
    if concs is None:
        concs = list(range(1, len(shifts) + 1))
    return SessionData(
        concentrations=concs,
        delta_lambdas=shifts,
        session_id=sid,
        lod_ppm=lod,
        sensitivity_nm_per_ppm=sens,
    )


# ---------------------------------------------------------------------------
# SessionData dataclass
# ---------------------------------------------------------------------------

class TestSessionData:
    def test_basic_construction(self):
        sd = SessionData(concentrations=[0.1, 0.5], delta_lambdas=[-0.1, -0.5])
        assert len(sd.concentrations) == 2
        assert sd.lod_ppm is None

    def test_optional_fields(self):
        sd = SessionData([1, 2], [-1.0, -2.0], lod_ppm=0.05, sensitivity_nm_per_ppm=-1.0)
        assert sd.lod_ppm == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# compare_sessions — identical sessions → reproducible
# ---------------------------------------------------------------------------

class TestCompareSessionsReproducible:
    def test_identical_sessions_not_significant(self):
        shifts = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(shifts, lod=0.05)
        b = _make_session(shifts, lod=0.05)
        result = compare_sessions(a, b)
        assert result.paired_t_significant is False

    def test_bland_altman_bias_zero_for_identical(self):
        shifts = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(shifts)
        b = _make_session(shifts)
        result = compare_sessions(a, b)
        assert result.bland_altman_bias == pytest.approx(0.0)

    def test_bland_altman_loa_symmetric_for_identical(self):
        shifts = [-0.2, -0.4, -0.8, -1.6, -3.2]
        a = _make_session(shifts)
        b = _make_session(shifts)
        result = compare_sessions(a, b)
        if result.bland_altman_loa_lower is not None:
            assert result.bland_altman_loa_lower == pytest.approx(0.0, abs=1e-9)
            assert result.bland_altman_loa_upper == pytest.approx(0.0, abs=1e-9)

    def test_rsd_near_zero_for_identical(self):
        shifts = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(shifts)
        b = _make_session(shifts)
        result = compare_sessions(a, b)
        if result.reproducibility_rsd_pct is not None:
            assert result.reproducibility_rsd_pct == pytest.approx(0.0, abs=1e-6)

    def test_f_variances_equal_for_identical(self):
        shifts = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(shifts)
        b = _make_session(shifts)
        result = compare_sessions(a, b)
        # F-test: identical variances → p should be 1.0 → equal
        assert result.f_variances_equal is True

    def test_reproducible_flag_true(self):
        shifts = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(shifts, lod=0.05)
        b = _make_session(shifts, lod=0.05)
        result = compare_sessions(a, b, lod_ppm=0.05)
        assert result.sessions_reproducible is True


# ---------------------------------------------------------------------------
# compare_sessions — systematically different sessions
# ---------------------------------------------------------------------------

class TestCompareSessionsDifferent:
    def test_large_bias_flagged_as_significant(self):
        a = _make_session([-0.1, -0.5, -1.0, -2.0, -5.0])
        b = _make_session([-1.1, -1.5, -2.0, -3.0, -6.0])  # 1 nm systematic shift
        result = compare_sessions(a, b)
        assert result.paired_t_significant is True
        assert result.bland_altman_bias == pytest.approx(-1.0)

    def test_loa_reflects_spread(self):
        rng = np.random.default_rng(1)
        base = [-0.1, -0.5, -1.0, -2.0, -5.0]
        a = _make_session(base)
        b_shifts = [v + float(rng.normal(0, 0.3)) for v in base]
        b = _make_session(b_shifts)
        result = compare_sessions(a, b)
        assert result.bland_altman_sd is not None
        assert result.bland_altman_sd > 0

    def test_delta_lod_populated(self):
        a = _make_session([-0.1, -0.5, -1.0], lod=0.05)
        b = _make_session([-0.1, -0.5, -1.0], lod=0.08)
        result = compare_sessions(a, b)
        assert result.delta_lod_ppm == pytest.approx(0.03)

    def test_delta_sensitivity_populated(self):
        a = _make_session([-0.1, -0.5, -1.0], sens=-0.10)
        b = _make_session([-0.1, -0.5, -1.0], sens=-0.12)
        result = compare_sessions(a, b)
        assert result.delta_sensitivity_nm_per_ppm == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# compare_sessions — return type & structure
# ---------------------------------------------------------------------------

class TestCompareSessionsStructure:
    def test_returns_cross_session_comparison(self):
        a = _make_session([-1.0, -2.0, -3.0])
        b = _make_session([-1.0, -2.0, -3.0])
        result = compare_sessions(a, b)
        assert isinstance(result, CrossSessionComparison)

    def test_paired_t_n_correct(self):
        a = _make_session([-0.1, -0.5, -1.0, -2.0])
        b = _make_session([-0.1, -0.5, -1.0, -2.0])
        result = compare_sessions(a, b)
        assert result.paired_t_n == 4

    def test_warnings_list_exists(self):
        a = _make_session([-0.1, -0.5, -1.0, -2.0])
        b = _make_session([-0.1, -0.5, -1.0, -2.0])
        result = compare_sessions(a, b)
        assert isinstance(result.warnings, list)

    def test_insufficient_data_gives_warning(self):
        a = _make_session([-0.1, -0.5])  # n=2, too few
        b = _make_session([-0.1, -0.5])
        result = compare_sessions(a, b)
        assert any("Insufficient" in w for w in result.warnings)

    def test_rsd_non_negative(self):
        a = _make_session([-0.1, -0.5, -1.0, -2.0, -5.0])
        b = _make_session([-0.15, -0.55, -1.05, -2.05, -5.05])
        result = compare_sessions(a, b)
        if result.reproducibility_rsd_pct is not None:
            assert result.reproducibility_rsd_pct >= 0.0

    def test_mann_whitney_populated_for_adequate_n(self):
        a = _make_session([-0.1, -0.5, -1.0, -2.0])
        b = _make_session([-0.1, -0.5, -1.0, -2.0])
        result = compare_sessions(a, b)
        assert result.mw_u_statistic is not None


# ---------------------------------------------------------------------------
# compare_lod_series
# ---------------------------------------------------------------------------

class TestCompareLodSeries:
    def test_returns_dict(self):
        result = compare_lod_series([0.05, 0.06, 0.055, 0.058, 0.062])
        assert isinstance(result, dict)

    def test_mandatory_keys(self):
        keys = {"mean_lod", "std_lod", "rsd_pct", "trend_slope_ppm_per_session",
                "trend_p_value", "drifting", "n_sessions"}
        result = compare_lod_series([0.05, 0.06, 0.055])
        assert keys <= set(result.keys())

    def test_stable_lod_not_drifting(self):
        stable = [0.05] * 6
        result = compare_lod_series(stable)
        assert result["drifting"] is False

    def test_degrading_lod_detected(self):
        degrading = [0.05 + i * 0.01 for i in range(8)]  # monotone increase
        result = compare_lod_series(degrading)
        assert result["drifting"] is True
        assert result["trend_slope_ppm_per_session"] > 0

    def test_too_few_sessions_returns_error(self):
        result = compare_lod_series([0.05, 0.06])
        assert "error" in result

    def test_rsd_pct_positive(self):
        result = compare_lod_series([0.05, 0.06, 0.055, 0.058])
        assert result["rsd_pct"] >= 0.0

    def test_n_sessions_matches_input(self):
        lods = [0.05, 0.06, 0.055, 0.058, 0.062]
        result = compare_lod_series(lods)
        assert result["n_sessions"] == 5


# ---------------------------------------------------------------------------
# Mann-Kendall extension to compare_lod_series
# ---------------------------------------------------------------------------

class TestCompareLodSeriesMannKendall:
    """Mann-Kendall non-parametric trend test in compare_lod_series."""

    def test_mann_kendall_keys_present(self):
        result = compare_lod_series([0.05, 0.06, 0.055, 0.058, 0.062])
        assert "mann_kendall_tau" in result
        assert "mann_kendall_p_value" in result
        assert "mann_kendall_trend" in result

    def test_mk_no_trend_for_stable_series(self):
        stable = [0.050] * 7
        result = compare_lod_series(stable)
        assert result["mann_kendall_trend"] == "no_significant_trend"

    def test_mk_increasing_trend_detected(self):
        """Monotone increasing series should give MK increasing trend."""
        degrading = [0.05 + i * 0.02 for i in range(8)]
        result = compare_lod_series(degrading)
        assert result["mann_kendall_trend"] == "increasing"
        assert cast(float, result["mann_kendall_tau"]) > 0

    def test_mk_decreasing_trend_detected(self):
        """Monotone decreasing series should give MK decreasing trend."""
        improving = [0.20 - i * 0.02 for i in range(8)]
        result = compare_lod_series(improving)
        assert result["mann_kendall_trend"] == "decreasing"
        assert cast(float, result["mann_kendall_tau"]) < 0

    def test_drifting_true_for_monotone_increase(self):
        degrading = [0.05 + i * 0.01 for i in range(8)]
        result = compare_lod_series(degrading)
        assert result["drifting"] is True

    def test_legacy_trend_p_value_key_present(self):
        """Backwards compatibility: 'trend_p_value' key must still be present."""
        result = compare_lod_series([0.05, 0.06, 0.055])
        assert "trend_p_value" in result

    def test_trend_p_value_equals_ols_p(self):
        """trend_p_value should equal trend_p_value_ols (same key, different names)."""
        result = compare_lod_series([0.05, 0.06, 0.055, 0.058])
        assert result["trend_p_value"] == result["trend_p_value_ols"]

    def test_mk_tau_in_minus_one_to_plus_one(self):
        result = compare_lod_series([0.05, 0.06, 0.055, 0.058, 0.062])
        tau = result["mann_kendall_tau"]
        assert tau is not None
        assert -1.0 <= float(tau) <= 1.0
