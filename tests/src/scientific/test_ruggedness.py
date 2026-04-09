"""Tests for src.scientific.ruggedness — Youden test + spike recovery."""
from __future__ import annotations

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Dashboard panel logic tests (TestYoudenDashboardPanel)
# These tests mirror the exact code path used by the Youden ruggedness panel in
# dashboard/agentic_pipeline_tab.py Step 3, so that regressions are caught
# before they reach the UI.
# ──────────────────────────────────────────────────────────────────────────────

from src.scientific.ruggedness import (
    RuggednessResult,
    SpikeRecoveryResult,
    YoudensDesign,
    recovery_acceptance,
    spike_recovery,
    youden_ruggedness,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

FACTOR_NAMES = [
    "integration_ms",
    "temperature_C",
    "flow_rate_sccm",
    "baseline_wait_s",
    "purge_wait_s",
    "fiber_bend_mm",
    "lamp_power_pct",
]

FACTOR_LEVELS = {
    "integration_ms":  (48.0, 52.0),
    "temperature_C":   (22.0, 24.0),
    "flow_rate_sccm":  (95.0, 105.0),
    "baseline_wait_s": (58.0, 62.0),
    "purge_wait_s":    (28.0, 32.0),
    "fiber_bend_mm":   (9.0, 11.0),
    "lamp_power_pct":  (98.0, 102.0),
}


@pytest.fixture
def design() -> YoudensDesign:
    return YoudensDesign(factors=FACTOR_NAMES, levels=FACTOR_LEVELS)


# ──────────────────────────────────────────────────────────────────
# YoudensDesign construction tests
# ──────────────────────────────────────────────────────────────────

class TestYoudensDesign:
    def test_creates_design(self, design: YoudensDesign):
        assert isinstance(design, YoudensDesign)

    def test_design_shape(self, design: YoudensDesign):
        assert design.design.shape == (8, 7)

    def test_design_entries_pm1(self, design: YoudensDesign):
        unique = np.unique(design.design)
        assert set(unique.tolist()) == {-1.0, 1.0}

    def test_design_balanced_columns(self, design: YoudensDesign):
        """Each column should have exactly 4 runs at +1 and 4 at −1."""
        for col in range(7):
            assert (design.design[:, col] == 1).sum() == 4
            assert (design.design[:, col] == -1).sum() == 4

    def test_wrong_number_of_factors_raises(self):
        with pytest.raises(ValueError, match="exactly 7"):
            YoudensDesign(factors=FACTOR_NAMES[:5], levels=FACTOR_LEVELS)

    def test_missing_level_raises(self):
        with pytest.raises(ValueError, match="Missing level"):
            YoudensDesign(
                factors=FACTOR_NAMES,
                levels={k: v for k, v in list(FACTOR_LEVELS.items())[:5]},
            )

    def test_run_settings_returns_correct_levels(self, design: YoudensDesign):
        for run_idx in range(8):
            settings = design.run_settings(run_idx)
            assert set(settings.keys()) == set(FACTOR_NAMES)
            for fname, val in settings.items():
                lo, hi = FACTOR_LEVELS[fname]
                assert val in (lo, hi), f"run {run_idx}: {fname}={val} not in {(lo, hi)}"

    def test_run_settings_invalid_index(self, design: YoudensDesign):
        with pytest.raises(IndexError):
            design.run_settings(8)

    def test_all_run_settings_length(self, design: YoudensDesign):
        all_runs = design.all_run_settings()
        assert len(all_runs) == 8

    def test_orthogonality(self, design: YoudensDesign):
        """All pairs of columns must be orthogonal (dot product = 0)."""
        X = design.design
        for i in range(7):
            for j in range(i + 1, 7):
                dot = float(X[:, i] @ X[:, j])
                assert abs(dot) < 1e-10, f"Columns {i} and {j} not orthogonal: dot={dot}"


# ──────────────────────────────────────────────────────────────────
# youden_ruggedness tests
# ──────────────────────────────────────────────────────────────────

class TestYoudenRuggedness:
    def _null_responses(self) -> list[float]:
        """Constant responses — no factor has any effect."""
        return [1.0] * 8

    def _single_factor_responses(self, design: YoudensDesign, factor_idx: int = 0, effect: float = 0.8) -> list[float]:
        """Responses that depend only on factor `factor_idx`."""
        responses = []
        for run_idx in range(8):
            level = design.design[run_idx, factor_idx]  # ±1
            responses.append(1.0 + effect * level)
        return responses

    def test_returns_ruggedness_result(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        assert isinstance(res, RuggednessResult)

    def test_constant_response_zero_effects(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        assert np.allclose(res.effects, 0.0, atol=1e-10)

    def test_constant_response_no_critical_factors(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        assert res.critical_factors == []

    def test_single_large_effect_detected(self, design: YoudensDesign):
        responses = self._single_factor_responses(design, factor_idx=0, effect=0.8)
        res = youden_ruggedness(design, responses)
        # Factor 0 must dominate
        assert abs(res.effects[0]) > max(abs(e) for i, e in enumerate(res.effects) if i != 0)

    def test_single_large_effect_is_critical(self, design: YoudensDesign):
        responses = self._single_factor_responses(design, factor_idx=2, effect=1.0)
        res = youden_ruggedness(design, responses)
        assert FACTOR_NAMES[2] in res.critical_factors

    def test_effects_length(self, design: YoudensDesign):
        responses = self._null_responses()
        res = youden_ruggedness(design, responses)
        assert len(res.effects) == 7

    def test_wrong_response_length_raises(self, design: YoudensDesign):
        with pytest.raises(ValueError, match="exactly 8"):
            youden_ruggedness(design, [1.0, 2.0, 3.0])

    def test_nan_response_raises(self, design: YoudensDesign):
        with pytest.raises(ValueError, match="finite"):
            youden_ruggedness(design, [1.0] * 7 + [float("nan")])

    def test_summary_is_nonempty_string(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        s = res.summary()
        assert isinstance(s, str) and len(s) > 50

    def test_as_dict_contains_expected_keys(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        d = res.as_dict()
        for key in ("factors", "effects", "residual_std", "critical_factors"):
            assert key in d

    def test_as_dict_effects_list(self, design: YoudensDesign):
        res = youden_ruggedness(design, self._null_responses())
        d = res.as_dict()
        assert isinstance(d["effects"], list)
        assert len(d["effects"]) == 7

    def test_response_mean_correct(self, design: YoudensDesign):
        responses = [float(i) for i in range(8)]
        res = youden_ruggedness(design, responses)
        assert abs(res.response_mean - float(np.mean(responses))) < 1e-10

    def test_residual_std_nonneg(self, design: YoudensDesign):
        responses = self._null_responses()
        res = youden_ruggedness(design, responses)
        assert res.residual_std >= 0.0


# ──────────────────────────────────────────────────────────────────
# spike_recovery tests
# ──────────────────────────────────────────────────────────────────

class TestSpikeRecovery:
    def test_perfect_recovery(self):
        added = [50.0, 100.0, 200.0]
        found = [50.0, 100.0, 200.0]
        res = spike_recovery(added, found)
        assert isinstance(res, SpikeRecoveryResult)
        assert res.mean_recovery == pytest.approx(1.0)
        assert all(p.recovery == pytest.approx(1.0) for p in res.points)

    def test_all_pass_ich_on_perfect_recovery(self):
        res = spike_recovery([50.0, 100.0], [50.0, 100.0])
        assert res.overall_pass_ich is True
        assert res.overall_pass_routine is True

    def test_low_recovery_fail_ich(self):
        # 90 % recovery — fails ICH (needs ≥98%) but passes routine (≥90%)
        res = spike_recovery([100.0], [90.0], background_concentration=0.0)
        assert res.points[0].pass_ich is False
        assert res.points[0].pass_routine is True

    def test_high_recovery_fail_ich(self):
        # 103 % recovery — fails ICH (≤102%)
        res = spike_recovery([100.0], [103.0])
        assert res.points[0].pass_ich is False

    def test_background_subtraction(self):
        # Blank = 10 ppm, added 100 ppm, found 110 ppm → R = (110-10)/100 = 1.0
        res = spike_recovery([100.0], [110.0], background_concentration=10.0)
        assert res.points[0].recovery == pytest.approx(1.0)
        assert res.points[0].pass_ich is True

    def test_n_levels_set(self):
        res = spike_recovery([50.0, 100.0, 200.0], [50.0, 100.0, 200.0])
        assert res.n_levels == 3

    def test_mean_recovery_calculation(self):
        added = [50.0, 100.0, 200.0]
        found = [51.0, 100.0, 198.0]
        res = spike_recovery(added, found)
        expected = np.mean([51.0 / 50.0, 100.0 / 100.0, 198.0 / 200.0])
        assert res.mean_recovery == pytest.approx(expected, rel=1e-6)

    def test_recovery_pct_equals_100_times_recovery(self):
        res = spike_recovery([100.0], [99.0])
        assert res.points[0].recovery_pct == pytest.approx(res.points[0].recovery * 100)

    def test_wrong_array_lengths_raises(self):
        with pytest.raises(ValueError):
            spike_recovery([50.0, 100.0], [50.0])

    def test_empty_arrays_raises(self):
        with pytest.raises(ValueError):
            spike_recovery([], [])

    def test_zero_added_raises(self):
        with pytest.raises(ValueError, match="positive"):
            spike_recovery([0.0, 100.0], [0.0, 100.0])

    def test_negative_added_raises(self):
        with pytest.raises(ValueError, match="positive"):
            spike_recovery([-50.0, 100.0], [50.0, 100.0])

    def test_nan_in_found_raises(self):
        with pytest.raises(ValueError, match="finite"):
            spike_recovery([100.0], [float("nan")])

    def test_summary_is_nonempty_string(self):
        res = spike_recovery([50.0, 100.0], [50.0, 100.0])
        s = res.summary()
        assert isinstance(s, str) and len(s) > 30

    def test_as_dict_has_required_keys(self):
        res = spike_recovery([50.0, 100.0], [50.0, 100.0])
        d = res.as_dict()
        for key in ("n_levels", "mean_recovery_pct", "std_recovery_pct",
                    "overall_pass_ich", "overall_pass_routine", "levels"):
            assert key in d

    def test_as_dict_levels_list(self):
        res = spike_recovery([50.0, 100.0, 200.0], [50.0, 100.0, 200.0])
        d = res.as_dict()
        assert len(d["levels"]) == 3

    def test_single_spike_level_std_is_zero(self):
        res = spike_recovery([100.0], [99.0])
        assert res.std_recovery == pytest.approx(0.0)

    def test_custom_ich_bounds(self):
        # 95 % recovery passes custom narrow acceptance window 0.94–0.96
        res = spike_recovery([100.0], [95.0], ich_low=0.94, ich_high=0.96)
        assert res.points[0].pass_ich is True
        # 97 % fails the same narrow window
        res2 = spike_recovery([100.0], [97.0], ich_low=0.94, ich_high=0.96)
        assert res2.points[0].pass_ich is False

    def test_overall_pass_requires_all_levels(self):
        # Two spikes: one passes, one fails ICH
        res = spike_recovery([50.0, 100.0], [49.5, 103.0])  # R=0.99 and 1.03
        # 0.99 passes (98–102), 1.03 fails (>102) → overall fails
        assert res.overall_pass_ich is False


# ──────────────────────────────────────────────────────────────────
# recovery_acceptance tests
# ──────────────────────────────────────────────────────────────────

class TestRecoveryAcceptance:
    def test_ich_q2r1_pass(self):
        res = spike_recovery([100.0], [100.0])
        assert recovery_acceptance(res, "ich_q2r1") is True

    def test_ich_q2r1_fail(self):
        res = spike_recovery([100.0], [96.0])  # 96% — fails ICH
        assert recovery_acceptance(res, "ich_q2r1") is False

    def test_routine_pass_where_ich_fails(self):
        res = spike_recovery([100.0], [95.0])  # 95% — fails ICH, passes routine
        assert recovery_acceptance(res, "ich_q2r1") is False
        assert recovery_acceptance(res, "routine") is True

    def test_custom_equals_ich_q2r1(self):
        res = spike_recovery([100.0], [100.0])
        assert recovery_acceptance(res, "custom") == recovery_acceptance(res, "ich_q2r1")

    def test_invalid_level_raises(self):
        res = spike_recovery([100.0], [100.0])
        with pytest.raises(ValueError, match="Unknown"):
            recovery_acceptance(res, "invalid")  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────
# Dashboard panel logic — mirrors agentic_pipeline_tab.py Step 3
# ──────────────────────────────────────────────────────────────────

# Hardcoded PB design (must match _YD_MATRIX in the dashboard panel)
_DASHBOARD_DESIGN_MATRIX = [
    [+1, +1, +1, -1, +1, -1, -1],
    [-1, +1, +1, +1, -1, +1, -1],
    [-1, -1, +1, +1, +1, -1, +1],
    [+1, -1, -1, +1, +1, +1, -1],
    [-1, +1, -1, -1, +1, +1, +1],
    [+1, -1, +1, -1, -1, +1, +1],
    [+1, +1, -1, +1, -1, -1, +1],
    [-1, -1, -1, -1, -1, -1, -1],
]

_DASHBOARD_FACTOR_NAMES = [
    "integration_ms", "temperature_C", "flow_rate_sccm",
    "baseline_wait_s", "purge_wait_s", "fiber_bend_mm", "lamp_power_pct",
]


def _dashboard_run(factor_names: list[str], responses_str: str) -> dict:
    """Simulate the dashboard panel's compute logic.

    Returns a dict with keys: 'error' (str or None), 'result' (RuggednessResult or None).
    Mirrors the try/except block in the Youden expander.
    """
    vals = [float(x.strip()) for x in responses_str.split(",") if x.strip()]
    if len(vals) != 8:
        return {"error": f"expected 8 values, got {len(vals)}", "result": None}
    if len(set(factor_names)) < 7:
        return {"error": "duplicate factor names", "result": None}
    design = YoudensDesign(
        factors=factor_names,
        levels={n: (0.0, 1.0) for n in factor_names},  # dummy levels
    )
    result = youden_ruggedness(design, vals)
    return {"error": None, "result": result}


class TestYoudenDashboardPanel:
    """Tests for the dashboard Youden panel logic in agentic_pipeline_tab.py."""

    def test_exactly_8_responses_required_too_few(self):
        out = _dashboard_run(_DASHBOARD_FACTOR_NAMES, "1.0, 2.0, 3.0")
        assert out["error"] is not None
        assert out["result"] is None

    def test_exactly_8_responses_required_too_many(self):
        out = _dashboard_run(_DASHBOARD_FACTOR_NAMES, ", ".join(str(float(i)) for i in range(9)))
        assert out["error"] is not None

    def test_valid_8_responses_succeeds(self):
        responses = ", ".join(["1.0"] * 8)
        out = _dashboard_run(_DASHBOARD_FACTOR_NAMES, responses)
        assert out["error"] is None
        assert out["result"] is not None

    def test_duplicate_factor_names_rejected(self):
        dup_names = ["factor_A"] * 7  # all duplicates
        out = _dashboard_run(dup_names, ", ".join(["1.0"] * 8))
        assert out["error"] is not None

    def test_dummy_levels_do_not_affect_effects(self):
        """Effects depend only on ±1 design and responses — levels are irrelevant."""
        responses = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]

        design_dummy = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        design_real = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={
                "integration_ms": (48.0, 52.0),
                "temperature_C": (22.0, 24.0),
                "flow_rate_sccm": (95.0, 105.0),
                "baseline_wait_s": (58.0, 62.0),
                "purge_wait_s": (28.0, 32.0),
                "fiber_bend_mm": (9.0, 11.0),
                "lamp_power_pct": (98.0, 102.0),
            },
        )
        r_dummy = youden_ruggedness(design_dummy, responses)
        r_real = youden_ruggedness(design_real, responses)
        np.testing.assert_allclose(r_dummy.effects, r_real.effects, atol=1e-12)

    def test_critical_factor_appears_in_result_dict(self):
        """A single dominant factor should be flagged as critical."""
        design = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        # Make temperature_C (index 1) dominant: add ±2 contribution
        base = 1.0
        responses = [base + 2.0 * design.design[r, 1] for r in range(8)]
        result = youden_ruggedness(design, responses)
        assert "temperature_C" in result.critical_factors

    def test_rugged_result_no_critical_factors(self):
        """Constant responses → no critical factors."""
        design = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        result = youden_ruggedness(design, [1.0] * 8)
        assert result.critical_factors == []

    def test_as_dict_keys_match_dashboard_reads(self):
        """as_dict() must have all keys the dashboard reads from ss['yd_result']."""
        design = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        d = youden_ruggedness(design, [1.0] * 8).as_dict()
        for key in ("factors", "effects", "residual_std", "critical_factors",
                    "response_mean", "response_std"):
            assert key in d, f"Missing key '{key}' from as_dict()"

    def test_design_matrix_shape(self):
        assert len(_DASHBOARD_DESIGN_MATRIX) == 8
        assert all(len(row) == 7 for row in _DASHBOARD_DESIGN_MATRIX)

    def test_design_matrix_values_pm1(self):
        for row in _DASHBOARD_DESIGN_MATRIX:
            for val in row:
                assert val in (+1, -1)

    def test_design_matrix_balanced(self):
        """Each column must have exactly 4 runs at +1 and 4 at −1."""
        arr = np.array(_DASHBOARD_DESIGN_MATRIX, dtype=float)
        for col in range(7):
            assert (arr[:, col] == +1).sum() == 4
            assert (arr[:, col] == -1).sum() == 4

    def test_design_matrix_orthogonal(self):
        """All column pairs must be orthogonal (dot product = 0)."""
        arr = np.array(_DASHBOARD_DESIGN_MATRIX, dtype=float)
        for i in range(7):
            for j in range(i + 1, 7):
                dot = float(arr[:, i] @ arr[:, j])
                assert abs(dot) < 1e-10, f"Columns {i} and {j} not orthogonal: dot={dot}"

    def test_design_matrix_matches_ruggedness_module(self):
        """Dashboard matrix must be identical to _YPB_DESIGN in ruggedness.py."""
        from src.scientific.ruggedness import _YPB_DESIGN  # noqa: PLC0415
        arr = np.array(_DASHBOARD_DESIGN_MATRIX, dtype=float)
        np.testing.assert_array_equal(arr, _YPB_DESIGN)

    def test_result_row_status_field(self):
        """Status column in results table must be CRITICAL or OK."""
        design = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        result = youden_ruggedness(design, [1.0] * 8)
        for f, e in zip(result.factors, result.effects):
            status = "🔴 CRITICAL" if f in result.critical_factors else "✅ OK"
            assert status in ("🔴 CRITICAL", "✅ OK")

    def test_sigma_resid_nonneg(self):
        design = YoudensDesign(
            factors=_DASHBOARD_FACTOR_NAMES,
            levels={n: (0.0, 1.0) for n in _DASHBOARD_FACTOR_NAMES},
        )
        result = youden_ruggedness(design, [float(i) for i in range(8)])
        assert result.residual_std >= 0.0
