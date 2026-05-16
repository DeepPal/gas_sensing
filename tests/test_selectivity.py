"""
tests.test_selectivity
=======================
Unit tests for src.scientific.selectivity:
  - compute_cross_sensitivity
  - selectivity_matrix
  - selectivity_from_calibration_data
  - SelectivityResult (summary_table, to_dict)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.scientific.selectivity import (
    SelectivityResult,
    compute_cross_sensitivity,
    selectivity_from_calibration_data,
    selectivity_matrix,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _three_gas_sensitivities():
    """Ethanol, IPA, Methanol with realistic LSPR slopes (Δλ/ppm)."""
    return {
        "Ethanol": -2.10,
        "IPA": -1.80,
        "Methanol": -0.95,
    }


def _calibration_data():
    concs = np.array([0.5, 1.0, 2.0, 5.0])
    return {
        "Ethanol": (concs, -2.10 * concs),
        "IPA": (concs, -1.80 * concs),
        "Methanol": (concs, -0.95 * concs),
    }


# ---------------------------------------------------------------------------
# compute_cross_sensitivity
# ---------------------------------------------------------------------------


class TestComputeCrossSensitivity:
    def test_returns_dict(self):
        k = compute_cross_sensitivity(
            target_sensitivity=-2.0,
            interferent_sensitivities={"IPA": -1.8, "Methanol": -0.9},
        )
        assert isinstance(k, dict)

    def test_correct_k_values(self):
        k = compute_cross_sensitivity(
            target_sensitivity=-2.0,
            interferent_sensitivities={"IPA": -1.0},
        )
        assert k["IPA"] == pytest.approx(0.5, rel=1e-6)

    def test_negative_k_for_opposite_sign(self):
        k = compute_cross_sensitivity(
            target_sensitivity=-2.0,
            interferent_sensitivities={"Noise": 1.0},
        )
        assert k["Noise"] < 0.0

    def test_same_sensitivity_gives_one(self):
        k = compute_cross_sensitivity(
            target_sensitivity=-2.0,
            interferent_sensitivities={"Twin": -2.0},
        )
        assert k["Twin"] == pytest.approx(1.0, rel=1e-9)

    def test_zero_sensitivity_raises(self):
        with pytest.raises(ValueError, match="zero"):
            compute_cross_sensitivity(0.0, {"IPA": -1.0})

    def test_empty_interferents(self):
        k = compute_cross_sensitivity(-2.0, {})
        assert k == {}


# ---------------------------------------------------------------------------
# selectivity_matrix
# ---------------------------------------------------------------------------


class TestSelectivityMatrix:
    def test_returns_selectivity_result(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        assert isinstance(result, SelectivityResult)

    def test_matrix_shape(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        n = len(_three_gas_sensitivities())
        assert result.matrix.shape == (n, n)

    def test_diagonal_is_one(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        np.testing.assert_allclose(np.diag(result.matrix), 1.0, rtol=1e-9)

    def test_k_values_correct(self):
        """K[Ethanol, IPA] = S_IPA / S_Ethanol = -1.80 / -2.10."""
        result = selectivity_matrix(_three_gas_sensitivities())
        gases = result.gases
        i_eth = gases.index("Ethanol")
        i_ipa = gases.index("IPA")
        expected = -1.80 / -2.10
        assert result.matrix[i_eth, i_ipa] == pytest.approx(expected, rel=1e-4)

    def test_asymmetry(self):
        """K[i,j] ≠ K[j,i] in general (unless sensitivities are equal)."""
        result = selectivity_matrix(_three_gas_sensitivities())
        assert result.matrix[0, 1] != result.matrix[1, 0]

    def test_rankings_sorted_descending(self):
        """Interferents should be ranked by |K| descending."""
        result = selectivity_matrix(_three_gas_sensitivities())
        for ranks in result.rankings.values():
            abs_k = [abs(k) for _, k in ranks]
            assert abs_k == sorted(abs_k, reverse=True)

    def test_all_gases_in_rankings(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        gases = set(result.gases)
        for tgt, ranks in result.rankings.items():
            ranked_names = {name for name, _ in ranks}
            assert ranked_names == gases - {tgt}

    def test_worst_interferent_per_target(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        for tgt, (worst_name, _worst_k) in result.worst_interferents.items():
            # worst_name should be at top of ranking
            assert result.rankings[tgt][0][0] == worst_name

    def test_interpretation_is_string(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        for v in result.interpretation.values():
            assert isinstance(v, str) and len(v) > 0

    def test_too_few_gases_raises(self):
        with pytest.raises(ValueError, match="2"):
            selectivity_matrix({"Ethanol": -2.0})

    def test_to_dict_serialisable(self):
        import json

        result = selectivity_matrix(_three_gas_sensitivities())
        d = result.to_dict()
        # Must be JSON-serialisable (no numpy arrays)
        json.dumps(d)

    def test_summary_table_contains_all_gases(self):
        result = selectivity_matrix(_three_gas_sensitivities())
        table = result.summary_table()
        for gas in result.gases:
            assert gas in table


# ---------------------------------------------------------------------------
# selectivity_from_calibration_data
# ---------------------------------------------------------------------------


class TestSelectivityFromCalibrationData:
    def test_returns_selectivity_result(self):
        result = selectivity_from_calibration_data(_calibration_data())
        assert isinstance(result, SelectivityResult)

    def test_sensitivities_match_slopes(self):
        """Sensitivities should equal the known linear slopes."""
        result = selectivity_from_calibration_data(_calibration_data())
        assert result.sensitivities["Ethanol"] == pytest.approx(-2.10, rel=1e-3)
        assert result.sensitivities["IPA"] == pytest.approx(-1.80, rel=1e-3)
        assert result.sensitivities["Methanol"] == pytest.approx(-0.95, rel=1e-3)

    def test_diagonal_still_one(self):
        result = selectivity_from_calibration_data(_calibration_data())
        np.testing.assert_allclose(np.diag(result.matrix), 1.0, rtol=1e-9)

    def test_huber_regression_produces_result(self):
        result = selectivity_from_calibration_data(_calibration_data(), regression="huber")
        assert isinstance(result, SelectivityResult)
        assert len(result.gases) == 3

    def test_unknown_regression_raises(self):
        with pytest.raises(ValueError, match="regression"):
            selectivity_from_calibration_data(_calibration_data(), regression="nonsense")

    def test_single_point_raises(self):
        data = {
            "A": (np.array([1.0]), np.array([-2.0])),
            "B": (np.array([1.0, 2.0]), np.array([-1.0, -2.0])),
        }
        with pytest.raises(ValueError, match="2"):
            selectivity_from_calibration_data(data)
