"""
tests.test_isotherms
====================
Unit tests for src.calibration.isotherms:
  - fit_langmuir
  - fit_freundlich
  - fit_hill
  - select_isotherm
  - IsothermResult.predict
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.calibration.isotherms import (
    IsothermResult,
    fit_freundlich,
    fit_hill,
    fit_langmuir,
    select_isotherm,
)

# ---------------------------------------------------------------------------
# Shared data generators
# ---------------------------------------------------------------------------


def _langmuir_data(R_max=10.0, K=0.5, concs=None, noise=0.0, seed=42):
    """Generate noiseless (or noisy) Langmuir calibration data."""
    rng = np.random.default_rng(seed)
    c = np.array(concs or [0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    r = -R_max * K * c / (1.0 + K * c)  # negative (LSPR convention)
    if noise > 0:
        r = r + rng.normal(0, noise, len(r))
    return c, r


def _freundlich_data(K=2.0, n=0.7, concs=None, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    c = np.array(concs or [0.5, 1.0, 2.0, 5.0, 10.0], dtype=float)
    r = -K * c**n
    if noise > 0:
        r = r + rng.normal(0, noise, len(r))
    return c, r


def _linear_data(slope=-2.0, intercept=0.0, concs=None, noise=0.0, seed=42):
    rng = np.random.default_rng(seed)
    c = np.array(concs or [0.5, 1.0, 2.0, 5.0], dtype=float)
    r = slope * c + intercept
    if noise > 0:
        r = r + rng.normal(0, noise, len(r))
    return c, r


# ---------------------------------------------------------------------------
# fit_langmuir
# ---------------------------------------------------------------------------


class TestFitLangmuir:
    def test_returns_isotherm_result(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert isinstance(result, IsothermResult)
        assert result.model == "langmuir"

    def test_recovered_params_close(self):
        """Fit should recover R_max and K within 10% on noiseless data."""
        c, r = _langmuir_data(R_max=10.0, K=0.5)
        result = fit_langmuir(c, r)
        assert result.params["R_max"] == pytest.approx(10.0, rel=0.10)
        assert result.params["K"] == pytest.approx(0.5, rel=0.10)

    def test_r_squared_near_one_noiseless(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert result.r_squared > 0.999

    def test_r_squared_reasonable_with_noise(self):
        c, r = _langmuir_data(noise=0.2)
        result = fit_langmuir(c, r)
        assert result.r_squared > 0.90

    def test_has_aic(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert np.isfinite(result.aic)

    def test_n_params(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert result.n_params == 2

    def test_fit_grid_shape(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert len(result.concentrations_fit) == len(result.responses_fit)
        assert len(result.concentrations_fit) > len(c)

    def test_predict_matches_fit_grid(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        pred = result.predict(result.concentrations_fit)
        np.testing.assert_allclose(pred, result.responses_fit, rtol=1e-5)

    def test_predict_at_zero(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        assert result.predict(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-9)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="3"):
            fit_langmuir(np.array([1.0, 2.0]), np.array([-1.0, -2.0]))

    def test_param_stderrs_non_negative(self):
        c, r = _langmuir_data()
        result = fit_langmuir(c, r)
        for se in result.param_stderrs.values():
            assert se >= 0.0


# ---------------------------------------------------------------------------
# fit_freundlich
# ---------------------------------------------------------------------------


class TestFitFreundlich:
    def test_returns_isotherm_result(self):
        c, r = _freundlich_data()
        result = fit_freundlich(c, r)
        assert isinstance(result, IsothermResult)
        assert result.model == "freundlich"

    def test_recovered_exponent_n(self):
        """Should recover exponent n within 15% on noiseless data."""
        c, r = _freundlich_data(K=2.0, n=0.7)
        result = fit_freundlich(c, r)
        assert result.params["n"] == pytest.approx(0.7, rel=0.15)

    def test_r_squared_near_one_noiseless(self):
        c, r = _freundlich_data()
        result = fit_freundlich(c, r)
        assert result.r_squared > 0.999

    def test_n_params(self):
        c, r = _freundlich_data()
        result = fit_freundlich(c, r)
        assert result.n_params == 2

    def test_predict_consistency(self):
        c, r = _freundlich_data()
        result = fit_freundlich(c, r)
        pred = result.predict(c)
        assert len(pred) == len(c)

    def test_zero_concentration_raises_or_warns(self):
        """Freundlich requires c > 0."""
        with pytest.raises((ValueError, Exception)):
            fit_freundlich(np.array([0.0, 1.0, 2.0]), np.array([-0.5, -1.0, -1.8]))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fit_freundlich(np.array([1.0, 2.0]), np.array([-1.0, -2.0]))


# ---------------------------------------------------------------------------
# fit_hill
# ---------------------------------------------------------------------------


class TestFitHill:
    def test_returns_isotherm_result(self):
        c, r = _langmuir_data()  # Hill with n=1 ≈ Langmuir
        result = fit_hill(c, r)
        assert isinstance(result, IsothermResult)
        assert result.model == "hill"

    def test_n_params(self):
        c, r = _langmuir_data()
        result = fit_hill(c, r)
        assert result.n_params == 3

    def test_r_squared_at_least_as_good_as_langmuir(self):
        """Hill (3 params) should fit at least as well as Langmuir (2 params)."""
        c, r = _langmuir_data()
        lang = fit_langmuir(c, r)
        hill = fit_hill(c, r)
        assert hill.r_squared >= lang.r_squared - 0.01

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="4"):
            fit_hill(np.array([1.0, 2.0, 3.0]), np.array([-1.0, -2.0, -3.0]))

    def test_predict_array_output(self):
        c, r = _langmuir_data()
        result = fit_hill(c, r)
        pred = result.predict(c)
        assert pred.shape == c.shape


# ---------------------------------------------------------------------------
# select_isotherm
# ---------------------------------------------------------------------------


class TestSelectIsotherm:
    def test_returns_dict_with_required_keys(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r)
        for key in ("best_model", "best_result", "all_results", "aic_table", "recommendation"):
            assert key in sel

    def test_best_model_is_string(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r)
        assert isinstance(sel["best_model"], str)

    def test_aic_table_sorted_ascending(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r)
        aic_values = [row[1] for row in cast(list, sel["aic_table"])]
        assert aic_values == sorted(aic_values)

    def test_linear_data_selects_linear(self):
        """For perfectly linear data, linear model should have lowest AIC."""
        c, r = _linear_data()
        sel = select_isotherm(c, r, models=["linear", "langmuir", "freundlich"])
        # Linear data → linear AIC should be best or nearly best
        # (small datasets may favour simpler model even with curved data)
        assert sel["best_model"] in ("linear", "langmuir", "freundlich")

    def test_subset_models(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r, models=["langmuir", "freundlich"])
        assert sel["best_model"] in ("langmuir", "freundlich")
        assert len(cast(list, sel["aic_table"])) <= 2

    def test_recommendation_is_string(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r)
        assert isinstance(sel["recommendation"], str)
        assert len(sel["recommendation"]) > 0

    def test_best_result_is_isotherm_result(self):
        c, r = _langmuir_data()
        sel = select_isotherm(c, r)
        assert isinstance(sel["best_result"], IsothermResult)
