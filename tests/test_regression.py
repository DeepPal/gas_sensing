"""Tests for src.scientific.regression — robust linear regression utilities."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.scientific.regression import ransac, theil_sen, weighted_linear


def _linear(x: np.ndarray, slope: float, intercept: float, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    return slope * x + intercept + rng.normal(0, noise, x.size)


class TestWeightedLinear:
    def test_perfect_line_recovered(self):
        x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        y = 3.0 * x + 0.5
        w = np.ones_like(x)
        result = weighted_linear(x, y, w)
        assert result is not None
        assert result["model"] == "weighted_ols"
        assert abs(cast(float, result["slope"]) - 3.0) < 0.01
        assert abs(cast(float, result["intercept"]) - 0.5) < 0.01

    def test_too_few_points_returns_none(self):
        x = np.array([1.0])
        y = np.array([2.0])
        w = np.array([1.0])
        assert weighted_linear(x, y, w) is None

    def test_zero_weight_excluded(self):
        x = np.array([0.0, 1.0, 2.0, 100.0])
        y = np.array([0.0, 1.0, 2.0, -999.0])
        w = np.array([1.0, 1.0, 1.0, 0.0])  # outlier has zero weight
        result = weighted_linear(x, y, w)
        assert result is not None
        assert cast(float, result["slope"]) > 0

    def test_wrong_weight_size_returns_none(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        w = np.array([1.0])  # mismatched
        assert weighted_linear(x, y, w) is None

    def test_r2_is_high_for_linear_data(self):
        x = np.linspace(0, 10, 20)
        y = _linear(x, slope=2.0, intercept=1.0, noise=0.01)
        w = np.ones_like(x)
        result = weighted_linear(x, y, w)
        assert result is not None
        assert cast(float, result["r2"]) > 0.99


class TestTheilSen:
    def test_clean_line_recovered(self):
        x = np.linspace(0, 5, 20)
        y = _linear(x, slope=1.5, intercept=0.2, noise=0.02)
        result = theil_sen(x, y)
        assert result is not None
        assert result["model"] == "theil_sen"
        assert abs(cast(float, result["slope"]) - 1.5) < 0.1

    def test_robust_to_single_outlier(self):
        x = np.linspace(0.1, 5, 15)
        y = 2.0 * x + 1.0
        y_outlier = y.copy()
        y_outlier[7] = 500.0  # extreme outlier
        result = theil_sen(x, y_outlier)
        assert result is not None
        assert abs(cast(float, result["slope"]) - 2.0) < 0.5

    def test_single_point_returns_none(self):
        assert theil_sen(np.array([1.0]), np.array([2.0])) is None

    def test_r2_reasonable(self):
        x = np.linspace(0, 10, 30)
        y = _linear(x, slope=3.0, intercept=0.0, noise=0.1)
        result = theil_sen(x, y)
        assert result is not None
        assert cast(float, result["r2"]) > 0.95


class TestRansac:
    def test_clean_line_recovered(self):
        x = np.linspace(0, 5, 20)
        y = _linear(x, slope=2.0, intercept=0.5, noise=0.05)
        result = ransac(x, y)
        assert result is not None
        assert result["model"] == "ransac"
        assert abs(cast(float, result["slope"]) - 2.0) < 0.2

    def test_robust_to_outliers(self):
        x = np.linspace(0.1, 5, 20)
        y = 1.5 * x + 0.3
        y_outlier = y.copy()
        y_outlier[[0, 1, 2]] = -50.0  # gross outliers
        result = ransac(x, y_outlier)
        assert result is not None
        assert abs(cast(float, result["slope"]) - 1.5) < 0.5

    def test_too_few_points_returns_none(self):
        assert ransac(np.array([1.0, 2.0]), np.array([1.0, 2.0])) is None

    def test_rmse_is_finite(self):
        x = np.linspace(0, 10, 15)
        y = _linear(x, slope=1.0, intercept=0.0, noise=0.2)
        result = ransac(x, y)
        assert result is not None
        assert np.isfinite(result["rmse"])


class TestRegressionKeys:
    """All three estimators must return the same key set."""

    REQUIRED_KEYS = {"model", "slope", "intercept", "r2", "rmse",
                     "slope_stderr", "slope_ci_low", "slope_ci_high"}

    def _xy(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.1, 5, 20)
        return x, 2.0 * x + 1.0

    def test_weighted_linear_keys(self):
        x, y = self._xy()
        result = weighted_linear(x, y, np.ones_like(x))
        assert result is not None
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_theil_sen_keys(self):
        x, y = self._xy()
        result = theil_sen(x, y)
        assert result is not None
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_ransac_keys(self):
        x, y = self._xy()
        result = ransac(x, y)
        assert result is not None
        assert self.REQUIRED_KEYS.issubset(result.keys())
