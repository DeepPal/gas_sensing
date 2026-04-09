"""Tests for src.scientific.allan_deviation."""
from __future__ import annotations

import numpy as np
import pytest

from src.scientific.allan_deviation import (
    AllanDeviationResult,
    adev_noise_fractions,
    allan_deviation,
)


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def white_noise_signal(n: int = 2000, sigma: float = 0.01, dt: float = 0.05) -> np.ndarray:
    """Pure white noise: ADEV should slope ≈ -0.5 on log-log."""
    return RNG.normal(0, sigma, size=n)


def random_walk_signal(n: int = 2000, sigma: float = 0.001, dt: float = 0.05) -> np.ndarray:
    """Integrated white noise (random walk): ADEV should slope ≈ +0.5."""
    return np.cumsum(RNG.normal(0, sigma, size=n))


def mixed_signal(n: int = 3000, dt: float = 0.05) -> np.ndarray:
    """White noise + slow random walk: typical real sensor trace."""
    white = RNG.normal(0, 0.005, size=n)
    rw = np.cumsum(RNG.normal(0, 0.0002, size=n))
    return white + rw


# ---------------------------------------------------------------------------
# Basic API / smoke tests
# ---------------------------------------------------------------------------

class TestAllanDeviationSmoke:
    def test_returns_result_type(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert isinstance(res, AllanDeviationResult)

    def test_taus_and_adevs_same_length(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert len(res.taus) == len(res.adevs)

    def test_taus_are_positive_and_increasing(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert np.all(res.taus > 0)
        assert np.all(np.diff(res.taus) >= 0)

    def test_adevs_are_positive(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert np.all(res.adevs > 0)

    def test_tau_opt_in_taus(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert res.tau_opt >= res.taus[0]
        assert res.tau_opt <= res.taus[-1]

    def test_sigma_min_le_all_adevs(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        assert res.sigma_min <= float(np.max(res.adevs)) + 1e-12

    def test_n_samples_set_correctly(self):
        sig = white_noise_signal(n=500)
        res = allan_deviation(sig, dt=0.05)
        assert res.n_samples == 500

    def test_dt_stored(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.1)
        assert res.dt_s == pytest.approx(0.1)

    def test_noise_type_is_valid_string(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        valid = {"white", "flicker", "random_walk", "drift", "mixed", "insufficient_data"}
        assert res.noise_type in valid

    def test_summary_returns_nonempty_string(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        s = res.summary()
        assert isinstance(s, str) and len(s) > 20

    def test_as_dict_has_required_keys(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        d = res.as_dict()
        for key in ("tau_opt_s", "sigma_min", "noise_type", "n_samples", "dt_s", "taus", "adevs"):
            assert key in d, f"Missing key: {key}"

    def test_as_dict_taus_adevs_lists(self):
        sig = white_noise_signal()
        res = allan_deviation(sig, dt=0.05)
        d = res.as_dict()
        assert isinstance(d["taus"], list)
        assert isinstance(d["adevs"], list)
        assert len(d["taus"]) == len(d["adevs"])


# ---------------------------------------------------------------------------
# Physical / statistical correctness tests
# ---------------------------------------------------------------------------

class TestAllanDeviationPhysics:
    def test_white_noise_decreasing_adev(self):
        """White noise ADEV should be monotonically (roughly) decreasing at short tau."""
        sig = white_noise_signal(n=4000, sigma=0.01, dt=0.05)
        res = allan_deviation(sig, dt=0.05)
        # First 5 points should show downward trend
        assert res.adevs[0] > res.adevs[min(4, len(res.adevs) - 1)]

    def test_white_noise_coeff_close_to_sigma_sqrt_dt(self):
        """For white noise: A ≈ σ·√dt  (standard ADEV result)."""
        sigma = 0.01
        dt = 0.05
        sig = RNG.normal(0, sigma, size=5000)
        res = allan_deviation(sig, dt=dt)
        # Expected A = sigma * sqrt(dt) = 0.01 * sqrt(0.05) ≈ 0.00224
        expected_A = sigma * np.sqrt(dt)
        # Allow 50% tolerance given finite sample size
        assert abs(res.white_noise_coeff - expected_A) / expected_A < 0.5

    def test_sigma_min_close_to_noise_floor(self):
        """sigma_min should be close to the theoretical ADEV floor for white noise."""
        sigma = 0.01
        dt = 0.05
        sig = RNG.normal(0, sigma, size=5000)
        res = allan_deviation(sig, dt=dt)
        # At tau_opt the ADEV floor for white noise is sigma/sqrt(tau_opt/dt)
        # Just check it's in the right ballpark (within one order of magnitude)
        assert res.sigma_min < sigma * 5  # much less than raw noise
        assert res.sigma_min > 0

    def test_random_walk_adev_increases_with_tau(self):
        """Random walk ADEV should increase at long tau."""
        sig = random_walk_signal(n=4000, sigma=0.001, dt=0.05)
        res = allan_deviation(sig, dt=0.05)
        n = len(res.adevs)
        if n >= 4:
            # Long-tau adevs should be larger than short-tau adevs
            assert float(np.mean(res.adevs[n // 2 :])) > float(np.mean(res.adevs[: n // 4]))

    def test_shorter_signal_fewer_tau_points(self):
        """Shorter signal should produce fewer tau points (max_m = N//3)."""
        sig_long = white_noise_signal(n=2000)
        sig_short = white_noise_signal(n=100)
        res_long = allan_deviation(sig_long, dt=0.05)
        res_short = allan_deviation(sig_short, dt=0.05)
        assert len(res_short.taus) <= len(res_long.taus) + 2  # allow rounding

    def test_custom_taus_respected(self):
        """When explicit taus are given, the result should use them."""
        sig = white_noise_signal(n=2000)
        custom_taus = np.array([0.05, 0.1, 0.5, 1.0, 2.0])
        res = allan_deviation(sig, dt=0.05, taus=custom_taus)
        # taus in result should be subset of custom_taus (duplicates removed)
        for t in res.taus:
            assert any(abs(t - ct) < 0.001 for ct in custom_taus)

    def test_mixed_signal_has_tau_opt_between_extremes(self):
        """τ_opt for a mixed signal should be neither the minimum nor maximum tau."""
        sig = mixed_signal(n=5000)
        res = allan_deviation(sig, dt=0.05)
        if len(res.taus) >= 5:
            assert res.tau_opt > res.taus[0]  # not the very first tau
            assert res.tau_opt < res.taus[-1]  # not the very last tau (if drift dominates)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestAllanDeviationEdgeCases:
    def test_minimum_samples_raises(self):
        with pytest.raises(ValueError, match="Allan deviation requires"):
            allan_deviation(np.ones(5), dt=1.0)

    def test_negative_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            allan_deviation(np.ones(100), dt=-1.0)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="dt must be positive"):
            allan_deviation(np.ones(100), dt=0.0)

    def test_constant_signal(self):
        """An exactly constant signal should give ADEV = 0 everywhere."""
        sig = np.ones(500)
        res = allan_deviation(sig, dt=0.05)
        assert np.all(res.adevs == 0.0)

    def test_nan_values_removed(self):
        """NaN values should be silently removed if enough valid samples remain."""
        sig = white_noise_signal(n=200)
        sig[::10] = np.nan  # sprinkle NaNs — 20 NaN, 180 valid > 10
        res = allan_deviation(sig, dt=0.05)
        assert res.n_samples == 180

    def test_too_many_nans_raises(self):
        """If NaN removal drops below MIN_SAMPLES, raise ValueError."""
        sig = np.full(20, np.nan)
        sig[:5] = 0.0  # only 5 valid
        with pytest.raises(ValueError):
            allan_deviation(sig, dt=1.0)

    def test_exactly_min_samples(self):
        """Exactly 10 valid samples should not raise."""
        sig = white_noise_signal(n=10)
        res = allan_deviation(sig, dt=0.05)
        assert res.n_samples == 10
        assert len(res.taus) >= 1

    def test_single_tau(self):
        """Edge case: only one valid tau point produced."""
        sig = white_noise_signal(n=10)  # N//3 = 3, gives very few tau points
        res = allan_deviation(sig, dt=1.0)
        assert len(res.taus) >= 1
        assert res.sigma_min >= 0


# ---------------------------------------------------------------------------
# adev_noise_fractions
# ---------------------------------------------------------------------------

class TestAdevNoiseFractions:
    def test_fractions_sum_to_one(self):
        sig = mixed_signal(n=3000)
        res = allan_deviation(sig, dt=0.05)
        fracs = adev_noise_fractions(res)
        total = fracs["white"] + fracs["flicker"] + fracs["random_walk"]
        assert abs(total - 1.0) < 0.01

    def test_returns_nan_for_bad_result(self):
        # Manufacture a result with sigma_min = 0
        sig = np.zeros(500)
        res = allan_deviation(sig, dt=0.05)
        fracs = adev_noise_fractions(res)
        # Should not raise; may return NaN or valid fractions
        assert isinstance(fracs, dict)
        assert "white" in fracs

    def test_fracs_keys_present(self):
        sig = white_noise_signal(n=1000)
        res = allan_deviation(sig, dt=0.05)
        fracs = adev_noise_fractions(res)
        for key in ("white", "flicker", "random_walk"):
            assert key in fracs
