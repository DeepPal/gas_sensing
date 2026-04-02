"""Tests for src.simulation.noise_model — spectrometer noise model."""
import numpy as np
import pytest
from src.simulation.noise_model import DomainRandomizedNoise, NoiseModel, SpectrometerNoise


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def wl():
    return np.linspace(500.0, 900.0, 3648)


@pytest.fixture
def flat_spectrum():
    return np.full(3648, 0.5)


# ---------------------------------------------------------------------------
# SpectrometerNoise
# ---------------------------------------------------------------------------

class TestSpectrometerNoise:
    def test_is_noise_model_subclass(self):
        assert issubclass(SpectrometerNoise, NoiseModel)

    def test_output_shape_matches_input(self, flat_spectrum, wl, rng):
        nm = SpectrometerNoise()
        noisy = nm.apply(flat_spectrum, wl, rng)
        assert noisy.shape == flat_spectrum.shape

    def test_output_clipped_non_negative(self, flat_spectrum, wl, rng):
        nm = SpectrometerNoise()
        noisy = nm.apply(flat_spectrum, wl, rng)
        assert noisy.min() >= 0.0

    def test_output_clipped_max_one(self, flat_spectrum, wl, rng):
        nm = SpectrometerNoise()
        noisy = nm.apply(flat_spectrum, wl, rng)
        assert noisy.max() <= 1.0

    def test_noise_adds_variation(self, flat_spectrum, wl, rng):
        nm = SpectrometerNoise()
        noisy = nm.apply(flat_spectrum, wl, rng)
        # Noisy spectrum should differ from clean
        assert not np.allclose(noisy, flat_spectrum)

    def test_prnu_map_cached(self, flat_spectrum, wl, rng):
        nm = SpectrometerNoise()
        nm.apply(flat_spectrum, wl, rng)
        prnu1 = nm._prnu_map.copy()
        nm.apply(flat_spectrum, wl, rng)
        prnu2 = nm._prnu_map.copy()
        # PRNU is a fixed detector property — same between calls
        assert np.allclose(prnu1, prnu2)

    def test_zero_signal_gives_dark_readout_noise(self, wl, rng):
        nm = SpectrometerNoise(
            full_well_electrons=65000,
            dark_current_e_per_s=50.0,
            readout_noise_e=8.0,
            drift_amplitude=0.0,
            prnu_fraction=0.0,
        )
        zero = np.zeros(3648)
        noisy = nm.apply(zero, wl, rng)
        # Should be near zero with small noise (dark + readout << signal range)
        assert np.abs(noisy).max() < 0.01

    def test_high_signal_has_larger_variance_than_low(self, wl, rng):
        nm = SpectrometerNoise(prnu_fraction=0.0, drift_amplitude=0.0)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        low = np.full(3648, 0.05)
        high = np.full(3648, 0.8)
        # Run many trials
        low_var = np.var([nm.apply(low, wl, rng1)[1000] for _ in range(50)])
        high_var = np.var([nm.apply(high, wl, rng2)[1000] for _ in range(50)])
        assert high_var > low_var  # shot noise dominates at high signal


# ---------------------------------------------------------------------------
# DomainRandomizedNoise
# ---------------------------------------------------------------------------

class TestDomainRandomizedNoise:
    def test_is_noise_model_subclass(self):
        assert issubclass(DomainRandomizedNoise, NoiseModel)

    def test_output_shape(self):
        nm = DomainRandomizedNoise()
        rng = np.random.default_rng(1)
        wl = np.linspace(500, 900, 3648)
        spec = np.full(3648, 0.5)
        out = nm.apply(spec, wl, rng)
        assert out.shape == (3648,)

    def test_randomizes_each_call(self):
        nm = DomainRandomizedNoise()
        rng = np.random.default_rng(99)
        wl = np.linspace(500, 900, 3648)
        spec = np.full(3648, 0.5)
        out1 = nm.apply(spec, wl, rng)
        out2 = nm.apply(spec, wl, rng)
        # Different random seeds each call → different PRNU maps
        assert not np.allclose(out1, out2)
