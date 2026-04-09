"""
tests.test_preprocessing
=========================
Unit tests for src.preprocessing modules:
  - baseline   (als_baseline, polynomial_baseline, rolling_min_baseline, correct_baseline)
  - denoising  (savgol_smooth, gaussian_smooth, moving_average_smooth, wavelet_denoise, smooth_spectrum)
  - normalization (normalize_minmax, normalize_area, normalize_zscore, normalize_snv,
                   normalize_peak, normalize_spectrum)

All functions are pure (no I/O, no state), so tests only need numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _gaussian_spectrum(n: int = 500, center: float = 250, sigma: float = 30) -> np.ndarray:
    """Gaussian peak on a flat baseline, shape (n,)."""
    x = np.arange(n, dtype=float)
    return 5000.0 * np.exp(-0.5 * ((x - center) / sigma) ** 2) + 200.0


def _noisy_spectrum(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = _gaussian_spectrum(n)
    return base + rng.normal(0, 50, n)


# ===========================================================================
# Baseline
# ===========================================================================


class TestAlsBaseline:
    def test_output_shape_preserved(self):
        from src.preprocessing.baseline import als_baseline

        it = _gaussian_spectrum()
        out = als_baseline(it)
        assert out.shape == it.shape

    def test_short_array_returned_unchanged(self):
        from src.preprocessing.baseline import als_baseline

        it = np.array([1.0, 2.0, 3.0])
        out = als_baseline(it)
        np.testing.assert_array_equal(out, it)

    def test_baseline_reduces_mean(self):
        """After correction, mean should be smaller than raw signal."""
        from src.preprocessing.baseline import als_baseline

        it = _gaussian_spectrum()
        out = als_baseline(it)
        assert np.mean(np.abs(out)) < np.mean(it)

    def test_output_is_finite(self):
        from src.preprocessing.baseline import als_baseline

        it = _noisy_spectrum()
        out = als_baseline(it)
        assert np.all(np.isfinite(out))

    def test_flat_signal_near_zero_after_correction(self):
        """ALS on a perfectly flat signal should return near-zero residuals."""
        from src.preprocessing.baseline import als_baseline

        it = np.ones(200) * 1000.0
        out = als_baseline(it)
        assert np.max(np.abs(out)) < 100.0  # residuals should be small


class TestPolynomialBaseline:
    def test_output_shape(self):
        from src.preprocessing.baseline import polynomial_baseline

        wl = np.linspace(400, 900, 300)
        it = _gaussian_spectrum(300)
        out = polynomial_baseline(wl, it, order=2)
        assert out.shape == (300,)

    def test_short_array_returned_unchanged(self):
        from src.preprocessing.baseline import polynomial_baseline

        wl = np.array([1.0, 2.0])
        it = np.array([5.0, 6.0])
        out = polynomial_baseline(wl, it)
        np.testing.assert_array_equal(out, it)

    def test_output_finite(self):
        from src.preprocessing.baseline import polynomial_baseline

        wl = np.linspace(400, 900, 300)
        it = _gaussian_spectrum(300) + np.linspace(0, 500, 300)
        out = polynomial_baseline(wl, it, order=2)
        assert np.all(np.isfinite(out))


class TestRollingMinBaseline:
    def test_output_shape(self):
        from src.preprocessing.baseline import rolling_min_baseline

        it = _gaussian_spectrum()
        out = rolling_min_baseline(it)
        assert out.shape == it.shape

    def test_short_array(self):
        from src.preprocessing.baseline import rolling_min_baseline

        it = np.array([1.0, 2.0])
        out = rolling_min_baseline(it)
        np.testing.assert_array_equal(out, it)

    def test_non_negative_peak_region(self):
        from src.preprocessing.baseline import rolling_min_baseline

        it = _gaussian_spectrum()
        out = rolling_min_baseline(it)
        # Peak region should still be positive after baseline removal
        assert np.max(out) > 0


class TestAirplsBaseline:
    def test_output_shape_preserved(self):
        from src.preprocessing.baseline import airpls_baseline

        it = _gaussian_spectrum()
        out = airpls_baseline(it)
        assert out.shape == it.shape

    def test_short_array_returned_unchanged(self):
        from src.preprocessing.baseline import airpls_baseline

        it = np.array([1.0, 2.0, 3.0])
        out = airpls_baseline(it)
        np.testing.assert_array_equal(out, it)

    def test_output_is_finite(self):
        from src.preprocessing.baseline import airpls_baseline

        it = _noisy_spectrum()
        out = airpls_baseline(it)
        assert np.all(np.isfinite(out))

    def test_baseline_reduces_offset(self):
        """After correction, mean should be smaller than raw signal with offset."""
        from src.preprocessing.baseline import airpls_baseline

        it = _gaussian_spectrum() + 500.0  # add DC offset
        out = airpls_baseline(it)
        assert np.mean(np.abs(out)) < np.mean(it)

    def test_peak_survives(self):
        """The spectral peak should still be visible after correction."""
        from src.preprocessing.baseline import airpls_baseline

        it = _gaussian_spectrum()
        out = airpls_baseline(it)
        assert np.max(out) > 100.0  # peak should remain prominent

    def test_flat_baseline_near_zero(self):
        from src.preprocessing.baseline import airpls_baseline

        it = np.ones(200) * 1000.0
        out = airpls_baseline(it)
        assert np.max(np.abs(out)) < 200.0

    def test_linear_trend_removed(self):
        """airPLS should remove a linear drift baseline."""
        from src.preprocessing.baseline import airpls_baseline

        x = np.linspace(0, 1, 300)
        peak = 3000.0 * np.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)
        drift = 500.0 * x
        it = peak + drift + 100.0
        out = airpls_baseline(it)
        # peak region should retain signal
        assert np.max(out) > 500.0


class TestCorrectBaseline:
    def test_als_dispatch(self):
        from src.preprocessing.baseline import correct_baseline

        wl = np.linspace(400, 900, 200)
        it = _gaussian_spectrum(200)
        out = correct_baseline(wl, it, method="als")
        assert out.shape == (200,)

    def test_airpls_dispatch(self):
        from src.preprocessing.baseline import correct_baseline

        wl = np.linspace(400, 900, 200)
        it = _gaussian_spectrum(200)
        out = correct_baseline(wl, it, method="airpls")
        assert out.shape == (200,)

    def test_polynomial_dispatch(self):
        from src.preprocessing.baseline import correct_baseline

        wl = np.linspace(400, 900, 200)
        it = _gaussian_spectrum(200)
        out = correct_baseline(wl, it, method="polynomial")
        assert out.shape == (200,)

    def test_rolling_min_dispatch(self):
        from src.preprocessing.baseline import correct_baseline

        wl = np.linspace(400, 900, 200)
        it = _gaussian_spectrum(200)
        out = correct_baseline(wl, it, method="rolling_min")
        assert out.shape == (200,)

    def test_unknown_method_raises(self):
        from src.preprocessing.baseline import correct_baseline

        with pytest.raises((ValueError, KeyError, NotImplementedError)):
            correct_baseline(np.ones(10), np.ones(10), method="nonexistent")


# ===========================================================================
# Denoising
# ===========================================================================


class TestSavgolSmooth:
    def test_output_shape(self):
        from src.preprocessing.denoising import savgol_smooth

        it = _noisy_spectrum()
        assert savgol_smooth(it).shape == it.shape

    def test_short_array_passthrough(self):
        from src.preprocessing.denoising import savgol_smooth

        it = np.array([1.0, 2.0, 3.0])
        out = savgol_smooth(it)
        assert out.shape == it.shape

    def test_reduces_noise(self):
        from src.preprocessing.denoising import savgol_smooth

        rng = np.random.default_rng(42)
        clean = np.sin(np.linspace(0, 2 * np.pi, 200)) * 1000
        noisy = clean + rng.normal(0, 50, 200)
        smoothed = savgol_smooth(noisy, window=21)
        # Smoothed should be closer to clean than noisy
        assert np.std(smoothed - clean) < np.std(noisy - clean)

    def test_finite_output(self):
        from src.preprocessing.denoising import savgol_smooth

        it = _noisy_spectrum()
        assert np.all(np.isfinite(savgol_smooth(it)))


class TestGaussianSmooth:
    def test_output_shape(self):
        from src.preprocessing.denoising import gaussian_smooth

        it = _gaussian_spectrum()
        assert gaussian_smooth(it).shape == it.shape

    def test_finite_output(self):
        from src.preprocessing.denoising import gaussian_smooth

        it = _noisy_spectrum()
        assert np.all(np.isfinite(gaussian_smooth(it)))


class TestMovingAverageSmooth:
    def test_output_shape(self):
        from src.preprocessing.denoising import moving_average_smooth

        it = _gaussian_spectrum()
        assert moving_average_smooth(it).shape == it.shape

    def test_finite_output(self):
        from src.preprocessing.denoising import moving_average_smooth

        it = _noisy_spectrum()
        assert np.all(np.isfinite(moving_average_smooth(it)))


class TestWaveletDenoise:
    def test_output_shape(self):
        from src.preprocessing.denoising import wavelet_denoise

        it = _noisy_spectrum()
        out = wavelet_denoise(it)
        assert out.shape == it.shape

    def test_short_array_passthrough(self):
        from src.preprocessing.denoising import wavelet_denoise

        it = np.array([1.0, 2.0])
        out = wavelet_denoise(it)
        assert out.shape == it.shape

    def test_finite_output(self):
        from src.preprocessing.denoising import wavelet_denoise

        it = _noisy_spectrum()
        assert np.all(np.isfinite(wavelet_denoise(it)))


class TestSmoothSpectrum:
    @pytest.mark.parametrize("method", ["savgol", "gaussian", "moving_average", "wavelet"])
    def test_dispatch_methods(self, method):
        from src.preprocessing.denoising import smooth_spectrum

        it = _noisy_spectrum()
        out = smooth_spectrum(it, method=method)
        assert out.shape == it.shape
        assert np.all(np.isfinite(out))

    def test_empty_input(self):
        from src.preprocessing.denoising import smooth_spectrum

        out = smooth_spectrum(np.array([]))
        assert len(out) == 0


# ===========================================================================
# Normalization
# ===========================================================================


class TestNormalizeMinmax:
    def test_range_zero_to_one(self):
        from src.preprocessing.normalization import normalize_minmax

        it = _gaussian_spectrum()
        out = normalize_minmax(it)
        assert pytest.approx(np.min(out), abs=1e-9) == 0.0
        assert pytest.approx(np.max(out), abs=1e-9) == 1.0

    def test_flat_signal_returns_zeros(self):
        from src.preprocessing.normalization import normalize_minmax

        it = np.ones(100) * 500
        out = normalize_minmax(it)
        np.testing.assert_array_equal(out, np.zeros(100))

    def test_shape_preserved(self):
        from src.preprocessing.normalization import normalize_minmax

        it = _gaussian_spectrum(200)
        assert normalize_minmax(it).shape == (200,)


class TestNormalizeArea:
    def test_positive_signal(self):
        from src.preprocessing.normalization import normalize_area

        it = _gaussian_spectrum()
        out = normalize_area(it)
        assert np.all(np.isfinite(out))

    def test_zero_area_returns_zeros(self):
        from src.preprocessing.normalization import normalize_area

        it = np.zeros(100)
        out = normalize_area(it)
        np.testing.assert_array_equal(out, np.zeros(100))


class TestNormalizeZscore:
    def test_mean_zero(self):
        from src.preprocessing.normalization import normalize_zscore

        it = _gaussian_spectrum()
        out = normalize_zscore(it)
        assert abs(np.mean(out)) < 1e-9

    def test_std_one(self):
        from src.preprocessing.normalization import normalize_zscore

        it = _gaussian_spectrum()
        out = normalize_zscore(it)
        assert abs(np.std(out) - 1.0) < 1e-6

    def test_constant_signal_returns_zeros(self):
        from src.preprocessing.normalization import normalize_zscore

        it = np.ones(50) * 300.0
        out = normalize_zscore(it)
        np.testing.assert_array_equal(out, np.zeros(50))


class TestNormalizeSnv:
    def test_same_as_zscore(self):
        """SNV should be identical to z-score for 1-D spectra."""
        from src.preprocessing.normalization import normalize_snv, normalize_zscore

        it = _gaussian_spectrum()
        np.testing.assert_allclose(normalize_snv(it), normalize_zscore(it))


class TestNormalizePeak:
    def test_max_abs_is_one(self):
        from src.preprocessing.normalization import normalize_peak

        it = _gaussian_spectrum()
        out = normalize_peak(it)
        assert pytest.approx(np.max(np.abs(out)), abs=1e-9) == 1.0

    def test_zero_peak_returns_zeros(self):
        from src.preprocessing.normalization import normalize_peak

        it = np.zeros(50)
        out = normalize_peak(it)
        np.testing.assert_array_equal(out, np.zeros(50))


class TestNormalizeSpectrum:
    @pytest.mark.parametrize("method", ["minmax", "area", "zscore", "standard", "snv", "peak"])
    def test_dispatch_all_methods(self, method):
        from src.preprocessing.normalization import normalize_spectrum

        it = _gaussian_spectrum()
        out = normalize_spectrum(it, method=method)
        assert out.shape == it.shape
        assert np.all(np.isfinite(out))

    def test_unknown_method_raises(self):
        from src.preprocessing.normalization import normalize_spectrum

        with pytest.raises(ValueError, match="Unknown"):
            normalize_spectrum(np.ones(100), method="bad_method")

    def test_empty_array_passthrough(self):
        from src.preprocessing.normalization import normalize_spectrum

        out = normalize_spectrum(np.array([]))
        assert len(out) == 0
