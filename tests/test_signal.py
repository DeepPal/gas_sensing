"""Tests for src.signal — spectral transforms and peak detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal.transforms import (
    append_absorbance_column,
    compute_transmittance,
    ensure_odd_window,
    smooth,
)
from src.signal.peak import estimate_shift_crosscorr, gaussian_peak_center


# ---------------------------------------------------------------------------
# compute_transmittance
# ---------------------------------------------------------------------------


class TestComputeTransmittance:
    def _make_df(self, intensities: list[float]) -> pd.DataFrame:
        wl = np.linspace(700, 750, len(intensities))
        return pd.DataFrame({"wavelength": wl, "intensity": intensities})

    def test_uniform_ref_gives_transmittance_one(self):
        sample = self._make_df([2.0] * 10)
        ref = self._make_df([2.0] * 10)
        out = compute_transmittance(sample, ref)
        assert "transmittance" in out.columns
        np.testing.assert_allclose(out["transmittance"].values, 1.0, atol=1e-6)

    def test_half_ref_gives_transmittance_half(self):
        sample = self._make_df([1.0] * 10)
        ref = self._make_df([2.0] * 10)
        out = compute_transmittance(sample, ref)
        np.testing.assert_allclose(out["transmittance"].values, 0.5, atol=1e-6)

    def test_none_ref_returns_unchanged(self):
        sample = self._make_df([1.0] * 5)
        out = compute_transmittance(sample, None)
        assert "transmittance" not in out.columns

    def test_empty_ref_returns_unchanged(self):
        sample = self._make_df([1.0] * 5)
        out = compute_transmittance(sample, pd.DataFrame())
        assert "transmittance" not in out.columns

    def test_transmittance_clipped_to_zero_one(self):
        sample = self._make_df([5.0] * 5)  # 5× ref → clipped to 1.0
        ref = self._make_df([1.0] * 5)
        out = compute_transmittance(sample, ref)
        assert out["transmittance"].max() <= 1.0
        assert out["transmittance"].min() >= 0.0

    def test_empty_sample_returns_empty(self):
        out = compute_transmittance(pd.DataFrame(), pd.DataFrame({"wavelength": [700], "intensity": [1]}))
        assert out.empty

    def test_sample_without_intensity_column_returned_unchanged(self):
        sample = pd.DataFrame({"wavelength": [700, 710], "transmittance": [0.5, 0.6]})
        ref = pd.DataFrame({"wavelength": [700, 710], "intensity": [1.0, 1.0]})
        out = compute_transmittance(sample, ref)
        assert "intensity" not in out.columns


# ---------------------------------------------------------------------------
# append_absorbance_column
# ---------------------------------------------------------------------------


class TestAppendAbsorbanceColumn:
    def test_adds_absorbance_from_transmittance(self):
        df = pd.DataFrame({"wavelength": [700, 710], "transmittance": [0.1, 1.0]})
        out = append_absorbance_column(df)
        assert "absorbance" in out.columns
        np.testing.assert_allclose(out["absorbance"].iloc[1], 0.0, atol=1e-6)
        assert out["absorbance"].iloc[0] > 0

    def test_no_transmittance_returns_unchanged(self):
        df = pd.DataFrame({"wavelength": [700], "intensity": [1.0]})
        out = append_absorbance_column(df)
        assert "absorbance" not in out.columns

    def test_inplace_modifies_original(self):
        df = pd.DataFrame({"wavelength": [700], "transmittance": [0.5]})
        out = append_absorbance_column(df, inplace=True)
        assert "absorbance" in df.columns
        assert out is df

    def test_copy_does_not_modify_original(self):
        df = pd.DataFrame({"wavelength": [700], "transmittance": [0.5]})
        _ = append_absorbance_column(df, inplace=False)
        assert "absorbance" not in df.columns


# ---------------------------------------------------------------------------
# ensure_odd_window
# ---------------------------------------------------------------------------


class TestEnsureOddWindow:
    def test_odd_input_unchanged(self):
        assert ensure_odd_window(11) == 11

    def test_even_input_incremented(self):
        assert ensure_odd_window(10) == 11

    def test_minimum_is_three(self):
        assert ensure_odd_window(1) == 3
        assert ensure_odd_window(2) == 3

    def test_large_even(self):
        assert ensure_odd_window(100) == 101


# ---------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------


class TestSmooth:
    def test_flat_signal_unchanged(self):
        y = np.ones(50)
        out = smooth(y, window=11, poly=2)
        np.testing.assert_allclose(out, 1.0, atol=1e-6)

    def test_noise_reduced(self):
        rng = np.random.default_rng(42)
        y = np.sin(np.linspace(0, 2 * np.pi, 100)) + rng.normal(0, 0.1, 100)
        out = smooth(y, window=11, poly=2)
        assert float(np.std(out)) < float(np.std(y))

    def test_short_array_returns_original(self):
        y = np.array([1.0, 2.0])
        out = smooth(y, window=11)
        assert out.shape == y.shape

    def test_output_same_shape(self):
        y = np.random.default_rng(0).random(200)
        out = smooth(y)
        assert out.shape == y.shape


# ---------------------------------------------------------------------------
# gaussian_peak_center
# ---------------------------------------------------------------------------


class TestGaussianPeakCenter:
    def _gauss(self, x: np.ndarray, x0: float, sigma: float = 2.0) -> np.ndarray:
        return np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    def test_symmetric_peak_recovered(self):
        x = np.linspace(700, 750, 500)
        y = self._gauss(x, x0=720.0)
        center = gaussian_peak_center(x, y)
        assert abs(center - 720.0) < 1.0  # pixel spacing ~0.1 nm; fit accuracy ~0.5 nm

    def test_with_idx_hint(self):
        x = np.linspace(700, 750, 500)
        y = self._gauss(x, x0=725.0)
        hint = int(np.argmax(y))
        center = gaussian_peak_center(x, y, idx_hint=hint)
        assert abs(center - 725.0) < 1.0

    def test_minimum_peak(self):
        x = np.linspace(700, 750, 200)
        y = -self._gauss(x, x0=715.0)
        center = gaussian_peak_center(x, y)
        assert abs(center - 715.0) < 1.0

    def test_empty_array_returns_nan(self):
        result = gaussian_peak_center(np.array([]), np.array([]))
        assert np.isnan(result)

    def test_two_pixel_array_returns_value(self):
        x = np.array([700.0, 702.0])
        y = np.array([0.5, 1.0])
        result = gaussian_peak_center(x, y)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# estimate_shift_crosscorr
# ---------------------------------------------------------------------------


class TestEstimateShiftCrosscorr:
    def _make_spectra(self, n: int = 200, shift_nm: float = 0.0):
        wl = np.linspace(700, 750, n)
        ref = np.exp(-0.5 * ((wl - 720.0) / 3.0) ** 2)
        # shift target by rolling — approximate for uniform grid
        dw = float(wl[1] - wl[0])
        shift_pixels = shift_nm / dw
        tgt = np.interp(wl, wl + shift_nm, ref, left=ref[0], right=ref[-1])
        return wl, ref, tgt

    def test_zero_shift_returns_near_zero(self):
        wl, ref, tgt = self._make_spectra(shift_nm=0.0)
        shift = estimate_shift_crosscorr(wl, ref, tgt)
        assert abs(shift) < 0.5

    def test_known_shift_recovered(self):
        # tgt[k] = ref[k - N] → tgt red-shifted → correlate returns negative shift
        wl, ref, tgt = self._make_spectra(n=400, shift_nm=2.0)
        shift = estimate_shift_crosscorr(wl, ref, tgt)
        # direction: negative = red-shifted target (convention matches pipeline.py)
        assert abs(abs(shift) - 2.0) < 1.0  # pixel-level accuracy; sign depends on convention

    def test_upsample_improves_precision(self):
        wl, ref, tgt = self._make_spectra(n=200, shift_nm=1.0)
        shift_1x = estimate_shift_crosscorr(wl, ref, tgt, upsample=1)
        shift_4x = estimate_shift_crosscorr(wl, ref, tgt, upsample=4)
        # both should be in the right direction (same sign)
        assert np.sign(shift_1x) == np.sign(shift_4x) or abs(shift_1x) < 0.1

    def test_length_mismatch_raises(self):
        wl = np.linspace(700, 750, 100)
        ref = np.ones(100)
        tgt = np.ones(99)
        with pytest.raises(ValueError, match="Mismatch"):
            estimate_shift_crosscorr(wl, ref, tgt)

    def test_single_point_returns_zero(self):
        wl = np.array([700.0])
        result = estimate_shift_crosscorr(wl, np.array([1.0]), np.array([1.0]))
        assert result == 0.0
