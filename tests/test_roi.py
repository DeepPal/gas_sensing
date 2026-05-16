"""Tests for src.signal.roi — ROI scanning utilities."""
import numpy as np
import pandas as pd
import pytest

from src.signal.roi import compute_band_ratio_matrix, find_monotonic_wavelengths

# ---------------------------------------------------------------------------
# compute_band_ratio_matrix
# ---------------------------------------------------------------------------


class TestComputeBandRatioMatrix:
    def test_shape_preserved(self):
        Y = np.random.default_rng(0).random((10, 50))
        out = compute_band_ratio_matrix(Y, half_width=3)
        assert out.shape == Y.shape

    def test_flat_signal_ratio_is_one(self):
        """Flat signal → left mean == right mean → ratio ≈ 1."""
        Y = np.ones((3, 20))
        out = compute_band_ratio_matrix(Y, half_width=2)
        np.testing.assert_allclose(out, 1.0, atol=1e-10)

    def test_ascending_slope(self):
        """Rising signal: left mean < right mean → ratio < 1 for interior pixels."""
        Y = np.arange(1, 21, dtype=float).reshape(1, 20)
        out = compute_band_ratio_matrix(Y, half_width=2)
        # Interior pixels: left side < right side so ratio < 1
        assert np.all(out[0, 2:-2] < 1.0)

    def test_zero_denominator_protected(self):
        """Near-zero right segment must not produce NaN or Inf."""
        Y = np.zeros((1, 10))
        Y[0, :5] = 1.0  # left side non-zero, right side zero
        out = compute_band_ratio_matrix(Y, half_width=2)
        assert np.all(np.isfinite(out))

    def test_empty_wavelengths(self):
        """Zero-width wavelength axis returns empty-like array."""
        Y = np.empty((5, 0))
        out = compute_band_ratio_matrix(Y, half_width=2)
        assert out.shape == (5, 0)

    def test_half_width_clipped_to_one(self):
        """half_width=0 is clamped to 1 — should not crash."""
        Y = np.ones((2, 10))
        out = compute_band_ratio_matrix(Y, half_width=0)
        assert out.shape == (2, 10)

    def test_single_row(self):
        """Single-sample matrix works correctly."""
        Y = np.array([[2.0, 4.0, 6.0, 4.0, 2.0]])
        out = compute_band_ratio_matrix(Y, half_width=1)
        assert out.shape == (1, 5)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# find_monotonic_wavelengths
# ---------------------------------------------------------------------------


def _make_canonical(
    concs: list[float],
    peak_wl: float = 700.0,
    shift_per_ppm: float = -2.0,
    n_points: int = 50,
) -> dict[float, pd.DataFrame]:
    """Build a synthetic canonical dict with a known monotonic peak shift."""
    wl = np.linspace(680.0, 730.0, n_points)
    canonical = {}
    for c in concs:
        center = peak_wl + c * shift_per_ppm
        intensity = np.exp(-0.5 * ((wl - center) / 5.0) ** 2) + 0.01
        canonical[c] = pd.DataFrame({"wavelength": wl, "intensity": intensity})
    return canonical


class TestFindMonotonicWavelengths:
    def test_too_few_concentrations_returns_empty(self):
        canonical = _make_canonical([0.5, 1.0])  # only 2 — need ≥3
        result = find_monotonic_wavelengths(canonical, 680.0, 730.0)
        assert result["best_wavelength"] is None
        assert result["candidates"] == []

    def test_empty_wl_range_returns_empty(self):
        canonical = _make_canonical([0.5, 1.0, 2.0])
        result = find_monotonic_wavelengths(canonical, 800.0, 850.0)
        assert result["best_wavelength"] is None
        assert result["candidates"] == []

    def test_known_shift_found(self):
        """Synthetic Gaussian peak shifting −2 nm/ppm should be discovered."""
        concs = [0.5, 1.0, 2.0, 4.0]
        canonical = _make_canonical(concs, peak_wl=700.0, shift_per_ppm=-2.0)
        result = find_monotonic_wavelengths(
            canonical, 680.0, 730.0, min_r2=0.5, min_spearman=0.7
        )
        assert result["passing_count"] > 0
        assert result["best_wavelength"] is not None
        bwl = result["best_wavelength"]
        # Best wavelength should be near the peak region (695–710 nm)
        assert 685.0 <= bwl <= 720.0

    def test_no_shift_nothing_passes(self):
        """Flat peak (no shift with concentration) should yield no passing candidates."""
        concs = [0.5, 1.0, 2.0]
        # All identical spectra (no shift)
        wl = np.linspace(680.0, 730.0, 50)
        intensity = np.exp(-0.5 * ((wl - 700.0) / 5.0) ** 2)
        canonical = {c: pd.DataFrame({"wavelength": wl, "intensity": intensity}) for c in concs}
        result = find_monotonic_wavelengths(
            canonical, 680.0, 730.0, min_r2=0.8, min_spearman=0.9
        )
        assert result["passing_count"] == 0
        assert result["best_wavelength"] is None

    def test_signal_col_fallback(self):
        """If signal_col not present, falls back to 'intensity'."""
        concs = [0.5, 1.0, 2.0, 4.0]
        canonical = _make_canonical(concs)
        # Request nonexistent column → falls back to 'intensity'
        result = find_monotonic_wavelengths(
            canonical, 680.0, 730.0, signal_col="transmittance", min_r2=0.5, min_spearman=0.7
        )
        assert result["signal_type"] == "intensity"

    def test_result_keys_complete(self):
        """All expected keys are present in the result dict."""
        canonical = _make_canonical([0.5, 1.0, 2.0])
        result = find_monotonic_wavelengths(canonical, 680.0, 730.0)
        for key in [
            "best_wavelength",
            "best_feature_type",
            "candidates",
            "best_peak",
            "best_valley",
            "total_scanned",
            "passing_count",
            "peak_candidates",
            "valley_candidates",
            "signal_type",
            "preferred_feature",
        ]:
            assert key in result, f"missing key: {key}"

    def test_absorbance_prefers_valleys(self):
        """With absorbance signal, preferred_feature should be 'valley'."""
        wl = np.linspace(680.0, 730.0, 50)
        concs = [0.5, 1.0, 2.0]
        # Create absorbance column (a valley = dip)
        canonical = {}
        for i, c in enumerate(concs):
            center = 700.0 + i * (-1.5)
            absorption = 0.5 * np.exp(-0.5 * ((wl - center) / 5.0) ** 2)
            canonical[c] = pd.DataFrame({"wavelength": wl, "absorbance": absorption})
        result = find_monotonic_wavelengths(
            canonical, 680.0, 730.0, signal_col="absorbance"
        )
        assert result["preferred_feature"] == "valley"
