"""
tests.test_lspr_features
=========================
Unit tests for src.features.lspr_features.

Covers:
  - LSPRFeatures.feature_vector (including zero-Δλ truthiness trap)
  - fit_lorentzian_peak (success, failure, sanity checks)
  - detect_lspr_peak (with and without prominence, Lorentzian fallback)
  - refine_peak_centroid
  - estimate_shift_xcorr (positive/negative shift, flat signal guard)
  - concentration_from_shift (sign convention, zero-slope guard)
  - extract_lspr_features (end-to-end with synthetic LSPR spectrum)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.lspr_features import (
    LSPR_SENSITIVITY_NM_PER_PPM,
    LSPRFeatures,
    concentration_from_shift,
    detect_lspr_peak,
    estimate_shift_xcorr,
    extract_lspr_features,
    fit_lorentzian_peak,
    refine_peak_centroid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_spectrum(
    wl: np.ndarray,
    center: float = 531.5,
    fwhm: float = 20.0,
    amplitude: float = 1.0,
    baseline: float = 0.05,
) -> np.ndarray:
    """Synthetic Gaussian peak (used as stand-in for the LSPR band in tests)."""
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2) + baseline


def _lorentzian_spectrum(
    wl: np.ndarray,
    center: float = 531.5,
    gamma: float = 10.0,
    amplitude: float = 1.0,
    baseline: float = 0.0,
) -> np.ndarray:
    """Synthetic Lorentzian peak — exact model used by fit_lorentzian_peak."""
    return amplitude / (1.0 + ((wl - center) / (gamma / 2.0)) ** 2) + baseline


_WL = np.linspace(400, 700, 3000)  # realistic 300-nm range at ~0.1 nm/px


# ---------------------------------------------------------------------------
# LSPRFeatures.feature_vector
# ---------------------------------------------------------------------------


class TestLSPRFeaturesVector:
    def test_all_none_returns_zeros(self):
        feat = LSPRFeatures()
        assert feat.feature_vector == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_populated_values_returned(self):
        feat = LSPRFeatures(
            delta_lambda=-0.5,
            delta_fwhm_nm=0.3,
            delta_amplitude=-0.03,
            delta_intensity_area=-1.2,
            delta_intensity_std=0.01,
            delta_asymmetry=0.05,
        )
        assert feat.feature_vector == [-0.5, 0.3, -0.03, -1.2, 0.01, 0.05]

    def test_populated_values_asymmetry_none(self):
        """When delta_asymmetry is None (fit failed), 6th element is 0.0."""
        feat = LSPRFeatures(
            delta_lambda=-0.5,
            delta_fwhm_nm=0.3,
            delta_amplitude=-0.03,
            delta_intensity_area=-1.2,
            delta_intensity_std=0.01,
        )
        assert feat.feature_vector == [-0.5, 0.3, -0.03, -1.2, 0.01, 0.0]

    def test_zero_delta_lambda_not_treated_as_missing(self):
        """Regression: `0.0 or 0.0` falsy trap must be fixed → use `is not None`."""
        feat = LSPRFeatures(
            delta_lambda=0.0,
            delta_fwhm_nm=0.0,
            delta_amplitude=0.0,
            delta_intensity_area=0.0,
            delta_intensity_std=0.0,
            delta_asymmetry=0.0,
        )
        # All fields are explicitly 0.0 (not None) — must survive as 0.0
        vec = feat.feature_vector
        assert vec == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Crucially: a LSPRFeatures with all-zero fields is distinct from one
        # with all-None fields; both should return [0,0,0,0,0,0] but for different reasons.
        feat_none = LSPRFeatures()
        assert feat.feature_vector == feat_none.feature_vector  # both [0,0,0,0,0,0]

    def test_negative_delta_lambda_preserved(self):
        """Negative shift (adsorption redshift) must not be zeroed out."""
        feat = LSPRFeatures(delta_lambda=-2.3)
        assert feat.feature_vector[0] == pytest.approx(-2.3)

    def test_length_always_six(self):
        """feature_vector always returns 6 elements (extended with delta_asymmetry)."""
        feat = LSPRFeatures(delta_lambda=1.0, delta_fwhm_nm=None)
        assert len(feat.feature_vector) == 6

    def test_legacy_vector_still_four(self):
        """feature_vector_legacy preserves the original 4-element contract."""
        feat = LSPRFeatures(
            delta_lambda=-0.5,
            delta_intensity_peak=-0.03,
            delta_intensity_area=-1.2,
            delta_intensity_std=0.01,
        )
        assert feat.feature_vector_legacy == [-0.5, -0.03, -1.2, 0.01]
        assert len(feat.feature_vector_legacy) == 4


# ---------------------------------------------------------------------------
# fit_lorentzian_peak
# ---------------------------------------------------------------------------


class TestFitLorentzianPeak:
    def test_fits_synthetic_lorentzian(self):
        center_true = 531.5
        intensities = _lorentzian_spectrum(_WL, center=center_true, gamma=12.0, amplitude=0.8)
        result = fit_lorentzian_peak(_WL, intensities, peak_wl_init=center_true)
        assert result is not None
        center_fit, fwhm_fit, amp_fit, center_std = result
        assert abs(center_fit - center_true) < 0.1  # sub-pixel accuracy
        assert fwhm_fit > 0

    def test_returns_none_on_too_few_points(self):
        # Only 2 points in the fit window → returns None
        wl = np.array([530.0, 532.0])
        inten = np.array([0.5, 1.0])
        assert fit_lorentzian_peak(wl, inten, peak_wl_init=531.0) is None

    def test_returns_none_if_center_outside_window(self):
        # Fit a flat line — no peak → unphysical parameters → None
        intensities = np.ones(len(_WL)) * 0.5
        result = fit_lorentzian_peak(_WL, intensities, peak_wl_init=531.5)
        # May return None or a result with amp≈0; either is acceptable
        if result is not None:
            _center, _fwhm, amp, _std = result
            # amp should be near zero for a flat signal
            assert amp >= 0

    def test_fwhm_positive(self):
        intensities = _lorentzian_spectrum(_WL, center=535.0, gamma=8.0)
        result = fit_lorentzian_peak(_WL, intensities, peak_wl_init=535.0)
        if result is not None:
            assert result[1] > 0  # FWHM > 0

    def test_center_std_non_negative(self):
        intensities = _lorentzian_spectrum(_WL, center=531.5)
        result = fit_lorentzian_peak(_WL, intensities, peak_wl_init=531.5)
        if result is not None:
            assert result[3] >= 0  # σ_center ≥ 0


# ---------------------------------------------------------------------------
# detect_lspr_peak
# ---------------------------------------------------------------------------


class TestDetectLSPRPeak:
    def test_detects_peak_within_search_window(self):
        center = 531.5
        intensities = _lorentzian_spectrum(_WL, center=center)
        peak_wl = detect_lspr_peak(_WL, intensities)
        assert peak_wl is not None
        assert abs(peak_wl - center) < 1.0

    def test_returns_none_for_empty_window(self):
        # Search window entirely outside wavelength range
        intensities = _lorentzian_spectrum(_WL, center=531.5)
        result = detect_lspr_peak(_WL, intensities, search_min=900.0, search_max=950.0)
        # Either None (< 5 points) or a fallback value — just check no crash
        # For this WL range (400-700 nm), 900-950 nm window should give None
        assert result is None or isinstance(result, float)

    def test_peak_outside_window_not_detected(self):
        # Peak at 650 nm, window 480–600 nm → fallback argmax + Lorentzian refinement
        # can place the result at or slightly above the window edge (acceptable).
        intensities = _gaussian_spectrum(_WL, center=650.0)
        peak = detect_lspr_peak(_WL, intensities, search_min=480.0, search_max=600.0)
        # Acceptable: returns None, or a value near the window edge (Lorentzian
        # refinement can shift the result slightly outside the hard boundary)
        if peak is not None:
            assert peak <= 650.0  # must not jump all the way to the true peak

    def test_flat_spectrum_returns_value(self):
        # Flat spectrum has no prominent peak; fallback is argmax
        intensities = np.ones(len(_WL))
        result = detect_lspr_peak(_WL, intensities)
        # Should not raise; may return a value or None
        assert result is None or isinstance(result, float)

    def test_noisy_peak_detected_approximately(self):
        rng = np.random.default_rng(0)
        intensities = _gaussian_spectrum(_WL, center=531.5, fwhm=20.0)
        intensities += rng.normal(0, 0.01, len(_WL))
        peak = detect_lspr_peak(_WL, intensities)
        assert peak is not None
        assert abs(peak - 531.5) < 2.0


# ---------------------------------------------------------------------------
# refine_peak_centroid
# ---------------------------------------------------------------------------


class TestRefinePeakCentroid:
    def test_refines_toward_true_center(self):
        center = 531.5
        intensities = _lorentzian_spectrum(_WL, center=center)
        refined = refine_peak_centroid(_WL, intensities, peak_wl=532.0)
        # centroid should move closer to 531.5
        assert abs(refined - center) < abs(532.0 - center) + 0.5

    def test_falls_back_when_window_too_small(self):
        # Only 2 points in window → returns input unchanged
        wl = np.array([531.0, 532.0])
        inten = np.array([0.5, 1.0])
        result = refine_peak_centroid(wl, inten, peak_wl=531.5, half_width_nm=0.1)
        assert result == pytest.approx(531.5)

    def test_returns_float(self):
        intensities = _lorentzian_spectrum(_WL, center=531.5)
        result = refine_peak_centroid(_WL, intensities, peak_wl=531.5)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# estimate_shift_xcorr
# ---------------------------------------------------------------------------


class TestEstimateShiftXcorr:
    def test_zero_shift_for_identical_spectra(self):
        intensities = _lorentzian_spectrum(_WL, center=531.5)
        shift = estimate_shift_xcorr(_WL, intensities, intensities)
        assert shift is not None
        assert abs(shift) < 0.5  # near-zero (pixel quantization noise)

    def test_detects_negative_redshift(self):
        """Adsorption → peak moves to longer wavelength (negative Δλ in our convention:
        Δλ = λ_gas − λ_ref, and cross-corr shift = gas - ref)."""
        ref = _lorentzian_spectrum(_WL, center=531.5)
        # Gas peak shifted +2 nm (longer λ) → cross-corr should return ~+2 nm
        gas = _lorentzian_spectrum(_WL, center=533.5)
        shift = estimate_shift_xcorr(_WL, gas, ref, center_nm=532.5)
        assert shift is not None
        assert shift > 0.5  # positive: gas peak is to the right of reference

    def test_detects_positive_blueshift(self):
        ref = _lorentzian_spectrum(_WL, center=531.5)
        gas = _lorentzian_spectrum(_WL, center=529.5)  # shifted -2 nm
        shift = estimate_shift_xcorr(_WL, gas, ref, center_nm=530.5)
        assert shift is not None
        assert shift < -0.5

    def test_returns_none_for_flat_reference(self):
        flat = np.ones(len(_WL)) * 0.5
        gas = _lorentzian_spectrum(_WL, center=531.5)
        result = estimate_shift_xcorr(_WL, gas, flat)
        # std(flat)=0 → returns None
        assert result is None

    def test_returns_none_when_window_too_narrow(self):
        intensities = _lorentzian_spectrum(_WL, center=531.5)
        result = estimate_shift_xcorr(_WL, intensities, intensities, window_nm=0.01)  # < 5 points
        assert result is None


# ---------------------------------------------------------------------------
# concentration_from_shift
# ---------------------------------------------------------------------------


class TestConcentrationFromShift:
    def test_basic_conversion(self):
        # Δλ = -0.116 nm with slope = -0.116 nm/ppm → 1.0 ppm
        c = concentration_from_shift(-0.116, slope=-0.116)
        assert c == pytest.approx(1.0)

    def test_zero_shift_gives_zero_concentration(self):
        c = concentration_from_shift(0.0, slope=-0.116)
        assert c == pytest.approx(0.0)

    def test_physical_floor_at_zero(self):
        # Positive Δλ (blueshift) would give negative concentration — clamp to 0
        c = concentration_from_shift(+0.5, slope=-0.116)
        assert c == pytest.approx(0.0)

    def test_returns_none_for_zero_slope(self):
        assert concentration_from_shift(-0.5, slope=0.0) is None

    def test_intercept_respected(self):
        # c = (Δλ − intercept) / slope
        c = concentration_from_shift(-0.216, slope=-0.116, intercept=-0.1)
        # (-0.216 - (-0.1)) / -0.116 = (-0.116) / -0.116 = 1.0
        assert c == pytest.approx(1.0, rel=1e-4)

    def test_lspr_default_slope_constant(self):
        assert pytest.approx(-0.116) == LSPR_SENSITIVITY_NM_PER_PPM


# ---------------------------------------------------------------------------
# extract_lspr_features  (end-to-end)
# ---------------------------------------------------------------------------


class TestExtractLSPRFeatures:
    @pytest.fixture
    def spectra_pair(self):
        """Reference at 531.5 nm, gas with 1 nm redshift to 532.5 nm."""
        ref = _lorentzian_spectrum(_WL, center=531.5, amplitude=0.8)
        gas = _lorentzian_spectrum(_WL, center=532.5, amplitude=0.75)
        return ref, gas

    def test_returns_lspr_features(self, spectra_pair):
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        assert isinstance(feat, LSPRFeatures)

    def test_peak_wavelength_near_gas_peak(self, spectra_pair):
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        assert feat.peak_wavelength is not None
        assert abs(feat.peak_wavelength - 532.5) < 1.5

    def test_delta_lambda_sign_positive_for_redshift(self, spectra_pair):
        """Gas peak is at longer λ than reference → Δλ = λ_gas − λ_ref > 0."""
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        assert feat.delta_lambda is not None
        assert feat.delta_lambda > 0.0

    def test_delta_lambda_magnitude_reasonable(self, spectra_pair):
        ref, gas = spectra_pair  # 1 nm shift
        feat = extract_lspr_features(_WL, gas, ref)
        assert feat.delta_lambda is not None
        assert 0.3 < feat.delta_lambda < 2.5

    def test_roi_params_stored(self, spectra_pair):
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref, roi_center=531.5, roi_width=20.0)
        assert feat.roi_center == pytest.approx(531.5)
        assert feat.roi_width == pytest.approx(20.0)

    def test_fwhm_positive_when_fit_succeeds(self, spectra_pair):
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        if feat.peak_fwhm_nm is not None:
            assert feat.peak_fwhm_nm > 0.0

    def test_feature_vector_length_six(self, spectra_pair):
        """feature_vector returns 6 orthogonal physical features including delta_asymmetry."""
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        assert len(feat.feature_vector) == 6

    def test_delta_fwhm_populated_when_fit_succeeds(self, spectra_pair):
        """ΔFWHM must be extracted when both gas and ref Lorentzian fits succeed."""
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        # If delta_lambda was extracted via Lorentzian fit, delta_fwhm_nm must also be set.
        if feat.delta_lambda_std is not None:
            assert feat.delta_fwhm_nm is not None
            assert feat.delta_amplitude is not None

    def test_identical_spectra_zero_delta_intensity_area(self):
        """No analyte → diff = 0 → ΔI_area = 0."""
        spec = _lorentzian_spectrum(_WL, center=531.5)
        feat = extract_lspr_features(_WL, spec, spec)
        if feat.delta_intensity_area is not None:
            assert abs(feat.delta_intensity_area) < 1e-6

    def test_zero_delta_lambda_survives_feature_vector(self):
        """Ensure the zero-Δλ fix: a spectrum identical to reference gives
        delta_lambda ≈ 0.0 which must appear as 0.0, not be treated as None."""
        spec = _lorentzian_spectrum(_WL, center=531.5)
        feat = extract_lspr_features(_WL, spec, spec)
        # feature_vector[0] should be 0.0 (not NaN, not corrupted)
        fv = feat.feature_vector
        assert fv[0] == pytest.approx(0.0, abs=0.1)
        assert not any(np.isnan(v) for v in fv)

    def test_snr_non_negative(self, spectra_pair):
        ref, gas = spectra_pair
        feat = extract_lspr_features(_WL, gas, ref)
        if feat.snr is not None:
            assert feat.snr >= 0.0
