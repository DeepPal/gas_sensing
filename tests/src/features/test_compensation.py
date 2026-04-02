"""Tests for src.features.compensation — environmental drift compensation."""
import numpy as np
import pytest
from src.features.compensation import (
    AdaptiveDriftCorrector,
    EnvironmentalCompensator,
    differential_peak_correction,
    polynomial_detrend,
)


# ---------------------------------------------------------------------------
# EnvironmentalCompensator
# ---------------------------------------------------------------------------

class TestEnvironmentalCompensator:
    @pytest.fixture
    def fitted_comp(self):
        """Compensator fitted on perfect linear drift data."""
        comp = EnvironmentalCompensator(n_peaks=2, ref_temp_c=25.0, ref_humidity_pct=50.0)
        T = np.array([20.0, 23.0, 25.0, 27.0, 30.0])
        RH = np.array([40.0, 45.0, 50.0, 55.0, 60.0])
        # Peak 0: α_T = 0.05 nm/°C; Peak 1: α_T = 0.03 nm/°C; α_RH both 0.02
        shifts = np.column_stack([
            0.05 * (T - 25.0) + 0.02 * (RH - 50.0),
            0.03 * (T - 25.0) + 0.02 * (RH - 50.0),
        ])
        comp.fit(T, RH, shifts)
        return comp

    def test_fit_sets_is_fitted(self, fitted_comp):
        assert fitted_comp.is_fitted

    def test_fit_returns_alpha_keys(self):
        comp = EnvironmentalCompensator(n_peaks=1)
        result = comp.fit(
            np.array([24.0, 25.0, 26.0]),
            np.array([49.0, 50.0, 51.0]),
            np.array([[-0.05], [0.0], [0.05]]),
        )
        assert "alpha_temp_nm_per_c" in result
        assert "r2_per_peak" in result

    def test_compensate_removes_drift(self, fitted_comp):
        # At T=30, RH=60: drift peak 0 = 0.05*(30-25) + 0.02*(60-50) = 0.45 nm
        #                  drift peak 1 = 0.03*(30-25) + 0.02*(60-50) = 0.35 nm
        analyte_signal = np.array([-0.50, -0.30])
        observed = analyte_signal + np.array([0.45, 0.35])
        corrected = fitted_comp.compensate(observed, temp_c=30.0, humidity_pct=60.0)
        # Compensation accuracy: within 0.05 nm (fitting from 5 correlated T/RH points)
        np.testing.assert_allclose(corrected, analyte_signal, atol=0.05)

    def test_at_reference_conditions_no_change(self, fitted_comp):
        signal = np.array([-0.5, -0.3])
        corrected = fitted_comp.compensate(signal, temp_c=25.0, humidity_pct=50.0)
        np.testing.assert_allclose(corrected, signal, atol=1e-6)

    def test_compensate_batch_shape(self, fitted_comp):
        T = np.array([25.0, 26.0, 27.0])
        RH = np.array([50.0, 52.0, 54.0])
        shifts = np.column_stack([
            np.array([-0.5, -0.5, -0.5]),
            np.array([-0.3, -0.3, -0.3]),
        ])
        out = fitted_comp.compensate_batch(shifts, T, RH)
        assert out.shape == (3, 2)

    def test_unfitted_returns_raw_with_warning(self):
        comp = EnvironmentalCompensator(n_peaks=1)
        signal = np.array([-0.5])
        result = comp.compensate(signal, 27.0, 55.0)
        np.testing.assert_array_equal(result, signal)

    def test_save_load_roundtrip(self, fitted_comp, tmp_path):
        path = str(tmp_path / "comp.joblib")
        fitted_comp.save(path)
        loaded = EnvironmentalCompensator.load(path)
        assert loaded.is_fitted
        signal = np.array([-0.5, -0.3])
        orig = fitted_comp.compensate(signal, 28.0, 55.0)
        new = loaded.compensate(signal, 28.0, 55.0)
        np.testing.assert_allclose(orig, new, atol=1e-9)



# ---------------------------------------------------------------------------
# differential_peak_correction
# ---------------------------------------------------------------------------

class TestDifferentialPeakCorrection:
    def test_removes_common_mode_drift(self):
        # All 3 peaks drift by +0.1 nm (common mode)
        # Peak 0 also has analyte signal: -0.5 nm
        peaks = np.array([-0.4, 0.1, 0.1])  # peak0 = signal - drift; peaks 1,2 = drift only
        corrected = differential_peak_correction(peaks, [0], [1, 2])
        # Expected: -0.4 - 0.1 = -0.5 nm (drift removed)
        assert abs(corrected[0] - (-0.5)) < 1e-9

    def test_multi_analyte_peaks(self):
        # 4 peaks: [0,1] analyte-sensitive, [2,3] reference
        drift = 0.08
        peaks = np.array([-0.5 + drift, -0.2 + drift, drift, drift])
        corrected = differential_peak_correction(peaks, [0, 1], [2, 3])
        assert abs(corrected[0] - (-0.5)) < 1e-9
        assert abs(corrected[1] - (-0.2)) < 1e-9


# ---------------------------------------------------------------------------
# AdaptiveDriftCorrector
# ---------------------------------------------------------------------------

class TestAdaptiveDriftCorrector:
    def test_initial_frames_converge_ema(self):
        adc = AdaptiveDriftCorrector(n_peaks=1, alpha_ema=0.1, step_threshold_nm=0.5)
        # Feed clean air frames: gradual drift
        for i in range(20):
            shift = np.array([0.01 * i])  # slow drift
            adc.update_and_correct(shift)
        # Baseline should track the drift
        assert adc.current_baseline_nm is not None

    def test_analyte_step_detected_and_corrected(self):
        adc = AdaptiveDriftCorrector(n_peaks=1, alpha_ema=0.1, step_threshold_nm=0.2)
        # Establish baseline at 0
        for _ in range(30):
            adc.update_and_correct(np.array([0.0]))
        # Analyte step: shift to -0.5 nm
        corrected = adc.update_and_correct(np.array([-0.5]))
        # Should be close to -0.5 (baseline is ~0)
        assert abs(corrected[0] - (-0.5)) < 0.05

    def test_ema_frozen_during_analyte_exposure(self):
        adc = AdaptiveDriftCorrector(n_peaks=1, alpha_ema=0.1, step_threshold_nm=0.2)
        for _ in range(10):
            adc.update_and_correct(np.array([0.0]))
        baseline_before = adc.current_baseline_nm.copy()
        # Large step → frozen EMA
        for _ in range(5):
            adc.update_and_correct(np.array([-1.0]))
        baseline_after = adc.current_baseline_nm.copy()
        # Baseline should NOT have changed during analyte exposure
        np.testing.assert_allclose(baseline_before, baseline_after, atol=1e-9)

    def test_reset_clears_state(self):
        adc = AdaptiveDriftCorrector(n_peaks=2)
        for _ in range(5):
            adc.update_and_correct(np.array([0.05, 0.03]))
        adc.reset()
        assert adc.current_baseline_nm is None

    def test_multi_peak_shape(self):
        adc = AdaptiveDriftCorrector(n_peaks=3)
        out = adc.update_and_correct(np.array([0.1, 0.2, 0.3]))
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# polynomial_detrend
# ---------------------------------------------------------------------------

class TestPolynomialDetrend:
    def test_removes_linear_drift(self):
        t = np.linspace(0, 100, 101)
        drift = 0.001 * t  # 0.1 nm/s drift
        analyte_signal = np.where((t >= 30) & (t <= 70), -0.5, 0.0)
        observed = drift + analyte_signal
        # Clean air mask: before and after exposure
        mask = (t < 30) | (t > 70)
        detrended = polynomial_detrend(t, observed, mask, poly_degree=1)
        # In analyte region, detrended should be close to -0.5
        analyte_region = detrended[(t >= 35) & (t <= 65)]
        assert abs(analyte_region.mean() - (-0.5)) < 0.05

    def test_output_shape_same_as_input(self):
        t = np.arange(50, dtype=float)
        shifts = np.random.default_rng(0).normal(0, 0.01, (50, 2))
        mask = np.ones(50, dtype=bool)
        out = polynomial_detrend(t, shifts, mask, poly_degree=1)
        assert out.shape == (50, 2)

    def test_scalar_input_returns_1d(self):
        t = np.arange(20, dtype=float)
        shifts = 0.002 * t
        mask = np.ones(20, dtype=bool)
        out = polynomial_detrend(t, shifts, mask, poly_degree=1)
        assert out.ndim == 1
        assert out.shape == (20,)
