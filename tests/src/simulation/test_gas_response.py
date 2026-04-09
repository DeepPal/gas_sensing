"""Tests for src.simulation.gas_response — physics-complete simulation."""
import numpy as np
import pytest

from src.simulation.gas_response import (
    AnalyteProfile,
    ExposureEvent,
    PeakProfile,
    SensorConfig,
    SpectralSimulator,
    build_spectrum,
    make_analyte,
    make_multi_peak_sensor,
    make_single_peak_sensor,
)

# ---------------------------------------------------------------------------
# AnalyteProfile
# ---------------------------------------------------------------------------

class TestAnalyteProfile:
    def test_tau_decreases_with_concentration(self):
        a = make_analyte("X", 1, -0.5, tau_s=30.0)
        tau1 = a.tau(0, 1.0)
        tau10 = a.tau(0, 10.0)
        assert tau10 < tau1, "tau should decrease as concentration increases"

    def test_delta_lambda_eq_sign_preserved(self):
        a = make_analyte("X", 1, -0.5, tau_s=30.0)
        assert a.delta_lambda_eq(0, 1.0) < 0  # blue-shift

    def test_delta_lambda_eq_saturation(self):
        a = make_analyte("X", 1, -0.5, tau_s=30.0, kd_ppm=1.0)
        # At very high conc, response should saturate → approaches S × Kd
        low = abs(a.delta_lambda_eq(0, 0.01))
        high = abs(a.delta_lambda_eq(0, 1000.0))
        # Saturation: ratio should be much less than 1000/0.01 = 100000
        assert high / low < 500  # saturated

    def test_delta_lambda_eq_linear_at_low_conc(self):
        a = make_analyte("X", 1, -0.5, tau_s=30.0, kd_ppm=1000.0)
        # At c << Kd: Δλ ≈ S × c (linear)
        c1, c2 = 0.01, 0.02
        r = a.delta_lambda_eq(0, c2) / a.delta_lambda_eq(0, c1)
        assert abs(r - 2.0) < 0.01  # linear: doubling conc ≈ doubles shift


# ---------------------------------------------------------------------------
# SensorConfig
# ---------------------------------------------------------------------------

class TestSensorConfig:
    def test_sensitivity_matrix_shape(self):
        sensor = make_multi_peak_sensor([600.0, 700.0])
        sensor.analytes = [
            make_analyte("A", 2, -0.5),
            make_analyte("B", 2, -0.3),
        ]
        S = sensor.sensitivity_matrix
        assert S.shape == (2, 2)

    def test_wavelengths_correct_range(self):
        sensor = make_single_peak_sensor(700.0, wl_start=400.0, wl_end=900.0)
        wl = sensor.wavelengths
        assert abs(wl[0] - 400.0) < 1.0
        assert abs(wl[-1] - 900.0) < 1.0

    def test_analyte_n_peaks_mismatch_raises(self):
        # Constructing SensorConfig with mismatched analyte peaks raises ValueError
        with pytest.raises(ValueError):
            SensorConfig(
                peaks=[PeakProfile(center_nm=700.0)],  # 1 peak
                analytes=[
                    AnalyteProfile("X", [0.5, 0.3], [100.0, 100.0], [0.033, 0.033], [3.3, 3.3])
                ],  # 2 sensitivities for 1-peak sensor
            )

    def test_selectivity_matrix_none_for_single_analyte(self):
        sensor = make_single_peak_sensor(700.0)
        sensor.analytes = [make_analyte("A", 1, -0.5)]
        assert sensor.selectivity_matrix is None


# ---------------------------------------------------------------------------
# build_spectrum
# ---------------------------------------------------------------------------

class TestBuildSpectrum:
    def test_peak_at_expected_position(self):
        wl = np.linspace(600, 800, 1000)
        peaks = [PeakProfile(center_nm=700.0, fwhm_nm=20.0, amplitude=1.0)]
        spec = build_spectrum(wl, peaks)
        peak_idx = np.argmax(spec)
        assert abs(wl[peak_idx] - 700.0) < 1.0

    def test_offset_shifts_peak(self):
        wl = np.linspace(600, 800, 1000)
        peaks = [PeakProfile(center_nm=700.0, fwhm_nm=20.0, amplitude=1.0)]
        spec_ref = build_spectrum(wl, peaks, [0.0])
        spec_shifted = build_spectrum(wl, peaks, [-5.0])
        ref_peak = wl[np.argmax(spec_ref)]
        shifted_peak = wl[np.argmax(spec_shifted)]
        assert abs(shifted_peak - (ref_peak - 5.0)) < 1.0

    def test_multi_peak_both_visible(self):
        wl = np.linspace(600, 800, 2000)
        peaks = [
            PeakProfile(center_nm=650.0, fwhm_nm=10.0, amplitude=1.0),
            PeakProfile(center_nm=750.0, fwhm_nm=10.0, amplitude=0.8),
        ]
        spec = build_spectrum(wl, peaks)
        # Both peaks should have substantial signal above baseline
        peak1_region = spec[(wl > 640) & (wl < 660)]
        peak2_region = spec[(wl > 740) & (wl < 760)]
        assert peak1_region.max() > 0.9
        assert peak2_region.max() > 0.7

    def test_gaussian_peak_shape(self):
        wl = np.linspace(600, 800, 1000)
        peaks = [PeakProfile(center_nm=700.0, fwhm_nm=20.0, amplitude=1.0, shape="gaussian")]
        spec = build_spectrum(wl, peaks)
        # Should peak at 700 nm
        assert abs(wl[np.argmax(spec)] - 700.0) < 1.0

    def test_fano_peak_shape(self):
        wl = np.linspace(600, 800, 1000)
        peaks = [PeakProfile(center_nm=700.0, fwhm_nm=20.0, amplitude=1.0, shape="fano", fano_q=3.0)]
        spec = build_spectrum(wl, peaks)
        # Fano should be asymmetric — check it has valid values
        assert spec.max() > 0


# ---------------------------------------------------------------------------
# SpectralSimulator
# ---------------------------------------------------------------------------

class TestSpectralSimulator:
    @pytest.fixture
    def ethanol_sensor(self):
        sensor = make_single_peak_sensor(700.0)
        sensor.analytes = [make_analyte("Ethanol", 1, -0.5, tau_s=30.0, kd_ppm=100.0)]
        return sensor

    def test_reference_spectrum_shape(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        ref = sim.reference_spectrum()
        assert ref.shape == (3648,)

    def test_spectrum_at_state_returns_correct_shapes(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        spec, shifts = sim.spectrum_at_state({"Ethanol": 1.0}, 60.0)
        assert spec.shape == (3648,)
        assert len(shifts) == 1

    def test_gas_causes_shift_in_correct_direction(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        # Sensitivity is -0.5 nm/ppm → blue shift at equilibrium
        _, shifts = sim.spectrum_at_state({"Ethanol": 1.0}, 10000.0, add_noise=False)
        assert shifts[0] < 0.0, f"Expected blue shift (negative), got {shifts[0]}"

    def test_kinetics_response_grows_over_time(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        # With tau_s=30s at c=1ppm, use 3×, 30×, 300× tau for clear kinetic separation
        _, s3 = sim.spectrum_at_state({"Ethanol": 1.0}, 3.0, add_noise=False)
        _, s30 = sim.spectrum_at_state({"Ethanol": 1.0}, 30.0, add_noise=False)
        _, s300 = sim.spectrum_at_state({"Ethanol": 1.0}, 300.0, add_noise=False)
        # Response grows toward equilibrium: |s3| < |s30| < |s300|
        assert abs(s3[0]) < abs(s30[0]) < abs(s300[0])

    def test_higher_concentration_larger_shift(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        _, s1 = sim.spectrum_at_state({"Ethanol": 1.0}, 300.0, add_noise=False)
        _, s5 = sim.spectrum_at_state({"Ethanol": 5.0}, 300.0, add_noise=False)
        assert abs(s5[0]) > abs(s1[0])

    def test_no_gas_no_shift(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        _, shifts = sim.spectrum_at_state({}, 60.0, temp_c=25.0, humidity_pct=50.0, add_noise=False)
        assert abs(shifts[0]) < 1e-9

    def test_simulate_session_frame_count(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        events = [ExposureEvent("Ethanol", 1.0, 10.0, 20.0)]
        session = sim.simulate_session(events, total_duration_s=40.0, frame_rate_hz=2.0)
        # 40s × 2 Hz = 80 frames
        assert session.n_frames == 80

    def test_simulate_session_has_gas_signal(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        events = [ExposureEvent("Ethanol", 1.0, 10.0, 200.0)]
        session = sim.simulate_session(events, total_duration_s=300.0, frame_rate_hz=2.0, add_noise=False)
        # Peak shifts should not be all-zero during exposure
        shifts = session.true_peak_shifts  # (n_frames, 1)
        max_shift = abs(shifts[:, 0]).max()
        assert max_shift > 0.1, f"Expected significant gas response, got max shift {max_shift:.4f}"

    def test_multi_analyte_session(self):
        sensor = make_multi_peak_sensor([600.0, 700.0])
        sensor.analytes = [
            make_analyte("A", 2, -0.5, tau_s=20.0),
            make_analyte("B", 2, -0.3, tau_s=50.0),
        ]
        sim = SpectralSimulator(sensor, rng=np.random.default_rng(0))
        _, shifts = sim.spectrum_at_state({"A": 1.0, "B": 0.5}, 200.0, add_noise=False)
        assert len(shifts) == 2
        # Both peaks should show a shift
        assert all(s != 0.0 for s in shifts)

    def test_temperature_shift(self, ethanol_sensor):
        sim = SpectralSimulator(ethanol_sensor, rng=np.random.default_rng(0))
        _, s_ref = sim.spectrum_at_state({}, 0.0, temp_c=25.0, humidity_pct=50.0, add_noise=False)
        _, s_hot = sim.spectrum_at_state({}, 0.0, temp_c=35.0, humidity_pct=50.0, add_noise=False)
        # 10°C increase should shift peak
        assert s_hot[0] != s_ref[0]
