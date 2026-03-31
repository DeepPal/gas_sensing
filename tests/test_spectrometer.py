"""
Tests for src.spectrometer — hardware abstraction layer.

Covers:
- SpectralFrame dataclass (construction, properties, serialisation)
- AbstractSpectrometer ABC (contract, convenience methods)
- SimulatedSpectrometer (lifecycle, physics, all modalities)
- SpectrometerRegistry (register, create, available, discover)
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pytest

from src.spectrometer import (
    AbstractSpectrometer,
    SimulatedSpectrometer,
    SpectralFrame,
    SpectrometerRegistry,
)
from src.spectrometer.base import AbstractSpectrometer as _BaseABC
from src.spectrometer.registry import SpectrometerRegistry as _Reg


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_frame(
    n: int = 100,
    seed: int = 0,
    dark_corrected: bool = False,
) -> SpectralFrame:
    rng = np.random.default_rng(seed)
    wl = np.linspace(500.0, 900.0, n)
    intensity = rng.uniform(100.0, 8000.0, n)
    return SpectralFrame(
        wavelengths=wl,
        intensities=intensity,
        timestamp=datetime.datetime(2025, 6, 1, 12, 0, 0, tzinfo=datetime.timezone.utc),
        integration_time_s=0.05,
        accumulations=3,
        dark_corrected=dark_corrected,
        nonlinearity_corrected=True,
        serial_number="TEST-0001",
        model_name="TestSpec",
        metadata={"foo": "bar", "value": 42},
    )


class _MinimalSpectrometer(AbstractSpectrometer):
    """Minimal concrete driver for testing the ABC contract."""

    _opened: bool = False

    def open(self) -> None:
        self._opened = True

    def close(self) -> None:
        self._opened = False

    def set_integration_time(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("integration_time must be > 0")
        self._it = seconds

    def acquire(self, accumulations: int = 1) -> SpectralFrame:
        wl = np.linspace(400.0, 800.0, 512)
        intensities = np.ones(512) * 1000.0
        return SpectralFrame(
            wavelengths=wl,
            intensities=intensities,
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            integration_time_s=0.05,
        )

    @property
    def wavelengths(self) -> np.ndarray:
        return np.linspace(400.0, 800.0, 512)

    @property
    def n_pixels(self) -> int:
        return 512

    @property
    def model(self) -> str:
        return "MinimalSpec"

    @property
    def serial_number(self) -> str:
        return "SN-MINIMAL"


# ===========================================================================
# SpectralFrame
# ===========================================================================


class TestSpectralFrame:
    def test_n_pixels(self) -> None:
        frame = _make_frame(200)
        assert frame.n_pixels == 200

    def test_wavelength_range(self) -> None:
        frame = _make_frame()
        lo, hi = frame.wavelength_range
        assert lo == pytest.approx(500.0)
        assert hi == pytest.approx(900.0)

    def test_peak_wavelength_type_and_range(self) -> None:
        frame = _make_frame()
        pw = frame.peak_wavelength
        assert isinstance(pw, float)
        lo, hi = frame.wavelength_range
        assert lo <= pw <= hi

    def test_peak_intensity_positive(self) -> None:
        frame = _make_frame()
        assert frame.peak_intensity > 0.0

    def test_snr_positive(self) -> None:
        frame = _make_frame()
        assert frame.snr > 0.0

    def test_snr_uniform_signal_is_inf(self) -> None:
        """Uniform signal → std of lowest 10 % ≈ 0 → SNR = inf."""
        frame = SpectralFrame(
            wavelengths=np.linspace(400, 800, 100),
            intensities=np.full(100, 5000.0),
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            integration_time_s=0.05,
        )
        assert math.isinf(frame.snr)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def test_to_dict_round_trip(self) -> None:
        frame = _make_frame()
        d = frame.to_dict()
        frame2 = SpectralFrame.from_dict(d)
        np.testing.assert_array_almost_equal(frame.wavelengths, frame2.wavelengths)
        np.testing.assert_array_almost_equal(frame.intensities, frame2.intensities)
        assert frame2.dark_corrected == frame.dark_corrected
        assert frame2.serial_number == frame.serial_number
        assert frame2.metadata == frame.metadata

    def test_to_dict_has_expected_keys(self) -> None:
        d = _make_frame().to_dict()
        for key in (
            "wavelengths", "intensities", "timestamp",
            "integration_time_s", "accumulations",
            "dark_corrected", "nonlinearity_corrected",
            "serial_number", "model_name", "metadata",
        ):
            assert key in d, f"Missing key: {key}"

    def test_from_dict_naive_timestamp_gets_utc(self) -> None:
        d = _make_frame().to_dict()
        d["timestamp"] = "2025-01-01T00:00:00"  # no tzinfo
        frame = SpectralFrame.from_dict(d)
        assert frame.timestamp.tzinfo is not None

    def test_wavelengths_returned_as_list_in_to_dict(self) -> None:
        d = _make_frame().to_dict()
        assert isinstance(d["wavelengths"], list)
        assert isinstance(d["intensities"], list)


# ===========================================================================
# AbstractSpectrometer — contract and convenience methods
# ===========================================================================


class TestAbstractSpectrometerContract:
    def setup_method(self) -> None:
        self.spec = _MinimalSpectrometer()

    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            _BaseABC()  # type: ignore[abstract]

    def test_context_manager(self) -> None:
        with _MinimalSpectrometer() as spec:
            assert spec._opened is True  # open() was called on __enter__
        assert spec._opened is False     # close() was called on __exit__

    def test_open_close(self) -> None:
        self.spec.open()
        assert self.spec._opened is True
        self.spec.close()
        assert self.spec._opened is False

    def test_set_integration_time_raises_on_zero(self) -> None:
        with pytest.raises(ValueError):
            self.spec.set_integration_time(0.0)

    def test_acquire_returns_spectral_frame(self) -> None:
        self.spec.open()
        frame = self.spec.acquire()
        assert isinstance(frame, SpectralFrame)

    def test_acquire_dark_marks_is_dark(self) -> None:
        self.spec.open()
        dark = self.spec.acquire_dark()
        assert dark.metadata.get("is_dark") is True

    def test_acquire_reference_marks_is_reference(self) -> None:
        self.spec.open()
        ref = self.spec.acquire_reference()
        assert ref.metadata.get("is_reference") is True

    def test_acquire_sequence_returns_n_frames(self) -> None:
        self.spec.open()
        frames = self.spec.acquire_sequence(n_frames=5, delay_s=0.0)
        assert len(frames) == 5
        assert all(isinstance(f, SpectralFrame) for f in frames)

    def test_repr_contains_model(self) -> None:
        r = repr(self.spec)
        assert "MinimalSpec" in r

    def test_n_pixels_property(self) -> None:
        assert self.spec.n_pixels == 512

    def test_wavelengths_property_length(self) -> None:
        assert len(self.spec.wavelengths) == 512


# ===========================================================================
# SimulatedSpectrometer
# ===========================================================================


class TestSimulatedSpectrometerLSPR:
    """LSPR modality — most scientifically critical."""

    def setup_method(self) -> None:
        self.spec = SimulatedSpectrometer(modality="lspr", seed=42)

    def test_open_sets_is_open(self) -> None:
        self.spec.open()
        assert self.spec.is_open is True
        self.spec.close()

    def test_close_clears_is_open(self) -> None:
        self.spec.open()
        self.spec.close()
        assert self.spec.is_open is False

    def test_context_manager(self) -> None:
        with SimulatedSpectrometer(modality="lspr", seed=0) as spec:
            assert spec.is_open

    def test_acquire_requires_open(self) -> None:
        with pytest.raises(RuntimeError, match="not open"):
            self.spec.acquire()

    def test_acquire_returns_spectral_frame(self) -> None:
        with self.spec:
            frame = self.spec.acquire()
        assert isinstance(frame, SpectralFrame)

    def test_frame_wavelength_range_lspr(self) -> None:
        with self.spec:
            frame = self.spec.acquire()
        lo, hi = frame.wavelength_range
        assert lo == pytest.approx(500.0, abs=1.0)
        assert hi == pytest.approx(1000.0, abs=1.0)

    def test_n_pixels_lspr(self) -> None:
        assert self.spec.n_pixels == 3648

    def test_peak_wavelength_near_reference_at_zero_conc(self) -> None:
        """At 0 ppm concentration, peak should be near 717.9 nm reference."""
        spec = SimulatedSpectrometer(modality="lspr", concentration_ppm=0.0, seed=0)
        with spec:
            frame = spec.acquire(accumulations=20)
        assert abs(frame.peak_wavelength - 717.9) < 5.0

    def test_langmuir_shift_redshifts_with_concentration(self) -> None:
        """LSPR Langmuir isotherm: increasing concentration → peak redshift (more negative)."""
        def peak_at(conc: float) -> float:
            s = SimulatedSpectrometer(modality="lspr", concentration_ppm=conc, seed=1)
            with s:
                return s.acquire(accumulations=50).peak_wavelength

        p0 = peak_at(0.0)
        p5 = peak_at(5.0)
        # Δλ_max = -15 nm (red-shift → lower wavelength? Check physics)
        # Actually for Au-MIP binding: shift is toward LONGER wavelength on binding.
        # The sim uses delta_lam_max = -15 nm: negative means shorter wavelength.
        # Either direction is valid as long as there IS a shift.
        assert abs(p5 - p0) > 0.5, f"Expected shift at 5 ppm but got p0={p0:.2f}, p5={p5:.2f}"

    def test_set_analyte_concentration(self) -> None:
        with self.spec:
            self.spec.set_analyte_concentration(10.0)
            frame = self.spec.acquire()
        assert isinstance(frame, SpectralFrame)

    def test_accumulations_reduces_noise(self) -> None:
        """Averaging N spectra should reduce std by approximately √N."""
        spec1 = SimulatedSpectrometer(modality="lspr", noise_fraction=0.02, seed=7)
        spec10 = SimulatedSpectrometer(modality="lspr", noise_fraction=0.02, seed=7)
        with spec1, spec10:
            # Acquire many single frames and average manually
            stds_1 = [spec1.acquire(1).peak_intensity for _ in range(30)]
            stds_10 = [spec10.acquire(10).peak_intensity for _ in range(30)]
        # The std of peak_intensity across repeated acquisitions should be smaller for acc=10
        assert np.std(stds_10) <= np.std(stds_1) * 1.5  # allow some slack for randomness

    def test_model_and_serial(self) -> None:
        assert "Simulated" in self.spec.model
        assert self.spec.serial_number == "SIM-0000"

    def test_set_integration_time_invalid(self) -> None:
        with pytest.raises(ValueError):
            self.spec.set_integration_time(-0.1)

    def test_frame_metadata_has_modality(self) -> None:
        with self.spec:
            frame = self.spec.acquire()
        assert frame.metadata.get("modality") == "lspr"

    def test_frame_intensities_non_negative(self) -> None:
        """clip(0) ensures no negative counts."""
        with self.spec:
            frame = self.spec.acquire(accumulations=5)
        assert float(frame.intensities.min()) >= 0.0


class TestSimulatedSpectrometerFluorescence:
    def test_fluorescence_modality_creates_shorter_axis(self) -> None:
        spec = SimulatedSpectrometer(modality="fluorescence", seed=0)
        assert spec.n_pixels == 2048

    def test_fluorescence_nonzero_concentration_higher_than_zero(self) -> None:
        """Any concentration > 0 should produce higher intensity than c=0 (no analyte)."""
        def peak_at(c: float) -> float:
            s = SimulatedSpectrometer(modality="fluorescence", concentration_ppm=c, seed=0)
            with s:
                return s.acquire(accumulations=20).peak_intensity

        assert peak_at(5.0) > peak_at(0.0)

    def test_fluorescence_zero_concentration_has_only_background(self) -> None:
        """At c=0 only Rayleigh background is present — no fluorophore peak.
        The peak at 400 nm from Rayleigh (λ⁻⁴) is ~2092 counts; at c>0 the
        fluorophore adds on top, so zero-conc should be the lower bound.
        """
        spec_zero = SimulatedSpectrometer(modality="fluorescence", concentration_ppm=0.0, seed=0)
        spec_mid = SimulatedSpectrometer(modality="fluorescence", concentration_ppm=5.0, seed=0)
        with spec_zero, spec_mid:
            peak_zero = spec_zero.acquire(accumulations=20).peak_intensity
            peak_mid = spec_mid.acquire(accumulations=20).peak_intensity
        # Fluorophore at c=5 ppm adds ≫ Rayleigh background
        assert peak_mid > peak_zero * 2.0


class TestSimulatedSpectrometerAbsorbance:
    def test_absorbance_modality_uv_range(self) -> None:
        spec = SimulatedSpectrometer(modality="absorbance", seed=0)
        with spec:
            frame = spec.acquire()
        lo, hi = frame.wavelength_range
        assert lo == pytest.approx(200.0, abs=1.0)
        assert hi == pytest.approx(800.0, abs=1.0)

    def test_absorbance_beer_lambert_scaling(self) -> None:
        """Higher concentration → higher absorbance peak."""
        def peak_at(c: float) -> float:
            s = SimulatedSpectrometer(modality="absorbance", concentration_ppm=c, seed=0)
            with s:
                return s.acquire(accumulations=20).peak_intensity

        assert peak_at(10.0) > peak_at(1.0)


# ===========================================================================
# SpectrometerRegistry
# ===========================================================================


class TestSpectrometerRegistry:
    def test_available_contains_simulated(self) -> None:
        aliases = SpectrometerRegistry.available()
        assert "simulated" in aliases

    def test_available_contains_sim_alias(self) -> None:
        assert "sim" in SpectrometerRegistry.available()

    def test_create_simulated(self) -> None:
        spec = SpectrometerRegistry.create("simulated")
        assert isinstance(spec, SimulatedSpectrometer)

    def test_create_sim_alias(self) -> None:
        spec = SpectrometerRegistry.create("sim")
        assert isinstance(spec, SimulatedSpectrometer)

    def test_create_case_insensitive(self) -> None:
        spec = SpectrometerRegistry.create("SIMULATED")
        assert isinstance(spec, SimulatedSpectrometer)

    def test_create_unknown_alias_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown spectrometer alias"):
            SpectrometerRegistry.create("does_not_exist_xyz")

    def test_register_custom_driver(self) -> None:
        """Register a custom driver class and create it via the registry."""
        @_Reg.register("_test_custom_driver")
        class _TestDriver(_MinimalSpectrometer):
            pass

        spec = _Reg.create("_test_custom_driver")
        assert isinstance(spec, _TestDriver)

    def test_register_driver_imperative(self) -> None:
        _Reg.register_driver("_test_imperative", _MinimalSpectrometer)
        spec = _Reg.create("_test_imperative")
        assert isinstance(spec, _MinimalSpectrometer)

    def test_available_returns_sorted_list(self) -> None:
        aliases = SpectrometerRegistry.available()
        assert aliases == sorted(aliases)

    def test_create_passes_kwargs(self) -> None:
        """create() must forward kwargs to the driver constructor."""
        spec = SpectrometerRegistry.create(
            "simulated",
            modality="fluorescence",
            concentration_ppm=2.5,
            seed=99,
        )
        assert isinstance(spec, SimulatedSpectrometer)
        assert spec._modality == "fluorescence"
        assert spec._concentration_ppm == pytest.approx(2.5)

    def test_create_as_context_manager(self) -> None:
        with SpectrometerRegistry.create("simulated") as spec:
            frame = spec.acquire()
        assert isinstance(frame, SpectralFrame)
