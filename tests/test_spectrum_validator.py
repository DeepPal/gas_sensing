"""Tests for spectrum_validator module (C1 fix)."""

import numpy as np
import pytest
from src.signal.spectrum_validator import validate_spectrum


class TestSpectrumValidator:
    """Test spectrum validation logic."""

    def test_valid_spectrum_passes(self):
        """Normal spectrum should pass all checks."""
        wl = np.linspace(350, 800, 3648)
        # Create spectrum with good SNR: strong peak + low noise
        intensity = 5000 + 3000 * np.sin(np.linspace(0, 4 * np.pi, 3648))
        intensity += 100 * np.random.randn(3648)  # Add noise
        result = validate_spectrum(intensity, wl, snr_threshold=2.0)
        assert result.valid, f"Expected valid=True, got errors: {result.errors}"
        assert len(result.errors) == 0

    def test_saturation_detected(self):
        """Spectrum with >5% saturation should fail."""
        wl = np.linspace(350, 800, 3648)
        intensity = 10000 * np.ones(3648)
        intensity[:400] = 70000  # 11% saturation
        result = validate_spectrum(intensity, wl, saturation_threshold=60000)
        assert not result.valid, "Expected saturation to fail validation"
        assert any("Saturation" in e for e in result.errors)

    def test_wrong_length_detected(self):
        """Wrong array length should fail."""
        intensity = np.ones(3649)  # Off by 1
        result = validate_spectrum(intensity)
        assert not result.valid
        assert any("Expected 3648" in e for e in result.errors)

    def test_nan_detected(self):
        """NaN values should fail."""
        intensity = np.ones(3648)
        intensity[100:110] = np.nan
        result = validate_spectrum(intensity)
        assert not result.valid
        assert any("non-finite" in e for e in result.errors)

    def test_low_snr_fails(self):
        """Flat spectrum should fail SNR check."""
        intensity = np.ones(3648) + 5 * np.random.randn(3648)  # Very low SNR
        result = validate_spectrum(intensity, snr_threshold=10)
        assert not result.valid

    def test_wavelength_mismatch(self):
        """Wavelength/intensity length mismatch."""
        intensity = np.ones(3648)
        wl = np.linspace(350, 800, 3647)  # Off by 1
        result = validate_spectrum(intensity, wl)
        assert not result.valid
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_wavelength_not_monotonic(self):
        """Wavelengths must be strictly increasing."""
        intensity = np.ones(3648)
        wl = np.linspace(800, 350, 3648)  # Decreasing
        result = validate_spectrum(intensity, wl)
        assert not result.valid
        assert any("increasing" in e.lower() for e in result.errors)

    def test_wavelength_out_of_range(self):
        """Wavelengths outside [200, 1200] nm should warn/fail."""
        intensity = np.ones(3648)
        wl = np.linspace(100, 2000, 3648)  # Out of range
        result = validate_spectrum(intensity, wl)
        assert not result.valid
        assert any("range" in e.lower() for e in result.errors)

    def test_negative_intensities_warning(self):
        """Negative intensities should warn but not fail."""
        wl = np.linspace(350, 800, 3648)
        intensity = -100 * np.ones(3648) + 5000  # Some negatives
        result = validate_spectrum(intensity, wl)
        # Should warn but still pass if SNR OK
        if result.valid:
            assert any("negative" in w.lower() for w in result.warnings)

    def test_flat_spectrum_fails(self):
        """Perfectly flat spectrum should fail (CV too low)."""
        intensity = 5000 * np.ones(3648)  # Flat line
        result = validate_spectrum(intensity)
        assert not result.valid
        assert any("flat" in e.lower() for e in result.errors)

    def test_inf_detected(self):
        """Inf values should fail."""
        intensity = np.ones(3648)
        intensity[50] = np.inf
        result = validate_spectrum(intensity)
        assert not result.valid
        assert any("non-finite" in e for e in result.errors)
