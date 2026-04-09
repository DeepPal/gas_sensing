"""Tests for reference_validator module (C3 fix)."""

import numpy as np
import pytest
from src.signal.reference_validator import validate_reference_spectrum


class TestReferenceValidator:
    """Test reference spectrum validation logic."""

    def test_valid_reference_passes(self):
        """Stable reference spectrum should pass."""
        ref = 5000 + np.random.randn(3648) * 50  # Low RSD
        result = validate_reference_spectrum(ref, max_rsd_pct=5.0)
        assert result.valid, f"Expected valid=True, got: {result.recommendations}"
        assert result.rsd_pct < 2.0

    def test_high_rsd_fails(self):
        """Reference with RSD > limit should fail."""
        ref = 5000 * (1 + 0.15 * np.random.randn(3648))  # ~15% RSD
        result = validate_reference_spectrum(ref, max_rsd_pct=5.0)
        assert not result.valid, "Expected high RSD to fail"
        assert result.rsd_pct > 5.0
        assert any("RSD" in rec for rec in result.recommendations)

    def test_saturation_detected(self):
        """Saturated reference should fail."""
        ref = 5000 * np.ones(3648)
        ref[:100] = 70000  # Saturated region
        result = validate_reference_spectrum(ref)
        assert result.saturated
        assert not result.valid
        assert any("SATURATED" in rec for rec in result.recommendations)

    def test_has_nans(self):
        """NaN in reference should fail."""
        ref = 5000 * np.ones(3648)
        ref[:10] = np.nan
        result = validate_reference_spectrum(ref)
        assert result.has_nans
        assert not result.valid
        assert any("NaN" in rec for rec in result.recommendations)

    def test_signal_similar_to_reference(self):
        """If signal ≈ ref, that's good for LSPR (as expected)."""
        ref = 5000 * np.ones(3648)
        signal = 4900 * np.ones(3648)  # Slightly lower (normal for LSPR)
        result = validate_reference_spectrum(ref, signal, max_rsd_pct=10.0)
        # Should be valid (no adverse recommendations for signal < ref)
        assert result.valid or len(result.recommendations) == 0

    def test_low_signal_warning(self):
        """Very low reference signal should warn."""
        ref = 500 * np.ones(3648)  # Very low
        result = validate_reference_spectrum(ref)
        assert result.low_signal
        assert any("low" in rec.lower() for rec in result.recommendations)

    def test_rsd_calculation(self):
        """RSD should be calculated correctly."""
        ref = np.array([100.0, 110.0, 90.0, 105.0, 95.0] * 730)  # Exactly 3650 → clip to 3648
        ref = ref[:3648]
        # mean ≈ 100, std ≈ 7.07, RSD ≈ 7.07%
        result = validate_reference_spectrum(ref, max_rsd_pct=10.0)
        assert 5.0 < result.rsd_pct < 10.0

    def test_perfect_constant_reference(self):
        """Perfect constant reference (RSD=0) should pass."""
        ref = 5000 * np.ones(3648)
        result = validate_reference_spectrum(ref, max_rsd_pct=5.0)
        assert result.valid
        assert result.rsd_pct < 0.01

    def test_inf_detected(self):
        """Inf in reference should fail."""
        ref = 5000 * np.ones(3648)
        ref[100] = np.inf
        result = validate_reference_spectrum(ref)
        assert result.has_nans  # Inf is treated as non-finite
        assert not result.valid
