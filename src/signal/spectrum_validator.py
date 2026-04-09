"""Input validation for spectral data before pipeline processing.

This module provides comprehensive validation of spectral measurements to catch
bad data BEFORE it propagates through the pipeline. Validates:
- Shape and length (must be 3648 points for CCS200)
- Saturation (>5% of pixels above threshold)
- SNR (estimated from peak isolation and noise floor)
- NaN/Inf values (finite check)
- Wavelength monotonicity

Usage
-----
::
    from src.signal.spectrum_validator import validate_spectrum

    result = validate_spectrum(
        intensities,
        wavelengths=wl,
        saturation_threshold=60000.0,
        snr_threshold=3.0
    )
    result.raise_on_error()  # Raises if validation fails
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of spectrum validation."""

    valid: bool
    """True if spectrum passed all validation checks."""

    errors: list[str]
    """List of critical errors (validation failed if non-empty)."""

    warnings: list[str]
    """List of warnings (validation passed but quality is reduced)."""

    def raise_on_error(self) -> None:
        """Raise ValueError if validation failed.

        Raises
        ------
        ValueError
            If ``valid=False``, raises with all error messages listed.
        """
        if not self.valid:
            msg = "Spectrum validation failed:\n" + "\n".join(f"  ✗ {e}" for e in self.errors)
            raise ValueError(msg)


def validate_spectrum(
    intensities: np.ndarray,
    wavelengths: np.ndarray | None = None,
    expected_points: int = 3648,
    saturation_threshold: float = 60000.0,
    snr_threshold: float = 3.0,
    dark_noise_max: float = 100.0,
) -> ValidationResult:
    """Comprehensive spectrum validation before pipeline entry.

    Tests:
    1. Array conversion and shape
    2. Expected length (CCS200 = 3648 points)
    3. Saturation detection (>5% above threshold)
    4. Finite values (no NaN/Inf)
    5. Signal-to-noise ratio
    6. Wavelength monotonicity (if provided)

    Parameters
    ----------
    intensities : np.ndarray
        Intensity array (should be shape (3648,))
    wavelengths : np.ndarray, optional
        Wavelength axis (must match intensities length if provided)
    expected_points : int
        Expected spectrum length (default 3648 for CCS200)
    saturation_threshold : float
        Hard limit for saturation detection (counts)
    snr_threshold : float
        Minimum acceptable SNR ratio
    dark_noise_max : float
        Maximum acceptable dark noise std dev

    Returns
    -------
    ValidationResult
        Contains valid bool, error list, warning list
    """
    errors = []
    warnings = []

    # ────────────────────────────────────────────────────────────────
    # Type & Shape Checks
    # ────────────────────────────────────────────────────────────────
    try:
        intensities = np.asarray(intensities, dtype=np.float64)
    except (TypeError, ValueError) as e:
        errors.append(f"Cannot convert intensities to array: {e}")
        return ValidationResult(False, errors, warnings)

    if intensities.ndim != 1:
        errors.append(f"Intensities must be 1-D, got shape {intensities.shape}")

    if len(intensities) != expected_points:
        errors.append(f"Expected {expected_points} points, got {len(intensities)}")

    # Early return if shape is wrong
    if errors:
        return ValidationResult(False, errors, warnings)

    # ────────────────────────────────────────────────────────────────
    # Wavelength Consistency
    # ────────────────────────────────────────────────────────────────
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths, dtype=np.float64)
        if len(wavelengths) != len(intensities):
            errors.append(
                f"Wavelength axis length mismatch: "
                f"{len(wavelengths)} vs {len(intensities)}"
            )
        if len(wavelengths) > 1 and not np.all(np.diff(wavelengths) > 0):
            errors.append("Wavelength axis must be strictly increasing")
        if len(wavelengths) > 0:
            if wavelengths[0] < 200 or wavelengths[-1] > 1200:
                errors.append(
                    f"Wavelengths out of expected range [200, 1200] nm: "
                    f"[{wavelengths[0]:.1f}, {wavelengths[-1]:.1f}]"
                )

    # ────────────────────────────────────────────────────────────────
    # Value Range & Finite Checks
    # ────────────────────────────────────────────────────────────────
    if not np.isfinite(intensities).all():
        n_bad = np.sum(~np.isfinite(intensities))
        errors.append(f"Found {n_bad} non-finite values (NaN/Inf) in spectrum")

    if np.any(intensities < 0):
        n_neg = np.sum(intensities < 0)
        warnings.append(
            f"Found {n_neg} negative intensities (expected 0 counts). "
            f"May indicate DMA/ underflow issues."
        )

    if np.any(intensities > saturation_threshold):
        n_sat = np.sum(intensities > saturation_threshold)
        sat_pct = 100.0 * n_sat / len(intensities)
        if sat_pct > 5:
            errors.append(
                f"Saturation: {n_sat} points ({sat_pct:.1f}%) exceed threshold "
                f"({saturation_threshold:.0f} counts)"
            )
        else:
            warnings.append(
                f"Minor saturation: {n_sat} points ({sat_pct:.2f}%) at threshold"
            )

    # ────────────────────────────────────────────────────────────────
    # Signal-to-Noise Ratio
    # ────────────────────────────────────────────────────────────────
    signal_max = np.max(intensities)
    signal_min = np.min(intensities)
    signal_rms = np.sqrt(np.mean(intensities**2))

    # Estimate noise from dark region (low intensity points)
    dark_idx = intensities < np.percentile(intensities, 10)
    if np.sum(dark_idx) > 10:
        noise_std = np.std(intensities[dark_idx])
        if noise_std > dark_noise_max:
            warnings.append(
                f"Dark noise high: σ_dark = {noise_std:.2f} counts "
                f"(typical: <{dark_noise_max})"
            )

    # Estimate SNR from peak
    signal_range = signal_max - signal_min
    snr = signal_range / (np.std(intensities) + 1e-9)
    if snr < snr_threshold:
        errors.append(f"SNR too low: {snr:.2f} (require > {snr_threshold:.1f})")

    # ────────────────────────────────────────────────────────────────
    # Consistency Checks
    # ────────────────────────────────────────────────────────────────
    # Spectrum should have recognizable structure (not flat line)
    spectrum_std = np.std(intensities)
    spectrum_mean = np.mean(intensities)
    cv = spectrum_std / (spectrum_mean + 1e-9)
    if cv < 0.01:
        errors.append(
            f"Spectrum appears flat (CV={cv:.4f}, expected >0.01). "
            f"Shape is not resolved."
        )

    # Most of the energy should be in one region (peak)
    peak_concentration = np.max(intensities) / (np.sum(intensities) + 1e-9)
    if peak_concentration < 0.001:
        warnings.append(f"Peak is diffuse (max/sum={peak_concentration:.4f})")

    valid = len(errors) == 0

    if errors:
        log.error(f"❌ Spectrum validation FAILED: {len(errors)} error(s)")
        for err in errors:
            log.error(f"     {err}")

    if warnings and valid:
        log.warning(f"⚠️  Spectrum validation OK but {len(warnings)} warning(s)")
        for warn in warnings:
            log.warning(f"     {warn}")
    elif warnings and not valid:
        for warn in warnings:
            log.warning(f"     {warn}")

    return ValidationResult(valid, errors, warnings)
