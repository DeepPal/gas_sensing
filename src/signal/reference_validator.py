"""Validation of reference (blank) spectra for LSPR normalization.

A poor reference spectrum → bad Δλ signal → invalid feature extraction → wrong LOD.

This module validates that reference (zero-gas) spectra are suitable for use in
LSPR wavelength-shift calculations. A stable, saturate-free reference with low noise
is critical for reliable differential signal analysis.

Usage
-----
::
    from src.signal.reference_validator import validate_reference_spectrum

    result = validate_reference_spectrum(
        reference_spectrum,
        signal_spectrum=None,
        max_rsd_pct=5.0
    )
    if not result.valid:
        print(result.recommendations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ReferenceValidationResult:
    """Result of reference spectrum validation."""

    valid: bool
    """True if reference is suitable for LSPR shift calculations."""

    rsd_pct: float
    """Relative standard deviation (repeatability) of the reference (%)."""

    low_signal: bool
    """True if reference intensity is very low (SNR risk)."""

    saturated: bool
    """True if any pixel saturated."""

    has_nans: bool
    """True if reference contains NaN/Inf."""

    recommendations: list[str]
    """List of actionable recommendations if validation failed or warned."""


def validate_reference_spectrum(
    reference_spectrum: np.ndarray,
    signal_spectrum: np.ndarray | None = None,
    max_rsd_pct: float = 5.0,
    saturation_threshold: float = 60000.0,
) -> ReferenceValidationResult:
    """Validate reference spectrum for use in LSPR diff signal calculation.

    A poor reference → bad Δλ signal → invalid feature extraction.

    Parameters
    ----------
    reference_spectrum : np.ndarray
        Blank/zero-gas reference intensities (baseline response)
    signal_spectrum : np.ndarray, optional
        Concurrent signal measurement (for sanity check)
    max_rsd_pct : float
        Maximum acceptable RSD in reference (IUPAC recommends <2-3%)
    saturation_threshold : float
        Reference must not be saturated (no pixel > threshold)

    Validation Checks
    -----------------
    1. No NaN/Inf values
    2. RSD of reference < max_rsd_pct (repeatability/stability)
    3. Not saturated (no pixel > threshold)
    4. Signal meets physical expectations (e.g., for LSPR: response < reference)
    5. Signal level adequate for SNR

    Returns
    -------
    ReferenceValidationResult
        Contains valid bool, RSD, and actionable recommendations
    """
    recommendations = []

    # ────────────────────────────────────────────────────────────────
    # Basic Validity
    # ────────────────────────────────────────────────────────────────
    ref = np.asarray(reference_spectrum, dtype=np.float64)

    if not np.isfinite(ref).all():
        return ReferenceValidationResult(
            valid=False,
            rsd_pct=-1.0,
            low_signal=False,
            saturated=False,
            has_nans=True,
            recommendations=[
                "Reference contains NaN/Inf — unable to use. "
                "Recapture reference spectrum with working spectrometer."
            ],
        )

    # ────────────────────────────────────────────────────────────────
    # Repeatability (RSD) — Key marker of stability
    # ────────────────────────────────────────────────────────────────
    ref_mean = np.mean(ref)
    ref_std = np.std(ref)
    rsd_pct = 100.0 * ref_std / (ref_mean + 1e-12)

    has_high_rsd = rsd_pct > max_rsd_pct
    if has_high_rsd:
        recommendations.append(
            f"Reference RSD = {rsd_pct:.2f}% exceeds limit of {max_rsd_pct:.1f}%. "
            f"Reference may be unstable (drifting baseline). "
            f"Consider: (1) Longer stabilization time before measurement, "
            f"(2) Check for temperature or humidity drift, "
            f"(3) Increase integration time for better count statistics."
        )

    # ────────────────────────────────────────────────────────────────
    # Saturation
    # ────────────────────────────────────────────────────────────────
    is_saturated = np.any(ref > saturation_threshold)
    if is_saturated:
        n_sat = np.sum(ref > saturation_threshold)
        sat_pct = 100.0 * n_sat / len(ref)
        recommendations.append(
            f"Reference SATURATED: {n_sat} pixels ({sat_pct:.1f}%) exceed threshold. "
            f"Saturated reference invalid for LSPR shift calculation. "
            f"Action: Reduce integration time or lower lamp intensity, "
            f"then recapture reference."
        )

    # ────────────────────────────────────────────────────────────────
    # Signal Sanity Check (if signal provided)
    # ────────────────────────────────────────────────────────────────
    if signal_spectrum is not None:
        sig = np.asarray(signal_spectrum, dtype=np.float64)

        # For LSPR: when gas adsorbs, peak intensity decreases
        # (due to damping of LSPR resonance)
        # So expect: signal < reference (at least at peak)
        ratio_mean = np.mean(sig) / (np.mean(ref) + 1e-12)

        if ratio_mean > 1.2:
            recommendations.append(
                f"Signal HIGHER than reference on average (ratio={ratio_mean:.2f}). "
                f"Expected signal ≤ reference for LSPR adsorption. "
                f"Check: (1) Is gas flow direction correct? "
                f"(2) Was reference captured with gas present (should be zero-gas)? "
                f"(3) Is sensor surface contaminated?"
            )

    # ────────────────────────────────────────────────────────────────
    # Signal Level Check
    # ────────────────────────────────────────────────────────────────
    is_low_signal = ref_mean < 1000.0
    if is_low_signal:
        recommendations.append(
            f"Reference signal very low ({ref_mean:.0f} counts). "
            f"May result in poor SNR in final Δλ measurement. "
            f"Action: Increase integration time or improve optical coupling."
        )

    valid = not (has_high_rsd or is_saturated or not np.isfinite(ref).all())

    if not valid:
        log.error(f"❌ Reference spectrum INVALID:")
        for rec in recommendations:
            log.error(f"     {rec[:80]}...")

    return ReferenceValidationResult(
        valid=valid,
        rsd_pct=float(rsd_pct),
        low_signal=bool(is_low_signal),
        saturated=bool(is_saturated),
        has_nans=not np.isfinite(ref).all(),
        recommendations=recommendations,
    )
