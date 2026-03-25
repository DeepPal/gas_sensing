"""
src.preprocessing.quality
==========================
Spectrum quality metrics: SNR, noise estimation, saturation and validity checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseMetrics:
    """Bundle of noise / quality statistics for one spectrum."""

    rms: float
    """Root-mean-square of the intensity values."""
    mad: float
    """Median Absolute Deviation (scaled to σ for Gaussian noise: × 1.4826)."""
    spectral_entropy: float
    """Shannon entropy of the normalised intensity distribution."""
    snr: float
    """Signal-to-noise ratio (peak / edge noise)."""


def compute_snr(
    intensities: np.ndarray,
    signal_region: tuple[int, int] | None = None,
    noise_region: tuple[int, int] | None = None,
) -> float:
    """Compute signal-to-noise ratio.

    Parameters
    ----------
    intensities:
        1-D intensity array.
    signal_region:
        ``(start, end)`` index slice for the signal band.
        ``None`` → use the global peak.
    noise_region:
        ``(start, end)`` index slice for the noise band.
        ``None`` → use the spectrum edges (outer 10% on each side).

    Returns
    -------
    float
        SNR value; 0.0 if the noise floor is essentially zero.
    """
    if len(intensities) == 0:
        return 0.0

    if signal_region is not None:
        signal = float(np.mean(np.abs(intensities[signal_region[0] : signal_region[1]])))
    else:
        # Peak height above local baseline (estimated from spectrum edges).
        # Using raw max would inflate SNR for high-offset spectra where the
        # DC baseline dominates the "signal" numerator.
        edge = max(1, len(intensities) // 10)
        baseline_est = float(np.mean(
            np.concatenate([intensities[:edge], intensities[-edge:]])
        ))
        signal = max(0.0, float(np.max(intensities)) - baseline_est)

    if noise_region is not None:
        noise = float(np.std(intensities[noise_region[0] : noise_region[1]]))
    else:
        edge = max(1, len(intensities) // 10)
        noise = float(np.std(np.concatenate([intensities[:edge], intensities[-edge:]])))

    return signal / noise if noise > 1e-10 else 0.0


def estimate_noise_metrics(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    signal_region: tuple[int, int] | None = None,
    noise_region: tuple[int, int] | None = None,
) -> NoiseMetrics:
    """Compute a full suite of noise metrics for a spectrum."""
    if len(intensities) == 0:
        return NoiseMetrics(rms=0.0, mad=0.0, spectral_entropy=0.0, snr=0.0)

    rms = float(np.sqrt(np.mean(np.square(intensities))))
    mad = float(np.median(np.abs(intensities - np.median(intensities))) * 1.4826)

    spec = np.abs(intensities - intensities.min())
    total = spec.sum()
    if total > 0:
        prob = spec / total
        entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))
    else:
        entropy = 0.0

    snr = compute_snr(intensities, signal_region, noise_region)
    return NoiseMetrics(rms=rms, mad=mad, spectral_entropy=entropy, snr=snr)


def check_saturation(
    intensities: np.ndarray,
    threshold: float = 60_000.0,
) -> bool:
    """Return ``True`` if any intensity exceeds *threshold* (saturated detector)."""
    return bool(np.any(intensities >= threshold))


def check_finite(intensities: np.ndarray) -> bool:
    """Return ``True`` if all values are finite (no NaN or Inf)."""
    return bool(np.all(np.isfinite(intensities)))


def is_valid_spectrum(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    min_snr: float = 4.0,
    saturation_threshold: float = 60_000.0,
) -> tuple[bool, list[str]]:
    """Run all hard-block and soft-warning checks on a spectrum.

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data arrays.
    min_snr:
        Minimum SNR for a soft warning (not a hard block).
    saturation_threshold:
        Intensity level above which the spectrum is considered saturated
        (hard block).

    Returns
    -------
    tuple[bool, list[str]]
        ``(valid, messages)`` — ``valid`` is ``False`` only on hard-block
        conditions; ``messages`` lists all warnings and errors.
    """
    messages: list[str] = []
    valid = True

    if len(wavelengths) == 0 or len(intensities) == 0:
        return False, ["Empty spectrum."]

    if len(wavelengths) != len(intensities):
        return False, [
            f"Length mismatch: wavelengths={len(wavelengths)}, intensities={len(intensities)}."
        ]

    if not check_finite(intensities):
        valid = False
        messages.append("Spectrum contains NaN or Inf values.")

    if check_saturation(intensities, saturation_threshold):
        valid = False
        messages.append(
            f"Detector saturation: max intensity {np.max(intensities):.0f} "
            f"≥ {saturation_threshold:.0f}."
        )

    if valid:
        snr = compute_snr(intensities)
        if snr < min_snr:
            messages.append(f"Low SNR warning: {snr:.1f} < {min_snr} (soft warning, not a block).")

    return valid, messages
