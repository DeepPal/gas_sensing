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

    # AC RMS (removes DC offset so the metric reflects noise, not signal level)
    rms = float(np.sqrt(np.mean(np.square(intensities - np.mean(intensities)))))
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


class NoiseFloorTracker:
    """Rolling per-frame estimate of the spectrometer dark noise floor.

    The noise floor is estimated from the spectral edge pixels (regions far
    from any LSPR peak) on every frame.  A rolling median over recent frames
    gives a stable, outlier-robust baseline.

    Usage
    -----
    ::

        tracker = NoiseFloorTracker()
        for frame in acquisition_loop:
            tracker.update(frame.intensities)
            if tracker.is_degrading:
                alert("Noise floor elevated — check lamp / dark current")
            current_floor = tracker.noise_floor_rms

    This distinguishes three noise components that have different physical
    causes and require different interventions:

    * **Shot noise**: scales as √I — handled by weighted Lorentzian fit
    * **Dark noise**: fixed per frame — this tracker measures it
    * **Structural baseline shift**: slow drift — tracked by DriftAgent
    """

    def __init__(
        self,
        window_frames: int = 100,
        edge_fraction: float = 0.05,
        degradation_factor: float = 2.0,
    ) -> None:
        """
        Parameters
        ----------
        window_frames:
            Number of recent frames used for the rolling noise estimate.
        edge_fraction:
            Fraction of spectrum pixels (each end) used as the off-peak noise
            region.  0.05 = first and last 5% of channels.
        degradation_factor:
            Alert if recent noise floor exceeds baseline by this factor.
            Default 2.0 = noise doubled (significant lamp or CCD degradation).
        """
        from collections import deque
        self._window = window_frames
        self._edge_frac = edge_fraction
        self._factor = degradation_factor
        self._history: deque[float] = deque(maxlen=window_frames)

    def update(self, intensities: np.ndarray) -> float:
        """Record one frame's noise floor estimate and return it.

        Parameters
        ----------
        intensities:
            Raw (pre-smoothing) 1-D intensity array.

        Returns
        -------
        float
            Estimated noise floor RMS for this frame.
        """
        if len(intensities) == 0:
            return 0.0
        n = len(intensities)
        edge = max(5, int(n * self._edge_frac))
        noise_region = np.concatenate([intensities[:edge], intensities[-edge:]])
        rms = float(np.std(noise_region))
        self._history.append(rms)
        return rms

    @property
    def noise_floor_rms(self) -> float:
        """Median noise floor RMS over the rolling window (counts)."""
        if not self._history:
            return 0.0
        return float(np.median(list(self._history)))

    @property
    def is_degrading(self) -> bool:
        """True if recent noise floor is > ``degradation_factor`` × early baseline.

        Requires at least 20 frames.  Uses the first 10% of the window as
        the baseline reference (avoids warm-up transient contaminating the alert).
        """
        h = list(self._history)
        if len(h) < 20:
            return False
        n_base = max(5, len(h) // 10)
        baseline = float(np.median(h[:n_base]))
        recent = float(np.median(h[-n_base:]))
        return baseline > 1e-12 and recent > baseline * self._factor

    def reset(self) -> None:
        """Clear history (e.g., after hardware reconfiguration)."""
        self._history.clear()


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


def _mad_zscores(values: np.ndarray) -> np.ndarray:
    """Compute robust Z-scores using the median absolute deviation.

    Uses the same constant (0.6745) as the legacy implementation in
    gas_analysis.core.signal_proc.basic to ensure parity.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def detect_outliers(
    spectra: list[np.ndarray],
    threshold: float = 3.0,
) -> list[bool]:
    """Flag spectra whose aggregate statistics deviate by more than *threshold* Z-scores.

    Uses five metrics (mean, std, max, min, range) with classical Z-scores,
    falling back to MAD-based Z-scores when variance is near zero.

    Parameters
    ----------
    spectra:
        List of 1-D intensity arrays (must all be the same length).
    threshold:
        Z-score threshold above which a spectrum is flagged as an outlier.

    Returns
    -------
    list[bool]
        One boolean per spectrum; ``True`` means the spectrum is an outlier.
    """
    from scipy import stats as _stats

    if len(spectra) < 2:
        return [False] * len(spectra)

    X = np.vstack(spectra)
    metrics: dict[str, np.ndarray] = {
        "mean": np.mean(X, axis=1),
        "std": np.std(X, axis=1),
        "max": np.max(X, axis=1),
        "min": np.min(X, axis=1),
        "range": np.max(X, axis=1) - np.min(X, axis=1),
    }

    outliers = np.zeros(len(spectra), dtype=bool)
    for values in metrics.values():
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue

        # Primary: classical Z-score
        z_scores = np.abs(_stats.zscore(values, nan_policy="omit"))

        # Fall back to MAD-based Z-scores when variance is ~0
        if np.isnan(z_scores).all() or np.allclose(values, values[0]):
            z_scores = np.abs(_mad_zscores(values))
        else:
            nan_mask = np.isnan(z_scores)
            if np.any(nan_mask):
                z_scores[nan_mask] = np.abs(_mad_zscores(values))[nan_mask]

        outliers |= z_scores > threshold

    return outliers.tolist()
