"""
src.simulation.noise_model
===========================
Physically grounded spectrometer noise model.

Noise sources modelled (in order of signal chain)
---------------------------------------------------
1. **Shot noise** (Poisson) — dominant at high signal levels.
   σ_shot = √(I × QE × gain) where I is intensity in [0,1] fractional units.
2. **Dark current** — thermal electrons accumulated during integration.
   σ_dark = √(dark_rate × integration_s)  (Poisson statistics).
3. **Readout noise** — electronics noise per pixel per readout; Gaussian,
   signal-independent.
4. **Fixed-pattern noise (PRNU)** — pixel-to-pixel quantum efficiency variation;
   multiplicative, scales with signal level.
5. **Baseline drift** — slow low-frequency variation from source/lamp intensity
   fluctuations; correlated across wavelengths.

Usage
-----
::

    nm = SpectrometerNoise(
        full_well_electrons=65000,
        dark_current_e_per_s=50.0,
        readout_noise_e=8.0,
        integration_time_s=0.05,
    )
    rng = np.random.default_rng(42)
    noisy_spectrum = nm.apply(clean_spectrum, wavelengths, rng)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

import numpy as np

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class NoiseModel(ABC):
    """Abstract base for all noise models."""

    @abstractmethod
    def apply(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply noise to a clean spectrum.

        Parameters
        ----------
        spectrum:
            Clean (noise-free) intensity array, values in [0, 1] (fractional
            detector response, not raw ADC counts).
        wavelengths:
            Wavelength axis (nm), same shape as *spectrum*.
        rng:
            Seeded NumPy random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Noisy spectrum, same shape as *spectrum*.
        """


# ---------------------------------------------------------------------------
# Default: SpectrometerNoise
# ---------------------------------------------------------------------------


@dataclass
class SpectrometerNoise(NoiseModel):
    """Realistic CCD/CMOS spectrometer noise model.

    All noise components are physics-based and expressed in fractional
    intensity units (same scale as the input *spectrum*).

    Parameters
    ----------
    full_well_electrons:
        Detector full-well capacity (electrons). Converts fractional
        intensity → electrons for Poisson shot noise calculation.
        Typical CCD: 50 000–100 000 e⁻; CMOS: 10 000–30 000 e⁻.
    dark_current_e_per_s:
        Dark current in electrons/pixel/second. Typical cooled CCD: 1–10 e⁻/s.
        Room-temperature CMOS: 10–200 e⁻/s.
    readout_noise_e:
        RMS readout noise in electrons/pixel. Typical CCD: 3–15 e⁻.
        CMOS: 1–5 e⁻.
    integration_time_s:
        Integration (exposure) time in seconds. Determines total dark current.
    prnu_fraction:
        Pixel response non-uniformity as a fraction (e.g. 0.01 = 1% variation).
        Applied as a multiplicative Gaussian perturbation to each pixel.
    drift_amplitude:
        Amplitude of low-frequency source/lamp intensity drift as a fraction
        of signal (e.g. 0.005 = 0.5%). Modelled as a smooth sinusoid.
    drift_period_pixels:
        Spatial period (in pixels) of the drift oscillation.
    """

    full_well_electrons: float = 65_000.0
    dark_current_e_per_s: float = 50.0
    readout_noise_e: float = 8.0
    integration_time_s: float = 0.05
    prnu_fraction: float = 0.01
    drift_amplitude: float = 0.003
    drift_period_pixels: int = 500
    _prnu_map: np.ndarray | None = field(default=None, repr=False)

    def _get_prnu_map(self, n_pixels: int, rng: np.random.Generator) -> np.ndarray:
        """Return (and cache) the PRNU map — constant across frames."""
        if self._prnu_map is None or len(self._prnu_map) != n_pixels:
            # PRNU is a fixed property of the detector — frozen per-instance
            # but requires the pixel count at first call.
            object.__setattr__(
                self,
                "_prnu_map",
                1.0 + rng.normal(0.0, self.prnu_fraction, n_pixels),
            )
        return self._prnu_map  # type: ignore[return-value]

    def apply(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply all noise sources to *spectrum*.

        Returns a clipped spectrum in [0, 1] fractional units.
        """
        n = len(spectrum)

        # --- 1. PRNU (multiplicative, fixed pattern) -----------------------
        prnu = self._get_prnu_map(n, rng)
        s = spectrum * prnu

        # --- 2. Shot noise (Poisson on signal electrons) -------------------
        signal_e = np.maximum(s, 0.0) * self.full_well_electrons
        shot_e = rng.normal(0.0, np.sqrt(np.maximum(signal_e, 0.0)))
        shot_frac = shot_e / self.full_well_electrons

        # --- 3. Dark current (Poisson) -------------------------------------
        dark_e = self.dark_current_e_per_s * self.integration_time_s
        dark_noise_e = rng.normal(0.0, np.sqrt(dark_e), n)
        dark_frac = dark_noise_e / self.full_well_electrons

        # --- 4. Readout noise (Gaussian, signal-independent) ---------------
        readout_frac = rng.normal(0.0, self.readout_noise_e / self.full_well_electrons, n)

        # --- 5. Baseline drift (smooth, correlated across wavelengths) -----
        phase = rng.uniform(0.0, 2.0 * np.pi)
        pixel_idx = np.arange(n, dtype=float)
        drift = self.drift_amplitude * np.sin(
            2.0 * np.pi * pixel_idx / self.drift_period_pixels + phase
        )

        noisy = s + shot_frac + dark_frac + readout_frac + drift
        return cast(np.ndarray, np.clip(noisy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Domain-randomized noise — used for sim→real transfer learning
# ---------------------------------------------------------------------------


class DomainRandomizedNoise(NoiseModel):
    """Noise model that randomizes parameters each call.

    Samples detector parameters from realistic ranges on every ``apply()``
    call, producing diverse training data that generalises to unseen detectors.

    This implements *domain randomization* for sim-to-real transfer:
    the ML model sees a wide distribution of detector qualities during training
    and learns features that are robust to detector noise variance.
    """

    def __init__(
        self,
        full_well_range: tuple[float, float] = (30_000.0, 100_000.0),
        dark_current_range: tuple[float, float] = (5.0, 200.0),
        readout_noise_range: tuple[float, float] = (2.0, 20.0),
        integration_time_s: float = 0.05,
        prnu_range: tuple[float, float] = (0.005, 0.03),
        drift_amplitude_range: tuple[float, float] = (0.001, 0.01),
    ) -> None:
        self._fw_range = full_well_range
        self._dark_range = dark_current_range
        self._rn_range = readout_noise_range
        self._int_s = integration_time_s
        self._prnu_range = prnu_range
        self._drift_range = drift_amplitude_range

    def apply(
        self,
        spectrum: np.ndarray,
        wavelengths: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Sample random detector parameters, then apply noise."""
        nm = SpectrometerNoise(
            full_well_electrons=float(rng.uniform(*self._fw_range)),
            dark_current_e_per_s=float(rng.uniform(*self._dark_range)),
            readout_noise_e=float(rng.uniform(*self._rn_range)),
            integration_time_s=self._int_s,
            prnu_fraction=float(rng.uniform(*self._prnu_range)),
            drift_amplitude=float(rng.uniform(*self._drift_range)),
        )
        return nm.apply(spectrum, wavelengths, rng)
