"""
src.spectrometer.simulated
==========================
Simulated spectrometer for offline development, testing, and demonstration.

Generates physically plausible spectra for three common spectroscopic
modalities so the full analysis pipeline can be exercised without hardware:

``'lspr'``
    LSPR sensor — Lorentzian peak at ~717.9 nm
    that red-shifts with analyte concentration following a Langmuir
    adsorption isotherm.  Thermal drift modelled as a linear trend.

``'fluorescence'``
    Broadband emission with a Gaussian fluorophore peak.  Intensity
    increases linearly with concentration (Beer-Lambert regime).

``'absorbance'``
    UV-Vis absorbance following Beer-Lambert law.  Returns absorbance
    units directly (log10 scale); peak position configurable.

Usage
-----
::

    from src.spectrometer.simulated import SimulatedSpectrometer

    with SimulatedSpectrometer(modality="lspr", seed=42) as spec:
        spec.set_integration_time(0.05)
        dark = spec.acquire_dark()
        ref  = spec.acquire_reference()
        # Expose sensor to 1 ppm analyte
        spec.set_analyte_concentration(1.0)
        frame = spec.acquire(accumulations=3)
        print(frame.peak_wavelength)  # ~718.8 nm (red-shifted from ref 717.9 nm)
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Literal, cast

import numpy as np

from src.spectrometer.base import AbstractSpectrometer, SpectralFrame

# ---------------------------------------------------------------------------
# Physical constants and defaults (example LSPR configuration for CCS200)
# ---------------------------------------------------------------------------

_LSPR_N_PIXELS: int = 3648
_LSPR_WL_START: float = 500.0   # nm
_LSPR_WL_END:   float = 1000.0  # nm
_LSPR_PEAK_REF: float = 717.9   # nm  (example reference peak; override via peak_wavelength_nm)
_LSPR_FWHM:     float = 55.0    # nm  (typical LSPR linewidth)
_LSPR_AMP:      float = 8000.0  # counts at peak (50 ms integration, 12-bit detector)

_FLUORESCENCE_N_PIXELS: int = 2048
_FLUORESCENCE_WL_START: float = 400.0
_FLUORESCENCE_WL_END:   float = 900.0

_ABSORBANCE_N_PIXELS: int = 2048
_ABSORBANCE_WL_START: float = 200.0
_ABSORBANCE_WL_END:   float = 800.0

SpectrometerModality = Literal["lspr", "fluorescence", "absorbance"]


class SimulatedSpectrometer(AbstractSpectrometer):
    """Physically realistic simulated spectrometer.

    Parameters
    ----------
    modality :
        Spectroscopic modality to simulate.  One of
        ``'lspr'``, ``'fluorescence'``, ``'absorbance'``.
    concentration_ppm :
        Initial analyte concentration in ppm (default 0 = blank).
    noise_fraction :
        Shot-noise level as a fraction of peak signal (default 0.001 = 0.1%).
    drift_nm_per_hour :
        Linear wavelength drift rate for LSPR modality (default 0.02 nm/h).
    seed :
        NumPy RNG seed for reproducibility.  None = non-deterministic.
    peak_wavelength_nm :
        Override the reference peak centre for LSPR / fluorescence.
    integration_time_s :
        Starting integration time (can be changed via
        :meth:`set_integration_time`).
    """

    _MODEL = "SimulatedSpectrometer"
    _SERIAL = "SIM-0000"

    def __init__(
        self,
        modality: SpectrometerModality = "lspr",
        concentration_ppm: float = 0.0,
        noise_fraction: float = 0.001,
        drift_nm_per_hour: float = 0.02,
        seed: int | None = 42,
        peak_wavelength_nm: float | None = None,
        integration_time_s: float = 0.05,
    ) -> None:
        self._modality = modality
        self._concentration_ppm = float(concentration_ppm)
        self._noise_fraction = float(noise_fraction)
        self._drift_nm_per_hour = float(drift_nm_per_hour)
        self._rng = np.random.default_rng(seed)
        self._integration_time_s = float(integration_time_s)
        self._opened = False
        self._frame_index = 0
        self._open_time: datetime.datetime | None = None

        # Wavelength axis
        if modality == "lspr":
            self._wl = np.linspace(_LSPR_WL_START, _LSPR_WL_END, _LSPR_N_PIXELS)
            self._peak_ref = peak_wavelength_nm or _LSPR_PEAK_REF
        elif modality == "fluorescence":
            self._wl = np.linspace(_FLUORESCENCE_WL_START, _FLUORESCENCE_WL_END, _FLUORESCENCE_N_PIXELS)
            self._peak_ref = peak_wavelength_nm or 650.0
        else:  # absorbance
            self._wl = np.linspace(_ABSORBANCE_WL_START, _ABSORBANCE_WL_END, _ABSORBANCE_N_PIXELS)
            self._peak_ref = peak_wavelength_nm or 450.0

    # ------------------------------------------------------------------
    # AbstractSpectrometer interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        self._opened = True
        self._open_time = datetime.datetime.now(datetime.timezone.utc)
        self._frame_index = 0

    def close(self) -> None:
        self._opened = False

    def set_integration_time(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError(f"Integration time must be > 0, got {seconds} s")
        self._integration_time_s = float(seconds)

    def acquire(self, accumulations: int = 1) -> SpectralFrame:
        if not self._opened:
            raise RuntimeError("Spectrometer is not open. Call open() first.")

        # Average N independent acquisitions (reduces shot noise by √N)
        stack = np.stack([self._generate_spectrum() for _ in range(max(1, accumulations))])
        intensities = stack.mean(axis=0)

        ts = datetime.datetime.now(datetime.timezone.utc)
        self._frame_index += 1

        return SpectralFrame(
            wavelengths=self._wl.copy(),
            intensities=intensities,
            timestamp=ts,
            integration_time_s=self._integration_time_s,
            accumulations=accumulations,
            dark_corrected=False,
            nonlinearity_corrected=True,
            serial_number=self._SERIAL,
            model_name=f"{self._MODEL}/{self._modality}",
            metadata={
                "modality": self._modality,
                "concentration_ppm": self._concentration_ppm,
                "frame_index": self._frame_index,
                "noise_fraction": self._noise_fraction,
            },
        )

    @property
    def wavelengths(self) -> np.ndarray:
        return cast(np.ndarray, self._wl.copy())

    @property
    def n_pixels(self) -> int:
        return len(self._wl)

    @property
    def model(self) -> str:
        return self._MODEL

    @property
    def serial_number(self) -> str:
        return self._SERIAL

    @property
    def integration_time_s(self) -> float:
        return self._integration_time_s

    @property
    def is_open(self) -> bool:
        return self._opened

    # ------------------------------------------------------------------
    # Simulation-specific controls
    # ------------------------------------------------------------------

    def set_analyte_concentration(self, concentration_ppm: float) -> None:
        """Change the simulated analyte concentration (ppm)."""
        self._concentration_ppm = float(concentration_ppm)

    def set_noise_level(self, noise_fraction: float) -> None:
        """Change the shot-noise level (fraction of peak signal)."""
        self._noise_fraction = float(noise_fraction)

    # ------------------------------------------------------------------
    # Internal spectrum generation
    # ------------------------------------------------------------------

    def _elapsed_hours(self) -> float:
        if self._open_time is None:
            return 0.0
        dt = datetime.datetime.now(datetime.timezone.utc) - self._open_time
        return dt.total_seconds() / 3600.0

    def _generate_spectrum(self) -> np.ndarray:
        """Generate one physically plausible spectrum for the current state."""
        if self._modality == "lspr":
            return self._lspr_spectrum()
        elif self._modality == "fluorescence":
            return self._fluorescence_spectrum()
        else:
            return self._absorbance_spectrum()

    def _lspr_spectrum(self) -> np.ndarray:
        """Lorentzian LSPR peak with Langmuir concentration response + thermal drift."""
        c = self._concentration_ppm
        # Langmuir shift: Δλ = Δλ_max * c / (K_D + c)
        # Example LSPR defaults: Δλ_max = -15 nm, K_D = 5 ppm
        delta_lam_max = -15.0
        k_d = 5.0
        delta_lam = delta_lam_max * c / (k_d + c) if c > 0 else 0.0

        # Thermal drift: linear trend in peak position
        drift = self._drift_nm_per_hour * self._elapsed_hours()

        peak_center = self._peak_ref + delta_lam + drift

        # Lorentzian line shape: I(λ) = A / [1 + ((λ - λ₀) / (Γ/2))²]
        gamma_half = _LSPR_FWHM / 2.0
        lorentzian = _LSPR_AMP / (1.0 + ((self._wl - peak_center) / gamma_half) ** 2)

        # Broadband scattering background (exponential decay)
        background = 200.0 * np.exp(-0.003 * (self._wl - _LSPR_WL_START))

        # Shot noise: Poisson approximation = sqrt(signal) × noise_fraction × sqrt(max)
        signal = lorentzian + background
        noise_scale = self._noise_fraction * math.sqrt(_LSPR_AMP)
        noise = self._rng.normal(0.0, noise_scale, size=len(self._wl))

        return np.clip(signal + noise, 0.0, 65535.0)

    def _fluorescence_spectrum(self) -> np.ndarray:
        """Gaussian emission peak — intensity scales linearly with concentration."""
        c = self._concentration_ppm
        # Intensity: Beer-Lambert linear regime (low c)
        peak_amplitude = 20000.0 * c / max(c, 0.01)  # saturates at high c
        if c <= 0.0:
            peak_amplitude = 0.0

        sigma = 25.0  # nm
        gaussian = peak_amplitude * np.exp(-0.5 * ((self._wl - self._peak_ref) / sigma) ** 2)

        # Rayleigh scattering background (λ⁻⁴)
        background = 300.0 * (self._peak_ref / self._wl) ** 4

        signal = gaussian + background
        noise = self._rng.normal(0.0, self._noise_fraction * max(float(signal.max()), 1.0), size=len(self._wl))
        return np.clip(signal + noise, 0.0, None)

    def _absorbance_spectrum(self) -> np.ndarray:
        """Beer-Lambert absorbance spectrum: A = ε·c·l."""
        c = self._concentration_ppm
        # Molar absorptivity ε ≈ 1.5×10⁴ L/(mol·cm), path 1 cm, c in ppm
        molar_mass = 46.07  # g/mol (Ethanol-equivalent)
        eps = 1.5e4          # L/(mol·cm)
        path = 1.0           # cm
        c_molar = (c * 1e-6 * 1000.0) / molar_mass  # ppm → mol/L (dilute aqueous)
        a_peak = eps * c_molar * path

        sigma = 30.0  # nm
        absorbance = a_peak * np.exp(-0.5 * ((self._wl - self._peak_ref) / sigma) ** 2)

        noise = self._rng.normal(0.0, self._noise_fraction, size=len(self._wl))
        return np.clip(absorbance + noise, 0.0, None)
