"""
spectraagent.drivers.simulation
================================
Simulation hardware driver — generates synthetic Lorentzian LSPR spectra.

Used for offline development, demos, and CI where no physical spectrometer
is available. The peak is centred at 720 nm with slight per-frame jitter to
simulate realistic drift.
"""
from __future__ import annotations

import time

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

_N_PIXELS: int = 3648
_WL_MIN: float = 500.0
_WL_MAX: float = 900.0
_PEAK_NM: float = 720.0
_GAMMA_NM: float = 9.0       # Lorentzian half-width at half-maximum
_AMPLITUDE: float = 0.8
_JITTER_STD: float = 0.02    # nm — per-frame peak position jitter


class SimulationDriver(AbstractHardwareDriver):
    """Synthetic spectrometer driver for testing and demos.

    Generates a Lorentzian peak at ~720 nm with Gaussian noise and slight
    per-frame jitter. ``read_spectrum()`` blocks for ``integration_time_ms``
    to faithfully simulate real acquisition timing.

    Parameters
    ----------
    integration_time_ms:
        How long ``read_spectrum()`` sleeps before returning (default 50 ms →
        ~20 Hz equivalent).
    noise_level:
        Standard deviation of Gaussian noise added to the spectrum.
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        noise_level: float = 0.002,
    ) -> None:
        self._integration_time_ms = integration_time_ms
        self._noise_level = noise_level
        self._connected: bool = False
        self._wavelengths: np.ndarray = np.linspace(_WL_MIN, _WL_MAX, _N_PIXELS)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    def read_spectrum(self) -> np.ndarray:
        """Sleep for integration time, then return a synthetic spectrum."""
        time.sleep(self._integration_time_ms / 1000.0)
        peak_wl = _PEAK_NM + float(np.random.normal(0.0, _JITTER_STD))
        lorentz = _AMPLITUDE / (1.0 + ((self._wavelengths - peak_wl) / _GAMMA_NM) ** 2)
        noise = np.random.normal(0.0, self._noise_level, _N_PIXELS)
        return np.clip(lorentz + noise, 0.0, None)

    def get_integration_time_ms(self) -> float:
        return self._integration_time_ms

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_time_ms = ms

    @property
    def name(self) -> str:
        return "Simulation"

    @property
    def is_connected(self) -> bool:
        return self._connected
