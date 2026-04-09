"""
spectraagent.drivers.base
=========================
Abstract base class for all hardware spectrometer drivers.

Third-party plugins implement this interface and register via:

    [project.entry-points."spectraagent.hardware"]
    my_driver = "mypkg.driver:MyDriver"
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractHardwareDriver(ABC):
    """Spectrometer hardware driver interface.

    ``connect()`` must be called before any other method.
    ``disconnect()`` must be called when the session ends (use try/finally).

    ``read_spectrum()`` is a **blocking** call — it returns when the next
    frame is available. The acquisition loop calls it in a dedicated thread.
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the connection to the spectrometer hardware."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release all hardware resources."""

    @abstractmethod
    def get_wavelengths(self) -> np.ndarray:
        """Return the wavelength calibration array, shape ``(N,)``, in nm.

        Call once after ``connect()`` and cache the result — the calibration
        does not change within a session.
        """

    @abstractmethod
    def read_spectrum(self) -> np.ndarray:
        """Block until the next spectrum frame is available, then return it.

        Returns
        -------
        np.ndarray
            Intensity array, shape ``(N,)``, same length as ``get_wavelengths()``.
        """

    @abstractmethod
    def get_integration_time_ms(self) -> float:
        """Return the current integration time in milliseconds."""

    @abstractmethod
    def set_integration_time_ms(self, ms: float) -> None:
        """Set the integration time in milliseconds."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable driver name, e.g. ``'ThorLabs CCS200'``."""

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """True if ``connect()`` has succeeded and ``disconnect()`` not yet called."""
