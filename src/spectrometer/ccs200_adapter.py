"""
src.spectrometer.ccs200_adapter
================================
Adapter that wraps the Thorlabs CCS200 native driver so it conforms to
the :class:`~src.spectrometer.base.AbstractSpectrometer` ABC.

This allows the CCS200 to be:
- Created via :class:`~src.spectrometer.registry.SpectrometerRegistry` using
  the ``"ccs200"`` alias.
- Used interchangeably with :class:`~src.spectrometer.simulated.SimulatedSpectrometer`
  in any code written against the ABC.
- Tested with the simulated driver, then switched to real hardware by changing
  one string.

Usage
-----
::

    from src.spectrometer import SpectrometerRegistry

    # Real hardware
    with SpectrometerRegistry.create("ccs200") as spec:
        spec.set_integration_time(0.05)
        frame = spec.acquire(accumulations=3)
        print(f"Peak at {frame.peak_wavelength:.2f} nm")

    # Swap to simulated — same code, no changes
    with SpectrometerRegistry.create("simulated", modality="lspr") as spec:
        frame = spec.acquire()

Notes
-----
The adapter delegates entirely to
:class:`~gas_analysis.acquisition.ccs200_native.CCS200Spectrometer`.
If the native driver is unavailable (DLL missing, not on Windows, no
hardware) an :class:`ImportError` is raised at construction time, not at
import time — so the module is safe to import on any platform.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np

from src.spectrometer.base import AbstractSpectrometer, SpectralFrame

log = logging.getLogger(__name__)

_NATIVE_MODEL = "Thorlabs CCS200"
_NATIVE_SERIAL_FALLBACK = "CCS200-unknown"


class CCS200Adapter(AbstractSpectrometer):
    """AbstractSpectrometer wrapper around the Thorlabs CCS200 native driver.

    Parameters
    ----------
    resource_string :
        VISA resource string, e.g. ``"USB0::0x1313::0x8089::..."``
        Pass ``None`` to auto-discover the first connected CCS200.
    integration_time_s :
        Initial integration time in seconds (default 0.05 s = 50 ms).
    """

    def __init__(
        self,
        resource_string: str | None = None,
        integration_time_s: float = 0.05,
    ) -> None:
        # Defer import so module is safe on Linux / CI
        try:
            from gas_analysis.acquisition.ccs200_native import CCS200Spectrometer
        except ImportError as exc:
            raise ImportError(
                "Thorlabs CCS200 native driver unavailable.  "
                "Ensure gas_analysis.acquisition.ccs200_native is importable "
                "and the Thorlabs DLL is installed.\n"
                f"Original error: {exc}"
            ) from exc

        self._driver: Any = CCS200Spectrometer(
            resource_string=resource_string,
            integration_time_s=integration_time_s,
        )
        self._integration_time_s: float = float(integration_time_s)
        self._accumulations: int = 1
        self._is_open: bool = False
        self._frame_index: int = 0

    # ------------------------------------------------------------------
    # AbstractSpectrometer interface
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the CCS200 USB connection.

        The native driver opens the connection during ``__init__``, but we
        call ``set_integration_time`` here to ensure the hardware is ready.
        """
        self._driver.set_integration_time(self._integration_time_s)
        self._is_open = True
        log.info("CCS200Adapter: opened (integration=%.3f s)", self._integration_time_s)

    def close(self) -> None:
        """Close the CCS200 connection and release the DLL handle."""
        try:
            if hasattr(self._driver, "close"):
                self._driver.close()
        except Exception as exc:
            log.warning("CCS200Adapter: error during close — %s", exc)
        finally:
            self._is_open = False
            log.info("CCS200Adapter: closed")

    def set_integration_time(self, seconds: float) -> None:
        """Set CCD integration time.

        Parameters
        ----------
        seconds :
            Integration time in seconds.  Must be > 0.

        Raises
        ------
        ValueError
            If *seconds* ≤ 0.
        """
        if seconds <= 0:
            raise ValueError(f"integration_time must be > 0, got {seconds}")
        self._integration_time_s = float(seconds)
        self._driver.set_integration_time(self._integration_time_s)
        log.debug("CCS200Adapter: integration_time set to %.4f s", seconds)

    def acquire(self, accumulations: int = 1) -> SpectralFrame:
        """Acquire one spectrum, averaging *accumulations* scans.

        Parameters
        ----------
        accumulations :
            Number of scans to average.  Each scan takes approximately
            ``integration_time_s + 0.4 s`` (hardware transfer overhead).

        Returns
        -------
        SpectralFrame
            Timestamped spectral frame.

        Raises
        ------
        RuntimeError
            If :meth:`open` has not been called.
        """
        if not self._is_open:
            raise RuntimeError(
                "CCS200Adapter is not open.  Call open() or use as a context manager."
            )

        accumulations = max(1, int(accumulations))
        scans: list[np.ndarray] = []

        for _ in range(accumulations):
            data = self._driver.get_data()
            scans.append(np.asarray(data, dtype=np.float64))

        intensities = np.mean(scans, axis=0) if len(scans) > 1 else scans[0]
        # Clip negatives (dark-noise floor can go slightly below zero)
        intensities = np.clip(intensities, 0.0, None)

        wl = np.asarray(self._driver.get_wavelengths(), dtype=np.float64)

        ts = datetime.datetime.now(datetime.timezone.utc)
        self._frame_index += 1

        return SpectralFrame(
            wavelengths=wl,
            intensities=intensities,
            timestamp=ts,
            integration_time_s=self._integration_time_s,
            accumulations=accumulations,
            dark_corrected=False,
            nonlinearity_corrected=False,
            serial_number=self.serial_number,
            model_name=self.model,
            metadata={
                "frame_index": self._frame_index,
                "driver": "ccs200_native",
            },
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wavelengths(self) -> np.ndarray:
        """Wavelength axis from the CCS200 calibration coefficients."""
        return np.asarray(self._driver.get_wavelengths(), dtype=np.float64)

    @property
    def n_pixels(self) -> int:
        """Number of CCD pixels (always 3648 for CCS200)."""
        return int(getattr(self._driver, "NUM_PIXELS", 3648))

    @property
    def model(self) -> str:
        return _NATIVE_MODEL

    @property
    def serial_number(self) -> str:
        try:
            return str(getattr(self._driver, "serial_number", _NATIVE_SERIAL_FALLBACK))
        except Exception:
            return _NATIVE_SERIAL_FALLBACK

    @property
    def is_open(self) -> bool:
        return self._is_open
