"""
src.spectrometer.base
======================
Hardware-agnostic spectrometer interface for research-grade instruments.

Architecture
------------
All spectrometer drivers implement :class:`AbstractSpectrometer`.  Users write
against this interface; swapping hardware requires only changing the driver
passed at construction.  This follows the same pattern used by Thorlabs'
own Kinesis / ThorCam SDK abstraction layers.

Supported hardware targets
--------------------------
- **Thorlabs CCS series** (CCS100, CCS175, CCS200): via the existing
  ``gas_analysis.acquisition.ccs200`` DLL wrapper — set ``model="CCS200"``.
- **Simulated spectrometer**: for offline development and CI — see
  :mod:`src.spectrometer.simulated`.
- **File-backed spectrometer**: replay previously recorded sessions — see
  :class:`FileSpectrometer`.
- **Custom drivers**: subclass :class:`AbstractSpectrometer` and register via
  :meth:`SpectrometerRegistry.register`.

Quick start
-----------
::

    from src.spectrometer import SpectrometerRegistry

    with SpectrometerRegistry.create("simulated") as spec:
        spec.set_integration_time(0.05)
        dark = spec.acquire_dark()
        ref  = spec.acquire_reference()
        frame = spec.acquire()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# SpectralFrame — the atomic unit of all spectral data
# ---------------------------------------------------------------------------


@dataclass
class SpectralFrame:
    """A single acquired spectrum with full provenance metadata.

    Every measurement in the pipeline is represented as a
    :class:`SpectralFrame`.  The dataclass is intentionally flat so it
    serialises to JSON/HDF5 without helper code.

    Parameters
    ----------
    wavelengths :
        Calibrated wavelength axis in nm, shape ``(n_pixels,)``.
    intensities :
        Raw detector counts (or a.u. for simulated), shape ``(n_pixels,)``.
    timestamp :
        UTC acquisition time.  Always timezone-aware.
    integration_time_s :
        Detector integration time in seconds.
    accumulations :
        Number of scans co-added on-hardware before returning to host.
    dark_corrected :
        True when a dark spectrum has been subtracted.
    nonlinearity_corrected :
        True when the driver applied a non-linearity correction polynomial.
    serial_number :
        Instrument serial number — for audit trail / traceability.
    model_name :
        Instrument model string (e.g. ``"CCS200/M"``, ``"USB2000+"``).
    metadata :
        Driver-specific extra fields (temperature, firmware version, etc.).
    """

    wavelengths: np.ndarray
    intensities: np.ndarray
    timestamp: datetime.datetime
    integration_time_s: float
    accumulations: int = 1
    dark_corrected: bool = False
    nonlinearity_corrected: bool = False
    serial_number: str = ""
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_pixels(self) -> int:
        """Number of detector pixels."""
        return int(len(self.wavelengths))

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """(lambda_min_nm, lambda_max_nm) of the wavelength axis."""
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    @property
    def peak_wavelength(self) -> float:
        """Wavelength of maximum intensity (nm)."""
        return float(self.wavelengths[int(np.argmax(self.intensities))])

    @property
    def peak_intensity(self) -> float:
        """Maximum intensity value."""
        return float(np.max(self.intensities))

    @property
    def snr(self) -> float:
        """Crude signal-to-noise estimate: peak / std of the lower 10% of signal."""
        sorted_i = np.sort(self.intensities)
        noise = float(np.std(sorted_i[: max(1, len(sorted_i) // 10)]))
        return float(self.peak_intensity / noise) if noise > 0 else float("inf")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict (numpy arrays become lists)."""
        return {
            "wavelengths": self.wavelengths.tolist(),
            "intensities": self.intensities.tolist(),
            "timestamp": self.timestamp.isoformat(),
            "integration_time_s": self.integration_time_s,
            "accumulations": self.accumulations,
            "dark_corrected": self.dark_corrected,
            "nonlinearity_corrected": self.nonlinearity_corrected,
            "serial_number": self.serial_number,
            "model_name": self.model_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpectralFrame:
        """Reconstruct a :class:`SpectralFrame` from a serialised dict."""
        ts_raw = d["timestamp"]
        if isinstance(ts_raw, str):
            ts = datetime.datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
        else:
            ts = ts_raw
        return cls(
            wavelengths=np.asarray(d["wavelengths"], dtype=float),
            intensities=np.asarray(d["intensities"], dtype=float),
            timestamp=ts,
            integration_time_s=float(d["integration_time_s"]),
            accumulations=int(d.get("accumulations", 1)),
            dark_corrected=bool(d.get("dark_corrected", False)),
            nonlinearity_corrected=bool(d.get("nonlinearity_corrected", False)),
            serial_number=str(d.get("serial_number", "")),
            model_name=str(d.get("model_name", "")),
            metadata=dict(d.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# AbstractSpectrometer — the driver interface every hardware must satisfy
# ---------------------------------------------------------------------------


class AbstractSpectrometer(ABC):
    """Hardware-agnostic spectrometer driver interface.

    Implement this abstract class to add any spectrometer to the pipeline.
    The interface is intentionally minimal — drivers are free to expose
    additional hardware-specific methods (e.g. ``set_trigger_mode``), but
    the pipeline only requires what is declared here.

    Context-manager support is built in::

        with MySpectrometer() as spec:
            spec.set_integration_time(0.1)
            frame = spec.acquire()
            # spec.close() called automatically on exit

    Implementing a new driver
    -------------------------
    1. Subclass :class:`AbstractSpectrometer`.
    2. Implement all ``@abstractmethod`` members.
    3. Register with :class:`SpectrometerRegistry` (optional but recommended)::

           @SpectrometerRegistry.register("my_instrument")
           class MySpectrometer(AbstractSpectrometer):
               ...
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Connect to the physical instrument.

        For USB/VISA instruments: open the device handle.
        For simulated drivers: initialise internal state.
        Raises ``RuntimeError`` if the device is not reachable.
        """

    @abstractmethod
    def close(self) -> None:
        """Disconnect and release all hardware resources.

        Must be idempotent — calling ``close()`` on an already-closed
        connection must not raise.
        """

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @abstractmethod
    def set_integration_time(self, seconds: float) -> None:
        """Set the detector integration time.

        Parameters
        ----------
        seconds :
            Integration time in seconds.  The driver is responsible for
            clamping to hardware-supported limits and raising
            ``ValueError`` for out-of-range values.
        """

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    @abstractmethod
    def acquire(self, accumulations: int = 1) -> SpectralFrame:
        """Acquire one spectrum.

        Parameters
        ----------
        accumulations :
            Number of sequential scans to average on-hardware before
            returning to the host (reduces readout noise by √N).

        Returns
        -------
        SpectralFrame
            Fully-populated spectrum with provenance metadata.
        """

    # ------------------------------------------------------------------
    # Instrument metadata (read-only properties)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        """Calibrated wavelength axis in nm, shape ``(n_pixels,)``."""

    @property
    @abstractmethod
    def n_pixels(self) -> int:
        """Number of physical detector pixels."""

    @property
    @abstractmethod
    def model(self) -> str:
        """Instrument model string (e.g. ``'CCS200/M'``, ``'USB2000+'``)."""

    @property
    @abstractmethod
    def serial_number(self) -> str:
        """Factory serial number for traceability."""

    @property
    def integration_time_s(self) -> float:
        """Current integration time in seconds (may not be supported by all drivers)."""
        return float("nan")

    @property
    def is_open(self) -> bool:
        """True when the hardware connection is active."""
        return False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> AbstractSpectrometer:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # High-level convenience methods (implemented here, not overridden)
    # ------------------------------------------------------------------

    def acquire_dark(self, accumulations: int = 10) -> SpectralFrame:
        """Acquire a dark (shutter-closed) reference spectrum.

        Call this with the light source blocked.  The returned frame is
        a regular :class:`SpectralFrame` with ``metadata["is_dark"] = True``.
        Subtract from sample spectra to remove detector offset and thermal noise.
        """
        frame = self.acquire(accumulations=accumulations)
        frame.metadata["is_dark"] = True
        return frame

    def acquire_reference(self, accumulations: int = 10) -> SpectralFrame:
        """Acquire a white-light / blank reference spectrum.

        Use this to normalise subsequent sample spectra.  The returned
        frame has ``metadata["is_reference"] = True``.
        """
        frame = self.acquire(accumulations=accumulations)
        frame.metadata["is_reference"] = True
        return frame

    def acquire_sequence(
        self,
        n_frames: int,
        delay_s: float = 0.0,
    ) -> list[SpectralFrame]:
        """Acquire a time-series of ``n_frames`` spectra.

        Parameters
        ----------
        n_frames :
            Number of frames to collect.
        delay_s :
            Inter-frame delay in seconds (≥ 0).  Actual timing depends on
            integration time and hardware readout latency.

        Returns
        -------
        list[SpectralFrame]
            Frames in acquisition order.
        """
        import time

        frames: list[SpectralFrame] = []
        for _ in range(n_frames):
            frames.append(self.acquire())
            if delay_s > 0:
                time.sleep(delay_s)
        return frames

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, sn={self.serial_number!r})"
