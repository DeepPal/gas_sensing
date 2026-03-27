"""
spectraagent.drivers.thorlabs
==============================
Hardware driver adapter for ThorLabs CCS200 spectrometer.

Wraps ``gas_analysis.acquisition.ccs200_realtime.RealtimeAcquisitionService``
(callback-based) into the blocking ``AbstractHardwareDriver`` interface using
a thread-safe queue. The acquisition thread fills the queue; ``read_spectrum()``
blocks until the next frame arrives.

Hardware-specific notes (from calibrated experience):
- Integration time 50 ms → ~20 Hz (410 ms total including cooldown)
- Error -1073807343: device connected but not powered
- Error -1073807339: stale VISA handle — reconnect required
- Always call ``disconnect()`` in a try/finally block
"""
from __future__ import annotations

import logging
import queue
from typing import Any

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

log = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 4
_READ_TIMEOUT_S = 5.0


class ThorlabsCCSDriver(AbstractHardwareDriver):
    """Blocking driver adapter for ThorLabs CCS200 via RealtimeAcquisitionService.

    The underlying service runs a callback thread continuously. This adapter
    bridges the callback → blocking-read interface using a bounded Queue.

    Parameters
    ----------
    integration_time_ms:
        Spectrometer integration time in milliseconds.
    resource_string:
        VISA resource string, e.g. ``"USB0::0x1313::0x8089::M00840499::RAW"``.
        Pass ``None`` for auto-discovery.
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        resource_string: str | None = None,
    ) -> None:
        # Import here so that missing DLL does not crash at module import time
        from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService

        self._svc = RealtimeAcquisitionService(
            integration_time_ms=integration_time_ms,
            resource_string=resource_string,
        )
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._wavelengths: np.ndarray | None = None
        self._connected: bool = False

    def connect(self) -> None:
        """Connect to CCS200 and start the acquisition thread."""
        self._svc.connect()
        self._wavelengths = self._svc.wavelengths.copy()
        self._svc.register_callback(self._on_sample)
        self._svc.start()
        self._connected = True
        log.info(
            "ThorlabsCCSDriver connected — %d pixels, %.1f–%.1f nm",
            len(self._wavelengths),
            self._wavelengths[0],
            self._wavelengths[-1],
        )

    def disconnect(self) -> None:
        """Stop acquisition and release the VISA handle."""
        try:
            self._svc.stop()
        finally:
            self._connected = False
            log.info("ThorlabsCCSDriver disconnected")

    def get_wavelengths(self) -> np.ndarray:
        if self._wavelengths is None:
            raise RuntimeError("Not connected — call connect() first")
        return self._wavelengths.copy()

    def read_spectrum(self) -> np.ndarray:
        """Block until the next frame arrives from the acquisition thread."""
        try:
            return self._frame_queue.get(timeout=_READ_TIMEOUT_S)
        except queue.Empty as exc:
            raise RuntimeError(
                f"No spectrum received within {_READ_TIMEOUT_S} s — "
                "check hardware connection"
            ) from exc

    def get_integration_time_ms(self) -> float:
        return float(self._svc.integration_time_ms)

    def set_integration_time_ms(self, ms: float) -> None:
        self._svc.integration_time_ms = ms

    @property
    def name(self) -> str:
        return "ThorLabs CCS200"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _on_sample(self, data: dict[str, Any]) -> None:
        """Callback invoked by the acquisition thread on each new frame."""
        frame = np.array(data["intensities"], dtype=np.float64)
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put_nowait(frame)
