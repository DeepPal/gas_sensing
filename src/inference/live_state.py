"""
src.inference.live_state
========================
Thread-safe live data store for real-time sensor acquisition.

Architecture
------------
The store is a **module-level singleton** that decouples the high-frequency
acquisition thread (~20 Hz) from the lower-frequency Streamlit UI (~1 Hz).

::

    Acquisition thread              Streamlit process
    ─────────────────               ─────────────────
    LiveDataStore.push(result, raw) LiveDataStore.get_latest(500)
    LiveDataStore.set_running(True) LiveDataStore.is_running()

Thread safety is guaranteed by a ``threading.Lock`` wrapping all mutations.
The circular buffer (``collections.deque`` with maxlen) ensures that memory
stays bounded regardless of session length.

Public API
----------
- ``LiveDataStore``   — module-level singleton instance (import and use directly)
- ``_LiveDataStore``  — underlying class (use for unit testing with fresh state)
"""

from __future__ import annotations

from collections import deque
import threading
from typing import Any

import numpy as np


class _LiveDataStore:
    """Thread-safe circular buffer for real-time pipeline results.

    Parameters
    ----------
    maxlen:
        Maximum number of result dicts to retain.  At 20 Hz, 2000 entries
        represents ~100 seconds of data — enough for the Streamlit chart.
    """

    def __init__(self, maxlen: int = 2000) -> None:
        self._deque: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        # Spectral data (written once at connection, read frequently)
        self._wavelengths: np.ndarray | None = None
        self._latest_intensities: np.ndarray | None = None

        # Session metadata
        self._session_meta: dict[str, Any] = {}
        self._is_running: bool = False
        self._sample_count: int = 0
        self._last_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Writer API (acquisition thread)
    # ------------------------------------------------------------------

    def set_wavelengths(self, wavelengths: np.ndarray) -> None:
        """Store the spectrometer wavelength axis (call once per session)."""
        with self._lock:
            self._wavelengths = np.asarray(wavelengths, dtype=float)

    def set_session_meta(self, meta: dict[str, Any]) -> None:
        """Store session metadata (gas label, start time, etc.)."""
        with self._lock:
            self._session_meta = dict(meta)

    def set_running(self, running: bool) -> None:
        """Set the acquisition-active flag."""
        with self._lock:
            self._is_running = running

    def push(
        self,
        result_dict: dict[str, Any],
        raw_intensities: np.ndarray | None = None,
    ) -> None:
        """Enqueue a processed frame (called ~20 Hz from the acquisition thread).

        Parameters
        ----------
        result_dict:
            Flat dict with pipeline results (peak_wavelength, concentration_ppm,
            snr, gas_type, etc.).
        raw_intensities:
            Optional raw spectrum for the live chart display.
        """
        # Copy outside the lock — np.asarray can allocate; holding the lock
        # during allocation blocks the dashboard reader thread unnecessarily.
        result_copy = dict(result_dict)
        intensities_copy = (
            np.asarray(raw_intensities, dtype=float).copy()
            if raw_intensities is not None
            else None
        )
        with self._lock:
            self._deque.append(result_copy)
            self._sample_count += 1
            self._last_result = result_copy
            if intensities_copy is not None:
                self._latest_intensities = intensities_copy

    # ------------------------------------------------------------------
    # Reader API (Streamlit / dashboard thread)
    # ------------------------------------------------------------------

    def get_latest(self, n: int = 500) -> list[dict[str, Any]]:
        """Return the last *n* result dicts as a thread-safe snapshot.

        Parameters
        ----------
        n:
            Maximum number of results to return.

        Returns
        -------
        list of dict
            Most-recent first (index 0 = oldest of the window).
        """
        with self._lock:
            items = list(self._deque)
        return items[-n:]

    def get_latest_spectrum(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return ``(wavelengths, intensities)`` for the live spectrum chart.

        Returns ``None`` if no frame has been received yet.
        """
        with self._lock:
            if self._wavelengths is None or self._latest_intensities is None:
                return None
            return self._wavelengths.copy(), self._latest_intensities.copy()

    def get_session_meta(self) -> dict[str, Any]:
        """Return a snapshot of the current session metadata."""
        with self._lock:
            return dict(self._session_meta)

    def is_running(self) -> bool:
        """Return True if acquisition is currently active."""
        with self._lock:
            return self._is_running

    def get_sample_count(self) -> int:
        """Return the total number of samples pushed since the last clear."""
        with self._lock:
            return self._sample_count

    def get_last_result(self) -> dict[str, Any] | None:
        """Return the most recent result dict, or None if the buffer is empty."""
        with self._lock:
            return dict(self._last_result) if self._last_result else None

    def get_buffer_size(self) -> int:
        """Return the number of results currently in the buffer."""
        with self._lock:
            return len(self._deque)

    def get_wavelengths(self) -> np.ndarray | None:
        """Return the wavelength axis, or None if not yet set."""
        with self._lock:
            return self._wavelengths.copy() if self._wavelengths is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all state (call between sessions)."""
        with self._lock:
            self._deque.clear()
            self._latest_intensities = None
            self._session_meta = {}
            self._is_running = False
            self._sample_count = 0
            self._last_result = None
            # Keep _wavelengths — spectrometer axis doesn't change between sessions


# Module-level singleton — import this directly
LiveDataStore = _LiveDataStore(maxlen=2000)
