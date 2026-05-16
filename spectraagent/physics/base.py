"""
spectraagent.physics.base
=========================
Abstract base class for sensor physics plugins.

Each plugin encapsulates the signal model for a specific sensor type
(LSPR, SPR, UV-Vis, etc.). Plugins are registered via:

    [project.entry-points."spectraagent.sensor_physics"]
    lspr = "spectraagent.physics.lspr:LSPRPlugin"
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AbstractSensorPhysicsPlugin(ABC):
    """Sensor physics plugin interface.

    Methods operate only on pre-processed numpy arrays — never raw bytes or
    file handles. All methods must be safe to call from multiple threads
    (they should be stateless or use immutable state only).
    """

    @abstractmethod
    def detect_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> float | None:
        """Detect the primary spectral peak and return its wavelength in nm.

        Returns ``None`` if no valid peak is found (e.g. saturated or noisy
        spectrum).
        """

    @abstractmethod
    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,
    ) -> dict[str, float]:
        """Extract physics-meaningful features from one spectrum.

        Parameters
        ----------
        wavelengths:
            Wavelength calibration array, shape ``(N,)``.
        intensities:
            Raw spectrum intensities, shape ``(N,)``.
        reference:
            Reference spectrum intensities if available, shape ``(N,)``.
        cached_ref:
            Plugin-specific pre-computed reference object returned by
            ``compute_reference_cache()``. Pass this every frame to avoid
            redundant computation on the reference.

        Returns
        -------
        dict[str, float]
            At minimum: ``{"delta_lambda": float, "snr": float}``.
            Additional keys are plugin-specific.
        """

    @abstractmethod
    def compute_reference_cache(
        self,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> object:
        """Pre-compute an expensive reference calculation once.

        The returned object is passed as ``cached_ref`` to every subsequent
        ``extract_features()`` call, eliminating redundant computation
        (e.g. Lorentzian fitting of the reference peak).
        """

    @abstractmethod
    def calibration_priors(self) -> dict:
        """Return calibration model priors for this sensor type.

        Returns
        -------
        dict
            ``{"models": ["Langmuir", "Linear", ...], "bounds": {...}}``
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable plugin name, e.g. ``'LSPR'``."""
