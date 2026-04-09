"""
spectraagent.physics.lspr
==========================
Lorentzian-peak sensor physics plugin.

Adapts the function-based API in ``src.features.lspr_features`` to the
``AbstractSensorPhysicsPlugin`` interface.  This plugin works for ANY sensor
that produces one or more Lorentzian-shaped extinction/absorption peaks in
the measured spectrum — it is NOT specific to LSPR sensors.

Primary signal: peak wavelength SHIFT Δλ = λ_analyte − λ_reference (nm).
Negative Δλ = blue-shift; positive Δλ = red-shift.  The plugin auto-detects
the peak position(s) from the reference spectrum on first use.
"""
from __future__ import annotations

import numpy as np

from spectraagent.physics.base import AbstractSensorPhysicsPlugin
from src.features.lspr_features import (
    LSPR_SEARCH_MAX_NM,
    LSPR_SEARCH_MIN_NM,
    LSPRReference,
    compute_lspr_reference,
    detect_all_peaks,
    detect_lspr_peak,
    extract_lspr_features,
)


class LSPRPlugin(AbstractSensorPhysicsPlugin):
    """Lorentzian-peak sensor physics plugin.

    Sensor-agnostic: the search window defaults cover the full visible/NIR
    range (configured via ``spectraagent.toml``).  The actual peak position(s)
    are discovered at runtime from the reference spectrum, not hardcoded.

    Supports both single-peak and multi-peak sensor configurations:
    - Single peak  → one Δλ channel per analyte
    - Multiple peaks → multi-channel feature vector; distinct spectral
      signatures enable discrimination of multiple analytes or cross-
      interference characterisation from a single broadband sensor.

    Parameters
    ----------
    search_min_nm, search_max_nm:
        Wavelength search window (nm).  Defaults match the module-level
        constants (full visible/NIR range).  Narrow this in
        ``spectraagent.toml`` for sensors with a known spectral band.
    """

    def __init__(
        self,
        search_min_nm: float = LSPR_SEARCH_MIN_NM,
        search_max_nm: float = LSPR_SEARCH_MAX_NM,
    ) -> None:
        self._search_min = search_min_nm
        self._search_max = search_max_nm
        # Discovered at runtime from the reference spectrum — never hardcoded.
        self._roi_center: float | None = None
        self._ref_peak_wls: list[float] = []  # all detected reference peaks

    def update_from_reference(self, peak_wls: list[float]) -> None:
        """Called after reference capture to register detected peak positions.

        Stores all detected peaks and sets the primary ROI center from the
        most prominent peak.  Subsequent calls to ``extract_features`` will
        use these positions instead of any hardcoded default.
        """
        if peak_wls:
            self._ref_peak_wls = list(peak_wls)
            self._roi_center = peak_wls[0]  # primary peak (highest prominence)

    def detect_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> float | None:
        """Return the single most-prominent peak (primary channel)."""
        return detect_lspr_peak(
            wavelengths, intensities, self._search_min, self._search_max
        )

    def detect_peaks(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> list[float]:
        """Return ALL detected peaks — enables multi-peak feature extraction."""
        return detect_all_peaks(
            wavelengths, intensities, self._search_min, self._search_max
        )

    def compute_reference_cache(
        self,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> LSPRReference:
        """Pre-compute Lorentzian fit of reference spectrum (call once per session)."""
        return compute_lspr_reference(
            wavelengths, reference,
            search_min=self._search_min,
            search_max=self._search_max,
        )

    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,
    ) -> dict[str, float]:
        """Extract features for all detected peaks.

        Returns a flat dict with keys indexed by peak number:
        ``delta_lambda_0``, ``delta_lambda_1``, ... for multi-peak sensors.
        Single-peak sensors return the same keys without index suffix for
        backward compatibility (``delta_lambda``, ``peak_wavelength``, ...).
        """
        ref = reference if reference is not None else intensities
        lspr_ref = cached_ref if isinstance(cached_ref, LSPRReference) else None

        def _f(val: float | None) -> float:
            return float(val) if val is not None else 0.0

        # Determine peak positions to process
        peak_centers: list[float | None]
        if self._ref_peak_wls:
            peak_centers = list(self._ref_peak_wls)
        else:
            peak_centers = [self._roi_center]  # None → auto-detect inside extract_lspr_features

        out: dict[str, float] = {}
        for i, center in enumerate(peak_centers):
            result = extract_lspr_features(
                wavelengths,
                intensities,
                reference_intensities=ref,
                lspr_ref=lspr_ref if i == 0 else None,  # cached ref is for primary peak only
                roi_center=center,
            )
            suffix = "" if len(peak_centers) == 1 else f"_{i}"
            if result is None:
                out.update({
                    f"delta_lambda{suffix}": 0.0,
                    f"snr{suffix}": 0.0,
                    f"peak_wavelength{suffix}": 0.0,
                })
            else:
                out.update({
                    f"delta_lambda{suffix}": _f(result.delta_lambda),
                    f"snr{suffix}": _f(result.snr),
                    f"peak_wavelength{suffix}": _f(result.peak_wavelength),
                    f"delta_intensity_peak{suffix}": _f(result.delta_intensity_peak),
                    f"delta_intensity_area{suffix}": _f(result.delta_intensity_area),
                    f"delta_fwhm_nm{suffix}": _f(result.delta_fwhm_nm),
                    f"delta_amplitude{suffix}": _f(result.delta_amplitude),
                })

        # Always expose primary alias for backward compatibility
        if "delta_lambda_0" in out and "delta_lambda" not in out:
            out["delta_lambda"] = out["delta_lambda_0"]
            out["peak_wavelength"] = out["peak_wavelength_0"]
            out["snr"] = out["snr_0"]

        return out

    def calibration_priors(self) -> dict:
        return {
            "models": ["Langmuir", "Freundlich", "Hill", "Linear"],
            "bounds": {
                "Langmuir": {"Bmax": (0.0, 10.0), "Kd": (1e-6, 1.0)},
                "Linear": {"slope": (-100.0, 0.0), "intercept": (-5.0, 5.0)},
            },
        }

    @property
    def name(self) -> str:
        return "LSPR"
