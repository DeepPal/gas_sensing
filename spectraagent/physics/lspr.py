"""
spectraagent.physics.lspr
==========================
LSPR sensor physics plugin.

Wraps ``src.features.lspr_features`` — the underlying Lorentzian peak
detection and feature extraction code is unchanged. This class only
adapts the function-based API to the ``AbstractSensorPhysicsPlugin``
interface.

Physics: Au nanoparticle LSPR sensor. Primary signal is peak wavelength
SHIFT delta_lambda = lambda_gas - lambda_reference (nm). Negative delta_lambda
= blue-shift on analyte adsorption.
"""
from __future__ import annotations

import numpy as np

from src.features.lspr_features import (
    LSPR_SEARCH_MAX_NM,
    LSPR_SEARCH_MIN_NM,
    LSPRReference,
    compute_lspr_reference,
    detect_lspr_peak,
    extract_lspr_features,
)
from spectraagent.physics.base import AbstractSensorPhysicsPlugin


class LSPRPlugin(AbstractSensorPhysicsPlugin):
    """LSPR sensor physics plugin — wraps ``src.features.lspr_features``.

    Parameters
    ----------
    search_min_nm, search_max_nm:
        Wavelength window for peak search. Defaults match the Au-MIP sensor
        constants in ``lspr_features.py``.
    """

    def __init__(
        self,
        search_min_nm: float = LSPR_SEARCH_MIN_NM,
        search_max_nm: float = LSPR_SEARCH_MAX_NM,
    ) -> None:
        self._search_min = search_min_nm
        self._search_max = search_max_nm

    def detect_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
    ) -> float | None:
        return detect_lspr_peak(
            wavelengths, intensities, self._search_min, self._search_max
        )

    def compute_reference_cache(
        self,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> LSPRReference:
        """Pre-compute Lorentzian fit of reference spectrum (call once per session)."""
        return compute_lspr_reference(
            wavelengths, reference, self._search_min, self._search_max
        )

    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,
    ) -> dict[str, float]:
        # extract_lspr_features requires reference_intensities; use intensities
        # as a neutral stand-in when no reference is provided (shift will be 0).
        ref = reference if reference is not None else intensities
        lspr_ref = cached_ref if isinstance(cached_ref, LSPRReference) else None
        result = extract_lspr_features(
            wavelengths,
            intensities,
            reference_intensities=ref,
            lspr_ref=lspr_ref,
        )
        if result is None:
            return {"delta_lambda": 0.0, "snr": 0.0, "peak_wavelength": 0.0}

        # Use getattr with None fallback for each optional field
        def _f(val: float | None) -> float:
            return float(val) if val is not None else 0.0

        return {
            "delta_lambda": _f(result.delta_lambda),
            "snr": _f(result.snr),
            "peak_wavelength": _f(result.peak_wavelength),
            "delta_intensity_peak": _f(result.delta_intensity_peak),
            "delta_intensity_area": _f(result.delta_intensity_area),
        }

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
