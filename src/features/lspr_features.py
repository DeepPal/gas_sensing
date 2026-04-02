"""Compatibility shim — all content has moved to spectral_features.py.

Import from ``src.features.spectral_features`` directly for new code.
This module re-exports everything for backwards compatibility.
"""
from src.features.spectral_features import *  # noqa: F401, F403
from src.features.spectral_features import (  # noqa: F401
    LSPR_SEARCH_MAX_NM,
    LSPR_SEARCH_MIN_NM,
    LSPRFeatures,
    LSPRReference,
    KineticFeatures,
    compute_lspr_reference,
    concentration_from_shift,
    detect_all_peaks,
    detect_lspr_peak,
    estimate_shift_xcorr,
    estimate_response_kinetics,
    extract_lspr_features,
    refine_peak_centroid,
)
