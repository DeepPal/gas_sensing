"""Spectral sensor feature extraction public API."""

from src.features.cross_peak_features import (
    CrossPeakPCA,
    extract_cross_peak_features,
    pattern_match_scores,
    shift_ratios,
    shift_vector_direction,
    shift_vector_norm,
    spectral_angle,
    spectral_entropy,
    spectral_similarity_scores,
    cosine_similarity_to_reference,
)
from src.features.compensation import (
    AdaptiveDriftCorrector,
    EnvironmentalCompensator,
    differential_peak_correction,
    polynomial_detrend,
)
# Primary import from the sensor-agnostic module name
from src.features.spectral_features import (
    LSPR_SEARCH_MAX_NM,
    LSPR_SEARCH_MIN_NM,
    LSPRFeatures,
    LSPRReference,
    compute_lspr_reference,
    concentration_from_shift,
    detect_all_peaks,
    detect_lspr_peak,
    estimate_shift_xcorr,
    extract_lspr_features,
    refine_peak_centroid,
)
# lspr_features kept as a compatibility shim — imports from spectral_features

__all__ = [
    "LSPRFeatures",
    "LSPRReference",
    "LSPR_SEARCH_MIN_NM",
    "LSPR_SEARCH_MAX_NM",
    "compute_lspr_reference",
    "extract_lspr_features",
    "detect_lspr_peak",
    "detect_all_peaks",
    "estimate_shift_xcorr",
    "concentration_from_shift",
    "refine_peak_centroid",
    # Cross-peak features
    "CrossPeakPCA",
    "extract_cross_peak_features",
    "pattern_match_scores",
    "shift_ratios",
    "shift_vector_direction",
    "shift_vector_norm",
    "spectral_angle",
    "spectral_entropy",
    "spectral_similarity_scores",
    "cosine_similarity_to_reference",
    # Environmental compensation
    "AdaptiveDriftCorrector",
    "EnvironmentalCompensator",
    "differential_peak_correction",
    "polynomial_detrend",
]
