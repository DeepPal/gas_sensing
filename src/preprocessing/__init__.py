"""
src.preprocessing
=================
Pure spectral signal processing functions.

Public API — import from here, not from submodules:

    from src.preprocessing import smooth_spectrum, als_baseline, compute_snr
"""

from src.preprocessing.baseline import als_baseline, correct_baseline, polynomial_baseline
from src.preprocessing.denoising import (
    savgol_smooth,
    smooth_spectrum,
    spike_rejection,
    wavelet_denoise,
)
from src.preprocessing.normalization import normalize_spectrum
from src.preprocessing.quality import (
    NoiseFloorTracker,
    NoiseMetrics,
    check_saturation,
    compute_snr,
    estimate_noise_metrics,
    is_valid_spectrum,
)

__all__ = [
    "smooth_spectrum",
    "savgol_smooth",
    "spike_rejection",
    "wavelet_denoise",
    "correct_baseline",
    "als_baseline",
    "polynomial_baseline",
    "normalize_spectrum",
    "compute_snr",
    "estimate_noise_metrics",
    "is_valid_spectrum",
    "check_saturation",
    "NoiseMetrics",
    "NoiseFloorTracker",
]
