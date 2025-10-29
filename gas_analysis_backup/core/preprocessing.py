"""Preprocessing utilities for spectral data."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import sparse, stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import spsolve
from typing import Tuple, List, Dict, Optional

def baseline_correction(wavelength: np.ndarray, intensity: np.ndarray,
                       method: str = 'polynomial', poly_order: int = 2) -> np.ndarray:
    """Remove baseline from spectrum using various methods.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        method: 'polynomial', 'rolling_min', or 'als' (asymmetric least squares)
        poly_order: Order of polynomial for 'polynomial' method
    
    Returns:
        Baseline-corrected intensity array
    """
    if method == 'polynomial':
        # Fit polynomial to estimate baseline
        coef = np.polyfit(wavelength, intensity, poly_order)
        baseline = np.polyval(coef, wavelength)
        return intensity - baseline
    
    elif method == 'rolling_min':
        # Rolling minimum with interpolation
        window = max(3, len(wavelength) // 20)  # 5% of points
        if window % 2 == 0:
            window += 1
        rolling_min = pd.Series(intensity).rolling(window=window, center=True).min()
        rolling_min = rolling_min.bfill().ffill()
        return intensity - rolling_min.values
    
    elif method == 'als':
        # Asymmetric Least Squares (better for complex baselines)
        lam = 1e5  # Smoothness
        p = 0.01   # Asymmetry
        L = len(wavelength)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        for i in range(10):  # Usually converges in < 10 iterations
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*intensity)
            w = p * (intensity > z) + (1-p) * (intensity < z)
        return intensity - z
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")


def _ensure_window(window: int, length: int) -> int:
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    return min(window, max(3, length - (1 - length % 2)))


def smooth_spectrum(intensity: np.ndarray, window: int = 11, poly_order: int = 2,
                   method: str = 'savgol') -> np.ndarray:
    """Smooth spectrum using various methods.
    
    Args:
        intensity: Intensity array
        window: Window size (must be odd)
        poly_order: Polynomial order for Savitzky-Golay
        method: 'savgol', 'moving_average', or 'gaussian'
    
    Returns:
        Smoothed intensity array
    """
    if len(intensity) == 0:
        return intensity

    window = _ensure_window(window, len(intensity))

    if method == 'savgol':
        return savgol_filter(intensity, window, min(poly_order, window - 1))
    
    if method == 'moving_average':
        kernel = np.ones(window) / window
        return np.convolve(intensity, kernel, mode='same')

    if method == 'gaussian':
        sigma = max(1.0, window / 6.0)
        return gaussian_filter1d(intensity, sigma=sigma)
    
    raise ValueError(f"Unknown smoothing method: {method}")


def preprocess_spectrum(wavelengths: np.ndarray,
                       intensity: np.ndarray,
                       smooth_window: int = 31,
                       poly_order: int = 3,
                       extra_smooth: bool = False,
                       baseline_order: Optional[int] = None,
                       smoothing_method: str = 'savgol') -> np.ndarray:
    """Smooth, baseline-correct, and normalize a spectrum."""
    if len(wavelengths) == 0:
        return intensity

    window = _ensure_window(smooth_window, len(wavelengths))

    smoothed = smooth_spectrum(intensity, window=window, poly_order=poly_order,
                               method=smoothing_method)

    if extra_smooth:
        smoothed = gaussian_filter1d(smoothed, sigma=max(1.0, window / 6.0))
        smoothed = smooth_spectrum(smoothed, window=window, poly_order=poly_order,
                                   method=smoothing_method)

    n_points = max(1, len(wavelengths) // 10)
    edges_wl = np.concatenate([wavelengths[:n_points], wavelengths[-n_points:]])
    edges_int = np.concatenate([smoothed[:n_points], smoothed[-n_points:]])
    order = baseline_order if baseline_order is not None else min(3, len(edges_wl) - 1)
    order = max(1, order)
    coef = np.polyfit(edges_wl, edges_int, order)
    baseline = np.polyval(coef, wavelengths)

    corrected = smoothed - baseline
    denom = np.max(np.abs(corrected)) or 1.0
    return corrected / denom


def normalize_spectrum(intensity: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """Normalize spectrum using various methods.
    
    Args:
        intensity: Intensity array
        method: 'minmax', 'standard', or 'area'
    
    Returns:
        Normalized intensity array
    """
    if method == 'minmax':
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        if max_val - min_val < 1e-10:
            return np.zeros_like(intensity)
        return (intensity - min_val) / (max_val - min_val)
    
    elif method == 'standard':
        mean = np.mean(intensity)
        std = np.std(intensity)
        if std < 1e-10:
            return np.zeros_like(intensity)
        return (intensity - mean) / std
    
    elif method == 'area':
        area = np.trapz(intensity)
        if abs(area) < 1e-10:
            return np.zeros_like(intensity)
        return intensity / area
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_snr(intensity: np.ndarray, signal_region: Tuple[int, int] = None,
                noise_region: Tuple[int, int] = None) -> float:
    """Compute signal-to-noise ratio.
    
    Args:
        intensity: Intensity array
        signal_region: (start, end) indices for signal region
        noise_region: (start, end) indices for noise region
    
    Returns:
        SNR value
    """
    if signal_region is None:
        # Use maximum as signal
        signal = np.max(np.abs(intensity))
    else:
        signal = np.mean(np.abs(intensity[signal_region[0]:signal_region[1]]))
    
    if noise_region is None:
        # Use edges of spectrum for noise
        edge_size = len(intensity) // 10
        noise_left = intensity[:edge_size]
        noise_right = intensity[-edge_size:]
        noise = np.std(np.concatenate([noise_left, noise_right]))
    else:
        noise = np.std(intensity[noise_region[0]:noise_region[1]])
    
    if noise < 1e-10:
        return 0.0
    return float(signal / noise)


@dataclass
class NoiseMetrics:
    rms: float
    mad: float
    spectral_entropy: float
    snr: float


def estimate_noise_metrics(wavelengths: np.ndarray,
                           intensity: np.ndarray,
                           signal_region: Optional[Tuple[int, int]] = None,
                           noise_region: Optional[Tuple[int, int]] = None) -> NoiseMetrics:
    """Estimate noise characteristics on a processed spectrum."""
    if len(intensity) == 0:
        return NoiseMetrics(rms=0.0, mad=0.0, spectral_entropy=0.0, snr=0.0)

    rms = float(np.sqrt(np.mean(np.square(intensity))))
    mad = float(np.median(np.abs(intensity - np.median(intensity))) * 1.4826)

    spec = np.abs(intensity - intensity.min())
    total = spec.sum()
    if total > 0:
        prob = spec / total
        spectral_entropy = float(-np.sum(prob * np.log2(prob + 1e-12)))
    else:
        spectral_entropy = 0.0

    snr = compute_snr(intensity, signal_region, noise_region)
    return NoiseMetrics(rms=rms, mad=mad, spectral_entropy=spectral_entropy, snr=snr)


def downsample_spectrum(wavelengths: np.ndarray,
                        intensity: np.ndarray,
                        factor: Optional[int] = None,
                        target_points: Optional[int] = None,
                        method: str = 'average') -> Tuple[np.ndarray, np.ndarray]:
    """Reduce spectral resolution for faster processing."""
    wavelengths = np.asarray(wavelengths)
    intensity = np.asarray(intensity)

    if len(wavelengths) == 0:
        return wavelengths, intensity

    if factor is None and target_points is None:
        return wavelengths, intensity

    if target_points is not None and target_points > 0:
        factor = max(1, len(wavelengths) // target_points)
    factor = max(1, factor or 1)

    if factor == 1:
        return wavelengths, intensity

    if method == 'average':
        usable = len(wavelengths) - (len(wavelengths) % factor)
        if usable == 0:
            return wavelengths, intensity
        wl_view = wavelengths[:usable].reshape(-1, factor)
        it_view = intensity[:usable].reshape(-1, factor)
        return wl_view.mean(axis=1), it_view.mean(axis=1)

    # Fallback to interpolation
    new_len = max(2, len(wavelengths) // factor)
    wl_ds = np.linspace(wavelengths[0], wavelengths[-1], new_len)
    int_ds = np.interp(wl_ds, wavelengths, intensity)
    return wl_ds, int_ds


def _mad_zscores(values: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using the median absolute deviation."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad < 1e-9:
        return np.zeros_like(values, dtype=float)
    return 0.6745 * (values - median) / mad


def detect_outliers(spectra: List[np.ndarray], threshold: float = 3.0) -> List[bool]:
    """Detect outlier spectra using various metrics.
    
    Args:
        spectra: List of intensity arrays
        threshold: Z-score threshold for outlier detection
    
    Returns:
        List of boolean flags (True = outlier)
    """
    if len(spectra) < 2:
        return [False] * len(spectra)
    
    # Stack spectra and compute metrics
    X = np.vstack(spectra)
    metrics = {
        'mean': np.mean(X, axis=1),
        'std': np.std(X, axis=1),
        'max': np.max(X, axis=1),
        'min': np.min(X, axis=1),
        'range': np.ptp(X, axis=1),
    }
    
    # Compute Z-scores for each metric
    outliers = np.zeros(len(spectra), dtype=bool)
    for values in metrics.values():
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            continue

        # Primary attempt: classical z-score
        z_scores = np.abs(stats.zscore(values, nan_policy='omit'))

        # stats.zscore returns NaN when variance is ~0; fall back to MAD-based z-scores
        if np.isnan(z_scores).all() or np.allclose(values, values[0]):
            z_scores = np.abs(_mad_zscores(values))
        else:
            nan_mask = np.isnan(z_scores)
            if np.any(nan_mask):
                z_scores[nan_mask] = np.abs(_mad_zscores(values))[nan_mask]

        outliers |= z_scores > threshold

    return outliers.tolist()
