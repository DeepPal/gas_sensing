"""Adaptive peak detection and analysis for gas sensors."""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from scipy import stats

__all__ = ['find_peaks_adaptive', 'analyze_peak_adaptive']

@dataclass
class AdaptivePeakParams:
    """Dynamic peak detection parameters."""
    prominence_min: float    # Minimum peak prominence
    prominence_rel: float    # Relative prominence threshold
    width_min: float        # Minimum peak width (points)
    width_max: float        # Maximum peak width (points)
    distance_min: float     # Minimum peak distance (points)
    snr_threshold: float    # Minimum SNR for valid peak
    quality_threshold: float # Minimum quality score

def estimate_noise_level(signal_data: np.ndarray, 
                        window_size: int = 5) -> Tuple[float, float, float]:
    """Estimate noise level using detrended fluctuation analysis."""
    noise_windows = []
    trend_windows = []
    
    for i in range(len(signal_data) - window_size + 1):
        window = signal_data[i:i+window_size]
        # Detrend window
        x = np.arange(window_size)
        slope, intercept = np.polyfit(x, window, 1)
        trend = slope * x + intercept
        detrended = window - trend
        
        noise_windows.append(np.std(detrended))
        trend_windows.append(np.std(trend))
    
    # Calculate noise statistics
    noise_level = np.median(noise_windows)
    trend_level = np.median(trend_windows)
    noise_variability = stats.iqr(noise_windows) / noise_level
    
    return noise_level, trend_level, noise_variability

def get_adaptive_parameters(signal_data: np.ndarray,
                          wavelengths: np.ndarray,
                          region: str = 'auto') -> AdaptivePeakParams:
    """Calculate adaptive peak detection parameters based on signal characteristics."""
    # Get noise characteristics
    noise_level, trend_level, noise_variability = estimate_noise_level(signal_data)
    
    # Base parameters for each region
    base_params = {
        'EtOH': {
            'prominence_min': 0.05,
            'prominence_rel': 0.1,
            'width_min': 3,
            'width_max': 30,
            'distance_min': 10,
            'snr_threshold': 3.0,
            'quality_threshold': 0.5
        },
        'IPA': {
            'prominence_min': 0.1,
            'prominence_rel': 0.15,
            'width_min': 5,
            'width_max': 20,
            'distance_min': 15,
            'snr_threshold': 3.0,
            'quality_threshold': 0.5
        },
        'MeOH': {
            'prominence_min': 0.2,
            'prominence_rel': 0.2,
            'width_min': 8,
            'width_max': 15,
            'distance_min': 20,
            'snr_threshold': 3.0,
            'quality_threshold': 0.5
        }
    }
    
    # Get base parameters
    params = base_params.get(region, base_params['IPA']).copy()
    
    # Adjust based on noise level
    noise_factor = max(1.0, noise_level / 0.01)  # Reference noise level: 0.01
    params['prominence_min'] *= noise_factor
    params['snr_threshold'] *= np.sqrt(noise_factor)
    
    # Adjust based on noise variability
    if noise_variability > 0.5:  # High noise variability
        params['width_min'] *= 1.5
        params['distance_min'] *= 1.5
        params['quality_threshold'] *= 1.2
    
    # Adjust based on trend level
    if trend_level > noise_level * 2:  # Strong baseline drift
        params['prominence_rel'] *= 1.5
        params['quality_threshold'] *= 1.2
    
    return AdaptivePeakParams(**params)

def find_peaks_adaptive(signal: np.ndarray,
                      wavelengths: np.ndarray,
                      region: str,
                      peak_wavelength: float) -> Tuple[np.ndarray, Dict]:
    """Find peaks with adaptive parameters."""
    # Determine window size based on wavelength resolution
    wavelength_resolution = np.mean(np.diff(wavelengths))
    window_points = int(2.0 / wavelength_resolution)  # 2 nm window
    window_size = max(3, min(window_points, len(signal) // 4))
    
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1
    
    # Smooth signal
    smoothed = savgol_filter(signal, window_size, 3)
    
    # Calculate noise level
    diff = np.diff(smoothed)
    mad = np.median(np.abs(diff - np.median(diff)))
    noise_level = 1.4826 * mad  # Estimate of standard deviation
    
    # Determine prominence and width
    prominence = max(noise_level * 3, np.std(signal) / 4)
    min_width = max(3, int(1.0 / wavelength_resolution))  # At least 1 nm wide
    max_width = min(int(5.0 / wavelength_resolution), len(signal) // 4)  # At most 5 nm wide
    
    # Find peaks
    peaks, properties = find_peaks(
        smoothed,
        prominence=prominence,
        width=(min_width, max_width),
        distance=min_width * 2,
        height=(None, None),  # No height constraints
        rel_height=0.5  # Use half prominence for width calculation
    )
    
    # Filter peaks by wavelength
    if len(peaks) > 0:
        # Find peaks within ±5 nm of target
        wavelength_mask = np.abs(wavelengths[peaks] - peak_wavelength) <= 5.0
        peaks = peaks[wavelength_mask]
        
        if len(peaks) > 0:
            # Sort by prominence
            prominences = properties['prominences'][wavelength_mask]
            sort_idx = np.argsort(prominences)[::-1]
            peaks = peaks[sort_idx]
            
            # Take most prominent peak
            peaks = peaks[:1]
            
            # Update properties
            properties = {
                key: val[wavelength_mask][sort_idx][:1] if isinstance(val, np.ndarray)
                else val for key, val in properties.items()
            }
    
    return peaks, properties

def analyze_peak_adaptive(signal: np.ndarray,
                       wavelengths: np.ndarray,
                       peak_idx: int,
                       peak_info: Dict) -> Dict:
    """Analyze peak characteristics."""
    # Extract peak properties
    prominence = peak_info['prominences'][0]
    width = peak_info['widths'][0]
    left_base = int(peak_info['left_bases'][0])
    right_base = int(peak_info['right_bases'][0])
    
    # Calculate peak position and height
    peak_pos = wavelengths[peak_idx]
    peak_height = signal[peak_idx]
    
    # Calculate baseline using linear fit
    base_x = np.array([wavelengths[left_base], wavelengths[right_base]])
    base_y = np.array([signal[left_base], signal[right_base]])
    base_fit = np.polyfit(base_x, base_y, 1)
    baseline = np.polyval(base_fit, wavelengths[peak_idx])
    
    # Calculate noise using MAD in baseline regions
    left_noise = signal[max(0, left_base-10):left_base+1]
    right_noise = signal[right_base:min(len(signal), right_base+11)]
    noise_region = np.concatenate([left_noise, right_noise])
    noise = 1.4826 * np.median(np.abs(noise_region - np.median(noise_region)))
    
    # Calculate SNR
    snr = (peak_height - baseline) / noise if noise > 0 else 0
    
    # Calculate quality metrics
    peak_range = wavelengths[right_base] - wavelengths[left_base]
    asymmetry = abs(wavelengths[peak_idx] - (wavelengths[left_base] + wavelengths[right_base])/2) / peak_range
    quality = prominence * snr * (1 - asymmetry) / peak_range if peak_range > 0 else 0
    
    # Calculate uncertainties
    wavelength_resolution = np.mean(np.diff(wavelengths))
    pos_uncertainty = max(width / (2 * np.sqrt(2 * np.log(2))), wavelength_resolution)
    height_uncertainty = noise
    
    # Calculate R-squared
    peak_region = signal[left_base:right_base+1]
    wavelength_region = wavelengths[left_base:right_base+1]
    
    # Fit Gaussian with baseline
    def gaussian_with_baseline(x, amp, cen, wid, m, b):
        # Avoid overflow in exponential
        exponent = -0.5 * ((x - cen) / wid)**2
        # Clip exponent to avoid overflow
        exponent = np.clip(exponent, -700, 700)
        return amp * np.exp(exponent) + m*x + b
    
    try:
        # Initial guesses
        amp_guess = peak_height - baseline
        width_guess = width * wavelength_resolution
        p0 = [
            max(amp_guess, 1e-10),  # amplitude
            peak_pos,                 # center
            max(width_guess, 0.1),    # width
            base_fit[0],             # slope
            base_fit[1]              # intercept
        ]
        
        # Fit with bounds
        bounds = (
            [1e-10, wavelength_region[0], 0.1, -np.inf, -np.inf],  # lower bounds
            [np.inf, wavelength_region[-1], 10.0, np.inf, np.inf]  # upper bounds
        )
        
        popt, pcov = curve_fit(gaussian_with_baseline, wavelength_region, peak_region,
                           p0=p0, bounds=bounds, maxfev=2000)
        fitted = gaussian_with_baseline(wavelength_region, *popt)
        residuals = peak_region - fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((peak_region - np.mean(peak_region))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Update peak position and uncertainty based on fit
        if r_squared > 0.8:  # Relaxed R² threshold
            peak_pos = popt[1]
            uncertainties = np.sqrt(np.diag(pcov))
            pos_uncertainty = min(pos_uncertainty, uncertainties[1])
    except Exception as e:
        r_squared = 0
    
    return {
        'peak_pos': peak_pos,
        'peak_height': peak_height,
        'prominence': prominence,
        'width': width,
        'snr': snr,
        'quality': quality,
        'pos_uncertainty': pos_uncertainty,
        'height_uncertainty': height_uncertainty,
        'r_squared': r_squared,
        'asymmetry': asymmetry
    }
