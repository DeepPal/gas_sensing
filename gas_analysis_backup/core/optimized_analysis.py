"""Optimized spectral analysis for NCF gas sensor."""

import numpy as np
from scipy import signal, stats
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PeakResult:
    """Peak analysis result."""
    wavelength: float      # Peak center wavelength (nm)
    height: float         # Peak height
    width: float          # FWHM (nm)
    area: float          # Integrated peak area
    asymmetry: float     # Peak asymmetry
    snr: float          # Signal-to-noise ratio
    quality: float       # Overall quality score

@dataclass
class ShiftResult:
    """Shift analysis result."""
    wavelength: float     # Reference wavelength (nm)
    shift: float         # Detected shift (nm)
    confidence: float    # Confidence in shift (0-1)
    peak_ref: PeakResult # Reference peak
    peak_sam: PeakResult # Sample peak


def adaptive_smooth(wavelengths: np.ndarray,
                   intensity: np.ndarray,
                   region: str = 'auto') -> np.ndarray:
    """Adaptive smoothing based on region and noise level."""
    # Estimate noise using differences
    diff = np.diff(intensity)
    noise_level = np.median(np.abs(diff)) * 1.4826  # MAD to sigma
    
    # Adjust parameters based on region and noise
    if region == 'EtOH' or noise_level > 0.05:
        # Strong smoothing for noisy regions
        intensity = gaussian_filter1d(intensity, sigma=3)
        window = 41
        poly_order = 4
    elif region == 'IPA' or noise_level > 0.02:
        # Moderate smoothing
        intensity = gaussian_filter1d(intensity, sigma=2)
        window = 31
        poly_order = 3
    else:  # MeOH or auto
        # Light smoothing for clean regions
        window = 21
        poly_order = 3
    
    if noise_level > 0.1:
        # Extra smoothing for very noisy data
        intensity = gaussian_filter1d(intensity, sigma=5)
    
    # Ensure window is odd
    window = min(window, len(wavelengths)-1)
    if window % 2 == 0:
        window += 1
    
    # Apply Savitzky-Golay filter
    return signal.savgol_filter(intensity, window, poly_order)

def robust_baseline(wavelengths: np.ndarray,
                   intensity: np.ndarray,
                   region: str = 'auto') -> np.ndarray:
    """Robust baseline estimation using iterative polynomial fitting."""
    # Adjust parameters based on region
    if region == 'EtOH':
        poly_order = 4  # Higher order for complex baselines
        num_iter = 4
    elif region == 'IPA':
        poly_order = 3  # Moderate complexity
        num_iter = 3
    else:  # MeOH or auto
        poly_order = 2  # Simple baseline
        num_iter = 2
    
    # Iterative polynomial fitting
    x = np.arange(len(intensity))
    baseline = np.zeros_like(intensity)
    working_y = intensity.copy()
    
    for _ in range(num_iter):
        # Fit polynomial
        coef = np.polyfit(x, working_y, poly_order)
        baseline = np.polyval(coef, x)
        
        # Update working data - only keep points below baseline
        diff = working_y - baseline
        working_y[diff > 0] = baseline[diff > 0]
    
    return baseline

def find_peak_position(x: np.ndarray, y: np.ndarray, initial_pos: float) -> float:
    """Find peak position using center of mass in local window."""
    # Find local window around initial position
    idx = np.argmin(np.abs(x - initial_pos))
    window = 5  # points on each side
    start = max(0, idx - window)
    end = min(len(x), idx + window + 1)
    
    # Get local region
    x_local = x[start:end]
    y_local = y[start:end]
    
    # Subtract baseline
    baseline = min(y_local)
    y_base = y_local - baseline
    
    # Calculate center of mass
    if np.sum(y_base) > 0:
        return np.sum(x_local * y_base) / np.sum(y_base)
    return x[idx]

def analyze_peak_shape(x: np.ndarray, y: np.ndarray, peak_pos: float) -> Tuple[float, float, float]:
    """Analyze peak shape to get height, width, and asymmetry."""
    # Find peak index
    peak_idx = np.argmin(np.abs(x - peak_pos))
    
    # Get height above baseline
    baseline = np.median(y)
    height = y[peak_idx] - baseline
    
    # Find width at half maximum
    half_height = height / 2 + baseline
    above_half = y >= half_height
    if np.sum(above_half) >= 2:
        left_idx = np.where(above_half)[0][0]
        right_idx = np.where(above_half)[0][-1]
        width = x[right_idx] - x[left_idx]
    else:
        width = 5.0  # Default width
    
    # Calculate asymmetry using areas
    left_area = np.trapz(y[:peak_idx] - baseline, x[:peak_idx])
    right_area = np.trapz(y[peak_idx:] - baseline, x[peak_idx:])
    asymmetry = left_area / right_area if right_area != 0 else 1.0
    
    return height, width, asymmetry

def analyze_peak(wavelengths: np.ndarray,
                intensity: np.ndarray,
                peak_idx: int,
                window: int = 20,
                debug: bool = False) -> PeakResult:
    """Analyze peak characteristics.
    
    Args:
        wavelengths: Wavelength array
        intensity: Intensity array
        peak_idx: Index of peak center
        window: Analysis window size
        debug: Print debug information
    
    Returns:
        Peak analysis result
    """
    # Extract peak region
    start_idx = max(0, peak_idx - window)
    end_idx = min(len(wavelengths), peak_idx + window)
    
    peak_region = intensity[start_idx:end_idx]
    peak_wl = wavelengths[start_idx:end_idx]
    
    # Basic measurements
    height = intensity[peak_idx]
    
    # Fit Gaussian for robust width estimation
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-((x - cen) / wid)**2)
    
    try:
        popt, _ = curve_fit(gaussian, peak_wl, peak_region,
                           p0=[height, wavelengths[peak_idx], 5])
        width = abs(2.355 * popt[2])  # FWHM = 2.355 * sigma
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        # Fallback to simple FWHM calculation
        half_height = (height + np.min(peak_region)) / 2
        above_half = peak_region >= half_height
        if np.sum(above_half) >= 2:
            left_idx = np.where(above_half)[0][0]
            right_idx = np.where(above_half)[0][-1]
            width = peak_wl[right_idx] - peak_wl[left_idx]
        else:
            width = np.nan
    
    # Calculate area using trapezoidal integration
    area = np.trapz(peak_region - np.min(peak_region), peak_wl)
    
    # Calculate asymmetry using moments
    center_mass = np.sum(peak_wl * peak_region) / np.sum(peak_region)
    left_moment = np.sum(peak_region[peak_wl <= center_mass])
    right_moment = np.sum(peak_region[peak_wl > center_mass])
    asymmetry = left_moment / right_moment if right_moment != 0 else np.inf
    
    # Calculate prominence
    prominence = height - max(
        np.min(peak_region[:peak_idx-start_idx+1]),
        np.min(peak_region[peak_idx-start_idx:])
    )
    
    # Find refined peak position
    peak_pos = find_peak_position(peak_wl, peak_region, wavelengths[peak_idx])
    
    # Analyze peak shape
    height, width, asymmetry = analyze_peak_shape(peak_wl, peak_region, peak_pos)
    
    # Calculate signal and noise
    # Use sliding window for noise estimation
    window_size = 5
    noise_windows = []
    for i in range(len(peak_region) - window_size):
        window = peak_region[i:i+window_size]
        # Detrend window
        slope, intercept = np.polyfit(range(window_size), window, 1)
        detrended = window - (slope * np.arange(window_size) + intercept)
        noise_windows.append(np.std(detrended))
    
    # Use median of window noise estimates
    noise = np.median(noise_windows)
    
    # Calculate SNR
    signal = height  # Peak height above baseline
    snr = signal / max(noise, signal * 0.01)  # Prevent division by zero
    snr = np.clip(snr, 0, 20)  # Reasonable limits
    
    # Update peak wavelength
    wavelengths[peak_idx] = peak_pos
    
    # Calculate quality score
    quality_metrics = [
        min(1.0, snr / 10),                    # SNR contribution
        min(1.0, prominence / height),          # Prominence contribution
        min(1.0, 1 / abs(asymmetry - 1)),      # Symmetry contribution
        min(1.0, 1 / abs(width - 10))          # Width contribution
    ]
    quality = np.mean(quality_metrics)
    
    return PeakResult(
        wavelength=float(wavelengths[peak_idx]),
        height=float(height),
        width=float(width),
        area=float(area),
        asymmetry=float(asymmetry),
        snr=float(snr),
        quality=float(quality)
    )


def find_spectral_shifts(wavelengths: np.ndarray,
                        reference: np.ndarray,
                        sample: np.ndarray,
                        region: str = 'auto',
                        peak_wavelength: Optional[float] = None,
                        expected_shift: Optional[float] = None,
                        debug: bool = False) -> List[ShiftResult]:
    """Find spectral shifts using multiple methods."""
    # Initialize results
    results = []
    
    # 1. Preprocess spectra
    ref_smooth = adaptive_smooth(wavelengths, reference, region)
    sam_smooth = adaptive_smooth(wavelengths, sample, region)
    
    ref_base = robust_baseline(wavelengths, ref_smooth, region)
    sam_base = robust_baseline(wavelengths, sam_smooth, region)
    
    ref_proc = ref_smooth - ref_base
    sam_proc = sam_smooth - sam_base
    
    # Normalize
    ref_norm = ref_proc / np.max(np.abs(ref_proc))
    sam_norm = sam_proc / np.max(np.abs(sam_proc))
    
    if debug:
        print(f"\nProcessing {region} region:")
        print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    
    # Find peaks around target wavelength
    if peak_wavelength is not None:
        # Use target wavelength
        peak_idx = np.argmin(np.abs(wavelengths - peak_wavelength))
        window = 50  # points
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(wavelengths), peak_idx + window)
        
        # Analyze peak
        ref_peak = analyze_peak(wavelengths, ref_norm, peak_idx)
        sam_peak = analyze_peak(wavelengths, sam_norm, peak_idx)
        
        # Calculate shift
        shift = sam_peak.wavelength - ref_peak.wavelength
        
        # Calculate confidence
        # 1. Peak quality
        quality_conf = min(ref_peak.quality, sam_peak.quality)
        
        # 2. SNR
        min_snr = min(ref_peak.snr, sam_peak.snr)
        snr_conf = 1.0 / (1.0 + np.exp(-0.5 * (min_snr - 3)))  # Sigmoid centered at SNR=3
        
        # 3. Shift reasonableness
        if expected_shift is not None:
            max_shift = abs(expected_shift) * 2  # Allow up to 2x expected shift
        else:
            max_shift = 5.0  # nm
        shift_conf = np.exp(-0.5 * (abs(shift) / max_shift)**2)
        
        # Combine scores
        confidence = (
            0.4 * quality_conf +
            0.4 * snr_conf +
            0.2 * shift_conf
        )
        
        # Create result
        results.append(ShiftResult(
            wavelength=peak_wavelength,
            shift=shift,
            confidence=confidence,
            peak_ref=ref_peak,
            peak_sam=sam_peak
        ))
    else:
        # Find all peaks
        peak_params = {
            'EtOH': {'prominence': 0.05, 'width': (3, 30), 'distance': 10},
            'IPA': {'prominence': 0.1, 'width': (5, 20), 'distance': 15},
            'MeOH': {'prominence': 0.2, 'width': (8, 15), 'distance': 20}
        }.get(region, {'prominence': 0.1, 'width': (5, 20), 'distance': 15})
    
        # Find peaks in reference and sample
        ref_peaks, _ = signal.find_peaks(ref_norm, **peak_params)
        sam_peaks, _ = signal.find_peaks(sam_norm, **peak_params)
        
        # Match peaks between reference and sample
        for ref_idx in ref_peaks:
            # Find closest sample peak
            if len(sam_peaks) > 0:
                sam_idx = sam_peaks[np.argmin(np.abs(
                    wavelengths[sam_peaks] - wavelengths[ref_idx]
                ))]
            else:
                continue
            
            # Skip if peaks are too far apart
            if abs(wavelengths[sam_idx] - wavelengths[ref_idx]) > 10:
                continue
            
            # Analyze peaks
            ref_peak = analyze_peak(wavelengths, ref_norm, ref_idx)
            sam_peak = analyze_peak(wavelengths, sam_norm, sam_idx)
            
            # Calculate shift
            shift = sam_peak.wavelength - ref_peak.wavelength
            
            # Calculate confidence
            # 1. Peak quality
            quality_conf = min(ref_peak.quality, sam_peak.quality)
            
            # 2. SNR
            min_snr = min(ref_peak.snr, sam_peak.snr)
            snr_conf = 1.0 / (1.0 + np.exp(-0.5 * (min_snr - 3)))
            
            # 3. Shift reasonableness
            max_shift = 5.0  # nm
            shift_conf = np.exp(-0.5 * (abs(shift) / max_shift)**2)
            
            # Combine scores
            confidence = (
                0.4 * quality_conf +
                0.4 * snr_conf +
                0.2 * shift_conf
            )
            
            # Create result
            results.append(ShiftResult(
                wavelength=wavelengths[ref_idx],
                shift=shift,
                confidence=confidence,
                peak_ref=ref_peak,
                peak_sam=sam_peak
            ))
            
    
    # Sort by confidence
    results.sort(key=lambda x: x.confidence, reverse=True)
    
    # Return results
    return results

def analyze_concentration_response(shifts: List[Dict],
                                region: str = 'auto') -> Dict:
    """Analyze concentration response with advanced statistics."""
    # Extract data
    concentrations = np.array([s['concentration'] for s in shifts])
    peak_shifts = np.array([s['shift'] for s in shifts])
    confidences = np.array([s['confidence'] for s in shifts])
    
    # Remove outliers using modified z-score
    def remove_outliers(x: np.ndarray, threshold: float = 3.5) -> np.ndarray:
        median = np.median(x)
        mad = np.median(np.abs(x - median)) * 1.4826
        modified_z = np.abs(x - median) / mad if mad > 0 else np.zeros_like(x)
        return modified_z < threshold
    
    # Apply outlier removal
    valid = remove_outliers(peak_shifts)
    x = concentrations[valid]
    y = peak_shifts[valid]
    w = confidences[valid]
    
    if len(x) < 3:
        return {
            'sensitivity': np.nan,
            'r_squared': np.nan,
            'rmse': np.nan,
            'lod': np.nan,
            'loq': np.nan,
            'points_used': len(x),
            'outliers_removed': len(concentrations) - len(x)
        }
    
    # Try different fit models
    def linear(x, a, b):
        return a * x + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def langmuir(x, vmax, k):
        return vmax * x / (k + x)
    
    models = [
        ('linear', linear, [1, 0]),
        ('quadratic', quadratic, [0.1, 1, 0]),
        ('langmuir', langmuir, [max(y), np.median(x)])
    ]
    
    best_fit = None
    best_aic = np.inf
    
    for name, func, p0 in models:
        try:
            # Weighted fit
            popt, pcov = curve_fit(func, x, y, p0=p0, sigma=1/w, absolute_sigma=True)
            
            # Calculate AIC
            y_fit = func(x, *popt)
            n = len(x)
            k = len(p0)
            rss = np.sum(w * (y - y_fit)**2)
            aic = n * np.log(rss/n) + 2*k
            
            if aic < best_aic:
                best_aic = aic
                best_fit = {
                    'name': name,
                    'function': func,
                    'parameters': popt,
                    'covariance': pcov,
                    'y_fit': y_fit
                }
        except:
            continue
    
    if best_fit is None:
        return {
            'sensitivity': np.nan,
            'r_squared': np.nan,
            'rmse': np.nan,
            'lod': np.nan,
            'loq': np.nan,
            'points_used': len(x),
            'outliers_removed': len(concentrations) - len(x)
        }
    
    # Calculate metrics
    residuals = y - best_fit['y_fit']
    ss_res = np.sum(w * residuals**2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w))**2)
    r_squared = 1 - ss_res/ss_tot
    
    # Calculate weighted RMSE
    rmse = np.sqrt(np.average(residuals**2, weights=w))
    
    # Estimate sensitivity at origin for non-linear models
    if best_fit['name'] == 'linear':
        sensitivity = best_fit['parameters'][0]
    else:
        x_test = np.linspace(0, min(x), 100)
        y_test = best_fit['function'](x_test, *best_fit['parameters'])
        sensitivity = (y_test[1] - y_test[0]) / (x_test[1] - x_test[0])
    
    # Calculate LOD and LOQ
    noise_level = np.std(residuals)
    lod = 3.3 * noise_level / abs(sensitivity)
    loq = 10.0 * noise_level / abs(sensitivity)
    
    return {
        'model': best_fit['name'],
        'sensitivity': float(sensitivity),
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'lod': float(lod),
        'loq': float(loq),
        'points_used': int(len(x)),
        'outliers_removed': int(len(concentrations) - len(x)),
        'parameters': [float(p) for p in best_fit['parameters']]
    }
