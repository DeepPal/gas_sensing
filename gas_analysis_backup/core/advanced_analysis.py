"""Advanced spectral analysis for NCF gas sensor."""

import numpy as np
from scipy import signal, stats
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PeakFeatures:
    """Extracted peak features."""
    wavelength: float      # Peak center wavelength (nm)
    height: float         # Peak height
    width: float          # FWHM (nm)
    area: float          # Integrated peak area
    asymmetry: float     # Peak asymmetry
    snr: float          # Signal-to-noise ratio
    quality: float       # Overall quality score

@dataclass
class ShiftAnalysis:
    """Results of shift analysis."""
    wavelength: float     # Reference wavelength (nm)
    shift: float         # Detected shift (nm)
    method: str          # Analysis method used
    confidence: float    # Confidence in shift detection (0-1)
    snr: float          # Signal-to-noise ratio
    quality: float      # Overall quality score

def adaptive_smooth(wavelengths: np.ndarray,
                   intensity: np.ndarray,
                   noise_threshold: float = 0.01) -> np.ndarray:
    """Adaptive smoothing based on local noise level."""
    # Estimate local noise
    diff = np.diff(intensity)
    noise = np.std(diff) / np.sqrt(2)
    
    if noise > noise_threshold:
        # Strong smoothing for noisy data
        window = 41
        poly_order = 4
        # Multiple passes
        smoothed = intensity
        for _ in range(2):
            smoothed = savgol_filter(smoothed, window, poly_order)
            smoothed = gaussian_filter1d(smoothed, sigma=2)
    else:
        # Light smoothing for clean data
        window = 21
        poly_order = 3
        smoothed = savgol_filter(intensity, window, poly_order)
    
    return smoothed

def robust_baseline(wavelengths: np.ndarray,
                   intensity: np.ndarray,
                   poly_order: int = 3,
                   chunk_size: int = 50) -> np.ndarray:
    """Robust baseline estimation using chunked polynomial fitting."""
    # Split into chunks
    n_chunks = len(wavelengths) // chunk_size
    chunks = np.array_split(np.arange(len(wavelengths)), n_chunks)
    
    # Find minimum points in each chunk
    min_points = []
    for chunk in chunks:
        if len(chunk) > 0:
            min_idx = chunk[np.argmin(intensity[chunk])]
            min_points.append((wavelengths[min_idx], intensity[min_idx]))
    
    min_points = np.array(min_points)
    
    # Fit polynomial to minima
    coef = np.polyfit(min_points[:, 0], min_points[:, 1], poly_order)
    baseline = np.polyval(coef, wavelengths)
    
    return baseline

def extract_peak_features(wavelengths: np.ndarray,
                         intensity: np.ndarray,
                         peak_idx: int,
                         window: int = 20,
                         debug: bool = False) -> PeakFeatures:
    """Extract detailed features of a peak."""
    # Define analysis window
    start_idx = max(0, peak_idx - window)
    end_idx = min(len(wavelengths), peak_idx + window)
    
    peak_region = intensity[start_idx:end_idx]
    peak_wl = wavelengths[start_idx:end_idx]
    
    # Basic features
    height = intensity[peak_idx]
    
    # Calculate FWHM
    half_height = (height + np.min(peak_region)) / 2
    above_half = peak_region >= half_height
    if np.sum(above_half) >= 2:
        left_idx = np.where(above_half)[0][0]
        right_idx = np.where(above_half)[0][-1]
        width = peak_wl[right_idx] - peak_wl[left_idx]
    else:
        width = np.nan
    
    # Calculate area
    area = np.trapz(peak_region, peak_wl)
    
    # Calculate asymmetry
    left_area = np.trapz(peak_region[:peak_idx-start_idx],
                        peak_wl[:peak_idx-start_idx])
    right_area = np.trapz(peak_region[peak_idx-start_idx:],
                         peak_wl[peak_idx-start_idx:])
    asymmetry = left_area / right_area if right_area != 0 else np.inf
    
    # Calculate SNR and quality score
    # Use robust noise estimation
    signal = height - np.median(peak_region)
    noise_region = peak_region - np.median(peak_region)
    noise = np.median(np.abs(noise_region)) * 1.4826  # MAD to sigma
    snr = signal / noise if noise > 0 else 0
    
    # Calculate peak prominence
    prominence = height - max(peak_region[0], peak_region[-1])
    
    # Calculate peak sharpness
    if not np.isnan(width) and width > 0:
        sharpness = prominence / width
    else:
        sharpness = 0
    
    # Calculate quality score
    quality_metrics = [
        min(1.0, snr / 5),           # SNR contribution
        min(1.0, prominence / signal), # Prominence contribution
        min(1.0, sharpness),          # Sharpness contribution
        min(1.0, 1 / abs(asymmetry - 1))  # Symmetry contribution
    ]
    quality = np.mean(quality_metrics)
    
    if debug:
        print(f"\nPeak analysis at {wavelengths[peak_idx]:.1f} nm:")
        print(f"- Height: {height:.3f}")
        print(f"- Width: {width:.1f} nm")
        print(f"- Area: {area:.3f}")
        print(f"- Asymmetry: {asymmetry:.2f}")
        print(f"- SNR: {snr:.1f}")
        print(f"- Prominence: {prominence:.3f}")
        print(f"- Sharpness: {sharpness:.3f}")
        print(f"- Quality: {quality:.3f}")
    
    return PeakFeatures(
        wavelength=float(wavelengths[peak_idx]),
        height=float(height),
        width=float(width),
        area=float(area),
        asymmetry=float(asymmetry),
        snr=float(snr),
        quality=float(quality)
    )

def analyze_peak_shift(wavelengths: np.ndarray,
                      reference: np.ndarray,
                      sample: np.ndarray,
                      peak_wavelength: float,
                      window: int = 50,
                      debug: bool = False) -> List[ShiftAnalysis]:
    """Analyze peak shift using multiple methods.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        sample: Sample spectrum
        peak_wavelength: Target peak wavelength
        window: Analysis window size (points)
        debug: Print debug information
    
    Returns:
        List of shift analyses from different methods
    """
    # Initialize results list
    results = []
    
    # Find peak region
    peak_idx = np.argmin(np.abs(wavelengths - peak_wavelength))
    start_idx = max(0, peak_idx - window)
    end_idx = min(len(wavelengths), peak_idx + window)
    
    # Extract region of interest
    wl = wavelengths[start_idx:end_idx]
    ref = reference[start_idx:end_idx]
    sam = sample[start_idx:end_idx]
    
    if len(wl) < 10:
        if debug:
            print("Warning: Window too small for analysis")
        return results
    
    if debug:
        print(f"\nAnalyzing shift at {peak_wavelength:.1f} nm")
        print(f"Window: {wl[0]:.1f} - {wl[-1]:.1f} nm")
    
    # 1. Peak tracking method
    # Find local maxima near expected peak
    ref_peaks, _ = find_peaks(ref)
    sam_peaks, _ = find_peaks(sam)
    
    # Get closest peaks to target wavelength
    if len(ref_peaks) > 0 and len(sam_peaks) > 0:
        ref_idx = ref_peaks[np.argmin(np.abs(wl[ref_peaks] - peak_wavelength))]
        sam_idx = sam_peaks[np.argmin(np.abs(wl[sam_peaks] - peak_wavelength))]
        
        # Extract features
        ref_peak = extract_peak_features(wl, ref, ref_idx, window=window//2, debug=debug)
        sam_peak = extract_peak_features(wl, sam, sam_idx, window=window//2, debug=debug)
    else:
        # If no peaks found, use target wavelength
        center_idx = len(wl) // 2
        ref_peak = extract_peak_features(wl, ref, center_idx, window=window//2, debug=debug)
        sam_peak = extract_peak_features(wl, sam, center_idx, window=window//2, debug=debug)
    
    peak_shift = sam_peak.wavelength - ref_peak.wavelength
    peak_conf = min(ref_peak.quality, sam_peak.quality)
    
    results.append(ShiftAnalysis(
        wavelength=peak_wavelength,
        shift=peak_shift,
        method='peak_tracking',
        confidence=peak_conf,
        snr=min(ref_peak.snr, sam_peak.snr),
        quality=peak_conf
    ))
    
    # 2. Cross-correlation method
    # Normalize signals
    ref_norm = (ref - np.mean(ref)) / np.std(ref)
    sam_norm = (sam - np.mean(sam)) / np.std(sam)
    
    # Compute correlation
    corr = signal.correlate(sam_norm, ref_norm, mode='full')
    lags = signal.correlation_lags(len(sam_norm), len(ref_norm))
    max_idx = np.argmax(corr)
    
    # Fit parabola for sub-pixel precision
    if max_idx > 0 and max_idx < len(corr)-1:
        fit_x = np.array([-1, 0, 1])
        fit_y = corr[max_idx-1:max_idx+2]
        coef = np.polyfit(fit_x, fit_y, 2)
        max_x = -coef[1] / (2 * coef[0])
        max_lag = lags[max_idx] + max_x
    else:
        max_lag = lags[max_idx]
    
    # Convert lag to wavelength
    wl_step = np.mean(np.diff(wl))
    corr_shift = max_lag * wl_step
    
    # Calculate confidence based on correlation quality
    # Normalize correlation by signal energy
    ref_energy = np.sqrt(np.sum(ref_norm**2))
    sam_energy = np.sqrt(np.sum(sam_norm**2))
    corr_norm = corr[max_idx] / (ref_energy * sam_energy)
    
    # Scale to 0-1 range
    corr_conf = (corr_norm + 1) / 2
    
    results.append(ShiftAnalysis(
        wavelength=peak_wavelength,
        shift=corr_shift,
        method='cross_correlation',
        confidence=float(corr_conf),
        snr=float(corr[max_idx] / np.std(corr)),
        quality=float(corr_conf)
    ))
    
    # 3. Centroid method
    def weighted_centroid(x, y):
        # Remove baseline
        baseline = np.min(y)
        y_base = y - baseline
        # Use positive values only
        mask = y_base > 0
        if np.sum(mask) > 0:
            return np.sum(x[mask] * y_base[mask]) / np.sum(y_base[mask])
        return np.mean(x)
    
    ref_cent = weighted_centroid(wl, ref)
    sam_cent = weighted_centroid(wl, sam)
    cent_shift = sam_cent - ref_cent
    
    # Calculate confidence based on multiple factors
    # 1. Peak quality
    quality_conf = min(ref_peak.quality, sam_peak.quality)
    
    # 2. Shift reasonableness
    max_expected_shift = (wl[-1] - wl[0]) * 0.1  # Max 10% of window
    shift_conf = 1.0 - min(1.0, abs(cent_shift) / max_expected_shift)
    
    # 3. SNR
    snr_conf = min(1.0, min(ref_peak.snr, sam_peak.snr) / 10)
    
    # Combine confidences
    cent_conf = np.mean([quality_conf, shift_conf, snr_conf])
    
    results.append(ShiftAnalysis(
        wavelength=peak_wavelength,
        shift=cent_shift,
        method='centroid',
        confidence=cent_conf,
        snr=min(ref_peak.snr, sam_peak.snr),
        quality=cent_conf
    ))
    
    # 4. Peak fitting method
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-((x - cen) / wid)**2)
    
    try:
        # Fit reference peak
        ref_popt, _ = curve_fit(gaussian, wl, ref,
                               p0=[np.max(ref), peak_wavelength, 5])
        # Fit sample peak
        sam_popt, _ = curve_fit(gaussian, wl, sam,
                               p0=[np.max(sam), peak_wavelength, 5])
        
        # Calculate shift
        fit_shift = sam_popt[1] - ref_popt[1]
        
        # Calculate confidence based on fit quality
        ref_resid = np.sum((ref - gaussian(wl, *ref_popt))**2)
        sam_resid = np.sum((sam - gaussian(wl, *sam_popt))**2)
        fit_conf = 1.0 / (1.0 + ref_resid + sam_resid)
        
        results.append(ShiftAnalysis(
            wavelength=peak_wavelength,
            shift=fit_shift,
            method='peak_fitting',
            confidence=fit_conf,
            snr=min(ref_peak.snr, sam_peak.snr),
            quality=fit_conf
        ))
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        pass
    
    return results

def get_consensus_shift(results: List[ShiftAnalysis]) -> Tuple[float, float]:
    """Get consensus shift from multiple methods."""
    if not results:
        return 0.0, 0.0
    
    # Weight each method by its confidence
    weights = np.array([r.confidence for r in results])
    shifts = np.array([r.shift for r in results])
    
    if len(shifts) == 1:
        return shifts[0], 0.0
    
    # Remove outliers using weighted median absolute deviation
    weighted_median = np.average(shifts, weights=weights)
    deviations = np.abs(shifts - weighted_median)
    mad = np.median(deviations)
    
    # Keep shifts within 3 MAD
    mask = deviations <= 3 * mad
    if np.sum(mask) > 0:
        weights = weights[mask]
        shifts = shifts[mask]
    
    if len(shifts) == 0:
        return 0.0, 0.0
    
    # Calculate weighted average
    consensus_shift = np.average(shifts, weights=weights)
    
    # Calculate weighted standard error
    if len(shifts) > 1:
        # Use weighted variance formula
        sum_weights = np.sum(weights)
        weighted_var = np.sum(weights * (shifts - consensus_shift)**2) / (sum_weights - np.sum(weights**2)/sum_weights)
        uncertainty = np.sqrt(weighted_var / len(shifts))
    else:
        uncertainty = 0.0
    
    return consensus_shift, uncertainty

def analyze_concentration_response(concentrations: List[float],
                                shifts: List[float],
                                uncertainties: List[float] = None) -> Dict:
    """Analyze concentration response curve using weighted least squares."""
    if uncertainties is None:
        uncertainties = np.ones_like(shifts)
    
    # Convert to arrays
    x = np.array(concentrations)
    y = np.array(shifts)
    dy = np.array(uncertainties)
    
    # Replace zero uncertainties
    dy[dy == 0] = np.min(dy[dy > 0]) if np.any(dy > 0) else 1.0
    
    # Weighted least squares
    weights = 1 / (dy**2)
    
    # Calculate weighted means
    wx = np.average(x, weights=weights)
    wy = np.average(y, weights=weights)
    
    # Calculate weighted covariance and variance
    wxy = np.average((x - wx) * (y - wy), weights=weights)
    wxx = np.average((x - wx)**2, weights=weights)
    
    # Calculate regression parameters
    slope = wxy / wxx
    intercept = wy - slope * wx
    
    # Calculate R² and other statistics
    y_fit = slope * x + intercept
    residuals = y - y_fit
    
    # Weighted R²
    ss_tot = np.sum(weights * (y - wy)**2)
    ss_res = np.sum(weights * residuals**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    # Standard errors
    n = len(x)
    if n > 2:
        # Weighted standard error of slope
        slope_stderr = np.sqrt(1 / (wxx * np.sum(weights)))
        
        # Weighted RMSE
        weighted_mse = ss_res / (n - 2)
        rmse = np.sqrt(weighted_mse)
        
        # Calculate p-value
        t_stat = slope / slope_stderr
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        slope_stderr = np.nan
        rmse = np.nan
        p_value = np.nan
    
    # Calculate LOD and LOQ
    if not np.isnan(rmse) and slope != 0:
        lod = 3.3 * rmse / abs(slope)
        loq = 10.0 * rmse / abs(slope)
    else:
        lod = np.nan
        loq = np.nan
    
    # Calculate confidence intervals
    if not np.isnan(slope_stderr) and n > 2:
        conf_level = 0.95
        t_value = stats.t.ppf((1 + conf_level) / 2, n - 2)
        slope_ci = (float(slope - t_value * slope_stderr),
                   float(slope + t_value * slope_stderr))
    else:
        slope_ci = (np.nan, np.nan)
    
    return {
        'sensitivity': float(slope),  # nm/ppm
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'p_value': float(p_value),
        'rmse': float(rmse),
        'lod': float(lod),
        'loq': float(loq),
        'slope_stderr': float(slope_stderr),
        'slope_ci': slope_ci
    }
