"""Advanced spectral analysis for gas sensing."""

import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ResonancePeak:
    """Detailed resonance peak characteristics."""
    wavelength: float          # Peak center
    intensity: float          # Peak height/depth
    fwhm: float              # Full width at half maximum
    area: float              # Integrated peak area
    sensitivity: float       # Wavelength shift per ppm
    r_squared: float         # Linearity of shift
    snr: float              # Signal-to-noise ratio
    drift: float            # Temporal drift rate
    cross_sensitivity: Dict[str, float]  # Response to other gases

def find_resonance_peaks(wavelength: np.ndarray, 
                        intensity: np.ndarray,
                        window_pts: int = 50,
                        noise_window: int = 20,
                        prominence_factor: float = 0.1,
                        width_bounds: Tuple[int, int] = (5, 100)) -> List[Dict]:
    """Find resonance peaks with advanced characterization.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        window_pts: Points for local baseline
        noise_window: Points for noise estimation
        prominence_factor: Minimum prominence relative to range
        width_bounds: (min_width, max_width) in points
    
    Returns:
        List of peak characteristics
    """
    # 1. Preprocessing
    # Smooth with adaptive window
    window = min(len(intensity) // 20, 51)
    if window % 2 == 0:
        window += 1
    y_smooth = savgol_filter(intensity, window, 3)
    
    # Local baseline correction
    y_base = np.zeros_like(intensity)
    for i in range(len(intensity)):
        start = max(0, i - window_pts//2)
        end = min(len(intensity), i + window_pts//2)
        y_base[i] = np.percentile(intensity[start:end], 10)
    y_corr = y_smooth - y_base
    
    # 2. Peak Detection
    # Find both positive peaks and negative dips
    all_peaks = []
    for invert in [False, True]:
        y = -y_corr if invert else y_corr
        
        # Find peaks with prominence threshold
        peaks, properties = find_peaks(
            y,
            prominence=(prominence_factor * np.ptp(y), None),
            width=width_bounds,
            rel_height=0.5
        )
        
        if len(peaks) == 0:
            continue
            
        # Analyze each peak
        widths_result = peak_widths(y, peaks, rel_height=0.5)
        for i, peak_idx in enumerate(peaks):
            # Basic properties
            peak_wl = wavelength[peak_idx]
            peak_height = y[peak_idx]
            
            # Width analysis
            width_pts = widths_result[0][i]
            left_idx = int(peak_idx - width_pts/2)
            right_idx = int(peak_idx + width_pts/2)
            left_idx = max(0, min(len(wavelength)-1, left_idx))
            right_idx = max(0, min(len(wavelength)-1, right_idx))
            
            fwhm = abs(wavelength[right_idx] - wavelength[left_idx])
            
            # Area calculation
            peak_region = slice(left_idx, right_idx+1)
            area = np.trapz(y[peak_region], wavelength[peak_region])
            
            # Local SNR
            noise_start = max(0, peak_idx - noise_window)
            noise_end = min(len(y), peak_idx + noise_window + 1)
            noise_level = np.std(y[noise_start:noise_end])
            snr = abs(peak_height) / noise_level if noise_level > 0 else 0
            
            # Peak shape analysis
            try:
                # Fit Gaussian to estimate peak shape
                x_fit = wavelength[peak_region]
                y_fit = y[peak_region]
                p0 = [peak_height, peak_wl, fwhm/2.355]  # Initial guess
                
                def gaussian(x, amplitude, center, sigma):
                    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
                
                popt, _ = curve_fit(gaussian, x_fit, y_fit, p0=p0)
                fit_quality = np.corrcoef(y_fit, gaussian(x_fit, *popt))[0,1]
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                fit_quality = 0.0
            
            all_peaks.append({
                'wavelength': float(peak_wl),
                'intensity': float(peak_height * (-1 if invert else 1)),
                'fwhm': float(fwhm),
                'area': float(area * (-1 if invert else 1)),
                'snr': float(snr),
                'prominence': float(properties['prominences'][i]),
                'is_dip': invert,
                'shape_quality': float(fit_quality),
                'left_idx': left_idx,
                'right_idx': right_idx
            })
    
    # Sort by SNR
    all_peaks.sort(key=lambda x: x['snr'], reverse=True)
    return all_peaks


def analyze_concentration_response(peaks_by_conc: Dict[float, List[Dict]],
                                min_r2: float = 0.9,
                                min_points: int = 3) -> List[Dict]:
    """Analyze peak shifts vs concentration to find sensitive regions.
    
    Args:
        peaks_by_conc: Dict mapping concentration to peak list
        min_r2: Minimum R² for valid calibration
        min_points: Minimum points for calibration
        
    Returns:
        List of peaks with calibration metrics
    """
    if not peaks_by_conc:
        return []
    
    # Track peaks across concentrations
    tracked_peaks = {}  # wavelength -> list of (conc, peak) pairs
    
    # Start with lowest concentration as reference
    min_conc = min(peaks_by_conc.keys())
    ref_peaks = peaks_by_conc[min_conc]
    
    for ref_peak in ref_peaks:
        ref_wl = ref_peak['wavelength']
        tracked_peaks[ref_wl] = [(min_conc, ref_peak)]
    
    # Track each peak through increasing concentrations
    for conc in sorted(peaks_by_conc.keys())[1:]:
        current_peaks = peaks_by_conc[conc]
        matched = set()
        
        # For each reference peak, find closest match
        for ref_wl, peak_history in tracked_peaks.items():
            last_peak = peak_history[-1][1]
            best_match = None
            best_dist = float('inf')
            
            for peak in current_peaks:
                if peak not in matched:
                    dist = abs(peak['wavelength'] - last_peak['wavelength'])
                    if dist < best_dist and dist < 5.0:  # Max 5nm shift
                        best_dist = dist
                        best_match = peak
            
            if best_match is not None:
                tracked_peaks[ref_wl].append((conc, best_match))
                matched.add(best_match)
    
    # Analyze each tracked peak
    calibrated_peaks = []
    for ref_wl, peak_history in tracked_peaks.items():
        if len(peak_history) < min_points:
            continue
        
        # Extract concentration-wavelength pairs
        concs, peaks = zip(*peak_history)
        concs = np.array(concs)
        wavelengths = np.array([p['wavelength'] for p in peaks])
        intensities = np.array([p['intensity'] for p in peaks])
        
        # Fit linear calibration
        slope, intercept, r_value, p_value, std_err = stats.linregress(concs, wavelengths)
        r2 = r_value**2
        
        if r2 >= min_r2:
            # Compute residuals and uncertainties
            y_pred = slope * concs + intercept
            residuals = wavelengths - y_pred
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Confidence intervals
            conf_level = 0.95
            degrees_of_freedom = len(concs) - 2
            t_value = stats.t.ppf((1 + conf_level) / 2, degrees_of_freedom)
            
            # Standard errors
            x_mean = np.mean(concs)
            x_std = np.std(concs)
            
            slope_se = std_err
            intercept_se = rmse * np.sqrt(1/len(concs) + x_mean**2/(x_std**2 * (len(concs)-1)))
            
            # Confidence intervals
            slope_ci = (slope - t_value*slope_se, slope + t_value*slope_se)
            intercept_ci = (intercept - t_value*intercept_se, intercept + t_value*intercept_se)
            
            # LOD and LOQ
            blank_std = rmse  # Using RMSE as estimate of blank standard deviation
            lod = 3.3 * blank_std / abs(slope)
            loq = 10.0 * blank_std / abs(slope)
            
            calibrated_peaks.append({
                'reference_wavelength': float(ref_wl),
                'sensitivity': float(slope),  # nm/ppm
                'intercept': float(intercept),
                'r_squared': float(r2),
                'p_value': float(p_value),
                'rmse': float(rmse),
                'slope_ci': (float(slope_ci[0]), float(slope_ci[1])),
                'intercept_ci': (float(intercept_ci[0]), float(intercept_ci[1])),
                'lod': float(lod),
                'loq': float(loq),
                'n_points': len(concs),
                'concentration_range': (float(min(concs)), float(max(concs))),
                'wavelength_range': (float(min(wavelengths)), float(max(wavelengths))),
                'mean_snr': float(np.mean([p['snr'] for p in peaks])),
                'intensity_range': (float(min(intensities)), float(max(intensities)))
            })
    
    # Sort by sensitivity and R²
    calibrated_peaks.sort(key=lambda x: (x['r_squared'], abs(x['sensitivity'])), reverse=True)
    return calibrated_peaks


def analyze_cross_sensitivity(calibrated_peaks: List[Dict],
                           peaks_by_gas: Dict[str, Dict[float, List[Dict]]]) -> Dict[str, List[Dict]]:
    """Analyze how each peak responds to different gases.
    
    Args:
        calibrated_peaks: List of calibrated peaks from analyze_concentration_response
        peaks_by_gas: Dict mapping gas name to its peaks_by_conc dict
        
    Returns:
        Dict mapping gas to list of peaks with cross-sensitivity metrics
    """
    results = {}
    
    for target_gas, target_peaks_by_conc in peaks_by_gas.items():
        gas_results = []
        
        # Analyze each calibrated peak
        for peak in calibrated_peaks:
            ref_wl = peak['reference_wavelength']
            cross_sensitivity = {}
            
            # Check response to each interfering gas
            for other_gas, other_peaks_by_conc in peaks_by_gas.items():
                if other_gas == target_gas:
                    continue
                
                # Find matching peak in other gas
                other_concs = []
                other_shifts = []
                
                for conc, peaks in other_peaks_by_conc.items():
                    # Find closest peak
                    closest = min(peaks, key=lambda p: abs(p['wavelength'] - ref_wl),
                                default=None)
                    if closest is not None:
                        other_concs.append(conc)
                        other_shifts.append(closest['wavelength'] - ref_wl)
                
                if len(other_concs) >= 3:
                    # Compute interfering sensitivity
                    slope, _, r_value, _, _ = stats.linregress(other_concs, other_shifts)
                    cross_sensitivity[other_gas] = {
                        'sensitivity': float(slope),
                        'r_squared': float(r_value**2)
                    }
            
            # Add cross-sensitivity metrics
            result = peak.copy()
            result['cross_sensitivity'] = cross_sensitivity
            
            # Compute selectivity ratios
            target_sensitivity = abs(peak['sensitivity'])
            selectivity_ratios = {}
            for gas, metrics in cross_sensitivity.items():
                interfering_sensitivity = abs(metrics['sensitivity'])
                if interfering_sensitivity > 0:
                    selectivity_ratios[gas] = target_sensitivity / interfering_sensitivity
                else:
                    selectivity_ratios[gas] = float('inf')
            
            result['selectivity_ratios'] = selectivity_ratios
            gas_results.append(result)
        
        results[target_gas] = gas_results
    
    return results
