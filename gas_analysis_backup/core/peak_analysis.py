"""Peak detection and analysis for spectral data."""

import numpy as np
from scipy.signal import find_peaks, peak_widths
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Peak:
    """Represents a spectral peak/dip with quality metrics."""
    wavelength: float          # Peak position
    intensity: float          # Peak height/depth
    width: float             # FWHM
    prominence: float        # Peak prominence
    symmetry: float          # Symmetry score (0-1)
    snr: float              # Local signal-to-noise ratio
    is_dip: bool            # True if valley/dip
    quality_score: float    # Overall quality metric

def find_peaks_advanced(wavelength: np.ndarray, intensity: np.ndarray,
                       smoothed: bool = True,
                       n_peaks: int = None,
                       min_snr: float = 3.0,
                       min_prominence_ratio: float = 0.1) -> List[Peak]:
    """Find spectral peaks/dips with quality metrics.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        smoothed: Whether intensity is already smoothed
        n_peaks: Maximum number of peaks to return (highest quality)
        min_snr: Minimum SNR for peak detection
        min_prominence_ratio: Minimum prominence relative to range
    
    Returns:
        List of Peak objects sorted by quality
    """
    from .preprocessing import smooth_spectrum
    
    # Smooth if needed
    y = intensity if smoothed else smooth_spectrum(intensity)
    y_range = np.ptp(y)
    
    # Find both peaks and dips
    peaks_list = []
    for is_dip, yy in [(False, y), (True, -y)]:
        # Find peaks with prominence threshold
        peaks, properties = find_peaks(
            yy,
            prominence=(min_prominence_ratio * y_range, None),
            width=3,
            rel_height=0.5
        )
        
        if len(peaks) == 0:
            continue
        
        # Compute widths at half prominence
        widths_result = peak_widths(yy, peaks, rel_height=0.5)
        widths = widths_result[0]
        
        # Analyze each peak
        for idx, peak_idx in enumerate(peaks):
            # Peak position and height
            peak_x = wavelength[peak_idx]
            peak_y = y[peak_idx]
            
            # Width (FWHM)
            # Safely compute width in wavelength units
            left_idx = int(max(0, min(len(wavelength)-1, peak_idx - widths[idx]/2)))
            right_idx = int(max(0, min(len(wavelength)-1, peak_idx + widths[idx]/2)))
            width_wl = abs(wavelength[right_idx] - wavelength[left_idx])
            
            # Prominence
            prominence = properties['prominences'][idx]
            
            # Symmetry: compare left and right sides
            half_width = int(widths[idx]/2)
            left_idx = max(0, peak_idx - half_width)
            right_idx = min(len(y), peak_idx + half_width)
            left_side = y[left_idx:peak_idx]
            right_side = y[peak_idx:right_idx]
            min_len = min(len(left_side), len(right_side))
            if min_len > 0:
                left_norm = left_side[-min_len:]
                right_norm = right_side[:min_len][::-1]
                symmetry = 1.0 - np.mean(np.abs(left_norm - right_norm)) / y_range
            else:
                symmetry = 0.0
            
            # Local SNR
            window = slice(max(0, peak_idx - 10), min(len(y), peak_idx + 11))
            signal = abs(peak_y - np.mean(y[window]))
            noise = np.std(y[window])
            snr = signal / noise if noise > 0 else 0.0
            
            # Quality score combines multiple metrics
            quality = (
                (snr / min_snr) *                    # SNR factor
                (prominence / y_range) *             # Prominence factor
                symmetry *                           # Symmetry factor
                np.exp(-width_wl / 100.0)           # Width factor (prefer sharper peaks)
            )
            
            if snr >= min_snr:
                peaks_list.append(Peak(
                    wavelength=float(peak_x),
                    intensity=float(peak_y),
                    width=float(width_wl),
                    prominence=float(prominence),
                    symmetry=float(symmetry),
                    snr=float(snr),
                    is_dip=is_dip,
                    quality_score=float(quality)
                ))
    
    # Sort by quality and optionally limit number
    peaks_list.sort(key=lambda p: p.quality_score, reverse=True)
    if n_peaks is not None:
        peaks_list = peaks_list[:n_peaks]
    
    return peaks_list


def track_peaks(peaks_by_concentration: Dict[float, List[Peak]],
                wavelength_tolerance: float = 5.0) -> Dict[int, List[Tuple[float, Peak]]]:
    """Track peaks across concentrations to identify consistent features.
    
    Args:
        peaks_by_concentration: Dict mapping concentration to peak list
        wavelength_tolerance: Maximum wavelength difference to consider same peak
    
    Returns:
        Dict mapping peak ID to list of (concentration, peak) pairs
    """
    if not peaks_by_concentration:
        return {}
    
    # Start with peaks from lowest concentration
    min_conc = min(peaks_by_concentration.keys())
    tracked_peaks = {
        i: [(min_conc, peak)]
        for i, peak in enumerate(peaks_by_concentration[min_conc])
    }
    
    # Track each peak through increasing concentrations
    for conc in sorted(peaks_by_concentration.keys())[1:]:
        current_peaks = peaks_by_concentration[conc]
        
        # For each tracked peak, find closest match in current concentration
        matched = set()
        for peak_id, peak_history in tracked_peaks.items():
            last_peak = peak_history[-1][1]
            
            # Find closest unmatched peak within tolerance
            best_match = None
            best_dist = wavelength_tolerance
            for peak in current_peaks:
                if peak not in matched:
                    dist = abs(peak.wavelength - last_peak.wavelength)
                    if dist < best_dist:
                        best_dist = dist
                        best_match = peak
            
            if best_match is not None:
                tracked_peaks[peak_id].append((conc, best_match))
                matched.add(best_match)
        
        # Add any unmatched peaks as new tracks
        next_id = max(tracked_peaks.keys()) + 1 if tracked_peaks else 0
        for peak in current_peaks:
            if peak not in matched:
                tracked_peaks[next_id] = [(conc, peak)]
                next_id += 1
    
    return tracked_peaks


def analyze_peak_shifts(tracked_peaks: Dict[int, List[Tuple[float, Peak]]],
                       min_points: int = 3) -> Dict[int, Dict[str, float]]:
    """Analyze wavelength shifts vs concentration for tracked peaks.
    
    Args:
        tracked_peaks: Output from track_peaks()
        min_points: Minimum number of points for valid calibration
    
    Returns:
        Dict mapping peak ID to calibration metrics
    """
    results = {}
    
    for peak_id, peak_history in tracked_peaks.items():
        if len(peak_history) < min_points:
            continue
        
        # Extract concentrations and wavelengths
        concs, peaks = zip(*peak_history)
        concs = np.array(concs)
        wavelengths = np.array([p.wavelength for p in peaks])
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(concs, wavelengths)
        
        # Residuals and RMSE
        predicted = intercept + slope * concs
        residuals = wavelengths - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        
        # LOD and LOQ (3σ and 10σ method)
        if abs(slope) > 1e-10:
            lod = 3.0 * rmse / abs(slope)
            loq = 10.0 * rmse / abs(slope)
        else:
            lod = loq = float('inf')
        
        # Quality metrics for calibration
        quality_scores = [p.quality_score for p in peaks]
        mean_quality = float(np.mean(quality_scores))
        min_quality = float(np.min(quality_scores))
        
        results[peak_id] = {
            'n_points': len(peak_history),
            'wavelength_range': (float(min(wavelengths)), float(max(wavelengths))),
            'concentration_range': (float(min(concs)), float(max(concs))),
            'slope': float(slope),
            'intercept': float(intercept),
            'r2': float(r_value**2),
            'p_value': float(p_value),
            'rmse': float(rmse),
            'lod': float(lod),
            'loq': float(loq),
            'mean_quality': mean_quality,
            'min_quality': min_quality
        }
    
    return results
