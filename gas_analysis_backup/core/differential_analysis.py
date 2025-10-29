"""Differential spectral analysis for NCF gas sensor.

This module implements a differential analysis approach:
1. Compute transmittance (T = I_sample/I_ref)
2. Find regions of maximum change
3. Track spectral shifts using cross-correlation
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks, correlate, correlation_lags
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpectralShift:
    """Detected spectral shift."""
    wavelength_range: Tuple[float, float]  # Region where shift was detected
    shift_nm: float                        # Magnitude of shift in nm
    correlation: float                     # Quality of shift detection
    transmittance_change: float           # Change in transmittance
    snr: float                            # Signal-to-noise ratio

def preprocess_spectra(wavelengths: np.ndarray,
                      reference: np.ndarray,
                      sample: np.ndarray,
                      window: int = 31) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess reference and sample spectra.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        sample: Sample spectrum
        window: Smoothing window size
    
    Returns:
        Tuple of (processed reference, processed sample)
    """
    # 1. Ensure positive values (add small offset if needed)
    min_val = min(np.min(reference), np.min(sample))
    if min_val < 0:
        offset = abs(min_val) + 1e-6
        reference = reference + offset
        sample = sample + offset
    
    # 2. Savitzky-Golay smoothing
    window = min(window, len(wavelengths) - 1)
    if window % 2 == 0:
        window += 1
    ref_smooth = signal.savgol_filter(reference, window, 3)
    sam_smooth = signal.savgol_filter(sample, window, 3)
    
    return ref_smooth, sam_smooth

def compute_transmittance(reference: np.ndarray,
                         sample: np.ndarray,
                         min_ref: float = 0.1) -> np.ndarray:
    """Compute transmittance spectrum.
    
    Args:
        reference: Reference spectrum
        sample: Sample spectrum
        min_ref: Minimum reference intensity to avoid division by zero
    
    Returns:
        Transmittance spectrum
    """
    # Avoid division by zero
    valid = reference > min_ref
    T = np.ones_like(reference)
    T[valid] = sample[valid] / reference[valid]
    
    # Clip to reasonable range
    T = np.clip(T, 0, 2)
    return T

def find_spectral_shifts(wavelengths: np.ndarray,
                        reference: np.ndarray,
                        sample: np.ndarray,
                        roi: Optional[Tuple[float, float]] = None,
                        window_size: int = 50,
                        debug: bool = False) -> List[SpectralShift]:
    """Find spectral shifts between reference and sample.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        sample: Sample spectrum
        roi: Optional region of interest (start, end) in nm
        window_size: Size of analysis window in points
        debug: Print debug information
    
    Returns:
        List of detected spectral shifts
    """
    # 1. Preprocess spectra
    ref_proc, sam_proc = preprocess_spectra(wavelengths, reference, sample)
    
    # 2. Compute transmittance
    T = compute_transmittance(ref_proc, sam_proc)
    
    # 3. Apply ROI if specified
    if roi is not None:
        mask = (wavelengths >= roi[0]) & (wavelengths <= roi[1])
        wavelengths = wavelengths[mask]
        T = T[mask]
        ref_proc = ref_proc[mask]
        sam_proc = sam_proc[mask]
    
    if debug:
        print(f"\nAnalyzing spectral shifts:")
        print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
        print(f"Points in range: {len(wavelengths)}")
    
    # 4. Find regions of significant change
    dT = np.gradient(T)
    change_points = signal.find_peaks(np.abs(dT), prominence=0.001)[0]
    
    shifts = []
    for cp in change_points:
        # Define analysis window
        start_idx = max(0, cp - window_size//2)
        end_idx = min(len(wavelengths), cp + window_size//2)
        
        if end_idx - start_idx < 10:  # Need enough points
            continue
        
        # Extract window
        wl_window = wavelengths[start_idx:end_idx]
        ref_window = ref_proc[start_idx:end_idx]
        sam_window = sam_proc[start_idx:end_idx]
        
        # Compute cross-correlation
        corr = signal.correlate(sam_window, ref_window, mode='full')
        lags = signal.correlation_lags(len(sam_window), len(ref_window))
        peak_idx = np.argmax(corr)
        lag = lags[peak_idx]
        
        # Convert lag to wavelength shift
        wl_step = np.mean(np.diff(wl_window))
        shift_nm = lag * wl_step
        
        # Compute SNR
        noise = np.std(sam_window - ref_window)
        signal = np.abs(np.mean(sam_window) - np.mean(ref_window))
        snr = signal / noise if noise > 0 else 0
        
        # Record shift if significant
        if abs(shift_nm) > 0.1 and snr > 1.0:
            shifts.append(SpectralShift(
                wavelength_range=(wl_window[0], wl_window[-1]),
                shift_nm=shift_nm,
                correlation=corr[peak_idx],
                transmittance_change=np.mean(np.abs(T[start_idx:end_idx] - 1)),
                snr=snr
            ))
            
            if debug:
                print(f"\nShift detected at {wl_window[0]:.1f}-{wl_window[-1]:.1f} nm:")
                print(f"- Shift: {shift_nm:.2f} nm")
                print(f"- Correlation: {corr[peak_idx]:.3f}")
                print(f"- SNR: {snr:.1f}")
    
    # Sort by correlation quality
    shifts.sort(key=lambda x: x.correlation, reverse=True)
    return shifts

def analyze_concentration_series(wavelengths: np.ndarray,
                               reference: np.ndarray,
                               samples: List[np.ndarray],
                               concentrations: List[float],
                               roi: Optional[Tuple[float, float]] = None,
                               debug: bool = False) -> Dict:
    """Analyze concentration series using differential approach.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        samples: List of sample spectra
        concentrations: Concentration values (ppm)
        roi: Optional region of interest
        debug: Print debug information
    
    Returns:
        Dict with analysis results
    """
    if debug:
        print(f"\nAnalyzing concentration series:")
        print(f"- {len(samples)} samples")
        print(f"- Concentrations: {sorted(set(concentrations))} ppm")
        if roi:
            print(f"- ROI: {roi[0]:.1f} - {roi[1]:.1f} nm")
    
    # Analyze each sample
    all_shifts = []
    for i, (sample, conc) in enumerate(zip(samples, concentrations)):
        shifts = find_spectral_shifts(
            wavelengths, reference, sample,
            roi=roi, debug=debug and i < 3
        )
        
        if shifts:
            # Use strongest shift
            shift = shifts[0]
            all_shifts.append({
                'concentration': conc,
                'shift': shift.shift_nm,
                'correlation': shift.correlation,
                'snr': shift.snr,
                'wavelength_range': shift.wavelength_range
            })
    
    if not all_shifts:
        if debug:
            print("No significant shifts detected")
        return {}
    
    # Convert to arrays for analysis
    conc_array = np.array([s['concentration'] for s in all_shifts])
    shift_array = np.array([s['shift'] for s in all_shifts])
    
    # Linear regression
    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(
        conc_array, shift_array
    )
    
    # Calculate metrics
    residuals = shift_array - (slope * conc_array + intercept)
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Estimate LOD and LOQ
    noise_level = np.std(residuals)
    lod = 3.3 * noise_level / abs(slope)
    loq = 10.0 * noise_level / abs(slope)
    
    results = {
        'sensitivity': float(slope),  # nm/ppm
        'r_squared': float(r_value**2),
        'p_value': float(p_value),
        'rmse': float(rmse),
        'lod': float(lod),
        'loq': float(loq),
        'shifts': all_shifts
    }
    
    if debug:
        print(f"\nResults:")
        print(f"- Sensitivity: {slope:.3f} nm/ppm")
        print(f"- R²: {r_value**2:.3f}")
        print(f"- RMSE: {rmse:.3f} nm")
        print(f"- LOD: {lod:.2f} ppm")
        print(f"- LOQ: {loq:.2f} ppm")
    
    return results
