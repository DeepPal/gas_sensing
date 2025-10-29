"""Spectral shift analysis for NCF gas sensor."""

import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks, correlate, correlation_lags
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .sensor_config import get_config_for_roi, GasFeatures
from .preprocessing import preprocess_spectrum

@dataclass
class ShiftResult:
    """Result of spectral shift analysis."""
    wavelength: float      # Center wavelength of shift
    shift_nm: float       # Magnitude of shift
    intensity_change: float  # Change in intensity
    snr: float           # Signal-to-noise ratio

def find_shift_regions(wavelengths: np.ndarray,
                      reference: np.ndarray,
                      sample: np.ndarray,
                      roi: Optional[Tuple[float, float]] = None,
                      peak_wavelength: Optional[float] = None,
                      min_snr: float = 2.0,
                      debug: bool = False) -> List[ShiftResult]:
    """Find regions with significant spectral shifts.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        sample: Sample spectrum
        roi: Optional (start, end) wavelength range
        min_snr: Minimum SNR for valid shift
        debug: Print debug info
    
    Returns:
        List of detected shifts
    """
    # Determine configuration for ROI if available
    config: Optional[GasFeatures] = None
    if roi:
        try:
            config = get_config_for_roi(tuple(float(x) for x in roi))
        except ValueError:
            config = None

    # Apply ROI if specified
    if roi:
        mask = (wavelengths >= roi[0]) & (wavelengths <= roi[1])
        wavelengths = wavelengths[mask]
        reference = reference[mask]
        sample = sample[mask]
    
    if debug:
        print(f"\nAnalyzing spectral region: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
        print(f"Points in region: {len(wavelengths)}")
    
    # Get processing parameters from config
    window = config.smooth_window if config else 31
    poly_order = config.smooth_poly if config else 3
    extra_smooth = config.extra_smooth if config else False

    # Preprocess spectra
    baseline_order = config.baseline_poly_order if config else None

    ref_proc = preprocess_spectrum(
        wavelengths,
        reference,
        smooth_window=window,
        poly_order=poly_order,
        extra_smooth=extra_smooth,
        baseline_order=baseline_order
    )
    sam_proc = preprocess_spectrum(
        wavelengths,
        sample,
        smooth_window=window,
        poly_order=poly_order,
        extra_smooth=extra_smooth,
        baseline_order=baseline_order
    )
    
    # Compute difference spectrum
    diff = sam_proc - ref_proc
    
    # Initialize shifts list
    shifts = []
    
    # If peak_wavelength is specified, focus on that region
    if peak_wavelength is None and config:
        peak_wavelength = config.peak_wavelength

    if peak_wavelength is not None:
        # Find nearest point to peak_wavelength
        peak_idx = np.argmin(np.abs(wavelengths - peak_wavelength))
        # Define window around peak (wider for better correlation)
        window = 50  # points
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(wavelengths), peak_idx + window)
        
        if debug:
            print(f"\nAnalyzing peak region:")
            print(f"- Center: {wavelengths[peak_idx]:.1f} nm")
            print(f"- Window: {wavelengths[start_idx]:.1f} - {wavelengths[end_idx]:.1f} nm")
        
        # Calculate shift using cross-correlation
        ref_region = ref_proc[start_idx:end_idx]
        sam_region = sam_proc[start_idx:end_idx]
        
        # Normalize regions
        ref_norm = (ref_region - np.mean(ref_region)) / np.std(ref_region)
        sam_norm = (sam_region - np.mean(sam_region)) / np.std(sam_region)
        
        # Cross-correlate with interpolation for sub-pixel precision
        corr = np.correlate(sam_norm, ref_norm, mode='full')
        lags = np.arange(-(len(ref_norm)-1), len(ref_norm))
        max_idx = np.argmax(corr)
        
        # Fit parabola to peak for sub-pixel precision
        if max_idx > 0 and max_idx < len(corr)-1:
            fit_x = np.array([-1, 0, 1])
            fit_y = corr[max_idx-1:max_idx+2]
            coef = np.polyfit(fit_x, fit_y, 2)
            max_x = -coef[1] / (2 * coef[0])  # Peak of parabola
            max_lag = lags[max_idx] + max_x
        else:
            max_lag = lags[max_idx]
            
        if debug:
            print(f"Cross-correlation:")
            print(f"- Max correlation: {corr[max_idx]:.3f}")
            print(f"- Lag: {max_lag:.3f} points")
        
        # Convert lag to wavelength shift
        wl_step = np.mean(np.diff(wavelengths[start_idx:end_idx]))
        shift_nm = max_lag * wl_step
        
        if debug:
            print(f"Wavelength shift:")
            print(f"- Step size: {wl_step:.3f} nm/point")
            print(f"- Raw shift: {shift_nm:.3f} nm")
        
        # Calculate SNR using peak region
        # Use RMS of difference in peak region for signal
        diff_region = sam_region - ref_region
        signal = np.sqrt(np.mean(diff_region**2))
        # Use MAD for robust noise estimation
        noise = np.median(np.abs(diff_region - np.median(diff_region))) * 1.4826
        snr = signal / noise if noise > 0 else 0
        
        if debug:
            print(f"Signal quality:")
            print(f"- Signal RMS: {signal:.6f}")
            print(f"- Noise level: {noise:.6f}")
            print(f"- SNR: {snr:.1f}")
        
        if snr >= min_snr:
            shifts = [ShiftResult(
                wavelength=float(wavelengths[peak_idx]),
                shift_nm=float(shift_nm),
                intensity_change=float(signal),
                snr=float(snr)
            )]
        return shifts
    
    # Otherwise, find all significant peaks
    shifts = []
    for sign in [1, -1]:
        peaks, properties = find_peaks(
            sign * diff,
            height=0.001,
            distance=10,
            prominence=0.001,
            width=(3, 50)
        )
        
        if len(peaks) == 0:
            continue
            
        for i, peak_idx in enumerate(peaks):
            # Define analysis window
            # Use wider window for better SNR calculation
            window = int(max(properties['widths'][i] * 2, 20))
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(diff), peak_idx + window)
            
            # Use peak prominence for better sensitivity
            prominence = properties['prominences'][i]
            peak_width = properties['widths'][i]
            
            # Calculate enhanced SNR using prominence and width
            signal = prominence * np.sqrt(peak_width)  # Area-based signal
            # Use median absolute deviation for robust noise estimation
            noise_region = diff[start_idx:end_idx]
            noise = np.median(np.abs(noise_region - np.median(noise_region))) * 1.4826
            snr = signal / noise if noise > 0 else 0
            
            if snr >= min_snr:
                shifts.append(ShiftResult(
                    wavelength=float(wavelengths[peak_idx]),
                    shift_nm=float(wavelengths[peak_idx] - wavelengths[peak_idx-1]),
                    intensity_change=float(signal),
                    snr=float(snr)
                ))
                
                if debug:
                    print(f"\nShift detected at {wavelengths[peak_idx]:.1f} nm:")
                    print(f"- Magnitude: {wavelengths[peak_idx] - wavelengths[peak_idx-1]:.3f} nm")
                    print(f"- Intensity change: {signal:.3f}")
                    print(f"- SNR: {snr:.1f}")
    
    # Sort by SNR
    shifts.sort(key=lambda x: x.snr, reverse=True)
    return shifts

def analyze_concentration_series(wavelengths: np.ndarray,
                               reference: np.ndarray,
                               samples: List[np.ndarray],
                               concentrations: List[float],
                               roi: Optional[Tuple[float, float]] = None,
                               peak_wavelength: Optional[float] = None,
                               debug: bool = False) -> Dict:
    """Analyze concentration series for spectral shifts.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        samples: List of sample spectra
        concentrations: List of concentrations (ppm)
        roi: Optional wavelength range to analyze
        peak_wavelength: Optional specific wavelength to track
        debug: Print debug info
    """
    config: Optional[GasFeatures] = None
    if roi:
        try:
            config = get_config_for_roi(tuple(float(x) for x in roi))
        except ValueError:
            config = None

    if peak_wavelength is None and config:
        peak_wavelength = config.peak_wavelength

    if debug:
        print(f"\nAnalyzing concentration series:")
        print(f"- {len(samples)} samples")
        print(f"- Concentrations: {sorted(set(concentrations))} ppm")
        if roi:
            print(f"- ROI: {roi[0]:.1f} - {roi[1]:.1f} nm")
        if config:
            print(f"- Expected shift: {config.expected_shift:.1f} nm/ppm")
            print(f"- Direction: {'blue' if config.shift_direction < 0 else 'red'} shift")
    
    # Analyze each sample
    all_shifts = []
    for i, sample in enumerate(samples):
        # Find shifts
        shifts = find_shift_regions(
            wavelengths, reference, sample,
            roi=roi,
            peak_wavelength=peak_wavelength,
            debug=debug and i < 3
        )

        if shifts:
            # Use strongest shift
            shift = shifts[0]
            # Apply direction correction if configured
            if config:
                shift.shift_nm *= config.shift_direction
            
            all_shifts.append({
                'concentration': concentrations[i],
                'wavelength': shift.wavelength,
                'shift': shift.shift_nm,
                'intensity_change': shift.intensity_change,
                'snr': shift.snr
            })
            
            if debug and i < 3:
                print(f"\nSample {i+1} at {concentrations[i]} ppm:")
                print(f"Peak at {shift.wavelength:.1f} nm:")
                print(f"- Shift: {shift.shift_nm:.3f} nm")
                print(f"- SNR: {shift.snr:.1f}")
    
    if not all_shifts:
        if debug:
            print("No significant shifts detected")
        return {}
    
    # Convert to arrays for analysis
    conc_array = np.array([s['concentration'] for s in all_shifts])
    shift_array = np.array([s['shift'] for s in all_shifts])
    
    # Sort by concentration for better plotting
    sort_idx = np.argsort(conc_array)
    conc_array = conc_array[sort_idx]
    shift_array = shift_array[sort_idx]
    
    # Linear regression
    from scipy import stats
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
