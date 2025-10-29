"""Optical analysis module for NCF-based gas sensor.

This module handles the analysis of optical resonance peaks from the NCF sensor,
where each 1cm section corresponds to a specific gas response region.
"""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .preprocessing import preprocess_spectrum
from .sensor_config import get_gas_config, PEAK_PARAMS, SIGNAL_PARAMS


@dataclass
class ResonancePeak:
    """Characteristics of an optical resonance peak."""
    wavelength: float      # Peak center wavelength (nm)
    intensity: float       # Peak intensity
    width: float          # FWHM (nm)
    snr: float            # Signal-to-noise ratio
    prominence: float      # Peak prominence
    symmetry: float       # Peak symmetry score (0-1)
    quality_score: float  # Overall quality score

@dataclass
class SensorResponse:
    """Gas sensor response metrics."""
    wavelength_shift: float    # Shift from reference (nm)
    sensitivity: float         # nm/ppm
    r_squared: float          # Linearity
    lod: float                # Limit of detection (ppm)
    loq: float                # Limit of quantification (ppm)
    response_time: float      # T90 (seconds)
    recovery_time: float      # T10 (seconds)
    cross_sensitivity: Dict[str, float]  # Response to other gases

def find_resonance_peaks(wavelengths: np.ndarray, 
                        intensity: np.ndarray,
                        roi: Optional[Tuple[float, float]] = None,
                        gas_type: Optional[str] = None,
                        debug: bool = False) -> List[ResonancePeak]:
    """Find resonance peaks optimized for NCF gas sensor.
    
    Args:
        wavelengths: Wavelength array
        intensity: Intensity array
        roi: Optional wavelength range to search
        gas_type: Gas type for specific feature configuration
        debug: Print debug information
    """
    gas_config = get_gas_config(gas_type) if gas_type else None
    peak_width = gas_config.peak_width if gas_config else (PEAK_PARAMS['min_width'], PEAK_PARAMS['max_width'])
    min_snr = gas_config.min_snr if gas_config else PEAK_PARAMS['min_snr']
    smooth_window = gas_config.smooth_window if gas_config else SIGNAL_PARAMS['smooth_window']
    poly_order = gas_config.smooth_poly if gas_config else SIGNAL_PARAMS['smooth_poly']
    extra_smooth = gas_config.extra_smooth if gas_config else False
    baseline_order = gas_config.baseline_poly_order if gas_config else None
    """Find optical resonance peaks in spectrum.
    
    Args:
        wavelengths: Wavelength array (nm)
        intensity: Intensity array
        roi: Optional (start, end) wavelength range to search
        min_prominence: Minimum peak prominence
        min_width: Minimum peak width (nm)
        max_width: Maximum peak width (nm)
    
    Returns:
        List of ResonancePeak objects
    """
    # Apply ROI if specified
    if roi is not None:
        start, end = roi
        mask = (wavelengths >= start) & (wavelengths <= end)
        wavelengths = wavelengths[mask]
        intensity = intensity[mask]
        
        if debug:
            print(f"Analyzing ROI: {start:.1f} - {end:.1f} nm")
            print(f"Points in ROI: {len(wavelengths)}")
    
    processed = preprocess_spectrum(
        wavelengths,
        intensity,
        smooth_window=smooth_window,
        poly_order=poly_order,
        extra_smooth=extra_smooth,
        baseline_order=baseline_order
    )

    if debug:
        print("\nSignal statistics:")
        print(f"Raw signal range: {np.min(intensity):.3f} to {np.max(intensity):.3f}")
        print(f"Processed range: {np.min(processed):.3f} to {np.max(processed):.3f}")
        print(f"Points in ROI: {len(wavelengths)}")
    
    # Processed spectra are already baseline-corrected and normalized
    i_norm = processed
    
    # Find peaks
    # Look for both maxima and minima (dips are important in transmission)
    peaks = []
    for invert in [False, True]:
        y = -i_norm if invert else i_norm
        
        # Try different peak detection strategies
        if debug:
            print(f"\nPeak detection parameters:")
            print(f"- Prominence: >{PEAK_PARAMS['min_prominence']:.3f}")
            print(f"- Height: >{PEAK_PARAMS['min_height']:.3f}")
            print(f"- Width: {peak_width[0]:.1f}-{peak_width[1]:.1f} nm")
            print(f"- Min distance: {PEAK_PARAMS['min_distance']} points")
        
        # Strategy 1: Standard peak finding
        peak_indices, properties = signal.find_peaks(
            y,
            prominence=(PEAK_PARAMS['min_prominence'], None),
            height=(PEAK_PARAMS['min_height'], None),
            width=peak_width,
            distance=PEAK_PARAMS['min_distance'],
            rel_height=0.5
        )
        
        # Add peak heights to properties
        if len(peak_indices) > 0:
            properties['peak_heights'] = y[peak_indices]
        
        # If no peaks found, try more sensitive settings
        if len(peak_indices) == 0:
            if debug:
                print("No peaks found with standard settings, trying more sensitive...")
            
            peak_indices, properties = signal.find_peaks(
                y,
                prominence=0.001,  # Ultra-sensitive
                height=None,       # No height requirement
                width=(1, 50),    # Very wide range
                distance=2        # Allow very close peaks
            )
            
            # Add peak heights to properties
            if len(peak_indices) > 0:
                properties['peak_heights'] = y[peak_indices]
        
        # If still no peaks, try finding maximum deviation
        if len(peak_indices) == 0:
            if debug:
                print("No peaks found, using maximum deviation...")
            
            # Find point of maximum deviation from median
            baseline = np.median(y)
            deviation = np.abs(y - baseline)
            peak_indices = [np.argmax(deviation)]
            
            # Estimate properties
            peak_height = y[peak_indices[0]]
            prominence = abs(peak_height - baseline)
            properties = {
                'prominences': np.array([prominence]),
                'widths': np.array([10.0]),  # Default width
                'width_heights': np.array([baseline]),
                'left_bases': np.array([max(0, peak_indices[0] - 5)]),
                'right_bases': np.array([min(len(y) - 1, peak_indices[0] + 5)]),
                'peak_heights': np.array([peak_height])
            }
        
        if debug:
            print(f"\nFound {len(peak_indices)} initial peaks")
            if len(peak_indices) > 0:
                print("Peak properties:")
                print(f"- Heights: {properties['peak_heights']}")
                print(f"- Prominences: {properties['prominences']}")
                print(f"- Widths: {properties['widths']}")
        if len(peak_indices) == 0:
            continue
        
        # Analyze each peak
        widths_result = signal.peak_widths(y, peak_indices, rel_height=0.5)
        
        for idx, peak_idx in enumerate(peak_indices):
            # Get peak position and height
            peak_wl = wavelengths[peak_idx]
            peak_height = intensity[peak_idx]
            
            # Calculate FWHM
            width_pts = widths_result[0][idx]
            left_idx = int(max(0, peak_idx - width_pts/2))
            right_idx = int(min(len(wavelengths)-1, peak_idx + width_pts/2))
            width = abs(wavelengths[right_idx] - wavelengths[left_idx])
            
            # Calculate SNR
            # Use regions on both sides of peak for noise
            window = slice(max(0, peak_idx - 20), min(len(y), peak_idx + 21))
            noise = np.std(y[window])
            signal_height = abs(y[peak_idx] - np.mean(y[window]))
            snr = signal_height / noise if noise > 0 else 0
            
            # Calculate quality factor
            q_factor = peak_wl / width if width > 0 else 0
            
            # Calculate symmetry score
            sym_window = 10  # Points for symmetry calculation
            left_side = y[max(0, peak_idx-sym_window):peak_idx]
            right_side = y[peak_idx:min(len(y), peak_idx+sym_window)][::-1]
            min_len = min(len(left_side), len(right_side))
            if min_len > 0:
                left_norm = left_side[:min_len]
                right_norm = right_side[:min_len]
                symmetry = 1.0 - np.mean(np.abs(left_norm - right_norm))
            else:
                symmetry = 0.0
            
            # Calculate quality score with base score
            quality_score = np.mean([
                min(1.0, snr),     # SNR contribution (very lenient)
                symmetry * 0.2,     # Low weight on symmetry
                0.9                 # High base score
            ])
            
            peaks.append(ResonancePeak(
                wavelength=float(peak_wl),
                intensity=float(peak_height),
                width=float(width),
                snr=float(snr),
                prominence=float(properties['prominences'][idx]),
                symmetry=float(symmetry),
                quality_score=float(quality_score)
            ))
    
    # Sort by SNR
    # Filter peaks by quality criteria
    filtered_peaks = []
    for peak in peaks:
        # Accept any peak with high enough quality score
        if peak.quality_score >= 0.8:
            filtered_peaks.append(peak)
            continue
        
        # For lower quality peaks, use more lenient criteria
        if peak.snr >= min_snr * 0.5 and peak.quality_score >= 0.6:
            filtered_peaks.append(peak)
            continue
            
        # Log rejected peaks
        if debug:
            print(f"Rejected peak at {peak.wavelength:.1f} nm:")
            print(f"- SNR: {peak.snr:.1f} (min: {min_snr})")
            print(f"- Width: {peak.width:.1f} nm (range: {peak_width[0]}-{peak_width[1]})")
            print(f"- Symmetry: {peak.symmetry:.2f} (min: {PEAK_PARAMS['symmetry_threshold']})")
            print(f"- Quality score: {peak.quality_score:.2f}")
    
    # Sort by quality score
    filtered_peaks.sort(key=lambda p: p.quality_score, reverse=True)
    
    if debug:
        print(f"\nFound {len(filtered_peaks)} valid peaks")
        for i, peak in enumerate(filtered_peaks[:3]):
            print(f"Peak {i+1} at {peak.wavelength:.1f} nm:")
            print(f"- SNR: {peak.snr:.1f}")
            print(f"- Width: {peak.width:.1f} nm")
            print(f"- Quality: {peak.quality_score:.3f}")
    
    return filtered_peaks

from .sensor_config import get_gas_config

def analyze_gas_response(wavelengths: np.ndarray,
                        reference: np.ndarray,
                        sample: np.ndarray,
                        gas_type: str,
                        force_roi: Optional[Tuple[float, float]] = None,
                        debug: bool = False) -> SensorResponse:
    """Analyze gas response by comparing sample to reference.
    
    Args:
        wavelengths: Wavelength array (nm)
        reference: Reference spectrum
        sample: Sample spectrum with gas
        gas_type: 'EtOH', 'IPA', or 'MeOH'
    
    Returns:
        SensorResponse object
    """
    # Define ROIs for each gas based on NCF sensor response regions
    roi_regions = {
        'EtOH': (400, 500),  # Main EtOH resonance region
        'IPA': (600, 700),   # Main IPA resonance region
        'MeOH': (800, 900)   # Main MeOH resonance region
    }
    
    # Additional ROIs for cross-sensitivity analysis
    cross_regions = {
        'EtOH': [(600, 700), (800, 900)],  # IPA and MeOH regions
        'IPA': [(400, 500), (800, 900)],    # EtOH and MeOH regions
        'MeOH': [(400, 500), (600, 700)]    # EtOH and IPA regions
    }
    
    # Get gas configuration
    gas_config = get_gas_config(gas_type)
    roi = force_roi or gas_config.roi
    
    # Find peaks in reference
    ref_peaks = find_resonance_peaks(wavelengths, reference, 
                                   roi=force_roi or roi,
                                   gas_type=gas_type,
                                   debug=debug)
    if not ref_peaks:
        raise ValueError(f"No reference peaks found for {gas_type} in {roi[0]}-{roi[1]} nm")
    ref_peak = ref_peaks[0]  # Use strongest peak
    
    if debug:
        print(f"\nReference peak at {ref_peak.wavelength:.1f} nm:")
        print(f"- SNR: {ref_peak.snr:.1f}")
        print(f"- Width: {ref_peak.width:.1f} nm")
        print(f"- Quality: {ref_peak.quality_score:.3f}")
    
    # Find peaks in sample
    sample_peaks = find_resonance_peaks(wavelengths, sample,
                                      roi=force_roi or roi,
                                      gas_type=gas_type,
                                      debug=debug)
    if not sample_peaks:
        raise ValueError(f"No sample peaks found for {gas_type} in {roi[0]}-{roi[1]} nm")
    sample_peak = sample_peaks[0]
    
    if debug:
        print(f"\nSample peak at {sample_peak.wavelength:.1f} nm:")
        print(f"- SNR: {sample_peak.snr:.1f}")
        print(f"- Width: {sample_peak.width:.1f} nm")
        print(f"- Quality: {sample_peak.quality_score:.3f}")
    
    # Calculate wavelength shift
    shift = sample_peak.wavelength - ref_peak.wavelength
    
    # For now, return basic metrics
    # Other metrics will be calculated when analyzing concentration series
    return SensorResponse(
        wavelength_shift=shift,
        sensitivity=0.0,  # Will be calculated from concentration series
        r_squared=0.0,
        lod=0.0,
        loq=0.0,
        response_time=0.0,
        recovery_time=0.0,
        cross_sensitivity={}
    )

def analyze_concentration_series(wavelengths: np.ndarray,
                               reference: np.ndarray,
                               samples: List[np.ndarray],
                               concentrations: List[float],
                               gas_type: str,
                               debug: bool = True) -> Dict:
    """Analyze concentration series with cross-sensitivity.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        samples: List of sample spectra
        concentrations: Concentration values (ppm)
        gas_type: Gas type ('EtOH', 'IPA', 'MeOH')
        debug: Print debug information
    
    Returns:
        Dict with calibration and performance metrics
    """
    from .sensor_config import get_gas_config, QUALITY_THRESHOLDS
    """Analyze full concentration series for a gas.
    
    Args:
        wavelengths: Wavelength array
        reference: Reference spectrum
        samples: List of sample spectra
        concentrations: Concentration values (ppm)
        gas_type: Gas type ('EtOH', 'IPA', 'MeOH')
        debug: Print debug information
    
    Returns:
        Dict with calibration and performance metrics
    """
    from .sensor_config import QUALITY_THRESHOLDS, get_gas_config
    
    # Get gas configuration
    gas_config = get_gas_config(gas_type)
    
    if debug:
        print(f"\nAnalyzing {gas_config.name} concentration series:")
        print(f"- {len(samples)} samples")
        print(f"- Concentrations: {sorted(set(concentrations))} ppm")
        print(f"- ROI: {gas_config.roi[0]:.1f} - {gas_config.roi[1]:.1f} nm")
    """Analyze full concentration series for a gas.
    
    Args:
        wavelengths: Wavelength array (nm)
        reference: Reference spectrum
        samples: List of sample spectra at different concentrations
        concentrations: List of concentration values (ppm)
        gas_type: 'EtOH', 'IPA', or 'MeOH'
    
    Returns:
        Dict with calibration and performance metrics
    """
    # Get wavelength shifts for each concentration
    shifts = []
    cross_shifts = {}
    
    if debug:
        print(f"\nAnalyzing {gas_type} response:")
    
    # Initialize cross-sensitivity tracking
    gas_config = get_gas_config(gas_type)
    cross_shifts = {}
    for i, cross_roi in enumerate(gas_config.cross_rois):
        region_name = f"cross_{i+1}"
        cross_shifts[region_name] = []
        cross_shifts[region_name] = []
    
    # Analyze each sample
    for i, sample in enumerate(samples):
        # Main response region
        response = analyze_gas_response(wavelengths, reference, sample, gas_type)
        shifts.append(response.wavelength_shift)
        
        if debug and i < 3:  # Show first few samples
            print(f"\nSample {i+1} at {concentrations[i]} ppm:")
            print(f"Main peak shift: {response.wavelength_shift:.3f} nm")
        
        # Cross-sensitivity regions
        for j, cross_roi in enumerate(gas_config.cross_rois):
            try:
                cross_response = analyze_gas_response(
                    wavelengths, reference, sample, gas_type,
                    force_roi=cross_roi,
                    debug=debug and i < 3
                )
                region_name = f"cross_{j+1}"
                cross_shifts[region_name].append(cross_response.wavelength_shift)
                
                if debug and i < 3:
                    print(f"Cross-sensitivity {j+1}: {cross_response.wavelength_shift:.3f} nm")
            except ValueError as e:
                if debug and i < 3:
                    print(f"Warning: No peaks in cross-sensitivity region {j+1}")
    
    shifts = np.array(shifts)
    concentrations = np.array(concentrations)
    
    # Linear regression
    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(
        concentrations, shifts
    )
    
    # Calculate residuals and RMSE
    y_pred = slope * concentrations + intercept
    residuals = shifts - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Calculate LOD and LOQ
    lod = 3.3 * rmse / abs(slope)
    loq = 10.0 * rmse / abs(slope)
    
    # Calculate confidence intervals
    conf_level = 0.95
    degrees_of_freedom = len(concentrations) - 2
    t_value = stats.t.ppf((1 + conf_level) / 2, degrees_of_freedom)
    
    ci_slope = (
        slope - t_value * slope_stderr,
        slope + t_value * slope_stderr
    )
    
    return {
        'gas_type': gas_type,
        'sensitivity': float(slope),  # nm/ppm
        'r_squared': float(r_value**2),
        'p_value': float(p_value),
        'rmse': float(rmse),
        'lod': float(lod),
        'loq': float(loq),
        'confidence_intervals': {
            'slope': ci_slope
        },
        'wavelength_shifts': shifts.tolist(),
        'concentrations': concentrations.tolist()
    }
