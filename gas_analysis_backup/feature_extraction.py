"""
Feature extraction and peak detection for gas analysis.
"""
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simps
import logging

logger = logging.getLogger(__name__)

def detect_peaks(spectrum: pd.DataFrame, 
                height: float = 0.1, 
                distance: int = 50,
                prominence: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Detect peaks in a spectrum.
    
    Args:
        spectrum: DataFrame with 'wavelength' and 'intensity' columns
        height: Required height of peaks
        distance: Required minimum distance (in index points) between peaks
        prominence: Required prominence of peaks
        
    Returns:
        Dictionary containing peak information:
        - 'peaks': Indices of the peaks in the spectrum
        - 'properties': Properties of the peaks
    """
    try:
        y = spectrum['intensity'].values
        peaks, properties = find_peaks(
            y, 
            height=height,
            distance=distance,
            prominence=prominence
        )
        
        # Calculate peak widths
        if len(peaks) > 0:
            widths = peak_widths(y, peaks, rel_height=0.5)
            properties['widths'] = widths[0]
            properties['width_heights'] = widths[1]
            properties['left_ips'] = widths[2]
            properties['right_ips'] = widths[3]
        
        return {
            'peaks': peaks,
            'properties': properties
        }
    except Exception as e:
        logger.error(f"Error detecting peaks: {str(e)}")
        return {'peaks': np.array([]), 'properties': {}}

def extract_peak_features(spectrum: pd.DataFrame, 
                        peaks_info: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Extract features from detected peaks.
    
    Args:
        spectrum: DataFrame with 'wavelength' and 'intensity' columns
        peaks_info: Dictionary containing peak information from detect_peaks()
        
    Returns:
        DataFrame with peak features
    """
    peaks = peaks_info['peaks']
    properties = peaks_info['properties']
    
    if len(peaks) == 0:
        return pd.DataFrame()
    
    # Create a DataFrame for peak features
    features = []
    
    for i, peak in enumerate(peaks):
        # Get peak position and intensity
        wavelength = spectrum['wavelength'].iloc[peak]
        intensity = properties['peak_heights'][i]
        
        # Calculate peak area
        left = int(np.floor(properties['left_ips'][i]))
        right = int(np.ceil(properties['right_ips'][i])) + 1
        
        # Ensure indices are within bounds
        left = max(0, left)
        right = min(len(spectrum), right)
        
        # Calculate area under the peak
        x = spectrum['wavelength'].iloc[left:right].values
        y = spectrum['intensity'].iloc[left:right].values
        area = simps(y, x)
        
        # Calculate full width at half maximum (FWHM)
        fwhm = properties['widths'][i] * (spectrum['wavelength'].iloc[1] - spectrum['wavelength'].iloc[0])
        
        features.append({
            'peak_index': i,
            'wavelength': wavelength,
            'intensity': intensity,
            'area': area,
            'fwhm': fwhm,
            'prominence': properties['prominences'][i] if 'prominences' in properties else np.nan,
            'left_base': spectrum['wavelength'].iloc[int(np.floor(properties['left_ips'][i]))],
            'right_base': spectrum['wavelength'].iloc[int(np.ceil(properties['right_ips'][i]))]
        })
    
    return pd.DataFrame(features)

def extract_statistical_features(spectrum: pd.DataFrame) -> Dict[str, float]:
    """
    Extract statistical features from a spectrum.
    
    Args:
        spectrum: DataFrame with 'intensity' column
        
    Returns:
        Dictionary of statistical features
    """
    y = spectrum['intensity'].values
    
    return {
        'mean': float(np.mean(y)),
        'std': float(np.std(y)),
        'max': float(np.max(y)),
        'min': float(np.min(y)),
        'range': float(np.max(y) - np.min(y)),
        'median': float(np.median(y)),
        'q1': float(np.percentile(y, 25)),
        'q3': float(np.percentile(y, 75)),
        'iqr': float(np.percentile(y, 75) - np.percentile(y, 25)),
        'skew': float(pd.Series(y).skew()),
        'kurtosis': float(pd.Series(y).kurtosis())
    }

def extract_all_features(spectrum: pd.DataFrame, 
                       peaks_info: Dict[str, np.ndarray]) -> Dict:
    """
    Extract all features from a spectrum.
    
    Args:
        spectrum: DataFrame with 'wavelength' and 'intensity' columns
        peaks_info: Dictionary containing peak information from detect_peaks()
        
    Returns:
        Dictionary containing all extracted features
    """
    # Extract peak features
    peak_features = extract_peak_features(spectrum, peaks_info)
    
    # Extract statistical features
    stats_features = extract_statistical_features(spectrum)
    
    # Combine all features
    features = {
        'num_peaks': len(peaks_info['peaks']),
        **stats_features
    }
    
    # Add peak-specific features
    if not peak_features.empty:
        # Get the most prominent peak
        main_peak = peak_features.loc[peak_features['intensity'].idxmax()]
        features.update({
            'main_peak_wavelength': main_peak['wavelength'],
            'main_peak_intensity': main_peak['intensity'],
            'main_peak_area': main_peak['area'],
            'main_peak_fwhm': main_peak['fwhm']
        })
    
    return features
