"""
Core analysis functionality for gas spectral data.
"""
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from scipy import signal
from scipy.stats import linregress
from dataclasses import dataclass
import logging
import pandas as pd
from config.config_loader import load_config
from .data_loader import preprocess_spectra

logger = logging.getLogger(__name__)

@dataclass
class Peak:
    """Container for peak information."""
    position: float
    intensity: float
    width: float
    left_base: float
    right_base: float

class GasAnalyzer:
    """Main class for gas spectral analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the gas analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.peaks: List[Peak] = []
        
    def find_peaks(self, wavelengths: np.ndarray, spectrum: np.ndarray, 
                  **kwargs) -> List[Peak]:
        """
        Find peaks in a spectrum.
        
        Args:
            wavelengths: Array of wavelength values
            spectrum: Array of intensity values
            **kwargs: Additional arguments for peak detection
            
        Returns:
            List of detected peaks
        """
        # Get peak detection parameters from config or use defaults
        pd_cfg = self.config.get('peak_detection', {}) if isinstance(self.config, dict) else {}
        prominence = kwargs.get('prominence', pd_cfg.get('prominence', 0.1))
        width = kwargs.get('width', pd_cfg.get('width', 5))
        distance = kwargs.get('distance', pd_cfg.get('distance', 10))
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            spectrum,
            prominence=prominence,
            width=width,
            distance=distance
        )
        
        # Create Peak objects
        self.peaks = []
        for i, idx in enumerate(peak_indices):
            self.peaks.append(Peak(
                position=wavelengths[idx],
                intensity=spectrum[idx],
                width=properties['widths'][i] if 'widths' in properties else 0,
                left_base=wavelengths[properties['left_bases'][i]] if 'left_bases' in properties else 0,
                right_base=wavelengths[properties['right_bases'][i]] if 'right_bases' in properties else 0
            ))
            
        return self.peaks
    
    def calculate_concentration(self, peak: Peak, calibration_factor: float = 1.0) -> float:
        """
        Calculate gas concentration from peak properties.
        
        Args:
            peak: Detected peak
            calibration_factor: Calibration factor (intensity to concentration)
            
        Returns:
            Calculated concentration
        """
        return peak.intensity * calibration_factor
    
    def analyze_spectrum(self, wavelengths: np.ndarray, spectrum: np.ndarray) -> Dict:
        """
        Perform complete analysis on a spectrum.
        
        Args:
            wavelengths: Array of wavelength values
            spectrum: Array of intensity values
            
        Returns:
            Dictionary containing analysis results
        """
        # Find peaks
        peaks = self.find_peaks(wavelengths, spectrum)
        
        # Calculate peak areas (simplified)
        peak_areas = [p.intensity * p.width for p in peaks]
        
        # Basic statistics
        mean_intensity = float(np.mean(spectrum))
        std_intensity = float(np.std(spectrum))
        
        return {
            'peaks': peaks,
            'peak_areas': peak_areas,
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'snr': mean_intensity / std_intensity if std_intensity > 0 else 0
        }

    def analyze(self, data: Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]) -> Dict[str, pd.DataFrame]:
        if isinstance(data, pd.DataFrame):
            raw = data.copy()
        elif isinstance(data, (tuple, list)) and len(data) >= 2:
            raw = pd.DataFrame({'wavelength': np.asarray(data[0]), 'intensity': np.asarray(data[1])})
        elif isinstance(data, dict) and 'wavelength' in data and 'intensity' in data:
            raw = pd.DataFrame({'wavelength': np.asarray(data['wavelength']), 'intensity': np.asarray(data['intensity'])})
        else:
            raise TypeError("Unsupported data format for analyze().")

        pre_cfg = self.config.get('preprocessing', {})
        baseline_correction = bool(pre_cfg.get('baseline_correction', True))
        normalize = bool(pre_cfg.get('normalize', True))

        processed = preprocess_spectra(
            raw,
            reference=None,
            baseline_correction=baseline_correction,
            normalize=normalize,
        )

        return {
            'Raw': raw,
            'Processed': processed,
        }
