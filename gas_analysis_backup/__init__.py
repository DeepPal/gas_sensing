"""
Gas Analysis Package
-------------------
A professional package for gas spectral analysis with advanced processing
and machine learning capabilities.
"""

__version__ = "0.1.0"

# Import key components for easier access
from .data_loader import load_spectral_data, preprocess_spectra
from .analyzer import GasAnalyzer
from .visualization import plot_spectrum, plot_spectra, plot_calibration_curve

__all__ = [
    'load_spectral_data',
    'preprocess_spectra',
    'GasAnalyzer',
    'plot_spectrum',
    'plot_spectra',
    'plot_calibration_curve'
]
