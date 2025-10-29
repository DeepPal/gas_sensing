"""Configuration for NCF-based gas sensor analysis."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class GasFeatures:
    """Features for specific gas detection."""
    name: str                    # Gas name
    description: str             # Human readable description
    roi: Tuple[float, float]    # Main resonance region (nm)
    cross_rois: List[Tuple[float, float]]  # Cross-sensitivity regions
    expected_shift: float       # Expected nm/ppm sensitivity (magnitude)
    min_snr: float             # Minimum SNR for valid peak
    peak_width: Tuple[float, float]  # Min/Max FWHM (nm)
    baseline_poly_order: int    # Polynomial order for baseline
    peak_wavelength: float      # Nominal resonance wavelength (nm)
    shift_direction: int        # +1 red shift, -1 blue shift
    smooth_window: int = 31     # Savitzky-Golay window length
    smooth_poly: int = 3        # Savitzky-Golay polynomial order
    extra_smooth: bool = False  # Apply additional smoothing pass
    color: str = "blue"         # Plotting helper

# Sensor configuration
SENSOR_CONFIG = {
    'EtOH': GasFeatures(
        name='Ethanol',
        description='First NCF section - EtOH sensitive',
        roi=(196, 296),          # First NCF section
        cross_rois=[
            (496, 596),          # Second NCF section
            (796, 896)           # Third NCF section
        ],
        expected_shift=2.0,
        min_snr=2.0,
        peak_width=(3, 20),
        baseline_poly_order=3,
        peak_wavelength=236.5,
        shift_direction=-1,
        smooth_window=41,
        smooth_poly=4,
        extra_smooth=True,
        color='blue'
    ),
    'IPA': GasFeatures(
        name='Isopropanol',
        description='Second NCF section - IPA sensitive',
        roi=(496, 596),         # Second NCF section
        cross_rois=[
            (196, 296),         # First NCF section
            (796, 896)          # Third NCF section
        ],
        expected_shift=1.5,
        min_snr=2.0,
        peak_width=(3, 20),
        baseline_poly_order=3,
        peak_wavelength=549.3,
        shift_direction=1,
        smooth_window=41,
        smooth_poly=4,
        extra_smooth=True,
        color='green'
    ),
    'MeOH': GasFeatures(
        name='Methanol',
        description='Third NCF section - MeOH sensitive',
        roi=(796, 896),        # Third NCF section
        cross_rois=[
            (196, 296),        # First NCF section
            (496, 596)         # Second NCF section
        ],
        expected_shift=1.0,
        min_snr=2.0,
        peak_width=(3, 20),
        baseline_poly_order=3,
        peak_wavelength=872.7,
        shift_direction=-1,
        smooth_window=31,
        smooth_poly=3,
        extra_smooth=False,
        color='red'
    )
}

# Peak detection parameters
PEAK_PARAMS = {
    'min_prominence': 0.01,
    'min_height': -np.inf,
    'min_distance': 5,
    'min_width': 1.0,
    'max_width': 40.0,
    'symmetry_threshold': 0.2
}

# Signal processing parameters
SIGNAL_PARAMS = {
    'smooth_window': 31,       # Very wide smoothing
    'smooth_poly': 4,         # Higher order polynomial
    'baseline_regions': [      # Regions for baseline estimation
        (195, 210),           # Low wavelength baseline
        (1000, 1019)          # High wavelength baseline
    ],
    'snr_window': 100        # Very wide SNR window
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_snr': 1.0,           # Extremely low SNR threshold
    'min_r2': 0.60,          # Extremely lenient R²
    'max_rmse': 5.0,         # Allow very high variation
    'min_sensitivity': 0.05,  # Extremely low sensitivity threshold
    'max_cross_sensitivity': 0.8  # Allow very high cross-sensitivity
}

def get_gas_config(gas_type: str) -> GasFeatures:
    """Get configuration for specific gas."""
    if gas_type not in SENSOR_CONFIG:
        raise ValueError(f"Unknown gas type: {gas_type}")
    return SENSOR_CONFIG[gas_type]


def get_config_for_roi(roi: Tuple[float, float]) -> GasFeatures:
    """Get configuration by region of interest."""
    roi_key = tuple(float(x) for x in roi)
    for config in SENSOR_CONFIG.values():
        if tuple(config.roi) == roi_key:
            return config
    raise ValueError(f"Unknown ROI: {roi}")


def sensor_config_dict() -> Dict[str, Dict[str, object]]:
    """Return sensor configuration as dictionaries for legacy consumers."""
    out: Dict[str, Dict[str, object]] = {}
    for gas, cfg in SENSOR_CONFIG.items():
        out[gas] = {
            'name': cfg.name,
            'description': cfg.description,
            'region': cfg.roi,
            'peak': cfg.peak_wavelength,
            'expected_shift': cfg.expected_shift * cfg.shift_direction,
            'shift_direction': cfg.shift_direction,
            'color': cfg.color,
            'cross_rois': cfg.cross_rois,
            'min_snr': cfg.min_snr,
            'peak_width': cfg.peak_width,
            'baseline_poly_order': cfg.baseline_poly_order,
            'smooth_window': cfg.smooth_window,
            'smooth_poly': cfg.smooth_poly,
            'extra_smooth': cfg.extra_smooth,
        }
    return out
