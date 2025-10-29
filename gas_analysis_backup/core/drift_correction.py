"""Real-time drift correction for gas sensors."""

import numpy as np
from scipy import signal, stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DriftParameters:
    """Parameters for drift correction."""
    window_size: int        # Size of moving window (points)
    poly_order: int        # Order of polynomial fit
    threshold: float       # Threshold for drift detection
    min_points: int       # Minimum points for estimation
    max_correction: float # Maximum allowed correction

def estimate_drift_parameters(signal_data: np.ndarray,
                            region: str = 'auto') -> DriftParameters:
    """Estimate optimal drift correction parameters."""
    # Calculate signal characteristics
    signal_std = np.std(signal_data)
    signal_range = np.ptp(signal_data)
    
    # Base parameters for each region
    base_params = {
        'EtOH': {
            'window_size': 60,  # 1 minute
            'poly_order': 2,
            'threshold': 0.1,
            'min_points': 10,
            'max_correction': 0.5
        },
        'IPA': {
            'window_size': 120,  # 2 minutes
            'poly_order': 2,
            'threshold': 0.15,
            'min_points': 15,
            'max_correction': 0.75
        },
        'MeOH': {
            'window_size': 180,  # 3 minutes
            'poly_order': 3,
            'threshold': 0.2,
            'min_points': 20,
            'max_correction': 1.0
        }
    }
    
    # Get base parameters
    params = base_params.get(region, base_params['IPA']).copy()
    
    # Adjust window size based on data length
    params['window_size'] = min(params['window_size'],
                              len(signal_data) // 4)
    
    # Adjust threshold based on signal variability
    noise_level = stats.iqr(np.diff(signal_data)) / np.sqrt(2)
    params['threshold'] = max(params['threshold'],
                            3 * noise_level / signal_range)
    
    # Use simple polynomial order
    params['poly_order'] = 2
    
    return DriftParameters(**params)

def detect_drift(signal_data: np.ndarray,
                timestamps: np.ndarray,
                params: DriftParameters) -> Tuple[bool, float]:
    """Detect if significant drift is present."""
    if len(signal_data) < params.min_points:
        return False, 0.0
    
    # Fit polynomial to estimate drift
    coeffs = np.polyfit(timestamps - timestamps[0],
                       signal_data,
                       params.poly_order)
    
    # Calculate drift rate (use highest order non-zero coefficient)
    for i, coeff in enumerate(coeffs[:-1]):  # Exclude constant term
        if abs(coeff) > 1e-10:  # Numerical threshold
            drift_rate = coeff * (params.poly_order - i)
            break
    else:
        drift_rate = 0.0
    
    # Check if drift exceeds threshold
    max_drift = drift_rate * (timestamps[-1] - timestamps[0])
    signal_range = np.ptp(signal_data)
    
    has_drift = abs(max_drift) > params.threshold * signal_range
    
    return has_drift, drift_rate

def estimate_drift_trend(signal_data: np.ndarray,
                        timestamps: np.ndarray,
                        params: DriftParameters) -> np.ndarray:
    """Estimate drift trend using robust polynomial fitting."""
    if len(signal_data) < params.min_points:
        return np.zeros_like(signal_data)
    
    # Normalize time to improve numerical stability
    t = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    
    # Iterative robust fitting
    weights = np.ones_like(signal_data)
    for _ in range(3):  # Number of iterations
        # Weighted polynomial fit
        coeffs = np.polyfit(t, signal_data, params.poly_order,
                           w=weights)
        trend = np.polyval(coeffs, t)
        
        # Update weights based on residuals
        residuals = signal_data - trend
        mad = stats.median_abs_deviation(residuals)
        weights = 1 / (1 + (residuals / (3 * mad))**2)
    
    return trend

def correct_drift(signal_data: np.ndarray,
                 timestamp: float,
                 region: str = 'auto',
                 reference_value: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """Apply real-time drift correction."""
    # Get drift parameters
    params = estimate_drift_parameters(signal_data, region)
    
    # Initialize output
    corrected_data = signal_data.copy()
    correction_info = {
        'has_drift': False,
        'drift_rate': 0.0,
        'correction_applied': np.zeros_like(signal_data),
        'params': params
    }
    
    # Calculate drift correction
    if reference_value is not None:
        # Use reference value as baseline
        correction = signal_data - reference_value
    else:
        # Use mean as baseline
        correction = signal_data - np.mean(signal_data)
    
    # Limit maximum correction
    max_corr = params.max_correction * np.ptp(signal_data)
    correction = np.clip(correction, -max_corr, max_corr)
    
    # Apply correction
    corrected_data = signal_data - correction
    correction_info['correction_applied'] = correction
    
    return corrected_data, correction_info

def analyze_drift_stability(signal_data: np.ndarray,
                          timestamps: np.ndarray,
                          region: str = 'auto') -> Dict:
    """Analyze drift stability and characteristics."""
    # Get drift parameters
    params = estimate_drift_parameters(signal_data, timestamps, region)
    
    # Calculate drift metrics
    drift_metrics = {}
    
    # Short-term stability (using differential)
    diff = np.diff(signal_data)
    drift_metrics['short_term_noise'] = stats.iqr(diff) / np.sqrt(2)
    
    # Long-term stability (using polynomial fit)
    trend = estimate_drift_trend(signal_data, timestamps, params)
    drift_metrics['long_term_drift'] = np.ptp(trend)
    
    # Calculate Allan deviation
    tau_values = []
    adev_values = []
    
    for m in range(1, len(signal_data)//4):
        # Divide data into chunks
        n_chunks = len(signal_data) // m
        if n_chunks < 2:
            break
            
        chunks = signal_data[:n_chunks*m].reshape(n_chunks, m)
        chunk_means = np.mean(chunks, axis=1)
        
        # Calculate Allan variance
        diffs = np.diff(chunk_means)
        avar = np.sum(diffs**2) / (2 * (n_chunks - 1))
        adev = np.sqrt(avar)
        
        tau = m * np.mean(np.diff(timestamps))
        tau_values.append(tau)
        adev_values.append(adev)
    
    drift_metrics['tau'] = np.array(tau_values)
    drift_metrics['adev'] = np.array(adev_values)
    
    # Find optimal averaging time
    min_adev_idx = np.argmin(adev_values)
    drift_metrics['optimal_tau'] = tau_values[min_adev_idx]
    drift_metrics['min_adev'] = adev_values[min_adev_idx]
    
    return drift_metrics
