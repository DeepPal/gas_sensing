"""Stability and drift analysis for gas sensors."""

import numpy as np
from scipy import stats, signal
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StabilityResult:
    """Results from stability analysis."""
    # Short-term stability (1 min)
    noise_short: float      # nm
    drift_short: float      # nm/min
    
    # Long-term stability (60 min)
    noise_long: float       # nm
    drift_long: float       # nm/min
    
    # Allan deviation analysis
    tau_opt: float         # Optimal averaging time (min)
    adev_opt: float        # Minimum Allan deviation (nm)
    
    # Temperature effects
    temp_sensitivity: float  # nm/°C
    temp_hysteresis: float  # nm
    
    # Environmental effects
    pressure_coeff: float   # nm/kPa
    humidity_coeff: float   # nm/%RH

def calculate_allan_deviation(data: np.ndarray,
                            rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Allan deviation for stability analysis."""
    # Generate tau values
    max_tau = len(data) // 10
    tau = np.logspace(0, np.log10(max_tau), 50).astype(int)
    tau = np.unique(tau)
    
    # Calculate Allan deviation
    adev = np.zeros(len(tau))
    adev_err = np.zeros(len(tau))
    
    for i, t in enumerate(tau):
        # Split data into chunks
        n_chunks = len(data) // t
        if n_chunks < 2:
            break
            
        # Calculate means of chunks
        chunks = data[:n_chunks*t].reshape(n_chunks, t)
        chunk_means = np.mean(chunks, axis=1)
        
        # Calculate Allan variance
        diffs = np.diff(chunk_means)
        avar = np.sum(diffs**2) / (2 * (n_chunks - 1))
        adev[i] = np.sqrt(avar)
        
        # Estimate error
        adev_err[i] = adev[i] / np.sqrt(2 * (n_chunks - 1))
    
    # Convert tau to minutes
    tau = tau / (rate * 60)
    
    return tau, adev, adev_err

def analyze_drift(time: np.ndarray,
                 signal: np.ndarray,
                 window: int = 60) -> Dict:
    """Analyze signal drift over time."""
    # Detrend signal to separate drift
    # Detrend signal using linear fit
    x = np.arange(len(signal))
    coeffs = np.polyfit(x, signal, 1)
    trend = np.polyval(coeffs, x)
    drift = signal - trend
    
    # Calculate drift rates
    time_min = time / 60  # Convert to minutes
    drift_rate = np.polyfit(time_min, drift, 1)[0]  # nm/min
    
    # Calculate noise levels
    noise = np.std(trend)
    
    # Short-term analysis (1 min window)
    short_windows = []
    for i in range(0, len(time), window):
        if i + window <= len(time):
            window_data = trend[i:i+window]
            short_windows.append({
                'noise': np.std(window_data),
                'drift': np.polyfit(time_min[i:i+window], 
                                  drift[i:i+window], 1)[0]
            })
    
    # Calculate stability metrics
    noise_short = np.mean([w['noise'] for w in short_windows])
    drift_short = np.mean([w['drift'] for w in short_windows])
    
    # Long-term analysis (60 min)
    if len(time_min) >= 60:
        noise_long = np.std(trend)
        drift_long = drift_rate
    else:
        noise_long = np.nan
        drift_long = np.nan
    
    return {
        'time': time_min.tolist(),
        'baseline': trend.tolist(),
        'drift': drift.tolist(),
        'noise': noise,
        'noise_short': noise_short,
        'drift_short': drift_short,
        'noise_long': noise_long,
        'drift_long': drift_long
    }

def analyze_temperature_stability(signal: np.ndarray,
                               temperature: np.ndarray,
                               temp_range: Tuple[float, float]) -> Dict:
    """Analyze temperature effects on stability."""
    # Calculate temperature sensitivity
    temp_sens = np.polyfit(temperature, signal, 1)[0]
    
    # Analyze hysteresis
    temp_up = temperature[:-1] < temperature[1:]
    temp_down = temperature[:-1] > temperature[1:]
    
    signal_up = signal[:-1][temp_up]
    signal_down = signal[:-1][temp_down]
    
    # Interpolate to common temperature points
    temp_common = np.linspace(temp_range[0], temp_range[1], 100)
    signal_up_interp = np.interp(temp_common, 
                                temperature[:-1][temp_up],
                                signal_up)
    signal_down_interp = np.interp(temp_common,
                                  temperature[:-1][temp_down],
                                  signal_down)
    
    # Calculate hysteresis
    hysteresis = np.mean(np.abs(signal_up_interp - signal_down_interp))
    
    return {
        'temperature_sensitivity': float(temp_sens),
        'hysteresis': float(hysteresis),
        'temp_range': list(temp_range),
        'temp_response_up': signal_up_interp.tolist(),
        'temp_response_down': signal_down_interp.tolist(),
        'temp_points': temp_common.tolist()
    }

def analyze_environmental_stability(signal: np.ndarray,
                                 pressure: np.ndarray,
                                 humidity: np.ndarray) -> Dict:
    """Analyze environmental effects on stability."""
    # Calculate pressure coefficient
    pressure_coeff = np.polyfit(pressure, signal, 1)[0]
    
    # Calculate humidity coefficient
    humidity_coeff = np.polyfit(humidity, signal, 1)[0]
    
    # Calculate correlation coefficients
    pressure_corr = stats.pearsonr(pressure, signal)[0]
    humidity_corr = stats.pearsonr(humidity, signal)[0]
    
    return {
        'pressure_coefficient': float(pressure_coeff),
        'humidity_coefficient': float(humidity_coeff),
        'pressure_correlation': float(pressure_corr),
        'humidity_correlation': float(humidity_corr),
        'pressure_range': [float(np.min(pressure)), float(np.max(pressure))],
        'humidity_range': [float(np.min(humidity)), float(np.max(humidity))]
    }

def analyze_stability(time: np.ndarray,
                     signal: np.ndarray,
                     temperature: Optional[np.ndarray] = None,
                     pressure: Optional[np.ndarray] = None,
                     humidity: Optional[np.ndarray] = None,
                     sample_rate: float = 1.0) -> Dict:
    """Comprehensive stability analysis."""
    # Basic drift analysis
    drift_results = analyze_drift(time, signal)
    
    # Allan deviation analysis
    tau, adev, adev_err = calculate_allan_deviation(signal, sample_rate)
    
    # Find optimal averaging time
    tau_opt = tau[np.argmin(adev)]
    adev_opt = np.min(adev)
    
    # Temperature stability
    if temperature is not None:
        temp_results = analyze_temperature_stability(
            signal, temperature,
            (np.min(temperature), np.max(temperature))
        )
    else:
        temp_results = {
            'temperature_sensitivity': np.nan,
            'hysteresis': np.nan,
            'temp_range': [np.nan, np.nan],
            'temp_response_up': [],
            'temp_response_down': [],
            'temp_points': []
        }
    
    # Environmental stability
    if pressure is not None and humidity is not None:
        env_results = analyze_environmental_stability(
            signal, pressure, humidity
        )
    else:
        env_results = {
            'pressure_coefficient': np.nan,
            'humidity_coefficient': np.nan,
            'pressure_correlation': np.nan,
            'humidity_correlation': np.nan,
            'pressure_range': [np.nan, np.nan],
            'humidity_range': [np.nan, np.nan]
        }
    
    # Combine results
    results = {
        **drift_results,
        'tau': tau.tolist(),
        'adev': adev.tolist(),
        'adev_err': adev_err.tolist(),
        'tau_opt': tau_opt,
        'adev_opt': adev_opt,
        **temp_results,
        **env_results
    }
    
    return results
