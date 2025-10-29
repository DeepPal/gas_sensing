"""Stability analysis for spectral time series."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StabilityMetrics:
    """Metrics for evaluating spectral stability."""
    mean_intensity: float
    std_intensity: float
    mean_wavelength: float
    std_wavelength: float
    drift_rate: float      # Linear drift in peak position
    noise_level: float     # Average frame-to-frame variation
    duration: float        # Time span in seconds
    quality_score: float   # Overall stability score

def analyze_frame_stability(frames: List[np.ndarray],
                          timestamps: List[float],
                          wavelengths: np.ndarray,
                          peak_wavelength: Optional[float] = None,
                          window_size: int = 5) -> Dict[str, np.ndarray]:
    """Analyze stability metrics across frames.
    
    Args:
        frames: List of intensity arrays
        timestamps: List of frame timestamps
        wavelengths: Shared wavelength array
        peak_wavelength: Optional wavelength to track
        window_size: Window for rolling statistics
    
    Returns:
        Dict with stability metrics over time
    """
    from .preprocessing import smooth_spectrum
    from .peak_analysis import find_peaks_advanced
    
    n_frames = len(frames)
    if n_frames < 2:
        raise ValueError("Need at least 2 frames")
    
    # Initialize metrics
    metrics = {
        'mean_intensity': np.zeros(n_frames),
        'std_intensity': np.zeros(n_frames),
        'peak_position': np.zeros(n_frames),
        'peak_intensity': np.zeros(n_frames),
        'frame_diff': np.zeros(n_frames-1),
        'rolling_std': np.zeros(n_frames),
    }
    
    # Analyze each frame
    for i, frame in enumerate(frames):
        metrics['mean_intensity'][i] = np.mean(frame)
        metrics['std_intensity'][i] = np.std(frame)
        
        # Find peak (either at specified wavelength or strongest peak)
        if peak_wavelength is not None:
            # Find closest point to peak_wavelength
            idx = np.argmin(np.abs(wavelengths - peak_wavelength))
            metrics['peak_position'][i] = wavelengths[idx]
            metrics['peak_intensity'][i] = frame[idx]
        else:
            # Find strongest peak
            peaks = find_peaks_advanced(wavelengths, frame, n_peaks=1)
            if peaks:
                metrics['peak_position'][i] = peaks[0].wavelength
                metrics['peak_intensity'][i] = peaks[0].intensity
            else:
                metrics['peak_position'][i] = np.nan
                metrics['peak_intensity'][i] = np.nan
        
        # Frame-to-frame difference
        if i > 0:
            metrics['frame_diff'][i-1] = np.mean(np.abs(frame - frames[i-1]))
        
        # Rolling standard deviation
        if i >= window_size:
            metrics['rolling_std'][i] = np.std(metrics['mean_intensity'][i-window_size:i])
    
    return metrics


def find_stable_regions(metrics: Dict[str, np.ndarray],
                       min_duration: float = 10.0,
                       intensity_threshold: float = 0.02,
                       position_threshold: float = 1.0) -> List[Tuple[int, int]]:
    """Find regions of stability in metrics.
    
    Args:
        metrics: Output from analyze_frame_stability()
        min_duration: Minimum duration in seconds
        intensity_threshold: Maximum allowed intensity variation
        position_threshold: Maximum allowed peak position variation
    
    Returns:
        List of (start_idx, end_idx) pairs
    """
    n_frames = len(metrics['mean_intensity'])
    
    # Combine stability criteria
    stable = np.ones(n_frames, dtype=bool)
    
    # Intensity stability
    intensity_stable = metrics['rolling_std'] < (intensity_threshold * np.mean(metrics['mean_intensity']))
    stable &= intensity_stable
    
    # Peak position stability (if available)
    if not np.all(np.isnan(metrics['peak_position'])):
        position_diff = np.abs(np.diff(metrics['peak_position']))
        position_stable = np.zeros_like(stable)
        position_stable[1:] = position_diff < position_threshold
        position_stable[0] = position_stable[1]
        stable &= position_stable
    
    # Find continuous stable regions
    regions = []
    start_idx = None
    
    for i, is_stable in enumerate(stable):
        if is_stable and start_idx is None:
            start_idx = i
        elif not is_stable and start_idx is not None:
            if i - start_idx >= min_duration:
                regions.append((start_idx, i))
            start_idx = None
    
    # Handle last region
    if start_idx is not None and n_frames - start_idx >= min_duration:
        regions.append((start_idx, n_frames))
    
    return regions


def compute_stability_metrics(frames: List[np.ndarray],
                            timestamps: List[float],
                            wavelengths: np.ndarray,
                            region: Tuple[int, int]) -> StabilityMetrics:
    """Compute comprehensive stability metrics for a region.
    
    Args:
        frames: List of intensity arrays
        timestamps: List of frame timestamps
        wavelengths: Shared wavelength array
        region: (start_idx, end_idx) for region
    
    Returns:
        StabilityMetrics object
    """
    start_idx, end_idx = region
    region_frames = frames[start_idx:end_idx]
    region_times = timestamps[start_idx:end_idx]
    
    # Stack frames
    X = np.vstack(region_frames)
    
    # Basic statistics
    mean_intensity = float(np.mean(X))
    std_intensity = float(np.std(X))
    
    # Track peak
    from .peak_analysis import find_peaks_advanced, track_peaks
    peak_positions = []
    for frame in region_frames:
        peaks = find_peaks_advanced(wavelengths, frame, n_peaks=1)
        if peaks:
            peak_positions.append(peaks[0].wavelength)
    
    if peak_positions:
        mean_wavelength = float(np.mean(peak_positions))
        std_wavelength = float(np.std(peak_positions))
        
        # Compute drift rate
        times = np.array(region_times) - region_times[0]
        if len(times) > 1:
            slope, _ = np.polyfit(times, peak_positions, 1)
            drift_rate = float(abs(slope))
        else:
            drift_rate = 0.0
    else:
        mean_wavelength = std_wavelength = drift_rate = float('nan')
    
    # Noise level from frame differences
    diffs = []
    for i in range(1, len(region_frames)):
        diff = np.mean(np.abs(region_frames[i] - region_frames[i-1]))
        diffs.append(diff)
    noise_level = float(np.mean(diffs)) if diffs else float('nan')
    
    # Duration
    duration = float(region_times[-1] - region_times[0])
    
    # Quality score combines multiple factors
    if not np.isnan(std_wavelength):
        intensity_stability = 1.0 / (1.0 + std_intensity/mean_intensity)
        wavelength_stability = 1.0 / (1.0 + std_wavelength)
        drift_factor = np.exp(-drift_rate)
        noise_factor = 1.0 / (1.0 + noise_level)
        duration_factor = np.minimum(1.0, duration/60.0)  # Saturate at 60s
        
        quality_score = float(np.mean([
            intensity_stability,
            wavelength_stability,
            drift_factor,
            noise_factor,
            duration_factor
        ]))
    else:
        quality_score = 0.0
    
    return StabilityMetrics(
        mean_intensity=mean_intensity,
        std_intensity=std_intensity,
        mean_wavelength=mean_wavelength,
        std_wavelength=std_wavelength,
        drift_rate=drift_rate,
        noise_level=noise_level,
        duration=duration,
        quality_score=quality_score
    )
