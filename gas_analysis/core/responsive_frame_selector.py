"""
Responsive Frame Selection Module

This module implements time series analysis to identify the most responsive frames
from long-duration gas sensing experiments (500-700 frames per trial).

Key functions:
- find_responsive_frames: Main function to select top N responsive frames
- compute_response_metric: Calculate response strength for each frame
- visualize_time_series: Plot response over time with selected frames highlighted
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def compute_response_metric(frame: pd.DataFrame, 
                            air_ref: pd.DataFrame,
                            roi_range: Tuple[float, float] = (675, 689),
                            metric_type: str = 'integrated') -> float:
    """
    Compute a response metric for a single frame.
    
    Args:
        frame: DataFrame with columns [wavelength, intensity]
        air_ref: Air reference DataFrame
        roi_range: Wavelength ROI for analysis (nm)
        metric_type: 'integrated', 'peak', 'shift', or 'snr'
    
    Returns:
        response_metric: Higher value = stronger gas response
    """
    wl = frame['wavelength'].values
    intensity = frame['intensity'].values
    
    # Interpolate air reference to frame wavelengths
    ref_intensity = np.interp(wl, air_ref['wavelength'].values, air_ref['intensity'].values)
    
    # Debug: Print stats for first few frames
    if hasattr(compute_response_metric, 'debug_count'):
        compute_response_metric.debug_count += 1
    else:
        compute_response_metric.debug_count = 1
        
    if compute_response_metric.debug_count <= 1:
        print("\nDEBUG: Frame 0 stats:")
        print(f"  WL range: {wl.min():.2f}-{wl.max():.2f} nm")
        print(f"  Intensity stats: min={intensity.min():.4f}, max={intensity.max():.4f}, mean={intensity.mean():.4f}")
        print(f"  Ref stats: min={ref_intensity.min():.4f}, max={ref_intensity.max():.4f}, mean={ref_intensity.mean():.4f}")
    
    # Compute transmittance
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.where(ref_intensity > 0, intensity / ref_intensity, 0.0)
    
    if compute_response_metric.debug_count <= 1:
        print(f"  T raw stats: min={T.min():.4f}, max={T.max():.4f}, mean={T.mean():.4f}")
        n_over_1 = np.sum(T > 1.0)
        print(f"  T > 1.0 count: {n_over_1} ({n_over_1/len(T)*100:.1f}%)")
    
    T = np.clip(T, 1e-6, 1.0)
    
    # Compute absorbance
    A = -np.log10(T)
    
    if compute_response_metric.debug_count <= 1:
        print(f"  A stats: min={A.min():.4f}, max={A.max():.4f}, mean={A.mean():.4f}")
    
    # Filter to ROI
    roi_mask = (wl >= roi_range[0]) & (wl <= roi_range[1])
    A_roi = A[roi_mask]
    wl_roi = wl[roi_mask]
    
    if compute_response_metric.debug_count <= 1:
        print(f"  ROI ({roi_range[0]}-{roi_range[1]} nm) points: {len(A_roi)}")
        if len(A_roi) > 0:
            print(f"  A_roi stats: min={A_roi.min():.4f}, max={A_roi.max():.4f}, mean={A_roi.mean():.4f}")
    
    if len(A_roi) == 0:
        return 0.0
    
    # Calculate metric based on type
    if metric_type == 'integrated':
        # Integrated absorbance in ROI (most robust)
        metric = float(np.sum(A_roi))
    
    elif metric_type == 'peak':
        # Peak absorbance value
        metric = float(np.max(A_roi))
    
    elif metric_type == 'shift':
        # Magnitude of peak shift from expected baseline
        baseline_peak = 680.5  # Expected air baseline peak (nm)
        peak_wl = wl_roi[np.argmax(A_roi)]
        metric = float(abs(peak_wl - baseline_peak))
    
    elif metric_type == 'snr':
        # Signal-to-noise ratio
        signal = np.max(A_roi)
        # Estimate noise from first 100 wavelength points (outside ROI)
        noise = np.std(A[:100]) if len(A) >= 100 else np.std(A)
        metric = float(signal / noise) if noise > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}")
    
    return metric


def find_responsive_frames(frames: List[pd.DataFrame],
                          air_ref: pd.DataFrame,
                          top_n: int = 15,
                          roi_range: Tuple[float, float] = (675, 689),
                          metric_type: str = 'integrated',
                          smooth_window: int = 5,
                          method: str = 'top_n') -> Tuple[List[int], pd.DataFrame, Dict]:
    """
    Identify the most responsive frames from a long time series.
    
    Args:
        frames: List of all frames (typically 570-695 DataFrames)
        air_ref: Air reference DataFrame
        top_n: Number of responsive frames to select (default: 15)
        roi_range: Wavelength ROI for analysis (nm)
        metric_type: Type of response metric ('integrated', 'peak', 'shift', 'snr')
        smooth_window: Window size for smoothing time series
        method: Selection method ('top_n' or 'window')
    
    Returns:
        responsive_indices: List of frame indices showing strongest response
        time_series_df: DataFrame with time series data for visualization
        metadata: Dict with selection info and statistics
    """
    n_frames = len(frames)
    print(f"Analyzing {n_frames} frames to find {top_n} most responsive...")
    
    # Step 1: Compute response metric for each frame
    response_metrics = []
    for i, frame in enumerate(frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{n_frames}...")
        
        metric = compute_response_metric(frame, air_ref, roi_range, metric_type)
        response_metrics.append(metric)
    
    response_metrics = np.array(response_metrics)
    
    # Step 2: Smooth time series to remove noise
    if smooth_window > 1 and n_frames >= smooth_window:
        smoothed = np.convolve(response_metrics, np.ones(smooth_window)/smooth_window, mode='same')
    else:
        smoothed = response_metrics.copy()
    
    # Step 3: Select responsive frames based on method
    if method == 'top_n':
        # Select top N frames with highest individual response
        top_indices = np.argsort(smoothed)[-top_n:]
        top_indices = sorted(top_indices)  # Sort chronologically
        
    elif method == 'window':
        # Find continuous window with highest average response
        if top_n <= n_frames:
            window_sums = np.convolve(smoothed, np.ones(top_n), mode='valid')
            best_window_start = np.argmax(window_sums)
            top_indices = list(range(best_window_start, best_window_start + top_n))
        else:
            # If top_n > n_frames, use all frames
            top_indices = list(range(n_frames))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 4: Build time series DataFrame for visualization
    time_series_df = pd.DataFrame({
        'frame_index': np.arange(n_frames),
        'response_metric': response_metrics,
        'smoothed_response': smoothed,
        'selected': [i in top_indices for i in range(n_frames)]
    })
    
    # Step 5: Calculate metadata
    selected_metrics = response_metrics[top_indices]
    metadata = {
        'n_total_frames': n_frames,
        'n_selected': len(top_indices),
        'selection_method': method,
        'metric_type': metric_type,
        'roi_range': roi_range,
        'mean_response': float(np.mean(response_metrics)),
        'max_response': float(np.max(response_metrics)),
        'selected_mean_response': float(np.mean(selected_metrics)),
        'selected_std_response': float(np.std(selected_metrics)),
        'response_enhancement': float(np.mean(selected_metrics) / np.mean(response_metrics)) if np.mean(response_metrics) > 0 else 1.0,
        'first_selected_frame': int(top_indices[0]),
        'last_selected_frame': int(top_indices[-1]),
        'selection_span_frames': int(top_indices[-1] - top_indices[0] + 1),
    }
    
    print(f"✓ Selected {len(top_indices)} frames:")
    print(f"  Range: frames {top_indices[0]} to {top_indices[-1]} (span: {metadata['selection_span_frames']} frames)")
    print(f"  Mean response: {metadata['selected_mean_response']:.3f} (enhancement: {metadata['response_enhancement']:.2f}x)")
    
    return top_indices, time_series_df, metadata


def visualize_time_series(time_series_df: pd.DataFrame,
                         title: str = "Response Time Series",
                         save_path: Optional[Path] = None) -> None:
    """
    Visualize the response time series with selected frames highlighted.
    
    Args:
        time_series_df: DataFrame from find_responsive_frames
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot full time series
    ax.plot(time_series_df['frame_index'], 
           time_series_df['response_metric'], 
           'o-', markersize=2, linewidth=0.5, alpha=0.3, label='Raw response', color='gray')
    
    # Plot smoothed time series
    ax.plot(time_series_df['frame_index'],
           time_series_df['smoothed_response'],
           '-', linewidth=2, label='Smoothed response', color='blue')
    
    # Highlight selected frames
    selected_df = time_series_df[time_series_df['selected']]
    ax.scatter(selected_df['frame_index'],
              selected_df['smoothed_response'],
              s=50, color='red', zorder=5, label='Selected frames', marker='o')
    
    # Formatting
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Response Metric', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Time series plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def average_frames_intensity(frames: List[pd.DataFrame],
                            frame_indices: List[int],
                            method: str = 'mean') -> pd.DataFrame:
    """
    Average selected frames in intensity space.
    
    Args:
        frames: List of all frames
        frame_indices: Indices of frames to average
        method: 'mean' or 'median'
    
    Returns:
        averaged_spectrum: DataFrame [wavelength, intensity]
    """
    # Select frames
    selected_frames = [frames[i] for i in frame_indices]
    
    # Get common wavelength grid
    base_wavelength = selected_frames[0]['wavelength'].values
    
    # Interpolate all frames to common grid
    intensities = []
    for frame in selected_frames:
        intensity = np.interp(
            base_wavelength,
            frame['wavelength'].values,
            frame['intensity'].values
        )
        intensities.append(intensity)
    
    intensities = np.array(intensities)
    
    # Average
    if method == 'mean':
        avg_intensity = np.mean(intensities, axis=0)
    elif method == 'median':
        avg_intensity = np.median(intensities, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    result = pd.DataFrame({
        'wavelength': base_wavelength,
        'intensity': avg_intensity
    })
    
    return result


# ============================================================================
# Example usage
# ============================================================================

if __name__ == '__main__':
    # Example: Process one trial
    from pathlib import Path
    import pandas as pd
    
    # Load air reference
    air_ref = pd.read_csv('Kevin_Data/Acetone/air1.csv', header=0, names=['wavelength', 'intensity'])
    
    # Load all frames from a trial
    trial_path = Path('Kevin_Data/Acetone/3ppm/T1')
    csv_files = sorted(trial_path.glob('*.csv'))[:100]  # Test with first 100 frames
    
    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, header=0, names=['wavelength', 'intensity'])
        frames.append(df)
    
    print(f"Loaded {len(frames)} frames")
    
    # Find responsive frames
    responsive_indices, time_series, metadata = find_responsive_frames(
        frames, 
        air_ref,
        top_n=15,
        roi_range=(675, 689),
        metric_type='integrated',
        method='top_n'
    )
    
    # Visualize
    visualize_time_series(time_series, title="3ppm Trial 1 Response", save_path=Path('test_time_series.png'))
    
    # Average responsive frames
    avg_spectrum = average_frames_intensity(frames, responsive_indices, method='mean')
    
    print(f"\nAveraged spectrum shape: {avg_spectrum.shape}")
    print(f"Wavelength range: {avg_spectrum['wavelength'].min():.2f} - {avg_spectrum['wavelength'].max():.2f} nm")
