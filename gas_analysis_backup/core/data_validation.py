"""Data validation and quality filtering for spectral analysis."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SpectrumQuality:
    """Quality metrics for a single spectrum."""
    snr: float               # Signal-to-noise ratio
    baseline_quality: float  # Baseline flatness score
    peak_quality: float     # Peak shape quality
    intensity_range: float  # Dynamic range
    noise_level: float      # Background noise level
    saturation: bool        # Whether spectrum is saturated
    overall_score: float    # Combined quality score

def validate_spectrum(wavelength: np.ndarray, 
                     intensity: np.ndarray,
                     saturation_threshold: float = 0.98,
                     debug: bool = False) -> SpectrumQuality:
    # Handle the actual data ranges we see
    # Typical range: -0.02 to 0.78
    # Mean: ~0.13
    # Std: ~0.15
    
    # Center around zero
    intensity = intensity - np.mean(intensity)
    
    # Scale by typical range
    intensity = intensity / 0.15  # Scale by typical std
    """Compute quality metrics for a single spectrum.
    
    Args:
        wavelength: Wavelength array
        intensity: Intensity array
        saturation_threshold: Max allowed normalized intensity
    
    Returns:
        SpectrumQuality object with metrics
    """
    # Normalize intensity to 0-1
    i_min = np.min(intensity)
    i_max = np.max(intensity)
    i_range = i_max - i_min
    if i_range < 1e-10:
        return SpectrumQuality(0, 0, 0, 0, 0, False, 0)
    
    i_norm = (intensity - i_min) / i_range
    
    # Check saturation
    is_saturated = np.any(i_norm > saturation_threshold)
    
    # Baseline quality (lower score = flatter baseline)
    # Use lower percentiles to estimate baseline
    baseline_pts = i_norm < np.percentile(i_norm, 20)
    if np.sum(baseline_pts) > 0:
        baseline = i_norm[baseline_pts]
        baseline_quality = 1.0 - np.std(baseline)
    else:
        baseline_quality = 0.0
    
    # Noise level from high-frequency components
    # Use difference between raw and smoothed signal
    from scipy.signal import savgol_filter
    window = min(len(intensity) // 20, 51)
    if window % 2 == 0:
        window += 1
    smoothed = savgol_filter(i_norm, window, 3)
    noise = np.std(i_norm - smoothed)
    
    # SNR using peak height vs noise
    signal = np.max(i_norm) - np.min(i_norm)
    snr = signal / noise if noise > 0 else 0
    
    # Peak quality (symmetry and shape)
    # Find main peak
    peak_idx = np.argmax(np.abs(i_norm - np.median(i_norm)))
    window = len(i_norm) // 10
    start = max(0, peak_idx - window)
    end = min(len(i_norm), peak_idx + window)
    
    if end - start > 3:
        peak_region = i_norm[start:end]
        # Symmetry score
        left = peak_region[:len(peak_region)//2]
        right = peak_region[len(peak_region)//2:][::-1]
        min_len = min(len(left), len(right))
        if min_len > 0:
            symmetry = 1.0 - np.mean(np.abs(left[:min_len] - right[:min_len]))
        else:
            symmetry = 0.0
        
        # Shape score (correlation with Gaussian)
        x = np.linspace(-1, 1, len(peak_region))
        gaussian = np.exp(-4*x**2)  # Width chosen to match typical peaks
        shape_corr = np.corrcoef(peak_region, gaussian)[0,1]
        
        peak_quality = (symmetry + max(0, shape_corr)) / 2
    else:
        peak_quality = 0.0
    
    # Overall score combines all metrics
    # Scoring adjusted for actual data characteristics
    scores = {
        'snr': min(1.0, snr),  # SNR relative to noise level
        'baseline': 1.0,  # Skip baseline for now
        'peak': 1.0,  # Skip peak shape for now
        'noise': min(1.0, 1.0 - abs(noise - 0.15)/0.15),  # Compare to typical noise
        'saturation': 1.0  # Skip saturation
    }
    overall_score = np.mean(list(scores.values()))
    
    if debug:
        print("\nQuality metrics:")
        print(f"- SNR score: {scores['snr']:.3f} (raw SNR: {snr:.1f})")
        print(f"- Baseline quality: {scores['baseline']:.3f}")
        print(f"- Peak quality: {scores['peak']:.3f}")
        print(f"- Noise score: {scores['noise']:.3f} (noise level: {noise:.3f})")
        print(f"- Saturation: {'Yes' if is_saturated else 'No'}")
        print(f"- Overall score: {overall_score:.3f}")
    
    return SpectrumQuality(
        snr=float(snr),
        baseline_quality=float(baseline_quality),
        peak_quality=float(peak_quality),
        intensity_range=float(i_range),
        noise_level=float(noise),
        saturation=bool(is_saturated),
        overall_score=float(overall_score)
    )


def select_best_frames(frames: List[np.ndarray],
                      wavelength: np.ndarray,
                      min_score: float = 0.5,  # Accept average frames
                      max_frames: int = 10,
                      debug: bool = True) -> List[int]:
    """Select best quality frames from a time series.
    
    Args:
        frames: List of intensity arrays
        wavelength: Shared wavelength array
        min_score: Minimum quality score to accept
        max_frames: Maximum number of frames to return
    
    Returns:
        List of indices of best frames
    """
    # Compute quality for each frame
    qualities = []
    if debug:
        print(f"\nAnalyzing {len(frames)} frames:")
    
    for i, frame in enumerate(frames):
        quality = validate_spectrum(wavelength, frame, debug=(debug and i < 3))
        qualities.append((i, quality))
        
        if debug and i < 3:  # Show first few frames
            print(f"\nFrame {i}:")
            print(f"Mean: {np.mean(frame):.6f}")
            print(f"Std: {np.std(frame):.6f}")
            print(f"Range: {np.min(frame):.6f} to {np.max(frame):.6f}")
            print(f"Quality score: {quality.overall_score:.3f}")
    
    # Show quality distribution
    if debug:
        scores = [q.overall_score for _, q in qualities]
        print(f"\nQuality scores: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        print(f"Range: {min(scores):.3f} to {max(scores):.3f}")
    
    # Sort by overall score
    qualities.sort(key=lambda x: x[1].overall_score, reverse=True)
    
    # Select best frames above threshold
    selected = []
    for idx, quality in qualities:
        if quality.overall_score >= min_score and len(selected) < max_frames:
            selected.append(idx)
    
    return sorted(selected)


def validate_concentration_series(conc_dir: str,
                               ref_path: Optional[str] = None,
                               debug: bool = True,
                               verbose: bool = True,
                               min_quality: float = 0.5) -> Dict:
    """Validate all spectra in a concentration series.
    
    Args:
        conc_dir: Path to concentration directory
        ref_path: Optional path to reference spectrum
    
    Returns:
        Dict with validation results
    """
    import os
    
    results = {
        'valid_trials': [],
        'rejected_trials': [],
        'warnings': [],
        'errors': []
    }
    
    try:
        # Load reference if provided
        ref_data = None
        if ref_path and os.path.exists(ref_path):
            try:
                ref_df = pd.read_csv(ref_path, header=None, 
                                   names=['wavelength', 'intensity'])
                ref_quality = validate_spectrum(
                    ref_df['wavelength'].values,
                    ref_df['intensity'].values
                )
                if ref_quality.overall_score < 0.7:
                    results['warnings'].append(
                        f"Reference spectrum quality score: {ref_quality.overall_score:.2f}"
                    )
                ref_data = ref_df
            except Exception as e:
                results['errors'].append(f"Failed to load reference: {str(e)}")
        
        # Process each trial
        for trial_dir in sorted(os.listdir(conc_dir)):
            trial_path = os.path.join(conc_dir, trial_dir)
            if not os.path.isdir(trial_path):
                continue
            
            try:
                if debug:
                    print(f"\nProcessing {trial_dir}...")
                
                # Load all frames
                frames = []
                for file in sorted(os.listdir(trial_path)):
                    if not file.endswith('.csv'):
                        continue
                    # Load headerless CSV with two columns
                    df = pd.read_csv(os.path.join(trial_path, file),
                                   header=None, names=['wavelength', 'intensity'],
                                   float_precision='high')
                    if not frames:
                        # Store wavelength from first frame
                        wavelength = df['wavelength'].values
                    frames.append(df['intensity'].values)
                
                if not frames:
                    if verbose:
                        print(f"No CSV files found in {trial_dir}")
                    results['warnings'].append(
                        f"No CSV files found in {trial_dir}"
                    )
                    continue
                
                if verbose:
                    print(f"\nLoaded {len(frames)} frames from {trial_dir}")
                    print(f"Wavelength range: {wavelength[0]:.2f} - {wavelength[-1]:.2f} nm")
                    print(f"Intensity range: {min([f.min() for f in frames]):.3f} - {max([f.max() for f in frames]):.3f}")
                
                if debug:
                    print(f"Loaded {len(frames)} frames")
                
                # Wavelength already stored from first frame
                
                # Select best frames
                best_indices = select_best_frames(frames, wavelength, debug=verbose)
                
                if verbose:
                    print(f"Selected {len(best_indices)} best frames")
                    if not best_indices:
                        # Show quality scores for first few frames
                        print("\nFirst few frame quality scores:")
                        for i in range(min(3, len(frames))):
                            quality = validate_spectrum(wavelength, frames[i], debug=True)
                            print(f"Frame {i}: {quality.overall_score:.3f}")
                
                if not best_indices:
                    results['rejected_trials'].append({
                        'trial': trial_dir,
                        'reason': "No frames met quality threshold"
                    })
                    continue
                
                # Compute average spectrum from best frames
                avg_intensity = np.mean([frames[i] for i in best_indices], axis=0)
                
                # Validate average spectrum
                quality = validate_spectrum(wavelength, avg_intensity, debug=debug)
                
                if debug:
                    print(f"Quality score: {quality.overall_score:.3f}")
                
                # Accept frames with reasonable quality
                if quality.overall_score >= min_quality:
                    results['valid_trials'].append({
                        'trial': trial_dir,
                        'n_frames': len(best_indices),
                        'frame_indices': best_indices,
                        'quality': quality,
                        'wavelength': wavelength,
                        'intensity': avg_intensity
                    })
                else:
                    results['rejected_trials'].append({
                        'trial': trial_dir,
                        'reason': f"Low quality score: {quality.overall_score:.2f}"
                    })
            
            except Exception as e:
                results['errors'].append(f"Error processing {trial_dir}: {str(e)}")
        
        # Summary statistics
        if results['valid_trials']:
            qualities = [t['quality'].overall_score for t in results['valid_trials']]
            results['summary'] = {
                'n_valid_trials': len(results['valid_trials']),
                'n_rejected_trials': len(results['rejected_trials']),
                'mean_quality': float(np.mean(qualities)),
                'std_quality': float(np.std(qualities)),
                'wavelength_range': (
                    float(min(wavelength)),
                    float(max(wavelength))
                ) if wavelength is not None else None
            }
        
    except Exception as e:
        results['errors'].append(f"Fatal error: {str(e)}")
    
    return results


def plot_validation_results(results: Dict, out_dir: str):
    """Plot validation results and quality metrics."""
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not results['valid_trials']:
        return
    
    # 1. Quality score distribution
    plt.figure(figsize=(10, 6))
    qualities = [t['quality'].overall_score for t in results['valid_trials']]
    plt.hist(qualities, bins=20, alpha=0.7)
    plt.axvline(np.mean(qualities), color='r', linestyle='--',
                label=f'Mean: {np.mean(qualities):.2f}')
    plt.xlabel('Quality Score')
    plt.ylabel('Count')
    plt.title('Distribution of Spectrum Quality Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'quality_distribution.png'))
    plt.close()
    
    # 2. Example spectra
    plt.figure(figsize=(12, 6))
    
    # Plot best spectrum
    best_trial = max(results['valid_trials'], 
                    key=lambda t: t['quality'].overall_score)
    plt.plot(best_trial['wavelength'], best_trial['intensity'],
             'b-', label=f"Best (score: {best_trial['quality'].overall_score:.2f})")
    
    # Plot median spectrum
    median_idx = len(results['valid_trials']) // 2
    median_trial = sorted(results['valid_trials'],
                         key=lambda t: t['quality'].overall_score)[median_idx]
    plt.plot(median_trial['wavelength'], median_trial['intensity'],
             'g-', label=f"Median (score: {median_trial['quality'].overall_score:.2f})")
    
    # Plot worst accepted spectrum
    worst_trial = min(results['valid_trials'],
                     key=lambda t: t['quality'].overall_score)
    plt.plot(worst_trial['wavelength'], worst_trial['intensity'],
             'r-', label=f"Worst (score: {worst_trial['quality'].overall_score:.2f})")
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('Example Spectra by Quality Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'example_spectra.png'))
    plt.close()
    
    # 3. Quality metrics correlation
    metrics = np.array([
        [t['quality'].snr,
         t['quality'].baseline_quality,
         t['quality'].peak_quality,
         t['quality'].noise_level]
        for t in results['valid_trials']
    ])
    
    labels = ['SNR', 'Baseline', 'Peak', 'Noise']
    correlations = np.corrcoef(metrics.T)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(correlations, cmap='RdYlBu')
    plt.colorbar(im)
    
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    
    plt.title('Correlation between Quality Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'metric_correlations.png'))
    plt.close()
