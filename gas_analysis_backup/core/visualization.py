"""Advanced visualization for gas sensor analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple
try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    sns = None
from scipy import stats
from scipy.optimize import curve_fit
from ..core.optimized_analysis import adaptive_smooth, robust_baseline

def set_style():
    """Set publication-quality plot style."""
    if sns is not None:
        sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.linewidth': 1.0,
        'axes.labelpad': 10,
        'figure.subplot.top': 0.9,
        'figure.subplot.bottom': 0.1,
        'figure.subplot.left': 0.1,
        'figure.subplot.right': 0.9,
        'figure.subplot.hspace': 0.4,
        'figure.subplot.wspace': 0.3
    })

def plot_spectral_analysis(wavelengths: np.ndarray,
                          reference: np.ndarray,
                          sample: np.ndarray,
                          shifts: list,
                          config: dict,
                          title: str,
                          out_path: str):
    # Close any existing figures
    plt.close('all')
    """Create publication-quality spectral analysis plot."""
    set_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1], width_ratios=[1, 0.8])
    
    # 1. Full spectra with preprocessing
    ax1 = fig.add_subplot(gs[0, :])
    
    # Process and plot spectra
    try:
        # Process spectra
        ref_smooth = adaptive_smooth(wavelengths, reference, config['name'])
        sam_smooth = adaptive_smooth(wavelengths, sample, config['name'])
        
        ref_base = robust_baseline(wavelengths, ref_smooth, config['name'])
        sam_base = robust_baseline(wavelengths, sam_smooth, config['name'])
        
        ref_proc = ref_smooth - ref_base
        sam_proc = sam_smooth - sam_base
        
        ref_norm = ref_proc / np.max(np.abs(ref_proc))
        sam_norm = sam_proc / np.max(np.abs(sam_proc))
        
        # Plot raw and processed
        ax1.plot(wavelengths, reference, 'k-', alpha=0.3, label='Reference (raw)')
        ax1.plot(wavelengths, sample, 'r-', alpha=0.3, label='Sample (raw)')
        ax1.plot(wavelengths, ref_norm, 'k--', label='Reference (proc)')
        ax1.plot(wavelengths, sam_norm, 'r--', label='Sample (proc)')
        
        # Store processed data for peak analysis
        processed = {}
        processed.update({
            'ref_norm': ref_norm,
            'sam_norm': sam_norm,
            'ref_smooth': ref_smooth,
            'sam_smooth': sam_smooth,
            'ref_base': ref_base,
            'sam_base': sam_base
        })
    except Exception as e:
        print(f"Warning: Processing failed - {e}")
        ax1.plot(wavelengths, reference, 'k-', label='Reference')
        ax1.plot(wavelengths, sample, 'r-', label='Sample')
        processed = None
    
    # Highlight ROI
    roi = config['region']
    ax1.axvspan(roi[0], roi[1], color=config['color'], alpha=0.1,
                label=f'{config["name"]} ROI')
    
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Normalized Intensity')
    ax1.set_title('Full Spectral Range')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # 2. Peak region with detailed analysis
    ax2 = fig.add_subplot(gs[1, 0])
    window = 50
    peak_wl = config['peak']
    peak_idx = np.argmin(np.abs(wavelengths - peak_wl))
    start_idx = max(0, peak_idx - window)
    end_idx = min(len(wavelengths), peak_idx + window)
    
    # Process and plot peak region
    try:
        # Process spectra
        ref_smooth = adaptive_smooth(wavelengths, reference, config['name'])
        sam_smooth = adaptive_smooth(wavelengths, sample, config['name'])
        
        ref_base = robust_baseline(wavelengths, ref_smooth, config['name'])
        sam_base = robust_baseline(wavelengths, sam_smooth, config['name'])
        
        ref_proc = ref_smooth - ref_base
        sam_proc = sam_smooth - sam_base
        
        ref_norm = ref_proc / np.max(np.abs(ref_proc))
        sam_norm = sam_proc / np.max(np.abs(sam_proc))
        
        # Plot processed data
        ax2.plot(wavelengths[start_idx:end_idx],
                 ref_norm[start_idx:end_idx],
                 'k-', label='Reference')
        ax2.plot(wavelengths[start_idx:end_idx],
                 sam_norm[start_idx:end_idx],
                 'r-', label='Sample')
    except Exception as e:
        print(f"Warning: Processing failed - {e}")
        # Plot raw data if processing failed
        ax2.plot(wavelengths[start_idx:end_idx],
                 reference[start_idx:end_idx],
                 'k-', label='Reference')
        ax2.plot(wavelengths[start_idx:end_idx],
                 sample[start_idx:end_idx],
                 'r-', label='Sample')
    
    # Mark shifts with confidence-based coloring
    for shift in shifts:
        # Draw arrow
        ax2.annotate(
            f"{shift.shift:.2f} nm",
            xy=(shift.wavelength + shift.shift, 0),
            xytext=(shift.wavelength, 0),
            arrowprops=dict(
                arrowstyle='->',
                color=config['color'],
                alpha=shift.confidence,
                shrinkA=0,
                shrinkB=0
            ),
            ha='center', va='bottom'
        )
    
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Normalized Intensity')
    ax2.set_title('Peak Region Analysis')
    ax2.legend(loc='upper right')
    
    # 3. SNR and baseline analysis
    ax3 = fig.add_subplot(gs[1, 1])
    if shifts and processed is not None:
        shift = shifts[0]  # Best shift
        window_size = 5
        noise_windows = []
        
        try:
            # Calculate noise in peak region
            peak_data = processed['ref_norm'][start_idx:end_idx]
            for i in range(len(peak_data) - window_size + 1):
                # Detrend window
                window = peak_data[i:i+window_size]
                slope, intercept = np.polyfit(range(window_size), window, 1)
                detrended = window - (slope * np.arange(window_size) + intercept)
                noise_windows.append(np.std(detrended))
            
            noise_median = np.median(noise_windows)
            noise_mean = np.mean(noise_windows)
            noise_std = np.std(noise_windows)
            
            # Plot noise analysis
            x_noise = np.arange(len(noise_windows))
            ax3.plot(x_noise, noise_windows, 'k-', alpha=0.5, label='Local Noise')
            ax3.axhline(y=noise_median, color='r', linestyle='--',
                       label=f'Median: {noise_median:.3f}')
            ax3.axhline(y=noise_mean, color='g', linestyle=':',
                       label=f'Mean: {noise_mean:.3f}')
            ax3.fill_between(x_noise,
                            noise_mean - noise_std,
                            noise_mean + noise_std,
                            color='gray', alpha=0.2,
                            label=f'±σ: {noise_std:.3f}')
            
            ax3.set_xlabel('Window Index')
            ax3.set_ylabel('Local Noise (σ)')
            ax3.set_title('Noise Analysis')
            ax3.legend(loc='upper right')
            
            # Store noise statistics
            processed.update({
                'noise_windows': noise_windows,
                'noise_median': noise_median,
                'noise_mean': noise_mean,
                'noise_std': noise_std
            })
        except Exception as e:
            print(f"Warning: Noise analysis failed - {e}")
            ax3.text(0.5, 0.5, 'Noise analysis failed',
                     ha='center', va='center',
                     transform=ax3.transAxes)
    
    # 4. Peak characteristics
    ax4 = fig.add_subplot(gs[2, 0])
    if shifts:
        shift = shifts[0]
        # Fit and plot Gaussians
        def gaussian(x, amp, cen, wid):
            return amp * np.exp(-((x - cen) / wid)**2)
            
        x_fit = np.linspace(wavelengths[start_idx], wavelengths[end_idx], 100)
        try:
            # Fit reference peak
            ref_data = processed['ref_norm'][start_idx:end_idx]
            popt_ref, _ = curve_fit(gaussian, wavelengths[start_idx:end_idx], ref_data,
                                   p0=[np.max(ref_data), wavelengths[peak_idx], 5])
            y_ref = gaussian(x_fit, *popt_ref)
            
            # Fit sample peak
            sam_data = processed['sam_norm'][start_idx:end_idx]
            popt_sam, _ = curve_fit(gaussian, wavelengths[start_idx:end_idx], sam_data,
                                   p0=[np.max(sam_data), wavelengths[peak_idx], 5])
            y_sam = gaussian(x_fit, *popt_sam)
            
            # Store fit parameters
            processed.update({
                'ref_fit_params': popt_ref,
                'sam_fit_params': popt_sam
            })
        except Exception as e:
            print(f"Warning: Gaussian fitting failed - {e}")
            y_ref = np.zeros_like(x_fit)
            y_sam = np.zeros_like(x_fit)
        
        ax4.plot(wavelengths[start_idx:end_idx],
                processed['ref_norm'][start_idx:end_idx],
                'ko', alpha=0.5, label='Reference Data')
        ax4.plot(x_fit, y_ref, 'k-', label='Reference Fit')
        ax4.plot(wavelengths[start_idx:end_idx],
                processed['sam_norm'][start_idx:end_idx],
                'ro', alpha=0.5, label='Sample Data')
        ax4.plot(x_fit, y_sam, 'r-', label='Sample Fit')
        
        ax4.set_xlabel('Wavelength (nm)')
        ax4.set_ylabel('Normalized Intensity')
        ax4.set_title('Peak Fitting')
        ax4.legend()
    
    # 5. Quality metrics
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    if shifts:
        shift = shifts[0]
        metrics = (
            f"Peak Analysis Metrics:\\n\\n"
            f"Reference Peak ({shift.peak_ref.wavelength:.1f} nm):\\n"
            f"SNR: {shift.peak_ref.snr:.1f}\\n"
            f"FWHM: {shift.peak_ref.width:.1f} nm\\n"
            f"Asymmetry: {shift.peak_ref.asymmetry:.2f}\\n"
            f"Quality: {shift.peak_ref.quality:.3f}\\n\\n"
            f"Sample Peak ({shift.peak_sam.wavelength:.1f} nm):\\n"
            f"SNR: {shift.peak_sam.snr:.1f}\\n"
            f"FWHM: {shift.peak_sam.width:.1f} nm\\n"
            f"Asymmetry: {shift.peak_sam.asymmetry:.2f}\\n"
            f"Quality: {shift.peak_sam.quality:.3f}\\n\\n"
            f"Shift Analysis:\\n"
            f"Magnitude: {shift.shift:.3f} ± {processed.get('shift_error', 0):.3f} nm\\n"
            f"Confidence: {shift.confidence:.3f}"
        )
        ax5.text(0.05, 0.95, metrics,
                va='top', ha='left',
                transform=ax5.transAxes,
                fontfamily='monospace')
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration(results: dict,
                    config: dict,
                    title: str,
                    out_path: str):
    # Close any existing figures
    plt.close('all')
    """Create publication-quality calibration curve plot."""
    set_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[1.2, 0.8])
    
    # Extract data
    # Extract data from shifts list
    shifts = results.get('shifts', [])
    if not shifts:
        print("Warning: No shift data available")
        return
        
    try:
        x = np.array([s['concentration'] for s in shifts])
        y = np.array([s['shift'] for s in shifts])
        conf = np.array([s['confidence'] for s in shifts])
    except (KeyError, TypeError) as e:
        print(f"Warning: Invalid shift data format - {e}")
        return
    
    # Calculate errors if not provided
    if 'shift_errors' in results:
        yerr = np.array(results['shift_errors'])
    else:
        # Use confidence-based errors
        yerr = np.abs(y) * (1 - conf)
    
    # 1. Main calibration plot with error bars
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot data points with error bars and color by confidence
    if len(conf) > 0:
        colors = plt.cm.viridis(conf/np.max(conf))
        for i, (xi, yi, yerri, color) in enumerate(zip(x, y, yerr, colors)):
            ax1.errorbar(xi, yi, yerr=yerri, fmt='o',
                        color=color, capsize=3, capthick=1,
                        markersize=8, alpha=0.7)
    else:
        ax1.text(0.5, 0.5, 'No valid data points',
                 ha='center', va='center',
                 transform=ax1.transAxes)
    
    if not np.isnan(results['sensitivity']):
        # Generate fit line with confidence bands
        x_fit = np.linspace(0, max(x) * 1.1, 100)
        
        if results['model'] == 'linear':
            y_fit = results['parameters'][0] * x_fit + results['parameters'][1]
            model_label = 'Linear'
            equation = f'y = {results["parameters"][0]:.3f}x + {results["parameters"][1]:.3f}'
        elif results['model'] == 'quadratic':
            y_fit = (results['parameters'][0] * x_fit**2 +
                    results['parameters'][1] * x_fit +
                    results['parameters'][2])
            model_label = 'Quadratic'
            equation = f'y = {results["parameters"][0]:.3f}x² + {results["parameters"][1]:.3f}x + {results["parameters"][2]:.3f}'
        else:  # langmuir
            y_fit = (results['parameters'][0] * x_fit /
                    (results['parameters'][1] + x_fit))
            model_label = 'Langmuir'
            equation = f'y = {results["parameters"][0]:.3f}x / ({results["parameters"][1]:.3f} + x)'
        
        # Calculate confidence bands
        y_std = np.zeros_like(x_fit)
        for i, x_i in enumerate(x_fit):
            if results['model'] == 'linear':
                X = np.array([x_i, 1.0])
                y_std[i] = np.sqrt(X @ results['covariance'] @ X)
            else:
                # Monte Carlo for non-linear confidence bands
                y_samples = []
                for _ in range(1000):
                    params = np.random.multivariate_normal(
                        results['parameters'],
                        results['covariance']
                    )
                    if results['model'] == 'quadratic':
                        y_samples.append(params[0] * x_i**2 + params[1] * x_i + params[2])
                    else:  # langmuir
                        y_samples.append(params[0] * x_i / (params[1] + x_i))
                y_std[i] = np.std(y_samples)
        
        # Plot fit and confidence bands
        ax1.plot(x_fit, y_fit, '--',
                color=config['color'],
                label=f'{model_label} fit')
        ax1.fill_between(x_fit, y_fit - 2*y_std, y_fit + 2*y_std,
                        color=config['color'], alpha=0.2,
                        label='95% Confidence')
        
        # Add expected response
        y_expected = config['expected_shift'] * x_fit
        ax1.plot(x_fit, y_expected, ':',
                color='gray',
                label='Expected')
        
        # Add equation
        ax1.text(0.05, 0.95, equation,
                transform=ax1.transAxes,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.8),
                fontfamily='monospace')
    
    ax1.set_xlabel('Concentration (ppm)')
    ax1.set_ylabel('Wavelength Shift (nm)')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    
    # 2. Residuals plot
    ax2 = fig.add_subplot(gs[1, 0])
    if not np.isnan(results['sensitivity']):
        if results['model'] == 'linear':
            y_pred = results['parameters'][0] * x + results['parameters'][1]
        elif results['model'] == 'quadratic':
            y_pred = (results['parameters'][0] * x**2 +
                     results['parameters'][1] * x +
                     results['parameters'][2])
        else:  # langmuir
            y_pred = (results['parameters'][0] * x /
                     (results['parameters'][1] + x))
        
        residuals = y - y_pred
        colors = plt.cm.viridis(conf/np.max(conf))
        for i, (xi, yi, yerri, color) in enumerate(zip(x, residuals, yerr, colors)):
            ax2.errorbar(xi, yi, yerr=yerri,
                        fmt='o', color=color,
                        capsize=3, capthick=1,
                        markersize=8, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Add residual statistics
        stats_text = (
            f"Residual Statistics:\\n"
            f"Mean: {np.mean(residuals):.3f} nm\\n"
            f"Std: {np.std(residuals):.3f} nm\\n"
            f"RMSE: {results['rmse']:.3f} nm"
        )
        ax2.text(0.95, 0.95, stats_text,
                transform=ax2.transAxes,
                va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.8),
                fontfamily='monospace')
        
        ax2.set_xlabel('Concentration (ppm)')
        ax2.set_ylabel('Residuals (nm)')
        ax2.set_title('Residuals Analysis')
    
    # 3. Performance metrics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    metrics = (
        f"Model Performance:\n\n"
        f"Model: {results.get('model', 'N/A')}\n"
        f"Sensitivity: {results.get('sensitivity', np.nan):.3f} nm/ppm\n"
        f"Uncertainty: ±{results.get('sensitivity_err', 0):.3f} nm/ppm\n"
        f"R²: {results.get('r_squared', np.nan):.3f}\n"
        f"RMSE: {results.get('rmse', np.nan):.3f} nm\n\n"
        f"Detection Limits:\n"
        f"LOD: {results.get('lod', np.nan):.2f} ppm\n"
        f"LOQ: {results.get('loq', np.nan):.2f} ppm\n\n"
        f"Data Quality:\n"
        f"Points used: {results.get('points_used', 0)}\n"
        f"Outliers removed: {results.get('outliers_removed', 0)}\n"
        f"Mean confidence: {np.mean(conf) if len(conf) > 0 else np.nan:.3f}"
    )
    
    ax3.text(0.05, 0.95, metrics,
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8),
            fontfamily='monospace')
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_sensitivity(results: Dict[str, dict],
                         config: dict,
                         title: str,
                         out_path: str):
    # Close any existing figures
    plt.close('all')
    """Create cross-sensitivity analysis plot."""
    set_style()
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, height_ratios=[1, 1.5], width_ratios=[1, 1])
    
    # 1. Sensitivity comparison
    ax1 = fig.add_subplot(gs[0, 0])
    gases = list(results.keys())
    sensitivities = [results[gas]['sensitivity'] for gas in gases]
    errors = [results[gas]['sensitivity_err'] for gas in gases]
    
    bars = ax1.bar(gases, sensitivities, yerr=errors,
                  capsize=5, alpha=0.7)
    
    # Color bars by gas
    for gas, bar in zip(gases, bars):
        bar.set_color(config[gas]['color'])
    
    ax1.set_xlabel('Gas')
    ax1.set_ylabel('Sensitivity (nm/ppm)')
    ax1.set_title('Sensitivity Comparison')
    
    # 2. LOD comparison
    ax2 = fig.add_subplot(gs[0, 1])
    lods = [results[gas]['lod'] for gas in gases]
    lod_errors = [results[gas]['lod_err'] for gas in gases]
    
    bars = ax2.bar(gases, lods, yerr=lod_errors,
                  capsize=5, alpha=0.7)
    
    for gas, bar in zip(gases, bars):
        bar.set_color(config[gas]['color'])
    
    ax2.set_xlabel('Gas')
    ax2.set_ylabel('LOD (ppm)')
    ax2.set_title('Limit of Detection Comparison')
    
    # 3. Cross-correlation matrix
    ax3 = fig.add_subplot(gs[1, :])
    n_gases = len(gases)
    correlation_matrix = np.zeros((n_gases, n_gases))
    
    for i, gas1 in enumerate(gases):
        for j, gas2 in enumerate(gases):
            # Extract data from shifts list
            shifts1 = results[gas1].get('shifts', [])
            shifts2 = results[gas2].get('shifts', [])
            
            if not shifts1 or not shifts2:
                print("Warning: No shift data available")
                return
            
            try:
                x1 = np.array([s['concentration'] for s in shifts1])
                y1 = np.array([s['shift'] for s in shifts1])
                x2 = np.array([s['concentration'] for s in shifts2])
                y2 = np.array([s['shift'] for s in shifts2])
            except (KeyError, TypeError) as e:
                print(f"Warning: Invalid shift data format - {e}")
                return
            
            # Calculate correlation between responses
            # Interpolate to common concentration points
            common_x = np.linspace(max(min(x1), min(x2)),
                                 min(max(x1), max(x2)), 100)
            y1_interp = np.interp(common_x, x1, y1)
            y2_interp = np.interp(common_x, x2, y2)
            
            correlation_matrix[i, j] = np.corrcoef(y1_interp, y2_interp)[0, 1]
    
    # Plot correlation matrix
    im = ax3.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax3, label='Response Correlation')
    
    # Add correlation values
    for i in range(n_gases):
        for j in range(n_gases):
            text = f"{correlation_matrix[i, j]:.2f}"
            ax3.text(j, i, text, ha='center', va='center')
    
    ax3.set_xticks(range(n_gases))
    ax3.set_yticks(range(n_gases))
    ax3.set_xticklabels(gases)
    ax3.set_yticklabels(gases)
    ax3.set_title('Cross-Sensitivity Analysis')
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_stability_analysis(time_series: dict,
                          config: dict,
                          title: str,
                          out_path: str):
    # Close any existing figures
    plt.close('all')
    """Create stability and drift analysis plot."""
    set_style()
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1.2, 0.8])
    
    # 1. Time series with drift
    ax1 = fig.add_subplot(gs[0, :])
    t = np.array(time_series['time'])  # minutes
    baseline = np.array(time_series['baseline'])
    drift = np.array(time_series['drift'])
    noise = np.array(time_series['noise'])
    
    ax1.plot(t, baseline, 'k-', label='Baseline')
    ax1.plot(t, baseline + drift, 'r--', label='Drift')
    ax1.fill_between(t, baseline - noise, baseline + noise,
                    color='gray', alpha=0.2, label='Noise Band')
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Signal')
    ax1.set_title('Temporal Stability')
    ax1.legend()
    
    # 2. Allan deviation
    ax2 = fig.add_subplot(gs[1, 0])
    tau = np.array(time_series['tau'])
    adev = np.array(time_series['adev'])
    adev_err = np.array(time_series['adev_err'])
    
    ax2.errorbar(tau, adev, yerr=adev_err, fmt='ko-',
                capsize=3, label='Measured')
    
    # Fit theoretical slopes
    tau_fit = np.logspace(np.log10(min(tau)), np.log10(max(tau)), 100)
    white_noise = adev[0] * (tau_fit[0]/tau_fit)**0.5
    random_walk = adev[-1] * (tau_fit/tau_fit[-1])**0.5
    
    ax2.loglog(tau_fit, white_noise, 'r--',
               label='White Noise (τ⁻¹/²)')
    ax2.loglog(tau_fit, random_walk, 'b--',
               label='Random Walk (τ¹/²)')
    
    ax2.set_xlabel('Averaging Time τ (minutes)')
    ax2.set_ylabel('Allan Deviation σ(τ)')
    ax2.set_title('Allan Deviation Analysis')
    ax2.grid(True, which="both")
    ax2.legend()
    
    # 3. Statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    
    stats = (
        f"Stability Metrics:\\n\\n"
        f"Short-term (1 min):\\n"
        f"Noise: {time_series['noise_short']:.3f} nm\\n"
        f"Drift: {time_series['drift_short']:.3f} nm/min\\n\\n"
        f"Long-term (60 min):\\n"
        f"Noise: {time_series['noise_long']:.3f} nm\\n"
        f"Drift: {time_series['drift_long']:.3f} nm/min\\n\\n"
        f"Optimal averaging time:\\n"
        f"τ = {time_series['tau_opt']:.1f} min\\n"
        f"σ(τ) = {time_series['adev_opt']:.3f} nm"
    )
    
    ax3.text(0.05, 0.95, stats,
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.8),
            fontfamily='monospace')
    
    plt.suptitle(title, fontsize=14, y=0.95)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
