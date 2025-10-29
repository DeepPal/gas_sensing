"""Advanced statistical analysis and visualization for research."""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SensorMetrics:
    """Comprehensive sensor performance metrics."""
    sensitivity: float          # nm/ppm
    r_squared: float           # Linearity
    lod: float                 # Limit of detection
    loq: float                # Limit of quantification
    response_time: float      # T90 (seconds)
    recovery_time: float      # T10 (seconds)
    drift_rate: float         # nm/hour
    repeatability: float      # RSD of repeated measurements
    cross_sensitivity: Dict[str, float]  # Response to other gases
    confidence_intervals: Dict[str, Tuple[float, float]]  # 95% CIs

def analyze_sensor_performance(wavelengths: np.ndarray,
                             intensities: np.ndarray,
                             concentrations: np.ndarray,
                             timestamps: Optional[np.ndarray] = None,
                             reference: Optional[np.ndarray] = None,
                             debug: bool = True) -> SensorMetrics:
    """Compute comprehensive sensor performance metrics.
    
    Args:
        wavelengths: Wavelength array
        intensities: List of intensity arrays for each measurement
        concentrations: Concentration values
        timestamps: Optional measurement timestamps
        reference: Optional reference spectrum
    
    Returns:
        SensorMetrics object
    """
    # Convert to transmittance if reference provided
    if reference is not None:
        # Interpolate reference to match wavelength grid
        ref_int = np.interp(wavelengths, wavelengths, reference)
        # Compute transmittance (spectra x wavelengths) / wavelengths
        transmittances = intensities / ref_int[None, :]
        transmittances = np.clip(transmittances, 0, 1)
        y_data = transmittances
        
        if debug:
            print(f"Reference range: {reference.min():.3f} - {reference.max():.3f}")
            print(f"Transmittance range: {transmittances.min():.3f} - {transmittances.max():.3f}")
    else:
        y_data = intensities
    
    if debug:
        print(f"\nSignal analysis:")
        print(f"Using {'transmittance' if reference is not None else 'intensity'} data")
        print(f"Data shape: {y_data.shape} (spectra x wavelengths)")
    
    # 1. Find resonance wavelengths
    peak_wavelengths = []
    
    # Average spectrum for peak finding
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)
    
    # Find main peak/dip
    peak_idx = np.argmax(np.abs(y_mean - np.median(y_mean)))
    window = len(wavelengths) // 20
    start = max(0, peak_idx - window)
    end = min(len(wavelengths), peak_idx + window)
    
    if debug:
        print(f"\nPeak analysis:")
        print(f"Main feature at {wavelengths[peak_idx]:.2f} nm")
        print(f"Analysis window: {wavelengths[start]:.2f} - {wavelengths[end]:.2f} nm")
    
    # Find peak position for each spectrum
    for y in y_data:
        # Use weighted centroid in peak region
        weights = np.abs(y[start:end] - np.median(y[start:end]))
        peak_wl = np.average(wavelengths[start:end], weights=weights)
        peak_wavelengths.append(peak_wl)
    
    # 2. Calibration Analysis
    peak_wavelengths = np.array(peak_wavelengths)
    
    # Linear regression
    slope, intercept, r_value, p_value, slope_stderr = stats.linregress(
        concentrations, peak_wavelengths
    )
    
    # R-squared
    r_squared = r_value ** 2
    
    # Residual analysis
    y_pred = slope * concentrations + intercept
    residuals = peak_wavelengths - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Confidence Intervals (95%)
    conf_level = 0.95
    degrees_of_freedom = len(concentrations) - 2
    t_value = stats.t.ppf((1 + conf_level) / 2, degrees_of_freedom)
    
    # Standard errors
    x_mean = np.mean(concentrations)
    x_std = np.std(concentrations)
    
    slope_se = slope_stderr
    intercept_se = rmse * np.sqrt(1/len(concentrations) + 
                                 x_mean**2/(x_std**2 * (len(concentrations)-1)))
    
    # Confidence intervals
    slope_ci = (slope - t_value*slope_se, slope + t_value*slope_se)
    intercept_ci = (intercept - t_value*intercept_se, 
                   intercept + t_value*intercept_se)
    
    # Prediction interval for new observations
    x_new = np.linspace(min(concentrations), max(concentrations), 100)
    y_new = slope * x_new + intercept
    
    pi = t_value * rmse * np.sqrt(1 + 1/len(concentrations) + 
                                 (x_new - x_mean)**2 / 
                                 (x_std**2 * (len(concentrations)-1)))
    
    prediction_intervals = (y_new - pi, y_new + pi)
    
    # 3. LOD and LOQ
    blank_std = rmse  # Using RMSE as estimate of blank standard deviation
    lod = 3.3 * blank_std / abs(slope)
    loq = 10.0 * blank_std / abs(slope)
    
    # 4. Response Time Analysis (if timestamps provided)
    response_time = recovery_time = float('nan')
    if timestamps is not None:
        # Normalize response
        response = (peak_wavelengths - min(peak_wavelengths)) / (max(peak_wavelengths) - min(peak_wavelengths))
        
        # Find response time (T90)
        t90_idx = np.where(response >= 0.9)[0]
        if len(t90_idx) > 0:
            response_time = timestamps[t90_idx[0]] - timestamps[0]
        
        # Find recovery time (T10)
        t10_idx = np.where(response <= 0.1)[0]
        if len(t10_idx) > 0:
            recovery_time = timestamps[t10_idx[-1]] - timestamps[0]
    
    # 5. Drift Analysis
    if timestamps is not None:
        # Linear fit to peak positions over time
        hours = (timestamps - timestamps[0]) / 3600
        drift_slope, _ = np.polyfit(hours, peak_wavelengths, 1)
        drift_rate = abs(drift_slope)  # nm/hour
    else:
        drift_rate = float('nan')
    
    # 6. Repeatability
    # Use RSD of measurements at same concentration
    unique_concs = np.unique(concentrations)
    rsds = []
    for conc in unique_concs:
        mask = concentrations == conc
        if np.sum(mask) > 1:
            rsd = np.std(peak_wavelengths[mask]) / np.mean(peak_wavelengths[mask])
            rsds.append(rsd)
    repeatability = np.mean(rsds) if rsds else float('nan')
    
    # 7. Package results
    confidence_intervals = {
        'slope': slope_ci,
        'intercept': intercept_ci,
        'prediction': prediction_intervals
    }
    
    return SensorMetrics(
        sensitivity=float(slope),
        r_squared=float(r_squared),
        lod=float(lod),
        loq=float(loq),
        response_time=float(response_time),
        recovery_time=float(recovery_time),
        drift_rate=float(drift_rate),
        repeatability=float(repeatability),
        cross_sensitivity={},  # Filled by separate analysis
        confidence_intervals=confidence_intervals
    )

def plot_research_figures(wavelengths: np.ndarray,
                         intensities: List[np.ndarray],
                         concentrations: np.ndarray,
                         timestamps: Optional[np.ndarray] = None,
                         metrics: Optional[SensorMetrics] = None,
                         out_dir: str = 'output/research'):
    """Generate publication-quality figures.
    
    Args:
        wavelengths: Wavelength array
        intensities: List of intensity arrays
        concentrations: Concentration values
        timestamps: Optional measurement timestamps
        metrics: Optional SensorMetrics object
        out_dir: Output directory for figures
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    # Set publication style
    # Use clean style
    plt.style.use('seaborn')
    
    # 1. Spectral Overview
    plt.figure(figsize=(10, 6))
    unique_concs = np.unique(concentrations)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_concs)))
    
    for conc, color in zip(unique_concs, colors):
        mask = concentrations == conc
        mean_spectrum = np.mean([intensities[i] for i in np.where(mask)[0]], axis=0)
        std_spectrum = np.std([intensities[i] for i in np.where(mask)[0]], axis=0)
        
        plt.plot(wavelengths, mean_spectrum, color=color, 
                label=f'{conc:.1f} ppm', alpha=0.8)
        plt.fill_between(wavelengths, 
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        color=color, alpha=0.2)
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Spectral Response vs. Concentration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spectral_overview.png'), dpi=300)
    plt.close()
    
    # 2. Calibration Curve with Confidence Intervals
    if metrics is not None:
        plt.figure(figsize=(8, 6))
        
        # Data points
        unique_concs = np.unique(concentrations)
        peak_wavelengths = []
        peak_stds = []
        
        for conc in unique_concs:
            mask = concentrations == conc
            peaks = []
            for i in np.where(mask)[0]:
                peak_idx = np.argmax(np.abs(intensities[i] - np.median(intensities[i])))
                peaks.append(wavelengths[peak_idx])
            peak_wavelengths.append(np.mean(peaks))
            peak_stds.append(np.std(peaks))
        
        plt.errorbar(unique_concs, peak_wavelengths, yerr=peak_stds,
                    fmt='o', color='blue', label='Data', capsize=5)
        
        # Fit line
        x_fit = np.linspace(min(concentrations), max(concentrations), 100)
        y_fit = metrics.sensitivity * x_fit + metrics.confidence_intervals['intercept'][0]
        
        plt.plot(x_fit, y_fit, 'r-', label=f'Fit (R² = {metrics.r_squared:.3f})')
        
        # Prediction intervals
        y_lower, y_upper = metrics.confidence_intervals['prediction']
        plt.fill_between(x_fit, y_lower, y_upper, color='red', alpha=0.2,
                        label='95% Prediction Interval')
        
        plt.xlabel('Concentration (ppm)')
        plt.ylabel('Peak Wavelength (nm)')
        plt.title('Sensor Calibration')
        
        # Add metrics annotation
        text = (f'Sensitivity: {metrics.sensitivity:.2f} nm/ppm\n'
               f'LOD: {metrics.lod:.2f} ppm\n'
               f'LOQ: {metrics.loq:.2f} ppm')
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'calibration.png'), dpi=300)
        plt.close()
    
    # 3. Response Dynamics (if timestamps available)
    if timestamps is not None:
        plt.figure(figsize=(10, 6))
        
        # Find peak wavelengths over time
        peak_wavelengths = []
        for intensity in intensities:
            peak_idx = np.argmax(np.abs(intensity - np.median(intensity)))
            peak_wavelengths.append(wavelengths[peak_idx])
        
        # Normalize response
        response = (peak_wavelengths - min(peak_wavelengths)) / (max(peak_wavelengths) - min(peak_wavelengths))
        
        # Plot response curve
        plt.plot((timestamps - timestamps[0])/60, response, 'b-', label='Response')
        
        if metrics is not None and not np.isnan(metrics.response_time):
            # Add T90 and T10 lines
            plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=0.1, color='r', linestyle='--', alpha=0.5)
            
            # Add time annotations
            plt.text(metrics.response_time/60, 0.9, f'T90 = {metrics.response_time:.1f}s',
                    verticalalignment='bottom')
            plt.text(metrics.recovery_time/60, 0.1, f'T10 = {metrics.recovery_time:.1f}s',
                    verticalalignment='top')
        
        plt.xlabel('Time (min)')
        plt.ylabel('Normalized Response')
        plt.title('Sensor Response Dynamics')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'dynamics.png'), dpi=300)
        plt.close()
    
    # 4. Repeatability Analysis
    unique_concs = np.unique(concentrations)
    if len(unique_concs) > 0:
        plt.figure(figsize=(8, 6))
        
        data = []
        labels = []
        
        for conc in unique_concs:
            mask = concentrations == conc
            peaks = []
            for i in np.where(mask)[0]:
                peak_idx = np.argmax(np.abs(intensities[i] - np.median(intensities[i])))
                peaks.append(wavelengths[peak_idx])
            data.append(peaks)
            labels.extend([f'{conc:.1f} ppm'] * len(peaks))
        
        # Box plot
        plt.boxplot(data, labels=[f'{c:.1f}' for c in unique_concs])
        
        # Add individual points
        for i, d in enumerate(data):
            x = np.random.normal(i+1, 0.04, size=len(d))
            plt.plot(x, d, 'o', alpha=0.5, color='blue')
        
        plt.xlabel('Concentration (ppm)')
        plt.ylabel('Peak Wavelength (nm)')
        plt.title('Measurement Repeatability')
        
        if metrics is not None:
            plt.text(0.05, 0.95, f'RSD = {metrics.repeatability:.3%}',
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'repeatability.png'), dpi=300)
        plt.close()
    
    # 5. Residual Analysis
    if metrics is not None:
        plt.figure(figsize=(12, 4))
        
        # Calculate residuals
        y_pred = metrics.sensitivity * concentrations + metrics.confidence_intervals['intercept'][0]
        peak_wavelengths = []
        for intensity in intensities:
            peak_idx = np.argmax(np.abs(intensity - np.median(intensity)))
            peak_wavelengths.append(wavelengths[peak_idx])
        residuals = np.array(peak_wavelengths) - y_pred
        
        # Subplot 1: Residuals vs Fitted
        plt.subplot(121)
        plt.scatter(y_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Fitted Values (nm)')
        plt.ylabel('Residuals (nm)')
        plt.title('Residuals vs Fitted')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Q-Q plot
        plt.subplot(122)
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Normal Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'residuals.png'), dpi=300)
        plt.close()

def generate_latex_table(metrics: Dict[str, SensorMetrics], 
                        output_path: str):
    """Generate LaTeX table with performance metrics."""
    
    template = r"""
\begin{table}[htbp]
\centering
\caption{Sensor Performance Metrics}
\begin{tabular}{lccccc}
\hline
Gas & Sensitivity & R$^2$ & LOD & Response Time & Repeatability \\
 & (nm/ppm) & & (ppm) & (s) & (\%) \\
\hline
%s
\hline
\end{tabular}
\label{tab:performance}
\end{table}
"""
    
    rows = []
    for gas, m in metrics.items():
        row = (f"{gas} & {m.sensitivity:.2f} & {m.r_squared:.3f} & "
               f"{m.lod:.2f} & {m.response_time:.1f} & {m.repeatability*100:.1f} \\\\")
        rows.append(row)
    
    table = template % '\n'.join(rows)
    
    with open(output_path, 'w') as f:
        f.write(table)
