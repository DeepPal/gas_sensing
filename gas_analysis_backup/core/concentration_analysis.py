"""Advanced concentration response analysis for gas sensors."""

import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CalibrationResult:
    """Calibration analysis result."""
    model: str                # Model type (linear, quadratic, langmuir)
    sensitivity: float        # Initial sensitivity (nm/ppm)
    sensitivity_err: float    # Sensitivity uncertainty
    r_squared: float         # R-squared value
    rmse: float             # Root mean square error (nm)
    lod: float             # Limit of detection (ppm)
    lod_err: float         # LOD uncertainty
    loq: float             # Limit of quantification (ppm)
    loq_err: float         # LOQ uncertainty
    points_used: int        # Number of points used in fit
    outliers_removed: int   # Number of outliers removed
    parameters: List[float] # Model parameters
    covariance: List[List[float]]  # Parameter covariance matrix

def group_by_concentration(concentrations: np.ndarray,
                         shifts: np.ndarray,
                         confidences: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group measurements by concentration and calculate weighted statistics."""
    unique_conc = np.unique(concentrations)
    mean_shifts = []
    std_shifts = []
    mean_conf = []
    
    for conc in unique_conc:
        mask = concentrations == conc
        shifts_at_conc = shifts[mask]
        conf_at_conc = confidences[mask]
        
        # Weighted mean and std using confidences
        weights = conf_at_conc / np.sum(conf_at_conc)
        mean_shift = np.average(shifts_at_conc, weights=weights)
        mean_shifts.append(mean_shift)
        
        # Weighted standard deviation
        variance = np.average((shifts_at_conc - mean_shift)**2, weights=weights)
        std_shifts.append(np.sqrt(variance))
        mean_conf.append(np.mean(conf_at_conc))
    
    return (np.array(unique_conc), np.array(mean_shifts), 
            np.array(std_shifts), np.array(mean_conf))

def remove_outliers(x: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    """Remove outliers using modified z-score."""
    median = np.median(x)
    mad = np.median(np.abs(x - median)) * 1.4826
    modified_z = np.abs(x - median) / mad if mad > 0 else np.zeros_like(x)
    return modified_z < threshold

def fit_models(x: np.ndarray, 
              y: np.ndarray, 
              yerr: np.ndarray,
              w: np.ndarray) -> Dict:
    """Fit different models and select best using AIC."""
    def linear(x, a, b):
        return a * x + b
    
    def quadratic(x, a, b, c):
        return a * x**2 + b * x + c
    
    def langmuir(x, vmax, k):
        return vmax * x / (k + x)
    
    models = [
        ('linear', linear, [1, 0]),
        ('quadratic', quadratic, [0.1, 1, 0]),
        ('langmuir', langmuir, [max(y), np.median(x)])
    ]
    
    best_fit = None
    best_aic = np.inf
    
    for name, func, p0 in models:
        try:
            # Weighted fit with uncertainties
            sigma = yerr / w  # Scale uncertainties by confidence
            popt, pcov = curve_fit(func, x, y, p0=p0, 
                                 sigma=sigma, absolute_sigma=True)
            
            # Calculate AIC with weighted residuals
            y_fit = func(x, *popt)
            n = len(x)
            k = len(p0)
            rss = np.sum(w * ((y - y_fit) / sigma)**2)
            aic = n * np.log(rss/n) + 2*k
            
            if aic < best_aic:
                best_aic = aic
                best_fit = {
                    'name': name,
                    'function': func,
                    'parameters': popt,
                    'covariance': pcov,
                    'y_fit': y_fit,
                    'aic': aic
                }
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
    
    return best_fit

def calculate_metrics(best_fit: Dict,
                     x: np.ndarray,
                     y: np.ndarray,
                     yerr: np.ndarray,
                     w: np.ndarray) -> Dict:
    """Calculate performance metrics with uncertainties."""
    # Basic metrics
    y_fit = best_fit['y_fit']
    residuals = y - y_fit
    ss_res = np.sum(w * residuals**2)
    ss_tot = np.sum(w * (y - np.average(y, weights=w))**2)
    r_squared = 1 - ss_res/ss_tot
    rmse = np.sqrt(np.average(residuals**2, weights=w))
    
    # Sensitivity and uncertainty
    if best_fit['name'] == 'linear':
        sensitivity = best_fit['parameters'][0]
        sensitivity_err = np.sqrt(best_fit['covariance'][0,0])
    else:
        # Monte Carlo for non-linear sensitivity
        n_samples = 1000
        sensitivities = []
        x_test = np.linspace(0, min(x), 100)
        
        for _ in range(n_samples):
            params = np.random.multivariate_normal(
                best_fit['parameters'],
                best_fit['covariance']
            )
            y_mc = best_fit['function'](x_test, *params)
            sens = (y_mc[1] - y_mc[0]) / (x_test[1] - x_test[0])
            sensitivities.append(sens)
        
        sensitivity = np.mean(sensitivities)
        sensitivity_err = np.std(sensitivities)
    
    # LOD and LOQ with uncertainty propagation
    noise_level = np.sqrt(np.average(residuals**2, weights=w))
    noise_err = np.std(residuals) / np.sqrt(len(residuals))
    
    # Relative error propagation
    rel_error = np.sqrt((sensitivity_err/sensitivity)**2 + 
                       (noise_err/noise_level)**2)
    
    lod = 3.3 * noise_level / abs(sensitivity)
    lod_err = lod * rel_error
    
    loq = 10.0 * noise_level / abs(sensitivity)
    loq_err = loq * rel_error
    
    return {
        'model': best_fit['name'],
        'sensitivity': float(sensitivity),
        'sensitivity_err': float(sensitivity_err),
        'r_squared': float(r_squared),
        'rmse': float(rmse),
        'lod': float(lod),
        'lod_err': float(lod_err),
        'loq': float(loq),
        'loq_err': float(loq_err),
        'parameters': [float(p) for p in best_fit['parameters']],
        'covariance': best_fit['covariance'].tolist()
    }

def analyze_concentration_response(shifts: List[Dict],
                                region: str = 'auto') -> Dict:
    """Analyze concentration response with advanced statistics."""
    # Extract data
    concentrations = np.array([s['concentration'] for s in shifts])
    peak_shifts = np.array([s['shift'] for s in shifts])
    confidences = np.array([s['confidence'] for s in shifts])
    
    # Group by concentration
    x, y, yerr, w = group_by_concentration(
        concentrations, peak_shifts, confidences)
    
    # Remove outliers
    valid = remove_outliers(y)
    x = x[valid]
    y = y[valid]
    yerr = yerr[valid]
    w = w[valid]
    
    if len(x) < 3:
        return {
            'sensitivity': np.nan,
            'sensitivity_err': np.nan,
            'r_squared': np.nan,
            'rmse': np.nan,
            'lod': np.nan,
            'lod_err': np.nan,
            'loq': np.nan,
            'loq_err': np.nan,
            'points_used': len(x),
            'outliers_removed': len(concentrations) - len(x)
        }
    
    # Fit models and select best
    best_fit = fit_models(x, y, yerr, w)
    
    if best_fit is None:
        return {
            'sensitivity': np.nan,
            'sensitivity_err': np.nan,
            'r_squared': np.nan,
            'rmse': np.nan,
            'lod': np.nan,
            'lod_err': np.nan,
            'loq': np.nan,
            'loq_err': np.nan,
            'points_used': len(x),
            'outliers_removed': len(concentrations) - len(x)
        }
    
    # Calculate metrics
    results = calculate_metrics(best_fit, x, y, yerr, w)
    results.update({
        'points_used': int(len(x)),
        'outliers_removed': int(len(concentrations) - len(x)),
        'concentrations': x.tolist(),
        'shifts': y.tolist(),
        'shift_errors': yerr.tolist(),
        'weights': w.tolist()
    })
    
    return results
