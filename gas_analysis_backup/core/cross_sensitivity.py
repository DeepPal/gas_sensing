"""Cross-sensitivity analysis for multi-gas sensing."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CrossSensitivityResult:
    """Results from cross-sensitivity analysis."""
    primary_gas: str
    interfering_gas: str
    sensitivity_ratio: float      # Ratio of sensitivities
    interference_factor: float    # % response at equal concentration
    selectivity: float           # Ratio of LODs
    correlation: float           # Response correlation coefficient
    p_value: float              # Statistical significance

def calculate_cross_sensitivity(primary_results: Dict,
                              interfering_results: Dict) -> CrossSensitivityResult:
    """Calculate cross-sensitivity metrics between two gases."""
    # Extract data from shifts list
    shifts1 = primary_results.get('shifts', [])
    shifts2 = interfering_results.get('shifts', [])
    
    if not shifts1 or not shifts2:
        print("Warning: No shift data available")
        return None
        
    try:
        x1 = np.array([s['concentration'] for s in shifts1])
        y1 = np.array([s['shift'] for s in shifts1])
        x2 = np.array([s['concentration'] for s in shifts2])
        y2 = np.array([s['shift'] for s in shifts2])
    except (KeyError, TypeError) as e:
        print(f"Warning: Invalid shift data format - {e}")
        return None
    
    # Interpolate to common concentration points
    common_x = np.linspace(max(min(x1), min(x2)),
                          min(max(x1), max(x2)), 100)
    y1_interp = np.interp(common_x, x1, y1)
    y2_interp = np.interp(common_x, x2, y2)
    
    # Calculate sensitivity ratio
    sensitivity_ratio = abs(
        interfering_results['sensitivity'] /
        primary_results['sensitivity']
    )
    
    # Calculate interference factor at equal concentration
    interference_factor = abs(np.mean(y2_interp / y1_interp)) * 100
    
    # Calculate selectivity (ratio of LODs)
    selectivity = interfering_results['lod'] / primary_results['lod']
    
    # Calculate response correlation
    correlation, p_value = stats.pearsonr(y1_interp, y2_interp)
    
    return CrossSensitivityResult(
        primary_gas=primary_results.get('gas', 'primary'),
        interfering_gas=interfering_results.get('gas', 'interfering'),
        sensitivity_ratio=float(sensitivity_ratio),
        interference_factor=float(interference_factor),
        selectivity=float(selectivity),
        correlation=float(correlation),
        p_value=float(p_value)
    )

def analyze_cross_sensitivities(results: Dict[str, Dict]) -> Dict[str, Dict[str, CrossSensitivityResult]]:
    """Analyze cross-sensitivities between all gas pairs."""
    gases = list(results.keys())
    cross_sensitivities = {}
    
    for primary in gases:
        cross_sensitivities[primary] = {}
        for interfering in gases:
            if primary != interfering:
                cross_sensitivities[primary][interfering] = calculate_cross_sensitivity(
                    results[primary], results[interfering]
                )
    
    return cross_sensitivities

def calculate_selectivity_coefficients(cross_sensitivities: Dict[str, Dict[str, CrossSensitivityResult]]) -> Dict[str, float]:
    """Calculate overall selectivity coefficients for each gas."""
    selectivity_coeffs = {}
    
    for primary in cross_sensitivities:
        # Average of inverse interference factors
        interfering_gases = cross_sensitivities[primary]
        selectivity = np.mean([
            100 / result.interference_factor
            for result in interfering_gases.values()
        ])
        selectivity_coeffs[primary] = float(selectivity)
    
    return selectivity_coeffs

def analyze_temperature_effects(results: Dict[str, Dict],
                             temperatures: np.ndarray) -> Dict[str, Dict]:
    """Analyze temperature effects on sensor response."""
    temp_effects = {}
    
    for gas, gas_results in results.items():
        # Extract temperature-dependent responses
        concentrations = np.array(gas_results['concentrations'])
        shifts = np.array(gas_results['shifts'])
        
        # Calculate temperature coefficients
        temp_coeff = np.polyfit(temperatures, shifts/concentrations, 1)[0]
        
        # Calculate temperature-compensated responses
        shifts_comp = shifts - temp_coeff * (temperatures - np.mean(temperatures))[:, None]
        
        # Analyze improvement
        original_rmse = gas_results['rmse']
        comp_rmse = np.std(shifts_comp - np.mean(shifts_comp))
        
        temp_effects[gas] = {
            'temperature_coefficient': float(temp_coeff),  # nm/ppm/°C
            'mean_temperature': float(np.mean(temperatures)),
            'temperature_range': float(np.ptp(temperatures)),
            'original_rmse': float(original_rmse),
            'compensated_rmse': float(comp_rmse),
            'improvement': float((original_rmse - comp_rmse) / original_rmse * 100)
        }
    
    return temp_effects

def analyze_humidity_effects(results: Dict[str, Dict],
                           humidity_levels: np.ndarray) -> Dict[str, Dict]:
    """Analyze humidity effects on sensor response."""
    humidity_effects = {}
    
    for gas, gas_results in results.items():
        # Extract humidity-dependent responses
        concentrations = np.array(gas_results['concentrations'])
        shifts = np.array(gas_results['shifts'])
        
        # Calculate humidity coefficients
        humidity_coeff = np.polyfit(humidity_levels, shifts/concentrations, 1)[0]
        
        # Calculate humidity-compensated responses
        shifts_comp = shifts - humidity_coeff * (humidity_levels - np.mean(humidity_levels))[:, None]
        
        # Analyze improvement
        original_rmse = gas_results['rmse']
        comp_rmse = np.std(shifts_comp - np.mean(shifts_comp))
        
        humidity_effects[gas] = {
            'humidity_coefficient': float(humidity_coeff),  # nm/ppm/%RH
            'mean_humidity': float(np.mean(humidity_levels)),
            'humidity_range': float(np.ptp(humidity_levels)),
            'original_rmse': float(original_rmse),
            'compensated_rmse': float(comp_rmse),
            'improvement': float((original_rmse - comp_rmse) / original_rmse * 100)
        }
    
    return humidity_effects
