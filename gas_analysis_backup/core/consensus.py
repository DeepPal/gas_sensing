"""Weighted consensus analysis for multi-method peak detection."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ShiftMethod:
    """Information about a shift detection method."""
    name: str           # Method name
    shift: float       # Detected shift
    uncertainty: float # Shift uncertainty
    confidence: float  # Confidence score (0-1)
    weight: float     # Method weight (0-1)

def calculate_method_weights(methods: List[ShiftMethod]) -> np.ndarray:
    """Calculate weights for each method based on performance."""
    # Base weights from confidence scores
    weights = np.array([method.confidence for method in methods])
    
    # Adjust weights based on agreement
    shifts = np.array([method.shift for method in methods])
    uncertainties = np.array([method.uncertainty for method in methods])
    
    # Calculate weighted mean and its uncertainty
    wmean = np.average(shifts, weights=weights)
    
    # Calculate normalized residuals
    residuals = np.abs(shifts - wmean) / uncertainties
    agreement_scores = 1 / (1 + residuals)
    
    # Update weights
    weights *= agreement_scores
    
    # Normalize weights
    weights /= np.sum(weights)
    
    return weights

def detect_outliers(methods: List[ShiftMethod]) -> np.ndarray:
    """Detect outlier methods using modified z-score."""
    shifts = np.array([method.shift for method in methods])
    uncertainties = np.array([method.uncertainty for method in methods])
    
    # Calculate median and MAD of normalized shifts
    norm_shifts = shifts / uncertainties
    median_shift = np.median(norm_shifts)
    mad = stats.median_abs_deviation(norm_shifts)
    
    # Calculate modified z-scores
    modified_z = 0.6745 * np.abs(norm_shifts - median_shift) / mad
    
    # Identify outliers
    is_valid = modified_z < 3.5  # Standard threshold
    
    return is_valid

def calculate_consensus_shift(methods: List[ShiftMethod]) -> Tuple[float, float, float]:
    """Calculate consensus shift using weighted average."""
    # Extract shifts, uncertainties, and confidences
    shifts = np.array([method.shift for method in methods])
    uncertainties = np.array([method.uncertainty for method in methods])
    confidences = np.array([method.confidence for method in methods])
    weights = calculate_method_weights(methods)
    
    # Convert to arrays
    shifts = np.array(shifts)
    uncertainties = np.array(uncertainties)
    confidences = np.array(confidences)
    weights = np.array(weights)
    
    # Remove outliers using modified z-score
    median_shift = np.median(shifts)
    mad = stats.median_abs_deviation(shifts)
    if mad > 0:
        modified_z = 0.6745 * np.abs(shifts - median_shift) / mad
        valid_idx = modified_z < 3.5  # Threshold for outliers
    else:
        valid_idx = np.ones_like(shifts, dtype=bool)
    
    if not any(valid_idx):
        return 0.0, np.inf, 0.0
    
    # Update arrays to remove outliers
    shifts = shifts[valid_idx]
    uncertainties = uncertainties[valid_idx]
    confidences = confidences[valid_idx]
    weights = weights[valid_idx]
    
    # Calculate combined weights
    total_weights = weights * confidences
    if np.any(uncertainties > 0):
        total_weights /= uncertainties
    total_weights /= np.sum(total_weights)
    
    # Calculate weighted average
    consensus_shift = np.sum(shifts * total_weights)
    
    # Calculate uncertainty
    uncertainty = np.sqrt(np.sum((uncertainties * total_weights)**2))
    
    # Calculate overall confidence
    confidence = np.mean(confidences) / np.mean(uncertainties**2)
    sys_var = np.average((shifts - consensus_shift)**2, weights=weights)
    stat_var = 1 / np.sum(weights / uncertainties**2)
    total_uncertainty = np.sqrt(stat_var + sys_var)
    
    # Calculate overall confidence
    n_methods = len(shifts)
    agreement_factor = 1 - np.std(shifts) / (np.abs(consensus_shift) + 1e-10)
    weight_factor = 1 - stats.entropy(weights) / np.log(n_methods)
    confidence = np.clip(
        0.5 * (agreement_factor + weight_factor) *
        np.mean([m.confidence for m in methods]),
        0, 1
    )
    
    return consensus_shift, total_uncertainty, confidence

def analyze_method_performance(results: List[Dict[str, List[ShiftMethod]]],
                             true_shifts: Optional[np.ndarray] = None) -> Dict[str, Dict]:
    """Analyze performance of different shift detection methods."""
    method_stats = {}
    
    # Get unique method names
    all_methods = set()
    for result in results:
        for methods in result.values():
            all_methods.update(m.name for m in methods)
    
    for method_name in all_methods:
        stats_dict = {
            'bias': [],
            'precision': [],
            'accuracy': [],
            'reliability': [],
            'confidence': []
        }
        
        for i, result in enumerate(results):
            # Collect all instances of this method
            method_results = []
            for methods in result.values():
                for method in methods:
                    if method.name == method_name:
                        method_results.append(method)
            
            if not method_results:
                continue
            
            # Calculate metrics
            shifts = np.array([m.shift for m in method_results])
            uncertainties = np.array([m.uncertainty for m in method_results])
            confidences = np.array([m.confidence for m in method_results])
            
            if true_shifts is not None:
                # Calculate bias and accuracy
                errors = shifts - true_shifts[i]
                stats_dict['bias'].append(np.mean(errors))
                stats_dict['accuracy'].append(np.sqrt(np.mean(errors**2)))
            
            # Calculate precision
            stats_dict['precision'].append(np.mean(uncertainties))
            
            # Calculate reliability
            n_valid = np.sum(detect_outliers(method_results))
            stats_dict['reliability'].append(n_valid / len(method_results))
            
            # Store confidence
            stats_dict['confidence'].append(np.mean(confidences))
        
        # Calculate summary statistics
        method_stats[method_name] = {
            'mean_bias': np.mean(stats_dict['bias']) if stats_dict['bias'] else np.nan,
            'std_bias': np.std(stats_dict['bias']) if stats_dict['bias'] else np.nan,
            'mean_precision': np.mean(stats_dict['precision']),
            'mean_accuracy': np.mean(stats_dict['accuracy']) if stats_dict['accuracy'] else np.nan,
            'mean_reliability': np.mean(stats_dict['reliability']),
            'mean_confidence': np.mean(stats_dict['confidence'])
        }
    
    return method_stats

def combine_shift_methods(methods: List[ShiftMethod]) -> Tuple[float, float, float]:
    """Combine multiple shift detection methods into consensus result."""
    if not methods:
        return 0.0, np.inf, 0.0
    return calculate_consensus_shift(methods)
