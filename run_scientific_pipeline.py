#!/usr/bin/env python3
"""
Scientific Gas Sensing Pipeline - CODEMAP Aligned Implementation

This script implements the complete 8-step pipeline as defined in CODEMAP.md:

┌─────────────────────────────────────────────────────────────────────────────┐
│  PIPELINE FLOW (CODEMAP.md compliant)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. DATA LOADING & PREPROCESSING                                             │
│     - scan_experiment_root() → Discover CSV files                            │
│     - load_spectrum() → Load individual spectra                              │
│     - compute_transmittance() → T = I/I_ref                                  │
│     - compute_absorbance() → A = -log10(T)                                   │
│                                                                              │
│  2. FRAME SELECTION & AVERAGING                                              │
│     - find_stable_block() → Detect stable measurement region                 │
│     - find_response_peak_frames() → Select top-N responsive frames           │
│     - average_selected_frames() → Average frames                             │
│                                                                              │
│  3. CANONICAL SPECTRUM GENERATION                                            │
│     - select_canonical_per_concentration() → One spectrum per conc.          │
│     - baseline_correct_canonical() → Baseline correction                     │
│                                                                              │
│  4. ROI DISCOVERY & FEATURE DETECTION                                        │
│     - scan_roi_windows() → Find optimal ROI                                  │
│     - find_peak_wavelength() → Detect spectral features                      │
│                                                                              │
│  5. CALIBRATION                                                              │
│     - calibrate_wavelength_shift() → Δλ method                               │
│     - calibrate_absorbance_amplitude() → ΔA method                           │
│     - Linear/WLS fitting with feature selection                              │
│                                                                              │
│  6. VALIDATION & UNCERTAINTY                                                 │
│     - LOOCV (Leave-One-Out Cross-Validation)                                 │
│     - Bootstrap confidence intervals (95% CI)                                │
│     - LOD/LOQ calculation (3σ/10σ method)                                    │
│     - Spearman correlation for monotonicity                                  │
│                                                                              │
│  7. DYNAMICS ANALYSIS                                                        │
│     - compute_t90_t10() → Response/recovery times                            │
│                                                                              │
│  8. VISUALIZATION & REPORTING                                                │
│     - Publication-quality plots                                              │
│     - JSON/CSV outputs                                                       │
│     - Summary reports                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

Reference benchmark (ZnO-coated NCF sensor for acetone):
- Sensitivity: 0.116 nm/ppm
- ROI: 675-689 nm  
- LOD: 3.26 ppm
- T90: 26 s, T10: 32 s

Author: Scientific Pipeline Enhancement
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress, spearmanr
from scipy.optimize import curve_fit
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import seaborn as sns
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import load_config


class ScientificGasPipeline:
    """
    A scientifically rigorous gas sensing analysis pipeline.
    
    Design Principles:
    1. Minimal preprocessing to preserve signal integrity
    2. Response-based frame selection (peak response, not just stable)
    3. Physics-informed ROI constraints
    4. Robust statistical validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.config = load_config(str(self.config_path))
        self.results: Dict = {}

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> Path:
        if config_path:
            return Path(config_path).expanduser().resolve()
        return (Path(__file__).resolve().parent / 'config' / 'config.yaml').resolve()

    @staticmethod
    def _read_file_hash(path: Path) -> Optional[str]:
        try:
            with open(path, 'rb') as f:
                data = f.read()
            return hashlib.sha256(data).hexdigest()
        except Exception:
            return None

    @staticmethod
    def _current_git_commit(repo_root: Path) -> Optional[str]:
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=str(repo_root),
                capture_output=True,
                check=True,
            )
            commit = result.stdout.decode('utf-8', errors='ignore').strip()
            return commit or None
        except Exception:
            return None

    def _build_provenance(self, gas_name: str, data_dir: Path, ref_path: Optional[Path],
                          roi_range: Tuple[float, float], best_method: Optional[str],
                          best_abs_method: Optional[str]) -> Dict[str, object]:
        repo_root = Path(__file__).resolve().parent
        versions = {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': sp.__version__,
            'sklearn': sk.__version__,
            'matplotlib': matplotlib.__version__,
        }
        return {
            'generated_at': datetime.now().isoformat(),
            'git_commit': self._current_git_commit(repo_root),
            'config_path': str(self.config_path),
            'config_sha256': self._read_file_hash(self.config_path),
            'data_dir': str(Path(data_dir).resolve()),
            'reference_file': str(Path(ref_path).resolve()) if ref_path else None,
            'roi_range_nm': list(roi_range),
            'selected_methods': {
                'wavelength_shift': best_method,
                'absorbance': best_abs_method,
            },
            'environment': {
                'platform': platform.platform(),
                'versions': versions,
            },
            'script': str(Path(__file__).resolve()),
            'gas': gas_name,
        }
        
    def load_spectrum(self, path: Path) -> pd.DataFrame:
        """Load a single spectrum CSV file."""
        try:
            df = pd.read_csv(path, header=None, names=['wavelength', 'intensity'])
            df = df.dropna().sort_values('wavelength').reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e}")
            return pd.DataFrame()
    
    def compute_transmittance(self, sample: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """
        Compute transmittance T = I_sample / I_reference
        
        This is the fundamental measurement for optical gas sensing.
        """
        if sample.empty or reference.empty:
            return sample.copy()
        
        # Interpolate reference to sample wavelength grid
        ref_interp = np.interp(
            sample['wavelength'].values,
            reference['wavelength'].values,
            reference['intensity'].values
        )
        
        # Compute transmittance with protection against division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            T = sample['intensity'].values / np.maximum(ref_interp, 1e-10)
            T = np.clip(T, 0, 2.0)  # Physical bounds
        
        result = sample.copy()
        result['transmittance'] = T
        return result
    
    def compute_absorbance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute absorbance A = -log10(T)
        
        Beer-Lambert law: A = εcl where ε is molar absorptivity,
        c is concentration, l is path length.
        """
        if 'transmittance' not in df.columns:
            return df
        
        result = df.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            A = -np.log10(np.clip(df['transmittance'].values, 1e-6, None))
            A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        result['absorbance'] = A
        return result
    
    def find_response_peak_frames(
        self,
        frames: List[pd.DataFrame],
        reference: pd.DataFrame,
        roi_range: Tuple[float, float],
        n_select: int = 10,
        baseline_frames: int = 10
    ) -> Tuple[List[int], Dict]:
        """
        Select frames showing maximum gas response within ROI.
        
        Scientific rationale:
        - Gas exposure causes wavelength shift in the ROI
        - We want frames at steady-state response, not during transient
        - Use integrated response metric within ROI
        
        Args:
            frames: List of spectrum DataFrames
            reference: Air reference spectrum
            roi_range: (min_wl, max_wl) for ROI
            n_select: Number of frames to select
            baseline_frames: Number of initial frames for baseline
            
        Returns:
            selected_indices: Frame indices with strongest response
            metadata: Selection statistics
        """
        if not frames:
            return [], {}
        
        roi_min, roi_max = roi_range
        response_metrics = []
        
        # Compute response metric for each frame
        for i, frame in enumerate(frames):
            df_t = self.compute_transmittance(frame, reference)
            
            # Extract ROI
            mask = (df_t['wavelength'] >= roi_min) & (df_t['wavelength'] <= roi_max)
            roi_data = df_t.loc[mask]
            
            if roi_data.empty:
                response_metrics.append(0.0)
                continue
            
            # Response metric: deviation from unity transmittance in ROI
            # Gas absorption causes T < 1, so we measure 1 - mean(T)
            mean_T = roi_data['transmittance'].mean()
            response = abs(1.0 - mean_T)
            response_metrics.append(response)
        
        response_metrics = np.array(response_metrics)
        
        # Compute baseline (first N frames, assumed to be air/low concentration)
        baseline_response = np.mean(response_metrics[:baseline_frames]) if len(response_metrics) >= baseline_frames else 0.0
        
        # Normalize response relative to baseline
        delta_response = response_metrics - baseline_response
        
        # Select frames with highest response (steady-state region)
        # Prefer frames in the latter half to avoid transient
        n_frames = len(frames)
        weights = np.ones(n_frames)
        weights[:n_frames//4] = 0.5  # Reduce weight of early frames (transient)
        
        weighted_response = delta_response * weights
        
        # Select top N frames
        if n_select >= n_frames:
            selected = list(range(n_frames))
        else:
            selected = np.argsort(weighted_response)[-n_select:]
            selected = sorted(selected)  # Keep chronological order
        
        metadata = {
            'n_total_frames': n_frames,
            'n_selected': len(selected),
            'baseline_response': float(baseline_response),
            'max_response': float(np.max(response_metrics)),
            'selected_mean_response': float(np.mean(response_metrics[selected])),
            'response_enhancement': float(np.mean(response_metrics[selected]) / (baseline_response + 1e-10)),
        }
        
        return list(selected), metadata
    
    def average_selected_frames(
        self,
        frames: List[pd.DataFrame],
        indices: List[int],
        reference: pd.DataFrame
    ) -> pd.DataFrame:
        """Average selected frames with transmittance and absorbance."""
        if not indices or not frames:
            return pd.DataFrame()
        
        selected_frames = [frames[i] for i in indices if 0 <= i < len(frames)]
        if not selected_frames:
            return pd.DataFrame()
        
        # Use first frame's wavelength grid
        base_wl = selected_frames[0]['wavelength'].values
        
        # Average intensities
        intensities = []
        for f in selected_frames:
            interp_int = np.interp(base_wl, f['wavelength'].values, f['intensity'].values)
            intensities.append(interp_int)
        
        avg_intensity = np.mean(intensities, axis=0)
        
        result = pd.DataFrame({
            'wavelength': base_wl,
            'intensity': avg_intensity
        })
        
        # Add transmittance and absorbance
        result = self.compute_transmittance(result, reference)
        result = self.compute_absorbance(result)
        
        return result
    
    def find_peak_wavelength(
        self,
        df: pd.DataFrame,
        roi_range: Tuple[float, float],
        method: str = 'centroid'
    ) -> float:
        """
        Find characteristic wavelength within ROI.
        
        Methods:
        - 'centroid': Intensity-weighted center (robust to noise)
        - 'minimum': Wavelength of minimum transmittance (absorption peak)
        - 'derivative': Zero-crossing of first derivative
        """
        roi_min, roi_max = roi_range
        mask = (df['wavelength'] >= roi_min) & (df['wavelength'] <= roi_max)
        roi = df.loc[mask].copy()
        
        if roi.empty:
            return np.nan
        
        wl = roi['wavelength'].values
        
        # Use transmittance if available, else intensity
        if 'transmittance' in roi.columns:
            signal = roi['transmittance'].values
        else:
            signal = roi['intensity'].values
        
        if method == 'centroid':
            # Intensity-weighted centroid (inverted for absorption)
            weights = 1.0 - signal / (np.max(signal) + 1e-10)
            weights = np.maximum(weights, 0)
            if np.sum(weights) > 0:
                return float(np.sum(wl * weights) / np.sum(weights))
            return float(np.mean(wl))
        
        elif method == 'minimum':
            # Wavelength of minimum transmittance
            return float(wl[np.argmin(signal)])
        
        elif method == 'derivative':
            # Zero-crossing of smoothed first derivative
            if len(signal) < 7:
                return float(np.mean(wl))
            smoothed = savgol_filter(signal, min(7, len(signal)//2*2+1), 2)
            deriv = np.gradient(smoothed, wl)
            # Find zero-crossings
            zero_crossings = np.where(np.diff(np.sign(deriv)))[0]
            if len(zero_crossings) > 0:
                return float(wl[zero_crossings[len(zero_crossings)//2]])
            return float(wl[np.argmin(signal)])
        
        return float(np.mean(wl))
    
    def scan_roi_windows(
        self,
        canonical_spectra: Dict[float, pd.DataFrame],
        scan_range: Tuple[float, float] = (500, 900),
        window_sizes: List[float] = [10, 15, 20, 25],
        step: float = 5.0,
        method: str = 'centroid'
    ) -> Dict:
        """
        Scan across wavelength range to find optimal ROI window.
        
        Scientific rationale:
        - The optimal ROI may not be at the expected location
        - Scan multiple window sizes and positions
        - Select based on R² and monotonicity (Spearman ρ)
        - Prioritize regions with Linear Relationship (High R²) and Low LOD
        - **NEW**: Prioritize Sensitivity > Benchmark (previous reported value)
        
        Returns:
            Best ROI configuration and all candidates
        """
        candidates = []
        
        # Get thresholds from config
        discovery_cfg = self.config.get('roi', {}).get('discovery', {})
        min_r2_gate = discovery_cfg.get('gates', {}).get('min_r2', 0.6)
        high_quality_r2 = discovery_cfg.get('best_sensitivity_min_r2', 0.95)
        
        # Get Benchmark Sensitivity
        benchmarks = self.config.get('benchmarks', {}).get('baseline_znc_ncf', {})
        benchmark_sensitivity = benchmarks.get('sensitivity_nm_per_ppm', 0.116)
        
        weights = discovery_cfg.get('weights', {'r2': 0.6, 'slope': 0.0, 'snr': 0.4})
        w_r2 = weights.get('r2', 0.6)
        w_spearman = weights.get('snr', 0.4) 
        
        # Determine methods to scan
        if method == 'auto':
            methods_to_scan = ['centroid', 'minimum', 'derivative']
        else:
            methods_to_scan = [method]

        for window_size in window_sizes:
            half_w = window_size / 2
            center = scan_range[0] + half_w
            
            while center + half_w <= scan_range[1]:
                roi = (center - half_w, center + half_w)
                
                for scan_method in methods_to_scan:
                    try:
                        result = self.calibrate_wavelength_shift(
                            canonical_spectra, roi, scan_method
                        )
                        
                        if 'error' not in result:
                            # Metrics
                            r2 = result['r2']
                            spearman = abs(result['spearman_r'])
                            slope = abs(result['slope'])
                            lod = result['lod_ppm']
                            
                            # Valid Candidate Gate
                            if r2 < min_r2_gate:
                                continue

                            # Score: Weighted combination for fallback
                            score = w_r2 * r2 + w_spearman * spearman
                            
                            candidates.append({
                                'roi_center': center,
                                'roi_width': window_size,
                                'roi_range': list(roi),
                                'method': scan_method,
                                'r2': r2,
                                'spearman_r': result['spearman_r'],
                                'slope': result['slope'], # Signed slope
                                'lod_ppm': lod,
                                'score': score,
                                'full_result': result
                            })
                    except Exception:
                        pass
                
                center += step
        
        if not candidates:
            return {'error': 'No valid ROI found. Check signal quality or lower min_r2 gate.'}
        
        # ---------------------------------------------------------
        # Scientific Selection Logic: "Performance & Sensitivity"
        # 1. Linearity: Must have R² >= High Quality Threshold
        # 2. Benchmark Check: beat sensitivity 0.116 nm/ppm?
        # 3. Optimization: Best LOD among those who pass.
        # ---------------------------------------------------------
        
        linear_candidates = [c for c in candidates if c['r2'] >= high_quality_r2]
        
        # Split into "High Sensitivity" vs "Standard"
        # We look for absolute slope > benchmark
        high_sensitivity_candidates = [c for c in linear_candidates if abs(c['slope']) > benchmark_sensitivity]
        
        best = None
        selection_reason = ""
        
        if high_sensitivity_candidates:
            # We have candidates that are BOTH Linear AND Sensitivity > Benchmark
            # Sort by LOD (desc quality -> asc LOD)
            valid_lod = [c for c in high_sensitivity_candidates if np.isfinite(c['lod_ppm'])]
            if valid_lod:
                valid_lod.sort(key=lambda x: (x['lod_ppm'], -abs(x['slope'])))
                best = valid_lod[0]
                selection_reason = f"Linear (R²>={high_quality_r2}) AND High Sensitivity (>{benchmark_sensitivity})"
            else:
                # Fallback to just highest slope
                high_sensitivity_candidates.sort(key=lambda x: abs(x['slope']), reverse=True)
                best = high_sensitivity_candidates[0]
                selection_reason = f"Linear & High Sensitivity (LOD infinite)"
                
        elif linear_candidates:
            # No candidate beat the benchmark, but we have linear ones.
            # Select best LOD among these.
            print(f"[WARNING] No candidate beat benchmark sensitivity ({benchmark_sensitivity} nm/ppm) with R²>={high_quality_r2}.")
            valid_lod = [c for c in linear_candidates if np.isfinite(c['lod_ppm'])]
            if valid_lod:
                valid_lod.sort(key=lambda x: (x['lod_ppm'], -abs(x['slope'])))
                best = valid_lod[0]
                selection_reason = f"Linear (R²>={high_quality_r2}) - Benchmark missed"
            else:
                linear_candidates.sort(key=lambda x: abs(x['slope']), reverse=True)
                best = linear_candidates[0]
                selection_reason = f"Linear (R²>={high_quality_r2}) - Max Sensitivity (LOD inf)"
                
        else:
            # Fallback: No highly linear region found.
            candidates.sort(key=lambda x: x['score'], reverse=True)
            best = candidates[0]
            selection_reason = f"Fallback (R² < {high_quality_r2})"
            print(f"[WARNING] No linear region found. Best R²={best['r2']:.4f}")
            
        print(f"[DEBUG] Optimal Selection: {selection_reason}")
        print(f"        ROI {best['roi_range']} nm | Method: {best['method']}")
        print(f"        LOD: {best['lod_ppm']:.4f} ppm | Sensitivity: {abs(best['slope']):.4f} nm/ppm | R²: {best['r2']:.4f}")
        
        # Sort all candidates for reporting (by score by default)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'best_roi': best['roi_range'],
            'best_center': best['roi_center'],
            'best_width': best['roi_width'],
            'best_method': best['method'],
            'best_r2': best['r2'],
            'best_spearman': best['spearman_r'],
            'best_slope': best['slope'],
            'best_lod': best['lod_ppm'],
            'best_score': best['score'],
            'calibration': best['full_result'],
            'n_candidates': len(candidates),
            'top_10_candidates': [
                {k: v for k, v in c.items() if k != 'full_result'}
                for c in candidates[:10]
            ]
        }
    
    def calibrate_wavelength_shift(
        self,
        canonical_spectra: Dict[float, pd.DataFrame],
        roi_range: Tuple[float, float],
        method: str = 'centroid'
    ) -> Dict:
        """
        Perform wavelength shift calibration.
        
        Scientific basis:
        - Gas adsorption changes effective refractive index
        - This causes wavelength shift Δλ proportional to concentration
        - Δλ = S × C where S is sensitivity (nm/ppm)
        
        Returns:
            Calibration results including slope, R², LOD, etc.
        """
        if not canonical_spectra:
            return {'error': 'No spectra provided'}
        
        concentrations = sorted(canonical_spectra.keys())
        peak_wavelengths = []
        
        for conc in concentrations:
            df = canonical_spectra[conc]
            peak_wl = self.find_peak_wavelength(df, roi_range, method)
            peak_wavelengths.append(peak_wl)
        
        concs = np.array(concentrations)
        peaks = np.array(peak_wavelengths)
        
        # Remove NaN values
        valid = np.isfinite(peaks)
        if np.sum(valid) < 3:
            return {'error': 'Insufficient valid data points'}
        
        concs = concs[valid]
        peaks = peaks[valid]
        
        # Reference wavelength (lowest concentration)
        ref_wl = peaks[0]
        delta_lambda = peaks - ref_wl
        
        # Linear regression: Δλ = slope × C + intercept
        slope, intercept, r_value, p_value, std_err = linregress(concs, delta_lambda)
        
        # Spearman correlation for monotonicity check
        spearman_r, spearman_p = spearmanr(concs, delta_lambda)
        
        # Residuals and RMSE
        predicted = slope * concs + intercept
        residuals = delta_lambda - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Noise estimation from residuals
        noise_std = np.std(residuals)
        
        # LOD = 3σ/slope (3-sigma criterion)
        lod = 3 * noise_std / abs(slope) if abs(slope) > 1e-10 else np.inf
        
        # LOQ = 10σ/slope
        loq = 10 * noise_std / abs(slope) if abs(slope) > 1e-10 else np.inf
        
        # Bootstrap confidence intervals for slope
        n_bootstrap = 500
        bootstrap_slopes = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(concs), len(concs), replace=True)
            if len(np.unique(idx)) < 3:
                continue
            try:
                b_slope, _, _, _, _ = linregress(concs[idx], delta_lambda[idx])
                bootstrap_slopes.append(b_slope)
            except:
                pass
        
        if bootstrap_slopes:
            slope_ci_low = np.percentile(bootstrap_slopes, 2.5)
            slope_ci_high = np.percentile(bootstrap_slopes, 97.5)
        else:
            slope_ci_low = slope - 1.96 * std_err
            slope_ci_high = slope + 1.96 * std_err
        
        return {
            'concentrations': concs.tolist(),
            'peak_wavelengths': peaks.tolist(),
            'delta_lambda': delta_lambda.tolist(),
            'reference_wavelength': float(ref_wl),
            'slope': float(slope),
            'slope_unit': 'nm/ppm',
            'intercept': float(intercept),
            'r2': float(r_value**2),
            'r_value': float(r_value),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'rmse': float(rmse),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'noise_std': float(noise_std),
            'lod_ppm': float(lod),
            'loq_ppm': float(loq),
            'slope_ci_95': [float(slope_ci_low), float(slope_ci_high)],
            'n_points': len(concs),
            'method': method,
            'roi_range': list(roi_range),
        }
    
    def calibrate_absorbance_amplitude(
        self,
        canonical_spectra: Dict[float, pd.DataFrame],
        roi_range: Tuple[float, float],
        method: str = 'window_avg'
    ) -> Dict:
        """
        Perform absorbance amplitude (ΔA) calibration per CODEMAP Step 5.
        
        ΔA = A(C) - A(0)
        
        Methods:
        - 'raw': Direct absorbance value at peak
        - 'window_avg': ±2 nm window average (more robust)
        - 'derivative': First derivative dA/dλ
        - 'differential': Subtract reference wavelength
        """
        if not canonical_spectra:
            return {'error': 'No spectra provided'}
        
        concentrations = sorted(canonical_spectra.keys())
        absorbance_values = []
        roi_min, roi_max = roi_range
        
        for conc in concentrations:
            df = canonical_spectra[conc]
            if 'absorbance' not in df.columns:
                absorbance_values.append(np.nan)
                continue
            
            mask = (df['wavelength'] >= roi_min) & (df['wavelength'] <= roi_max)
            roi = df.loc[mask]
            
            if roi.empty:
                absorbance_values.append(np.nan)
                continue
            
            if method == 'raw':
                # Peak absorbance value
                abs_val = roi['absorbance'].max()
            elif method == 'window_avg':
                # Average absorbance in ROI (±2 nm window around peak)
                peak_idx = roi['absorbance'].idxmax()
                peak_wl = roi.loc[peak_idx, 'wavelength']
                window_mask = (df['wavelength'] >= peak_wl - 2) & (df['wavelength'] <= peak_wl + 2)
                abs_val = df.loc[window_mask, 'absorbance'].mean()
            elif method == 'derivative':
                # First derivative maximum
                wl = roi['wavelength'].values
                A = roi['absorbance'].values
                if len(A) > 3:
                    dA = np.gradient(A, wl)
                    abs_val = np.max(np.abs(dA))
                else:
                    abs_val = roi['absorbance'].max()
            elif method == 'differential':
                # Differential: peak - baseline region
                peak_A = roi['absorbance'].max()
                # Use edges of ROI as baseline
                baseline_A = (roi['absorbance'].iloc[:3].mean() + roi['absorbance'].iloc[-3:].mean()) / 2
                abs_val = peak_A - baseline_A
            else:
                abs_val = roi['absorbance'].mean()
            
            absorbance_values.append(abs_val)
        
        concs = np.array(concentrations)
        A_vals = np.array(absorbance_values)
        
        valid = np.isfinite(A_vals)
        if np.sum(valid) < 3:
            return {'error': 'Insufficient valid data points'}
        
        concs = concs[valid]
        A_vals = A_vals[valid]
        
        # Reference absorbance (lowest concentration)
        ref_A = A_vals[0]
        delta_A = A_vals - ref_A
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(concs, delta_A)
        spearman_r, spearman_p = spearmanr(concs, delta_A)
        
        predicted = slope * concs + intercept
        residuals = delta_A - predicted
        rmse = np.sqrt(np.mean(residuals**2))
        noise_std = np.std(residuals)
        
        lod = 3 * noise_std / abs(slope) if abs(slope) > 1e-10 else np.inf
        loq = 10 * noise_std / abs(slope) if abs(slope) > 1e-10 else np.inf
        
        return {
            'concentrations': concs.tolist(),
            'absorbance_values': A_vals.tolist(),
            'delta_A': delta_A.tolist(),
            'reference_absorbance': float(ref_A),
            'slope': float(slope),
            'slope_unit': 'AU/ppm',
            'intercept': float(intercept),
            'r2': float(r_value**2),
            'spearman_r': float(spearman_r),
            'rmse': float(rmse),
            'lod_ppm': float(lod),
            'loq_ppm': float(loq),
            'method': method,
            'roi_range': list(roi_range),
        }
    
    def compute_loocv(
        self,
        concs: np.ndarray,
        values: np.ndarray
    ) -> Dict:
        """
        Leave-One-Out Cross-Validation per CODEMAP Step 6.
        
        For each point, fit model on remaining points and predict left-out point.
        """
        n = len(concs)
        if n < 4:
            return {'error': 'Insufficient points for LOOCV'}
        
        predictions = np.zeros(n)
        residuals = np.zeros(n)
        
        for i in range(n):
            # Leave out point i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            
            # Fit on remaining points
            slope_i, intercept_i, _, _, _ = linregress(concs[mask], values[mask])
            
            # Predict left-out point
            predictions[i] = slope_i * concs[i] + intercept_i
            residuals[i] = values[i] - predictions[i]
        
        # LOOCV metrics
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((values - np.mean(values))**2)
        r2_cv = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        rmse_cv = np.sqrt(np.mean(residuals**2))
        
        return {
            'r2_cv': float(r2_cv),
            'rmse_cv': float(rmse_cv),
            'predictions': predictions.tolist(),
            'residuals': residuals.tolist(),
            'n_folds': n,
        }
    
    def compute_t90_t10(
        self,
        time_series: np.ndarray,
        frame_rate: float = 1.0,
        baseline_frames: int = 10
    ) -> Dict:
        """
        Compute T90 (response time) and T10 (recovery time) per CODEMAP Step 7.
        
        T90: Time to reach 90% of steady-state response
        T10: Time to recover to 10% of response (from peak)
        
        Args:
            time_series: Array of response values over time
            frame_rate: Frames per second
            baseline_frames: Number of initial frames for baseline
        """
        if len(time_series) < baseline_frames + 10:
            return {'error': 'Insufficient time series data'}
        
        # Baseline (first N frames)
        baseline = np.mean(time_series[:baseline_frames])
        baseline_std = np.std(time_series[:baseline_frames])
        
        # Find response region (above baseline + 3σ)
        threshold = baseline + 3 * baseline_std
        response_mask = time_series > threshold
        
        if not np.any(response_mask):
            return {'error': 'No significant response detected'}
        
        # Find peak response
        peak_idx = np.argmax(time_series)
        peak_value = time_series[peak_idx]
        
        # Response amplitude
        amplitude = peak_value - baseline
        
        # T90: Time to reach 90% of amplitude
        target_90 = baseline + 0.9 * amplitude
        t90_idx = None
        for i in range(baseline_frames, peak_idx + 1):
            if time_series[i] >= target_90:
                t90_idx = i
                break
        
        t90 = (t90_idx - baseline_frames) / frame_rate if t90_idx else np.nan
        
        # T10: Time to recover to 10% of amplitude (from peak)
        target_10 = baseline + 0.1 * amplitude
        t10_idx = None
        for i in range(peak_idx, len(time_series)):
            if time_series[i] <= target_10:
                t10_idx = i
                break
        
        t10 = (t10_idx - peak_idx) / frame_rate if t10_idx else np.nan
        
        return {
            'baseline': float(baseline),
            'peak_value': float(peak_value),
            'amplitude': float(amplitude),
            't90_seconds': float(t90) if np.isfinite(t90) else None,
            't10_seconds': float(t10) if np.isfinite(t10) else None,
            'peak_frame': int(peak_idx),
            'frame_rate': float(frame_rate),
        }
    
    def scan_experiment_directory(self, root_dir: Path) -> Dict[float, Dict[str, List[Path]]]:
        """
        Scan experiment directory structure.
        
        Expected structure:
        root_dir/
            {concentration}ppm/
                T1/
                    *.csv
                T2/
                    *.csv
        """
        mapping = {}
        
        for conc_dir in sorted(root_dir.iterdir()):
            if not conc_dir.is_dir():
                continue
            
            # Parse concentration from directory name
            name = conc_dir.name.lower()
            try:
                # Handle formats like "1ppm", "10ppm", "0.5ppm"
                conc_str = name.replace('ppm', '').strip()
                conc = float(conc_str)
            except ValueError:
                continue
            
            trials = {}
            for trial_dir in sorted(conc_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                
                frames = sorted(trial_dir.glob('*.csv'))
                if frames:
                    trials[trial_dir.name] = frames
            
            if trials:
                mapping[conc] = trials
        
        return mapping
    
    def run_analysis(
        self,
        data_dir: Path,
        ref_path: Path,
        output_dir: Path,
        gas_name: str = 'Unknown',
        n_frames_select: int = 10
    ) -> Dict:
        """
        Run complete analysis pipeline.
        
        Steps:
        1. Load reference spectrum
        2. Scan experiment directory
        3. For each concentration:
           a. For each trial:
              - Load frames
              - Select response-peak frames
              - Average selected frames
           b. Average trials for canonical spectrum
        4. Perform wavelength shift calibration
        5. Generate outputs
        """
        print(f"\n{'='*60}")
        print(f"Scientific Gas Sensing Pipeline - {gas_name}")
        print(f"{'='*60}")
        
        # Get ROI configuration
        roi_cfg = self.config.get('roi', {}).get('per_gas_overrides', {}).get(gas_name, {})
        roi_range_cfg = roi_cfg.get('range', {})
        roi_min = roi_range_cfg.get('min_wavelength', 670.0)
        roi_max = roi_range_cfg.get('max_wavelength', 695.0)
        roi_range = (roi_min, roi_max)
        
        expected_center = roi_cfg.get('validation', {}).get('expected_center', (roi_min + roi_max) / 2)
        
        print(f"\nConfiguration:")
        print(f"  ROI: {roi_min:.1f} - {roi_max:.1f} nm")
        print(f"  Expected center: {expected_center:.1f} nm")
        print(f"  Frames to select per trial: {n_frames_select}")
        
        # Load reference
        print(f"\nLoading reference: {ref_path}")
        reference = self.load_spectrum(ref_path)
        if reference.empty:
            return {'error': f'Failed to load reference: {ref_path}'}
        print(f"  Reference loaded: {len(reference)} points, {reference['wavelength'].min():.1f}-{reference['wavelength'].max():.1f} nm")
        
        # Scan experiment directory
        print(f"\nScanning data directory: {data_dir}")
        mapping = self.scan_experiment_directory(data_dir)
        
        if not mapping:
            return {'error': f'No valid data found in {data_dir}'}
        
        print(f"  Found {len(mapping)} concentrations: {sorted(mapping.keys())} ppm")
        
        # Process each concentration
        canonical_spectra = {}
        processing_metadata = {}
        
        for conc in sorted(mapping.keys()):
            trials = mapping[conc]
            print(f"\n[{conc} ppm] Processing {len(trials)} trials...")
            
            trial_spectra = []
            trial_metadata = {}
            
            for trial_name, frame_paths in trials.items():
                # Load frames
                frames = [self.load_spectrum(p) for p in frame_paths]
                frames = [f for f in frames if not f.empty]
                
                if not frames:
                    print(f"  {trial_name}: No valid frames")
                    continue
                
                # Select response-peak frames
                selected_idx, sel_meta = self.find_response_peak_frames(
                    frames, reference, roi_range, n_select=n_frames_select
                )
                
                if not selected_idx:
                    print(f"  {trial_name}: No frames selected")
                    continue
                
                # Average selected frames
                avg_spectrum = self.average_selected_frames(frames, selected_idx, reference)
                
                if avg_spectrum.empty:
                    continue
                
                trial_spectra.append(avg_spectrum)
                trial_metadata[trial_name] = sel_meta
                
                print(f"  {trial_name}: {len(frames)} frames -> selected {len(selected_idx)}, enhancement={sel_meta.get('response_enhancement', 0):.2f}x")
            
            if not trial_spectra:
                print(f"  [WARNING] No valid trials for {conc} ppm")
                continue
            
            # Average trials for canonical spectrum
            base_wl = trial_spectra[0]['wavelength'].values
            avg_data = {'wavelength': base_wl}
            
            for col in ['intensity', 'transmittance', 'absorbance']:
                if col in trial_spectra[0].columns:
                    values = [np.interp(base_wl, t['wavelength'].values, t[col].values) for t in trial_spectra]
                    avg_data[col] = np.mean(values, axis=0)
            
            canonical_spectra[conc] = pd.DataFrame(avg_data)
            processing_metadata[conc] = {
                'n_trials': len(trial_spectra),
                'trials': trial_metadata
            }
        
        if not canonical_spectra:
            return {'error': 'No canonical spectra generated'}
        
        print(f"\n{'='*60}")
        print("Calibration Analysis")
        print(f"{'='*60}")
        
        # First: Scan for optimal ROI
        print("\n[ROI SCAN] Searching for optimal wavelength window...")
        scan_result = self.scan_roi_windows(
            canonical_spectra,
            scan_range=(500, 900),
            window_sizes=[10, 15, 20, 25, 30],
            step=5.0,
            method='auto'
        )
        
        roi_scan_results = None
        if 'error' not in scan_result:
            roi_scan_results = scan_result
            print(f"\n  Best ROI found: {scan_result['best_roi'][0]:.1f}-{scan_result['best_roi'][1]:.1f} nm")
            print(f"  Best R²: {scan_result['best_r2']:.4f}")
            print(f"  Best Spearman rho: {scan_result['best_spearman']:.4f}")
            print(f"  Best sensitivity: {scan_result['best_slope']:.4f} nm/ppm")
            print(f"  Scanned {scan_result['n_candidates']} candidates")
            
            # Show top 5 candidates
            print("\n  Top 5 ROI candidates:")
            for i, cand in enumerate(scan_result['top_10_candidates'][:5]):
                print(f"    {i+1}. {cand['roi_range'][0]:.0f}-{cand['roi_range'][1]:.0f} nm: "
                      f"R²={cand['r2']:.3f}, rho={cand['spearman_r']:.3f}, slope={cand['slope']:.4f}")
            
            # Use best ROI for final calibration
            roi_range = tuple(scan_result['best_roi'])
        
        # Perform calibration with multiple methods using best ROI
        print(f"\n[CALIBRATION] Using ROI: {roi_range[0]:.1f}-{roi_range[1]:.1f} nm")
        methods = ['centroid', 'minimum']
        calibration_results = {}
        
        for method in methods:
            result = self.calibrate_wavelength_shift(canonical_spectra, roi_range, method)
            calibration_results[method] = result
            
            if 'error' not in result:
                print(f"\n{method.upper()} method:")
                print(f"  Sensitivity: {result['slope']:.4f} nm/ppm")
                print(f"  R²: {result['r2']:.4f}")
                print(f"  LOD: {result['lod_ppm']:.2f} ppm")
                print(f"  Spearman rho: {result['spearman_r']:.4f}")
        
        # Select best method by R²
        best_method = max(
            [m for m in methods if 'error' not in calibration_results[m]],
            key=lambda m: calibration_results[m]['r2'],
            default=None
        )
        
        if best_method:
            print(f"\n-> Best wavelength shift method: {best_method} (R² = {calibration_results[best_method]['r2']:.4f})")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 5b: Absorbance Amplitude (ΔA) Calibration per CODEMAP
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n[DA CALIBRATION] Absorbance amplitude method...")
        absorbance_results = {}
        for method in ['raw', 'window_avg', 'differential']:
            result = self.calibrate_absorbance_amplitude(canonical_spectra, roi_range, method)
            absorbance_results[method] = result
            if 'error' not in result:
                print(f"  {method}: R²={result['r2']:.4f}, slope={result['slope']:.4f} AU/ppm")
        
        best_abs_method = max(
            [m for m in absorbance_results if 'error' not in absorbance_results[m]],
            key=lambda m: absorbance_results[m]['r2'],
            default=None
        )
        if best_abs_method:
            print(f"-> Best absorbance method: {best_abs_method} (R² = {absorbance_results[best_abs_method]['r2']:.4f})")
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: LOOCV Validation per CODEMAP
        # ═══════════════════════════════════════════════════════════════════
        loocv_results = {}
        if best_method and 'error' not in calibration_results[best_method]:
            cal = calibration_results[best_method]
            concs = np.array(cal['concentrations'])
            delta_lambda = np.array(cal['delta_lambda'])
            loocv = self.compute_loocv(concs, delta_lambda)
            loocv_results['wavelength_shift'] = loocv
            if 'error' not in loocv:
                print(f"\n[LOOCV] wavelength shift: R²_CV={loocv['r2_cv']:.4f}, RMSE_CV={loocv['rmse_cv']:.4f}")
                calibration_results[best_method]['r2_cv'] = loocv['r2_cv']
                calibration_results[best_method]['rmse_cv'] = loocv['rmse_cv']
        
        if best_abs_method and 'error' not in absorbance_results[best_abs_method]:
            cal = absorbance_results[best_abs_method]
            concs = np.array(cal['concentrations'])
            delta_A = np.array(cal['delta_A'])
            loocv = self.compute_loocv(concs, delta_A)
            loocv_results['absorbance'] = loocv
            if 'error' not in loocv:
                print(f"[LOOCV] absorbance: R²_CV={loocv['r2_cv']:.4f}, RMSE_CV={loocv['rmse_cv']:.4f}")
                absorbance_results[best_abs_method]['r2_cv'] = loocv['r2_cv']
                absorbance_results[best_abs_method]['rmse_cv'] = loocv['rmse_cv']
        
        # Save outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CODEMAP-compliant directory structure
        (output_dir / 'plots').mkdir(exist_ok=True)
        (output_dir / 'metrics').mkdir(exist_ok=True)
        (output_dir / 'reports').mkdir(exist_ok=True)
        (output_dir / 'canonical_spectra').mkdir(exist_ok=True)
        
        # Save calibration results
        provenance = self._build_provenance(
            gas_name=gas_name,
            data_dir=Path(data_dir),
            ref_path=Path(ref_path) if ref_path else None,
            roi_range=roi_range,
            best_method=best_method,
            best_abs_method=best_abs_method,
        )

        results = {
            'gas': gas_name,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': 'CODEMAP_aligned_v1.0',
            'roi_range': list(roi_range),
            'expected_center': expected_center,
            'n_concentrations': len(canonical_spectra),
            'concentrations': sorted(canonical_spectra.keys()),
            'calibration_wavelength_shift': calibration_results,
            'calibration_absorbance': absorbance_results,
            'best_method_wavelength_shift': best_method,
            'best_method_absorbance': best_abs_method,
            'loocv_validation': loocv_results,
            'roi_scan': roi_scan_results,
            'processing': processing_metadata,
            'benchmark_comparison': {
                'paper_sensitivity': 0.116,
                'paper_lod': 3.26,
                'paper_roi': [675, 689],
                'achieved_sensitivity': calibration_results.get(best_method, {}).get('slope', np.nan),
                'achieved_lod': calibration_results.get(best_method, {}).get('lod_ppm', np.nan),
                'achieved_r2': calibration_results.get(best_method, {}).get('r2', np.nan),
                'achieved_spearman': calibration_results.get(best_method, {}).get('spearman_r', np.nan),
            },
            'provenance': provenance,
        }
        
        # ═══════════════════════════════════════════════════════════════════
        # STEP 8: Save outputs per CODEMAP structure
        # ═══════════════════════════════════════════════════════════════════
        
        # Save main calibration JSON to metrics/
        json_path = output_dir / 'metrics' / 'calibration_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OUTPUT] Calibration metrics: {json_path}")
        
        # Save absorbance calibration separately
        abs_json_path = output_dir / 'metrics' / 'absorbance_amplitude_calibration.json'
        with open(abs_json_path, 'w') as f:
            json.dump({'absorbance_calibration': absorbance_results, 'best_method': best_abs_method}, f, indent=2, default=str)
        
        # Save canonical spectra
        for conc, df in canonical_spectra.items():
            df.to_csv(output_dir / 'canonical_spectra' / f'{conc}ppm.csv', index=False)
        
        # ═══════════════════════════════════════════════════════════════════
        # Generate ALL scientific plots
        # ═══════════════════════════════════════════════════════════════════
        print("\n[PLOTS] Generating publication-quality figures...")
        
        # 1. Main calibration curve (Δλ vs C)
        self._plot_calibration(calibration_results, best_method, output_dir, gas_name)
        
        # 2. Spectral overlay with ROI highlight
        self._plot_spectral_overlay(canonical_spectra, roi_range, output_dir, gas_name)
        
        # 3. Method comparison bar chart
        self._plot_method_comparison(calibration_results, absorbance_results, output_dir, gas_name)
        
        # 4. Absorbance calibration curve (ΔA vs C)
        self._plot_absorbance_calibration(absorbance_results, best_abs_method, output_dir, gas_name)
        
        # 5. Residual analysis (diagnostic plot)
        self._plot_residual_analysis(calibration_results, best_method, output_dir, gas_name)
        
        # 6. ROI heatmap showing R² across wavelength windows
        if roi_scan_results:
            self._plot_roi_heatmap(roi_scan_results, output_dir, gas_name)
        
        # 7. Comprehensive 6-panel diagnostic figure
        self._plot_comprehensive_diagnostic(
            canonical_spectra, calibration_results, absorbance_results,
            best_method, best_abs_method, roi_range, output_dir, gas_name
        )
        
        # 8. Peak wavelength tracking plot
        self._plot_peak_tracking(calibration_results, best_method, output_dir, gas_name)
        
        # Generate summary report
        self._generate_summary_report(results, output_dir, gas_name)
        
        print(f"\n[OUTPUT] All outputs saved to: {output_dir}")
        
        return results
    
    def _plot_method_comparison(self, wavelength_results: Dict, absorbance_results: Dict, 
                                output_dir: Path, gas_name: str):
        """Generate method comparison plot per CODEMAP."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Δλ methods comparison
            ax1 = axes[0]
            methods = []
            r2_values = []
            for method, result in wavelength_results.items():
                if 'error' not in result:
                    methods.append(method)
                    r2_values.append(result['r2'])
            
            if methods:
                colors = ['#2E86AB' if r2 == max(r2_values) else '#A0A0A0' for r2 in r2_values]
                ax1.bar(methods, r2_values, color=colors, edgecolor='black')
                ax1.set_ylabel('R²', fontsize=11)
                ax1.set_title('Wavelength Shift (Δλ) Methods', fontsize=12, fontweight='bold')
                ax1.set_ylim(0, 1.05)
                ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target R²=0.95')
                ax1.legend()
            
            # ΔA methods comparison
            ax2 = axes[1]
            methods = []
            r2_values = []
            for method, result in absorbance_results.items():
                if 'error' not in result:
                    methods.append(method)
                    r2_values.append(result['r2'])
            
            if methods:
                colors = ['#9BBB59' if r2 == max(r2_values) else '#A0A0A0' for r2 in r2_values]
                ax2.bar(methods, r2_values, color=colors, edgecolor='black')
                ax2.set_ylabel('R²', fontsize=11)
                ax2.set_title('Absorbance Amplitude (ΔA) Methods', fontsize=12, fontweight='bold')
                ax2.set_ylim(0, 1.05)
                ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Target R²=0.95')
                ax2.legend()
            
            plt.suptitle(f'{gas_name} - Method Comparison', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_dir / 'plots' / 'method_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"[WARNING] Failed to generate method comparison plot: {e}")
    
    def _plot_absorbance_calibration(self, results: Dict, best_method: str, output_dir: Path, gas_name: str):
        """Generate absorbance amplitude calibration plot (ΔA vs C)."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            if not best_method or 'error' in results.get(best_method, {}):
                return
            
            data = results[best_method]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            concs = np.array(data['concentrations'])
            delta_A = np.array(data['delta_A'])
            
            # Data points
            ax.scatter(concs, delta_A, s=80, c='#9BBB59', edgecolors='black', 
                      linewidths=1, zorder=5, label='Measured')
            
            # Fit line
            x_fit = np.linspace(0, max(concs) * 1.1, 100)
            y_fit = data['slope'] * x_fit + data['intercept']
            ax.plot(x_fit, y_fit, 'g-', linewidth=2, label=f'Linear fit (R² = {data["r2"]:.4f})')
            
            ax.set_xlabel('Concentration (ppm)', fontsize=12)
            ax.set_ylabel('Absorbance Change ΔA (AU)', fontsize=12)
            ax.set_title(f'{gas_name} Absorbance Calibration\nSensitivity: {data["slope"]:.6f} AU/ppm', 
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(concs) * 1.1)
            
            plt.tight_layout()
            plot_path = output_dir / 'plots' / 'absorbance_calibration.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Absorbance calibration: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate absorbance calibration plot: {e}")
    
    def _plot_residual_analysis(self, results: Dict, best_method: str, output_dir: Path, gas_name: str):
        """Generate residual analysis diagnostic plot."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            from scipy import stats
            
            if not best_method or 'error' in results.get(best_method, {}):
                return
            
            data = results[best_method]
            concs = np.array(data['concentrations'])
            delta_lambda = np.array(data['delta_lambda'])
            predicted = data['slope'] * concs + data['intercept']
            residuals = delta_lambda - predicted
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Residuals vs Fitted
            ax1 = axes[0, 0]
            ax1.scatter(predicted, residuals, s=60, c='#2E86AB', edgecolors='black')
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax1.set_xlabel('Fitted Values (nm)', fontsize=11)
            ax1.set_ylabel('Residuals (nm)', fontsize=11)
            ax1.set_title('Residuals vs Fitted', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # 2. Q-Q Plot
            ax2 = axes[0, 1]
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Normal Q-Q Plot', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # 3. Residuals vs Concentration
            ax3 = axes[1, 0]
            ax3.scatter(concs, residuals, s=60, c='#2E86AB', edgecolors='black')
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax3.set_xlabel('Concentration (ppm)', fontsize=11)
            ax3.set_ylabel('Residuals (nm)', fontsize=11)
            ax3.set_title('Residuals vs Concentration', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # 4. Histogram of Residuals
            ax4 = axes[1, 1]
            ax4.hist(residuals, bins=max(5, len(residuals)//2), color='#2E86AB', 
                    edgecolor='black', alpha=0.7)
            ax4.axvline(x=0, color='red', linestyle='--', linewidth=1)
            ax4.set_xlabel('Residuals (nm)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Residual Distribution', fontsize=12, fontweight='bold')
            
            # Add statistics annotation
            stats_text = f'Mean: {np.mean(residuals):.4f}\nStd: {np.std(residuals):.4f}'
            ax4.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                        fontsize=10, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle(f'{gas_name} - Residual Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_dir / 'plots' / 'residual_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Residual analysis: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate residual analysis plot: {e}")
    
    def _plot_roi_heatmap(self, roi_scan_results: Dict, output_dir: Path, gas_name: str):
        """Generate ROI scan heatmap showing R² across wavelength windows."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            candidates = roi_scan_results.get('top_10_candidates', [])
            if not candidates:
                return
            
            # Get all candidates for visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 1. R² vs ROI center
            ax1 = axes[0]
            centers = [c['roi_center'] for c in candidates]
            r2_vals = [c['r2'] for c in candidates]
            slopes = [c['slope'] for c in candidates]
            
            scatter = ax1.scatter(centers, r2_vals, c=np.abs(slopes), cmap='viridis', 
                                 s=100, edgecolors='black', linewidths=0.5)
            ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='R²=0.95')
            
            # Highlight best
            best_center = roi_scan_results.get('best_center')
            best_r2 = roi_scan_results.get('best_r2')
            if best_center and best_r2:
                ax1.scatter([best_center], [best_r2], s=200, c='red', marker='*', 
                           zorder=10, label=f'Best: {best_center:.0f} nm')
            
            ax1.set_xlabel('ROI Center (nm)', fontsize=11)
            ax1.set_ylabel('R²', fontsize=11)
            ax1.set_title('ROI Scan Results', fontsize=12, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='|Slope| (nm/ppm)')
            
            # 2. Top candidates bar chart
            ax2 = axes[1]
            top_n = min(10, len(candidates))
            labels = [f"{c['roi_range'][0]:.0f}-{c['roi_range'][1]:.0f}" for c in candidates[:top_n]]
            r2_top = [c['r2'] for c in candidates[:top_n]]
            colors = ['#2E86AB' if i == 0 else '#A0A0A0' for i in range(top_n)]
            
            bars = ax2.barh(range(top_n), r2_top, color=colors, edgecolor='black')
            ax2.set_yticks(range(top_n))
            ax2.set_yticklabels(labels)
            ax2.set_xlabel('R²', fontsize=11)
            ax2.set_ylabel('ROI Window (nm)', fontsize=11)
            ax2.set_title('Top 10 ROI Candidates', fontsize=12, fontweight='bold')
            ax2.axvline(x=0.95, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlim(0, 1.05)
            ax2.invert_yaxis()
            
            plt.suptitle(f'{gas_name} - ROI Discovery', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_dir / 'plots' / 'roi_scan_results.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] ROI scan results: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate ROI heatmap: {e}")
    
    def _plot_comprehensive_diagnostic(self, canonical_spectra: Dict, wl_results: Dict, 
                                       abs_results: Dict, best_wl: str, best_abs: str,
                                       roi_range: Tuple, output_dir: Path, gas_name: str):
        """Generate comprehensive 6-panel diagnostic figure for publication."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig = plt.figure(figsize=(16, 12))
            
            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            concs = sorted(canonical_spectra.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
            
            # Panel A: Full spectrum overlay
            ax1 = fig.add_subplot(gs[0, 0])
            for i, conc in enumerate(concs):
                df = canonical_spectra[conc]
                if 'transmittance' in df.columns:
                    ax1.plot(df['wavelength'], df['transmittance'], 
                            color=colors[i], label=f'{conc} ppm', linewidth=1)
            ax1.axvspan(roi_range[0], roi_range[1], alpha=0.2, color='red')
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Transmittance')
            ax1.set_title('(A) Full Spectrum', fontweight='bold')
            ax1.legend(fontsize=8, loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Panel B: ROI zoom
            ax2 = fig.add_subplot(gs[0, 1])
            for i, conc in enumerate(concs):
                df = canonical_spectra[conc]
                mask = (df['wavelength'] >= roi_range[0]) & (df['wavelength'] <= roi_range[1])
                roi_df = df.loc[mask]
                if 'transmittance' in roi_df.columns and not roi_df.empty:
                    ax2.plot(roi_df['wavelength'], roi_df['transmittance'],
                            color=colors[i], label=f'{conc} ppm', linewidth=1.5)
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Transmittance')
            ax2.set_title(f'(B) ROI Detail ({roi_range[0]:.0f}-{roi_range[1]:.0f} nm)', fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Panel C: Absorbance in ROI
            ax3 = fig.add_subplot(gs[0, 2])
            for i, conc in enumerate(concs):
                df = canonical_spectra[conc]
                mask = (df['wavelength'] >= roi_range[0]) & (df['wavelength'] <= roi_range[1])
                roi_df = df.loc[mask]
                if 'absorbance' in roi_df.columns and not roi_df.empty:
                    ax3.plot(roi_df['wavelength'], roi_df['absorbance'],
                            color=colors[i], label=f'{conc} ppm', linewidth=1.5)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('Absorbance (AU)')
            ax3.set_title('(C) Absorbance in ROI', fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Panel D: Δλ Calibration
            ax4 = fig.add_subplot(gs[1, 0])
            if best_wl and 'error' not in wl_results.get(best_wl, {}):
                data = wl_results[best_wl]
                c = np.array(data['concentrations'])
                dl = np.array(data['delta_lambda'])
                ax4.scatter(c, dl, s=80, c='#2E86AB', edgecolors='black', zorder=5)
                x_fit = np.linspace(0, max(c) * 1.1, 100)
                y_fit = data['slope'] * x_fit + data['intercept']
                ax4.plot(x_fit, y_fit, 'r-', linewidth=2)
                ax4.set_xlabel('Concentration (ppm)')
                ax4.set_ylabel('Δλ (nm)')
                ax4.set_title(f'(D) Wavelength Shift (R²={data["r2"]:.4f})', fontweight='bold')
                ax4.grid(True, alpha=0.3)
            
            # Panel E: ΔA Calibration
            ax5 = fig.add_subplot(gs[1, 1])
            if best_abs and 'error' not in abs_results.get(best_abs, {}):
                data = abs_results[best_abs]
                c = np.array(data['concentrations'])
                da = np.array(data['delta_A'])
                ax5.scatter(c, da, s=80, c='#9BBB59', edgecolors='black', zorder=5)
                x_fit = np.linspace(0, max(c) * 1.1, 100)
                y_fit = data['slope'] * x_fit + data['intercept']
                ax5.plot(x_fit, y_fit, 'g-', linewidth=2)
                ax5.set_xlabel('Concentration (ppm)')
                ax5.set_ylabel('ΔA (AU)')
                ax5.set_title(f'(E) Absorbance Change (R²={data["r2"]:.4f})', fontweight='bold')
                ax5.grid(True, alpha=0.3)
            
            # Panel F: Sensitivity comparison
            ax6 = fig.add_subplot(gs[1, 2])
            methods = []
            sensitivities = []
            method_types = []
            
            for m, r in wl_results.items():
                if 'error' not in r:
                    methods.append(f'Δλ-{m}')
                    sensitivities.append(abs(r['slope']))
                    method_types.append('Δλ')
            for m, r in abs_results.items():
                if 'error' not in r:
                    methods.append(f'ΔA-{m}')
                    sensitivities.append(abs(r['slope']) * 1000)  # Scale for visibility
                    method_types.append('ΔA')
            
            if methods:
                colors_bar = ['#2E86AB' if t == 'Δλ' else '#9BBB59' for t in method_types]
                ax6.bar(range(len(methods)), sensitivities, color=colors_bar, edgecolor='black')
                ax6.set_xticks(range(len(methods)))
                ax6.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
                ax6.set_ylabel('Sensitivity')
                ax6.set_title('(F) Method Sensitivity Comparison', fontweight='bold')
            
            # Panel G: Peak wavelength tracking
            ax7 = fig.add_subplot(gs[2, 0])
            if best_wl and 'error' not in wl_results.get(best_wl, {}):
                data = wl_results[best_wl]
                c = np.array(data['concentrations'])
                peaks = np.array(data['peak_wavelengths'])
                ax7.plot(c, peaks, 'o-', color='#2E86AB', markersize=10, linewidth=2)
                ax7.set_xlabel('Concentration (ppm)')
                ax7.set_ylabel('Peak Wavelength (nm)')
                ax7.set_title('(G) Peak Wavelength vs Concentration', fontweight='bold')
                ax7.grid(True, alpha=0.3)
            
            # Panel H: Metrics summary table
            ax8 = fig.add_subplot(gs[2, 1:])
            ax8.axis('off')
            
            # Create metrics table
            if best_wl and 'error' not in wl_results.get(best_wl, {}):
                wl_data = wl_results[best_wl]
                table_data = [
                    ['Metric', 'Value', 'Unit'],
                    ['Sensitivity (Δλ)', f'{wl_data["slope"]:.4f}', 'nm/ppm'],
                    ['R²', f'{wl_data["r2"]:.4f}', '-'],
                    ['R²_CV (LOOCV)', f'{wl_data.get("r2_cv", "N/A")}', '-'],
                    ['LOD', f'{wl_data["lod_ppm"]:.2f}', 'ppm'],
                    ['LOQ', f'{wl_data["loq_ppm"]:.2f}', 'ppm'],
                    ['Spearman ρ', f'{wl_data["spearman_r"]:.4f}', '-'],
                    ['RMSE', f'{wl_data["rmse"]:.4f}', 'nm'],
                    ['ROI', f'{roi_range[0]:.0f}-{roi_range[1]:.0f}', 'nm'],
                ]
                
                table = ax8.table(cellText=table_data, loc='center', cellLoc='center',
                                 colWidths=[0.4, 0.3, 0.2])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1.2, 1.5)
                
                # Style header row
                for i in range(3):
                    table[(0, i)].set_facecolor('#2E86AB')
                    table[(0, i)].set_text_props(color='white', fontweight='bold')
                
                ax8.set_title('(H) Calibration Metrics Summary', fontweight='bold', pad=20)
            
            plt.suptitle(f'{gas_name} - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.98)
            
            plot_path = output_dir / 'plots' / 'comprehensive_diagnostic.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Comprehensive diagnostic: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate comprehensive diagnostic: {e}")
    
    def _plot_peak_tracking(self, results: Dict, best_method: str, output_dir: Path, gas_name: str):
        """Generate peak wavelength tracking plot."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            if not best_method or 'error' in results.get(best_method, {}):
                return
            
            data = results[best_method]
            concs = np.array(data['concentrations'])
            peaks = np.array(data['peak_wavelengths'])
            ref_wl = data['reference_wavelength']
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Panel 1: Absolute peak wavelength
            ax1 = axes[0]
            ax1.plot(concs, peaks, 'o-', color='#2E86AB', markersize=12, linewidth=2, 
                    markeredgecolor='black', markeredgewidth=1)
            ax1.axhline(y=ref_wl, color='gray', linestyle='--', alpha=0.5, 
                       label=f'Reference: {ref_wl:.2f} nm')
            ax1.set_xlabel('Concentration (ppm)', fontsize=12)
            ax1.set_ylabel('Peak Wavelength (nm)', fontsize=12)
            ax1.set_title('Absolute Peak Position', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Panel 2: Wavelength shift with error bars (if available)
            ax2 = axes[1]
            delta_lambda = peaks - ref_wl
            ax2.errorbar(concs, delta_lambda, fmt='o-', color='#2E86AB', markersize=12, 
                        linewidth=2, markeredgecolor='black', markeredgewidth=1,
                        capsize=5, capthick=1)
            
            # Add fit line
            slope = data['slope']
            intercept = data['intercept']
            x_fit = np.linspace(0, max(concs) * 1.1, 100)
            y_fit = slope * x_fit + intercept
            ax2.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7, 
                    label=f'Fit: Δλ = {slope:.4f}×C + {intercept:.4f}')
            
            ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax2.set_xlabel('Concentration (ppm)', fontsize=12)
            ax2.set_ylabel('Wavelength Shift Δλ (nm)', fontsize=12)
            ax2.set_title('Wavelength Shift vs Concentration', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'{gas_name} - Peak Wavelength Tracking', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_dir / 'plots' / 'peak_tracking.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Peak tracking: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate peak tracking plot: {e}")
    
    def _generate_summary_report(self, results: Dict, output_dir: Path, gas_name: str):
        """Generate summary.md report per CODEMAP."""
        try:
            best_wl = results.get('best_method_wavelength_shift')
            best_abs = results.get('best_method_absorbance')
            
            wl_cal = results.get('calibration_wavelength_shift', {}).get(best_wl, {}) if best_wl else {}
            abs_cal = results.get('calibration_absorbance', {}).get(best_abs, {}) if best_abs else {}
            bench = results.get('benchmark_comparison', {})
            
            report = f"""# {gas_name} Gas Sensing Analysis Report

## Pipeline: CODEMAP Aligned v1.0
Generated: {results.get('timestamp', 'N/A')}

---

## Summary

| Metric | Achieved | Paper Benchmark |
|--------|----------|-----------------|
| **Sensitivity (Δλ)** | {wl_cal.get('slope', 'N/A'):.4f} nm/ppm | {bench.get('paper_sensitivity', 'N/A')} nm/ppm |
| **R²** | {wl_cal.get('r2', 'N/A'):.4f} | ~0.95 |
| **LOD** | {wl_cal.get('lod_ppm', 'N/A'):.2f} ppm | {bench.get('paper_lod', 'N/A')} ppm |
| **Spearman ρ** | {wl_cal.get('spearman_r', 'N/A'):.4f} | >0.9 |
| **ROI** | {results.get('roi_range', ['N/A', 'N/A'])[0]:.0f}-{results.get('roi_range', ['N/A', 'N/A'])[1]:.0f} nm | 675-689 nm |

---

## Calibration Details

### Wavelength Shift (Δλ) Method
- **Best method**: {best_wl}
- **Equation**: Δλ = {wl_cal.get('slope', 0):.4f} × C + {wl_cal.get('intercept', 0):.4f}
- **95% CI for slope**: [{wl_cal.get('slope_ci_95', [0,0])[0]:.4f}, {wl_cal.get('slope_ci_95', [0,0])[1]:.4f}]
- **LOOCV R²**: {wl_cal.get('r2_cv', 'N/A')}

### Absorbance Amplitude (ΔA) Method
- **Best method**: {best_abs}
- **R²**: {abs_cal.get('r2', 'N/A')}
- **Sensitivity**: {abs_cal.get('slope', 'N/A')} AU/ppm

---

## Data Processing

- **Concentrations**: {results.get('concentrations', [])} ppm
- **ROI discovered**: {results.get('roi_range', [])} nm
- **Frame selection**: Response-peak based

---

## Output Files

### Metrics (JSON)
- `metrics/calibration_metrics.json` - Complete calibration results with all methods
- `metrics/absorbance_amplitude_calibration.json` - ΔA calibration results

### Plots (PNG, 300 DPI)
- `plots/calibration_curve.png` - Main Δλ vs concentration calibration
- `plots/absorbance_calibration.png` - ΔA vs concentration calibration
- `plots/spectral_overlay.png` - Full spectrum + ROI zoom
- `plots/method_comparison.png` - Δλ vs ΔA method comparison
- `plots/residual_analysis.png` - 4-panel residual diagnostics (Q-Q, histogram, etc.)
- `plots/roi_scan_results.png` - ROI discovery heatmap and top candidates
- `plots/peak_tracking.png` - Peak wavelength tracking vs concentration
- `plots/comprehensive_diagnostic.png` - 8-panel publication figure

### Data (CSV)
- `canonical_spectra/*.csv` - Averaged spectra per concentration

---

*Generated by Scientific Gas Sensing Pipeline (CODEMAP aligned)*
"""
            
            report_path = output_dir / 'reports' / 'summary.md'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"[OUTPUT] Summary report: {report_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate summary report: {e}")
    
    def _plot_calibration(self, results: Dict, best_method: str, output_dir: Path, gas_name: str):
        """Generate publication-quality calibration plot."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            if not best_method or 'error' in results.get(best_method, {}):
                return
            
            data = results[best_method]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            concs = np.array(data['concentrations'])
            delta_lambda = np.array(data['delta_lambda'])
            
            # Data points
            ax.scatter(concs, delta_lambda, s=80, c='#2E86AB', edgecolors='black', 
                      linewidths=1, zorder=5, label='Measured')
            
            # Fit line
            x_fit = np.linspace(0, max(concs) * 1.1, 100)
            y_fit = data['slope'] * x_fit + data['intercept']
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'Linear fit (R² = {data["r2"]:.4f})')
            
            # Confidence band
            ci_low, ci_high = data['slope_ci_95']
            y_low = ci_low * x_fit + data['intercept']
            y_high = ci_high * x_fit + data['intercept']
            ax.fill_between(x_fit, y_low, y_high, alpha=0.2, color='red', label='95% CI')
            
            # Labels and formatting
            ax.set_xlabel('Concentration (ppm)', fontsize=12)
            ax.set_ylabel('Wavelength Shift Δλ (nm)', fontsize=12)
            ax.set_title(f'{gas_name} Calibration Curve\nSensitivity: {data["slope"]:.4f} nm/ppm, LOD: {data["lod_ppm"]:.2f} ppm', 
                        fontsize=14, fontweight='bold')
            
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(concs) * 1.1)
            
            # Add equation annotation
            eq_text = f'Δλ = {data["slope"]:.4f}×C + {data["intercept"]:.4f}'
            ax.annotate(eq_text, xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plot_path = output_dir / 'plots' / 'calibration_curve.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Calibration plot: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate calibration plot: {e}")
    
    def _plot_spectral_overlay(self, spectra: Dict, roi_range: Tuple, output_dir: Path, gas_name: str):
        """Generate spectral overlay plot."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Color map
            concs = sorted(spectra.keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(concs)))
            
            # Full spectrum
            ax1 = axes[0]
            for i, conc in enumerate(concs):
                df = spectra[conc]
                if 'transmittance' in df.columns:
                    ax1.plot(df['wavelength'], df['transmittance'], 
                            color=colors[i], label=f'{conc} ppm', linewidth=1)
            
            ax1.axvspan(roi_range[0], roi_range[1], alpha=0.2, color='red', label='ROI')
            ax1.set_xlabel('Wavelength (nm)', fontsize=11)
            ax1.set_ylabel('Transmittance', fontsize=11)
            ax1.set_title('Full Spectrum', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # ROI zoom
            ax2 = axes[1]
            for i, conc in enumerate(concs):
                df = spectra[conc]
                mask = (df['wavelength'] >= roi_range[0]) & (df['wavelength'] <= roi_range[1])
                roi_df = df.loc[mask]
                if 'transmittance' in roi_df.columns and not roi_df.empty:
                    ax2.plot(roi_df['wavelength'], roi_df['transmittance'],
                            color=colors[i], label=f'{conc} ppm', linewidth=1.5)
            
            ax2.set_xlabel('Wavelength (nm)', fontsize=11)
            ax2.set_ylabel('Transmittance', fontsize=11)
            ax2.set_title(f'ROI Detail ({roi_range[0]:.0f}-{roi_range[1]:.0f} nm)', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'{gas_name} Spectral Response', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plot_path = output_dir / 'plots' / 'spectral_overlay.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OUTPUT] Spectral overlay: {plot_path}")
            
        except Exception as e:
            print(f"[WARNING] Failed to generate spectral overlay: {e}")


def run_pipeline(gas: str = 'Acetone', frames: int = 10, output: str = None, config: str = None):
    """Reusable entry point for programmatic execution."""
    # Define data paths
    data_paths = {
        'Acetone': ('Kevin_Data/Acetone', 'Kevin_Data/Acetone/air1.csv'),
        'Ethanol': ('Kevin_Data/Ethanol', 'Kevin_Data/Ethanol/air for ethanol ref.csv'),
        'Isopropanol': ('Kevin_Data/Isopropanol', 'Kevin_Data/Isopropanol/Air_ref for IPA.csv'),
        'Methanol': ('Kevin_Data/Methanol', 'Kevin_Data/Methanol/air ref after purging _N2.csv'),
        'Toluene': ('Kevin_Data/Toluene', 'Kevin_Data/Toluene/toluene_ref air.csv'),
        'Xylene': ('Kevin_Data/Xylene', 'Kevin_Data/Xylene/air ref xylene.csv'),
        'MixVOC': ('Kevin_Data/mix VOC', 'Kevin_Data/mix VOC/air.csv'),
    }

    if gas not in data_paths:
        raise ValueError(f"Unknown gas '{gas}'")

    data_dir, ref_path = data_paths[gas]
    data_dir = PROJECT_ROOT / data_dir
    ref_path = PROJECT_ROOT / ref_path

    output_dir = output if output else f'output/scientific/{gas}'
    output_dir = PROJECT_ROOT / output_dir

    pipeline = ScientificGasPipeline(config)
    results = pipeline.run_analysis(
        data_dir=data_dir,
        ref_path=ref_path,
        output_dir=output_dir,
        gas_name=gas,
        n_frames_select=frames
    )
    return results, output_dir


def main(gas=None, frames=10, output=None, config=None):
    if gas is None:
        parser = argparse.ArgumentParser(
            description='Scientific Gas Sensing Analysis Pipeline',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python run_scientific_pipeline.py --gas Acetone
  python run_scientific_pipeline.py --gas Ethanol --frames 15
  python run_scientific_pipeline.py --gas Acetone --output custom_output
            """
        )
        
        parser.add_argument('--gas', type=str, default='Acetone',
                           choices=['Acetone', 'Ethanol', 'Isopropanol', 'Methanol', 'Toluene', 'Xylene', 'MixVOC'],
                           help='Gas to analyze (default: Acetone)')
        parser.add_argument('--frames', type=int, default=10,
                           help='Number of response-peak frames to select per trial (default: 10)')
        parser.add_argument('--output', type=str, default=None,
                           help='Output directory (default: output/{gas}_scientific)')
        parser.add_argument('--config', type=str, default=None,
                           help='Path to custom config.yaml')

        args = parser.parse_args()
        gas = args.gas
        frames = args.frames
        output = args.output
        config = args.config

    results, output_dir = run_pipeline(
        gas=gas,
        frames=frames,
        output=output,
        config=config
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    best = results.get('best_method')
    if best and best in results.get('calibration', {}):
        cal = results['calibration'][best]
        bench = results.get('benchmark_comparison', {})
        
        print(f"Gas: {gas}")
        print(f"Best method: {best}")
        print(f"Sensitivity: {cal['slope']:.4f} nm/ppm (paper: {bench.get('paper_sensitivity', 'N/A')} nm/ppm)")
        print(f"R²: {cal['r2']:.4f}")
        print(f"LOD: {cal['lod_ppm']:.2f} ppm (paper: {bench.get('paper_lod', 'N/A')} ppm)")
        print(f"Spearman ρ: {cal['spearman_r']:.4f}")
        print(f"\nOutputs: {output_dir}")

    return results


if __name__ == '__main__':
    main()
