import os
import re
import json
import math
import yaml
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Set

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, correlate
from scipy.optimize import curve_fit, nnls
from scipy.stats import linregress, t, probplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, LinearRegression
import scipy as sp
import sklearn as sk

from config.config_loader import load_config
CONFIG = load_config()
REPO_ROOT = Path(__file__).resolve().parents[2]

from .preprocessing import (
    estimate_noise_metrics,
    baseline_correction,
    smooth_spectrum,
    normalize_spectrum,
    downsample_spectrum,
    detect_outliers,
)
from ..advanced.deconvolution_ica import (
    fit_ica_from_canonical,
    save_ica_outputs,
)
from ..advanced.mcr_als import (
    fit_mcrals_from_canonical,
    save_mcrals_outputs,
)


def _pelt_changepoint_detection(signal: np.ndarray, penalty: float = 3.0, min_size: int = 5) -> List[int]:
    """
    PELT (Pruned Exact Linear Time) change-point detection.
    Returns list of change-point indices where signal statistics change.
    """
    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0.0)
    n = len(signal)
    if n < 2 * min_size:
        return []
    
    # Cost function: sum of squared residuals from mean
    def cost(start: int, end: int) -> float:
        if end <= start:
            return 0.0
        segment = signal[start:end]
        if len(segment) == 0:
            return 0.0
        mean_val = float(np.mean(segment))
        return float(np.sum((segment - mean_val) ** 2))
    
    # PELT algorithm
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    R = [0]
    cp = {}
    
    for t in range(min_size, n + 1):
        candidates = []
        for s in R:
            if t - s >= min_size:
                val = F[s] + cost(s, t) + penalty
                candidates.append((val, s))
        
        if candidates:
            best_val, best_s = min(candidates)
            F[t] = best_val
            cp[t] = best_s
            # Prune: keep only indices s where F[s] + cost(s,t) <= F[t]
            R = [s for s in R if F[s] + cost(s, t) <= F[t]]
            R.append(t)
    
    # Backtrack to get change-points
    changepoints = []
    t = n
    while t > 0 and t in cp:
        s = cp[t]
        if s > 0:
            changepoints.append(s)
        t = s
    
    return sorted(set(changepoints))


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _write_json(path: Path, payload: Dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _autosave_env_coefficients_to_config(est: Dict[str, object]) -> bool:
    """Persist estimated environment coefficients into config/config.yaml if enabled.

    Returns True if write succeeded, else False.
    """
    try:
        cfg_path = REPO_ROOT / 'config' / 'config.yaml'
        if not cfg_path.exists():
            return False
        with cfg_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        env = cfg.get('environment', {}) if isinstance(cfg.get('environment', {}), dict) else {}
        if not env.get('autosave_coefficients', False):
            return False
        coeffs = env.get('coefficients', {}) if isinstance(env.get('coefficients', {}), dict) else {}
        t = est.get('temperature', None)
        h = est.get('humidity', None)
        if t is not None:
            coeffs['temperature'] = float(t)
        if h is not None:
            coeffs['humidity'] = float(h)
        env['coefficients'] = coeffs
        cfg['environment'] = env
        with cfg_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return True
    except Exception:
        return False


# ----------------------
# Multivariate calibration (PLSR)
# ----------------------

def _build_feature_matrix_from_canonical(canonical: Dict[float, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X (spectral matrix) and y (concentrations) from canonical dict.

    Returns (X, y, wavelengths). X has shape (n_samples, n_wavelengths).
    """
    if not canonical:
        raise ValueError("No canonical spectra provided for PLSR")

    items = sorted(canonical.items(), key=lambda kv: kv[0])
    concs = np.array([float(k) for k, _ in items], dtype=float)

    # Use the first spectrum as base grid
    base_wl = items[0][1]['wavelength'].to_numpy()
    sig_col = _signal_column(items[0][1])

    X_list: List[np.ndarray] = []
    for _, df in items:
        wl = df['wavelength'].to_numpy()
        sig_col_df = _signal_column(df)
        ysig = df[sig_col_df].to_numpy()
        if not np.array_equal(wl, base_wl):
            ysig = np.interp(base_wl, wl, ysig)
        X_list.append(ysig)

    X = np.vstack(X_list)
    return X, concs, base_wl


def _fit_plsr_calibration(canonical: Dict[float, pd.DataFrame],
                          cfg: Dict[str, object],
                          matrix_cache: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> Optional[Dict[str, object]]:
    """Fit PLSR to predict concentration from full spectrum, with simple CV model selection."""
    try:
        if matrix_cache is None:
            X, y, wl = _build_feature_matrix_from_canonical(canonical)
        else:
            X, y, wl = matrix_cache
    except ValueError:
        return None

    # Optional absorbance transform (useful if input is transmittance)
    if bool(cfg.get('absorbance', False)):
        # Only convert to absorbance if inputs look like transmittance (bounded in 0-1 range)
        # to avoid double -log10 when the canonical spectra are already absorbance.
        x_max = np.nanmax(X)
        x_min = np.nanmin(X)
        if np.isfinite(x_max) and np.isfinite(x_min) and x_min >= 0.0 and x_max <= 1.2:
            with np.errstate(divide='ignore', invalid='ignore'):
                X = -np.log10(np.clip(X, 1e-6, None))

    # Optional wavelength limits
    wl_min = cfg.get('wl_min', None)
    wl_max = cfg.get('wl_max', None)
    if wl_min is not None or wl_max is not None:
        mask = np.ones_like(wl, dtype=bool)
        if wl_min is not None:
            mask &= wl >= float(wl_min)
        if wl_max is not None:
            mask &= wl <= float(wl_max)
        if np.any(mask):
            X = X[:, mask]
            wl = wl[mask]

    # Optional feature preprocessing
    prep = str(cfg.get('feature_prep', 'raw')).lower()
    if prep in ('derivative', 'first_derivative') or 'derivative' in prep:
        X = np.gradient(X, wl, axis=1)
    if prep == 'snv' or 'snv' in prep:
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        X = (X - mu) / sd
    elif prep == 'mean_center':
        mu = X.mean(axis=0, keepdims=True)
        X = X - mu

    n_samples, n_features = X.shape
    if n_samples < 3:
        return None

    max_comp_cfg = int(cfg.get('max_components', 5))
    max_components = max(1, min(max_comp_cfg, n_samples - 1))
    best_cv_r2 = float('-inf')
    best_cv_rmse = float('inf')
    best_n = 1
    preds_cv_best = None
    r2_cv_curve: List[float] = []
    rmse_cv_curve: List[float] = []
    r2_in_curve: List[float] = []

    # Leave-one-out CV over components 1..max_components
    for n_comp in range(1, max_components + 1):
        cv_preds = np.zeros_like(y)
        for i in range(n_samples):
            mask = np.ones(n_samples, dtype=bool)
            mask[i] = False
            pls = PLSRegression(n_components=n_comp, scale=bool(cfg.get('scale', True)))
            pls.fit(X[mask], y[mask])
            cv_preds[i] = float(pls.predict(X[i:i+1]).ravel()[0])
        cv_r2 = float(r2_score(y, cv_preds))
        cv_rmse = float(np.sqrt(mean_squared_error(y, cv_preds)))
        r2_cv_curve.append(cv_r2)
        rmse_cv_curve.append(cv_rmse)
        # In-sample fit for this component count (diagnostic only)
        pls_full = PLSRegression(n_components=n_comp, scale=bool(cfg.get('scale', True)))
        pls_full.fit(X, y)
        preds_full = pls_full.predict(X).ravel()
        r2_in_n = float(r2_score(y, preds_full)) if np.isfinite(np.var(y)) else float('nan')
        r2_in_curve.append(r2_in_n)
        if np.isfinite(cv_r2) and cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_cv_rmse = cv_rmse
            best_n = n_comp
            preds_cv_best = cv_preds.copy()

    # Fit final model on all data with selected components
    pls_final = PLSRegression(n_components=best_n, scale=bool(cfg.get('scale', True)))
    pls_final.fit(X, y)
    preds_in = pls_final.predict(X).ravel()
    in_r2 = float(r2_score(y, preds_in)) if np.isfinite(np.var(y)) else float('nan')
    in_rmse = float(np.sqrt(mean_squared_error(y, preds_in)))

    # Optional coefficient-based wavelength selection to refine model
    selected_mask = None
    fs_cfg = cfg.get('feature_selection', {}) if isinstance(cfg, dict) else {}
    if fs_cfg.get('enabled', False):
        method = str(fs_cfg.get('method', 'coef_top_fraction')).lower()
        if method == 'coef_top_fraction':
            frac = float(fs_cfg.get('top_fraction', 0.2))
            frac = float(np.clip(frac, 0.01, 0.8))
            coef = np.abs(pls_final.coef_.ravel())
            if coef.size > 0 and np.any(coef > 0):
                k = max(1, int(round(frac * coef.size)))
                top_idx = np.argpartition(coef, -k)[-k:]
                selected_mask = np.zeros_like(coef, dtype=bool)
                selected_mask[top_idx] = True
        # Refit if we have a selection
        if selected_mask is not None and np.any(selected_mask):
            X_sel = X[:, selected_mask]
            # Recompute CV on selected subset
            cv_preds_sel = np.zeros_like(y)
            for i in range(n_samples):
                mask = np.ones(n_samples, dtype=bool)
                mask[i] = False
                pls_sel = PLSRegression(n_components=min(best_n, X_sel.shape[1]), scale=bool(cfg.get('scale', True)))
                pls_sel.fit(X_sel[mask], y[mask])
                cv_preds_sel[i] = float(pls_sel.predict(X_sel[i:i+1]).ravel()[0])
            cv_r2_sel = float(r2_score(y, cv_preds_sel))
            cv_rmse_sel = float(np.sqrt(mean_squared_error(y, cv_preds_sel)))
            # If improved, replace final model artifacts to reflect selection
            if np.isfinite(cv_r2_sel) and cv_r2_sel > best_cv_r2:
                best_cv_r2 = cv_r2_sel
                best_cv_rmse = cv_rmse_sel
                # Fit final selected model for coefficients export
                pls_final = PLSRegression(n_components=min(best_n, X_sel.shape[1]), scale=bool(cfg.get('scale', True)))
                pls_final.fit(X_sel, y)
                preds_in = pls_final.predict(X_sel).ravel()
                in_r2 = float(r2_score(y, preds_in)) if np.isfinite(np.var(y)) else float('nan')
                in_rmse = float(np.sqrt(mean_squared_error(y, preds_in)))
                # Update wavelength list to selected subset for export
                wl = wl[selected_mask]

    result: Dict[str, object] = {
        'model': 'plsr',
        'n_components': int(best_n),
        'wavelengths': wl.tolist(),
        'concentrations': y.tolist(),
        'predictions_in': preds_in.tolist(),
        'r2_in': in_r2,
        'rmse_in': in_rmse,
        'predictions_cv': preds_cv_best.tolist() if preds_cv_best is not None else None,
        'r2_cv': best_cv_r2,
        'rmse_cv': best_cv_rmse,
        'r2_cv_curve': r2_cv_curve,
        'rmse_cv_curve': rmse_cv_curve,
        'r2_in_curve': r2_in_curve,
        'n_components_candidates': list(range(1, max_components + 1)),
        'coef_': pls_final.coef_.ravel().tolist(),
    }

    # Bootstrap uncertainty on predictions (and derived metrics) if requested.
    bootstrap_cfg = cfg.get('bootstrap') if isinstance(cfg, dict) else None
    if not bootstrap_cfg:
        bootstrap_cfg = (CONFIG.get('calibration', {}) if isinstance(CONFIG, dict) else {}).get('bootstrap', {})

    bootstrap_summary: Optional[Dict[str, object]] = None
    if bootstrap_cfg and bootstrap_cfg.get('enabled', False) and n_samples >= 3:
        rng = np.random.default_rng(bootstrap_cfg.get('random_seed', None))
        iterations = int(max(1, bootstrap_cfg.get('iterations', 500)))
        sample_fraction = float(np.clip(bootstrap_cfg.get('sample_fraction', 0.8), 0.1, 1.0))
        min_unique = int(max(2, bootstrap_cfg.get('min_unique', 2)))

        pred_samples: List[np.ndarray] = []
        r2_samples: List[float] = []
        rmse_samples: List[float] = []

        draw_size = max(min_unique, int(round(sample_fraction * n_samples)))
        for _ in range(iterations):
            idx = rng.integers(0, n_samples, size=draw_size)
            if np.unique(idx).size < min_unique:
                continue
            try:
                pls_bs = PLSRegression(n_components=min(best_n, draw_size - 1), scale=bool(cfg.get('scale', True)))
                pls_bs.fit(X[idx], y[idx])
                preds_bs = pls_bs.predict(X).ravel()
            except Exception:
                continue
            pred_samples.append(preds_bs)
            try:
                r2_samples.append(float(r2_score(y, preds_bs)))
            except Exception:
                r2_samples.append(float('nan'))
            rmse_samples.append(float(np.sqrt(mean_squared_error(y, preds_bs))))

        if pred_samples:
            preds_arr = np.vstack(pred_samples)
            pred_lo = np.percentile(preds_arr, 2.5, axis=0)
            pred_hi = np.percentile(preds_arr, 97.5, axis=0)
            pred_med = np.median(preds_arr, axis=0)

            def _finite_stats(values: List[float]) -> Dict[str, float]:
                arr = np.array(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return {'mean': float('nan'), 'std': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan')}
                return {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    'ci_low': float(np.percentile(arr, 2.5)),
                    'ci_high': float(np.percentile(arr, 97.5)),
                }

            bootstrap_summary = {
                'iterations': len(pred_samples),
                'r2': _finite_stats(r2_samples),
                'rmse': _finite_stats(rmse_samples),
                'predictions': {
                    'ci_low': pred_lo.tolist(),
                    'ci_high': pred_hi.tolist(),
                    'median': pred_med.tolist(),
                    'concentrations': y.tolist(),
                },
            }

    result['bootstrap'] = bootstrap_summary

    return result


def _preprocess_dataframe(df: pd.DataFrame, *, stage: str) -> pd.DataFrame:
    settings = CONFIG.get('preprocessing', {})
    if not settings.get('enabled', False):
        return df

    stage_norm = stage.lower()
    if stage_norm == 'frame' and not settings.get('apply_to_frames', False):
        return df
    if stage_norm == 'transmittance' and not settings.get('apply_to_transmittance', True):
        return df

    if 'wavelength' not in df.columns:
        return df

    preferred_col = None
    if stage_norm == 'transmittance' and 'transmittance' in df.columns:
        preferred_col = 'transmittance'
    elif 'intensity' in df.columns:
        preferred_col = 'intensity'
    else:
        other_cols = [c for c in df.columns if c != 'wavelength']
        if other_cols:
            preferred_col = other_cols[0]
        else:
            return df

    is_transmittance = preferred_col == 'transmittance'

    wl = df['wavelength'].to_numpy(copy=True)
    signal = df[preferred_col].to_numpy(copy=True)

    smooth_cfg = settings.get('smooth', {})
    if smooth_cfg.get('enabled', False):
        window = smooth_cfg.get('window', 21)
        poly_order = smooth_cfg.get('poly_order', 3)
        method = smooth_cfg.get('method', 'savgol')
        signal = smooth_spectrum(signal, window=window, poly_order=poly_order, method=method)
        if smooth_cfg.get('extra_pass', False):
            signal = smooth_spectrum(signal, window=window, poly_order=poly_order, method=method)

    if not is_transmittance:
        baseline_cfg = settings.get('baseline', {})
        if baseline_cfg.get('enabled', False):
            method = baseline_cfg.get('method', 'polynomial')
            order = baseline_cfg.get('order', 2)
            signal = baseline_correction(wl, signal, method=method, poly_order=order)

    norm_cfg = settings.get('normalization', {})
    if norm_cfg.get('enabled', False):
        signal = normalize_spectrum(signal, method=norm_cfg.get('method', 'minmax'))

    # Optional environment compensation (additive offset in measurement units)
    try:
        env_cfg = CONFIG.get('environment', {})
        if env_cfg.get('enabled', False):
            apply_frames = bool(env_cfg.get('apply_to_frames', False))
            apply_trans = bool(env_cfg.get('apply_to_transmittance', True))
            if (stage_norm == 'frame' and apply_frames) or (stage_norm == 'transmittance' and apply_trans):
                ref = env_cfg.get('reference', {}) if isinstance(env_cfg.get('reference', {}), dict) else {}
                coeffs = env_cfg.get('coefficients', {}) if isinstance(env_cfg.get('coefficients', {}), dict) else {}
                override = env_cfg.get('override', {}) if isinstance(env_cfg.get('override', {}), dict) else {}

                # Determine environment values from DataFrame columns or overrides
                T_ref = float(ref.get('temperature', 25.0))
                H_ref = float(ref.get('humidity', 50.0))
                T_val = None
                H_val = None
                if 'temperature' in df.columns:
                    try:
                        T_val = float(pd.to_numeric(df['temperature'], errors='coerce').dropna().mean())
                    except Exception:
                        T_val = None
                if 'humidity' in df.columns:
                    try:
                        H_val = float(pd.to_numeric(df['humidity'], errors='coerce').dropna().mean())
                    except Exception:
                        H_val = None
                if T_val is None and override.get('temperature') is not None:
                    T_val = float(override.get('temperature'))
                if H_val is None and override.get('humidity') is not None:
                    H_val = float(override.get('humidity'))

                # Apply additive offset using provided coefficients
                offset = 0.0
                cT = coeffs.get('temperature', None)
                cH = coeffs.get('humidity', None)
                if cT is not None and T_val is not None:
                    offset += float(cT) * (T_val - T_ref)
                if cH is not None and H_val is not None:
                    offset += float(cH) * (H_val - H_ref)
                if offset != 0.0 and np.isfinite(offset):
                    signal = signal - float(offset)
    except Exception:
        pass

    ds_cfg = settings.get('downsample', {})
    new_wl = wl
    if ds_cfg.get('enabled', False):
        new_wl, signal = downsample_spectrum(
            wl,
            signal,
            factor=ds_cfg.get('factor'),
            target_points=ds_cfg.get('target_points'),
            method='average',
        )

    out_df = df.copy()
    if len(new_wl) != len(wl):
        out_df = pd.DataFrame({'wavelength': new_wl})
        for col in df.columns:
            if col == 'wavelength':
                continue
            source = df[col].to_numpy(copy=True)
            if len(source) != len(wl):
                continue
            out_df[col] = np.interp(new_wl, wl, source)
    out_df[preferred_col] = signal

    return out_df


def _record_outlier(metadata: Dict[str, object], conc: float, trial: str):
    metadata.setdefault('dropped_outliers', []).append({'concentration': conc, 'trial': trial})


def _save_run_metadata(out_root: str, metadata: Dict[str, object]) -> Path:
    metrics_dir = Path(out_root) / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_path = metrics_dir / 'run_metadata.json'
    _write_json(meta_path, metadata)
    return meta_path


def _archive_run(out_root: str, metadata: Dict[str, object]):
    archive_cfg = CONFIG.get('archiving', {})
    if not archive_cfg.get('enabled', False):
        return None

    out_path = Path(out_root)
    archive_dir = out_path / 'archives' / metadata['run_timestamp']
    archive_dir.mkdir(parents=True, exist_ok=True)

    for rel in ['metrics', 'plots', 'dynamics', 'reports', 'aggregated', 'stable_selected']:
        src = out_path / rel
        if src.exists():
            dst = archive_dir / rel
            _copy_tree(src, dst)

    for extra_file in ['run_metadata.json']:
        src = out_path / 'metrics' / extra_file
        if src.exists():
            shutil.copy2(src, archive_dir / extra_file)

    return archive_dir


def _invoke_report_generation(out_root: str, metadata: Dict[str, object]) -> Dict[str, object]:
    reporting = CONFIG.get('reporting', {})
    results: Dict[str, object] = {}
    if not reporting:
        return results

    run_dir = Path(out_root)
    reports_dir = run_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_script = REPO_ROOT / 'scripts' / 'generate_report.py'
    notebook_path = reports_dir / 'analysis_report.ipynb'

    if reporting.get('generate_notebook', True):
        if report_script.exists():
            cmd = [sys.executable, str(report_script), '--run', str(run_dir)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                results['notebook'] = str(notebook_path)
            except subprocess.CalledProcessError as exc:
                results['notebook_error'] = exc.stderr.decode('utf-8', errors='ignore')
            else:
                try:
                    summary_md = (reports_dir / 'summary.md')
                    md_text = summary_md.read_text(encoding='utf-8') if summary_md.exists() else '# Gas Analysis Report\n\nSummary not found.'
                    cells = []
                    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["# Gas Analysis Report\n", "\n", "This notebook aggregates key figures and metrics for peer review.\n"]})
                    cells.append({"cell_type": "markdown", "metadata": {}, "source": md_text.splitlines(True)})
                    # Embed key figures if they exist
                    try:
                        plots_dir = run_dir / 'plots'
                        def _rel(p):
                            try:
                                return os.path.relpath(str(p), start=str(reports_dir))
                            except Exception:
                                return str(p)
                        figs = [
                            ('## Concentration Response', plots_dir / 'concentration_response.png'),
                            ('## ROI Repeatability', plots_dir / 'roi_repeatability.png'),
                            ('## Multivariate CV R² Comparison', plots_dir / 'multivariate_cv_r2.png'),
                            ('## Selected Model: Predicted vs Actual', plots_dir / 'selected_pred_vs_actual.png'),
                            ('## MCR-ALS Components', plots_dir / 'mcr_als_components.png'),
                            ('## MCR-ALS Pred vs Actual', plots_dir / 'mcr_als_pred_vs_actual.png'),
                        ]
                        for title, path in figs:
                            if path.exists():
                                cells.append({"cell_type": "markdown", "metadata": {}, "source": [title + "\n"]})
                                cells.append({"cell_type": "markdown", "metadata": {}, "source": [f"![]({_rel(path)})\n"]})
                    except Exception:
                        pass
                    nb = {"cells": cells, "metadata": {"language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
                    with notebook_path.open('w', encoding='utf-8') as f:
                        json.dump(nb, f, indent=2)
                    results['notebook'] = str(notebook_path)
                except Exception as exc:
                    results['notebook_error'] = str(exc)

    if reporting.get('export_pdf', False) and notebook_path.exists():
        nbconvert_exe = reporting.get('nbconvert_executable', 'jupyter')
        cmd = [nbconvert_exe, 'nbconvert', '--to', 'pdf', str(notebook_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results['pdf'] = str(notebook_path.with_suffix('.pdf'))
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            results['pdf_error'] = str(exc)

    return results


def _gather_trend_records(run_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    archives_root = run_root / 'archives'
    if archives_root.exists():
        for arch in sorted(archives_root.iterdir()):
            perf_path = arch / 'metrics' / 'roi_performance.json'
            meta_path = arch / 'run_metadata.json'
            if perf_path.exists():
                try:
                    performance = json.loads(perf_path.read_text())
                    timestamp = arch.name
                    if meta_path.exists():
                        meta = json.loads(meta_path.read_text())
                        timestamp = meta.get('run_timestamp', timestamp)
                    records.append({'timestamp': timestamp, 'performance': performance})
                except json.JSONDecodeError:
                    continue

    current_perf_path = run_root / 'metrics' / 'roi_performance.json'
    current_meta_path = run_root / 'metrics' / 'run_metadata.json'
    if current_perf_path.exists():
        try:
            performance = json.loads(current_perf_path.read_text())
            timestamp = _timestamp()
            if current_meta_path.exists():
                meta = json.loads(current_meta_path.read_text())
                timestamp = meta.get('run_timestamp', timestamp)
            records.append({'timestamp': timestamp, 'performance': performance})
        except json.JSONDecodeError:
            pass

    unique = {rec['timestamp']: rec for rec in records}
    return [unique[k] for k in sorted(unique.keys())]


def generate_trend_plots(out_root: str) -> Dict[str, str]:
    reporting = CONFIG.get('reporting', {})
    if not reporting.get('trend_plots', True):
        return {}

    run_root = Path(out_root)
    records = _gather_trend_records(run_root)
    if len(records) < 2:
        return {}

    timestamps = []
    slopes = []
    r2_vals = []
    lod_vals = []

    for rec in records:
        perf = rec['performance']
        timestamps.append(rec['timestamp'])
        slopes.append(perf.get('regression_slope'))
        r2_vals.append(perf.get('regression_r2'))
        lod_vals.append(perf.get('lod_ppm'))

    plots_dir = run_root / 'plots' / 'trends'
    plots_dir.mkdir(parents=True, exist_ok=True)
    trend_path = plots_dir / 'roi_performance_trend.png'

    plt.figure(figsize=(10, 5))
    x = range(len(timestamps))
    plt.plot(x, slopes, marker='o', label='Slope (dT/ppm)')
    plt.plot(x, r2_vals, marker='s', label='R²')
    plt.plot(x, lod_vals, marker='^', label='LOD (ppm)')
    plt.xticks(x, timestamps, rotation=45, ha='right')
    plt.ylabel('Metric Value')
    plt.title('ROI Performance Trend Across Runs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(trend_path, dpi=300)
    plt.close()

    return {'roi_performance': str(trend_path)}

# IO and utilities
# ----------------------

def _read_csv_spectrum(path: str) -> pd.DataFrame:
    """Read a spectrum CSV and normalize columns to wavelength,intensity.
    Accepts headerless files as two columns.
    """
    try:
        df = pd.read_csv(path)
        if 'wavelength' not in df.columns or 'intensity' not in df.columns:
            df = pd.read_csv(path, header=None, names=['wavelength', 'intensity'])
        df = df[['wavelength', 'intensity']].copy()
        df['wavelength'] = pd.to_numeric(df['wavelength'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        df = df.dropna()
        return df.sort_values('wavelength').reset_index(drop=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read spectrum {path}: {e}")


def _sort_frame_paths(paths: Sequence[str]) -> List[str]:
    def _key(p: str) -> Tuple[int, float, float]:
        name = os.path.basename(p)

        # Prefer explicit timestamp in names like: t1_20241029_11h25m41s826ms
        # trial = t1, date = 20241029, time = 11:25:41.826
        ts_match = re.search(
            r"t(?P<trial>\d+)[^0-9]*(?P<date>\d{8})_(?P<hour>\d{1,2})h(?P<minute>\d{1,2})m(?P<second>\d{1,2})s(?P<msec>\d{1,3})ms",
            name,
        )
        if ts_match:
            try:
                date_str = ts_match.group('date')  # YYYYMMDD
                hour = int(ts_match.group('hour'))
                minute = int(ts_match.group('minute'))
                second = int(ts_match.group('second'))
                msec = int(ts_match.group('msec'))

                date_int = int(date_str)
                time_key = ((hour * 3600 + minute * 60 + second) * 1000.0) + float(msec)
                mtime = os.path.getmtime(p)
                return date_int, time_key, mtime
            except Exception:
                # Fall back to generic numeric+mtime ordering
                pass

        # Fallback: original behavior based on last numeric token and modification time
        digits = re.findall(r'(\d+)', name)
        if digits:
            try:
                idx = int(digits[-1])
            except ValueError:
                idx = math.inf
        else:
            idx = math.inf
        mtime = os.path.getmtime(p)
        return int(idx if math.isfinite(idx) else math.inf), float(idx if math.isfinite(idx) else math.inf), mtime

    return sorted(paths, key=_key)


def _ensure_response_signal(df: pd.DataFrame, ref_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    updated = df
    has_trans = 'transmittance' in df.columns
    if not has_trans and ref_df is not None and 'intensity' in df.columns:
        updated = compute_transmittance(df, ref_df)
    if 'absorbance' not in updated.columns:
        updated = _append_absorbance_column(updated)
    return updated


def _smooth_vector(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1 or values.size == 0:
        return values.astype(float)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode='same')


def _detect_segments_above_threshold(values: np.ndarray,
                                     high_threshold: float,
                                     low_threshold: float,
                                     min_length: int,
                                     pad: int) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    if values.size == 0:
        return segments

    min_length = max(1, int(min_length))
    pad = max(0, int(pad))
    current_start: Optional[int] = None
    below_counter = 0

    for idx, val in enumerate(values):
        if current_start is None:
            if np.isfinite(val) and val >= high_threshold:
                current_start = idx
                below_counter = 0
        else:
            if np.isfinite(val) and val >= low_threshold:
                below_counter = 0
            else:
                below_counter += 1
                if below_counter > 0:
                    end_idx = idx - below_counter
                    if end_idx < current_start:
                        end_idx = idx - 1
                    segment_length = (end_idx - current_start) + 1
                    if segment_length >= min_length:
                        seg_start = max(0, current_start - pad)
                        seg_end = min(values.size - 1, end_idx + pad)
                        segments.append((seg_start, seg_end))
                    current_start = None
                    below_counter = 0

    if current_start is not None:
        seg_end = values.size - 1
        segment_length = (seg_end - current_start) + 1
        if segment_length >= min_length:
            segments.append((max(0, current_start - pad), seg_end))

    # Merge overlapping or adjacent segments
    if not segments:
        return segments
    segments.sort()
    merged: List[Tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _compute_response_time_series(frames: Sequence[pd.DataFrame],
                                  ref_df: Optional[pd.DataFrame],
                                  *,
                                  dataset_label: Optional[str],
                                  response_cfg: Dict[str, object]) -> Tuple[pd.DataFrame, List[int], List[int]]:
    if not frames:
        return None, [], []

    processed = [_ensure_response_signal(df, ref_df) for df in frames]
    absorb_col = 'absorbance' if 'absorbance' in processed[0].columns else _signal_column(processed[0])

    base_wl = processed[0]['wavelength'].to_numpy()
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = float(base_wl.min())
    if max_wl_roi is None:
        max_wl_roi = float(base_wl.max())
    roi_mask = (base_wl >= min_wl_roi) & (base_wl <= max_wl_roi)
    if not np.any(roi_mask):
        roi_mask = np.ones_like(base_wl, dtype=bool)
    roi_wavelengths = base_wl[roi_mask]

    absorb_matrix = []
    mean_absorb = []
    roi_matrix = []
    for df in processed:
        wl = df['wavelength'].to_numpy()
        signal = df[absorb_col].to_numpy(dtype=float)
        if not np.array_equal(wl, base_wl):
            signal = np.interp(base_wl, wl, signal)
        absorb_matrix.append(signal)
        mean_absorb.append(float(np.nanmean(signal)))
        roi_matrix.append(signal[roi_mask])
    absorb_matrix = np.vstack(absorb_matrix)
    mean_absorb = np.array(mean_absorb, dtype=float)
    roi_matrix = np.vstack(roi_matrix)

    smooth_window = int(response_cfg.get('smooth_window', 5) or 5)
    if smooth_window > 1:
        window = _ensure_odd_window(smooth_window)
        absorb_matrix = savgol_filter(absorb_matrix, window_length=min(window, max(3, absorb_matrix.shape[1] - (absorb_matrix.shape[1] + 1) % 2)), polyorder=2, axis=1, mode='nearest')
        roi_matrix = savgol_filter(roi_matrix, window_length=min(window, max(3, roi_matrix.shape[1] - (roi_matrix.shape[1] + 1) % 2)), polyorder=2, axis=1, mode='nearest')

    baseline_target = int(response_cfg.get('baseline_frames', 12) or 12)
    baseline_target = max(1, min(baseline_target, absorb_matrix.shape[0]))
    baseline_indices = list(range(min(baseline_target, len(frames))))
    baseline_indices = [idx for idx in baseline_indices if 0 <= idx < len(frames)]
    if not baseline_indices:
        baseline_indices = [0]

    def _compute_baseline_outputs(indices: List[int]) -> Tuple[
        np.ndarray,
        np.ndarray,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        float,
    ]:
        valid = [idx for idx in indices if 0 <= idx < len(frames)]
        if not valid:
            valid = [0]
        base_matrix = absorb_matrix[valid]
        base_roi_matrix = roi_matrix[valid]
        baseline_ref = np.nanmean(base_matrix, axis=0)
        roi_baseline_ref = np.nanmean(base_roi_matrix, axis=0)
        baseline_mean_val = float(np.nanmean(mean_absorb[valid]))
        baseline_std_val = float(np.nanstd(mean_absorb[valid], ddof=1)) if len(valid) > 1 else 0.0
        baseline_std_val = float(np.nan_to_num(baseline_std_val, nan=0.0))
        centered = absorb_matrix - baseline_ref
        delta_mean_local = mean_absorb - baseline_mean_val
        roi_delta_local = roi_matrix - roi_baseline_ref
        # Determine whether the ROI extremum is a valley (default) or peak based on baseline reference
        extremum_mode = 'valley'
        if np.any(np.isfinite(roi_baseline_ref)):
            roi_finite = roi_baseline_ref[np.isfinite(roi_baseline_ref)]
            if roi_finite.size:
                median_val = float(np.nanmedian(roi_finite))
                min_val = float(np.nanmin(roi_finite))
                max_val = float(np.nanmax(roi_finite))
                if (max_val - median_val) > (median_val - min_val):
                    extremum_mode = 'peak'
        if roi_wavelengths.size:
            if extremum_mode == 'valley':
                baseline_peak_idx = int(np.nanargmin(np.where(np.isfinite(roi_baseline_ref), roi_baseline_ref, np.inf)))
            else:
                baseline_peak_idx = int(np.nanargmax(np.where(np.isfinite(roi_baseline_ref), roi_baseline_ref, -np.inf)))
        else:
            baseline_peak_idx = 0
        baseline_peak_idx = int(np.clip(baseline_peak_idx, 0, max(0, roi_wavelengths.size - 1)))

        wl_step = float(np.nanmedian(np.diff(roi_wavelengths))) if roi_wavelengths.size > 1 else 0.2
        wl_step = wl_step if np.isfinite(wl_step) and wl_step > 0 else 0.2
        search_radius = int(max(2, math.ceil(1.5 / wl_step)))

        peak_idx = np.full(len(frames), baseline_peak_idx, dtype=int)
        
        for frame_idx, row in enumerate(roi_matrix):
            if not np.any(np.isfinite(row)):
                continue
            start = max(0, baseline_peak_idx - search_radius)
            end = min(row.size, baseline_peak_idx + search_radius + 1)
            window = row[start:end].astype(float)
            if window.size == 0:
                continue
            if extremum_mode == 'valley':
                window[~np.isfinite(window)] = np.inf
                local_idx = int(np.argmin(window))
            else:
                window[~np.isfinite(window)] = -np.inf
                local_idx = int(np.argmax(window))
            peak_idx[frame_idx] = start + local_idx
        
        peak_idx = np.clip(peak_idx, 0, max(0, roi_wavelengths.size - 1))
        
        # Simple grid-based peak wavelengths (sub-pixel interpolation disabled for performance)
        peak_wls = roi_wavelengths[peak_idx] if roi_wavelengths.size else np.full(len(frames), float('nan'))
        baseline_peak_val = float(peak_wls[valid].mean()) if len(valid) > 0 and np.any(np.isfinite(peak_wls[valid])) else (float(roi_wavelengths[baseline_peak_idx]) if roi_wavelengths.size else float('nan'))
        delta_lambda_local = peak_wls - baseline_peak_val
        abs_delta_lambda_local = np.abs(delta_lambda_local)
        baseline_delta_local = delta_lambda_local[valid]
        lambda_sigma_local = float(np.nanstd(baseline_delta_local, ddof=1)) if baseline_delta_local.size > 1 else 0.0
        lambda_sigma_local = float(np.nan_to_num(lambda_sigma_local, nan=0.0))
        return (
            baseline_ref,
            roi_baseline_ref,
            baseline_mean_val,
            baseline_std_val,
            centered,
            delta_mean_local,
            roi_delta_local,
            peak_wls,
            baseline_peak_val,
            delta_lambda_local,
            abs_delta_lambda_local,
            lambda_sigma_local,
        )

    (
        baseline_reference,
        roi_baseline_reference,
        baseline_mean_abs,
        baseline_std_abs,
        centered_matrix,
        delta_mean,
        roi_delta_matrix,
        peak_wavelengths,
        baseline_peak_nm,
        delta_lambda,
        abs_delta_lambda,
        lambda_sigma,
    ) = _compute_baseline_outputs(baseline_indices)

    activation_delta = float(response_cfg.get('activation_delta', 0.01) or 0.01)
    sigma_multiplier = float(response_cfg.get('activation_sigma_multiplier', 1.5) or 1.5)
    noise_floor = float(response_cfg.get('noise_floor', 1e-4) or 1e-4)
    threshold = activation_delta + sigma_multiplier * max(lambda_sigma, noise_floor)

    slope_sigma_multiplier = float(response_cfg.get('slope_sigma_multiplier', 1.0) or 1.0)
    min_response_slope = float(response_cfg.get('min_response_slope', 0.0) or 0.0)

    direction = float(np.sign(np.nanmedian(delta_lambda[np.isfinite(delta_lambda)]))) if np.any(np.isfinite(delta_lambda)) else 1.0
    if not np.isfinite(direction) or direction == 0.0:
        direction = 1.0

    responsive_indices = [idx for idx, val in enumerate(abs_delta_lambda)
                          if np.isfinite(val) and val >= threshold]

    changepoint_cfg = response_cfg.get('changepoint', {}) if isinstance(response_cfg.get('changepoint', {}), dict) else {}
    responsive_segments: List[Tuple[int, int]] = []

    if changepoint_cfg.get('enabled', False):
        cp_signal_mode = str(changepoint_cfg.get('signal', 'abs_delta_lambda')).lower()
        if cp_signal_mode == 'delta_lambda':
            change_signal = np.copy(delta_lambda)
        elif cp_signal_mode == 'delta_mean':
            change_signal = np.copy(delta_mean)
        else:
            change_signal = np.copy(abs_delta_lambda)

        smooth_win = int(changepoint_cfg.get('smooth_window', 3) or 3)
        if smooth_win > 1:
            change_signal = _smooth_vector(change_signal, smooth_win)

        cp_method = str(changepoint_cfg.get('method', 'pelt')).lower()
        min_seg_size = int(changepoint_cfg.get('min_segment_size', 8) or 8)
        
        if cp_method == 'pelt':
            penalty = float(changepoint_cfg.get('penalty', 3.0) or 3.0)
            cps = _pelt_changepoint_detection(change_signal, penalty=penalty, min_size=min_seg_size)
            
            # Convert change-points to segments and filter by mean signal level
            segments = []
            boundaries = [0] + cps + [len(change_signal)]
            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if end - start >= min_seg_size:
                    seg_signal = change_signal[start:end]
                    seg_mean = float(np.nanmean(seg_signal))
                    # Keep segments with elevated signal
                    if seg_mean >= threshold * 0.8:
                        segments.append((start, end - 1))
            responsive_segments = segments
        else:
            # Fallback to threshold-based detection
            scale = float(changepoint_cfg.get('threshold_scale', 1.0) or 1.0)
            release_mult = float(changepoint_cfg.get('release_multiplier', 0.6) or 0.6)
            min_len = int(changepoint_cfg.get('min_length', 4) or 4)
            pad = int(changepoint_cfg.get('pad', 1) or 1)
            cp_high = threshold * scale
            cp_low = cp_high * release_mult
            segments = _detect_segments_above_threshold(change_signal, cp_high, cp_low, min_len, pad)
            responsive_segments = segments

        if responsive_segments:
            responsive_indices = sorted({idx for start, end in responsive_segments for idx in range(start, end + 1)
                                         if 0 <= idx < len(frames)})
            roi_only_indices = list(responsive_indices)

    # Ensure baseline frames occur before response onset if possible
    if responsive_indices:
        first_resp = min(responsive_indices)
        trimmed = [idx for idx in baseline_indices if idx < first_resp]
        if len(trimmed) >= max(1, baseline_target // 2):
            baseline_indices = trimmed
        elif first_resp > 0:
            start = max(0, first_resp - baseline_target)
            baseline_indices = list(range(start, first_resp)) or [max(0, first_resp - 1)]

        (
            baseline_reference,
            roi_baseline_reference,
            baseline_mean_abs,
            baseline_std_abs,
            centered_matrix,
            delta_mean,
            roi_delta_matrix,
            peak_wavelengths,
            baseline_peak_nm,
            delta_lambda,
            abs_delta_lambda,
            lambda_sigma,
        ) = _compute_baseline_outputs(baseline_indices)
        direction = float(np.sign(np.nanmedian(delta_lambda[np.isfinite(delta_lambda)]))) if np.any(np.isfinite(delta_lambda)) else 1.0
        if not np.isfinite(direction) or direction == 0.0:
            direction = 1.0
        threshold = activation_delta + sigma_multiplier * max(lambda_sigma, noise_floor)
        responsive_indices = [idx for idx, val in enumerate(abs_delta_lambda)
                              if np.isfinite(val) and val >= threshold]

    roi_only_indices = list(responsive_indices)

    min_activation_frames = int(response_cfg.get('min_activation_frames', 6) or 6)
    min_activation_fraction = float(response_cfg.get('min_activation_fraction', 0.08) or 0.08)
    required = max(min_activation_frames, int(math.ceil(min_activation_fraction * len(frames))))
    if responsive_indices and len(responsive_indices) < required:
        energy = abs_delta_lambda
        top_idx = np.argsort(energy)[::-1][:required]
        responsive_indices = sorted(set(list(responsive_indices) + top_idx.tolist()))

    if not responsive_indices and int(response_cfg.get('fallback_window', 4) or 4) > 0:
        responsive_indices = list(range(min(len(frames), int(response_cfg.get('fallback_window', 4) or 4))))

    slope_threshold = max(min_response_slope, slope_sigma_multiplier * max(lambda_sigma, noise_floor))
    selected_slope = float('nan')

    if responsive_indices:
        sorted_resp = sorted(responsive_indices)
        monotonic_indices = list(sorted_resp)
        monotonic_tol = float(response_cfg.get('monotonic_tolerance_nm', 0.05) or 0.0)
        if len(sorted_resp) >= 2:
            signed_trace = delta_lambda[sorted_resp] * direction
            diffs = np.diff(signed_trace)
            negative_diffs = np.where(diffs < -monotonic_tol)[0]
            if negative_diffs.size > 0:
                cutoff = negative_diffs[0] + 1
                trimmed = sorted_resp[:cutoff]
                if trimmed:
                    monotonic_indices = trimmed
        min_len_for_slope = max(3, min_activation_frames)
        candidate_indices = list(monotonic_indices)
        slope_pass = False
        while candidate_indices and len(candidate_indices) >= min_len_for_slope:
            candidate_lambda = delta_lambda[candidate_indices]
            if len(candidate_indices) >= 2:
                slope_val = linregress(candidate_indices, candidate_lambda).slope
            else:
                slope_val = 0.0
            if not np.isfinite(slope_val):
                slope_val = 0.0
            slope_along_dir = slope_val * direction
            selected_slope = slope_along_dir
            if slope_along_dir >= slope_threshold:
                slope_pass = True
                break
            candidate_indices = candidate_indices[1:]
        if slope_pass:
            responsive_indices = candidate_indices
        else:
            responsive_indices = []

    records = []
    segment_ids = np.full(len(mean_absorb), -1, dtype=int)
    if responsive_segments:
        for seg_id, (seg_start, seg_end) in enumerate(responsive_segments, start=1):
            seg_start = int(max(0, seg_start))
            seg_end = int(min(len(mean_absorb) - 1, seg_end))
            segment_ids[seg_start:seg_end + 1] = seg_id

    for idx, mean_val in enumerate(mean_absorb):
        records.append({
            'frame_index': idx,
            'mean_signal': float(mean_val),
            'delta_mean': float(delta_mean[idx]),
            'delta_lambda_nm': float(delta_lambda[idx]) if np.isfinite(delta_lambda[idx]) else np.nan,
            'delta_lambda_abs_nm': float(abs_delta_lambda[idx]) if np.isfinite(abs_delta_lambda[idx]) else np.nan,
            'is_responsive': 1 if idx in responsive_indices else 0,
            'peak_wavelength_nm': float(peak_wavelengths[idx]) if np.isfinite(peak_wavelengths[idx]) else np.nan,
            'segment_id': int(segment_ids[idx]) if segment_ids.size else -1,
        })

    df_series = pd.DataFrame(records)
    df_series['dataset_label'] = dataset_label
    df_series['threshold_nm'] = threshold
    df_series['baseline_mean'] = baseline_mean_abs
    df_series['baseline_std'] = baseline_std_abs
    df_series['baseline_std_abs'] = baseline_std_abs
    df_series['baseline_std_delta_lambda_nm'] = lambda_sigma
    df_series['baseline_peak_nm'] = baseline_peak_nm
    df_series['slope_threshold'] = slope_threshold
    df_series['responsive_slope'] = selected_slope
    df_series['baseline_frames'] = len(baseline_indices)
    df_series['activation_delta_nm'] = activation_delta
    df_series['roi_min_nm'] = float(min_wl_roi)
    df_series['roi_max_nm'] = float(max_wl_roi)
    df_series['response_direction'] = direction
    df_series['baseline_indices'] = [baseline_indices] * len(df_series)
    if responsive_segments:
        df_series['responsive_segments'] = [responsive_segments] * len(df_series)
    return df_series, responsive_indices, roi_only_indices
def _scale_reference_to_baseline(ref_df: Optional[pd.DataFrame],
                                 baseline_frames: Sequence[pd.DataFrame],
                                 percentile: float = 95.0) -> Tuple[Optional[pd.DataFrame], float]:
    """Scale a reference spectrum so that it matches a trial's baseline intensity."""
    if ref_df is None or not baseline_frames:
        return ref_df, 1.0

    if 'wavelength' not in ref_df.columns or 'intensity' not in ref_df.columns:
        return ref_df, 1.0

    ref_wl = ref_df['wavelength'].to_numpy(dtype=float)
    ref_int = ref_df['intensity'].to_numpy(dtype=float)
    if ref_wl.size == 0 or ref_int.size == 0:
        return ref_df, 1.0

    baseline_vals: List[float] = []
    for frame in baseline_frames:
        if frame is None or frame.empty:
            continue
        if 'wavelength' not in frame.columns or 'intensity' not in frame.columns:
            continue
        frame_wl = frame['wavelength'].to_numpy(dtype=float)
        frame_int = frame['intensity'].to_numpy(dtype=float)
        if frame_wl.size == 0 or frame_int.size == 0:
            continue
        interp = np.interp(ref_wl, frame_wl, frame_int)
        baseline_vals.append(float(np.percentile(interp, percentile)))

    if not baseline_vals:
        return ref_df, 1.0

    baseline_target = float(np.nanmean(baseline_vals))
    ref_percentile = float(np.percentile(ref_int, percentile))
    if ref_percentile <= 0 or not np.isfinite(baseline_target):
        return ref_df, 1.0

    scale_factor = baseline_target / ref_percentile
    if scale_factor <= 0 or not np.isfinite(scale_factor):
        return ref_df, 1.0

    scaled = ref_df.copy(deep=True)
    scaled['intensity'] = scaled['intensity'].astype(float) * scale_factor
    return scaled, float(scale_factor)


def _score_trial_quality(df: pd.DataFrame,
                         *,
                         roi_bounds: Tuple[Optional[float], Optional[float]],
                         expected_center: Optional[float]) -> Tuple[float, Dict[str, float]]:
    """Return a 0–1 quality score using SNR, contrast, and peak alignment."""
    details: Dict[str, float] = {}
    if df is None or df.empty or 'wavelength' not in df.columns:
        return 0.0, details

    wl = df['wavelength'].to_numpy(dtype=float)
    signal_col = _signal_column(df)
    signal = df[signal_col].to_numpy(dtype=float)

    min_wl, max_wl = roi_bounds
    if min_wl is None:
        min_wl = float(np.nanmin(wl))
    if max_wl is None:
        max_wl = float(np.nanmax(wl))
    mask = (wl >= min_wl) & (wl <= max_wl)
    if not np.any(mask):
        mask = np.ones_like(wl, dtype=bool)

    roi_signal = signal[mask]
    roi_wl = wl[mask]
    if roi_signal.size == 0:
        return 0.0, details

    baseline_window = min(len(signal), 200)
    baseline_noise = float(np.nanstd(signal[:baseline_window], ddof=1)) if baseline_window else 0.0
    if not np.isfinite(baseline_noise) or baseline_noise <= 0:
        baseline_noise = 1e-3

    peak_idx = int(np.nanargmax(roi_signal)) if np.any(np.isfinite(roi_signal)) else 0
    peak_val = float(roi_signal[peak_idx]) if roi_signal.size else 0.0
    snr = peak_val / baseline_noise
    details['snr'] = float(snr)

    snr_score = 0.0
    if snr >= 5:
        snr_score = 0.5
    elif snr >= 3:
        snr_score = 0.35
    elif snr >= 2:
        snr_score = 0.2
    elif snr >= 1:
        snr_score = 0.1

    contrast = float(np.nanmax(roi_signal) - np.nanmin(roi_signal)) if roi_signal.size else 0.0
    details['contrast'] = contrast
    contrast_score = 0.0
    if contrast >= 0.5:
        contrast_score = 0.3
    elif contrast >= 0.25:
        contrast_score = 0.2
    elif contrast >= 0.1:
        contrast_score = 0.1

    if expected_center is None:
        expected_center = float((min_wl + max_wl) / 2.0)
    peak_wl = float(roi_wl[peak_idx]) if roi_wl.size else expected_center
    shift = abs(peak_wl - expected_center)
    details['peak_wavelength'] = peak_wl
    details['expected_center'] = expected_center
    details['shift_nm'] = shift
    shift_score = max(0.0, 0.2 - min(shift, 1.0) * 0.2)

    total_score = min(1.0, snr_score + contrast_score + shift_score)
    details['total_score'] = total_score
    return total_score, details
def _simple_response_selection(frames: Sequence[pd.DataFrame],
                               ref_df: Optional[pd.DataFrame],
                               *,
                               dataset_label: Optional[str],
                               response_cfg: Dict[str, object]) -> Tuple[pd.DataFrame, List[int], List[int]]:
    """Lightweight frame selection that ranks frames by ROI absorbance energy."""
    if not frames:
        return None, [], []

    processed = [_ensure_response_signal(df, ref_df) for df in frames]
    base_wl = processed[0]['wavelength'].to_numpy()
    absorb_col = 'absorbance' if 'absorbance' in processed[0].columns else _signal_column(processed[0])

    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = float(base_wl.min())
    if max_wl_roi is None:
        max_wl_roi = float(base_wl.max())
    roi_mask = (base_wl >= min_wl_roi) & (base_wl <= max_wl_roi)
    if not np.any(roi_mask):
        roi_mask = np.ones_like(base_wl, dtype=bool)

    response_metrics: List[float] = []
    for df in processed:
        wl = df['wavelength'].to_numpy(dtype=float)
        signal = df[absorb_col].to_numpy(dtype=float)
        if not np.array_equal(wl, base_wl):
            signal = np.interp(base_wl, wl, signal)
        roi_signal = signal[roi_mask]
        metric = float(np.nansum(np.clip(roi_signal, 0.0, None)))
        response_metrics.append(metric)

    response_arr = np.array(response_metrics, dtype=float)
    smooth_window = int(response_cfg.get('simple_smooth_window', 5) or 5)
    if smooth_window > 1 and response_arr.size >= smooth_window:
        kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
        smoothed = np.convolve(response_arr, kernel, mode='same')
    else:
        smoothed = response_arr.copy()

    top_n = int(response_cfg.get('top_n_frames', 20) or 20)
    top_n = max(1, min(top_n, len(frames)))
    if np.all(~np.isfinite(smoothed)):
        smoothed = np.nan_to_num(smoothed, nan=0.0)

    finite_mask = np.isfinite(smoothed)
    if not np.any(finite_mask):
        smoothed = np.zeros_like(smoothed)
        finite_mask = np.ones_like(smoothed, dtype=bool)

    rank_indices = np.argsort(smoothed[finite_mask])
    finite_indices = np.where(finite_mask)[0][rank_indices]
    if finite_indices.size >= top_n:
        top_indices = finite_indices[-top_n:]
    else:
        fallback = np.arange(len(frames))[-top_n:]
        top_indices = np.unique(np.concatenate([finite_indices, fallback]))[-top_n:]

    top_indices = sorted(int(idx) for idx in top_indices)

    time_series_df = pd.DataFrame({
        'frame_index': np.arange(len(frames)),
        'response_metric': response_arr,
        'smoothed_response': smoothed,
        'selected': [idx in top_indices for idx in range(len(frames))],
    })

    return time_series_df, top_indices, top_indices
def _save_response_series(df: pd.DataFrame,
                          out_root: str,
                          concentration: float,
                          trial: str,
                          dataset_label: Optional[str]) -> Tuple[Path, Path]:
    series_dir = Path(out_root) / 'time_series'
    series_dir.mkdir(parents=True, exist_ok=True)
    safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
    prefix = f"{dataset_label or 'dataset'}_{concentration:g}_{safe_trial}"
    csv_path = series_dir / f"{prefix}.csv"
    df.to_csv(csv_path, index=False)

    plot_path = series_dir / f"{prefix}.png"
    try:
        # Build multi-panel visualization
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        indices = df['frame_index'].to_numpy(dtype=float)
        delta_peak = df.get('delta_peak', pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        delta_peak_raw = df.get('delta_peak_raw', pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        delta_mean = df.get('delta_mean', pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        threshold = df.get('threshold', pd.Series([np.nan])).iloc[0]
        responsive_mask = df['is_responsive'].to_numpy(dtype=int) == 1

        ax0 = axes[0]
        ax0.plot(indices, delta_peak_raw, marker='o', label='Δ peak (raw)')
        ax0.plot(indices, delta_peak, marker='^', label='Δ peak (directed)')
        if np.any(np.isfinite(delta_mean)):
            ax0.plot(indices, delta_mean, marker='s', label='Δ mean')
        if np.isfinite(threshold):
            ax0.axhline(threshold, color='r', linestyle='--', linewidth=1.0, label='activation threshold')
        if responsive_mask.any():
            ax0.scatter(indices[responsive_mask], delta_peak_raw[responsive_mask], color='red', zorder=3, label='responsive frames')
            ymin = np.nanmin(delta_peak_raw[responsive_mask]) if np.any(np.isfinite(delta_peak_raw[responsive_mask])) else np.nanmin(delta_peak_raw)
            ymax = np.nanmax(delta_peak_raw[responsive_mask]) if np.any(np.isfinite(delta_peak_raw[responsive_mask])) else np.nanmax(delta_peak_raw)
            ax0.fill_between(indices, ymin, ymax, where=responsive_mask,
                             color='red', alpha=0.08, step='mid')
        ax0.set_ylabel('Δ absorbance')
        ax0.legend(loc='best')
        ax0.grid(alpha=0.25)

        ax1 = axes[1]
        peak_wl = df.get('peak_wavelength_nm', pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        if np.isfinite(peak_wl).any():
            ax1.plot(indices, peak_wl, color='tab:blue', marker='.', label='peak λ')
        mean_signal = df.get('mean_signal', pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
        ax1b = ax1.twinx()
        ax1b.plot(indices, mean_signal, color='tab:orange', alpha=0.6, label='mean absorbance')
        if responsive_mask.any():
            ax1.fill_between(indices, np.nanmin(peak_wl) if np.isfinite(peak_wl).any() else 0,
                             np.nanmax(peak_wl) if np.isfinite(peak_wl).any() else 1,
                             where=responsive_mask, color='red', alpha=0.1, step='mid', label='responsive window')
        ax1.set_ylabel('Peak λ (nm)')
        ax1b.set_ylabel('Mean absorbance (a.u.)')
        ax1.grid(alpha=0.25)

        # Combine legends for bottom axis
        handles, labels = ax1.get_legend_handles_labels()
        handles_b, labels_b = ax1b.get_legend_handles_labels()
        if handles or handles_b:
            ax1.legend(handles + handles_b, labels + labels_b, loc='best')

        axes[-1].set_xlabel('Frame index (acquisition order)')

        title_label = dataset_label or 'dataset'
        fig.suptitle(f'Response diagnostics: {title_label} {concentration:g} ppm ({trial})', fontsize=11)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
    return csv_path, plot_path


def _summarize_responsive_delta(df: pd.DataFrame) -> Dict[str, object]:
    if df is None or df.empty:
        return {}

    try:
        delta_series = pd.to_numeric(df.get('delta_lambda_nm'), errors='coerce')
    except Exception:
        delta_series = pd.Series(dtype=float)

    if 'is_responsive' in df.columns:
        responsive_mask = pd.to_numeric(df['is_responsive'], errors='coerce').fillna(0).astype(int) == 1
    else:
        responsive_mask = pd.Series([False] * len(df))

    segment_ids = None
    try:
        if 'segment_id' in df.columns:
            segment_ids = pd.to_numeric(df['segment_id'], errors='coerce').fillna(-1).astype(int)
    except Exception:
        segment_ids = None

    segment_definitions: Optional[List[Tuple[int, int]]] = None
    if 'responsive_segments' in df.columns and len(df.index) > 0:
        try:
            raw_segments = df['responsive_segments'].iloc[0]
            if isinstance(raw_segments, list):
                segment_definitions = []
                for entry in raw_segments:
                    if isinstance(entry, (list, tuple)) and len(entry) == 2:
                        try:
                            start = int(entry[0])
                            end = int(entry[1])
                        except (TypeError, ValueError):
                            continue
                        if start <= end:
                            segment_definitions.append((start, end))
        except Exception:
            segment_definitions = None

    delta_values = delta_series.to_numpy(dtype=float) if delta_series.size else np.array([], dtype=float)
    responsive_values = delta_series[responsive_mask].to_numpy(dtype=float) if delta_series.size else np.array([], dtype=float)
    responsive_finite = responsive_values[np.isfinite(responsive_values)]
    all_finite = delta_values[np.isfinite(delta_values)]

    # Derive segments if definitions missing but segment ids present
    if segment_definitions is None and segment_ids is not None and segment_ids.size:
        segment_definitions = []
        current_id = None
        start_idx = None
        for idx, seg_id in enumerate(segment_ids):
            if seg_id > 0:
                if current_id != seg_id:
                    if current_id is not None and start_idx is not None:
                        segment_definitions.append((start_idx, idx - 1))
                    current_id = seg_id
                    start_idx = idx
            else:
                if current_id is not None and start_idx is not None:
                    segment_definitions.append((start_idx, idx - 1))
                current_id = None
                start_idx = None
        if current_id is not None and start_idx is not None:
            segment_definitions.append((start_idx, len(segment_ids) - 1))
        if not segment_definitions:
            segment_definitions = None

    baseline_peak_nm = float('nan')
    if 'baseline_peak_nm' in df.columns:
        try:
            baseline_vals = pd.to_numeric(df['baseline_peak_nm'], errors='coerce').dropna()
            if not baseline_vals.empty:
                baseline_peak_nm = float(baseline_vals.iloc[0])
        except Exception:
            baseline_peak_nm = float('nan')

    responsive_frame_count = int(responsive_mask.sum()) if hasattr(responsive_mask, 'sum') else 0
    total_frame_count = int(len(df))
    responsive_fraction = float(responsive_frame_count / max(1, total_frame_count))

    median_delta = float('nan')
    mean_delta = float('nan')
    std_delta = float('nan')
    max_abs_delta = float('nan')
    signed_consistency = float('nan')
    if responsive_finite.size:
        median_delta = float(np.nanmedian(responsive_finite))
        mean_delta = float(np.nanmean(responsive_finite))
        std_delta = float(np.nanstd(responsive_finite, ddof=1)) if responsive_finite.size > 1 else 0.0
        idx_max = int(np.abs(responsive_finite).argmax())
        max_abs_delta = float(responsive_finite[idx_max])
        total = responsive_finite.size
        if total > 0:
            same_sign = float(np.sum(np.sign(responsive_finite) == np.sign(median_delta)))
            signed_consistency = same_sign / total if total else float('nan')
    elif all_finite.size:
        mean_delta = float(np.nanmean(all_finite))
        std_delta = float(np.nanstd(all_finite, ddof=1)) if all_finite.size > 1 else 0.0
        idx_max = int(np.abs(all_finite).argmax())
        max_abs_delta = float(all_finite[idx_max])

    fallback_delta = float('nan')
    if all_finite.size:
        idx_fb = int(np.abs(all_finite).argmax())
        fallback_delta = float(all_finite[idx_fb])

    selected_delta = median_delta if np.isfinite(median_delta) else fallback_delta
    direction = float(np.sign(selected_delta)) if np.isfinite(selected_delta) and selected_delta != 0 else float('nan')

    median_peak_nm = baseline_peak_nm + median_delta if np.isfinite(baseline_peak_nm) and np.isfinite(median_delta) else float('nan')
    mean_peak_nm = baseline_peak_nm + mean_delta if np.isfinite(baseline_peak_nm) and np.isfinite(mean_delta) else float('nan')
    selected_peak_nm = baseline_peak_nm + selected_delta if np.isfinite(baseline_peak_nm) and np.isfinite(selected_delta) else float('nan')

    segment_count = 0
    segment_lengths: List[int] = []
    segment_coverage = float('nan')
    if segment_definitions:
        segment_count = len(segment_definitions)
        segment_lengths = [max(0, int(end) - int(start) + 1) for start, end in segment_definitions]
        total_len = sum(segment_lengths)
        if total_frame_count > 0:
            segment_coverage = total_len / float(total_frame_count)

    return {
        'responsive_frame_count': responsive_frame_count,
        'total_frame_count': total_frame_count,
        'responsive_fraction': responsive_fraction,
        'responsive_finite_count': int(responsive_finite.size),
        'median_delta_nm': median_delta if np.isfinite(median_delta) else float('nan'),
        'mean_delta_nm': mean_delta if np.isfinite(mean_delta) else float('nan'),
        'selected_delta_nm': selected_delta if np.isfinite(selected_delta) else float('nan'),
        'std_delta_nm': std_delta if np.isfinite(std_delta) else float('nan'),
        'max_abs_delta_nm': max_abs_delta if np.isfinite(max_abs_delta) else float('nan'),
        'fallback_delta_nm': fallback_delta if np.isfinite(fallback_delta) else float('nan'),
        'median_peak_nm': median_peak_nm if np.isfinite(median_peak_nm) else float('nan'),
        'mean_peak_nm': mean_peak_nm if np.isfinite(mean_peak_nm) else float('nan'),
        'selected_peak_nm': selected_peak_nm if np.isfinite(selected_peak_nm) else float('nan'),
        'baseline_peak_nm': baseline_peak_nm if np.isfinite(baseline_peak_nm) else float('nan'),
        'direction': direction if np.isfinite(direction) else float('nan'),
        'signed_consistency': signed_consistency if np.isfinite(signed_consistency) else float('nan'),
        'responsive_segment_count': float(segment_count),
        'responsive_segment_lengths': segment_lengths,
        'responsive_segment_coverage': segment_coverage if np.isfinite(segment_coverage) else float('nan'),
    }


def _aggregate_responsive_delta_maps(responsive_delta_by_conc: Dict[float, Dict[str, Dict[str, object]]]
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[float, Dict[str, float]]]:
    if not responsive_delta_by_conc:
        empty_df = pd.DataFrame()
        return empty_df, empty_df.copy(), {}

    rows_trial: List[Dict[str, object]] = []
    rows_conc: List[Dict[str, object]] = []
    summary_by_conc: Dict[float, Dict[str, float]] = {}

    for conc, trial_map in sorted(responsive_delta_by_conc.items(), key=lambda kv: kv[0]):
        if not isinstance(trial_map, dict) or not trial_map:
            continue

        trial_count = 0
        responsive_trial_count = 0
        total_responsive_frames = 0
        total_frames = 0

        selected_delta_vals: List[float] = []
        median_delta_vals: List[float] = []
        mean_delta_vals: List[float] = []
        selected_peak_vals: List[float] = []
        baseline_peak_vals: List[float] = []
        responsive_fraction_vals: List[float] = []
        direction_vals: List[float] = []
        segment_count_vals: List[float] = []
        segment_coverage_vals: List[float] = []
        segment_length_mean_vals: List[float] = []

        for trial, summary in sorted(trial_map.items()):
            if not isinstance(summary, dict) or not summary:
                continue
            trial_count += 1

            responsive_frames = int(summary.get('responsive_frame_count', 0) or 0)
            frame_total = int(summary.get('total_frame_count', 0) or 0)
            total_responsive_frames += responsive_frames
            total_frames += frame_total

            selected_delta = _safe_float(summary.get('selected_delta_nm'))
            median_delta = _safe_float(summary.get('median_delta_nm'))
            mean_delta = _safe_float(summary.get('mean_delta_nm'))
            selected_peak = _safe_float(summary.get('selected_peak_nm'))
            baseline_peak = _safe_float(summary.get('baseline_peak_nm'))
            responsive_fraction = _safe_float(summary.get('responsive_fraction'))
            direction = _safe_float(summary.get('direction'))

            seg_count = _safe_float(summary.get('responsive_segment_count'))
            seg_coverage = _safe_float(summary.get('responsive_segment_coverage'))
            seg_lengths = summary.get('responsive_segment_lengths') if isinstance(summary.get('responsive_segment_lengths'), list) else None
            mean_seg_len = float('nan')
            if isinstance(seg_lengths, list) and seg_lengths:
                try:
                    mean_seg_len = float(np.nanmean([float(v) for v in seg_lengths]))
                except Exception:
                    mean_seg_len = float('nan')

            if np.isfinite(selected_delta):
                selected_delta_vals.append(selected_delta)
            if np.isfinite(median_delta):
                median_delta_vals.append(median_delta)
            if np.isfinite(mean_delta):
                mean_delta_vals.append(mean_delta)
            if np.isfinite(selected_peak):
                selected_peak_vals.append(selected_peak)
            if np.isfinite(baseline_peak):
                baseline_peak_vals.append(baseline_peak)
            if np.isfinite(responsive_fraction):
                responsive_fraction_vals.append(responsive_fraction)
            if np.isfinite(direction) and direction != 0.0:
                direction_vals.append(float(np.sign(direction)))
            if np.isfinite(seg_count):
                segment_count_vals.append(seg_count)
            if np.isfinite(seg_coverage):
                segment_coverage_vals.append(seg_coverage)
            if np.isfinite(mean_seg_len):
                segment_length_mean_vals.append(mean_seg_len)

            if responsive_frames > 0:
                responsive_trial_count += 1

            row = {
                'concentration': float(conc),
                'trial': trial,
                'selected_delta_nm': selected_delta if np.isfinite(selected_delta) else float('nan'),
                'median_delta_nm': median_delta if np.isfinite(median_delta) else float('nan'),
                'mean_delta_nm': mean_delta if np.isfinite(mean_delta) else float('nan'),
                'selected_peak_nm': selected_peak if np.isfinite(selected_peak) else float('nan'),
                'median_peak_nm': _safe_float(summary.get('median_peak_nm')),
                'mean_peak_nm': _safe_float(summary.get('mean_peak_nm')),
                'baseline_peak_nm': baseline_peak if np.isfinite(baseline_peak) else float('nan'),
                'responsive_frame_count': responsive_frames,
                'total_frame_count': frame_total,
                'responsive_fraction': responsive_fraction if np.isfinite(responsive_fraction) else float('nan'),
                'std_delta_nm': _safe_float(summary.get('std_delta_nm')),
                'max_abs_delta_nm': _safe_float(summary.get('max_abs_delta_nm')),
                'fallback_delta_nm': _safe_float(summary.get('fallback_delta_nm')),
                'direction': direction if np.isfinite(direction) else float('nan'),
                'responsive_segment_count': seg_count if np.isfinite(seg_count) else float('nan'),
                'responsive_segment_coverage': seg_coverage if np.isfinite(seg_coverage) else float('nan'),
                'responsive_segment_mean_length': mean_seg_len if np.isfinite(mean_seg_len) else float('nan'),
            }
            rows_trial.append(row)

        if trial_count == 0:
            continue

        selected_arr = np.array(selected_delta_vals, dtype=float) if selected_delta_vals else np.array([])
        median_arr = np.array(median_delta_vals, dtype=float) if median_delta_vals else np.array([])
        mean_arr = np.array(mean_delta_vals, dtype=float) if mean_delta_vals else np.array([])
        selected_peak_arr = np.array(selected_peak_vals, dtype=float) if selected_peak_vals else np.array([])
        baseline_peak_arr = np.array(baseline_peak_vals, dtype=float) if baseline_peak_vals else np.array([])
        responsive_fraction_arr = np.array(responsive_fraction_vals, dtype=float) if responsive_fraction_vals else np.array([])
        segment_count_arr = np.array(segment_count_vals, dtype=float) if segment_count_vals else np.array([])
        segment_cov_arr = np.array(segment_coverage_vals, dtype=float) if segment_coverage_vals else np.array([])
        segment_len_arr = np.array(segment_length_mean_vals, dtype=float) if segment_length_mean_vals else np.array([])

        def _nan_stat(arr: np.ndarray, func: str) -> float:
            if arr.size == 0 or not np.isfinite(arr).any():
                return float('nan')
            if func == 'mean':
                return float(np.nanmean(arr))
            if func == 'median':
                return float(np.nanmedian(arr))
            if func == 'std':
                return float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
            if func == 'min':
                return float(np.nanmin(arr))
            if func == 'max':
                return float(np.nanmax(arr))
            return float('nan')

        dominant_direction = float('nan')
        if direction_vals:
            pos = sum(1 for d in direction_vals if d > 0)
            neg = sum(1 for d in direction_vals if d < 0)
            if pos or neg:
                dominant_direction = 1.0 if pos >= neg else -1.0

        summary = {
            'trial_count': float(trial_count),
            'responsive_trial_count': float(responsive_trial_count),
            'total_responsive_frames': float(total_responsive_frames),
            'total_frames': float(total_frames),
            'selected_delta_nm_median': _nan_stat(selected_arr, 'median'),
            'selected_delta_nm_mean': _nan_stat(selected_arr, 'mean'),
            'selected_delta_nm_std': _nan_stat(selected_arr, 'std'),
            'selected_delta_nm_min': _nan_stat(selected_arr, 'min'),
            'selected_delta_nm_max': _nan_stat(selected_arr, 'max'),
            'median_delta_nm_median': _nan_stat(median_arr, 'median'),
            'median_delta_nm_mean': _nan_stat(median_arr, 'mean'),
            'mean_delta_nm_mean': _nan_stat(mean_arr, 'mean'),
            'selected_peak_nm_median': _nan_stat(selected_peak_arr, 'median'),
            'selected_peak_nm_mean': _nan_stat(selected_peak_arr, 'mean'),
            'baseline_peak_nm_median': _nan_stat(baseline_peak_arr, 'median'),
            'responsive_fraction_mean': _nan_stat(responsive_fraction_arr, 'mean'),
            'responsive_fraction_median': _nan_stat(responsive_fraction_arr, 'median'),
            'dominant_direction': dominant_direction,
            'responsive_segment_count_mean': _nan_stat(segment_count_arr, 'mean'),
            'responsive_segment_count_median': _nan_stat(segment_count_arr, 'median'),
            'responsive_segment_coverage_mean': _nan_stat(segment_cov_arr, 'mean'),
            'responsive_segment_coverage_median': _nan_stat(segment_cov_arr, 'median'),
            'responsive_segment_mean_length': _nan_stat(segment_len_arr, 'mean'),
        }

        rows_conc.append({'concentration': float(conc), **summary})
        summary_by_conc[float(conc)] = summary

    per_trial_df = pd.DataFrame(rows_trial)
    per_conc_df = pd.DataFrame(rows_conc)
    return per_trial_df, per_conc_df, summary_by_conc


def _safe_float(val: object) -> float:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return float('nan')
    return fval if np.isfinite(fval) else float('nan')


def _compute_responsive_trend_fallback(summary: Dict[str, object]) -> Dict[str, float]:
    if not summary:
        return {}

    def _get(name: str) -> float:
        try:
            return float(summary.get(name))
        except (TypeError, ValueError):
            return float('nan')

    responsive_fraction = _get('responsive_fraction')
    signed_consistency = _get('signed_consistency')
    std_delta = _get('std_delta_nm')
    median_delta = _get('median_delta_nm')
    selected_delta = _get('selected_delta_nm')
    fallback_delta = _get('fallback_delta_nm')
    median_peak = _get('median_peak_nm')
    selected_peak = _get('selected_peak_nm')
    baseline_peak = _get('baseline_peak_nm')

    config = CONFIG.get('responsive_trend', {}) if isinstance(CONFIG, dict) else {}
    min_fraction = float(config.get('min_fraction', 0.1))
    min_consistency = float(config.get('min_consistency', 0.6))
    max_noise = float(config.get('max_std_nm', 5.0))

    usable_delta = selected_delta if np.isfinite(selected_delta) else median_delta
    usable_peak = selected_peak if np.isfinite(selected_peak) else median_peak
    fallback_peak = baseline_peak + fallback_delta if np.isfinite(baseline_peak) and np.isfinite(fallback_delta) else usable_peak

    quality_flags = [
        bool(np.isfinite(responsive_fraction) and responsive_fraction >= min_fraction),
        bool(np.isfinite(signed_consistency) and signed_consistency >= min_consistency),
        bool(np.isfinite(std_delta) and std_delta <= max_noise),
    ]

    quality_ok = all(quality_flags)

    return {
        'responsive_fraction': responsive_fraction,
        'signed_consistency': signed_consistency,
        'std_delta_nm': std_delta,
        'usable_delta_nm': usable_delta if np.isfinite(usable_delta) else float('nan'),
        'usable_peak_nm': usable_peak if np.isfinite(usable_peak) else float('nan'),
        'fallback_delta_nm': fallback_delta if np.isfinite(fallback_delta) else float('nan'),
        'fallback_peak_nm': fallback_peak if np.isfinite(fallback_peak) else float('nan'),
        'quality_ok': bool(quality_ok),
        'quality_flags': quality_flags,
    }


def _first_finite(values: Sequence[object]) -> float:
    for val in values:
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fval):
            return fval
    return float('nan')


def _weighted_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> Optional[Dict[str, float]]:
    if w is None or w.size != x.size:
        return None
    mask = np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return None
    xw = x[mask]
    yw = y[mask]
    ww = w[mask]
    ww = ww / np.nanmax(ww)
    lr = LinearRegression()
    lr.fit(xw.reshape(-1, 1), yw, sample_weight=ww)
    preds = lr.predict(x.reshape(-1, 1))
    residuals = y - preds
    r2_val = float(r2_score(yw, lr.predict(xw.reshape(-1, 1)))) if xw.size > 1 else float('nan')
    rmse_val = float(np.sqrt(np.nanmean(residuals ** 2)))
    if np.isnan(r2_val):
        return None
    return {
        'model': 'weighted_ols',
        'slope': float(lr.coef_[0]),
        'intercept': float(lr.intercept_),
        'r2': r2_val,
        'rmse': rmse_val,
        'slope_stderr': float('nan'),
        'slope_ci_low': float('nan'),
        'slope_ci_high': float('nan'),
        'preds': preds,
        'residuals': residuals,
    }


def _theil_sen(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    if x.size < 2:
        return None
    try:
        model = TheilSenRegressor(random_state=0)
        model.fit(x.reshape(-1, 1), y)
        preds = model.predict(x.reshape(-1, 1))
        residuals = y - preds
        r2_val = float(r2_score(y, preds))
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        return {
            'model': 'theil_sen',
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r2': r2_val,
            'rmse': rmse_val,
            'slope_stderr': float('nan'),
            'slope_ci_low': float('nan'),
            'slope_ci_high': float('nan'),
            'preds': preds,
            'residuals': residuals,
        }
    except Exception:
        return None


def _ransac(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
    if x.size < 3:
        return None
    try:
        base_estimator = LinearRegression()
        model = RANSACRegressor(base_estimator=base_estimator, random_state=0)
        model.fit(x.reshape(-1, 1), y)
        preds = model.predict(x.reshape(-1, 1))
        residuals = y - preds
        r2_val = float(r2_score(y, preds))
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        slope = float(model.estimator_.coef_[0]) if hasattr(model.estimator_, 'coef_') else float('nan')
        intercept = float(model.estimator_.intercept_) if hasattr(model.estimator_, 'intercept_') else float('nan')
        return {
            'model': 'ransac',
            'slope': slope,
            'intercept': intercept,
            'r2': r2_val,
            'rmse': rmse_val,
            'slope_stderr': float('nan'),
            'slope_ci_low': float('nan'),
            'slope_ci_high': float('nan'),
            'preds': preds,
            'residuals': residuals,
        }
    except Exception:
        return None


    models: List[Dict[str, object]] = []
    baseline_model = _linreg_model(concs, deltas)
    models.append(baseline_model)
    weighted_model = _weighted_linear(concs, deltas, weights)
    if weighted_model:
        models.append(weighted_model)
    ts_model = _theil_sen(concs, deltas)
    if ts_model:
        models.append(ts_model)
    ransac_model = _ransac(concs, deltas)
    if ransac_model:
        models.append(ransac_model)

    best_model = max(models, key=lambda m: (m.get('r2', float('-inf')), -m.get('rmse', float('inf'))))
    preds = best_model['preds']
    residuals = best_model['residuals']
    rmse = float(best_model['rmse'])
    r2 = float(best_model['r2'])
    slope = float(best_model['slope'])
    intercept = float(best_model['intercept'])
    slope_stderr = float(best_model.get('slope_stderr', float('nan')))
    slope_ci_low = float(best_model.get('slope_ci_low', float('nan')))
    slope_ci_high = float(best_model.get('slope_ci_high', float('nan')))
    model_name = str(best_model.get('model', 'ols'))

    def _loocv_scores(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        nloc = x.size
        if nloc < 3:
            return float('nan'), float('nan')
        y_pred = np.empty(nloc, dtype=float)
        for i in range(nloc):
            mask = np.ones(nloc, dtype=bool)
            mask[i] = False
            res_cv = linregress(x[mask], y[mask])
            y_pred[i] = res_cv.intercept + res_cv.slope * x[i]
        try:
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2_cv = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse_cv = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            return float(r2_cv), float(rmse_cv)
        except Exception:
            return float('nan'), float('nan')

    r2_cv, rmse_cv = _loocv_scores(concs, deltas)

    baseline_conc = float(concs.min())
    baseline_stats = summary_by_conc.get(baseline_conc) if baseline_conc in summary_by_conc else None
    baseline_noise = float('nan')
    if isinstance(baseline_stats, dict):
        baseline_noise = _first_finite([
            baseline_stats.get('selected_delta_nm_std'),
            baseline_stats.get('median_delta_nm_mean'),
        ])

    lod = (_compute_lod_lsq(slope=slope, sigma=baseline_noise) if np.isfinite(baseline_noise)
           else float('nan'))
    loq = (_compute_loq_lsq(slope=slope, sigma=baseline_noise) if np.isfinite(baseline_noise)
           else float('nan'))

    metrics_dir = Path(out_root) / 'metrics'
    plots_dir = Path(out_root) / 'plots'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    table_path = metrics_dir / 'responsive_delta_by_concentration.csv'
    df.to_csv(table_path, index=False)

    quality_true = sum(1 for q in quality_per_conc if q)
    total_quality = len(quality_per_conc)
    quality_ratio = (quality_true / total_quality) if total_quality else 0.0
    required_quality = max(min_quality_count, int(np.ceil(min_quality_ratio * total_quality))) if total_quality else min_quality_count
    calibration_quality_ok = (
        total_quality > 0
        and quality_true >= required_quality
        and np.isfinite(r2)
        and r2 >= min_r2_required
    )

    metrics_payload = {
        'dataset_label': dataset_label,
        'concentrations': concs.tolist(),
        'delta_nm': deltas.tolist(),
        'slope_nm_per_ppm': float(slope),
        'intercept_nm': float(intercept),
        'r2': r2,
        'rmse_nm': rmse,
        'slope_stderr': float(slope_stderr),
        'slope_ci_low': float(slope_ci_low),
        'slope_ci_high': float(slope_ci_high),
        'predictions': preds.tolist(),
        'residuals': residuals.tolist(),
        'r2_cv': r2_cv,
        'rmse_cv': rmse_cv,
        'lod_ppm': float(lod),
        'loq_ppm': float(loq),
        'baseline_noise_nm': float(baseline_noise),
        'trend_fallbacks': detailed_fallbacks,
        'quality_flags': {
            'min_fraction': float(CONFIG.get('responsive_trend', {}).get('min_fraction', 0.1)) if isinstance(CONFIG, dict) else 0.1,
            'min_consistency': float(CONFIG.get('responsive_trend', {}).get('min_consistency', 0.6)) if isinstance(CONFIG, dict) else 0.6,
            'max_std_nm': float(CONFIG.get('responsive_trend', {}).get('max_std_nm', 5.0)) if isinstance(CONFIG, dict) else 5.0,
            'min_quality_ratio': min_quality_ratio,
            'min_quality_count': min_quality_count,
            'min_r2': min_r2_required,
        },
        'quality_per_concentration': quality_per_conc,
        'quality_true_count': int(quality_true),
        'quality_total': int(total_quality),
        'quality_ratio': float(quality_ratio),
        'calibration_quality_ok': bool(calibration_quality_ok),
    }

    metrics_path = metrics_dir / 'responsive_delta_calibration.json'
    _write_json(metrics_path, metrics_payload)

    plot_path = plots_dir / 'responsive_delta_calibration.png'
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(concs, deltas, color='tab:blue', label='Responsive Δλ (median)')
        x_line = np.linspace(float(concs.min()), float(concs.max()), 200)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color='tab:orange', label=f'Linear fit (R²={r2:.3f})')
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Δλ (nm)')
        title = 'Responsive Δλ Calibration'
        if dataset_label:
            title += f' – {dataset_label}'
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        plot_path = None

    result = {
        'dataset_label': dataset_label,
        'concentrations': concs.tolist(),
        'delta_nm': deltas.tolist(),
        'slope_nm_per_ppm': float(slope),
        'intercept_nm': float(intercept),
        'r2': r2,
        'rmse_nm': rmse,
        'slope_stderr': float(slope_stderr),
        'slope_ci_low': float(slope_ci_low),
        'slope_ci_high': float(slope_ci_high),
        'predictions': preds.tolist(),
        'residuals': residuals.tolist(),
        'r2_cv': r2_cv,
        'rmse_cv': rmse_cv,
        'lod_ppm': float(lod),
        'loq_ppm': float(loq),
        'baseline_noise_nm': float(baseline_noise),
        'metrics_path': str(metrics_path),
        'table_path': str(table_path),
        'plot_path': str(plot_path) if plot_path else None,
        'trend_fallbacks': detailed_fallbacks,
        'quality_per_concentration': quality_per_conc,
        'quality_ratio': float(quality_ratio),
        'quality_true_count': int(quality_true),
        'quality_total': int(total_quality),
        'calibration_quality_ok': bool(calibration_quality_ok),
        'quality_thresholds': {
            'min_quality_ratio': min_quality_ratio,
            'min_quality_count': min_quality_count,
            'min_r2': min_r2_required,
        },
    }

    return result


def load_reference_csv(ref_path: str) -> pd.DataFrame:
    """Load the reference spectrum (wavelength,intensity)."""
    ref = _read_csv_spectrum(ref_path)
    if ref.empty:
        raise ValueError("Reference file is empty or invalid")
    return ref


def scan_experiment_root(root_dir: str) -> Dict[float, Dict[str, List[str]]]:
    """Scan the experiment root to build mapping: concentration -> trial -> frame csv paths.

    Expected structure:
      root/
        <concA>/
          <trial1>/
            frame_001.csv
            ...
          <trial2>/
        <concB>/
          ...
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    def _extract_conc(name: str) -> float:
        pats = [
            r'(\d+(?:\.\d+)?)\s*ppm', r'(\d+(?:\.\d+)?)\s*ppb', r'(\d+(?:\.\d+)?)\s*%',
            r'conc_?(\d+(?:\.\d+)?)', r'(\d+(?:\.\d+)?)'
        ]
        for p in pats:
            m = re.search(p, name.lower())
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, TypeError, AttributeError):
                    continue
        return 0.0

    mapping: Dict[float, Dict[str, List[str]]] = {}
    for conc_name in os.listdir(root_dir):
        conc_path = os.path.join(root_dir, conc_name)
        if not os.path.isdir(conc_path):
            continue
        conc_val = _extract_conc(conc_name)
        conc_trials = mapping.setdefault(conc_val, {})

        # Case 1: CSV frames directly under the concentration directory
        direct_frames = [os.path.join(conc_path, f)
                         for f in os.listdir(conc_path)
                         if f.lower().endswith('.csv')]
        if direct_frames:
            trial_key = conc_name
            if trial_key in conc_trials:
                suffix = 1
                while f"{trial_key}_{suffix}" in conc_trials:
                    suffix += 1
                trial_key = f"{trial_key}_{suffix}"
            conc_trials[trial_key] = direct_frames

        # Case 2: Trial subfolders
        for trial_name in os.listdir(conc_path):
            trial_path = os.path.join(conc_path, trial_name)
            if not os.path.isdir(trial_path):
                continue
            frames = [os.path.join(trial_path, f) for f in os.listdir(trial_path) if f.lower().endswith('.csv')]
            if frames:
                trial_key = f"{conc_name}/{trial_name}"
                if trial_key in conc_trials:
                    suffix = 1
                    while f"{trial_key}_{suffix}" in conc_trials:
                        suffix += 1
                    trial_key = f"{trial_key}_{suffix}"
                conc_trials[trial_key] = frames

    if not mapping:
        raise ValueError(f"No trials found under {root_dir}")
    return mapping


# ----------------------
# Batch inference with selected multivariate model
# ----------------------

def _load_canonical_from_saved_dir(out_root: str) -> Dict[float, pd.DataFrame]:
    saved_dir = Path(out_root) / 'stable_selected'
    canonical: Dict[float, pd.DataFrame] = {}
    if not saved_dir.exists():
        return canonical
    for f in sorted(saved_dir.glob('*.csv')):
        name = f.stem  # e.g., "0.5_stable"
        try:
            conc_str = name.split('_')[0]
            conc_val = float(conc_str)
        except Exception:
            continue
        try:
            df = pd.read_csv(f)
            if 'wavelength' in df.columns and ('intensity' in df.columns or 'transmittance' in df.columns):
                canonical[conc_val] = df
        except Exception:
            continue
    return canonical


def _collect_csv_files(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith('.csv'):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _prepare_sample_vector(df: pd.DataFrame, target_wl: np.ndarray, ref_df: Optional[pd.DataFrame]) -> np.ndarray:
    # Optionally compute transmittance and apply preprocessing similar to training
    calib_settings = CONFIG.get('calibration', {})
    use_trans = bool(calib_settings.get('use_transmittance', True))
    if ref_df is not None and use_trans:
        df = compute_transmittance(df, ref_df)
    df = _preprocess_dataframe(df, stage='transmittance')
    col = _signal_column(df)
    wl = df['wavelength'].to_numpy()
    sig = df[col].to_numpy()
    # Interpolate to target grid
    vec = np.interp(target_wl, wl, sig)
    return vec


def _apply_feature_prep_matrix(X: np.ndarray, wl: np.ndarray, cfg: Dict[str, object]) -> np.ndarray:
    prep = str(cfg.get('feature_prep', 'raw')).lower()
    Xp = X.copy()
    if prep in ('derivative', 'first_derivative') or 'derivative' in prep:
        Xp = np.gradient(Xp, wl, axis=1)
    if prep == 'snv' or 'snv' in prep:
        mu = Xp.mean(axis=1, keepdims=True)
        sd = Xp.std(axis=1, keepdims=True)
        sd[sd < 1e-9] = 1.0
        Xp = (Xp - mu) / sd
    elif prep == 'mean_center':
        mu = Xp.mean(axis=0, keepdims=True)
        Xp = Xp - mu
    return Xp


def predict_batch_with_selected_model(predict_dir: str, ref_path: Optional[str], out_root: str) -> Optional[str]:
    """Predict concentrations for CSV spectra in predict_dir using the selected multivariate model.

    Saves metrics/predictions_batch.csv and returns its path, or None on failure.
    """
    try:
        files = _collect_csv_files(predict_dir)
        if not files:
            return None
        # Determine selected method
        selected = None
        sel_json_path = Path(out_root) / 'metrics' / 'multivariate_selection.json'
        calib_json_path = Path(out_root) / 'metrics' / 'calibration_metrics.json'
        if sel_json_path.exists():
            try:
                with sel_json_path.open('r', encoding='utf-8') as f:
                    sel_obj = json.load(f)
                bm = sel_obj.get('best_method')
                if bm in {'plsr', 'ica', 'mcr_als'}:
                    selected = bm
            except Exception:
                pass
        if selected is None and calib_json_path.exists():
            try:
                with calib_json_path.open('r', encoding='utf-8') as f:
                    cm = json.load(f)
                sm = str(cm.get('selected_model', ''))
                if sm.startswith('plsr'):
                    selected = 'plsr'
                elif sm.startswith('ica'):
                    selected = 'ica'
                elif sm.startswith('mcr'):
                    selected = 'mcr_als'
            except Exception:
                pass
        if selected is None:
            return None

        ref_df = load_reference_csv(ref_path) if ref_path else None
        rows = []

        if selected == 'ica':
            # Load ICA metrics
            path = Path(out_root) / 'metrics' / 'deconvolution_ica.json'
            if not path.exists():
                return None
            obj = json.loads(path.read_text())
            wl = np.array(obj.get('wavelengths', []), dtype=float)
            comps = np.array(obj.get('components', []), dtype=float)
            bi = int(obj.get('best_component', 0))
            k = float(obj.get('best_linear_k', float('nan')))
            b = float(obj.get('best_linear_b', float('nan')))
            if not comps.size or not wl.size or bi >= comps.shape[0] or not np.isfinite(k):
                return None
            basis = comps[bi]
            denom = float(np.dot(basis, basis)) if basis.size else 0.0
            if denom <= 0:
                return None
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    s_amp = float(np.dot(vec, basis) / denom)
                    y_pred = b + k * s_amp
                    rows.append({'file': fp, 'predicted': float(y_pred), 'method': 'ica', 'component': bi, 'amplitude': s_amp})
                except Exception:
                    continue

        elif selected == 'mcr_als':
            # Load MCR metrics
            path = Path(out_root) / 'metrics' / 'deconvolution_mcr_als.json'
            if not path.exists():
                return None
            obj = json.loads(path.read_text())
            wl = np.array(obj.get('wavelengths', []), dtype=float)
            comps = np.array(obj.get('components', []), dtype=float)  # comps x wl
            bi = int(obj.get('best_component', 0))
            k = float(obj.get('best_linear_k', float('nan')))
            b = float(obj.get('best_linear_b', float('nan')))
            if not comps.size or not wl.size or bi >= comps.shape[0] or not np.isfinite(k):
                return None
            S = comps  # basis matrix
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    # NNLS to get contributions
                    c, _ = nnls(S.T, vec)
                    amp = float(c[bi]) if bi < len(c) else float('nan')
                    y_pred = b + k * amp if np.isfinite(amp) else float('nan')
                    rows.append({'file': fp, 'predicted': float(y_pred), 'method': 'mcr_als', 'component': bi, 'amplitude': amp})
                except Exception:
                    continue

        else:  # PLSR
            # Rebuild PLSR on saved canonical and predict
            canonical = _load_canonical_from_saved_dir(out_root)
            if not canonical:
                return None
            mv_cfg = CONFIG.get('calibration', {}).get('multivariate', {})
            pm = _fit_plsr_calibration(canonical, mv_cfg)
            if not pm:
                return None
            wl = np.array(pm.get('wavelengths', []), dtype=float)
            if wl.size == 0:
                return None
            # Build X_train on wl subset and fit final model
            X_train, y_train, wl_base = _build_feature_matrix_from_canonical(canonical)
            # Restrict to wl subset
            # Interpolate X_train rows to selected wl grid
            Xw = []
            for conc, df in sorted(canonical.items(), key=lambda kv: kv[0]):
                v = _prepare_sample_vector(df, wl, ref_df=None)  # already stable canonical; do not recompute trans
                Xw.append(v)
            Xw = np.vstack(Xw)
            Xw = _apply_feature_prep_matrix(Xw, wl, mv_cfg)
            n_comp = int(pm.get('n_components', 1))
            pls = PLSRegression(n_components=n_comp, scale=bool(mv_cfg.get('scale', True)))
            pls.fit(Xw, y_train)
            for fp in files:
                try:
                    df = _read_csv_spectrum(fp)
                    vec = _prepare_sample_vector(df, wl, ref_df)
                    vec2 = _apply_feature_prep_matrix(vec.reshape(1, -1), wl, mv_cfg)
                    y_pred = float(pls.predict(vec2).ravel()[0])
                    rows.append({'file': fp, 'predicted': y_pred, 'method': 'plsr'})
                except Exception:
                    continue

        if not rows:
            return None
        out_csv = Path(out_root) / 'metrics' / 'predictions_batch.csv'
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return str(out_csv)
    except Exception:
        return None


# ----------------------
# Transmittance
# ----------------------

def compute_transmittance(sample_df: pd.DataFrame, ref_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute transmittance T = I_sample / I_ref on wavelength grid of sample."""
    if sample_df.empty:
        return sample_df
    if ref_df is None or ref_df.empty:
        return sample_df
    if 'intensity' not in sample_df.columns:
        # Already in transmittance or another signal space
        return sample_df

    ref_int = np.interp(sample_df['wavelength'].values, ref_df['wavelength'].values, ref_df['intensity'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.clip(np.where(ref_int != 0, sample_df['intensity'].values / ref_int, 0.0), 0.0, 1.0)
    out = sample_df.copy()
    out['transmittance'] = T
    return out


def _append_absorbance_column(df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
    """Ensure an absorbance column A = -log10(T) exists when transmittance is available."""
    if 'transmittance' not in df.columns:
        return df if inplace else df.copy()

    target = df if inplace else df.copy()
    trans = target['transmittance'].to_numpy(dtype=float, copy=True)
    trans = np.clip(trans, 1e-6, None)
    target['absorbance'] = -np.log10(trans)
    return target


def _ensure_odd_window(window: int) -> int:
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    return w


def _apply_signal_strategy(df: pd.DataFrame, signal: str) -> pd.DataFrame:
    strategies = CONFIG.get('analysis', {}).get('signal_strategies', {}) if isinstance(CONFIG, dict) else {}
    strat = strategies.get(signal, {}) if isinstance(strategies, dict) else {}
    if not strat:
        return df

    out = df.copy(deep=True)
    wl = out['wavelength'].to_numpy(dtype=float, copy=True)
    y = out[signal].to_numpy(dtype=float, copy=True)

    smooth_cfg = strat.get('smooth', {}) if isinstance(strat, dict) else {}
    if smooth_cfg.get('enabled', False):
        method = smooth_cfg.get('method', 'savgol')
        window = _ensure_odd_window(smooth_cfg.get('window', 11))
        poly = int(max(1, smooth_cfg.get('poly_order', 3)))
        y = smooth_spectrum(y, window=window, poly_order=poly, method=method)

    baseline_cfg = strat.get('baseline', {}) if isinstance(strat, dict) else {}
    if baseline_cfg.get('enabled', False):
        method = baseline_cfg.get('method', 'als')
        order = int(baseline_cfg.get('order', 2))
        y = baseline_correction(wl, y, method=method, poly_order=order)

    normalize_cfg = strat.get('normalize', {}) if isinstance(strat, dict) else {}
    if normalize_cfg.get('enabled', False):
        method = normalize_cfg.get('method', 'minmax')
        y = normalize_spectrum(y, method=method)

    clip_cfg = strat.get('clip', {}) if isinstance(strat, dict) else {}
    if clip_cfg.get('enabled', False):
        vmin = clip_cfg.get('min', None)
        vmax = clip_cfg.get('max', None)
        y = np.clip(y, vmin if vmin is not None else y, vmax if vmax is not None else y)

    center_cfg = strat.get('center', {}) if isinstance(strat, dict) else {}
    if center_cfg.get('enabled', False):
        mode = str(center_cfg.get('mode', 'mean')).lower()
        if mode == 'median':
            y = y - float(np.nanmedian(y))
        else:
            y = y - float(np.nanmean(y))

    out[signal] = y
    return out


def _build_signal_views(processed: Dict[float, Dict[str, pd.DataFrame]],
                        raw: Dict[float, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[float, Dict[str, pd.DataFrame]]]:
    signal_views: Dict[str, Dict[float, Dict[str, pd.DataFrame]]] = {}
    for signal in ('intensity', 'transmittance', 'absorbance'):
        view_map: Dict[float, Dict[str, pd.DataFrame]] = {}
        source = raw if (signal == 'intensity' and raw) else processed
        for conc, trials in source.items():
            trial_views: Dict[str, pd.DataFrame] = {}
            for name, df in trials.items():
                if signal not in df.columns:
                    continue
                cols = ['wavelength', signal]
                extracted = df[cols].copy(deep=True)
                prepared = _apply_signal_strategy(extracted, signal)
                trial_views[name] = prepared
            if trial_views:
                view_map[conc] = trial_views
        if view_map:
            signal_views[signal] = view_map
    return signal_views


def _resolve_primary_signal(signal_views: Dict[str, Dict[float, Dict[str, pd.DataFrame]]]) -> str:
    analysis_cfg = CONFIG.get('analysis', {}) if isinstance(CONFIG, dict) else {}
    preferred = str(analysis_cfg.get('primary_signal', '') or '').lower()
    candidates = [preferred] if preferred else []
    if analysis_cfg.get('enable_absorbance', False):
        candidates.append('absorbance')
    candidates.extend(['transmittance', 'intensity'])
    for candidate in candidates:
        if candidate and candidate in signal_views:
            return candidate
    return next(iter(signal_views.keys())) if signal_views else 'intensity'


def compute_transmittance_on_frames(frames: List[pd.DataFrame], ref_df: pd.DataFrame) -> List[pd.DataFrame]:
    return [compute_transmittance(df, ref_df) for df in frames]


# ----------------------
# Stability on multi-frame spectral trials
# ----------------------

def _align_on_grid(frames: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (wl, Y, signal_col) using a signal present across all frames."""
    if not frames:
        raise ValueError("No frames provided for alignment")

    base = frames[0]
    wl = base['wavelength'].values
    signal_col = _select_common_signal(frames)
    if signal_col is None:
        signal_col = _signal_column(base)

    Y = []
    for df in frames:
        col = signal_col if signal_col in df.columns else _signal_column(df)
        vec = df[col].values
        if not np.array_equal(df['wavelength'].values, wl):
            vec = np.interp(wl, df['wavelength'].values, vec)
        Y.append(vec)

    return wl, np.vstack(Y), signal_col if signal_col in base.columns else _signal_column(base)


def find_stable_block(frames: List[pd.DataFrame],
                      diff_threshold: float = 0.01,
                      weight_mode: str = 'uniform',
                      top_k: Optional[int] = None,
                      min_block: Optional[int] = None,
                      **_unused: object) -> Tuple[int, int, np.ndarray]:
    """Identify a stable frame range and optional weighting for averaging.

    Returns the start/end indices of the longest stable block along with per-frame
    weights that bias averaging toward stronger intensity frames if requested."""
    wl, Y, _ = _align_on_grid(frames)

    if Y.shape[0] == 1:
        return 0, 0, np.array([1.0])
    diffs = []
    eps = 1e-9
    for i in range(1, Y.shape[0]):
        prev, curr = Y[i-1], Y[i]
        rng = max(prev.max() - prev.min(), eps)
        mad = np.mean(np.abs(curr - prev)) / rng
        diffs.append(mad)
    diffs = np.array(diffs)
    stable_flags = diffs < diff_threshold
    frame_stable = np.ones(Y.shape[0], dtype=bool)
    frame_stable[1:] = stable_flags

    best_len = 0
    best_start = 0
    curr_len = 0
    curr_start = 0
    for i, f in enumerate(frame_stable):
        if f:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > best_len:
                best_len = curr_len
                best_start = curr_start
        else:
            curr_len = 0
    if best_len <= 1:
        mid = Y.shape[0] // 2
        best_start = max(0, mid - 1)
        best_len = min(3, Y.shape[0] - best_start)

    start_idx = best_start
    end_idx = best_start + best_len - 1

    if min_block and min_block > 0:
        desired = min(int(min_block), Y.shape[0])
        current = end_idx - start_idx + 1
        if current < desired:
            padding = desired - current
            pad_left = padding // 2
            pad_right = padding - pad_left
            start_idx = max(0, start_idx - pad_left)
            end_idx = min(Y.shape[0] - 1, end_idx + pad_right)
            current = end_idx - start_idx + 1
            if current < desired:
                start_idx = max(0, end_idx - desired + 1)

    frame_window = np.zeros(Y.shape[0], dtype=bool)
    frame_window[start_idx:end_idx + 1] = True

    weights = np.zeros(Y.shape[0], dtype=float)
    weights[frame_window] = 1.0

    weight_mode = weight_mode.lower() if isinstance(weight_mode, str) else 'uniform'

    if weight_mode in {'intensity', 'max'}:
        block = Y[frame_window]
        if block.size > 0:
            if weight_mode == 'intensity':
                frame_scores = block.mean(axis=1)
            else:  # 'max'
                frame_scores = block.max(axis=1)
            frame_scores = np.clip(frame_scores, 1e-9, None)
            weights[frame_window] = frame_scores

    if top_k and top_k > 0:
        block_indices = np.flatnonzero(frame_window)
        if block_indices.size > top_k:
            block = Y[frame_window]
            frame_scores = block.mean(axis=1)
            top_idx = np.argsort(frame_scores)[-top_k:]
            selected = np.zeros_like(frame_window)
            selected[block_indices[top_idx]] = True
            frame_window = selected
            weights = np.zeros_like(weights)
            weights[frame_window] = 1.0

    total_weight = weights.sum()
    if total_weight <= 0:
        weights[frame_window] = 1.0
    return start_idx, end_idx, weights


def average_stable_block(frames: List[pd.DataFrame],
                         start_idx: int,
                         end_idx: int,
                         weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    base_wl = frames[0]['wavelength'].values
    start_idx = max(0, min(start_idx, len(frames) - 1))
    end_idx = max(start_idx, min(end_idx, len(frames) - 1))

    frame_indices = list(range(start_idx, end_idx + 1))
    if not frame_indices:
        return pd.DataFrame({'wavelength': base_wl})

    if weights is not None and weights.size == len(frames):
        frame_weights = weights[frame_indices]
    else:
        frame_weights = None

    common_cols = _common_signal_columns(frames)
    out = pd.DataFrame({'wavelength': base_wl})

    for col in common_cols:
        accum = np.zeros_like(base_wl, dtype=float)
        total_weight = 0.0
        for idx, frame_idx in enumerate(frame_indices):
            df = frames[frame_idx]
            if col not in df.columns:
                continue
            weight = frame_weights[idx] if frame_weights is not None else 1.0
            if weight <= 0:
                continue
            vec = df[col].to_numpy(dtype=float, copy=True)
            wl = df['wavelength'].to_numpy(dtype=float, copy=True)
            if not np.array_equal(wl, base_wl):
                vec = np.interp(base_wl, wl, vec)
            accum += weight * vec
            total_weight += weight
        if total_weight > 0:
            out[col] = accum / total_weight

    # If no common columns were averaged, fall back to the signal column of the first frame
    if len(out.columns) == 1:
        col = _signal_column(frames[0])
        accum = np.zeros_like(base_wl, dtype=float)
        total_weight = 0.0
        for idx, frame_idx in enumerate(frame_indices):
            df = frames[frame_idx]
            weight = frame_weights[idx] if frame_weights is not None else 1.0
            if weight <= 0 or col not in df.columns:
                continue
            vec = df[col].to_numpy(dtype=float, copy=True)
            wl = df['wavelength'].to_numpy(dtype=float, copy=True)
            if not np.array_equal(wl, base_wl):
                vec = np.interp(base_wl, wl, vec)
            accum += weight * vec
            total_weight += weight
        if total_weight > 0:
            out[col] = accum / total_weight

    return out


def average_top_frames(frames: List[pd.DataFrame], top_k: int = 5) -> pd.DataFrame:
    """Average the first `top_k` frames using intensity values before transmittance conversion."""
    if not frames:
        return pd.DataFrame()
    top_k = max(1, min(len(frames), top_k))
    selected = frames[:top_k]
    base_wl = selected[0]['wavelength'].values
    signal_col = 'intensity' if all('intensity' in df.columns for df in selected) else _select_common_signal(selected, ('intensity', 'transmittance', 'absorbance'))
    if signal_col is None:
        return pd.DataFrame({'wavelength': base_wl})

    accum = np.zeros_like(base_wl, dtype=float)
    count = 0
    for df in selected:
        wl = df['wavelength'].values
        if signal_col not in df.columns:
            continue
        values = df[signal_col].values
        if not np.array_equal(wl, base_wl):
            values = np.interp(base_wl, wl, values)
        accum += values
        count += 1
    avg_intensity = accum / max(count, 1)
    result = pd.DataFrame({'wavelength': base_wl})
    result[signal_col] = avg_intensity
    if signal_col != 'intensity':
        # Retain compatibility with downstream steps expecting an intensity column
        result['intensity'] = avg_intensity
    if signal_col == 'transmittance' and 'absorbance' not in result.columns:
        result = _append_absorbance_column(result)
    return result


def compute_roi_linearity(df: pd.DataFrame,
                          concentrations: List[float],
                          response_metric: str = 'transmittance') -> Dict[str, float]:
    """Compute linearity metrics for a single averaged spectrum across concentrations."""
    if response_metric not in df.columns:
        raise ValueError(f"Column '{response_metric}' not found in dataframe")
    wl = df['wavelength'].values
    signal = df[response_metric].values
    roi_cfg = CONFIG.get('roi', {})
    min_wl = roi_cfg.get('min_wavelength', wl.min())
    max_wl = roi_cfg.get('max_wavelength', wl.max())
    mask = (wl >= min_wl) & (wl <= max_wl)
    wl_roi = wl[mask]
    signal_roi = signal[mask]
    slope, intercept, r_value, _, _ = linregress(wl_roi, signal_roi)
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r2': float(r_value ** 2),
        'start_wavelength': float(wl_roi[0]) if wl_roi.size else float('nan'),
        'end_wavelength': float(wl_roi[-1]) if wl_roi.size else float('nan'),
    }


# ----------------------
# Canonical selection per concentration
# ----------------------

def select_canonical_per_concentration(stable_results: Dict[float, Dict[str, pd.DataFrame]]) -> Dict[float, pd.DataFrame]:
    """Aggregate trials per concentration using robust outlier rejection and weighted averaging."""
    canonical: Dict[float, pd.DataFrame] = {}
    for conc, trials in stable_results.items():
        if not trials:
            continue

        base_wl: Optional[np.ndarray] = None
        accum: Dict[str, np.ndarray] = {}
        weight_sums: Dict[str, float] = {}

        for trial_name, df in trials.items():
            wl = df['wavelength'].to_numpy()
            if base_wl is None:
                base_wl = wl
            elif not np.array_equal(base_wl, wl):
                # interpolate onto the first trial's wavelength grid when needed
                pass

            # Optional per-trial weight (e.g. quality score); defaults to 1.0
            weight = 1.0
            try:
                # trial_weights may be provided in the caller via a closure or outer scope
                trial_weights: Optional[Dict[float, Dict[str, float]]] = globals().get('TRIAL_WEIGHTS_FOR_CANONICAL')  # type: ignore[assignment]
            except Exception:
                trial_weights = None
            if isinstance(trial_weights, dict):
                conc_weights = trial_weights.get(conc, {}) if isinstance(trial_weights.get(conc, {}), dict) else {}
                w_val = conc_weights.get(trial_name, 1.0)
                try:
                    weight = float(w_val)
                except Exception:
                    weight = 1.0
            if not np.isfinite(weight) or weight <= 0.0:
                continue

            for col in df.columns:
                if col == 'wavelength':
                    continue
                series = df[col].to_numpy()
                if base_wl is not None and not np.array_equal(base_wl, wl):
                    series = np.interp(base_wl, wl, series)
                arr = series.astype(float)
                if col not in accum:
                    accum[col] = weight * arr
                    weight_sums[col] = weight
                else:
                    accum[col] += weight * arr
                    weight_sums[col] += weight

        if base_wl is None or not accum:
            continue

        canonical_df = pd.DataFrame({'wavelength': base_wl})
        for col, s in accum.items():
            wsum = weight_sums.get(col, 0.0)
            if wsum > 0.0:
                canonical_df[col] = s / wsum
        canonical[conc] = canonical_df

    return canonical


def _baseline_correct_canonical(canonical: Dict[float, pd.DataFrame]) -> Dict[float, pd.DataFrame]:
    # Return canonical spectra without subtracting the lowest concentration.
    # Previous behavior used the minimum concentration (e.g., 1 ppm) as a
    # surrogate “zero” baseline, which inverted the calibration trend when the
    # true baseline should be an external air reference. Until we wire in the
    # dedicated zero-gas reference, keep spectra in absolute wavelength space.
    return canonical


# ----------------------
# ROI and calibration (wavelength shift)
# ----------------------

def _smooth(y: np.ndarray, window: int = 11, poly: int = 2) -> np.ndarray:
    window = max(3, window if window % 2 == 1 else window + 1)
    window = min(window, len(y) - 1 if len(y) % 2 == 0 else len(y))
    window = max(3, window)
    try:
        return savgol_filter(y, window_length=window, polyorder=min(poly, window-1))
    except Exception:
        return y


def _apply_wavelength_limits(x: np.ndarray,
                             y: np.ndarray,
                             min_wl: Optional[float] = None,
                             max_wl: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.ones_like(x, dtype=bool)
    if min_wl is not None:
        mask &= (x >= min_wl)
    if max_wl is not None:
        mask &= (x <= max_wl)
    if mask.sum() == 0:
        return x, y
    return x[mask], y[mask]


def _resolve_roi_bounds(dataset_label: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
    min_global = roi_cfg.get('min_wavelength')
    max_global = roi_cfg.get('max_wavelength')
    overrides = roi_cfg.get('per_gas_overrides', {}) if isinstance(roi_cfg, dict) else {}
    if dataset_label and isinstance(overrides, dict):
        entry = overrides.get(dataset_label, {})
        if isinstance(entry, dict):
            rng = entry.get('range', {})
            if isinstance(rng, dict):
                min_override = rng.get('min_wavelength', min_global)
                max_override = rng.get('max_wavelength', max_global)
                return min_override, max_override
    return min_global, max_global


def _resolve_expected_center(dataset_label: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
    overrides = roi_cfg.get('per_gas_overrides', {}) if isinstance(roi_cfg, dict) else {}
    if dataset_label and isinstance(overrides, dict):
        entry = overrides.get(dataset_label, {})
        if isinstance(entry, dict):
            validation = entry.get('validation', {})
            if isinstance(validation, dict):
                return validation.get('expected_center'), validation.get('tolerance')
    validation_cfg = roi_cfg.get('validation', {}) if isinstance(roi_cfg, dict) else {}
    if isinstance(validation_cfg, dict):
        return validation_cfg.get('expected_center'), validation_cfg.get('tolerance')
    return None, None


def _refine_centroid_with_derivative(df: pd.DataFrame,
                                      centroid_cfg: Optional[Dict[str, object]],
                                      expected_center: Optional[float],
                                      span_nm: float = 2.0,
                                      smooth_window: int = 7) -> float:
    if expected_center is None or not np.isfinite(expected_center):
        return float('nan')

    centroid_cfg = centroid_cfg or {}
    try:
        span_nm = float(centroid_cfg.get('derivative_span_nm', span_nm) or span_nm)
    except Exception:
        span_nm = span_nm
    try:
        smooth_window = int(centroid_cfg.get('derivative_smooth_window', smooth_window) or smooth_window)
    except Exception:
        smooth_window = smooth_window

    if span_nm <= 0:
        return float('nan')

    x, y = _prepare_calibration_signal(df, centroid_cfg)
    if x.size < 3:
        return float('nan')

    mask = (x >= expected_center - span_nm) & (x <= expected_center + span_nm)
    if np.count_nonzero(mask) < 3:
        idx_closest = int(np.argmin(np.abs(x - expected_center)))
        return float(x[idx_closest])

    xx = x[mask]
    yy = y[mask]

    # Ensure odd window length within range
    smooth_window = max(5, smooth_window)
    smooth_window = _ensure_odd_window(smooth_window)
    if smooth_window >= xx.size:
        smooth_window = xx.size - 1 if xx.size % 2 == 0 else xx.size
    smooth_window = max(3, smooth_window)
    if smooth_window >= xx.size:
        # if still too large, fall back to gradient
        smooth_window = 0

    delta = float(np.mean(np.diff(xx))) if xx.size > 1 else 1.0
    try:
        if smooth_window >= 3:
            deriv = savgol_filter(
                yy,
                window_length=smooth_window,
                polyorder=min(3, smooth_window - 1),
                deriv=1,
                delta=delta,
                mode='interp',
            )
        else:
            raise RuntimeError('window_too_small')
    except Exception:
        deriv = np.gradient(yy, xx)

    idx_peak = int(np.argmax(np.abs(deriv)))
    idx_peak = int(np.clip(idx_peak, 0, xx.size - 1))
    return float(xx[idx_peak])


def _prepare_calibration_signal(df: pd.DataFrame,
                                centroid_cfg: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, np.ndarray]:
    centroid_cfg = centroid_cfg or {}
    ycol = _signal_column(df)
    if ycol not in df.columns:
        # Fallback: use first non-wavelength column if present
        candidates = [c for c in df.columns if c != 'wavelength']
        if not candidates:
            raise KeyError("No signal column available for calibration")
        ycol = candidates[0]
    x = df['wavelength'].values
    y = _smooth(df[ycol].values)
    min_wl = centroid_cfg.get('min_wavelength')
    max_wl = centroid_cfg.get('max_wavelength')
    if min_wl is not None or max_wl is not None:
        x, y = _apply_wavelength_limits(x, y, min_wl, max_wl)
    return x, y


def _estimate_shift_crosscorr(ref_wl: np.ndarray,
                              ref_signal: np.ndarray,
                              target_signal: np.ndarray,
                              upsample: int = 1) -> float:
    if len(ref_wl) != len(ref_signal) or len(ref_signal) != len(target_signal):
        raise ValueError("Mismatch between reference and target signal lengths for cross-correlation")

    if len(ref_wl) < 2:
        return 0.0

    if upsample > 1:
        dense_wl = np.linspace(ref_wl[0], ref_wl[-1], len(ref_wl) * upsample)
        ref = np.interp(dense_wl, ref_wl, ref_signal)
        tgt = np.interp(dense_wl, ref_wl, target_signal)
        dw = float(np.mean(np.diff(dense_wl)))
    else:
        dense_wl = ref_wl
        ref = ref_signal.copy()
        tgt = target_signal.copy()
        dw = float(np.mean(np.diff(ref_wl))) if len(ref_wl) > 1 else 0.0

    ref -= np.mean(ref)
    tgt -= np.mean(tgt)
    corr = correlate(tgt, ref, mode='full')
    lags = np.arange(-len(ref) + 1, len(tgt))
    best_lag = float(lags[np.argmax(corr)])
    return -best_lag * dw


def _gaussian_peak_center(x: np.ndarray,
                          y: np.ndarray,
                          idx_hint: Optional[int] = None,
                          half_width: int = 5) -> float:
    """Estimate sub-sample peak center using a Gaussian fit in a local window.

    Handles both maxima and minima by allowing negative amplitude. Falls back to
    centroid when fitting fails.
    """
    if x.size < 3:
        return float(x[idx_hint] if idx_hint is not None else (x[x.size//2] if x.size else np.nan))

    # initial index around apex
    if idx_hint is None:
        idx_hint = int(np.argmax(np.abs(y - np.median(y))))
    idx_hint = int(np.clip(idx_hint, 0, x.size - 1))

    half_width = max(1, min(int(half_width), x.size // 2))
    s = max(0, idx_hint - half_width)
    e = min(x.size - 1, idx_hint + half_width)
    xx = x[s:e+1]
    yy = y[s:e+1]

    # Guess baseline and amplitude
    baseline = float(np.median(yy))
    is_min = bool(yy.min() < (np.median(yy) - 0.25 * (yy.max() - yy.min())))
    if is_min:
        A0 = float(yy.min() - baseline)
        idx0 = int(np.argmin(yy))
    else:
        A0 = float(yy.max() - baseline)
        idx0 = int(np.argmax(yy))
    x0_0 = float(xx[idx0])
    sigma0 = max((xx[-1] - xx[0]) / 6.0, 1e-3)

    def gauss(xv, A, x0, sigma, C):
        sigma = max(sigma, 1e-6)
        return C + A * np.exp(-0.5 * ((xv - x0) / sigma) ** 2)

    p0 = [A0, x0_0, sigma0, baseline]
    bounds = ([-np.inf, xx[0]-5.0, 1e-6, -np.inf], [np.inf, xx[-1]+5.0, (xx[-1]-xx[0])*2.0, np.inf])
    try:
        popt, _ = curve_fit(gauss, xx, yy, p0=p0, bounds=bounds, maxfev=5000)
        x0 = float(popt[1])
        # clamp to window if needed
        if not np.isfinite(x0):
            raise RuntimeError('non-finite center')
        if x0 < xx[0] - 1.0 or x0 > xx[-1] + 1.0:
            raise RuntimeError('center outside window')
        return x0
    except Exception:
        # fallback to centroid inside window
        weights = np.abs(yy - baseline) + 1e-9
        return float(np.sum(xx * weights) / np.sum(weights))


def _compute_band_ratio_matrix(Y: np.ndarray, half_width: int) -> np.ndarray:
    n_samples, n_wl = Y.shape
    if n_wl == 0:
        return np.empty_like(Y)
    half_width = max(1, int(half_width))
    ratios = np.empty_like(Y, dtype=float)
    eps = 1e-12
    for i in range(n_samples):
        row = Y[i]
        for j in range(n_wl):
            left_start = max(0, j - half_width)
            left_end = j + 1
            right_start = j
            right_end = min(n_wl, j + half_width + 1)

            left_segment = row[left_start:left_end]
            right_segment = row[right_start:right_end]

            left_mean = float(np.mean(left_segment)) if left_segment.size else float(row[j])
            right_mean = float(np.mean(right_segment)) if right_segment.size else float(row[j])

            denom = right_mean if abs(right_mean) > eps else (eps if right_mean >= 0 else -eps)
            ratios[i, j] = left_mean / denom
    return ratios


def _find_peak_wavelength(df: pd.DataFrame, centroid_cfg: Optional[Dict[str, object]] = None) -> float:
    centroid_cfg = centroid_cfg or {}
    x, y = _prepare_calibration_signal(df, centroid_cfg)

    # Decide if peak is max or min using overall skew
    is_min_peak = (y.min() < (np.median(y) - 0.25 * (y.max() - y.min())))
    centroid_hint = centroid_cfg.get('centroid_hint')
    if centroid_hint is not None and np.isfinite(centroid_hint):
        idx = int(np.argmin(np.abs(x - centroid_hint)))
    else:
        idx = int(np.argmin(y) if is_min_peak else np.argmax(y))

    half_width = int(centroid_cfg.get('centroid_half_width', 5))
    half_width = max(1, min(half_width, len(x) // 2))
    s = max(0, idx - half_width)
    e = min(len(x) - 1, idx + half_width)
    xx = x[s:e+1]
    yy = y[s:e+1]

    weight_mode = str(centroid_cfg.get('centroid_weight', 'contrast')).lower()
    contrast = np.abs(yy - np.median(yy)) + 1e-9

    if is_min_peak:
        base_weight = (yy.max() - yy) + 1e-9
    else:
        base_weight = (yy - yy.min()) + 1e-9

    if weight_mode == 'contrast':
        weights = base_weight * contrast
    elif weight_mode == 'uniform':
        weights = np.ones_like(base_weight)
    else:
        weights = base_weight

    lam = float(np.sum(xx * weights) / np.sum(weights))
    return lam


def _transform_concentrations(concs: np.ndarray, mode: str) -> Tuple[np.ndarray, Dict[str, float]]:
    meta: Dict[str, float] = {}
    mode = mode.lower()
    if mode == 'log10':
        safe = np.where(concs <= 0, np.nan, concs)
        min_positive = np.nanmin(safe)
        offset = min_positive * 0.1 if min_positive and not np.isnan(min_positive) else 1e-3
        meta['offset'] = offset
        transformed = np.log10(np.clip(concs, offset, None))
    elif mode == 'log':
        safe = np.where(concs <= 0, np.nan, concs)
        min_positive = np.nanmin(safe)
        offset = min_positive * 0.1 if min_positive and not np.isnan(min_positive) else 1e-3
        meta['offset'] = offset
        transformed = np.log(np.clip(concs, offset, None))
    elif mode == 'sqrt':
        transformed = np.sqrt(np.clip(concs, 0.0, None))
    else:
        transformed = concs
    return transformed, meta


def _measure_peak_within_window(df: pd.DataFrame,
                                center_nm: float,
                                window_nm: float,
                                centroid_cfg: Optional[Dict[str, object]] = None) -> float:
    df_sorted = df.sort_values('wavelength').reset_index(drop=True)
    if df_sorted.empty:
        return float('nan')

    half = max(window_nm / 2.0, 0.1)
    min_wl = center_nm - half
    max_wl = center_nm + half
    mask = (df_sorted['wavelength'] >= min_wl) & (df_sorted['wavelength'] <= max_wl)
    subset = df_sorted.loc[mask].copy()

    expand_step = 0.5
    expand_limit = max(window_nm, 6.0)
    expand = expand_step
    while subset.empty and expand <= expand_limit:
        lower = min_wl - expand
        upper = max_wl + expand
        mask = (df_sorted['wavelength'] >= lower) & (df_sorted['wavelength'] <= upper)
        subset = df_sorted.loc[mask].copy()
        expand += expand_step

    if subset.empty:
        return float('nan')

    cfg = dict(centroid_cfg or {})
    cfg['min_wavelength'] = subset['wavelength'].min()
    cfg['max_wavelength'] = subset['wavelength'].max()
    cfg['centroid_hint'] = center_nm
    cfg.setdefault('centroid_half_width', max(3, int(np.ceil(subset.shape[0] / 6.0))))

    try:
        return _find_peak_wavelength(subset, centroid_cfg=cfg)
    except Exception:
        try:
            signal_col = _signal_column(subset)
            wl = subset['wavelength'].to_numpy(dtype=float)
            signal = subset[signal_col].to_numpy(dtype=float)
            weights = np.abs(signal - np.median(signal)) + 1e-9
            return float(np.sum(wl * weights) / np.sum(weights))
        except Exception:
            wl = subset['wavelength'].to_numpy(dtype=float)
            return float(np.nanmean(wl)) if wl.size else float('nan')


def _evaluate_roi_candidate(canonical_items: List[Tuple[float, pd.DataFrame]],
                            center_nm: float,
                            window_nm: float,
                            centroid_cfg: Optional[Dict[str, object]],
                            gates: Dict[str, float],
                            prior_center: Optional[float],
                            prior_weight: float,
                            weights: Dict[str, float]) -> Dict[str, object]:
    concs = np.array([c for c, _ in canonical_items], dtype=float)
    peaks: List[float] = []
    deltas: List[float] = []
    baseline_peak = float('nan')

    for _, df in canonical_items:
        peak = _measure_peak_within_window(df, center_nm, window_nm, centroid_cfg)
        peaks.append(float(peak))
        if not np.isfinite(baseline_peak) and np.isfinite(peak):
            baseline_peak = float(peak)
        if np.isfinite(peak) and np.isfinite(baseline_peak):
            deltas.append(float(peak) - float(baseline_peak))
        else:
            deltas.append(float('nan'))

    peaks_arr = np.array(peaks, dtype=float)
    deltas_arr = np.array(deltas, dtype=float)
    valid_mask = np.isfinite(peaks_arr) & np.isfinite(deltas_arr) & np.isfinite(concs)
    concs_valid = concs[valid_mask]
    deltas_valid = deltas_arr[valid_mask]

    slope = float('nan')
    intercept = float('nan')
    r2 = float('nan')
    rmse = float('nan')
    residuals = np.full_like(deltas_valid, float('nan'))
    if concs_valid.size >= 2:
        try:
            slope_lin, intercept_lin, r_val_lin, _, _ = linregress(concs_valid, deltas_valid)
            preds = intercept_lin + slope_lin * concs_valid
            residuals = deltas_valid - preds
            ss_tot = np.sum((deltas_valid - np.nanmean(deltas_valid)) ** 2)
            ss_res = np.sum(residuals ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
            slope = float(slope_lin)
            intercept = float(intercept_lin)
            rmse = float(np.sqrt(ss_res / residuals.size)) if residuals.size else float('nan')
        except Exception:
            slope = float('nan')
            intercept = float('nan')
            r2 = float('nan')
            rmse = float('nan')

    consistency = float('nan')
    if concs_valid.size >= 2:
        finite_mask = np.isfinite(deltas_valid)
        if np.count_nonzero(finite_mask) > 0:
            finite_deltas = deltas_valid[finite_mask]
            sign_ref = np.sign(np.nanmedian(finite_deltas))
            if sign_ref == 0:
                sign_ref = np.sign(slope)
            if sign_ref == 0:
                sign_ref = 1.0
            same = np.sum(np.sign(finite_deltas) == sign_ref)
            consistency = float(same / finite_deltas.size)

    snr = float('nan')
    if np.isfinite(slope) and np.isfinite(rmse) and rmse > 0:
        snr = float(abs(slope) / rmse)

    gate_min_r2 = float(gates.get('min_r2', 0.7))
    gate_min_consistency = float(gates.get('min_consistency', 0.8))
    gate_min_snr = float(gates.get('min_snr', 3.0))
    gate_min_abs_slope = float(gates.get('min_abs_slope', 0.02))
    gate_min_count = int(gates.get('min_conc_count', 3))

    valid_conc_count = int(np.unique(np.round(concs_valid, decimals=6)).size)
    slope_abs = abs(slope) if np.isfinite(slope) else float('nan')

    quality_flags = {
        'min_conc_count': bool(valid_conc_count >= gate_min_count),
        'min_r2': bool(np.isfinite(r2) and r2 >= gate_min_r2),
        'min_consistency': bool(np.isfinite(consistency) and consistency >= gate_min_consistency),
        'min_snr': bool(np.isfinite(snr) and snr >= gate_min_snr),
        'min_abs_slope': bool(np.isfinite(slope_abs) and slope_abs >= gate_min_abs_slope),
    }
    quality_ok = all(quality_flags.values())

    w_r2 = float(weights.get('r2', 1.5))
    w_slope = float(weights.get('slope', 0.6))
    w_snr = float(weights.get('snr', 0.4))
    penalty = prior_weight * abs(center_nm - prior_center) if (prior_center is not None and np.isfinite(prior_center)) else 0.0
    score_components = {
        'r2': float(r2),
        'abs_slope': float(abs(slope)) if np.isfinite(slope) else float('nan'),
        'snr': float(snr) if np.isfinite(snr) else float('nan'),
    }
    score = 0.0
    if np.isfinite(r2):
        score += w_r2 * r2
    if np.isfinite(slope):
        score += w_slope * abs(slope)
    if np.isfinite(snr):
        score += w_snr * snr
    score -= penalty

    candidate = {
        'center_nm': float(center_nm),
        'window_nm': float(window_nm),
        'min_wavelength_nm': float(center_nm - window_nm / 2.0),
        'max_wavelength_nm': float(center_nm + window_nm / 2.0),
        'slope_nm_per_ppm': float(slope) if np.isfinite(slope) else float('nan'),
        'intercept_nm': float(intercept) if np.isfinite(intercept) else float('nan'),
        'r2': float(r2) if np.isfinite(r2) else float('nan'),
        'rmse_nm': float(rmse) if np.isfinite(rmse) else float('nan'),
        'snr': float(snr) if np.isfinite(snr) else float('nan'),
        'consistency': float(consistency) if np.isfinite(consistency) else float('nan'),
        'valid_points': int(concs_valid.size),
        'unique_concentrations': valid_conc_count,
        'baseline_peak_nm': float(baseline_peak) if np.isfinite(baseline_peak) else float('nan'),
        'quality_flags': quality_flags,
        'quality_ok': bool(quality_ok),
        'score': float(score),
        'score_components': score_components,
        'deltas_nm': deltas_arr.tolist(),
        'peaks_nm': peaks_arr.tolist(),
    }
    if concs_valid.size:
        candidate['concentrations_ppm'] = concs_valid.tolist()
        candidate['deltas_valid_nm'] = deltas_valid.tolist()
        candidate['residuals_nm'] = residuals.tolist() if residuals.size else []
    else:
        candidate['concentrations_ppm'] = []
        candidate['deltas_valid_nm'] = []
        candidate['residuals_nm'] = []
    return candidate


def _discover_roi_in_band(canonical: Dict[float, pd.DataFrame],
                          dataset_label: Optional[str] = None,
                          out_root: Optional[str] = None) -> Dict[str, object]:
    print("[DEBUG] _discover_roi_in_band called")
    roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
    discovery_cfg = roi_cfg.get('discovery', {}) if isinstance(roi_cfg.get('discovery', {}), dict) else {}
    if not discovery_cfg.get('enabled', False):
        return {}

    band = discovery_cfg.get('band', [600.0, 700.0])
    band_min = float(band[0]) if isinstance(band, (list, tuple)) and len(band) >= 2 else 600.0
    band_max = float(band[1]) if isinstance(band, (list, tuple)) and len(band) >= 2 else 700.0
    window_nm = float(discovery_cfg.get('window_nm', 12.0))
    step_nm_cfg = discovery_cfg.get('step_nm', 0.2)
    try:
        step_nm = float(step_nm_cfg)
    except Exception:
        step_nm = 0.2
    if step_nm <= 0:
        step_nm = 0.2
    expected_center = discovery_cfg.get('expected_center', None)
    prior_center = float(expected_center) if expected_center is not None else None
    prior_weight = float(discovery_cfg.get('prior_weight', 0.03))
    gates = discovery_cfg.get('gates', {}) if isinstance(discovery_cfg.get('gates', {}), dict) else {}
    weights = discovery_cfg.get('weights', {}) if isinstance(discovery_cfg.get('weights', {}), dict) else {}

    calib_cfg = CONFIG.get('calibration', {}) if isinstance(CONFIG, dict) else {}
    centroid_cfg = calib_cfg.copy() if isinstance(calib_cfg, dict) else {}

    canonical_items = sorted(canonical.items(), key=lambda kv: kv[0])
    if not canonical_items:
        return {}

    # Build an instrument-aware set of candidate centers from the actual wavelength grid.
    try:
        wl_all: List[float] = []
        for _, df in canonical_items:
            if 'wavelength' in df.columns:
                wl_vals = df['wavelength'].to_numpy(dtype=float)
                wl_all.append(wl_vals)
        if wl_all:
            wl_concat = np.unique(np.concatenate(wl_all))
            mask_band = (wl_concat >= band_min) & (wl_concat <= band_max)
            wl_band = wl_concat[mask_band]
        else:
            wl_band = np.array([], dtype=float)
    except Exception:
        wl_band = np.array([], dtype=float)

    if wl_band.size == 0:
        # Fallback to uniform centers if wavelength grid could not be resolved.
        centers = np.arange(band_min, band_max + step_nm / 2.0, step_nm)
    else:
        # Use step_nm as an approximate stride in units of the native wavelength spacing.
        if wl_band.size > 1:
            diffs = np.diff(wl_band)
            median_spacing = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else step_nm
            if median_spacing <= 0:
                median_spacing = step_nm
            stride = max(1, int(round(step_nm / median_spacing)))
        else:
            stride = 1
        centers = wl_band[::stride]
    candidates: List[Dict[str, object]] = []
    for center_nm in centers:
        candidate = _evaluate_roi_candidate(
            canonical_items,
            float(center_nm),
            window_nm,
            centroid_cfg,
            gates,
            prior_center,
            prior_weight,
            weights,
        )
        candidates.append(candidate)

    if not candidates:
        return {}

    candidates_sorted = sorted(candidates, key=lambda c: c.get('score', float('-inf')), reverse=True)

    min_r2_for_sensitivity = float(discovery_cfg.get('best_sensitivity_min_r2', float('nan')))
    print(f"[DEBUG] min_r2_for_sensitivity = {min_r2_for_sensitivity}")
    print(f"[DEBUG] Is finite: {np.isfinite(min_r2_for_sensitivity)}")
    top_candidate = None
    if np.isfinite(min_r2_for_sensitivity):
        print(f"[DEBUG] Entering sensitivity-first selection logic")
        best_sens_candidate: Optional[Dict[str, object]] = None
        best_abs_slope = float('-inf')
        print(f"[DEBUG] Checking {len(candidates_sorted)} candidates for sensitivity-first selection")
        for i, cand in enumerate(candidates_sorted):
            print(f"[DEBUG] Candidate {i}: ROI {cand.get('roi_range')} nm, R²={cand.get('r2'):.4f}, slope={cand.get('slope_nm_per_ppm', cand.get('slope')):.4f}")
            if not cand.get('quality_ok'):
                print(f"[DEBUG]   Failed quality_ok")
                continue
            r2_val = cand.get('r2')
            slope_val = cand.get('slope_nm_per_ppm') or cand.get('slope')  # Fix: check both keys
            if not isinstance(r2_val, (int, float)) or not np.isfinite(r2_val):
                continue
            if r2_val < min_r2_for_sensitivity:
                print(f"[DEBUG]   Failed R² threshold: {r2_val} < {min_r2_for_sensitivity}")
                continue
            if not isinstance(slope_val, (int, float)) or not np.isfinite(slope_val):
                continue
            abs_slope = abs(float(slope_val))
            if abs_slope > best_abs_slope:
                best_abs_slope = abs_slope
                best_sens_candidate = cand
                print(f"[DEBUG]   New best sensitivity: {abs_slope:.4f}")
        if best_sens_candidate is not None:
            top_candidate = best_sens_candidate
            print(f"[DEBUG] Selected sensitivity-first candidate: ROI {top_candidate.get('roi_range')} nm")
        else:
            print(f"[DEBUG] No candidate met sensitivity criteria")
    else:
        print(f"[DEBUG] Skipping sensitivity-first selection (min_r2 is NaN)")
        
    if top_candidate is None:
        print(f"[DEBUG] Using fallback selection")
        top_candidate = next((c for c in candidates_sorted if c.get('quality_ok')), candidates_sorted[0])
        print(f"[DEBUG] Fallback selected: ROI {top_candidate.get('roi_range')} nm")

    discovery = {
        'selected': top_candidate,
        'candidates': candidates_sorted[:int(discovery_cfg.get('retain_top', 10))],
        'band': [band_min, band_max],
        'window_nm': window_nm,
        'step_nm': step_nm,
        'prior_center': prior_center,
        'prior_weight': prior_weight,
        'gates': gates,
        'weights': weights,
        'dataset_label': dataset_label,
    }

    if out_root:
        try:
            metrics_dir = Path(out_root) / 'metrics'
            metrics_dir.mkdir(parents=True, exist_ok=True)
            _write_json(metrics_dir / 'roi_discovery.json', discovery)
        except Exception:
            pass

    return discovery


def _merge_discovered_bounds(min_wl: float,
                             max_wl: float,
                             expected_center: Optional[float],
                             discovered_roi: Optional[Dict[str, object]]) -> Tuple[float, float, Optional[float], Optional[Dict[str, object]]]:
    info: Optional[Dict[str, object]] = None
    if isinstance(discovered_roi, dict):
        selected = discovered_roi.get('selected', {})
        if isinstance(selected, dict) and bool(selected.get('quality_ok', False)):
            sel_min = selected.get('min_wavelength_nm')
            sel_max = selected.get('max_wavelength_nm')
            try:
                sel_min = float(sel_min)
                sel_max = float(sel_max)
            except (TypeError, ValueError):
                sel_min = sel_max = None
            if sel_min is not None and sel_max is not None and np.isfinite(sel_min) and np.isfinite(sel_max) and sel_min < sel_max:
                min_wl = max(min_wl, sel_min)
                max_wl = min(max_wl, sel_max)
                info = {
                    'min_wavelength_nm': float(min_wl),
                    'max_wavelength_nm': float(max_wl),
                    'center_nm': float(selected.get('center_nm', (sel_min + sel_max) * 0.5)),
                    'window_nm': float(selected.get('window_nm', sel_max - sel_min)),
                    'score': selected.get('score'),
                }
                if expected_center is None and info.get('center_nm') is not None:
                    expected_center = info.get('center_nm')
    return min_wl, max_wl, expected_center, info


def _select_multi_roi_candidates(discovered_roi: Dict[str, object], max_features: int = 4) -> List[Dict[str, object]]:
    if not isinstance(discovered_roi, dict):
        return []

    candidates = discovered_roi.get('candidates') or []
    if not isinstance(candidates, list):
        candidates = []

    quality: List[Dict[str, object]] = []
    seen_centers: Set[float] = set()

    def _maybe_add(cand: Dict[str, object]):
        if not isinstance(cand, dict):
            return
        if not bool(cand.get('quality_ok', False)):
            return
        center = cand.get('center_nm')
        deltas = cand.get('deltas_valid_nm') or cand.get('deltas_nm')
        concs = cand.get('concentrations_ppm')
        if center is None or not isinstance(deltas, (list, tuple)) or not isinstance(concs, (list, tuple)):
            return
        if len(deltas) != len(concs) or len(deltas) < 2:
            return
        center_val = float(center)
        if center_val in seen_centers:
            return
        seen_centers.add(center_val)
        quality.append(cand)

    _maybe_add(discovered_roi.get('selected'))
    for cand in candidates:
        _maybe_add(cand)

    def _score(cand: Dict[str, object]) -> float:
        slope = cand.get('slope_nm_per_ppm')
        r2 = cand.get('r2')
        slope_val = abs(float(slope)) if isinstance(slope, (int, float)) else 0.0
        r2_val = float(r2) if isinstance(r2, (int, float)) else 0.0
        return (slope_val, r2_val)

    quality.sort(key=_score, reverse=True)
    return quality[:max_features]


def _compute_multi_roi_fusion_calibration(discovered_roi: Optional[Dict[str, object]],
                                          calib: Dict[str, object],
                                          out_root: str,
                                          dataset_label: Optional[str],
                                          max_features: int = 4) -> Optional[Dict[str, object]]:
    if not discovered_roi or not isinstance(calib, dict):
        return None

    concentrations = calib.get('concentrations') or []
    if not isinstance(concentrations, list) or len(concentrations) < 3:
        return None
    y = np.array(concentrations, dtype=float)
    if not np.all(np.isfinite(y)):
        return None

    selected = _select_multi_roi_candidates(discovered_roi, max_features=max_features)
    if len(selected) < 2:
        return None

    feature_vectors: List[np.ndarray] = []
    feature_details: List[Dict[str, object]] = []
    for cand in selected:
        deltas = cand.get('deltas_valid_nm') or cand.get('deltas_nm')
        concs = cand.get('concentrations_ppm') or []
        vec = np.array(deltas, dtype=float)
        conc_vec = np.array(concs, dtype=float)
        if vec.shape[0] != y.shape[0]:
            continue
        if not np.all(np.isfinite(vec)):
            continue
        if np.std(vec) < 1e-6:
            continue
        feature_vectors.append(vec)
        feature_details.append({
            'center_nm': float(cand.get('center_nm', float('nan'))),
            'slope_nm_per_ppm': float(cand.get('slope_nm_per_ppm', float('nan'))),
            'r2': float(cand.get('r2', float('nan'))),
            'snr': float(cand.get('snr', float('nan'))),
        })

    if len(feature_vectors) < 2:
        return None

    X = np.column_stack(feature_vectors)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))
    residuals = y_pred - y
    lod_ppm = float('nan')
    if residuals.size > 1:
        sigma = np.std(residuals, ddof=1)
        if np.isfinite(sigma):
            lod_ppm = 3.0 * sigma

    r2_cv = float('nan')
    rmse_cv = float('nan')
    cv_preds = None
    n = y.shape[0]
    if n >= 4:
        cv_preds = np.empty(n, dtype=float)
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            try:
                model_cv = LinearRegression()
                model_cv.fit(X_train, y_train)
                cv_preds[i] = model_cv.predict(X[i:i+1])[0]
            except Exception:
                cv_preds = None
                break
        if cv_preds is not None and np.all(np.isfinite(cv_preds)):
            r2_cv = r2_score(y, cv_preds)
            rmse_cv = math.sqrt(mean_squared_error(y, cv_preds))

    metrics = {
        'dataset': dataset_label,
        'n_points': int(n),
        'n_features': int(X.shape[1]),
        'feature_centers_nm': [fd['center_nm'] for fd in feature_details],
        'coefficients': model.coef_.tolist(),
        'intercept_ppm': float(model.intercept_),
        'r2': float(r2),
        'rmse_ppm': float(rmse),
        'lod_ppm': float(lod_ppm),
        'r2_cv': float(r2_cv),
        'rmse_cv_ppm': float(rmse_cv),
        'actual_concentrations_ppm': y.tolist(),
        'predicted_concentrations_ppm': y_pred.tolist(),
        'cv_predictions_ppm': cv_preds.tolist() if isinstance(cv_preds, np.ndarray) else None,
        'residuals_ppm': residuals.tolist(),
        'features': feature_details,
    }

    metrics_dir = os.path.join(out_root, 'metrics')
    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    fusion_metrics_path = os.path.join(metrics_dir, 'multi_roi_fusion_metrics.json')
    with open(fusion_metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(y, y_pred, color='#1f77b4', label='Train fit')
    if isinstance(cv_preds, np.ndarray) and np.all(np.isfinite(cv_preds)):
        ax.scatter(y, cv_preds, color='#ff7f0e', marker='s', label='LOOCV')
    min_c = min(np.min(y), np.min(y_pred))
    max_c = max(np.max(y), np.max(y_pred))
    ax.plot([min_c, max_c], [min_c, max_c], 'k--', linewidth=1, label='y = x')
    ax.set_xlabel('Actual concentration (ppm)')
    ax.set_ylabel('Predicted concentration (ppm)')
    ax.set_title('Multi-ROI Fusion Calibration')
    ax.grid(True, alpha=0.3)
    text_lines = [
        f"R² = {r2:.3f}",
        f"RMSE = {rmse:.3f} ppm",
    ]
    if np.isfinite(r2_cv):
        text_lines.append(f"R²_LOOCV = {r2_cv:.3f}")
    if np.isfinite(lod_ppm):
        text_lines.append(f"LOD ≈ {lod_ppm:.2f} ppm")
    ax.text(0.05, 0.05, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend()
    fig.tight_layout()
    fusion_plot_path = os.path.join(plots_dir, 'calibration_multi_roi_fusion.png')
    fig.savefig(fusion_plot_path, dpi=200)
    plt.close(fig)

    metrics['metrics_path'] = fusion_metrics_path
    metrics['plot_path'] = fusion_plot_path
    return metrics


def find_roi_and_calibration(canonical: Dict[float, pd.DataFrame],
                             dataset_label: Optional[str] = None,
                             responsive_delta: Optional[Dict[str, object]] = None,
                             discovered_roi: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Compute wavelength shift vs concentration and fit linear calibration.

    Returns dict with keys: 'concentrations', 'peak_wavelengths', 'slope', 'intercept',
    'r2', 'rmse', 'slope_se', 'slope_ci_low', 'slope_ci_high', 'lod', 'loq', 'roi_center'.
    If ``responsive_delta`` is provided, responsive-frame Δλ statistics are used to
    override the canonical calibration metrics while preserving canonical details in
    the ``canonical_model`` field for auditability.
    """
    if not canonical:
        raise ValueError("No canonical spectra provided")

    # Sort by concentration
    items = sorted(canonical.items(), key=lambda kv: kv[0])
    concs = np.array([kv[0] for kv in items], dtype=float)
    calib_cfg = CONFIG.get('calibration', {}) or {}
    shift_method = str(calib_cfg.get('shift_method', 'centroid')).lower()
    xcorr_upsample = int(max(1, calib_cfg.get('xcorr_upsample', 1)))

    # Apply ROI wavelength limits to calibration
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    if min_wl_roi is None:
        min_wl_roi = 500.0
    if max_wl_roi is None:
        max_wl_roi = 900.0

    expected_center, expected_tol = _resolve_expected_center(dataset_label)
    expected_min = None
    expected_max = None
    if expected_center is not None and expected_tol is not None and expected_tol > 0:
        expected_min = expected_center - expected_tol
        expected_max = expected_center + expected_tol
        if expected_min is not None:
            min_wl_roi = max(min_wl_roi, expected_min)
        if expected_max is not None:
            max_wl_roi = min(max_wl_roi, expected_max)
        if min_wl_roi >= max_wl_roi:
            min_wl_roi = expected_min if expected_min is not None else min_wl_roi
            max_wl_roi = expected_max if expected_max is not None else max_wl_roi

    min_wl_roi, max_wl_roi, expected_center, discovery_applied = _merge_discovered_bounds(
        min_wl_roi,
        max_wl_roi,
        expected_center,
        discovered_roi,
    )

    # Trim canonical spectra to ROI bounds (with minor expansion fallback)
    roi_center_guess = expected_center if expected_center is not None else 0.5 * (min_wl_roi + max_wl_roi)
    trimmed_canonical: Dict[float, pd.DataFrame] = {}
    for conc, df in items:
        df_sorted = df.sort_values('wavelength').reset_index(drop=True)
        mask = (df_sorted['wavelength'] >= min_wl_roi) & (df_sorted['wavelength'] <= max_wl_roi)
        df_roi = df_sorted.loc[mask].copy()
        expand_step = 0.5
        expand_limit = 3.0
        current_expand = expand_step
        while df_roi.empty and current_expand <= expand_limit:
            lower = min_wl_roi - current_expand
            upper = max_wl_roi + current_expand
            mask = (df_sorted['wavelength'] >= lower) & (df_sorted['wavelength'] <= upper)
            df_roi = df_sorted.loc[mask].copy()
            current_expand += expand_step
        if df_roi.empty and not df_sorted.empty:
            wl = df_sorted['wavelength'].to_numpy()
            nearest_indices = np.argsort(np.abs(wl - roi_center_guess))[:min(25, wl.size)]
            df_roi = df_sorted.iloc[np.sort(nearest_indices)].copy()
        trimmed_canonical[conc] = df_roi if not df_roi.empty else df_sorted.copy()
        if df_roi.empty and df_sorted.empty:
            trimmed_canonical[conc] = df.copy()
    
    # Apply Savitzky-Golay smoothing to ROI spectra for noise reduction
    smoothing_cfg = CONFIG.get('preprocessing', {}).get('smoothing', {}) if isinstance(CONFIG, dict) else {}
    smoothing_enabled = bool(smoothing_cfg.get('enabled', False))  # Default to False
    if smoothing_enabled:
        try:
            from scipy.signal import savgol_filter
            window_length = int(smoothing_cfg.get('window', 11))
            poly_order = int(smoothing_cfg.get('poly_order', 2))
            
            # Ensure window_length is odd and valid
            if window_length % 2 == 0:
                window_length += 1
            
            print(f"[SMOOTHING] Applying Savitzky-Golay filter (window={window_length}, poly={poly_order})")
            import sys; sys.stdout.flush()
            
            smoothed_canonical = {}
            for conc, df in trimmed_canonical.items():
                if df is None or df.empty:
                    smoothed_canonical[conc] = df
                    continue
                
                df_smooth = df.copy()
                signal_cols = [c for c in df.columns if c != 'wavelength']
                
                for col in signal_cols:
                    if col in df_smooth.columns and len(df_smooth) >= window_length:
                        try:
                            smoothed_signal = savgol_filter(
                                df_smooth[col].values,
                                window_length=window_length,
                                polyorder=poly_order,
                                mode='interp'
                            )
                            df_smooth[col] = smoothed_signal
                        except Exception as e:
                            print(f"[WARNING] Smoothing failed for {col}: {e}")
                
                smoothed_canonical[conc] = df_smooth
            
            trimmed_canonical = smoothed_canonical
            print(f"[SMOOTHING] Successfully smoothed {len(trimmed_canonical)} concentration spectra")
            import sys; sys.stdout.flush()
            
        except ImportError:
            print("[WARNING] scipy not available, skipping Savitzky-Golay smoothing")
            import sys; sys.stdout.flush()
        except Exception as e:
            print(f"[WARNING] Smoothing failed: {e}")
            import sys; sys.stdout.flush()

    items = sorted(trimmed_canonical.items(), key=lambda kv: kv[0])

    # Merge ROI limits into calibration config for feature tracking
    calib_cfg_with_limits = calib_cfg.copy() if isinstance(calib_cfg, dict) else {}
    calib_cfg_with_limits['min_wavelength'] = min_wl_roi
    calib_cfg_with_limits['max_wavelength'] = max_wl_roi
    print(f"\n[CALIBRATION] Restricting feature tracking to {min_wl_roi:.1f}–{max_wl_roi:.1f} nm")
    import sys; sys.stdout.flush()

    # Track multiple features
    feature_results = []

    # Base reference
    base_index = 0
    base_df = items[base_index][1]
    shift_cfg = CONFIG.get('roi', {}).get('shift', {}) if isinstance(CONFIG.get('roi', {}), dict) else {}
    seed_span_nm = float(shift_cfg.get('seed_derivative_span_nm', 2.0) or 2.0)
    seed_window = int(shift_cfg.get('seed_derivative_smooth_window', 7) or 7)
    derivative_seed = _refine_centroid_with_derivative(
        base_df,
        {
            **calib_cfg_with_limits,
            'derivative_span_nm': seed_span_nm,
            'derivative_smooth_window': seed_window,
        },
        expected_center,
        span_nm=seed_span_nm,
        smooth_window=seed_window,
    )
    base_center_centroid = _find_peak_wavelength(base_df, centroid_cfg=calib_cfg_with_limits)
    if np.isfinite(derivative_seed) and min_wl_roi <= derivative_seed <= max_wl_roi:
        base_center_centroid = derivative_seed
        calib_cfg_with_limits['centroid_hint'] = derivative_seed
        print(f"  [SEED] Derivative-based centroid refinement to {derivative_seed:.2f} nm")
        import sys; sys.stdout.flush()
    else:
        calib_cfg_with_limits['centroid_hint'] = base_center_centroid
    base_wl, base_signal = _prepare_calibration_signal(base_df, calib_cfg_with_limits)

    # 1) Centroid track (always computed)
    centroid_list: List[float] = []
    for _, df in items:
        centroid_list.append(_find_peak_wavelength(df, centroid_cfg=calib_cfg_with_limits))
    feature_results.append({'type': 'centroid', 'wavelengths': centroid_list, 'center': base_center_centroid})

    # 2) Cross-correlation track (sub-sample shifts)
    xcorr_list: List[float] = []
    if shift_method == 'xcorr':
        for _, df in items:
            try:
                target_wl, target_signal = _prepare_calibration_signal(df, calib_cfg_with_limits)
                target_resampled = np.interp(base_wl, target_wl, target_signal)
                shift_nm = _estimate_shift_crosscorr(base_wl, base_signal, target_resampled, upsample=xcorr_upsample)
                xcorr_list.append(base_center_centroid + shift_nm)
            except Exception:
                xcorr_list.append(np.nan)
        feature_results.append({'type': 'xcorr', 'wavelengths': xcorr_list, 'center': base_center_centroid})

    # 3) Gaussian fit track
    gaussian_list: List[float] = []
    hw = int(calib_cfg.get('centroid_half_width', 5))
    for _, df in items:
        x, y = _prepare_calibration_signal(df, calib_cfg_with_limits)
        # hint index from centroid for stability
        idx_hint = int(np.argmin(np.abs(x - _find_peak_wavelength(df, centroid_cfg=calib_cfg_with_limits))))
        gaussian_list.append(_gaussian_peak_center(x, y, idx_hint=idx_hint, half_width=hw))
    feature_results.append({'type': 'gaussian', 'wavelengths': gaussian_list, 'center': float(np.median(gaussian_list))})
    
    # Add best wavelength from intensity-based analysis
    try:
        ref_df = items[-1][1]  # Highest concentration
        intensity_col = 'intensity' if 'intensity' in ref_df.columns else None
        has_intensity_everywhere = intensity_col is not None and all(intensity_col in df.columns for _, df in items)
        if has_intensity_everywhere:
            # Quick intensity-based scan to find best wavelength
            best_wl_intensity = None
            best_r2_intensity = -1

            all_wl = ref_df['wavelength'].values
            wl_mask = (all_wl >= min_wl_roi) & (all_wl <= max_wl_roi)
            candidate_wl = all_wl[wl_mask][::10]  # Sample every 10th wavelength

            for test_wl in candidate_wl:
                intensities_at_wl = []
                for _, df in items:
                    df_wl = df['wavelength'].values
                    df_int = df[intensity_col].values
                    closest_idx = np.argmin(np.abs(df_wl - test_wl))
                    intensities_at_wl.append(df_int[closest_idx])

                if len(intensities_at_wl) == len(concs):
                    slope_i, intercept_i, r_val_i, _, _ = linregress(concs, intensities_at_wl)
                    r2_i = r_val_i ** 2
                    if r2_i > best_r2_intensity:
                        best_r2_intensity = r2_i
                        best_wl_intensity = test_wl

            if best_wl_intensity is not None and best_r2_intensity > 0.1:
                intensity_values = []
                for _, df in items:
                    df_wl = df['wavelength'].values
                    df_int = df[intensity_col].values
                    closest_idx = np.argmin(np.abs(df_wl - best_wl_intensity))
                    intensity_values.append(df_int[closest_idx])

                feature_results.append({
                    'type': 'intensity_best',
                    'wavelengths': intensity_values,
                    'center': best_wl_intensity,
                    'is_intensity_based': True,
                    'r2_precalculated': best_r2_intensity
                })
                print(f"  [SCAN] Found best intensity wavelength: {best_wl_intensity:.2f} nm (R²={best_r2_intensity:.4f})")
                import sys; sys.stdout.flush()
    except Exception as e:
        print(f"  [SCAN] Intensity scan failed: {e}")
    
    # Additional feature tracking (peaks/valleys)
    try:
        from scipy.signal import find_peaks

        # Use highest concentration spectrum for feature detection - APPLY LIMITS
        ref_df = items[-1][1]  # Highest concentration
        ref_wl_full = ref_df['wavelength'].values
        ref_signal_col = 'intensity' if 'intensity' in ref_df.columns else _signal_column(ref_df)
        if ref_signal_col not in ref_df.columns:
            other_cols = [c for c in ref_df.columns if c != 'wavelength']
            if not other_cols:
                raise KeyError("No usable signal column for feature detection")
            ref_signal_col = other_cols[0]
        ref_intensity_full = ref_df[ref_signal_col].values
        
        # Apply wavelength limits to reference spectrum
        mask_roi = (ref_wl_full >= min_wl_roi) & (ref_wl_full <= max_wl_roi)
        ref_wl = ref_wl_full[mask_roi]
        ref_intensity = ref_intensity_full[mask_roi]
        
        # Find peaks
        peaks, _ = find_peaks(ref_intensity, prominence=0.005, distance=50)
        # Find valleys (invert signal)
        valleys, _ = find_peaks(-ref_intensity, prominence=0.005, distance=50)
        
        # Track up to 4 additional features
        feature_count = 0
        max_features = 4
        
        for p in peaks:
            if feature_count >= max_features:
                break
            feature_wl = ref_wl[p]
            
            # Skip if outside ROI (safety check)
            if feature_wl < min_wl_roi or feature_wl > max_wl_roi:
                continue
            
            tracked_wavelengths = []
            
            for _, df in items:
                df_wl = df['wavelength'].values
                sig_col = ref_signal_col if ref_signal_col in df.columns else _signal_column(df)
                if sig_col not in df.columns:
                    continue
                df_intensity = df[sig_col].values
                # Find closest wavelength point
                closest_idx = np.argmin(np.abs(df_wl - feature_wl))
                tracked_wavelengths.append(df_wl[closest_idx])
            
            feature_results.append({
                'type': 'peak',
                'index': p,
                'reference_wavelength': feature_wl,
                'wavelengths': tracked_wavelengths,
                'center': feature_wl
            })
            feature_count += 1
        
        for v in valleys:
            if feature_count >= max_features:
                break
            feature_wl = ref_wl[v]
            
            # Skip if outside ROI (safety check)
            if feature_wl < min_wl_roi or feature_wl > max_wl_roi:
                continue
            
            tracked_wavelengths = []
            
            for _, df in items:
                df_wl = df['wavelength'].values
                sig_col = _signal_column(df)
                df_intensity = df[sig_col].values
                # Find closest wavelength point
                closest_idx = np.argmin(np.abs(df_wl - feature_wl))
                tracked_wavelengths.append(df_wl[closest_idx])
            
            feature_results.append({
                'type': 'valley',
                'index': v,
                'reference_wavelength': feature_wl,
                'wavelengths': tracked_wavelengths,
                'center': feature_wl
            })
            feature_count += 1
            
    except ImportError:
        pass  # scipy not available, skip additional features
    
    # Evaluate linearity for each feature and select best
    best_r2 = -1
    best_result = None
    best_peak_list = None
    shift_preferences = (CONFIG.get('roi', {}) or {}).get('shift', {}) if isinstance(CONFIG.get('roi', {}), dict) else {}
    min_r2_required = float(shift_preferences.get('min_r2_w', 0.8))
    min_slope_required = float(shift_preferences.get('min_slope_nm_per_ppm', 0.05))
    if min_slope_required <= 0:
        min_slope_required = 0.05
    if min_r2_required <= 0:
        min_r2_required = 0.25

    # Stricter quality gates to favor physically meaningful Δλ tracks
    quality_threshold = max(0.25, min_r2_required)
    peak_list = None
    rejection_log = []
    
    for result in feature_results:
        wavelengths = result['wavelengths']
        feature_type = result.get('type', 'unknown')
        center_wl = result.get('center', 0.0)
        is_intensity_based = result.get('is_intensity_based', False)
        
        if len(concs) > 1 and len(wavelengths) == len(concs):
            valid_indices = [i for i in range(len(concs)) if not np.isnan(wavelengths[i])]
            if len(valid_indices) > 1:
                valid_concs = concs[valid_indices]
                valid_wavelengths = np.array(wavelengths)[valid_indices]
                
                if center_wl < min_wl_roi or center_wl > max_wl_roi:
                    reason = f"{feature_type} at {center_wl:.2f} nm (outside ROI)"
                    rejection_log.append(reason)
                    print(f"  [FILTER] Rejecting {reason}")
                    import sys; sys.stdout.flush()
                    continue

                if expected_min is not None and expected_max is not None:
                    if center_wl < expected_min or center_wl > expected_max:
                        reason = f"{feature_type} at {center_wl:.2f} nm (outside expected {expected_center:.2f}±{expected_tol:.2f} nm)"
                        rejection_log.append(reason)
                        print(f"  [FILTER] Rejecting {reason}")
                        import sys; sys.stdout.flush()
                        continue
                
                # For intensity-based features, use pre-calculated R² or recalculate
                if is_intensity_based and 'r2_precalculated' in result:
                    r2 = result['r2_precalculated']
                    slope, intercept, _, p_value, std_err = linregress(valid_concs, valid_wavelengths)
                else:
                    slope, intercept, r_value, p_value, std_err = linregress(valid_concs, valid_wavelengths)
                    r2 = r_value ** 2
                
                wl_std = float(np.std(valid_wavelengths))
                wl_mean = float(np.mean(valid_wavelengths))
                cv = wl_std / wl_mean if wl_mean > 0 else float('inf')
                
                result['slope'] = slope
                result['intercept'] = intercept
                result['r2'] = r2
                result['p_value'] = p_value
                result['std_err'] = std_err
                result['cv'] = cv
                result['wl_std'] = wl_std
                
                if is_intensity_based and bool(CONFIG.get('roi', {}).get('shift', {}).get('prefer_expected_center', True)):
                    # deprioritize intensity-only tracks when centroid/gaussian options exist
                    reason = f"{feature_type} at {center_wl:.2f} nm (intensity-based not allowed)"
                    rejection_log.append(reason)
                    print(f"  [FILTER] Rejecting {reason}")
                    import sys; sys.stdout.flush()
                    continue

                if abs(slope) < min_slope_required:
                    reason = f"{feature_type} at {center_wl:.2f} nm (slope {slope:+.6f} below {min_slope_required:.6f} nm/ppm)"
                    rejection_log.append(reason)
                    print(f"  [FILTER] Rejecting {reason}")
                    import sys; sys.stdout.flush()
                    continue

                if r2 < quality_threshold or cv > 0.02:
                    reason = f"{feature_type} at {center_wl:.2f} nm (R2={r2:.4f}, CV={cv:.4f})"
                    rejection_log.append(reason)
                    print(f"  [FILTER] Rejecting {reason}")
                    import sys; sys.stdout.flush()
                    continue

                if (expected_center is not None and
                        abs(center_wl - expected_center) > max(expected_tol or 0.0, 0.5)):
                    reason = f"{feature_type} at {center_wl:.2f} nm (drifts > tolerance from expected {expected_center:.2f} nm)"
                    rejection_log.append(reason)
                    print(f"  [FILTER] Rejecting {reason}")
                    import sys; sys.stdout.flush()
                    continue
                
                # Format output based on feature type
                if is_intensity_based:
                    print(f"  [FEATURE] {feature_type:12s} at {center_wl:7.2f} nm: R2={r2:.4f} (intensity-based)")
                else:
                    print(f"  [FEATURE] {feature_type:12s} at {center_wl:7.2f} nm: R2={r2:.4f}, slope={slope:+.6f}, CV={cv:.4f}")
                import sys; sys.stdout.flush()
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_result = result
                    best_peak_list = valid_wavelengths.tolist()
    
    # Use the best feature for final calibration
    roi_min_wl = min_wl_roi
    roi_max_wl = max_wl_roi

    if best_result is not None:
        peak_list = best_peak_list
        base_center = np.clip(best_result['center'], roi_min_wl, roi_max_wl)
        is_intensity = best_result.get('is_intensity_based', False)
        method_label = "intensity" if is_intensity else "wavelength-shift"
        print(f"\n✓ Selected feature: {best_result['type']} at {base_center:.2f} nm (R² = {best_r2:.4f}, {method_label})")
        import sys; sys.stdout.flush()
        
        # Optional: ensemble tracking (weighted average of top features)
        ensemble_cfg = calib_cfg.get('ensemble', {}) if isinstance(calib_cfg, dict) else {}
        if ensemble_cfg.get('enabled', False) and len(feature_results) > 1:
            # Get top N features by R²
            top_n = int(ensemble_cfg.get('top_n', 3))
            valid_features = [fr for fr in feature_results if fr.get('r2') is not None and np.isfinite(fr.get('r2')) and fr.get('r2') > 0]
            if len(valid_features) >= 2:
                valid_features.sort(key=lambda x: x['r2'], reverse=True)
                top_features = valid_features[:top_n]
                
                # Weighted average of wavelengths
                ensemble_wavelengths = []
                for i in range(len(concs)):
                    weighted_sum = 0.0
                    weight_total = 0.0
                    for feat in top_features:
                        if i < len(feat['wavelengths']) and np.isfinite(feat['wavelengths'][i]):
                            weight = feat['r2']
                            weighted_sum += feat['wavelengths'][i] * weight
                            weight_total += weight
                    if weight_total > 0:
                        ensemble_wavelengths.append(weighted_sum / weight_total)
                    else:
                        ensemble_wavelengths.append(np.nan)
                
                # Check if ensemble improves R²
                valid_idx = [i for i in range(len(concs)) if np.isfinite(ensemble_wavelengths[i])]
                if len(valid_idx) > 1:
                    ens_concs = concs[valid_idx]
                    ens_wl = np.array(ensemble_wavelengths)[valid_idx]
                    ens_reg = linregress(ens_concs, ens_wl)
                    ens_r2 = ens_reg.rvalue ** 2
                    if ens_r2 > best_r2:
                        print(f"Ensemble tracking improved R²: {ens_r2:.4f} (combining {len(top_features)} features)")
                        peak_list = ens_wl.tolist()
                        best_r2 = ens_r2
    else:
        import sys
        print("\n⚠ No suitable feature passed ROI gates; falling back to centroid track")
        if rejection_log:
            print("   Rejection summary:")
            for reason in rejection_log[:10]:
                print(f"    • {reason}")
            if len(rejection_log) > 10:
                remaining = len(rejection_log) - 10
                print(f"    • … {remaining} more")
        sys.stdout.flush()
        peak_list = [np.clip(p, roi_min_wl, roi_max_wl) for p in centroid_list]
        base_center = np.clip(base_center_centroid, roi_min_wl, roi_max_wl)
        best_r2 = float('nan')

    if peak_list is None:
        # As an absolute fallback, copy centroid list to avoid runtime failure
        peak_list = [np.clip(p, roi_min_wl, roi_max_wl) for p in centroid_list]
    peaks = np.array(peak_list, dtype=float)
    peaks = np.clip(peaks, roi_min_wl, roi_max_wl)

    x_mode = str(calib_cfg.get('x_transform', 'linear')).lower()
    concs_x, transform_meta = _transform_concentrations(concs, x_mode)

    # Fit linear model: peak_wavelength vs concentration transform
    reg = linregress(concs_x, peaks)
    slope_lin, intercept_lin, rvalue_lin, pvalue_lin, stderr_lin = reg
    preds_lin = intercept_lin + slope_lin * concs_x
    resid_lin = peaks - preds_lin
    rmse_lin = float(np.sqrt(np.mean(resid_lin ** 2)))
    n = len(concs_x)
    dfree = max(1, n - 2)
    tcrit = t.ppf(0.975, dfree)
    ci_low_lin = slope_lin - tcrit * stderr_lin
    ci_high_lin = slope_lin + tcrit * stderr_lin
    
    linear_model = {
        'model': 'linear',
        'x_transform': x_mode,
        'slope': float(slope_lin),
        'intercept': float(intercept_lin),
        'r2': float(rvalue_lin ** 2),
        'rmse': float(rmse_lin),
        'slope_se': float(stderr_lin),
        'slope_ci_low': float(ci_low_lin),
        'slope_ci_high': float(ci_high_lin),
        'predictions': preds_lin.tolist(),
    }
    
    # Compute residual sum of squares for AIC calculation (needed for Langmuir comparison)
    ss_res_lin = float(np.sum((peaks - preds_lin) ** 2))
    ss_tot_lin = float(np.sum((peaks - np.mean(peaks)) ** 2))

    best_model = linear_model
    best_predictions = preds_lin

    # Cross-validation (LOOCV) for model robustness
    cv_cfg = calib_cfg.get('cv', {}) if isinstance(calib_cfg, dict) else {}
    cv_enabled = bool(cv_cfg.get('enabled', True))

    def _loocv_scores(model_type: str = 'linear', degree: int = 2, langmuir_params: tuple = None) -> Tuple[float, float]:
        """Return (r2_cv, rmse_cv) using leave-one-out CV on current peaks vs concs_x."""
        nloc = len(concs_x)
        if nloc < 3:
            return float('nan'), float('nan')
        y_pred = np.empty(nloc, dtype=float)
        for i in range(nloc):
            x_tr = np.delete(concs_x, i)
            y_tr = np.delete(peaks, i)
            x_te = concs_x[i]
            if model_type == 'linear':
                k, b = np.polyfit(x_tr, y_tr, 1)
                y_pred[i] = b + k * x_te
            elif model_type == 'robust_huber':
                try:
                    from sklearn.linear_model import HuberRegressor
                    hub = HuberRegressor(epsilon=float(robust_cfg.get('epsilon', 1.35)),
                                         alpha=float(robust_cfg.get('alpha', 1e-4)))
                    hub.fit(x_tr.reshape(-1, 1), y_tr)
                    y_pred[i] = float(hub.predict(np.array([[x_te]]))[0])
                except Exception:
                    return float('nan'), float('nan')
            elif model_type.startswith('poly'):
                deg = max(1, degree)
                try:
                    coeffs_cv = np.polyfit(x_tr, y_tr, deg)
                    y_pred[i] = float(np.polyval(coeffs_cv, x_te))
                except Exception:
                    return float('nan'), float('nan')
            elif model_type == 'langmuir':
                try:
                    from scipy.optimize import curve_fit
                    def langmuir_cv(C, a, b):
                        return (a * C) / (1.0 + b * C)
                    # Use provided params as initial guess
                    p0 = langmuir_params if langmuir_params else [0.05, 0.001]
                    popt_cv, _ = curve_fit(langmuir_cv, x_tr, y_tr, p0=p0, maxfev=5000, bounds=([0, 0], [np.inf, np.inf]))
                    y_pred[i] = float(langmuir_cv(x_te, *popt_cv))
                except Exception:
                    return float('nan'), float('nan')
            else:
                return float('nan'), float('nan')
        try:
            ss_res = float(np.sum((peaks - y_pred) ** 2))
            ss_tot = float(np.sum((peaks - np.mean(peaks)) ** 2))
            r2cv = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmsecv = float(np.sqrt(np.mean((peaks - y_pred) ** 2)))
            return float(r2cv), float(rmsecv)
        except Exception:
            return float('nan'), float('nan')

    if cv_enabled:
        r2cv_lin, rmsecv_lin = _loocv_scores('linear')
        linear_model['r2_cv'] = r2cv_lin
        linear_model['rmse_cv'] = rmsecv_lin

    robust_info: Optional[Dict[str, object]] = None
    robust_cfg = calib_cfg.get('robust', {}) if isinstance(calib_cfg, dict) else {}
    if robust_cfg.get('enabled', False) and np.unique(concs_x).size >= 2:
        try:
            from sklearn.linear_model import HuberRegressor
            from sklearn.metrics import r2_score

            epsilon = float(robust_cfg.get('epsilon', 1.35))
            alpha = float(robust_cfg.get('alpha', 0.0001))
            huber = HuberRegressor(epsilon=epsilon, alpha=alpha)
            huber.fit(concs_x.reshape(-1, 1), peaks)
            preds_robust = huber.predict(concs_x.reshape(-1, 1))
            slope_robust = float(huber.coef_[0])
            intercept_robust = float(huber.intercept_)
            r2_robust = float(r2_score(peaks, preds_robust))
            rmse_robust = float(np.sqrt(np.mean((peaks - preds_robust) ** 2)))

            robust_info = {
                'model': 'robust_huber',
                'slope': slope_robust,
                'intercept': intercept_robust,
                'r2': r2_robust,
                'rmse': rmse_robust,
                'epsilon': epsilon,
                'alpha': alpha,
                'predictions': preds_robust.tolist(),
            }

            prefer_robust = bool(robust_cfg.get('prefer', False))
            if prefer_robust and np.isfinite(r2_robust):
                current_r2 = best_model.get('r2', float('-inf'))
                if not np.isfinite(current_r2) or r2_robust >= current_r2:
                    best_model = {
                        'model': 'robust_huber',
                        'x_transform': x_mode,
                        'slope': slope_robust,
                        'intercept': intercept_robust,
                        'r2': r2_robust,
                        'rmse': rmse_robust,
                        'slope_se': float('nan'),
                        'slope_ci_low': float('nan'),
                        'slope_ci_high': float('nan'),
                        'predictions': preds_robust.tolist(),
                    }
                    best_predictions = preds_robust
            # CV for robust model
            if cv_enabled and robust_info and robust_info.get('model') == 'robust_huber':
                r2cv_rb, rmsecv_rb = _loocv_scores('robust_huber')
                robust_info['r2_cv'] = r2cv_rb
                robust_info['rmse_cv'] = rmsecv_rb
        except Exception as exc:  # noqa: BLE001
            robust_info = {'error': str(exc)}

    poly_info: Optional[Dict[str, object]] = None
    poly_enabled = str(calib_cfg.get('model', 'linear')).lower() in {'auto', 'polynomial'}
    poly_degree = int(calib_cfg.get('polynomial_degree', 2))
    if poly_enabled and len(concs_x) > poly_degree:
        try:
            coeffs = np.polyfit(concs_x, peaks, poly_degree)
            preds_poly = np.polyval(coeffs, concs_x)
            ss_res = float(np.sum((peaks - preds_poly) ** 2))
            ss_tot = float(np.sum((peaks - np.mean(peaks)) ** 2))
            r2_poly = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse_poly = float(np.sqrt(np.mean((peaks - preds_poly) ** 2)))
            derivative = np.polyder(coeffs)
            ref_conc = float(np.mean(concs_x))
            slope_poly = float(np.polyval(derivative, ref_conc))
            intercept_poly = float(np.polyval(coeffs, ref_conc) - slope_poly * ref_conc)
            poly_info = {
                'model': f'poly_deg_{poly_degree}',
                'coefficients': coeffs.tolist(),
                'ref_concentration': ref_conc,
                'slope': slope_poly,
                'intercept': intercept_poly,
                'r2': float(r2_poly),
                'rmse': rmse_poly,
                'predictions': preds_poly.tolist(),
            }
            # CV for polynomial model (compute before selection to check for overfitting)
            if cv_enabled and poly_info:
                r2cv_pl, rmsecv_pl = _loocv_scores('poly', degree=poly_degree)
                poly_info['r2_cv'] = r2cv_pl
                poly_info['rmse_cv'] = rmsecv_pl
            
            if np.isfinite(r2_poly):
                model_mode = str(calib_cfg.get('model', 'linear')).lower()
                # Only select polynomial if CV R² is positive (prevents overfitting)
                r2cv_poly = poly_info.get('r2_cv', float('nan')) if poly_info else float('nan')
                cv_acceptable = not cv_enabled or (np.isfinite(r2cv_poly) and r2cv_poly > 0.5)
                
                if (model_mode == 'polynomial') or (model_mode == 'auto' and r2_poly > best_model['r2'] and cv_acceptable):
                    best_model = {
                        'model': poly_info['model'],
                        'x_transform': x_mode,
                        'slope': poly_info['slope'],
                        'intercept': poly_info['intercept'],
                        'r2': poly_info['r2'],
                        'rmse': poly_info['rmse'],
                        'slope_se': float('nan'),
                        'slope_ci_low': float('nan'),
                        'slope_ci_high': float('nan'),
                        'predictions': preds_poly.tolist(),
                        'ref_concentration': ref_conc,
                    }
                    best_predictions = preds_poly
                    print(f"[INFO] Polynomial model selected (R²={r2_poly:.4f}, R²_CV={r2cv_poly:.4f})")
                elif model_mode == 'auto' and not cv_acceptable:
                    print(f"[WARNING] Polynomial model rejected due to poor CV (R²_CV={r2cv_poly:.4f})")
        except (np.linalg.LinAlgError, ValueError):
            poly_info = None

    # Langmuir saturation model: Δλ = (a*C) / (1 + b*C)
    langmuir_info: Optional[Dict[str, object]] = None
    langmuir_enabled = str(calib_cfg.get('model', 'linear')).lower() in {'auto', 'langmuir'}
    if langmuir_enabled and len(concs_x) >= 3:
        try:
            from scipy.optimize import curve_fit
            
            def langmuir_model(C, a, b):
                """Langmuir saturation model: Δλ = (a*C) / (1 + b*C)"""
                return (a * C) / (1.0 + b * C)
            
            # Initial guess: a ~ linear slope, b ~ 1/max_concentration
            a_init = slope_lin if np.isfinite(slope_lin) else 0.05
            b_init = 1.0 / np.max(concs_x) if np.max(concs_x) > 0 else 0.001
            
            # Fit Langmuir model
            popt, pcov = curve_fit(
                langmuir_model, 
                concs_x, 
                peaks,
                p0=[a_init, b_init],
                maxfev=5000,
                bounds=([0, 0], [np.inf, np.inf])  # Both parameters must be positive
            )
            
            a_lang, b_lang = popt
            preds_lang = langmuir_model(concs_x, a_lang, b_lang)
            
            # Compute R² and RMSE
            ss_res_lang = float(np.sum((peaks - preds_lang) ** 2))
            ss_tot_lang = float(np.sum((peaks - np.mean(peaks)) ** 2))
            r2_lang = 1.0 - ss_res_lang / ss_tot_lang if ss_tot_lang > 0 else float('nan')
            rmse_lang = float(np.sqrt(np.mean((peaks - preds_lang) ** 2)))
            
            # Compute effective slope at reference concentration (derivative)
            ref_conc_lang = float(np.mean(concs_x))
            # dΔλ/dC = a / (1 + b*C)²
            slope_lang = float(a_lang / ((1.0 + b_lang * ref_conc_lang) ** 2))
            intercept_lang = float(langmuir_model(ref_conc_lang, a_lang, b_lang) - slope_lang * ref_conc_lang)
            
            # Compute parameter uncertainties from covariance matrix
            perr = np.sqrt(np.diag(pcov))
            a_se = float(perr[0]) if len(perr) > 0 else float('nan')
            b_se = float(perr[1]) if len(perr) > 1 else float('nan')
            
            langmuir_info = {
                'model': 'langmuir',
                'parameter_a': float(a_lang),
                'parameter_b': float(b_lang),
                'parameter_a_se': a_se,
                'parameter_b_se': b_se,
                'ref_concentration': ref_conc_lang,
                'slope': slope_lang,
                'intercept': intercept_lang,
                'r2': float(r2_lang),
                'rmse': rmse_lang,
                'predictions': preds_lang.tolist(),
            }
            
            # Model selection: prefer Langmuir if R² is better (with AIC penalty for extra parameter)
            if np.isfinite(r2_lang):
                model_mode = str(calib_cfg.get('model', 'linear')).lower()
                
                # Compute AIC for model comparison (lower is better)
                n = len(concs_x)
                k_linear = 2  # slope + intercept
                k_langmuir = 2  # a + b
                
                aic_linear = n * np.log(ss_res_lin / n) + 2 * k_linear if ss_res_lin > 0 else np.inf
                aic_langmuir = n * np.log(ss_res_lang / n) + 2 * k_langmuir if ss_res_lang > 0 else np.inf
                
                langmuir_info['aic'] = float(aic_langmuir)
                linear_model['aic'] = float(aic_linear)
                
                # Select Langmuir if explicitly requested or if AIC is better in auto mode
                if model_mode == 'langmuir' or (model_mode == 'auto' and aic_langmuir < aic_linear - 2):
                    best_model = {
                        'model': 'langmuir',
                        'x_transform': x_mode,
                        'slope': slope_lang,
                        'intercept': intercept_lang,
                        'r2': r2_lang,
                        'rmse': rmse_lang,
                        'slope_se': float('nan'),  # Not directly applicable
                        'slope_ci_low': float('nan'),
                        'slope_ci_high': float('nan'),
                        'predictions': preds_lang.tolist(),
                        'ref_concentration': ref_conc_lang,
                        'parameter_a': float(a_lang),
                        'parameter_b': float(b_lang),
                        'aic': float(aic_langmuir),
                    }
                    best_predictions = preds_lang
            
            # CV for Langmuir model
            if cv_enabled and langmuir_info:
                r2cv_lang, rmsecv_lang = _loocv_scores('langmuir', langmuir_params=(a_lang, b_lang))
                langmuir_info['r2_cv'] = r2cv_lang
                langmuir_info['rmse_cv'] = rmsecv_lang
                
        except Exception as exc:  # noqa: BLE001
            langmuir_info = {'error': str(exc)}
            print(f"[WARNING] Langmuir model fitting failed: {exc}")

    # Final model selection metric (prefer CV if enabled)
    try:
        sel = str(result_payload.get('selected_model', 'linear')).lower()
        r2_cv_sel = float('nan')
        rmse_cv_sel = float('nan')
        if sel.startswith('poly') and isinstance(poly_info, dict):
            r2_cv_sel = float(poly_info.get('r2_cv', float('nan')))
            rmse_cv_sel = float(poly_info.get('rmse_cv', float('nan')))
        elif sel == 'langmuir' and isinstance(langmuir_info, dict):
            r2_cv_sel = float(langmuir_info.get('r2_cv', float('nan')))
            rmse_cv_sel = float(langmuir_info.get('rmse_cv', float('nan')))
        elif sel == 'robust_huber' and isinstance(robust_info, dict):
            r2_cv_sel = float(robust_info.get('r2_cv', float('nan')))
            rmse_cv_sel = float(robust_info.get('rmse_cv', float('nan')))
        elif sel == 'linear' and isinstance(linear_model, dict):
            r2_cv_sel = float(linear_model.get('r2_cv', float('nan')))
            rmse_cv_sel = float(linear_model.get('rmse_cv', float('nan')))
        elif sel == 'plsr_cv' and isinstance(result_payload.get('plsr_model', None), dict):
            pm = result_payload['plsr_model']
            r2_cv_sel = float(pm.get('r2_cv', float('nan')))
            rmse_cv_sel = float(pm.get('rmse_cv', float('nan')))
        result_payload['uncertainty'] = {
            'r2_cv': r2_cv_sel,
            'rmse_cv': rmse_cv_sel,
            'bootstrap': bootstrap_info,
        }
    except Exception:
        pass

    bootstrap_info: Optional[Dict[str, object]] = None
    bootstrap_cfg = calib_cfg.get('bootstrap', {}) if isinstance(calib_cfg, dict) else {}
    if bootstrap_cfg.get('enabled', False) and len(concs_x) >= 3:
        rng_seed = bootstrap_cfg.get('random_seed', None)
        rng = np.random.default_rng(rng_seed)
        iterations = int(max(1, bootstrap_cfg.get('iterations', 500)))
        sample_fraction = float(bootstrap_cfg.get('sample_fraction', 0.8))
        sample_fraction = float(np.clip(sample_fraction, 0.1, 1.0))
        min_unique = int(max(2, bootstrap_cfg.get('min_unique', 2)))

        slope_samples: List[float] = []
        intercept_samples: List[float] = []

        size = len(concs_x)
        for _ in range(iterations):
            draw_size = max(min_unique, int(round(sample_fraction * size)))
            indices = rng.integers(0, size, draw_size)
            x_bs = concs_x[indices]
            if np.unique(x_bs).size < min_unique:
                continue
            y_bs = peaks[indices]
            res_bs = linregress(x_bs, y_bs)
            slope_samples.append(float(res_bs.slope))
            intercept_samples.append(float(res_bs.intercept))

        if slope_samples:
            slopes_arr = np.array(slope_samples)
            intercepts_arr = np.array(intercept_samples)
            bootstrap_info = {
                'iterations': int(len(slopes_arr)),
                'slope_mean': float(np.mean(slopes_arr)),
                'slope_std': float(np.std(slopes_arr, ddof=1)) if len(slopes_arr) > 1 else 0.0,
                'slope_ci_low': float(np.percentile(slopes_arr, 2.5)),
                'slope_ci_high': float(np.percentile(slopes_arr, 97.5)),
                'intercept_mean': float(np.mean(intercepts_arr)),
                'intercept_std': float(np.std(intercepts_arr, ddof=1)) if len(intercepts_arr) > 1 else 0.0,
                'intercept_ci_low': float(np.percentile(intercepts_arr, 2.5)),
                'intercept_ci_high': float(np.percentile(intercepts_arr, 97.5)),
            }
        else:
            bootstrap_info = {'iterations': 0}

    # LOD and LOQ from residual-based noise (approximation)
    eps = 1e-12
    slope_for_lod = best_model['slope'] if abs(best_model['slope']) > eps else slope_lin
    best_predictions_arr = np.asarray(best_predictions, dtype=float) if best_predictions is not None else np.array([], dtype=float)
    predictions_list: Optional[List[float]] = None
    residuals_arr: Optional[np.ndarray] = None
    residual_summary: Optional[Dict[str, float]] = None
    if best_predictions_arr.size == peaks.size:
        predictions_list = best_predictions_arr.tolist()
        residuals_arr = peaks - best_predictions_arr
        try:
            residual_summary = {
                'mean': float(np.nanmean(residuals_arr)),
                'std': float(np.nanstd(residuals_arr, ddof=1)) if residuals_arr.size > 1 else 0.0,
                'max_abs': float(np.nanmax(np.abs(residuals_arr))),
                'min': float(np.nanmin(residuals_arr)),
                'max': float(np.nanmax(residuals_arr)),
            }
        except Exception:
            residual_summary = None
    else:
        residuals_arr = None
    rmse_best = float(np.sqrt(np.mean((peaks - best_predictions_arr) ** 2))) if best_predictions_arr.size == peaks.size else float('nan')
    if abs(slope_for_lod) < eps or not np.isfinite(rmse_best):
        lod = float('inf')
        loq = float('inf')
    else:
        lod = 3.0 * rmse_best / abs(slope_for_lod)
        loq = 10.0 * rmse_best / abs(slope_for_lod)

    roi_center = float(np.median(peaks))

    absolute_shift_info: Optional[Dict[str, object]] = None
    try:
        if concs.size >= 2:
            baseline_wavelength = float(peaks[0])
            delta_wavelengths = peaks - baseline_wavelength
            abs_delta = np.abs(delta_wavelengths)
            abs_reg = linregress(concs, abs_delta)
            abs_preds = abs_reg.intercept + abs_reg.slope * concs
            abs_residuals = abs_delta - abs_preds
            abs_rmse = float(np.sqrt(np.mean(abs_residuals ** 2)))
            absolute_shift_info = {
                'concentrations': concs.tolist(),
                'baseline_wavelength': baseline_wavelength,
                'delta_wavelengths': delta_wavelengths.tolist(),
                'absolute_delta_wavelengths': abs_delta.tolist(),
                'slope': float(abs_reg.slope),
                'intercept': float(abs_reg.intercept),
                'r2': float(abs_reg.rvalue ** 2),
                'rmse': abs_rmse,
                'predicted_absolute_delta': abs_preds.tolist(),
            }
        else:
            absolute_shift_info = None
    except Exception:
        absolute_shift_info = None

    # Summarize feature candidates if available
    feature_summary = []
    try:
        for fr in feature_results:
            if isinstance(fr, dict):
                feature_summary.append({
                    'type': fr.get('type'),
                    'r2': float(fr.get('r2')) if fr.get('r2') is not None and np.isfinite(fr.get('r2')) else None,
                    'slope': float(fr.get('slope')) if fr.get('slope') is not None and np.isfinite(fr.get('slope')) else None,
                    'intercept': float(fr.get('intercept')) if fr.get('intercept') is not None and np.isfinite(fr.get('intercept')) else None,
                })
    except Exception:
        pass

    result_payload = {
        'concentrations': concs.tolist(),
        'transformed_concentrations': concs_x.tolist(),
        'transform_meta': transform_meta,
        'peak_wavelengths': peaks.tolist(),
        'selected_model': best_model['model'],
        'slope': float(best_model['slope']),
        'intercept': float(best_model['intercept']),
        'r2': float(best_model['r2']),
        'rmse': float(best_model['rmse']),
        'slope_se': float(best_model.get('slope_se', float('nan'))),
        'slope_ci_low': float(best_model.get('slope_ci_low', float('nan'))),
        'slope_ci_high': float(best_model.get('slope_ci_high', float('nan'))),
        'lod': float(lod),
        'loq': float(loq),
        'roi_center': roi_center,
        'linear_model': linear_model,
        'polynomial_model': poly_info,
        'langmuir_model': langmuir_info,
        'robust_model': robust_info,
        'bootstrap': bootstrap_info,
        'feature_candidates': feature_summary,
        'predictions': predictions_list,
        'residuals': residuals_arr.tolist() if residuals_arr is not None else None,
        'residual_summary': residual_summary,
        'absolute_shift': absolute_shift_info,
        'calibration_mode': 'canonical',
    }

    result_payload['canonical_model'] = {
        'concentrations': concs.tolist(),
        'peak_wavelengths': peaks.tolist(),
        'slope_nm_per_ppm': float(best_model['slope']),
        'intercept_nm': float(best_model['intercept']),
        'r2': float(best_model['r2']),
        'rmse_nm': float(best_model['rmse']),
        'lod_ppm': float(lod),
        'loq_ppm': float(loq),
        'predictions': predictions_list,
        'residuals': residuals_arr.tolist() if residuals_arr is not None else None,
        'uncertainty': result_payload.get('uncertainty'),
    }

    # Optional multivariate calibration (PLSR)
    plsr_cfg = calib_cfg.get('multivariate', {}) if isinstance(calib_cfg, dict) else {}
    if plsr_cfg.get('enabled', False):
        plsr_res = _fit_plsr_calibration(canonical, plsr_cfg)
        if plsr_res is not None:
            result_payload['plsr_model'] = plsr_res

            # If config requests auto select by CV R^2, allow PLSR to be the reported model
            select_mode = str(plsr_cfg.get('select_mode', 'report_only')).lower()
            plsr_score = plsr_res.get('r2_cv', float('-inf'))
            try:
                best_score = float(best_model.get('r2', float('-inf')))
            except Exception:
                best_score = float('-inf')
            if select_mode == 'auto' and np.isfinite(plsr_score) and plsr_score > best_score:
                result_payload['selected_model'] = 'plsr_cv'
                result_payload['slope'] = float('nan')
                result_payload['intercept'] = float('nan')
                result_payload['r2'] = float(plsr_score)
                result_payload['rmse'] = float(plsr_res.get('rmse_cv', float('nan')))
                result_payload['peak_wavelengths'] = peaks.tolist()

    # Consolidated uncertainty summary (CV + bootstrap) for selected model
    try:
        sel = str(result_payload.get('selected_model', 'linear')).lower()
        r2_cv_sel = float('nan')
        rmse_cv_sel = float('nan')
        if sel.startswith('poly') and isinstance(poly_info, dict):
            r2_cv_sel = float(poly_info.get('r2_cv', float('nan')))
            rmse_cv_sel = float(poly_info.get('rmse_cv', float('nan')))
        elif sel == 'robust_huber' and isinstance(robust_info, dict):
            r2_cv_sel = float(robust_info.get('r2_cv', float('nan')))
            rmse_cv_sel = float(robust_info.get('rmse_cv', float('nan')))
        elif sel == 'linear' and isinstance(linear_model, dict):
            r2_cv_sel = float(linear_model.get('r2_cv', float('nan')))
            rmse_cv_sel = float(linear_model.get('rmse_cv', float('nan')))
        elif sel == 'plsr_cv' and isinstance(result_payload.get('plsr_model', None), dict):
            pm = result_payload['plsr_model']
            r2_cv_sel = float(pm.get('r2_cv', float('nan')))
            rmse_cv_sel = float(pm.get('rmse_cv', float('nan')))
        result_payload['uncertainty'] = {
            'r2_cv': r2_cv_sel,
            'rmse_cv': rmse_cv_sel,
            'bootstrap': bootstrap_info,
        }
    except Exception:
        pass

    if responsive_delta:
        try:
            conc_rd = np.asarray(responsive_delta.get('concentrations', []), dtype=float)
            delta_rd = np.asarray(responsive_delta.get('delta_nm', []), dtype=float)
            slope_rd = float(responsive_delta.get('slope_nm_per_ppm', float('nan')))
            intercept_rd = float(responsive_delta.get('intercept_nm', float('nan')))
            r2_rd = float(responsive_delta.get('r2', float('nan')))
            rmse_rd = float(responsive_delta.get('rmse_nm', float('nan')))
            lod_rd = float(responsive_delta.get('lod_ppm', float('nan')))
            loq_rd = float(responsive_delta.get('loq_ppm', float('nan')))
            r2_cv_rd = float(responsive_delta.get('r2_cv', float('nan')))
            rmse_cv_rd = float(responsive_delta.get('rmse_cv', float('nan')))
            baseline_peak = intercept_rd if np.isfinite(intercept_rd) else (float(peaks[0]) if peaks.size else float('nan'))

            if conc_rd.size >= 2 and np.isfinite(slope_rd) and np.isfinite(baseline_peak):
                abs_predictions = baseline_peak + slope_rd * conc_rd
                abs_observed = baseline_peak + delta_rd
                residuals_abs = delta_rd - slope_rd * conc_rd

                result_payload.update({
                    'calibration_mode': 'responsive_delta',
                    'concentrations': conc_rd.tolist(),
                    'transformed_concentrations': conc_rd.tolist(),
                    'transform_meta': {'mode': 'direct_delta'},
                    'peak_wavelengths': abs_observed.tolist(),
                    'selected_model': 'responsive_delta_linear',
                    'slope': slope_rd,
                    'intercept': baseline_peak,
                    'r2': r2_rd,
                    'rmse': rmse_rd,
                    'slope_se': float('nan'),
                    'slope_ci_low': float(responsive_delta.get('slope_ci_low', float('nan'))),
                    'slope_ci_high': float(responsive_delta.get('slope_ci_high', float('nan'))),
                    'lod': lod_rd if np.isfinite(lod_rd) else result_payload.get('lod'),
                    'loq': loq_rd if np.isfinite(loq_rd) else result_payload.get('loq'),
                    'predictions': abs_predictions.tolist(),
                    'residuals': residuals_abs.tolist(),
                    'residual_summary': {
                        'mean': float(np.nanmean(residuals_abs)),
                        'std': float(np.nanstd(residuals_abs, ddof=1)) if np.isfinite(residuals_abs).sum() > 1 else 0.0,
                        'max_abs': float(np.nanmax(np.abs(residuals_abs))) if residuals_abs.size else float('nan'),
                        'min': float(np.nanmin(residuals_abs)) if residuals_abs.size else float('nan'),
                        'max': float(np.nanmax(residuals_abs)) if residuals_abs.size else float('nan'),
                    },
                })

                result_payload['uncertainty'] = {
                    'r2_cv': r2_cv_rd,
                    'rmse_cv': rmse_cv_rd,
                    'baseline_noise_nm': float(responsive_delta.get('baseline_noise_nm', float('nan'))),
                }

                result_payload['responsive_delta'] = responsive_delta
                result_payload['canonical_model']['predictions'] = predictions_list
                result_payload['canonical_model']['residuals'] = residuals_arr.tolist() if residuals_arr is not None else None
        except Exception as exc:
            print(f"[WARNING] Failed to apply responsive Δλ override: {exc}")

    return result_payload


# ----------------------
# Saving
# ----------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_canonical_spectra(canonical: Dict[float, pd.DataFrame], out_root: str) -> List[str]:
    out_dir = os.path.join(out_root, 'stable_selected')
    _ensure_dir(out_dir)
    paths: List[str] = []
    for conc, df in canonical.items():
        fname = f"{conc:g}_stable.csv"
        fpath = os.path.join(out_dir, fname)
        df.to_csv(fpath, index=False)
        paths.append(fpath)
    return paths


def save_aggregated_spectra(aggregated: Dict[float, Dict[str, pd.DataFrame]], out_root: str) -> Dict[float, Dict[str, str]]:
    base_dir = os.path.join(out_root, 'aggregated')
    _ensure_dir(base_dir)
    saved: Dict[float, Dict[str, str]] = {}
    for conc, trials in aggregated.items():
        conc_dir = os.path.join(base_dir, f"{conc:g}")
        _ensure_dir(conc_dir)
        for old in os.listdir(conc_dir):
            if old.lower().endswith('.csv'):
                try:
                    os.remove(os.path.join(conc_dir, old))
                except OSError:
                    pass
        saved[conc] = {}
        for trial, df in trials.items():
            safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
            fname = f"{safe_trial or 'trial'}.csv"
            fpath = os.path.join(conc_dir, fname)
            df.to_csv(fpath, index=False)
            saved[conc][trial] = fpath
    return saved


def _signal_column(df: pd.DataFrame) -> str:
    if 'absorbance' in df.columns:
        return 'absorbance'
    if 'transmittance' in df.columns:
        return 'transmittance'
    return 'intensity'


def _select_common_signal(frames: Sequence[pd.DataFrame],
                          priority: Sequence[str] = ('transmittance', 'intensity', 'absorbance')) -> Optional[str]:
    if not frames:
        return None

    for col in priority:
        if all(col in df.columns for df in frames):
            return col
    return None


def _common_signal_columns(frames: Sequence[pd.DataFrame]) -> List[str]:
    if not frames:
        return []

    common = set(frames[0].columns) - {'wavelength'}
    for df in frames[1:]:
        common &= (set(df.columns) - {'wavelength'})
    return sorted(common)


def compute_noise_metrics_map(aggregated: Dict[float, Dict[str, pd.DataFrame]]) -> Dict[float, Dict[str, object]]:
    metrics: Dict[float, Dict[str, object]] = {}
    for conc, trials in aggregated.items():
        metrics[conc] = {}
        for trial, df in trials.items():
            col = _signal_column(df)
            nm = estimate_noise_metrics(df['wavelength'].values, df[col].values)
            metrics[conc][trial] = asdict(nm)
    return metrics


def save_noise_metrics(metrics: Dict[float, Dict[str, object]], out_root: str) -> str:
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    serializable = {str(conc): trials for conc, trials in metrics.items()}
    out_path = os.path.join(metrics_dir, 'noise_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    return out_path


def compute_environment_summary(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]]) -> Dict[str, object]:
    env_cfg = CONFIG.get('environment', {}) if isinstance(CONFIG, dict) else {}
    if not env_cfg:
        return {}
    ref = env_cfg.get('reference', {}) if isinstance(env_cfg.get('reference', {}), dict) else {}
    coeffs = env_cfg.get('coefficients', {}) if isinstance(env_cfg.get('coefficients', {}), dict) else {}
    override = env_cfg.get('override', {}) if isinstance(env_cfg.get('override', {}), dict) else {}
    T_ref = float(ref.get('temperature', 25.0))
    H_ref = float(ref.get('humidity', 50.0))
    cT = coeffs.get('temperature', None)
    cH = coeffs.get('humidity', None)
    t_vals: List[float] = []
    h_vals: List[float] = []
    offsets: List[float] = []
    for conc, trials in stable_by_conc.items():
        for trial, df in trials.items():
            T_val = None
            H_val = None
            if 'temperature' in df.columns:
                try:
                    T_val = float(pd.to_numeric(df['temperature'], errors='coerce').dropna().mean())
                except Exception:
                    T_val = None
            if 'humidity' in df.columns:
                try:
                    H_val = float(pd.to_numeric(df['humidity'], errors='coerce').dropna().mean())
                except Exception:
                    H_val = None
            if T_val is None and override.get('temperature') is not None:
                T_val = float(override.get('temperature'))
            if H_val is None and override.get('humidity') is not None:
                H_val = float(override.get('humidity'))
            if T_val is not None:
                t_vals.append(T_val)
            if H_val is not None:
                h_vals.append(H_val)
            off = 0.0
            if cT is not None and T_val is not None:
                off += float(cT) * (T_val - T_ref)
            if cH is not None and H_val is not None:
                off += float(cH) * (H_val - H_ref)
            if off != 0.0 and np.isfinite(off):
                offsets.append(float(off))
    info: Dict[str, object] = {
        'enabled': bool(env_cfg.get('enabled', False)),
        'apply_to_frames': bool(env_cfg.get('apply_to_frames', False)),
        'apply_to_transmittance': bool(env_cfg.get('apply_to_transmittance', True)),
        'reference': {'temperature': T_ref, 'humidity': H_ref},
        'coefficients': {'temperature': float(cT) if cT is not None else None,
                         'humidity': float(cH) if cH is not None else None},
        'override': {'temperature': override.get('temperature', None), 'humidity': override.get('humidity', None)},
        'temperature_mean': float(np.mean(t_vals)) if t_vals else None,
        'humidity_mean': float(np.mean(h_vals)) if h_vals else None,
        'offset_mean': float(np.mean(offsets)) if offsets else 0.0,
        'offset_std': float(np.std(offsets, ddof=1)) if len(offsets) > 1 else 0.0,
        'offset_count': int(len(offsets)),
    }
    return info


def compute_environment_coefficients(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]],
                                     calib: Dict[str, object]) -> Dict[str, object]:
    """Estimate first-order environment coefficients from calibration outputs.

    Fits y ≈ β0 + βC*x + cT*(T-Tref) + cH*(H-Href) on per-concentration means.
    Computes improvement vs concentration-only fit.
    """
    try:
        env_cfg = CONFIG.get('environment', {}) if isinstance(CONFIG, dict) else {}
        ref = env_cfg.get('reference', {}) if isinstance(env_cfg.get('reference', {}), dict) else {}
        T_ref = float(ref.get('temperature', 25.0))
        H_ref = float(ref.get('humidity', 50.0))

        conc_seq = np.asarray(calib.get('concentrations', []), dtype=float)
        x_seq = np.asarray(calib.get('transformed_concentrations', conc_seq), dtype=float)
        y_seq = np.asarray(calib.get('peak_wavelengths', []), dtype=float)
        if not (conc_seq.size and y_seq.size and conc_seq.size == y_seq.size):
            return {}

        # Collect per-concentration T/H means
        t_by_conc: Dict[float, float] = {}
        h_by_conc: Dict[float, float] = {}
        for conc, trials in stable_by_conc.items():
            t_vals = []
            h_vals = []
            for df in trials.values():
                if 'temperature' in df.columns:
                    tv = pd.to_numeric(df['temperature'], errors='coerce').dropna()
                    if not tv.empty:
                        t_vals.append(float(tv.mean()))
                if 'humidity' in df.columns:
                    hv = pd.to_numeric(df['humidity'], errors='coerce').dropna()
                    if not hv.empty:
                        h_vals.append(float(hv.mean()))
            if t_vals:
                t_by_conc[float(conc)] = float(np.mean(t_vals))
            if h_vals:
                h_by_conc[float(conc)] = float(np.mean(h_vals))

        T_arr = np.full_like(conc_seq, np.nan, dtype=float)
        H_arr = np.full_like(conc_seq, np.nan, dtype=float)
        for i, c in enumerate(conc_seq):
            if float(c) in t_by_conc:
                T_arr[i] = t_by_conc[float(c)]
            if float(c) in h_by_conc:
                H_arr[i] = h_by_conc[float(c)]

        # If all missing, nothing to do
        have_T = np.isfinite(T_arr).any()
        have_H = np.isfinite(H_arr).any()
        if not (have_T or have_H):
            return {}

        # Center env variables around reference
        dT = np.where(np.isfinite(T_arr), T_arr - T_ref, 0.0)
        dH = np.where(np.isfinite(H_arr), H_arr - H_ref, 0.0)

        # Concentration-only fit
        Xc = np.column_stack([np.ones_like(x_seq), x_seq])
        beta_c, *_ = np.linalg.lstsq(Xc, y_seq, rcond=None)
        yhat_c = Xc @ beta_c
        ss_res_c = float(np.sum((y_seq - yhat_c) ** 2))
        ss_tot = float(np.sum((y_seq - np.mean(y_seq)) ** 2))
        r2_c = 1.0 - ss_res_c / ss_tot if ss_tot > 0 else float('nan')
        rmse_c = float(np.sqrt(np.mean((y_seq - yhat_c) ** 2)))

        # Full fit with available env vars
        cols = [np.ones_like(x_seq), x_seq]
        names = ['intercept', 'beta_c']
        if have_T and (np.nanstd(T_arr) > 0):
            cols.append(dT)
            names.append('cT')
        if have_H and (np.nanstd(H_arr) > 0):
            cols.append(dH)
            names.append('cH')
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y_seq, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y_seq - yhat) ** 2))
        r2_full = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        rmse_full = float(np.sqrt(np.mean((y_seq - yhat) ** 2)))

        coeffs: Dict[str, float] = {}
        for name, val in zip(names, beta):
            coeffs[name] = float(val)

        out = {
            'estimated_coefficients': {
                'temperature': float(coeffs.get('cT')) if 'cT' in coeffs else None,
                'humidity': float(coeffs.get('cH')) if 'cH' in coeffs else None,
                'beta_c': float(coeffs.get('beta_c')),
                'intercept': float(coeffs.get('intercept')),
            },
            'r2_conc_only': float(r2_c),
            'r2_full': float(r2_full),
            'delta_r2': float(r2_full - r2_c) if np.isfinite(r2_full) and np.isfinite(r2_c) else float('nan'),
            'rmse_conc_only': float(rmse_c),
            'rmse_full': float(rmse_full),
            'delta_rmse': float(rmse_c - rmse_full) if np.isfinite(rmse_c) and np.isfinite(rmse_full) else float('nan'),
            'n_points': int(len(y_seq)),
        }
        return out
    except Exception:
        return {}

def summarize_quality_control(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]],
                              noise_metrics: Dict[float, Dict[str, object]]) -> Dict[str, object]:
    """Summarize simple QC metrics across dataset: SNR distribution and trial-to-trial RSD.

    Uses existing noise metrics to compute SNR stats and estimates trial-to-trial
    relative standard deviation (RSD) using mean signal per trial.
    """
    # Thresholds may be configured under CONFIG['quality']
    qcfg = CONFIG.get('quality', {}) if isinstance(CONFIG, dict) else {}
    min_snr_req = float(qcfg.get('min_snr', 10.0))
    max_rsd_req = float(qcfg.get('max_rsd', 5.0))  # percent

    # Aggregate SNRs from noise metrics
    snr_values: List[float] = []
    for conc, trials in (noise_metrics or {}).items():
        try:
            for tinfo in trials.values():
                snr = float(tinfo.get('snr', float('nan')))
                if np.isfinite(snr):
                    snr_values.append(snr)
        except Exception:
            continue

    # Estimate trial-to-trial RSD per concentration using average signal level
    rsd_by_conc: Dict[float, float] = {}
    for conc, trials in (stable_by_conc or {}).items():
        vals: List[float] = []
        for df in trials.values():
            col = _signal_column(df)
            arr = df[col].to_numpy(dtype=float)
            if arr.size:
                vals.append(float(np.mean(arr)))
        if len(vals) >= 2:
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            rsd = (std_v / mean_v * 100.0) if abs(mean_v) > 1e-12 else float('inf')
            rsd_by_conc[float(conc)] = rsd
    max_rsd = float(np.nanmax(list(rsd_by_conc.values()))) if rsd_by_conc else float('nan')

    qc = {
        'min_snr': float(np.nanmin(snr_values)) if snr_values else float('nan'),
        'median_snr': float(np.nanmedian(snr_values)) if snr_values else float('nan'),
        'max_rsd_percent': max_rsd,
        'snr_threshold': min_snr_req,
        'rsd_threshold_percent': max_rsd_req,
    }
    qc['snr_pass'] = bool(np.isfinite(qc['min_snr']) and qc['min_snr'] >= min_snr_req)
    qc['rsd_pass'] = bool(np.isfinite(qc['max_rsd_percent']) and qc['max_rsd_percent'] <= max_rsd_req)
    qc['overall_pass'] = qc['snr_pass'] and qc['rsd_pass']
    qc['rsd_by_concentration'] = {str(k): float(v) for k, v in rsd_by_conc.items()}
    return qc


def save_quality_summary(qc: Dict[str, object], out_root: str) -> str:
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, 'qc_summary.json')
    with open(out_path, 'w') as f:
        json.dump(qc, f, indent=2)
    return out_path


def save_aggregated_summary(aggregated: Dict[float, Dict[str, pd.DataFrame]],
                            noise_metrics: Dict[float, Dict[str, object]],
                            out_root: str) -> str:
    rows = []
    for conc, trials in aggregated.items():
        for trial, df in trials.items():
            col = _signal_column(df)
            arr = df[col].values
            nm = noise_metrics.get(conc, {}).get(trial, {})
            rows.append({
                'concentration': conc,
                'trial': trial,
                'signal_column': col,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'rms': nm.get('rms', np.nan),
                'mad': nm.get('mad', np.nan),
                'spectral_entropy': nm.get('spectral_entropy', np.nan),
                'snr': nm.get('snr', np.nan),
            })
    summary = pd.DataFrame(rows)
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, 'aggregated_summary.csv')
    summary.to_csv(out_path, index=False)
    return out_path


def _stack_trials_for_response(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    base_wl: Optional[np.ndarray] = None
    stacked: List[np.ndarray] = []
    conc_labels: List[float] = []
    avg_by_conc: Dict[float, np.ndarray] = {}

    for conc, trials in sorted(stable_by_conc.items(), key=lambda kv: kv[0]):
        trial_arrays: List[np.ndarray] = []
        for df in trials.values():
            col = _signal_column(df)
            wl = df['wavelength'].values
            y = df[col].values
            if base_wl is None:
                base_wl = wl
            elif not np.array_equal(base_wl, wl):
                y = np.interp(base_wl, wl, y)
            stacked.append(y)
            conc_labels.append(conc)
            trial_arrays.append(y)
        if trial_arrays:
            avg_by_conc[conc] = np.mean(trial_arrays, axis=0)

    if base_wl is None:
        raise ValueError("No spectra available for concentration response analysis")

    return (
        base_wl,
        np.vstack(stacked) if stacked else np.zeros((0, len(base_wl))),
        np.array(conc_labels, dtype=float),
        avg_by_conc,
    )


def compute_concentration_response(
    stable_by_conc: Dict[float, Dict[str, pd.DataFrame]],
    override_min_wavelength: Optional[float] = None,
    override_max_wavelength: Optional[float] = None,
    top_k_candidates: int = 0,
    debug_out_root: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[float, np.ndarray]]:
    wl, Y, concs, avg_by_conc = _stack_trials_for_response(stable_by_conc)
    if Y.size == 0:
        raise ValueError("No spectra available for concentration response analysis")

    slopes: List[float] = []
    intercepts: List[float] = []
    r2_vals: List[float] = []
    poly_r2_vals: List[float] = []
    corr_vals: List[float] = []
    residual_stds: List[float] = []

    roi_cfg = CONFIG.get('roi', {})
    debug_enabled = roi_cfg.get('debug_regressions', True)
    selection_metric = str(roi_cfg.get('selection_metric', 'r2')).lower()
    min_r2 = float(roi_cfg.get('min_r2', 0.0))
    r2_weight = float(np.clip(roi_cfg.get('r2_weight', 1.0), 0.0, 1.0))
    band_half_width_cfg = roi_cfg.get('band_half_width', None)
    band_window = int(max(1, roi_cfg.get('band_window', 0)))
    expected_trend = str(roi_cfg.get('expected_trend', 'any')).lower()
    min_corr = float(np.clip(roi_cfg.get('min_corr', 0.0), 0.0, 1.0))
    derivative_weight = float(np.clip(roi_cfg.get('derivative_weight', 0.0), 0.0, 1.0))
    ratio_weight = float(np.clip(roi_cfg.get('ratio_weight', 0.0), 0.0, 1.0))
    ratio_half_width = int(max(1, roi_cfg.get('ratio_half_width', 5)))
    slope_noise_weight = float(np.clip(roi_cfg.get('slope_noise_weight', 0.0), 0.0, 1.0))
    min_slope_to_noise = float(max(0.0, roi_cfg.get('min_slope_to_noise', 0.0)))
    min_abs_slope_cfg = float(max(0.0, roi_cfg.get('min_abs_slope', 0.0)))

    trend_modes_cfg = roi_cfg.get('trend_modes', None)
    allowed_modes = {'increasing', 'decreasing', 'any', 'valley', 'peak', 'dip'}
    if isinstance(trend_modes_cfg, (list, tuple, set)):
        trend_modes = [str(m).lower() for m in trend_modes_cfg if str(m).lower() in allowed_modes]
    else:
        trend_modes = []
    if not trend_modes:
        trend_modes = [expected_trend]
    if expected_trend == 'any' and 'any' not in trend_modes:
        trend_modes.append('any')
    trend_modes = list(dict.fromkeys(trend_modes))

    min_wavelength = override_min_wavelength if override_min_wavelength is not None else roi_cfg.get('min_wavelength', None)
    max_wavelength = override_max_wavelength if override_max_wavelength is not None else roi_cfg.get('max_wavelength', None)
    validation_cfg = roi_cfg.get('validation', {}) or {}
    alt_model_cfg = roi_cfg.get('alternative_models', {}) or {}
    adaptive_band_cfg = roi_cfg.get('adaptive_band', {}) or {}

    mask_wl = np.ones_like(wl, dtype=bool)
    if min_wavelength is not None:
        min_wavelength = float(min_wavelength)
        mask_wl &= (wl >= min_wavelength)
    if max_wavelength is not None:
        max_wavelength = float(max_wavelength)
        mask_wl &= (wl <= max_wavelength)

    if not np.any(mask_wl):
        raise ValueError("No wavelengths remain after applying ROI wavelength constraints")

    wl = wl[mask_wl]
    Y = Y[:, mask_wl]
    for conc_key, arr in avg_by_conc.items():
        avg_by_conc[conc_key] = arr[mask_wl]

    # Feature matrices
    derivative_matrix = None
    ratio_matrix = None
    if derivative_weight > 0:
        derivative_matrix = np.gradient(Y, wl, axis=1)
    if ratio_weight > 0:
        ratio_matrix = _compute_band_ratio_matrix(Y, ratio_half_width)

    alt_enabled = bool(alt_model_cfg.get('enabled', False))
    poly_degree = int(max(1, alt_model_cfg.get('polynomial_degree', 2))) if alt_enabled else 1
    adaptive_enabled = bool(adaptive_band_cfg.get('enabled', False))
    slope_fraction = float(np.clip(adaptive_band_cfg.get('slope_fraction', 0.6), 0.0, 1.0)) if adaptive_enabled else 0.0
    adaptive_max_half = int(adaptive_band_cfg.get('max_half_width', band_half_width_cfg if band_half_width_cfg is not None else 20)) if adaptive_enabled else 0

    for j in range(Y.shape[1]):
        column = Y[:, j]
        res = linregress(concs, column)
        preds_lin = res.intercept + res.slope * concs
        residual_std = float(np.std(column - preds_lin, ddof=1)) if column.size > 1 else 0.0
        slopes.append(float(res.slope))
        intercepts.append(float(res.intercept))
        r_val = float(res.rvalue) if not np.isnan(res.rvalue) else np.nan
        r_sq = float(r_val ** 2) if not np.isnan(r_val) else np.nan
        r2_vals.append(r_sq)
        corr_vals.append(r_val)
        residual_stds.append(residual_std)

        if alt_enabled and len(concs) > poly_degree:
            try:
                coeffs = np.polyfit(concs, column, poly_degree)
                preds = np.polyval(coeffs, concs)
                ss_res = float(np.sum((column - preds) ** 2))
                ss_tot = float(np.sum((column - np.mean(column)) ** 2))
                poly_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except np.linalg.LinAlgError:
                poly_r2 = np.nan
        else:
            poly_r2 = np.nan
        poly_r2_vals.append(poly_r2)

    slopes_arr = np.array(slopes)
    abs_slopes = np.abs(slopes_arr)
    r2_arr = np.array(r2_vals)
    poly_r2_arr = np.array(poly_r2_vals)
    corr_arr = np.array(corr_vals)
    residual_std_arr = np.array(residual_stds)

    if debug_enabled:
        debug_df = pd.DataFrame({
            'wavelength': wl,
            'slope': slopes_arr,
            'abs_slope': abs_slopes,
            'intercept': intercepts,
            'r2': r2_arr,
            'correlation': corr_arr,
            'residual_std': residual_std_arr,
        })
        if debug_out_root is not None:
            try:
                metrics_dir = os.path.join(debug_out_root, 'metrics')
                _ensure_dir(metrics_dir)
                debug_path = os.path.join(metrics_dir, 'debug_all_wavelength_regressions.csv')
            except Exception:
                debug_path = os.path.join(os.getcwd(), 'debug_all_wavelength_regressions.csv')
        else:
            debug_path = os.path.join(os.getcwd(), 'debug_all_wavelength_regressions.csv')
        try:
            debug_df.to_csv(debug_path, index=False)
            print(f"\n[DEBUG] Saved all {len(wl)} wavelength regressions to: {debug_path}")
            print(f"[DEBUG] Wavelength range: {wl.min():.1f} - {wl.max():.1f} nm")
            print(f"[DEBUG] Best R²: {r2_arr.max():.4f} at {wl[r2_arr.argmax()]:.2f} nm")
            print(f"[DEBUG] Max |slope|: {abs_slopes.max():.6f} at {wl[abs_slopes.argmax()]:.2f} nm")
            top5_r2_idx = np.argsort(r2_arr)[-5:][::-1]
            print("[DEBUG] Top 5 by R²:")
            for idx in top5_r2_idx:
                print(f"  λ={wl[idx]:.2f} nm: R²={r2_arr[idx]:.4f}, slope={slopes_arr[idx]:.6f}, corr={corr_arr[idx]:.4f}")
        except Exception as e:
            print(f"[DEBUG] Failed to save debug CSV: {e}")

    noise_per_band: Optional[np.ndarray] = None
    if slope_noise_weight > 0 or min_slope_to_noise > 0:
        noise_per_band = np.full_like(abs_slopes, np.nan, dtype=float)
        repeatability = CONFIG.get('_last_repeatability', {}) or {}
        global_stats = repeatability.get('global', {})
        global_std = float(global_stats.get('std_transmittance', 0.0) or 0.0)
        if global_std > 0:
            noise_per_band = abs_slopes / global_std

    slope_scores = abs_slopes.copy()
    r2_scores = r2_arr.copy()
    poly_r2_scores = poly_r2_arr.copy()
    corr_scores = corr_arr.copy()

    if derivative_matrix is not None and derivative_weight > 0:
        deriv_slopes = []
        for j in range(derivative_matrix.shape[1]):
            col = derivative_matrix[:, j]
            res = linregress(concs, col)
            deriv_slopes.append(float(res.slope))
        deriv_arr = np.abs(np.array(deriv_slopes))
        slope_scores = (1 - derivative_weight) * slope_scores + derivative_weight * deriv_arr

    if ratio_matrix is not None and ratio_weight > 0:
        ratio_slopes = []
        for j in range(ratio_matrix.shape[1]):
            col = ratio_matrix[:, j]
            res = linregress(concs, col)
            ratio_slopes.append(float(res.slope))
        ratio_arr = np.abs(np.array(ratio_slopes))
        slope_scores = (1 - ratio_weight) * slope_scores + ratio_weight * ratio_arr

    if band_window > 1:
        kernel = np.ones(band_window) / band_window
        slope_scores = np.convolve(slope_scores, kernel, mode='same')
        r2_scores = np.convolve(r2_scores, kernel, mode='same')
        if not np.all(np.isnan(poly_r2_scores)):
            poly_r2_scores = np.convolve(poly_r2_scores, kernel, mode='same')
        if not np.all(np.isnan(corr_scores)):
            corr_scores = np.convolve(corr_scores, kernel, mode='same')

    # Normalised scores for hybrid/alternative ROI selection
    max_abs_slope = np.nanmax(abs_slopes)
    norm_slopes = abs_slopes / max_abs_slope if max_abs_slope and not np.isnan(max_abs_slope) else np.zeros_like(abs_slopes)
    r2_clean = np.clip(r2_scores, 0.0, 1.0)
    poly_r2_clean = np.clip(poly_r2_scores, 0.0, 1.0)

    if selection_metric in ('poly', 'poly_r2') and alt_enabled:
        score = poly_r2_clean
        if min_r2 > 0:
            score[poly_r2_clean < min_r2] = 0.0
    elif selection_metric == 'slope':
        score = norm_slopes * (np.minimum(1.0, r2_clean / max(min_r2, 1e-6)) if min_r2 > 0 else 1.0)
    elif selection_metric == 'hybrid':
        slope_component = norm_slopes
        if alt_enabled and not np.all(np.isnan(poly_r2_clean)):
            r2_component = poly_r2_clean
        else:
            r2_component = r2_clean
        score = r2_weight * r2_component + (1.0 - r2_weight) * slope_component
        if min_r2 > 0:
            score[r2_clean < min_r2] = 0.0
    else:  # default to R^2
        score = poly_r2_clean if (alt_enabled and selection_metric == 'poly_r2') else r2_clean
        if min_r2 > 0:
            score[r2_clean < min_r2] = 0.0

    if noise_per_band is not None and slope_noise_weight > 0:
        noise_scaled = np.nan_to_num(noise_per_band, nan=0.0, posinf=0.0, neginf=0.0)
        score = (1.0 - slope_noise_weight) * score + slope_noise_weight * noise_scaled

    if noise_per_band is not None and min_slope_to_noise > 0:
        insufficient = (noise_per_band < min_slope_to_noise) | np.isnan(noise_per_band)
        score[insufficient] = 0.0

    if min_abs_slope_cfg > 0:
        low_slope_mask = abs_slopes < min_abs_slope_cfg
        score[low_slope_mask] = 0.0
    # Apply expected trend / correlation constraints across requested trend modes
    idx_all = np.arange(len(score))
    best_idx = None
    best_score_val = -np.inf

    def _apply_trend_filter(trend: str, corr_array: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(corr_array)
        if trend == 'decreasing' or trend == 'valley' or trend == 'dip':
            thresh = -min_corr if min_corr > 0 else 0.0
            mask &= corr_array <= thresh
            if min_corr > 0 and not mask.any():
                mask = (~np.isnan(corr_array)) & (corr_array < 0.0)
        elif trend == 'increasing' or trend == 'peak':
            thresh = min_corr if min_corr > 0 else 0.0
            mask &= corr_array >= thresh
            if min_corr > 0 and not mask.any():
                mask = (~np.isnan(corr_array)) & (corr_array > 0.0)
        else:  # 'any'
            if min_corr > 0:
                mask &= np.abs(corr_array) >= min_corr
        return mask

    for mode in trend_modes:
        eligible_mask = _apply_trend_filter(mode, corr_arr)
        if not eligible_mask.any():
            continue

        mode_score = score.copy()
        mode_score[~eligible_mask] = 0.0

        valid_idx = idx_all[eligible_mask & (~np.isnan(mode_score))]
        if valid_idx.size == 0:
            continue

        mode_best_idx = int(valid_idx[np.nanargmax(mode_score[valid_idx])])
        mode_best_score = float(mode_score[mode_best_idx])

        # Prefer higher scores; break ties with absolute slope magnitude
        if mode_best_score > best_score_val:
            best_score_val = mode_best_score
            best_idx = mode_best_idx
        elif mode_best_score == best_score_val and best_idx is not None:
            if abs_slopes[mode_best_idx] > abs_slopes[best_idx]:
                best_idx = mode_best_idx

    if best_idx is None:
        eligible_mask = ~np.isnan(score)
        valid_idx = idx_all[eligible_mask & (~np.isnan(score))]
        if valid_idx.size > 0:
            best_idx = int(valid_idx[np.nanargmax(score[valid_idx])])
        else:
            fallback_idx = idx_all[eligible_mask & (~np.isnan(r2_scores))]
            if fallback_idx.size > 0:
                best_idx = int(fallback_idx[np.nanargmax(r2_scores[fallback_idx])])
            else:
                best_idx = int(np.nanargmax(norm_slopes))

    default_half = max(3, min(25, max(3, len(wl) // 40)))
    half_width = int(band_half_width_cfg) if band_half_width_cfg is not None else default_half
    half_width = max(1, min(half_width, len(wl) // 2))
    roi_start_idx = max(0, best_idx - half_width)
    roi_end_idx = min(len(wl) - 1, best_idx + half_width)

    if adaptive_enabled and max_abs_slope > 0:
        target = slope_fraction * abs_slopes[best_idx]
        max_hw = max(1, min(adaptive_max_half, len(wl) // 2))
        left = best_idx
        while left > 0 and abs_slopes[left - 1] >= target and (best_idx - (left - 1)) <= max_hw:
            left -= 1
        right = best_idx
        while right < len(wl) - 1 and abs_slopes[right + 1] >= target and ((right + 1) - best_idx) <= max_hw:
            right += 1
        roi_start_idx = min(roi_start_idx, left)
        roi_end_idx = max(roi_end_idx, right)

    validation_result = {}
    expected_center = validation_cfg.get('expected_center')
    tolerance = float(validation_cfg.get('tolerance', 0.0))
    if expected_center is not None:
        observed = float(wl[best_idx])
        deviation = observed - float(expected_center)
        passes = abs(deviation) <= abs(tolerance)
        validation_result = {
            'expected_center': float(expected_center),
            'observed_center': observed,
            'tolerance': tolerance,
            'deviation': deviation,
            'within_tolerance': passes,
            'notes': validation_cfg.get('notes', ''),
        }

    # Build optional top-K candidate list (ranked by selection score)
    candidates: List[Dict[str, float]] = []
    if top_k_candidates and top_k_candidates > 0:
        idx_all = np.arange(len(score))
        valid = idx_all[~np.isnan(score)]
        if valid.size > 0:
            order = valid[np.argsort(score[valid])[::-1]]
            k = int(min(top_k_candidates, order.size))
            for idx in order[:k]:
                idx = int(idx)
                start_idx = max(0, idx - half_width)
                end_idx = min(len(wl) - 1, idx + half_width)
                slope_to_noise = float('nan')
                if noise_per_band is not None and 0 <= idx < noise_per_band.size:
                    slope_to_noise = float(noise_per_band[idx])
                candidates.append({
                    'wavelength': float(wl[idx]),
                    'r2': float(r2_arr[idx]) if not np.isnan(r2_arr[idx]) else float('nan'),
                    'slope': float(slopes_arr[idx]),
                    'slope_to_noise': slope_to_noise,
                    'corr': float(corr_arr[idx]) if not np.isnan(corr_arr[idx]) else float('nan'),
                    'score': float(score[idx]) if not np.isnan(score[idx]) else float('nan'),
                    'roi_start_wavelength': float(wl[start_idx]),
                    'roi_end_wavelength': float(wl[end_idx]),
                })

    response = {
        'wavelengths': wl.tolist(),
        'slopes': slopes_arr.tolist(),
        'intercepts': intercepts,
        'r_squared': r2_vals,
        'poly_r_squared': poly_r2_vals if alt_enabled else None,
        'correlations': corr_arr.tolist(),
        'max_correlation': float(corr_arr[best_idx]) if not np.isnan(corr_arr[best_idx]) else float('nan'),
        'max_slope': float(slopes_arr[best_idx]),
        'max_slope_wavelength': float(wl[best_idx]),
        'max_r_squared': float(r2_arr[best_idx]) if not np.isnan(r2_arr[best_idx]) else float('nan'),
        'max_poly_r_squared': float(poly_r2_arr[best_idx]) if alt_enabled and not np.isnan(poly_r2_arr[best_idx]) else None,
        'roi_selection_metric': selection_metric,
        'roi_score': float(score[best_idx]) if not np.isnan(score[best_idx]) else float('nan'),
        'roi_start_index': int(roi_start_idx),
        'roi_end_index': int(roi_end_idx),
        'roi_start_wavelength': float(wl[roi_start_idx]),
        'roi_end_wavelength': float(wl[roi_end_idx]),
        'validation': validation_result,
        'candidates': candidates,
    }

    return response, avg_by_conc


def compute_roi_repeatability(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]],
                              response: Dict[str, object]) -> Dict[str, object]:
    start = response['roi_start_wavelength']
    end = response['roi_end_wavelength']
    center = 0.5 * (start + end)
    roi_span = end - start

    repeatability: Dict[str, object] = {
        'indices': [],
        'per_concentration': {},
        'global': {},
    }

    global_vals: List[float] = []
    for conc, trials in stable_by_conc.items():
        trial_means: List[float] = []
        for trial_name, df in trials.items():
            col = _signal_column(df)
            wl = df['wavelength'].values
            y = df[col].values
            mask = (wl >= start) & (wl <= end)
            if not mask.any():
                # fallback: interpolate around center
                trial_means.append(float(np.interp(center, wl, y)))
                continue
            trial_means.append(float(np.nanmean(y[mask])))

        if not trial_means:
            continue
        arr = np.array(trial_means, dtype=float)
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0
        cv = float(std_val / mean_val) if mean_val else float('inf')
        repeatability['per_concentration'][str(conc)] = {
            'mean_transmittance': mean_val,
            'std_transmittance': std_val,
            'cv_transmittance': cv,
            'trial_count': int(len(trial_means)),
        }
        global_vals.extend(trial_means)

    if global_vals:
        gv = np.array(global_vals, dtype=float)
        repeatability['global'] = {
            'mean_transmittance': float(np.nanmean(gv)),
            'std_transmittance': float(np.nanstd(gv, ddof=1) if gv.size > 1 else 0.0),
            'cv_transmittance': float((np.nanstd(gv, ddof=1) / np.nanmean(gv)) if gv.size > 1 and np.nanmean(gv) else float('inf')),
            'count': int(gv.size),
        }
    repeatability['indices'] = [start, end]
    repeatability['roi_width'] = roi_span
    return repeatability


def summarize_top_comparison(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    for mode, payload in results.items():
        perf = payload.get('performance', {}) or {}
        roi_perf = perf.get('roi_performance', {}) if isinstance(perf, dict) else {}
        summary.append({
            'mode': mode,
            'canonical_count': payload.get('canonical_count'),
            'roi_max_r2': payload.get('response_stats', {}).get('max_r_squared'),
            'roi_max_slope': payload.get('response_stats', {}).get('max_slope'),
            'roi_center': payload.get('response_stats', {}).get('max_slope_wavelength'),
            'lod': roi_perf.get('lod'),
            'loq': roi_perf.get('loq'),
            'metrics_path': payload.get('metrics_path'),
            'plot_path': payload.get('plot_path'),
        })
    return summary


def compute_roi_performance(repeatability: Dict[str, object]) -> Dict[str, object]:
    per_conc = repeatability.get('per_concentration', {})
    if not per_conc:
        return {}

    concs: List[float] = []
    means: List[float] = []
    stds: List[float] = []
    cvs: List[float] = []

    for conc_str, stats in per_conc.items():
        try:
            conc_val = float(conc_str)
        except ValueError:
            continue
        concs.append(conc_val)
        means.append(float(stats.get('mean_transmittance', np.nan)))
        stds.append(float(stats.get('std_transmittance', np.nan)))
        cvs.append(float(stats.get('cv_transmittance', np.nan)))

    if len(concs) < 2:
        return {}

    order = np.argsort(concs)
    concs_arr = np.array(concs)[order]
    means_arr = np.array(means)[order]
    stds_arr = np.array(stds)[order]
    cvs_arr = np.array(cvs)[order]

    reg = linregress(concs_arr, means_arr)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r2 = float(reg.rvalue ** 2)
    rmse = float(np.sqrt(np.mean((means_arr - (intercept + slope * concs_arr)) ** 2)))

    dynamic_range = float(np.nanmax(means_arr) - np.nanmin(means_arr))
    ppm_span = float(np.nanmax(concs_arr) - np.nanmin(concs_arr)) or 1.0
    sensitivity = float(dynamic_range / ppm_span)

    global_stats = repeatability.get('global', {})
    global_std = float(global_stats.get('std_transmittance', np.nan) or 0.0)
    if slope == 0.0:
        lod = float('inf')
        loq = float('inf')
    else:
        lod = float(3 * global_std / abs(slope))
        loq = float(10 * global_std / abs(slope))

    performance = {
        'regression_slope': slope,
        'regression_intercept': intercept,
        'regression_r2': r2,
        'regression_rmse': rmse,
        'dynamic_range': dynamic_range,
        'dynamic_range_per_ppm': sensitivity,
        'mean_cv': float(np.nanmean(cvs_arr)),
        'max_cv': float(np.nanmax(cvs_arr)),
        'min_cv': float(np.nanmin(cvs_arr)),
        'lod_ppm': lod,
        'loq_ppm': loq,
        'ppm_span': ppm_span,
        'concentrations': concs_arr.tolist(),
        'mean_transmittance_per_concentration': means_arr.tolist(),
        'std_transmittance_per_concentration': stds_arr.tolist(),
        'cv_transmittance_per_concentration': cvs_arr.tolist(),
    }
    return performance


def save_roi_performance_metrics(performance: Dict[str, object], out_root: str) -> Optional[str]:
    if not performance:
        return None
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, 'roi_performance.json')
    with open(out_path, 'w') as f:
        json.dump(performance, f, indent=2)
    return out_path


def save_roi_discovery_plot(discovery: Dict[str, object], out_root: str) -> Optional[str]:
    candidates = discovery.get('candidates', []) if isinstance(discovery, dict) else []
    if not isinstance(candidates, list) or not candidates:
        return None

    centers: List[float] = []
    slopes: List[float] = []
    r2_vals: List[float] = []
    snr_vals: List[float] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        c = cand.get('center_nm')
        m = cand.get('slope_nm_per_ppm')
        r2 = cand.get('r2')
        snr = cand.get('snr')
        try:
            c_val = float(c)
            m_val = float(m)
        except Exception:
            continue
        centers.append(c_val)
        slopes.append(m_val)
        try:
            r2_vals.append(float(r2))
        except Exception:
            r2_vals.append(float('nan'))
        try:
            snr_vals.append(float(snr))
        except Exception:
            snr_vals.append(float('nan'))

    if not centers:
        return None

    centers_arr = np.array(centers, dtype=float)
    slopes_arr = np.array(slopes, dtype=float)
    r2_arr = np.array(r2_vals, dtype=float)

    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(plots_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(centers_arr, slopes_arr, c=r2_arr, cmap='viridis', s=30, edgecolor='none')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('R²')

    selected = discovery.get('selected', {}) if isinstance(discovery, dict) else {}
    if isinstance(selected, dict):
        try:
            sel_c = float(selected.get('center_nm'))
            sel_m = float(selected.get('slope_nm_per_ppm'))
            ax.scatter([sel_c], [sel_m], marker='*', color='red', s=120, label='Selected ROI')
        except Exception:
            pass

    ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Center Wavelength (nm)')
    ax.set_ylabel('Slope (nm/ppm)')
    ax.set_title('ROI Discovery: Sensitivity vs Wavelength')
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best')
    fig.tight_layout()

    out_path = os.path.join(plots_dir, 'roi_discovery.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def summarize_roi_performance(performance: Dict[str, object]) -> Optional[str]:
    if not performance:
        return None
    sensitivity = performance.get('regression_slope')
    r2 = performance.get('regression_r2')
    lod = performance.get('lod_ppm')
    loq = performance.get('loq_ppm')
    return (
        f"slope={sensitivity:.6f} dT/ppm, R²={r2:.3f}, LOD={lod:.3f} ppm, LOQ={loq:.3f} ppm"
        if all(v is not None for v in (sensitivity, r2, lod, loq)) else None
    )


def summarize_dynamics_metrics(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {}

    df = df.replace([np.inf, -np.inf], np.nan)
    metrics: Dict[str, object] = {
        'per_concentration': {},
    }

    for conc, group in df.groupby('concentration'):
        conc_key = str(float(conc))
        metrics['per_concentration'][conc_key] = {
            'mean_T90': float(group['response_time_T90'].mean(skipna=True)),
            'std_T90': float(group['response_time_T90'].std(skipna=True) or 0.0),
            'mean_T10': float(group['recovery_time_T10'].mean(skipna=True)),
            'std_T10': float(group['recovery_time_T10'].std(skipna=True) or 0.0),
            'count': int(group.shape[0]),
        }

    metrics['overall'] = {
        'mean_T90': float(df['response_time_T90'].mean(skipna=True)),
        'std_T90': float(df['response_time_T90'].std(skipna=True) or 0.0),
        'mean_T10': float(df['recovery_time_T10'].mean(skipna=True)),
        'std_T10': float(df['recovery_time_T10'].std(skipna=True) or 0.0),
        'count': int(df.shape[0]),
    }

    return metrics


def save_dynamics_summary(summary: Dict[str, object], out_root: str) -> str:
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, 'dynamics_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return out_path


def save_dynamics_error(message: str, out_root: str) -> str:
    return save_dynamics_summary({'error': message}, out_root)


def save_concentration_response_metrics(response: Dict[str, object],
                                        repeatability: Dict[str, object],
                                        out_root: str,
                                        name: str = 'concentration_response') -> str:
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, f'{name}.json')
    payload = response.copy()
    payload['roi_repeatability'] = repeatability
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return out_path


def save_concentration_response_plot(response: Dict[str, object],
                                     avg_by_conc: Dict[float, np.ndarray],
                                     out_root: str,
                                     name: str = 'concentration_response',
                                     clamp_to_roi: bool = True) -> Optional[str]:
    if not response:
        return None
    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, f'{name}.png')

    wl = np.array(response['wavelengths'])
    slopes = np.array(response['slopes'])
    roi_start = response['roi_start_wavelength']
    roi_end = response['roi_end_wavelength']
    # Compute best-by-R² and top-5 for visualization
    r2_arr = None
    best_r2_wl = None
    top5_wl = []
    max_abs_slope_wl = None
    try:
        r2_arr = np.array(response.get('r_squared', []), dtype=float)
        if r2_arr.size == wl.size and r2_arr.size > 0:
            best_idx = int(np.nanargmax(r2_arr))
            best_r2_wl = float(wl[best_idx])
            order = np.argsort(r2_arr)[-5:][::-1]
            top5_wl = [float(wl[i]) for i in order]
    except Exception:
        pass
    try:
        abs_sl = np.abs(slopes.astype(float))
        if abs_sl.size == wl.size and abs_sl.size > 0:
            max_abs_slope_wl = float(wl[int(np.nanargmax(abs_sl))])
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for conc, arr in sorted(avg_by_conc.items(), key=lambda kv: kv[0]):
        ax1.plot(wl, arr, label=f"{conc:g} ppm")
    ax1.axvspan(roi_start, roi_end, color='orange', alpha=0.2, label='ROI')
    # Mark best-by-R², max-|slope| and top-5
    if best_r2_wl is not None:
        ax1.axvline(best_r2_wl, color='red', linestyle='--', linewidth=1.0, label='Best R² λ')
    if max_abs_slope_wl is not None:
        ax1.axvline(max_abs_slope_wl, color='blue', linestyle='-.', linewidth=1.0, label='Max |slope| λ')
    if top5_wl:
        for w in top5_wl:
            ax1.axvline(w, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax1.set_ylabel('Transmittance')
    if best_r2_wl is not None and max_abs_slope_wl is not None:
        ax1.set_title(f'Average Transmittance per Concentration (Best R² λ={best_r2_wl:.2f} nm, Max |slope| λ={max_abs_slope_wl:.2f} nm)')
    else:
        ax1.set_title('Average Transmittance per Concentration')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(wl, slopes, label='Slope (ΔT / Δppm)')
    ax2.axvspan(roi_start, roi_end, color='orange', alpha=0.2)
    if best_r2_wl is not None:
        ax2.axvline(best_r2_wl, color='red', linestyle='--', linewidth=1.0, label='Best R² λ')
    if max_abs_slope_wl is not None:
        ax2.axvline(max_abs_slope_wl, color='blue', linestyle='-.', linewidth=1.0, label='Max |slope| λ')
    if top5_wl:
        for w in top5_wl:
            ax2.axvline(w, color='gray', linestyle=':', linewidth=0.6, alpha=0.5)
    ax2.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Slope')
    ax2.set_title('Concentration Response Gradient')
    ax2.grid(True, alpha=0.3)

    if clamp_to_roi:
        roi_cfg = CONFIG.get('roi', {})
        x_min = roi_cfg.get('min_wavelength')
        x_max = roi_cfg.get('max_wavelength')
        if x_min is not None and x_max is not None and float(x_min) < float(x_max):
            ax1.set_xlim(float(x_min), float(x_max))
            ax2.set_xlim(float(x_min), float(x_max))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return out_path


def save_roi_repeatability_plot(stable_by_conc: Dict[float, Dict[str, pd.DataFrame]],
                                response: Dict[str, object],
                                out_root: str) -> Optional[str]:
    start = response['roi_start_wavelength']
    end = response['roi_end_wavelength']
    if start == end:
        return None

    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, 'roi_repeatability.png')

    center = 0.5 * (start + end)
    fig, ax = plt.subplots(figsize=(8, 5))

    xs = []
    ys = []
    labels = []
    for conc, trials in sorted(stable_by_conc.items(), key=lambda kv: kv[0]):
        for trial, df in trials.items():
            col = _signal_column(df)
            wl = df['wavelength'].values
            y = df[col].values
            mask = (wl >= start) & (wl <= end)
            if not mask.any():
                val = float(np.interp(center, wl, y))
            else:
                val = float(np.nanmean(y[mask]))
            xs.append(conc)
            ys.append(val)
            labels.append(trial)

    if not xs:
        plt.close(fig)
        return None

    ax.scatter(xs, ys, alpha=0.7)
    ax.set_xlabel('Concentration (ppm)')
    ax.set_ylabel('Mean Transmittance in ROI')
    ax.set_title(f'ROI Repeatability ({start:.2f}–{end:.2f} nm)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_aggregated_plots(aggregated: Dict[float, Dict[str, pd.DataFrame]], out_root: str) -> Dict[float, Dict[str, str]]:
    plots_root = os.path.join(out_root, 'plots', 'aggregated')
    plot_paths: Dict[float, Dict[str, str]] = {}
    for conc, trials in aggregated.items():
        conc_dir = os.path.join(plots_root, f"{conc:g}")
        _ensure_dir(conc_dir)
        plot_paths[conc] = {}
        for trial, df in trials.items():
            col = _signal_column(df)
            y_label = 'Transmittance' if col == 'transmittance' else 'Intensity'
            safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
            fname = f"{safe_trial or 'trial'}.png"
            fpath = os.path.join(conc_dir, fname)

            plt.figure(figsize=(10, 6))
            plt.plot(df['wavelength'].values, df[col].values, label=trial)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel(y_label)
            plt.title(f"Conc {conc:g} - {trial}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fpath, dpi=300)
            plt.close()

            plot_paths[conc][trial] = fpath
    return plot_paths


def save_canonical_overlay(canonical: Dict[float, pd.DataFrame], out_root: str) -> Optional[str]:
    if not canonical:
        return None
    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, 'canonical_overlay.png')

    plt.figure(figsize=(10, 6))
    for conc, df in sorted(canonical.items(), key=lambda kv: kv[0]):
        col = _signal_column(df)
        plt.plot(df['wavelength'].values, df[col].values, label=f"{conc:g}")
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance' if 'transmittance' in next(iter(canonical.values())).columns else 'Intensity')
    plt.title('Canonical Spectra Overlay')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Concentration')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def save_environment_compensation_summary(info: Dict[str, object], out_root: str) -> str:
    metrics_dir = os.path.join(out_root, 'metrics')
    _ensure_dir(metrics_dir)
    out_path = os.path.join(metrics_dir, 'environment_compensation.json')
    with open(out_path, 'w') as f:
        json.dump(info, f, indent=2)
    return out_path


def write_run_summary(calib: Dict[str, object],
                      aggregated_paths: Dict[float, Dict[str, str]],
                      noise_metrics_path: str,
                      summary_csv_path: str,
                      canonical_plot_path: Optional[str],
                      response_metrics_path: str,
                      response_plot_path: Optional[str],
                      repeatability_plot_path: Optional[str],
                      performance_metrics_path: Optional[str],
                      dynamics_summary_path: Optional[str],
                      dynamics_plot_path: Optional[str],
                      metadata_path: Optional[str],
                      archive_path: Optional[str],
                      qc_summary_path: Optional[str],
                      report_artifacts: Dict[str, object],
                      trend_plots: Dict[str, str],
                      performance: Optional[Dict[str, object]],
                      dynamics_summary: Optional[Dict[str, object]],
                      out_root: str) -> str:
    reports_dir = os.path.join(out_root, 'reports')
    _ensure_dir(reports_dir)
    out_path = os.path.join(reports_dir, 'summary.md')

    lines = [
        '# Gas Analysis Run Summary',
        '',
        '## Calibration Results',
        '',
    ]
    try:
        sel_model = str(calib.get('selected_model', ''))
        if sel_model.endswith('_cv'):
            lines.append(f"- **Selected model**: {sel_model}")
    except Exception:
        pass
    try:
        slope_v = float(calib.get('slope', float('nan')))
        if np.isfinite(slope_v):
            lines.append(f"- **Slope**: {slope_v:.4f} nm/ppm")
    except Exception:
        pass
    try:
        intercept_v = float(calib.get('intercept', float('nan')))
        if np.isfinite(intercept_v):
            lines.append(f"- **Intercept**: {intercept_v:.4f} nm")
    except Exception:
        pass
    try:
        r2_v = float(calib.get('r2', float('nan')))
        if np.isfinite(r2_v):
            lines.append(f"- **R²**: {r2_v:.4f}")
    except Exception:
        pass
    try:
        rmse_v = float(calib.get('rmse', float('nan')))
        if np.isfinite(rmse_v):
            try:
                units = 'ppm' if str(calib.get('selected_model', '')).endswith('_cv') else 'nm'
            except Exception:
                units = 'nm'
            lines.append(f"- **RMSE**: {rmse_v:.4f} {units}")
    except Exception:
        pass
    try:
        lod_v = float(calib.get('lod', float('nan')))
        if np.isfinite(lod_v):
            lines.append(f"- **LOD**: {lod_v:.4f} ppm")
    except Exception:
        pass
    try:
        loq_v = float(calib.get('loq', float('nan')))
        if np.isfinite(loq_v):
            lines.append(f"- **LOQ**: {loq_v:.4f} ppm")
    except Exception:
        pass
    try:
        roi_v = float(calib.get('roi_center', float('nan')))
        if np.isfinite(roi_v):
            lines.append(f"- **ROI Center**: {roi_v:.4f} nm")
    except Exception:
        pass
    lines.append('')

    # Add CV metrics if available
    try:
        unc = calib.get('uncertainty', {}) if isinstance(calib, dict) else {}
        r2_cv = unc.get('r2_cv', None)
        rmse_cv = unc.get('rmse_cv', None)
        if r2_cv is not None and np.isfinite(r2_cv):
            lines.insert(9, f"- **R² (LOOCV)**: {float(r2_cv):.4f}")
        if rmse_cv is not None and np.isfinite(rmse_cv):
            lines.insert(10, f"- **RMSE (LOOCV)**: {float(rmse_cv):.4f} nm")
    except Exception:
        pass

    lines.extend([
        '## Aggregated Spectra',
        '',
        f'- **Noise metrics**: `{noise_metrics_path}`',
        f'- **Aggregated summary CSV**: `{summary_csv_path}`',
        f'- **Concentration response metrics**: `{response_metrics_path}`',
    ])
    # Optional: band-wise regressions CSV and per-trial plots
    try:
        dbg_csv = os.path.join(out_root, 'metrics', 'debug_all_wavelength_regressions.csv')
        if os.path.exists(dbg_csv):
            lines.append(f'- **Band-wise regressions CSV**: `{dbg_csv}`')
    except Exception:
        pass
    try:
        agg_dir = os.path.join(out_root, 'plots', 'aggregated')
        if os.path.isdir(agg_dir):
            lines.append(f'- **Per-trial aggregated plots folder**: `{agg_dir}`')
    except Exception:
        pass
    # ROI selection details and plots
    try:
        with open(response_metrics_path, 'r') as f:
            resp_json = json.load(f)
        lines.extend([
            '',
            '### ROI Selection Details',
        ])
        rsel = resp_json
        # Pre-compute useful arrays and stats for summary/profile
        wl_arr = np.array(rsel.get('wavelengths', []), dtype=float)
        slopes_arr_r = np.array(rsel.get('slopes', []), dtype=float)
        r2_arr_r = np.array(rsel.get('r_squared', []), dtype=float)
        best_r2_wl_val = None
        if wl_arr.size and r2_arr_r.size and wl_arr.size == r2_arr_r.size:
            try:
                best_idx_tmp = int(np.nanargmax(r2_arr_r))
                best_r2_wl_val = float(wl_arr[best_idx_tmp])
            except Exception:
                best_r2_wl_val = None
        max_slope_wl_val = rsel.get('max_slope_wavelength', None)
        roi_start_val = rsel.get('roi_start_wavelength', None)
        roi_end_val = rsel.get('roi_end_wavelength', None)
        # Slope-to-noise from global repeatability if present
        global_std_val = None
        try:
            rep = rsel.get('roi_repeatability', {}) if isinstance(rsel.get('roi_repeatability', {}), dict) else {}
            global_std_val = rep.get('global', {}).get('std_transmittance', None)
        except Exception:
            global_std_val = None
        stn_profile = None
        if (global_std_val is not None) and (float(global_std_val) > 0) and slopes_arr_r.size and (wl_arr.size == slopes_arr_r.size):
            try:
                stn_profile = np.abs(slopes_arr_r) / float(global_std_val)
            except Exception:
                stn_profile = None
        sel_metric = rsel.get('roi_selection_metric')
        if sel_metric:
            lines.append(f"- Selection metric: {sel_metric}")
        try:
            roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
            if roi_cfg:
                r2w = roi_cfg.get('r2_weight', None)
                bhw = roi_cfg.get('band_half_width', None)
                bw = roi_cfg.get('band_window', None)
                mincorr = roi_cfg.get('min_corr', None)
                minslope = roi_cfg.get('min_abs_slope', None)
                adapt = roi_cfg.get('adaptive_band', {}) or {}
                lines.append(f"- r2_weight: {r2w}")
                lines.append(f"- band_half_width: {bhw}")
                lines.append(f"- band_window: {bw}")
                lines.append(f"- min_corr: {mincorr}")
                lines.append(f"- min_abs_slope: {minslope}")
                if adapt:
                    lines.append(f"- adaptive_band: enabled={bool(adapt.get('enabled', False))}, slope_fraction={adapt.get('slope_fraction')}, max_half_width={adapt.get('max_half_width')}")
        except Exception:
            pass
        try:
            rs = rsel
            lines.append(f"- ROI: {rs.get('roi_start_wavelength'):.2f}–{rs.get('roi_end_wavelength'):.2f} nm")
            lines.append(f"- Max R²: {float(rs.get('max_r_squared', float('nan'))):.4f}")
            lines.append(f"- Max slope @ λ: {float(rs.get('max_slope_wavelength', float('nan'))):.2f} nm")
        except Exception:
            pass
        # Top candidates (limit 5)
        try:
            cands = rsel.get('candidates', [])
            if isinstance(cands, list) and cands:
                lines.append('- Top candidates:')
                for c in cands[:5]:
                    try:
                        lines.append(f"  - λ={float(c.get('wavelength')):.2f} nm, R²={float(c.get('r2')):.4f}, score={float(c.get('score')):.4f}")
                    except Exception:
                        continue
        except Exception:
            pass
        # Link plots if exist
        if response_plot_path:
            lines.append(f"- Response plot: `{response_plot_path}`")
            try:
                lines.extend([
                    '',
                    '#### Concentration Response',
                    f'![]({response_plot_path})',
                ])
            except Exception:
                pass
        # Generate slope-to-noise profile plot within ROI if available
        try:
            if stn_profile is not None and wl_arr.size:
                plots_dir = os.path.join(out_root, 'plots')
                _ensure_dir(plots_dir)
                sn_path = os.path.join(plots_dir, 'slope_to_noise_profile.png')
                figsn, axsn = plt.subplots(figsize=(10, 3))
                axsn.plot(wl_arr, stn_profile, color='purple', linewidth=1.2)
                if (roi_start_val is not None) and (roi_end_val is not None):
                    axsn.axvspan(float(roi_start_val), float(roi_end_val), color='orange', alpha=0.2, label='ROI')
                if best_r2_wl_val is not None:
                    axsn.axvline(best_r2_wl_val, color='red', linestyle='--', linewidth=1.0, label='Best R² λ')
                if max_slope_wl_val is not None:
                    axsn.axvline(float(max_slope_wl_val), color='blue', linestyle='-.', linewidth=1.0, label='Max |slope| λ')
                axsn.set_xlabel('Wavelength (nm)')
                axsn.set_ylabel('Slope-to-noise (|slope|/σ)')
                axsn.set_title('Slope-to-Noise Profile (within ROI)')
                axsn.grid(True, alpha=0.3)
                axsn.legend(loc='upper right')
                figsn.tight_layout()
                figsn.savefig(sn_path, dpi=200)
                plt.close(figsn)
                lines.append(f'- Slope-to-noise profile: `{sn_path}`')
                try:
                    relsn = os.path.relpath(sn_path, start=reports_dir)
                    lines.append(f'![Slope-to-noise profile]({relsn})')
                except Exception:
                    pass
        except Exception:
            pass
        try:
            fs_metrics = os.path.join(out_root, 'metrics', 'fullscan_concentration_response.json')
            fs_plot = os.path.join(out_root, 'plots', 'fullscan_concentration_response.png')
            if os.path.exists(fs_metrics):
                lines.append(f"- Full-scan response metrics: `{fs_metrics}`")
            if os.path.exists(fs_plot):
                lines.append(f"- Full-scan response plot: `{fs_plot}`")
                try:
                    lines.extend([
                        '',
                        '#### Full-scan Concentration Response',
                        f'![]({fs_plot})',
                    ])
                except Exception:
                    pass
        except Exception:
            pass
        # Per-Gas Summary table
        try:
            lines.extend(['', '## Per-Gas Summary', ''])
            header = '| ROI | Observed center (nm) | Best R² λ (nm) | Max |slope| λ (nm) | STN@BestR² | STN@Max|slope| | Selected | CV R² | RMSE | LOD | LOQ |'
            sep = '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|'
            lines.extend([header, sep])
            # Observed center from validation
            obs_center = None
            try:
                vj = rsel.get('validation', {}) if isinstance(rsel.get('validation', {}), dict) else {}
                oc = vj.get('observed_center', None)
                if oc is not None:
                    obs_center = float(oc)
            except Exception:
                obs_center = None
            roi_label = 'NA'
            try:
                if (roi_start_val is not None) and (roi_end_val is not None):
                    roi_label = f"{float(roi_start_val):.2f}–{float(roi_end_val):.2f}"
            except Exception:
                pass
            # Compute STN at specific λ via interpolation
            def _interp_stn(x):
                try:
                    if stn_profile is not None and wl_arr.size and stn_profile.size == wl_arr.size and (x is not None):
                        return float(np.interp(float(x), wl_arr, stn_profile))
                except Exception:
                    return float('nan')
                return float('nan')
            stn_best = _interp_stn(best_r2_wl_val)
            stn_maxs = _interp_stn(max_slope_wl_val)
            # Selected model and CV metrics
            sel_name = None
            sel_r2cv = float('nan')
            sel_rmsecv = float('nan')
            try:
                # Prefer explicit selection file if available
                sel_path = os.path.join(out_root, 'metrics', 'multivariate_selection.json')
                if os.path.exists(sel_path):
                    with open(sel_path, 'r') as f:
                        sjson = json.load(f)
                    # If calib carries selected_model, use that key for metrics
                    sm = str(calib.get('selected_model', '')).strip().lower()
                    if not sm:
                        sm = str(sjson.get('best_method', '')).strip().lower()
                    if sm:
                        if sm.startswith('plsr'):
                            sel_name = 'PLSR'
                            key = 'plsr'
                        elif sm.startswith('ica'):
                            sel_name = 'ICA'
                            key = 'ica'
                        elif sm.startswith('mcr'):
                            sel_name = 'MCR-ALS'
                            key = 'mcr_als'
                        else:
                            key = None
                        if key is not None:
                            scores = sjson.get('scores', {}) if isinstance(sjson.get('scores', {}), dict) else {}
                            sc = scores.get(key, {}) if isinstance(scores.get(key, {}), dict) else {}
                            r2c = sc.get('r2_cv', None)
                            rmsec = sc.get('rmse_cv', None)
                            if r2c is not None:
                                sel_r2cv = float(r2c)
                            if rmsec is not None:
                                sel_rmsecv = float(rmsec)
                # Fallback to PLSR model metrics
                if sel_name is None and isinstance(calib.get('plsr_model', None), dict):
                    pm = calib['plsr_model']
                    sel_name = 'PLSR'
                    if pm.get('r2_cv', None) is not None:
                        sel_r2cv = float(pm.get('r2_cv'))
                    if pm.get('rmse_cv', None) is not None:
                        sel_rmsecv = float(pm.get('rmse_cv'))
            except Exception:
                pass
            # LOD/LOQ from calibration
            lod_v = calib.get('lod', None)
            loq_v = calib.get('loq', None)
            try:
                lod_v = float(lod_v) if lod_v is not None else float('nan')
            except Exception:
                lod_v = float('nan')
            try:
                loq_v = float(loq_v) if loq_v is not None else float('nan')
            except Exception:
                loq_v = float('nan')
            # Compose row
            row = (
                f"| {roi_label} | "
                f"{(f'{obs_center:.2f}' if obs_center is not None else 'NA')} | "
                f"{(f'{best_r2_wl_val:.2f}' if best_r2_wl_val is not None else 'NA')} | "
                f"{(f'{float(max_slope_wl_val):.2f}' if max_slope_wl_val is not None else 'NA')} | "
                f"{(f'{stn_best:.3f}' if np.isfinite(stn_best) else 'NA')} | "
                f"{(f'{stn_maxs:.3f}' if np.isfinite(stn_maxs) else 'NA')} | "
                f"{(sel_name if sel_name is not None else 'NA')} | "
                f"{(f'{sel_r2cv:.4f}' if np.isfinite(sel_r2cv) else 'NA')} | "
                f"{(f'{sel_rmsecv:.4f}' if np.isfinite(sel_rmsecv) else 'NA')} | "
                f"{(f'{lod_v:.4f}' if np.isfinite(lod_v) else 'NA')} | "
                f"{(f'{loq_v:.4f}' if np.isfinite(loq_v) else 'NA')} |"
            )
            lines.append(row)
        except Exception:
            pass
    except Exception:
        pass

    # Link QC summary if present
    try:
        qc_path = os.path.join(out_root, 'metrics', 'qc_summary.json')
        if os.path.exists(qc_path):
            lines.append(f'- **Quality control summary**: `{qc_path}`')
    except Exception:
        pass

    # Link environment compensation summary if present
    try:
        env_path = os.path.join(out_root, 'metrics', 'environment_compensation.json')
        if os.path.exists(env_path):
            lines.append(f'- **Environment compensation**: `{env_path}`')
            try:
                with open(env_path, 'r') as f:
                    env_json = json.load(f)
                r2c = env_json.get('r2_conc_only', None)
                r2f = env_json.get('r2_full', None)
                dr2 = env_json.get('delta_r2', None)
                rmsec = env_json.get('rmse_conc_only', None)
                rmsef = env_json.get('rmse_full', None)
                drmse = env_json.get('delta_rmse', None)
                coef = env_json.get('estimated_coefficients', {}) if isinstance(env_json.get('estimated_coefficients', {}), dict) else {}
                ct = coef.get('temperature', None)
                ch = coef.get('humidity', None)
                # Append compact summary
                lines.extend([
                    '',
                    '### Environment Compensation Summary',
                ])
                if any(v is not None for v in [r2c, r2f, dr2, rmsec, rmsef, drmse]):
                    if r2c is not None:
                        lines.append(f"- R² (conc-only): {float(r2c):.4f}")
                    if r2f is not None:
                        lines.append(f"- R² (with env): {float(r2f):.4f}")
                    if dr2 is not None:
                        lines.append(f"- ΔR²: {float(dr2):.4f}")
                    if rmsec is not None:
                        lines.append(f"- RMSE (conc-only): {float(rmsec):.4f} nm")
                    if rmsef is not None:
                        lines.append(f"- RMSE (with env): {float(rmsef):.4f} nm")
                    if drmse is not None:
                        lines.append(f"- ΔRMSE: {float(drmse):.4f} nm")
                if any(v is not None for v in [ct, ch]):
                    lines.append(f"- Estimated cT: {ct if ct is not None else 'n/a'}")
                    lines.append(f"- Estimated cH: {ch if ch is not None else 'n/a'}")
            except Exception:
                pass
    except Exception:
        pass

    if canonical_plot_path:
        lines.append(f'- **Canonical overlay plot**: `{canonical_plot_path}`')
        try:
            rel = os.path.relpath(canonical_plot_path, start=reports_dir)
            lines.append(f'![Canonical overlay]({rel})')
        except Exception:
            pass
    # Link deconvolution artifacts if present
    try:
        ica_metrics = os.path.join(out_root, 'metrics', 'deconvolution_ica.json')
        if os.path.exists(ica_metrics):
            lines.append(f'- **Deconvolution (ICA)**: `{ica_metrics}`')
            comp_plot = os.path.join(out_root, 'plots', 'ica_components.png')
            pv_plot = os.path.join(out_root, 'plots', 'ica_pred_vs_actual.png')
            for title, pth in [('ICA components', comp_plot), ('ICA predicted vs actual', pv_plot)]:
                if os.path.exists(pth):
                    try:
                        rel = os.path.relpath(pth, start=reports_dir)
                        lines.append(f'![{title}]({rel})')
                    except Exception:
                        pass
                    # Optional bootstrap CIs for selected model
                    try:
                        bc = mv_cfg.get('bootstrap_ci', {}) if isinstance(mv_cfg, dict) else {}
                        if bc.get('enabled', False):
                            y_true_b = np.array(calib.get('selected_actual', []), dtype=float)
                            y_pred_b = np.array(calib.get('selected_predictions', []), dtype=float)
                            if y_true_b.size and y_true_b.size == y_pred_b.size:
                                iters = int(bc.get('iterations', 1000))
                                rng = np.random.default_rng(0)
                                n = y_true_b.size
                                r2_s = []
                                rmse_s = []
                                for _ in range(max(1, iters)):
                                    idx = rng.integers(0, n, size=n)
                                    yt = y_true_b[idx]
                                    yp = y_pred_b[idx]
                                    if np.isfinite(np.var(yt)) and np.var(yt) > 0:
                                        r2_s.append(float(r2_score(yt, yp)))
                                    else:
                                        r2_s.append(float('nan'))
                                    rmse_s.append(float(np.sqrt(mean_squared_error(yt, yp))))
                                def _nan_ci(arr, lo=2.5, hi=97.5):
                                    a = np.array(arr, dtype=float)
                                    return float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))
                                r2_lo, r2_hi = _nan_ci(r2_s)
                                rmse_lo, rmse_hi = _nan_ci(rmse_s)
                                unc = calib.get('uncertainty', {}) if isinstance(calib.get('uncertainty', {}), dict) else {}
                                unc['r2_cv_ci'] = [r2_lo, r2_hi]
                                unc['rmse_cv_ci'] = [rmse_lo, rmse_hi]
                                calib['uncertainty'] = unc
                    except Exception:
                        pass
        mcr_metrics = os.path.join(out_root, 'metrics', 'deconvolution_mcr_als.json')
        if os.path.exists(mcr_metrics):
            lines.append(f'- **Deconvolution (MCR-ALS)**: `{mcr_metrics}`')
            comp_plot = os.path.join(out_root, 'plots', 'mcr_als_components.png')
            pv_plot = os.path.join(out_root, 'plots', 'mcr_als_pred_vs_actual.png')
            for title, pth in [('MCR-ALS components', comp_plot), ('MCR-ALS predicted vs actual', pv_plot)]:
                if os.path.exists(pth):
                    try:
                        rel = os.path.relpath(pth, start=reports_dir)
                        lines.append(f'![{title}]({rel})')
                    except Exception:
                        pass
        # Link PLSR predicted vs actual and coefficient plots if present
        try:
            plsr_pv = os.path.join(out_root, 'plots', 'plsr_pred_vs_actual.png')
            plsr_coef = os.path.join(out_root, 'plots', 'plsr_coefficients.png')
            plsr_cv = os.path.join(out_root, 'plots', 'plsr_cv_curves.png')
            plsr_resid = os.path.join(out_root, 'plots', 'plsr_residuals.png')
            if os.path.exists(plsr_pv):
                rel = os.path.relpath(plsr_pv, start=reports_dir)
                lines.append(f'- **PLSR predicted vs actual**: `{plsr_pv}`')
                lines.append(f'![PLSR predicted vs actual]({rel})')
            if os.path.exists(plsr_coef):
                relc = os.path.relpath(plsr_coef, start=reports_dir)
                lines.append(f'- **PLSR coefficients**: `{plsr_coef}`')
                lines.append(f'![PLSR coefficients]({relc})')
            if os.path.exists(plsr_cv):
                relcv = os.path.relpath(plsr_cv, start=reports_dir)
                lines.append(f'- **PLSR CV curves**: `{plsr_cv}`')
                lines.append(f'![PLSR CV curves]({relcv})')
            if os.path.exists(plsr_resid):
                relrd = os.path.relpath(plsr_resid, start=reports_dir)
                lines.append(f'- **PLSR residual diagnostics**: `{plsr_resid}`')
                lines.append(f'![PLSR residual diagnostics]({relrd})')
        except Exception:
            pass
    except Exception:
        pass
    if response_plot_path:
        lines.append(f'- **Concentration response plot**: `{response_plot_path}`')
        try:
            rel = os.path.relpath(response_plot_path, start=reports_dir)
            lines.append(f'![Concentration response]({rel})')
        except Exception:
            pass
    if repeatability_plot_path:
        lines.append(f'- **ROI repeatability plot**: `{repeatability_plot_path}`')
        try:
            rel = os.path.relpath(repeatability_plot_path, start=reports_dir)
            lines.append(f'![ROI repeatability]({rel})')
        except Exception:
            pass
    # Link multivariate selection if present
    try:
        sel_path = os.path.join(out_root, 'metrics', 'multivariate_selection.json')
        if os.path.exists(sel_path):
            lines.append(f'- **Multivariate model selection**: `{sel_path}`')
            try:
                with open(sel_path, 'r') as f:
                    sel_json = json.load(f)
                bm = sel_json.get('best_method')
                br2 = sel_json.get('best_r2_cv')
                lines.extend([
                    '',
                    '### Multivariate Selection',
                ])
                if bm is not None:
                    if br2 is not None and np.isfinite(br2):
                        lines.append(f"- Best by CV R²: {bm} ({float(br2):.4f})")
                    else:
                        lines.append(f"- Best by CV R²: {bm}")
                # Candidate table
                scores = sel_json.get('scores', {}) if isinstance(sel_json.get('scores', {}), dict) else {}
                if scores:
                    lines.extend([
                        '',
                        '| Model | CV R² | RMSE | Selected |',
                        '|---|---:|---:|:---:|',
                    ])
                    def _fmt(v):
                        try:
                            return f"{float(v):.4f}"
                        except Exception:
                            return 'NA'
                    # Determine policy-selected model if available
                    sel_model = None
                    try:
                        if isinstance(calib, dict):
                            sm = str(calib.get('selected_model', '')).strip().lower()
                            if sm:
                                if sm.startswith('plsr'):
                                    sel_model = 'plsr'
                                elif sm.startswith('ica'):
                                    sel_model = 'ica'
                                elif sm.startswith('mcr'):
                                    sel_model = 'mcr_als'
                    except Exception:
                        sel_model = None
                    for key, label in [('plsr', 'PLSR'), ('ica', 'ICA'), ('mcr_als', 'MCR-ALS')]:
                        sc = scores.get(key, {}) if isinstance(scores.get(key, {}), dict) else {}
                        r2v = _fmt(sc.get('r2_cv'))
                        rmsev = _fmt(sc.get('rmse_cv'))
                        selmark = '✔' if (sel_model == key) else ''
                        lines.append(f"| {label} | {r2v} | {rmsev} | {selmark} |")
                    # Explicitly state selection
                    if sel_model:
                        try:
                            lines.append(f"- Selected by policy: {sel_model}")
                        except Exception:
                            pass
                # Selection policy summary (if configured)
                mv_cfg = CONFIG.get('calibration', {}).get('multivariate', {}) if isinstance(CONFIG, dict) else {}
                if mv_cfg:
                    try:
                        lines.append('')
                        lines.append('- Selection policy: '
                                     f"min_r2_cv={mv_cfg.get('min_r2_cv', 'NA')}, "
                                     f"improve_margin={mv_cfg.get('improve_margin', 'NA')}, "
                                     f"prefer_plsr_on_tie={mv_cfg.get('prefer_plsr_on_tie', 'NA')}")
                    except Exception:
                        pass
                # Embed CV R² comparison plot if exists
                try:
                    cv_plot = os.path.join(out_root, 'plots', 'multivariate_cv_r2.png')
                    if os.path.exists(cv_plot):
                        lines.append(f'- CV R² comparison plot: `{cv_plot}`')
                        try:
                            rel = os.path.relpath(cv_plot, start=reports_dir)
                            lines.append(f'![Multivariate CV R² Comparison]({rel})')
                        except Exception:
                            pass
                    # Embed selected model predicted vs actual if exists
                    sel_plot = os.path.join(out_root, 'plots', 'selected_pred_vs_actual.png')
                    if os.path.exists(sel_plot):
                        lines.append(f'- Selected model predicted vs actual: `{sel_plot}`')
                        try:
                            rel2 = os.path.relpath(sel_plot, start=reports_dir)
                            lines.append(f'![Selected model predicted vs actual]({rel2})')
                        except Exception:
                            pass
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    if performance_metrics_path:
        lines.append(f'- **ROI performance metrics**: `{performance_metrics_path}`')
    if dynamics_summary_path:
        lines.append(f'- **Dynamics summary**: `{dynamics_summary_path}`')
    if dynamics_plot_path:
        lines.append(f'- **Dynamics plot**: `{dynamics_plot_path}`')
        try:
            rel = os.path.relpath(dynamics_plot_path, start=reports_dir)
            lines.append(f'![Dynamics]({rel})')
        except Exception:
            pass
    if metadata_path:
        lines.append(f'- **Run metadata**: `{metadata_path}`')
    if archive_path:
        lines.append(f'- **Archive copy**: `{archive_path}`')

    perf_summary = summarize_roi_performance(performance or {})
    if perf_summary:
        lines.extend(['', '### ROI Performance Snapshot', '', f'- {perf_summary}'])

    if report_artifacts:
        lines.extend(['', '### Report Artifacts', ''])
        for key, value in report_artifacts.items():
            lines.append(f'- {key}: `{value}`')

    if trend_plots:
        lines.extend(['', '### Trend Plots', ''])
        for key, value in trend_plots.items():
            lines.append(f'- {key}: `{value}`')

    if dynamics_summary:
        lines.extend(['', '### Dynamics Overview', ''])
        overall = dynamics_summary.get('overall', {})
        if overall:
            lines.append('- Overall: '
                         f"T90={overall.get('mean_T90', float('nan')):.2f}s ± {overall.get('std_T90', 0.0):.2f}s, "
                         f"T10={overall.get('mean_T10', float('nan')):.2f}s ± {overall.get('std_T10', 0.0):.2f}s")
        per_conc = dynamics_summary.get('per_concentration', {})
        for conc_key, stats in sorted(per_conc.items(), key=lambda kv: float(kv[0])):
            lines.append(f"- {conc_key} ppm: "
                         f"T90={stats.get('mean_T90', float('nan')):.2f}s ± {stats.get('std_T90', 0.0):.2f}s, "
                         f"T10={stats.get('mean_T10', float('nan')):.2f}s ± {stats.get('std_T10', 0.0):.2f}s")

    # Recommendations section
    try:
        lines.extend(['', '## Recommendations', ''])
        # ROI recommendation
        roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
        roi_min = roi_cfg.get('min_wavelength', None)
        roi_max = roi_cfg.get('max_wavelength', None)
        if roi_min is not None and roi_max is not None:
            lines.append(f"- Keep analysis within ROI [{float(roi_min):.0f}, {float(roi_max):.0f}] nm; treat outside as contextual only.")
        # Validation recommendation (align expected center)
        try:
            with open(response_metrics_path, 'r') as f:
                rsj = json.load(f)
            v = rsj.get('validation', {}) if isinstance(rsj.get('validation', {}), dict) else {}
            expc = v.get('expected_center', None)
            obsc = v.get('observed_center', None)
            within = v.get('within_tolerance', None)
            tol = v.get('tolerance', None)
            if expc is not None and obsc is not None and within is False:
                lines.append(f"- Align prior: set expected_center≈{float(obsc):.1f} nm (current {float(expc):.1f}±{float(tol) if tol is not None else 'NA'}).")
        except Exception:
            pass
        # Smoothing recommendation
        try:
            sm = CONFIG.get('preprocessing', {}).get('smooth', {}) if isinstance(CONFIG, dict) else {}
            win = sm.get('window', None)
            if win and int(win) > 11:
                lines.append("- Reduce smoothing window (e.g., 11) to avoid shifting band apex visually.")
        except Exception:
            pass
        # Gating recommendation
        try:
            mv_cfg = CONFIG.get('calibration', {}).get('multivariate', {}) if isinstance(CONFIG, dict) else {}
            minr = mv_cfg.get('min_r2_cv', None)
            im = mv_cfg.get('improve_margin', None)
            if minr is not None and im is not None:
                lines.append(f"- Selection gating: min_r2_cv={float(minr):.2f}, improve_margin={float(im):.2f}. Relax slightly if no model is selected, or keep for conservatism.")
        except Exception:
            pass
        # Data quality recommendation
        lines.append("- If CV R² is modest, add more concentration levels/replicates to stabilize LOOCV.")
        # Debug recommendation
        dbg_csv = os.path.join(out_root, 'metrics', 'debug_all_wavelength_regressions.csv')
        if os.path.exists(dbg_csv):
            lines.append(f"- Use Top-5 R² in `{dbg_csv}` to finalize per-gas ROI centers (expect ±5–10 nm stability).")
    except Exception:
        pass

    lines.extend(['', '### Files per Concentration', ''])

    for conc, trials in sorted(aggregated_paths.items(), key=lambda kv: kv[0]):
        lines.append(f'- **{conc:g}**')
        for trial, path in trials.items():
            lines.append(f'  - `{trial}`: `{path}`')
        lines.append('')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


def save_calibration_outputs(calib: Dict[str, object], out_root: str, name_suffix: str = ''):
    metrics_dir = os.path.join(out_root, 'metrics')
    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    plots_dict = calib.setdefault('plots', {})

    cal_label = f"calibration{name_suffix}" if name_suffix else 'calibration'

    cal_path = os.path.join(metrics_dir, f'{cal_label}.csv')
    data = pd.DataFrame({
        'concentration': calib['concentrations'],
        'peak_wavelength': calib['peak_wavelengths'],
    })
    data.to_csv(cal_path, index=False)

    meta = calib.copy()
    meta_path = os.path.join(metrics_dir, f'{cal_label}_metrics.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(calib['concentrations'])
    y = np.array(calib['peak_wavelengths'])
    slope = calib['slope']
    intercept = calib['intercept']

    yerr = None
    resid_summary = calib.get('residual_summary', {})
    if isinstance(resid_summary, dict):
        try:
            y_std = float(resid_summary.get('std', float('nan')))
        except Exception:
            y_std = float('nan')
        if np.isfinite(y_std) and y_std > 0.0:
            yerr = np.full_like(y, y_std, dtype=float)

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt='o', color='tab:blue', ecolor='lightgray', elinewidth=1.0, capsize=3, label='Data ±σ')
    else:
        ax.scatter(x, y, label='Data')
    xx = np.linspace(x.min(), x.max(), 100)
    yy = intercept + slope * xx
    ax.plot(xx, yy, 'r-', label=f"Fit (R^2={calib['r2']:.3f})")

    target_slope = 0.116
    try:
        target_slope = float(target_slope)
    except Exception:
        target_slope = 0.116
    if x.size >= 1 and np.isfinite(target_slope):
        x0 = float(x.min())
        y0 = float(intercept + slope * x0)
        yy_target = y0 + target_slope * (xx - x0)
        ax.plot(xx, yy_target, 'k--', alpha=0.5, label=f'Target slope {target_slope:.3f} nm/ppm')
    ax.set_xlabel('Concentration (ppm)')
    ax.set_ylabel('Peak Wavelength (nm)')
    ax.set_title('Calibration: Wavelength Shift vs Concentration')
    ax.grid(True, alpha=0.3)
    ax.legend()

    text_lines = []
    try:
        text_lines.append(f"slope = {float(slope):.4f} nm/ppm")
    except Exception:
        pass
    try:
        text_lines.append(f"R² = {float(calib.get('r2', float('nan'))):.3f}")
    except Exception:
        pass
    try:
        lod_val = float(calib.get('lod', float('nan')))
        if np.isfinite(lod_val):
            text_lines.append(f"LOD = {lod_val:.2f} ppm")
    except Exception:
        pass
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
                ha='left', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f'{cal_label}.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    plots_dict['calibration'] = plot_path

    # Residual diagnostics plots
    conc_arr = np.array(calib.get('concentrations', []), dtype=float)
    residuals = np.array(calib.get('residuals') or [], dtype=float)
    preds = np.array(calib.get('predictions') or [], dtype=float)
    mask_conc = residuals.size and conc_arr.size == residuals.size
    mask_pred = mask_conc and preds.size == residuals.size
    if mask_conc:
        finite_mask = np.isfinite(conc_arr) & np.isfinite(residuals)
        conc_fin = conc_arr[finite_mask]
        resid_fin = residuals[finite_mask]
        pred_fin = preds[finite_mask] if mask_pred else None
        if resid_fin.size >= 3:
            fig_res, axes = plt.subplots(2, 2, figsize=(9, 6))
            axes[0, 0].scatter(conc_fin, resid_fin, s=25, alpha=0.8, color='tab:green')
            axes[0, 0].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
            axes[0, 0].set_xlabel('Concentration (ppm)')
            axes[0, 0].set_ylabel('Residual (nm)')
            axes[0, 0].set_title('Residuals vs Concentration')
            axes[0, 0].grid(True, alpha=0.3)

            if pred_fin is not None and np.any(np.isfinite(pred_fin)):
                axes[0, 1].scatter(pred_fin, resid_fin, s=25, alpha=0.8, color='tab:blue')
                axes[0, 1].set_xlabel('Predicted Wavelength (nm)')
                axes[0, 1].set_ylabel('Residual (nm)')
                axes[0, 1].set_title('Residuals vs Predicted')
            else:
                axes[0, 1].axis('off')
            axes[0, 1].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].hist(resid_fin, bins=max(8, int(np.sqrt(resid_fin.size))), color='gray', edgecolor='black', alpha=0.85)
            axes[1, 0].set_xlabel('Residual (nm)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Residual Histogram')

            try:
                (osm, osr), (slope_q, intercept_q, r_q) = probplot(resid_fin, dist='norm', fit=True)
                axes[1, 1].scatter(osm, osr, s=20, alpha=0.8)
                axes[1, 1].plot(osm, slope_q * np.array(osm) + intercept_q, 'r--', linewidth=1.0)
                axes[1, 1].set_title(f'Normal Q-Q (R={r_q:.3f})')
                axes[1, 1].set_xlabel('Theoretical quantiles')
                axes[1, 1].set_ylabel('Ordered residuals')
            except Exception:
                axes[1, 1].text(0.5, 0.5, 'Q-Q failed', ha='center', va='center')
                axes[1, 1].axis('off')

            fig_res.tight_layout()
            resid_plot_path = os.path.join(plots_dir, 'calibration_residuals.png')
            fig_res.savefig(resid_plot_path, dpi=200)
            plt.close(fig_res)
            plots_dict['calibration_residuals'] = resid_plot_path

    # Absolute wavelength shift plot
    abs_info = calib.get('absolute_shift') or {}
    abs_conc = np.array(abs_info.get('concentrations') or [], dtype=float)
    abs_delta = np.array(abs_info.get('absolute_delta_wavelengths') or [], dtype=float)
    if abs_conc.size and abs_delta.size == abs_conc.size:
        mask_abs = np.isfinite(abs_conc) & np.isfinite(abs_delta)
        conc_abs = abs_conc[mask_abs]
        delta_abs = abs_delta[mask_abs]
        if conc_abs.size >= 2:
            fit_vals = abs_info.get('predicted_absolute_delta')
            if fit_vals is not None:
                fit_vals = np.array(fit_vals, dtype=float)
                fit_vals = fit_vals[mask_abs] if fit_vals.size == abs_conc.size else None
            if fit_vals is None and np.isfinite(abs_info.get('slope', float('nan'))):
                slope_abs = float(abs_info.get('slope'))
                intercept_abs = float(abs_info.get('intercept', 0.0))
                fit_vals = intercept_abs + slope_abs * conc_abs
            fig_abs, ax_abs = plt.subplots(figsize=(7, 4))
            ax_abs.scatter(conc_abs, delta_abs, color='tab:orange', alpha=0.85, label='|Δλ| observations')
            if fit_vals is not None and np.size(fit_vals) == conc_abs.size:
                order = np.argsort(conc_abs)
                ax_abs.plot(conc_abs[order], np.asarray(fit_vals)[order], color='tab:red', linewidth=1.2, label='Linear fit')
            ax_abs.set_xlabel('Concentration (ppm)')
            ax_abs.set_ylabel('|Δλ| (nm)')
            ax_abs.set_title('Absolute Wavelength Shift vs Concentration')
            ax_abs.grid(True, alpha=0.3)
            ax_abs.legend(loc='upper left')
            fig_abs.tight_layout()
            abs_plot_path = os.path.join(plots_dir, 'absolute_shift_vs_concentration.png')
            fig_abs.savefig(abs_plot_path, dpi=200)
            plt.close(fig_abs)
            plots_dict['absolute_shift'] = abs_plot_path
            try:
                abs_info['plot_path'] = abs_plot_path
            except Exception:
                pass

    # PLSR artifacts
    if isinstance(calib, dict) and isinstance(calib.get('plsr_model', None), dict):
        pm = calib['plsr_model']
        y_true = np.array(pm.get('concentrations', []), dtype=float)
        y_pred_cv = pm.get('predictions_cv', None)
        y_pred_in = pm.get('predictions_in', None)
        y_pred = np.array(y_pred_cv if y_pred_cv is not None else (y_pred_in if y_pred_in is not None else []), dtype=float)
        if y_true.size and y_pred.size and y_true.size == y_pred.size:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y_true, y_pred, s=20)
            minv = float(min(y_true.min(), y_pred.min()))
            maxv = float(max(y_true.max(), y_pred.max()))
            ax2.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
            r2cv = pm.get('r2_cv', float('nan'))
            ax2.set_xlabel('Actual Concentration (ppm)')
            ax2.set_ylabel('Predicted Concentration (ppm)')
            ax2.set_title(f'PLSR CV (R^2={r2cv:.3f})')
            fig2.tight_layout()
            fig2.savefig(os.path.join(plots_dir, 'plsr_pred_vs_actual.png'), dpi=200)
            plt.close(fig2)

        wl_arr = np.array(pm.get('wavelengths', []), dtype=float)
        coef_arr = np.array(pm.get('coef_', []), dtype=float)
        if wl_arr.size and coef_arr.size and wl_arr.size == coef_arr.size:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.plot(wl_arr, coef_arr, linewidth=1)
            ax3.set_xlabel('Wavelength (nm)')
            ax3.set_ylabel('PLSR Coefficient')
            ax3.set_title('PLSR Coefficients')
            fig3.tight_layout()
            fig3.savefig(os.path.join(plots_dir, 'plsr_coefficients.png'), dpi=200)
            plt.close(fig3)

        # PLSR CV curves (R²/RMSE vs components) and residual diagnostics
        try:
            comps = pm.get('n_components_candidates', None)
            r2_curve = pm.get('r2_cv_curve', None)
            rmse_curve = pm.get('rmse_cv_curve', None)
            if isinstance(comps, list) and isinstance(r2_curve, list) and isinstance(rmse_curve, list):
                if len(comps) == len(r2_curve) == len(rmse_curve) and len(comps) > 0:
                    figc, axc1 = plt.subplots(figsize=(7, 4))
                    axc1.plot(comps, r2_curve, marker='o', color='tab:blue', label='R² (CV)')
                    axc1.set_xlabel('PLS components')
                    axc1.set_ylabel('R² (CV)', color='tab:blue')
                    axc1.tick_params(axis='y', labelcolor='tab:blue')
                    axc2 = axc1.twinx()
                    axc2.plot(comps, rmse_curve, marker='s', color='tab:red', label='RMSE (CV)')
                    axc2.set_ylabel('RMSE (CV)', color='tab:red')
                    axc2.tick_params(axis='y', labelcolor='tab:red')
                    figc.tight_layout()
                    figc.savefig(os.path.join(plots_dir, 'plsr_cv_curves.png'), dpi=200)
                    plt.close(figc)
        except Exception:
            pass

        # Residual diagnostics (use CV predictions if available; else in-sample)
        try:
            y_true = np.array(pm.get('concentrations', []), dtype=float)
            y_pred = pm.get('predictions_cv', None)
            if y_pred is None:
                y_pred = pm.get('predictions_in', None)
            y_pred = np.array(y_pred if y_pred is not None else [], dtype=float)
            if y_true.size and y_pred.size and y_true.size == y_pred.size:
                resid = y_true - y_pred
                figd, axs = plt.subplots(2, 2, figsize=(9, 6))
                # Residuals vs concentration
                axs[0, 0].scatter(y_true, resid, s=18, alpha=0.8)
                axs[0, 0].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
                axs[0, 0].set_xlabel('Concentration (ppm)')
                axs[0, 0].set_ylabel('Residuals (ppm)')
                axs[0, 0].set_title('Residuals vs Concentration')
                # Residuals vs predicted
                axs[0, 1].scatter(y_pred, resid, s=18, alpha=0.8)
                axs[0, 1].axhline(0.0, color='k', linestyle='--', linewidth=0.8)
                axs[0, 1].set_xlabel('Predicted (ppm)')
                axs[0, 1].set_ylabel('Residuals (ppm)')
                axs[0, 1].set_title('Residuals vs Predicted')
                # Histogram
                axs[1, 0].hist(resid, bins=max(8, int(np.sqrt(len(resid)))), color='gray', edgecolor='black', alpha=0.8)
                axs[1, 0].set_xlabel('Residual (ppm)')
                axs[1, 0].set_ylabel('Frequency')
                axs[1, 0].set_title('Residual Histogram')
                # Q-Q plot
                try:
                    (osm, osr), (slope, intercept, r) = probplot(resid, dist='norm', fit=True)
                    axs[1, 1].scatter(osm, osr, s=14, alpha=0.8)
                    axs[1, 1].plot(osm, slope * np.array(osm) + intercept, 'r--', linewidth=1)
                    axs[1, 1].set_title(f'Normal Q-Q (R={r:.3f})')
                    axs[1, 1].set_xlabel('Theoretical quantiles')
                    axs[1, 1].set_ylabel('Ordered residuals')
                except Exception:
                    axs[1, 1].text(0.5, 0.5, 'Q-Q plot failed', ha='center', va='center')
                    axs[1, 1].set_axis_off()
                figd.tight_layout()
                figd.savefig(os.path.join(plots_dir, 'plsr_residuals.png'), dpi=200)
                plt.close(figd)
        except Exception:
            pass

    # Selected model predicted vs actual plot (if multivariate auto-selected)
    try:
        y_true = np.array(calib.get('selected_actual', []), dtype=float)
        y_pred = np.array(calib.get('selected_predictions', []), dtype=float)
        if y_true.size and y_pred.size and y_true.size == y_pred.size:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.scatter(y_true, y_pred, s=20)
            minv = float(min(y_true.min(), y_pred.min()))
            maxv = float(max(y_true.max(), y_pred.max()))
            ax4.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
            sel_model = str(calib.get('selected_model', 'selected'))
            sel_mode = str(calib.get('selected_predictions_mode', 'cv'))
            r2cv = float(calib.get('uncertainty', {}).get('r2_cv', float('nan')))
            ax4.set_xlabel('Actual Concentration (ppm)')
            ax4.set_ylabel('Predicted Concentration (ppm)')
            ax4.set_title(f'Selected model ({sel_model}, {sel_mode}, R^2={r2cv:.3f})')
            fig4.tight_layout()
            fig4.savefig(os.path.join(plots_dir, 'selected_pred_vs_actual.png'), dpi=200)
            plt.close(fig4)
            # Save CSV of selected predictions
            try:
                sel_df = pd.DataFrame({'actual': y_true.astype(float), 'predicted': y_pred.astype(float)})
                sel_df.to_csv(os.path.join(metrics_dir, 'selected_predictions.csv'), index=False)
            except Exception:
                pass
    except Exception:
        pass

# High-level run helper (optional for CLI)
# ----------------------

def _resolve_gas_benchmarks(label: Optional[str]) -> Dict[str, float]:
    benchmarks = {
        'Acetone': {'slope_nm_per_ppm': 0.116, 'r2': 0.95, 'lod_ppm': 3.26, 'response_s': 26.0, 'recovery_s': 32.0},
        'Methanol': {'slope_nm_per_ppm': 0.081, 'r2': 0.88},
        'Ethanol': {'r2': 0.27},
        'Isopropanol': {'r2': 0.67},
        'Toluene': {'r2': 0.31},
        'Xylene': {'r2': 0.65},
    }
    return benchmarks.get(str(label), {})


def _apply_response_overrides(base_cfg: Dict[str, object], label: Optional[str]) -> Dict[str, object]:
    cfg = dict(base_cfg) if isinstance(base_cfg, dict) else {}
    overrides = cfg.get('overrides', {}) if isinstance(cfg.get('overrides', {}), dict) else {}
    override = overrides.get(str(label), {}) if label is not None else {}
    # Merge override onto base without mutating original
    merged = {k: v for k, v in cfg.items() if k != 'overrides'}
    if isinstance(override, dict):
        merged.update(override)
    merged['overrides'] = overrides
    merged['_applied_override'] = override
    return merged


def run_full_pipeline(root_dir: str, ref_path: str, out_root: str,
                      diff_threshold: float = 0.01,
                      avg_top_n: Optional[int] = None,
                      scan_full: bool = False,
                      top_k_candidates: int = 5,
                      dataset_label: Optional[str] = None) -> Dict[str, object]:
    """Run: scan → average frames per trial → preprocessing → calibration → persistence."""

    run_timestamp = _timestamp()
    metadata: Dict[str, object] = {
        'run_timestamp': run_timestamp,
        'data_dir': os.path.abspath(root_dir),
        'ref_path': os.path.abspath(ref_path) if ref_path else None,
        'out_root': os.path.abspath(out_root),
        'diff_threshold': diff_threshold,
        'avg_top_n': avg_top_n,
        'scan_full': scan_full,
        'top_k_candidates': top_k_candidates,
        'config_snapshot': CONFIG,
        'dataset_label': dataset_label,
        'preprocessing': CONFIG.get('preprocessing', {}),
        'archiving': CONFIG.get('archiving', {}),
        'reporting': CONFIG.get('reporting', {}),
        'trials': {},
    }

    mapping = scan_experiment_root(root_dir)
    metadata['concentrations_detected'] = [float(c) for c in sorted(mapping.keys())]
    ref_df = load_reference_csv(ref_path) if ref_path else None

    preproc_settings = CONFIG.get('preprocessing', {})
    calib_settings = CONFIG.get('calibration', {})
    use_trans = bool(calib_settings.get('use_transmittance', True))
    outlier_cfg = preproc_settings.get('outlier_rejection', {})
    apply_frames = preproc_settings.get('enabled', False) and preproc_settings.get('apply_to_frames', False)
    apply_trans = preproc_settings.get('enabled', False) and preproc_settings.get('apply_to_transmittance', True)
    dynamics_cfg = CONFIG.get('dynamics', {})
    dynamics_enabled = dynamics_cfg.get('enabled', True)
    baseline_n = int(dynamics_cfg.get('baseline_frames', 20) or 20) if isinstance(dynamics_cfg, dict) else 20
    baseline_n = max(1, baseline_n)
    response_cfg = _apply_response_overrides(
        CONFIG.get('response_series', {}) if isinstance(CONFIG.get('response_series', {}), dict) else {},
        dataset_label,
    )
    response_override_applied = response_cfg.pop('_applied_override', {}) if isinstance(response_cfg, dict) else {}
    stability_cfg = CONFIG.get('stability', {}) if isinstance(CONFIG.get('stability', {}), dict) else {}
    min_block_cfg = int(stability_cfg.get('min_block', 0)) if stability_cfg.get('min_block') else None

    stable_by_conc: Dict[float, Dict[str, pd.DataFrame]] = {}
    stable_raw_by_conc: Dict[float, Dict[str, pd.DataFrame]] = {}
    top_path: Dict[str, Dict[float, Dict[str, pd.DataFrame]]] = {}
    time_series_outputs: Dict[str, Dict[str, Dict[str, object]]] = {}
    responsive_delta_by_conc: Dict[float, Dict[str, Dict[str, object]]] = {}
    responsive_trend_fallback: Dict[float, Dict[str, float]] = {}
    for conc, trials in mapping.items():
        conc_key = float(conc)
        conc_entry: Dict[str, object] = {
            'raw_trial_count': len(trials),
            'retained_trial_count': 0,
        }
        metadata['trials'][str(conc_key)] = conc_entry
        trial_debug = metadata.setdefault('trial_debug', {}).setdefault(str(conc_key), {})

        processed_trials: Dict[str, pd.DataFrame] = {}
        raw_trials: Dict[str, pd.DataFrame] = {}
        averaged_intensity_trials: Dict[str, pd.DataFrame] = {}
        averaged_trans_trials: Dict[str, pd.DataFrame] = {}
        averaged_abs_trials: Dict[str, pd.DataFrame] = {}
        spectral_arrays: List[np.ndarray] = []
        trial_names: List[str] = []
        base_wavelengths: Optional[np.ndarray] = None
        wavelengths_consistent = True

        trial_quality_scores: Dict[str, float] = {}

        for trial, frames in trials.items():
            frames_sorted = _sort_frame_paths(frames)
            dfs = [_read_csv_spectrum(p) for p in frames_sorted]
            dfs = [df for df in dfs if not df.empty]
            info_entry: Dict[str, object] = {
                'frame_count': len(frames_sorted),
                'valid_frame_count': len(dfs),
            }
            trial_debug[trial] = info_entry
            if not dfs:
                info_entry['status'] = 'no_valid_frames'
                continue

            if apply_frames:
                dfs = [_preprocess_dataframe(df, stage='frame') for df in dfs]

            # Per-trial reference scaling and simple responsive frame selection
            responsive_indices: List[int] = []
            trial_ref_df = ref_df
            if ref_df is not None:
                baseline_frames = dfs[:baseline_n]
                trial_ref_df, _ = _scale_reference_to_baseline(ref_df, baseline_frames, percentile=95.0)

            if response_cfg.get('enabled', False):
                simple_series_df, simple_indices, _ = _simple_response_selection(
                    dfs,
                    trial_ref_df,
                    dataset_label=dataset_label,
                    response_cfg=response_cfg,
                )
                responsive_indices = list(simple_indices)
                info_entry['response_series'] = {
                    'responsive_frame_count': len(responsive_indices),
                }

                # Build full Δλ time-series per trial for downstream dynamics and reporting.
                try:
                    response_series_df, _, _ = _compute_response_time_series(
                        dfs,
                        trial_ref_df,
                        dataset_label=dataset_label,
                        response_cfg=response_cfg,
                    )
                except Exception as exc:
                    print(f"[WARNING] Failed to compute response time-series for conc={conc_key}, trial={trial}: {exc}")
                    response_series_df = None

                if response_series_df is not None:
                    try:
                        csv_path, series_plot = _save_response_series(
                            response_series_df,
                            out_root,
                            conc_key,
                            trial,
                            dataset_label,
                        )
                        conc_key_str = str(conc_key)
                        ts_conc = time_series_outputs.setdefault(conc_key_str, {})
                        ts_conc[trial] = {
                            'csv': str(csv_path),
                            'plot': str(series_plot),
                        }
                        info_entry['response_series'].update({
                            'csv': str(csv_path),
                            'plot': str(series_plot),
                        })
                    except Exception as exc:
                        info_entry.setdefault('response_series_error', str(exc))

            if response_cfg.get('restrict_to_responsive', False) and responsive_indices:
                frames_for_stability = [dfs[i] for i in responsive_indices if 0 <= i < len(dfs)]
                if not frames_for_stability:
                    frames_for_stability = dfs
            else:
                frames_for_stability = dfs

            weight_mode = stability_cfg.get('weight_mode', 'uniform') if isinstance(stability_cfg, dict) else 'uniform'
            top_k = stability_cfg.get('top_k', 0) if isinstance(stability_cfg, dict) else 0
            s, e, weights = find_stable_block(
                frames_for_stability,
                diff_threshold=diff_threshold,
                weight_mode=weight_mode,
                top_k=int(top_k) if top_k else None,
                min_block=int(stability_cfg.get('min_block', 0) or 0) if isinstance(stability_cfg, dict) else None,
            )

            # Responsive frame selection is handled by _simple_response_selection;
            # no additional ROI-only tightening is applied here.
            avg_df = average_stable_block(frames_for_stability, s, e, weights=weights)
            avg_df_trans = compute_transmittance(avg_df, trial_ref_df) if trial_ref_df is not None else avg_df.copy(deep=True)
            avg_df_with_abs = _append_absorbance_column(avg_df_trans)
            raw_trials[trial] = avg_df_with_abs.copy(deep=True)

            if apply_trans:
                avg_df_proc = _preprocess_dataframe(avg_df_with_abs, stage='transmittance')
            else:
                avg_df_proc = avg_df_with_abs.copy(deep=True)

            avg_df_proc = _append_absorbance_column(avg_df_proc, inplace=True)

            processed_trials[trial] = avg_df_proc

            # Trial quality scoring for canonical weighting
            try:
                roi_bounds = _resolve_roi_bounds(dataset_label)
            except Exception:
                roi_bounds = (None, None)
            expected_center = None
            try:
                roi_cfg = CONFIG.get('roi', {}) if isinstance(CONFIG, dict) else {}
                per_gas = roi_cfg.get('per_gas_overrides', {}) if isinstance(roi_cfg.get('per_gas_overrides', {}), dict) else roi_cfg.get('per_gas_overrides', {})
                gas_cfg = per_gas.get(dataset_label, {}) if dataset_label is not None and isinstance(per_gas, dict) else {}
                val_cfg = gas_cfg.get('validation', {}) if isinstance(gas_cfg.get('validation', {}), dict) else gas_cfg.get('validation', {})
                expected_center = val_cfg.get('expected_center', None)
            except Exception:
                expected_center = None
            q_score, _ = _score_trial_quality(avg_df_proc, roi_bounds=roi_bounds, expected_center=expected_center)
            trial_quality_scores[trial] = float(q_score)
            info_entry['quality_score'] = float(q_score)

            if avg_top_n:
                top_avg_int = average_top_frames(frames_for_stability, top_k=avg_top_n)
                averaged_intensity_trials[trial] = top_avg_int
                if trial_ref_df is not None:
                    top_avg_trans = compute_transmittance(top_avg_int, trial_ref_df)
                else:
                    top_avg_trans = top_avg_int.copy()
                averaged_trans_trials[trial] = top_avg_trans
                averaged_abs_trials[trial] = _append_absorbance_column(top_avg_trans)

            if outlier_cfg.get('enabled', False):
                col = _signal_column(avg_df)
                arr = avg_df[col].to_numpy()
                wl = avg_df['wavelength'].to_numpy()
                if base_wavelengths is None:
                    base_wavelengths = wl
                elif len(wl) != len(base_wavelengths) or not np.allclose(wl, base_wavelengths):
                    wavelengths_consistent = False
                spectral_arrays.append(arr)
                trial_names.append(trial)

        trial_debug['__processed__'] = sorted(processed_trials.keys())
        trial_debug['__raw__'] = sorted(raw_trials.keys())
        conc_entry['processed_trial_count'] = len(processed_trials)
        conc_entry['raw_trial_count_post'] = len(raw_trials)

        flagged_trials: set = set()
        if (outlier_cfg.get('enabled', False)
                and wavelengths_consistent
                and len(spectral_arrays) >= 2):
            threshold = outlier_cfg.get('threshold', 3.0)
            flags = detect_outliers(spectral_arrays, threshold=threshold)
            for trial_name, flag in zip(trial_names, flags):
                if flag:
                    flagged_trials.add(trial_name)
                    _record_outlier(metadata, conc_key, trial_name)
            if flags and all(bool(f) for f in flags):
                flagged_trials.clear()
                conc_entry['outlier_relaxed'] = True

        final_trials = {trial: df for trial, df in processed_trials.items() if trial not in flagged_trials}
        final_raw_trials = {trial: raw_trials[trial] for trial in processed_trials if trial not in flagged_trials and trial in raw_trials}
        final_intensity_top = {trial: averaged_intensity_trials[trial]
                               for trial in processed_trials
                               if trial not in flagged_trials and trial in averaged_intensity_trials}
        final_trans_top = {trial: averaged_trans_trials[trial]
                           for trial in processed_trials
                           if trial not in flagged_trials and trial in averaged_trans_trials}
        final_abs_top = {trial: averaged_abs_trials[trial]
                         for trial in processed_trials
                         if trial not in flagged_trials and trial in averaged_abs_trials}

        if not final_trials and processed_trials:
            final_trials = dict(processed_trials)
            final_raw_trials = {trial: raw_trials.get(trial, df) for trial, df in processed_trials.items()}
            conc_entry['filter_relaxed_all'] = True
        if not final_trials and raw_trials:
            final_trials = dict(raw_trials)
            final_raw_trials = dict(raw_trials)
            conc_entry['raw_fallback'] = True

        if final_trials:
            chosen_source = 'retained'
        elif processed_trials:
            chosen_source = 'restored_from_processed'
            final_trials = dict(processed_trials)
            final_raw_trials = {trial: raw_trials.get(trial, df) for trial, df in processed_trials.items()}
        elif raw_trials:
            chosen_source = 'restored_from_raw'
            final_trials = dict(raw_trials)
            final_raw_trials = dict(raw_trials)
        else:
            chosen_source = 'dropped'
            final_trials = {}
            final_raw_trials = {}

        conc_entry['retained_trial_count'] = len(final_trials)
        conc_entry['restoration_status'] = chosen_source

        # Build quality weights for canonical selection (per concentration)
        final_quality_scores: Dict[str, float] = {}
        for trial_name in final_trials.keys():
            w = trial_quality_scores.get(trial_name, 1.0)
            try:
                w = float(w)
            except Exception:
                w = 1.0
            if not np.isfinite(w) or w <= 0.0:
                w = 1.0
            final_quality_scores[trial_name] = w
        metadata.setdefault('trial_quality', {})[str(conc_key)] = final_quality_scores

        if final_trials:
            stable_by_conc[conc] = final_trials
            stable_raw_by_conc[conc] = final_raw_trials
            trial_debug['__final__'] = sorted(final_trials.keys())
            trial_debug['__final_raw__'] = sorted(final_raw_trials.keys())
            trial_debug['__final_intensity__'] = sorted(final_intensity_top.keys()) if final_intensity_top else []
            trial_debug['__final_trans__'] = sorted(final_trans_top.keys()) if final_trans_top else []
            trial_debug['__final_absorbance__'] = sorted(final_abs_top.keys()) if final_abs_top else []
            trial_debug['__status__'] = chosen_source
            if avg_top_n:
                if final_intensity_top:
                    top_path.setdefault('intensity', {})[conc] = final_intensity_top
                if final_trans_top:
                    top_path.setdefault('transmittance', {})[conc] = final_trans_top
                if final_abs_top:
                    top_path.setdefault('absorbance', {})[conc] = final_abs_top
        else:
            trial_debug['__final__'] = []
            trial_debug['__final_raw__'] = []
            trial_debug['__final_intensity__'] = []
            trial_debug['__final_trans__'] = []
            trial_debug['__final_absorbance__'] = []
            trial_debug['__status__'] = 'dropped'

    # Expose per-trial quality weights for canonical aggregation
    try:
        trial_quality_global: Dict[float, Dict[str, float]] = {}
        for conc_str, per_trial in metadata.get('trial_quality', {}).items():
            try:
                conc_val = float(conc_str)
            except Exception:
                continue
            if isinstance(per_trial, dict):
                q_map: Dict[str, float] = {}
                for t_name, w in per_trial.items():
                    try:
                        w_val = float(w)
                    except Exception:
                        w_val = 1.0
                    if not np.isfinite(w_val) or w_val <= 0.0:
                        w_val = 1.0
                    q_map[str(t_name)] = w_val
                trial_quality_global[conc_val] = q_map
        globals()['TRIAL_WEIGHTS_FOR_CANONICAL'] = trial_quality_global
    except Exception:
        globals()['TRIAL_WEIGHTS_FOR_CANONICAL'] = {}

    if not stable_by_conc:
        raise RuntimeError("No stable blocks found across trials")

    raw_to_save = stable_raw_by_conc if stable_raw_by_conc else stable_by_conc
    aggregated_paths = save_aggregated_spectra(raw_to_save, out_root)
    metadata['stable_concentrations'] = sorted(float(k) for k in stable_by_conc.keys())
    metadata['stable_counts'] = {str(float(k)): len(v) for k, v in stable_by_conc.items()}
    metadata['aggregated_paths'] = aggregated_paths

    signal_views = _build_signal_views(stable_by_conc, stable_raw_by_conc)
    if not signal_views:
        raise RuntimeError("No signal representations available after preprocessing")

    delta_per_trial_df, delta_per_conc_df, responsive_delta_summary = _aggregate_responsive_delta_maps(responsive_delta_by_conc)
    metadata['responsive_delta'] = {
        'per_trial': delta_per_trial_df.to_dict(orient='records') if not delta_per_trial_df.empty else [],
        'per_concentration': delta_per_conc_df.to_dict(orient='records') if not delta_per_conc_df.empty else [],
        'summary_by_concentration': responsive_delta_summary,
    }

    multivariate_cfg = calib_settings.get('multivariate', {}) if isinstance(calib_settings, dict) else {}
    multivariate_enabled = bool(multivariate_cfg.get('enabled', False))

    signal_results: Dict[str, Dict[str, object]] = {}
    for signal_name, view_map in signal_views.items():
        if not view_map:
            continue
        try:
            signal_root = Path(out_root) / 'signals' / signal_name
            signal_root.mkdir(parents=True, exist_ok=True)
            canonical_sig = select_canonical_per_concentration(view_map)
            canonical_paths_sig = save_canonical_spectra(canonical_sig, str(signal_root))
            canonical_plot_sig = save_canonical_overlay(canonical_sig, str(signal_root))

            matrix_cache_sig: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
            if multivariate_enabled and canonical_sig:
                try:
                    matrix_cache_sig = _build_feature_matrix_from_canonical(canonical_sig)
                except ValueError:
                    matrix_cache_sig = None

            # Get per-gas ROI bounds for proper calibration
            min_wl_roi_sig, max_wl_roi_sig = _resolve_roi_bounds(dataset_label)
            response_stats_sig, avg_by_conc_sig = compute_concentration_response(
                view_map,
                override_min_wavelength=min_wl_roi_sig,
                override_max_wavelength=max_wl_roi_sig,
                top_k_candidates=top_k_candidates,
                debug_out_root=str(signal_root),
            )
            repeatability_sig = compute_roi_repeatability(view_map, response_stats_sig)
            performance_sig = compute_roi_performance(repeatability_sig)

            baseline_std = float('nan')
            try:
                baseline_std = float(repeatability_sig.get('global', {}).get('std_transmittance', float('nan')))
            except Exception:
                baseline_std = float('nan')
            baseline_slope = performance_sig.get('regression_slope') if isinstance(performance_sig, dict) else None
            baseline_slope_to_noise = float('nan')
            if isinstance(baseline_slope, (int, float)) and np.isfinite(baseline_slope) and np.isfinite(baseline_std) and baseline_std not in (0.0,):
                baseline_slope_to_noise = abs(float(baseline_slope)) / float(baseline_std)

            mv_result_sig: Optional[Dict[str, object]] = None
            if multivariate_enabled:
                try:
                    mv_source = canonical_sig if canonical_sig else {}
                    mv_result = _fit_plsr_calibration(mv_source, multivariate_cfg, matrix_cache_sig) if mv_source else None
                    if mv_result is not None and isinstance(mv_result, dict):
                        try:
                            y_true = matrix_cache_sig[1] if matrix_cache_sig else None
                            baseline_den = baseline_std if np.isfinite(baseline_std) and baseline_std not in (0.0,) else float('nan')
                            preds_cv = mv_result.get('predictions_cv')
                            preds_in = mv_result.get('predictions_in')
                            preds_arr = None
                            if isinstance(preds_cv, list):
                                preds_arr = np.array(preds_cv, dtype=float)
                            elif isinstance(preds_in, list):
                                preds_arr = np.array(preds_in, dtype=float)
                            if preds_arr is not None and preds_arr.size and y_true is not None and np.isfinite(baseline_den) and baseline_den not in (0.0,):
                                slope_cov = np.polyfit(np.asarray(y_true, dtype=float), preds_arr, 1)[0]
                                mv_result['slope_to_noise'] = abs(float(slope_cov)) / baseline_den
                            mv_result['baseline_slope_to_noise'] = baseline_slope_to_noise
                        except Exception:
                            mv_result.setdefault('slope_to_noise', float('nan'))
                        mv_result_sig = mv_result
                except Exception as exc_mv:  # noqa: BLE001
                    mv_result_sig = {'error': str(exc_mv)}

            metrics_path_sig = save_concentration_response_metrics(
                response_stats_sig,
                repeatability_sig,
                str(signal_root),
                name=f'concentration_response_{signal_name}',
            )
            plot_path_sig = save_concentration_response_plot(
                response_stats_sig,
                avg_by_conc_sig,
                str(signal_root),
                name=f'concentration_response_{signal_name}',
                clamp_to_roi=True,
            )
            repeatability_plot_sig = save_roi_repeatability_plot(view_map, response_stats_sig, str(signal_root))
            performance_metrics_sig = save_roi_performance_metrics(performance_sig, str(signal_root))

            signal_results[signal_name] = {
                'canonical': canonical_sig,
                'canonical_paths': canonical_paths_sig,
                'canonical_plot': canonical_plot_sig,
                'response': response_stats_sig,
                'repeatability': repeatability_sig,
                'performance': performance_sig,
                'multivariate': mv_result_sig,
                'baseline': {
                    'std_transmittance': baseline_std,
                    'slope_to_noise': baseline_slope_to_noise,
                    'regression_r2': performance_sig.get('regression_r2') if isinstance(performance_sig, dict) else float('nan'),
                    'lod_ppm': performance_sig.get('lod_ppm') if isinstance(performance_sig, dict) else float('nan'),
                },
                'metrics_path': metrics_path_sig,
                'plot_path': plot_path_sig,
                'repeatability_plot': repeatability_plot_sig,
                'performance_metrics_path': performance_metrics_sig,
            }
        except Exception as exc:  # noqa: BLE001
            signal_results[signal_name] = {
                'error': str(exc),
            }

    def _compute_signal_score(entry: Dict[str, object]) -> Tuple[float, Dict[str, float]]:
        components: Dict[str, float] = {}
        perf = entry.get('performance') if isinstance(entry, dict) else None
        if not isinstance(perf, dict):
            components['r2'] = float('nan')
            components['lod_ppm'] = float('nan')
            components['sensitivity'] = float('nan')
        else:
            r2 = perf.get('regression_r2')
            lod = perf.get('lod_ppm')
            slope = perf.get('regression_slope')
            components['r2'] = float(r2) if isinstance(r2, (int, float)) and np.isfinite(r2) else float('nan')
            components['lod_ppm'] = float(lod) if isinstance(lod, (int, float)) and np.isfinite(lod) else float('nan')
            components['sensitivity'] = float(slope) if isinstance(slope, (int, float)) and np.isfinite(slope) else float('nan')

        baseline_info = entry.get('baseline') if isinstance(entry, dict) else {}
        baseline_slope_to_noise = float('nan')
        baseline_r2 = float('nan')
        if isinstance(baseline_info, dict):
            baseline_slope_to_noise = float(baseline_info.get('slope_to_noise', float('nan')))
            baseline_r2 = float(baseline_info.get('regression_r2', float('nan')))

        score = 0.0
        if np.isfinite(components.get('r2', float('nan'))):
            score += components['r2']
        if np.isfinite(components.get('lod_ppm', float('nan'))) and components['lod_ppm'] > 0:
            score += 1.0 / components['lod_ppm']
        if np.isfinite(components.get('sensitivity', float('nan'))):
            score += abs(components['sensitivity'])

        mv_entry = entry.get('multivariate') if isinstance(entry, dict) else None
        mv_r2 = float('nan')
        mv_rmse = float('nan')
        mv_slope_to_noise = float('nan')
        if isinstance(mv_entry, dict) and 'error' not in mv_entry:
            mv_r2 = mv_entry.get('r2_cv', float('nan'))
            mv_rmse = mv_entry.get('rmse_cv', float('nan'))
            mv_slope_to_noise = mv_entry.get('slope_to_noise', float('nan'))
        components['plsr_r2_cv'] = float(mv_r2) if isinstance(mv_r2, (int, float)) and np.isfinite(mv_r2) else float('nan')
        components['plsr_rmse_cv'] = float(mv_rmse) if isinstance(mv_rmse, (int, float)) and np.isfinite(mv_rmse) else float('nan')
        components['plsr_slope_to_noise'] = float(mv_slope_to_noise) if isinstance(mv_slope_to_noise, (int, float)) and np.isfinite(mv_slope_to_noise) else float('nan')
        components['baseline_r2'] = baseline_r2
        components['baseline_slope_to_noise'] = baseline_slope_to_noise
        if np.isfinite(components['plsr_r2_cv']):
            score += 2.0 * components['plsr_r2_cv']
        if np.isfinite(components['plsr_rmse_cv']) and components['plsr_rmse_cv'] > 0:
            score += 1.0 / components['plsr_rmse_cv']
        if np.isfinite(components['plsr_slope_to_noise']):
            score += components['plsr_slope_to_noise']

        return score, components

    for sig_name, result in signal_results.items():
        score, components = _compute_signal_score(result)
        result['score'] = score
        result['score_components'] = components

    ranked_signals = sorted(
        signal_results.items(),
        key=lambda kv: kv[1].get('score', float('-inf')),
        reverse=True,
    )

    primary_signal = _resolve_primary_signal(signal_views)
    if ranked_signals:
        best_signal = ranked_signals[0][0]
        if best_signal in signal_views:
            primary_signal = best_signal

    primary_map = signal_views[primary_signal]
    metadata['signal_analysis'] = {
        'available_signals': sorted(signal_views.keys()),
        'primary_signal': primary_signal,
        'ranked_signals': [name for name, _ in ranked_signals],
        'per_signal_results': {
            name: {
                'performance': result.get('performance'),
                'multivariate': result.get('multivariate'),
                'metrics_path': result.get('metrics_path'),
                'plot_path': result.get('plot_path'),
                'repeatability_plot': result.get('repeatability_plot'),
                'performance_metrics_path': result.get('performance_metrics_path'),
                'error': result.get('error'),
                'canonical_plot': result.get('canonical_plot'),
                'score': result.get('score'),
                'score_components': result.get('score_components'),
            }
            for name, result in signal_results.items()
        },
    }

    top_results: Dict[str, Dict[str, object]] = {}
    if avg_top_n and top_path:
        compare_dir = Path(out_root) / 'top_avg_comparison'
        compare_dir.mkdir(parents=True, exist_ok=True)
        for metric_type, data_map in top_path.items():
            if not data_map:
                continue
            subset_dir = compare_dir / metric_type
            save_aggregated_spectra(data_map, str(subset_dir))
            canonical_subset = select_canonical_per_concentration(data_map)
            # Get per-gas ROI bounds for proper calibration
            min_wl_roi_subset, max_wl_roi_subset = _resolve_roi_bounds(dataset_label)
            response_stats_subset, avg_by_conc_subset = compute_concentration_response(
                data_map,
                override_min_wavelength=min_wl_roi_subset,
                override_max_wavelength=max_wl_roi_subset,
                top_k_candidates=top_k_candidates,
                debug_out_root=str(subset_dir),
            )
            repeatability_subset = compute_roi_repeatability(data_map, response_stats_subset)
            performance_subset = compute_roi_performance(repeatability_subset)
            metrics_path_subset = save_concentration_response_metrics(response_stats_subset, repeatability_subset, str(subset_dir), name=f"concentration_response_{metric_type}")
            plot_path_subset = save_concentration_response_plot(response_stats_subset, avg_by_conc_subset, str(subset_dir), name=f"concentration_response_{metric_type}", clamp_to_roi=True)
            full_metrics_path_subset = None
            full_plot_path_subset = None
            if scan_full:
                full_min = float(CONFIG.get('roi', {}).get('min_wavelength', 500.0))
                full_max = float(CONFIG.get('roi', {}).get('max_wavelength', 900.0))
                fs_stats_subset, fs_avg_subset = compute_concentration_response(
                    data_map,
                    override_min_wavelength=full_min,
                    override_max_wavelength=full_max,
                    top_k_candidates=top_k_candidates,
                    debug_out_root=str(subset_dir),
                )
                full_metrics_path_subset = save_concentration_response_metrics(fs_stats_subset, repeatability_subset, str(subset_dir), name=f"fullscan_concentration_response_{metric_type}")
                full_plot_path_subset = save_concentration_response_plot(fs_stats_subset, fs_avg_subset, str(subset_dir), name=f"fullscan_concentration_response_{metric_type}", clamp_to_roi=False)
            top_results[metric_type] = {
                'canonical_count': len(canonical_subset),
                'response_stats': response_stats_subset,
                'repeatability': repeatability_subset,
                'performance': performance_subset,
                'metrics_path': metrics_path_subset,
                'plot_path': plot_path_subset,
                'fullscan_metrics_path': full_metrics_path_subset,
                'fullscan_plot_path': full_plot_path_subset,
                'canonical_raw': canonical_subset,
                'canonical': _baseline_correct_canonical(canonical_subset),
            }

    canonical_raw = select_canonical_per_concentration(stable_by_conc)
    canonical = _baseline_correct_canonical(canonical_raw)
    save_canonical_spectra(canonical, out_root)

    discovered_roi = _discover_roi_in_band(canonical, dataset_label=dataset_label, out_root=out_root)
    if discovered_roi:
        metadata['discovered_roi'] = discovered_roi
        try:
            roi_plot_path = save_roi_discovery_plot(discovered_roi, out_root)
            if roi_plot_path:
                metadata['discovered_roi_plot'] = roi_plot_path
        except Exception:
            pass

    # Optional QC-filtered calibration using existing trial quality weights
    qc_calib: Optional[Dict[str, object]] = None
    try:
        qc_cfg = calib_settings.get('qc', {}) if isinstance(calib_settings, dict) else {}
        min_q = float(qc_cfg.get('min_quality', 0.0))
        min_trials = int(qc_cfg.get('min_trials', 1))
        if min_trials < 1:
            min_trials = 1
        trial_weights_global = globals().get('TRIAL_WEIGHTS_FOR_CANONICAL', {})
        stable_qc: Dict[float, Dict[str, pd.DataFrame]] = {}
        if isinstance(trial_weights_global, dict):
            for conc_val, trials in stable_by_conc.items():
                weights_map = trial_weights_global.get(conc_val, {}) if isinstance(trial_weights_global.get(conc_val, {}), dict) else {}
                filtered: Dict[str, pd.DataFrame] = {}
                for t_name, df in trials.items():
                    w = weights_map.get(t_name, 1.0)
                    try:
                        w_val = float(w)
                    except Exception:
                        w_val = 1.0
                    if not np.isfinite(w_val):
                        w_val = 1.0
                    if w_val >= min_q:
                        filtered[t_name] = df
                if len(filtered) >= min_trials:
                    stable_qc[conc_val] = filtered
        if stable_qc:
            canonical_qc_raw = select_canonical_per_concentration(stable_qc)
            canonical_qc = _baseline_correct_canonical(canonical_qc_raw)
            qc_calib = find_roi_and_calibration(
                canonical_qc,
                dataset_label=dataset_label,
                responsive_delta=responsive_calib,
                discovered_roi=discovered_roi,
            )
            if isinstance(qc_calib, dict):
                qc_calib.setdefault('variant', 'qc_filtered')
                save_calibration_outputs(qc_calib, out_root, name_suffix='_qc')
                metadata['calibration_qc'] = qc_calib
    except Exception:
        qc_calib = None

    noise_metrics = compute_noise_metrics_map(primary_map)
    noise_metrics_path = save_noise_metrics(noise_metrics, out_root)
    summary_csv_path = save_aggregated_summary(primary_map, noise_metrics, out_root)
    # Quality control summary
    try:
        qc_summary = summarize_quality_control(primary_map, noise_metrics)
        qc_summary_path = save_quality_summary(qc_summary, out_root)
    except Exception:
        qc_summary_path = None
    aggregated_plot_paths = save_aggregated_plots(primary_map, out_root)
    canonical_plot_path = save_canonical_overlay(canonical, out_root)

    # Optional deconvolution (ICA and MCR-ALS)
    ica_artifacts: Dict[str, str] = {}
    mcr_artifacts: Dict[str, str] = {}
    ica_result: Optional[Dict[str, object]] = None
    mcr_result: Optional[Dict[str, object]] = None
    try:
        ica_cfg = CONFIG.get('advanced', {}).get('deconvolution', {}).get('ica', {})
        if ica_cfg and bool(ica_cfg.get('enabled', False)):
            ica_res = fit_ica_from_canonical(canonical, ica_cfg)
            if ica_res:
                ica_result = ica_res
                ica_artifacts = save_ica_outputs(ica_res, out_root)
    except Exception:
        ica_artifacts = {}
    try:
        mcr_cfg = CONFIG.get('advanced', {}).get('deconvolution', {}).get('mcr_als', {})
        if mcr_cfg and bool(mcr_cfg.get('enabled', False)):
            mcr_res = fit_mcrals_from_canonical(canonical, mcr_cfg)
            if mcr_res:
                mcr_result = mcr_res
                mcr_artifacts = save_mcrals_outputs(mcr_res, out_root)
    except Exception:
        mcr_artifacts = {}

    # Get per-gas ROI bounds for proper calibration
    min_wl_roi, max_wl_roi = _resolve_roi_bounds(dataset_label)
    responsive_calib = None
    if responsive_delta_summary:
        try:
            responsive_calib = perform_responsive_delta_calibration(
                responsive_delta_summary,
                out_root,
                dataset_label=dataset_label,
                trend_fallbacks=responsive_trend_fallback,
            )
            metadata['responsive_delta_calibration'] = responsive_calib
        except Exception as exc:  # noqa: BLE001
            print(f"[WARNING] Responsive Δλ calibration failed: {exc}")
            responsive_calib = None

    response_stats, avg_by_conc = compute_concentration_response(
        stable_by_conc,
        override_min_wavelength=min_wl_roi,
        override_max_wavelength=max_wl_roi,
        top_k_candidates=top_k_candidates,
        debug_out_root=out_root,
    )
    repeatability = compute_roi_repeatability(stable_by_conc, response_stats)
    performance = compute_roi_performance(repeatability)
    response_metrics_path = save_concentration_response_metrics(response_stats, repeatability, out_root, name='concentration_response')
    response_plot_path = save_concentration_response_plot(response_stats, avg_by_conc, out_root, name='concentration_response', clamp_to_roi=True)
    repeatability_plot_path = save_roi_repeatability_plot(stable_by_conc, response_stats, out_root)
    performance_metrics_path = save_roi_performance_metrics(performance, out_root)

    env_summary_path = None
    env_info: Dict[str, object] = {}
    try:
        env_info = compute_environment_summary(stable_by_conc)
        if env_info and (env_info.get('enabled') or env_info.get('offset_count', 0) > 0 or env_info.get('temperature_mean') is not None or env_info.get('humidity_mean') is not None):
            env_summary_path = save_environment_compensation_summary(env_info, out_root)
    except Exception:
        env_summary_path = None

    fullscan_metrics_path = None
    fullscan_plot_path = None
    if scan_full:
        full_min = float(CONFIG.get('roi', {}).get('min_wavelength', 500.0))
        full_max = float(CONFIG.get('roi', {}).get('max_wavelength', 900.0))
        fs_stats, fs_avg = compute_concentration_response(
            stable_by_conc,
            override_min_wavelength=full_min,
            override_max_wavelength=full_max,
            top_k_candidates=top_k_candidates,
            debug_out_root=out_root,
        )
        fullscan_metrics_path = save_concentration_response_metrics(fs_stats, repeatability, out_root, name='fullscan_concentration_response')
        fullscan_plot_path = save_concentration_response_plot(fs_stats, fs_avg, out_root, name='fullscan_concentration_response', clamp_to_roi=False)

    dynamics_summary_path: Optional[str] = None
    dynamics_plot_path: Optional[str] = None
    dynamics_summary: Dict[str, object] = {}
    if dynamics_enabled:
        try:
            from .dynamics import compute_response_recovery_times  # local import to avoid circular dependency

            dynamics_result = compute_response_recovery_times(root_dir, out_root, signal_column='intensity')
            df_dyn = dynamics_result['results'] if isinstance(dynamics_result['results'], pd.DataFrame) else pd.DataFrame(dynamics_result['results'])
            dynamics_summary = summarize_dynamics_metrics(df_dyn)
            dynamics_summary_path = save_dynamics_summary(dynamics_summary, out_root)
            dynamics_plot_path = dynamics_result.get('plot_path')
        except Exception as exc:
            dynamics_summary_path = save_dynamics_error(str(exc), out_root)

    calib = find_roi_and_calibration(
        canonical,
        dataset_label=dataset_label,
        responsive_delta=responsive_calib,
        discovered_roi=discovered_roi,
    )
    if responsive_calib:
        metadata.setdefault('responsive_delta', {})['calibration'] = responsive_calib

    # If configured, auto-select best multivariate model (PLSR/ICA/MCR) by CV R² with gating and override selection
    try:
        mv_cfg = CONFIG.get('calibration', {}).get('multivariate', {}) if isinstance(CONFIG, dict) else {}
        sel_mode = str(mv_cfg.get('select_mode', 'report_only')).lower()
        if mv_cfg.get('enabled', False) and sel_mode == 'auto':
            # Hyperparameters for robust selection
            min_r2_cv = float(mv_cfg.get('min_r2_cv', 0.0))
            improve_margin = float(mv_cfg.get('improve_margin', 0.02))
            prefer_plsr_on_tie = bool(mv_cfg.get('prefer_plsr_on_tie', True))

            # Gather candidates
            candidates = []  # list of (name, r2_cv, rmse_cv)
            plsr_r2 = float('-inf')
            if isinstance(calib, dict) and isinstance(calib.get('plsr_model', None), dict):
                pm = calib['plsr_model']
                r2cv = pm.get('r2_cv', None)
                rmsecv = pm.get('rmse_cv', None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(('plsr_cv', float(r2cv), float(rmsecv) if rmsecv is not None else float('nan')))
                    plsr_r2 = float(r2cv)
            if isinstance(ica_result, dict):
                r2cv = ica_result.get('r2_cv', None)
                rmsecv = ica_result.get('rmse_cv', None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(('ica_cv', float(r2cv), float(rmsecv) if rmsecv is not None else float('nan')))
            if isinstance(mcr_result, dict):
                r2cv = mcr_result.get('r2_cv', None)
                rmsecv = mcr_result.get('rmse_cv', None)
                if r2cv is not None and np.isfinite(r2cv):
                    candidates.append(('mcr_cv', float(r2cv), float(rmsecv) if rmsecv is not None else float('nan')))

            if candidates:
                # Sort by r2 desc
                candidates.sort(key=lambda t: t[1], reverse=True)
                best_method, best_r2, best_rmse = candidates[0]

                # Gate by min R²
                if not np.isfinite(best_r2) or best_r2 < min_r2_cv:
                    best_method = None

                # Prefer PLSR on tie within margin
                if best_method is not None and prefer_plsr_on_tie and np.isfinite(plsr_r2):
                    if best_method != 'plsr_cv' and (best_r2 - plsr_r2) < improve_margin:
                        # Switch to PLSR if within margin
                        best_method = 'plsr_cv'
                        # Update to PLSR metrics
                        for name, r2v, rmsev in candidates:
                            if name == 'plsr_cv':
                                best_r2, best_rmse = r2v, rmsev
                                break

                if best_method is not None:
                    # Override calibration summary to reflect selected multivariate model
                    calib['selected_model'] = best_method
                    calib['slope'] = float('nan')
                    calib['intercept'] = float('nan')
                    calib['r2'] = float(best_r2)
                    calib['rmse'] = float(best_rmse)
                    # Update uncertainty block
                    unc = calib.get('uncertainty', {}) if isinstance(calib.get('uncertainty', {}), dict) else {}
                    unc['r2_cv'] = float(best_r2)
                    unc['rmse_cv'] = float(best_rmse)
                    calib['uncertainty'] = unc

                    # Compute selected-model predictions for reporting
                    try:
                        y_true = np.array(calib.get('concentrations', []), dtype=float)
                        sel_preds = None
                        sel_mode_label = 'cv'
                        if best_method == 'plsr_cv' and isinstance(calib.get('plsr_model', None), dict):
                            pm = calib['plsr_model']
                            p_cv = pm.get('predictions_cv', None)
                            p_in = pm.get('predictions_in', None)
                            if p_cv is not None and len(p_cv) == len(y_true):
                                sel_preds = np.array(p_cv, dtype=float)
                                sel_mode_label = 'cv'
                            elif p_in is not None and len(p_in) == len(y_true):
                                sel_preds = np.array(p_in, dtype=float)
                                sel_mode_label = 'in'
                        elif best_method == 'ica_cv' and isinstance(ica_result, dict):
                            S = np.array(ica_result.get('scores', []), dtype=float)
                            bi = int(ica_result.get('best_component', 0))
                            if S.size and S.shape[0] == y_true.size and bi < S.shape[1]:
                                x = S[:, bi]
                                n = len(y_true)
                                if n >= 3:
                                    y_pred = np.empty(n, dtype=float)
                                    for i in range(n):
                                        xi = np.delete(x, i)
                                        yi = np.delete(y_true, i)
                                        try:
                                            k1, b1 = np.polyfit(xi, yi, 1)
                                            y_pred[i] = b1 + k1 * x[i]
                                        except Exception:
                                            y_pred[:] = np.nan
                                            break
                                    if np.all(np.isfinite(y_pred)):
                                        sel_preds = y_pred
                                        sel_mode_label = 'cv'
                                if sel_preds is None:
                                    try:
                                        k_in, b_in = np.polyfit(x, y_true, 1)
                                        sel_preds = b_in + k_in * x
                                        sel_mode_label = 'in'
                                    except Exception:
                                        pass
                        elif best_method == 'mcr_cv' and isinstance(mcr_result, dict):
                            C = np.array(mcr_result.get('contributions', []), dtype=float)
                            bi = int(mcr_result.get('best_component', 0))
                            if C.size and C.shape[0] == y_true.size and bi < C.shape[1]:
                                x = C[:, bi]
                                n = len(y_true)
                                if n >= 3:
                                    y_pred = np.empty(n, dtype=float)
                                    for i in range(n):
                                        xi = np.delete(x, i)
                                        yi = np.delete(y_true, i)
                                        try:
                                            k1, b1 = np.polyfit(xi, yi, 1)
                                            y_pred[i] = b1 + k1 * x[i]
                                        except Exception:
                                            y_pred[:] = np.nan
                                            break
                                    if np.all(np.isfinite(y_pred)):
                                        sel_preds = y_pred
                                        sel_mode_label = 'cv'
                                if sel_preds is None:
                                    try:
                                        k_in, b_in = np.polyfit(x, y_true, 1)
                                        sel_preds = b_in + k_in * x
                                        sel_mode_label = 'in'
                                    except Exception:
                                        pass
                        if sel_preds is not None and y_true.size == sel_preds.size:
                            calib['selected_actual'] = y_true.tolist()
                            calib['selected_predictions'] = sel_preds.astype(float).tolist()
                            calib['selected_predictions_mode'] = sel_mode_label
                    except Exception:
                        pass
    except Exception:
        pass

    save_calibration_outputs(calib, out_root)

    fusion_cfg = CONFIG.get('calibration', {}).get('multi_roi_fusion', {}) if isinstance(CONFIG, dict) else {}
    if bool(fusion_cfg.get('enabled', False)):
        max_feats = int(fusion_cfg.get('max_features', 4) or 4)
        fusion_result = _compute_multi_roi_fusion_calibration(
            discovered_roi,
            calib,
            out_root,
            dataset_label=dataset_label,
            max_features=max_feats,
        )
        if fusion_result:
            metadata['multi_roi_fusion'] = fusion_result

    # Compute T90/T10 response and recovery times from time-series data
    try:
        from .dynamics import compute_t90_t10_from_timeseries
        time_series_dir = os.path.join(out_root, 'time_series')
        if os.path.isdir(time_series_dir):
            dynamics_cfg = CONFIG.get('dynamics', {}) if isinstance(CONFIG, dict) else {}
            baseline_frames = int(dynamics_cfg.get('baseline_frames', 20))
            frame_rate_cfg = dynamics_cfg.get('frame_rate', None)
            min_amp_cfg = dynamics_cfg.get('min_response_amplitude_nm', 0.0)
            smooth_cfg = dynamics_cfg.get('timeseries_smoothing_window', 1)
            try:
                frame_rate_val = float(frame_rate_cfg) if frame_rate_cfg is not None else None
                if frame_rate_val is not None and frame_rate_val <= 0:
                    frame_rate_val = None
            except Exception:
                frame_rate_val = None
            try:
                min_amp_val = float(min_amp_cfg) if min_amp_cfg is not None else 0.0
            except Exception:
                min_amp_val = 0.0
            try:
                smooth_val = int(smooth_cfg) if smooth_cfg is not None else 1
            except Exception:
                smooth_val = 1
            if smooth_val < 1:
                smooth_val = 1

            dynamics_result = compute_t90_t10_from_timeseries(
                time_series_dir=time_series_dir,
                out_root=out_root,
                baseline_frames=baseline_frames,
                steady_state_frames=20,
                frame_rate=frame_rate_val,
                min_response_amplitude_nm=min_amp_val,
                smooth_window=smooth_val,
            )
            if dynamics_result and dynamics_result.get('summary'):
                print(f"[INFO] T90/T10 dynamics computed successfully")
        else:
            print(f"[WARNING] Time-series directory not found: {time_series_dir}")
    except Exception as e:
        print(f"[WARNING] Failed to compute T90/T10 dynamics: {e}")

    # After calibration, try estimating environment coefficients and update summary
    try:
        env_coeffs = compute_environment_coefficients(stable_by_conc, calib)
        if env_coeffs:
            env_info.update(env_coeffs)
            env_summary_path = save_environment_compensation_summary(env_info, out_root)
            # Optional: persist estimated coefficients into config if enabled
            try:
                est = env_coeffs.get('estimated_coefficients', {}) if isinstance(env_coeffs.get('estimated_coefficients', {}), dict) else {}
                if est and isinstance(CONFIG, dict):
                    env_cfg = CONFIG.get('environment', {}) if isinstance(CONFIG.get('environment', {}), dict) else {}
                    if env_cfg.get('autosave_coefficients', False):
                        _autosave_env_coefficients_to_config(est)
            except Exception:
                pass
    except Exception:
        pass

    if time_series_outputs:
        metadata['time_series_outputs'] = time_series_outputs

    report_results = _invoke_report_generation(out_root, metadata)
    trend_plots = generate_trend_plots(out_root)

    archive_dir = Path(out_root) / 'archives' / run_timestamp

    # Multivariate auto-selection summary (PLSR vs ICA vs MCR-ALS) by CV R²
    multi_select_path = None
    multi_cv_plot_path = None
    try:
        plsr_r2cv = None
        plsr_rmsecv = None
        if isinstance(calib, dict) and isinstance(calib.get('plsr_model', None), dict):
            pm = calib['plsr_model']
            plsr_r2cv = float(pm.get('r2_cv', float('nan')))
            plsr_rmsecv = float(pm.get('rmse_cv', float('nan')))
        ica_r2cv = float(ica_result.get('r2_cv')) if isinstance(ica_result, dict) and ica_result.get('r2_cv') is not None else None
        ica_rmsecv = float(ica_result.get('rmse_cv')) if isinstance(ica_result, dict) and ica_result.get('rmse_cv') is not None else None
        mcr_r2cv = float(mcr_result.get('r2_cv')) if isinstance(mcr_result, dict) and mcr_result.get('r2_cv') is not None else None
        mcr_rmsecv = float(mcr_result.get('rmse_cv')) if isinstance(mcr_result, dict) and mcr_result.get('rmse_cv') is not None else None

        scores = {
            'plsr': {'r2_cv': plsr_r2cv, 'rmse_cv': plsr_rmsecv, 'metrics': os.path.join(out_root, 'metrics', 'calibration_metrics.json')},
            'ica': {'r2_cv': ica_r2cv, 'rmse_cv': ica_rmsecv, 'metrics': ica_artifacts.get('metrics') if isinstance(ica_artifacts, dict) else None},
            'mcr_als': {'r2_cv': mcr_r2cv, 'rmse_cv': mcr_rmsecv, 'metrics': mcr_artifacts.get('metrics') if isinstance(mcr_artifacts, dict) else None},
        }
        best_method = None
        best_r2 = float('-inf')
        for method, sc in scores.items():
            r2v = sc.get('r2_cv')
            if r2v is not None and np.isfinite(r2v) and r2v > best_r2:
                best_r2 = r2v
                best_method = method
        multivar = {'scores': scores, 'best_method': best_method, 'best_r2_cv': best_r2}
        metrics_dir = os.path.join(out_root, 'metrics')
        _ensure_dir(metrics_dir)
        multi_select_path = os.path.join(metrics_dir, 'multivariate_selection.json')
        with open(multi_select_path, 'w') as f:
            json.dump(multivar, f, indent=2)

        # Generate a simple comparison plot of CV R² across methods
        try:
            labels = []
            values = []
            for lbl in ['plsr', 'ica', 'mcr_als']:
                r2v = scores.get(lbl, {}).get('r2_cv', None)
                if r2v is not None and np.isfinite(r2v):
                    labels.append(lbl.upper())
                    values.append(float(r2v))
            if labels:
                plots_dir = os.path.join(out_root, 'plots')
                _ensure_dir(plots_dir)
                plt.figure(figsize=(5, 3))
                xpos = np.arange(len(labels))
                plt.bar(xpos, values, color=['#4F81BD' if l != (best_method or '').upper() else '#9BBB59' for l in labels])
                plt.xticks(xpos, labels)
                plt.ylabel('CV R²')
                plt.ylim(0.0, 1.0)
                plt.title('Multivariate CV R² Comparison')
                plt.tight_layout()
                multi_cv_plot_path = os.path.join(plots_dir, 'multivariate_cv_r2.png')
                plt.savefig(multi_cv_plot_path, dpi=200)
                plt.close()
        except Exception:
            multi_cv_plot_path = None
    except Exception:
        multi_select_path = None

    # Reproducibility metadata
    try:
        versions = {
            'python': sys.version,
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': sp.__version__,
            'sklearn': sk.__version__,
            'matplotlib': matplotlib.__version__,
            'pyyaml': yaml.__version__,
        }
    except Exception:
        versions = {'python': sys.version}
    git_commit = None
    try:
        proc = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(REPO_ROOT), capture_output=True, check=True)
        git_commit = proc.stdout.decode('utf-8', errors='ignore').strip() or None
    except Exception:
        git_commit = None

    metadata.update({
        'calibration': calib,
        'roi_response': response_stats,
        'roi_repeatability': repeatability,
        'archiving': CONFIG.get('archiving', {}),
        'reporting': CONFIG.get('reporting', {}),
        'dynamics_config': dynamics_cfg,
        'roi_config': CONFIG.get('roi', {}),
        'outputs': {
            'noise_metrics': noise_metrics_path,
            'aggregated_summary': summary_csv_path,
            'quality_control': qc_summary_path,
            'environment_compensation': env_summary_path,
            'deconvolution_ica': ica_artifacts.get('metrics') if isinstance(ica_artifacts, dict) else None,
            'deconvolution_mcr_als': mcr_artifacts.get('metrics') if isinstance(mcr_artifacts, dict) else None,
            'multivariate_selection': multi_select_path,
            'multivariate_cv_plot': multi_cv_plot_path,
            'selected_predictions_csv': os.path.join(out_root, 'metrics', 'selected_predictions.csv'),
            'selected_pred_vs_actual_plot': os.path.join(out_root, 'plots', 'selected_pred_vs_actual.png'),
            'concentration_response_metrics': response_metrics_path,
            'roi_performance_metrics': performance_metrics_path,
            'dynamics_summary': dynamics_summary_path,
            'canonical_plot': canonical_plot_path,
            'concentration_response_plot': response_plot_path,
            'roi_repeatability_plot': repeatability_plot_path,
            'dynamics_plot': dynamics_plot_path,
        },
        'reports': report_results,
        'trend_plots': trend_plots,
        'archive_path': str(archive_dir),
        'aggregated_plot_paths': aggregated_plot_paths,
        'versions': versions,
        'git_commit': git_commit,
    })

    metadata_path = _save_run_metadata(out_root, metadata)
    archive_path = _archive_run(out_root, metadata)
    if archive_path is not None:
        metadata['archive_created'] = str(archive_path)
        metadata_path = _save_run_metadata(out_root, metadata)

    report_path = write_run_summary(
        calib,
        aggregated_paths,
        noise_metrics_path,
        summary_csv_path,
        canonical_plot_path,
        response_metrics_path,
        response_plot_path,
        repeatability_plot_path,
        performance_metrics_path,
        dynamics_summary_path,
        dynamics_plot_path,
        metadata_path,
        str(archive_path) if archive_path else None,
        qc_summary_path,
        report_results,
        trend_plots,
        performance,
        dynamics_summary,
        out_root,
    )

    return {
        'mapping': mapping,
        'aggregated_files': aggregated_paths,
        'aggregated_plots': aggregated_plot_paths,
        'canonical_count': len(canonical),
        'calibration': calib,
        'noise_metrics_path': noise_metrics_path,
        'aggregated_summary_csv': summary_csv_path,
        'canonical_plot': canonical_plot_path,
        'concentration_response_metrics': response_metrics_path,
        'concentration_response_plot': response_plot_path,
        'roi_repeatability_plot': repeatability_plot_path,
        'roi_performance_metrics': performance_metrics_path,
        'dynamics_summary': dynamics_summary_path,
        'dynamics_plot': dynamics_plot_path,
        'environment_compensation': env_summary_path,
        'deconvolution_ica': ica_artifacts,
        'deconvolution_mcr_als': mcr_artifacts,
        'multivariate_selection': multi_select_path,
        'run_metadata': metadata_path,
        'archive_path': str(archive_path) if archive_path else None,
        'report_artifacts': report_results,
        'top_avg_results': top_results,
        'top_avg_summary': summarize_top_comparison(top_results) if top_results else [],
        'trend_plots': trend_plots,
        'report': report_path,
    }
