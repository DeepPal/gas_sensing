import os
import re
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, correlate
from scipy.optimize import curve_fit
from scipy.stats import linregress, t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

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


def _fit_plsr_calibration(canonical: Dict[float, pd.DataFrame], cfg: Dict[str, object]) -> Optional[Dict[str, object]]:
    """Fit PLSR to predict concentration from full spectrum, with simple CV model selection."""
    try:
        X, y, wl = _build_feature_matrix_from_canonical(canonical)
    except ValueError:
        return None

    # Optional absorbance transform (useful if input is transmittance)
    if bool(cfg.get('absorbance', False)):
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

    return {
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
        'coef_': pls_final.coef_.ravel().tolist(),
    }


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

    if reporting.get('generate_notebook', True) and report_script.exists():
        cmd = [sys.executable, str(report_script), '--run', str(run_dir)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results['notebook'] = str(notebook_path)
        except subprocess.CalledProcessError as exc:
            results['notebook_error'] = exc.stderr.decode('utf-8', errors='ignore')

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


# ----------------------
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
# Transmittance
# ----------------------

def compute_transmittance(sample_df: pd.DataFrame, ref_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute transmittance T = I_sample / I_ref on wavelength grid of sample."""
    if sample_df.empty:
        return sample_df
    if ref_df is None or ref_df.empty:
        return sample_df
    ref_int = np.interp(sample_df['wavelength'].values, ref_df['wavelength'].values, ref_df['intensity'].values)
    with np.errstate(divide='ignore', invalid='ignore'):
        T = np.clip(np.where(ref_int != 0, sample_df['intensity'].values / ref_int, 0.0), 0.0, 1.0)
    out = sample_df.copy()
    out['transmittance'] = T
    return out


def compute_transmittance_on_frames(frames: List[pd.DataFrame], ref_df: pd.DataFrame) -> List[pd.DataFrame]:
    return [compute_transmittance(df, ref_df) for df in frames]
# ----------------------
# Stability on multi-frame spectral trials
# ----------------------

def _align_on_grid(frames: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (wl, Y, has_T). Y is frames x wavelengths matrix (prefers transmittance)."""
    base = frames[0]
    wl = base['wavelength'].values
    has_T = 'transmittance' in base.columns
    Y = []
    for df in frames:
        vec = df['transmittance'].values if has_T and 'transmittance' in df.columns else df['intensity'].values
        if not np.array_equal(df['wavelength'].values, wl):
            vec = np.interp(wl, df['wavelength'].values, vec)
        Y.append(vec)
    return wl, np.vstack(Y), has_T


def find_stable_block(frames: List[pd.DataFrame],
                      diff_threshold: float = 0.01,
                      weight_mode: str = 'uniform',
                      top_k: Optional[int] = None) -> Tuple[int, int, np.ndarray]:
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
    wl, Y, has_T = _align_on_grid(frames)
    start_idx = max(0, min(start_idx, Y.shape[0] - 1))
    end_idx = max(start_idx, min(end_idx, Y.shape[0] - 1))
    selected = Y[start_idx:end_idx + 1]
    if weights is not None and weights.size == Y.shape[0]:
        block_weights = weights[start_idx:end_idx + 1]
        if np.any(block_weights > 0):
            block_weights = block_weights / np.sum(block_weights)
            avg_vec = np.average(selected, axis=0, weights=block_weights)
        else:
            avg_vec = selected.mean(axis=0)
    else:
        avg_vec = selected.mean(axis=0)
    out = pd.DataFrame({'wavelength': wl})
    if has_T:
        out['transmittance'] = avg_vec
    else:
        out['intensity'] = avg_vec
    return out


def average_top_frames(frames: List[pd.DataFrame], top_k: int = 5) -> pd.DataFrame:
    """Average the first `top_k` frames using intensity values before transmittance conversion."""
    if not frames:
        return pd.DataFrame()
    top_k = max(1, min(len(frames), top_k))
    selected = frames[:top_k]
    base_wl = selected[0]['wavelength'].values
    accum = np.zeros_like(base_wl, dtype=float)
    count = 0
    for df in selected:
        wl = df['wavelength'].values
        if not np.array_equal(wl, base_wl):
            intensity = np.interp(base_wl, wl, df['intensity'].values)
        else:
            intensity = df['intensity'].values
        accum += intensity
        count += 1
    avg_intensity = accum / max(count, 1)
    return pd.DataFrame({'wavelength': base_wl, 'intensity': avg_intensity})


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
    """Aggregate trials per concentration using an averaged spectrum for robustness."""
    canonical: Dict[float, pd.DataFrame] = {}
    for conc, trials in stable_results.items():
        if not trials:
            continue

        base_wl: Optional[np.ndarray] = None
        accum: Dict[str, List[np.ndarray]] = {}
        weights_by_col: Dict[str, List[float]] = {}

        for df in trials.values():
            wl = df['wavelength'].to_numpy()
            if base_wl is None:
                base_wl = wl
            elif not np.array_equal(base_wl, wl):
                # Interpolate onto the first trial's wavelength grid
                wl_interp = base_wl
            else:
                wl_interp = wl

            for col in df.columns:
                if col == 'wavelength':
                    continue
                series = df[col].to_numpy()
                if base_wl is not None and not np.array_equal(base_wl, wl):
                    series = np.interp(base_wl, wl, series)
                accum.setdefault(col, []).append(series)

                metrics = estimate_noise_metrics(base_wl if base_wl is not None else wl, series)
                weight = float(max(metrics.snr, 1e-6) / (metrics.mad + 1e-6))
                weights_by_col.setdefault(col, []).append(weight)

        if base_wl is None:
            continue

        canonical_df = pd.DataFrame({'wavelength': base_wl})
        for col, stacks in accum.items():
            mat = np.vstack(stacks)
            weights = np.array(weights_by_col.get(col, [1.0] * len(stacks)), dtype=float)
            weights = np.clip(weights, 1e-6, None)
            weights /= np.sum(weights)
            canonical_df[col] = np.average(mat, axis=0, weights=weights)
        canonical[conc] = canonical_df

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
        mask &= (x >= float(min_wl))
    if max_wl is not None:
        mask &= (x <= float(max_wl))
    if not np.any(mask):
        raise ValueError("No data points remain after applying wavelength constraints")
    return x[mask], y[mask]


def _prepare_calibration_signal(df: pd.DataFrame,
                                centroid_cfg: Optional[Dict[str, object]] = None) -> Tuple[np.ndarray, np.ndarray]:
    centroid_cfg = centroid_cfg or {}
    ycol = 'transmittance' if 'transmittance' in df.columns else 'intensity'
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
    is_min = bool(yy.min() < baseline)
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


def find_roi_and_calibration(canonical: Dict[float, pd.DataFrame]) -> Dict[str, object]:
    """Compute wavelength shift vs concentration and fit linear calibration.

    Returns dict with keys: 'concentrations', 'peak_wavelengths', 'slope', 'intercept',
    'r2', 'rmse', 'slope_se', 'slope_ci_low', 'slope_ci_high', 'lod', 'loq', 'roi_center'.
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
    roi_cfg = CONFIG.get('roi', {}) or {}
    min_wl_roi = roi_cfg.get('min_wavelength', 500.0)
    max_wl_roi = roi_cfg.get('max_wavelength', 900.0)
    
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
    base_center_centroid = _find_peak_wavelength(base_df, centroid_cfg=calib_cfg_with_limits)
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
        # Quick intensity-based scan to find best wavelength
        best_wl_intensity = None
        best_r2_intensity = -1
        
        # Sample a subset of wavelengths for speed
        ref_df = items[-1][1]  # Highest concentration
        all_wl = ref_df['wavelength'].values
        wl_mask = (all_wl >= min_wl_roi) & (all_wl <= max_wl_roi)
        candidate_wl = all_wl[wl_mask][::10]  # Sample every 10th wavelength
        
        for test_wl in candidate_wl:
            intensities_at_wl = []
            for _, df in items:
                df_wl = df['wavelength'].values
                df_int = df['intensity'].values
                closest_idx = np.argmin(np.abs(df_wl - test_wl))
                intensities_at_wl.append(df_int[closest_idx])
            
            if len(intensities_at_wl) == len(concs):
                slope_i, intercept_i, r_val_i, _, _ = linregress(concs, intensities_at_wl)
                r2_i = r_val_i ** 2
                if r2_i > best_r2_intensity:
                    best_r2_intensity = r2_i
                    best_wl_intensity = test_wl
        
        # Add as a feature if found
        if best_wl_intensity is not None and best_r2_intensity > 0.1:
            # Track intensities at this wavelength across all concentrations
            intensity_values = []
            for _, df in items:
                df_wl = df['wavelength'].values
                df_int = df['intensity'].values
                closest_idx = np.argmin(np.abs(df_wl - best_wl_intensity))
                intensity_values.append(df_int[closest_idx])
            
            feature_results.append({
                'type': 'intensity_best',
                'wavelengths': intensity_values,  # Store intensities here for evaluation
                'center': best_wl_intensity,
                'is_intensity_based': True,  # Flag to handle differently
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
        ref_intensity_full = ref_df['intensity'].values
        
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
                df_intensity = df['intensity'].values
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
                df_intensity = df['intensity'].values
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
    quality_threshold = 0.05
    
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
                    print(f"  [FILTER] Rejecting {feature_type} at {center_wl:.2f} nm (outside ROI)")
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
                
                if r2 < quality_threshold and cv > 0.01:
                    print(f"  [FILTER] Rejecting {feature_type} at {center_wl:.2f} nm (R2={r2:.4f}, CV={cv:.4f})")
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
    if best_result is not None:
        peak_list = best_peak_list
        base_center = best_result['center']
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
        print("\n⚠ No suitable feature found, using original centroid method")
        import sys; sys.stdout.flush()

    peaks = np.array(peak_list, dtype=float)

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

    best_model = linear_model
    best_predictions = preds_lin

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
            if np.isfinite(r2_poly):
                model_mode = str(calib_cfg.get('model', 'linear')).lower()
                if (model_mode == 'polynomial') or (model_mode == 'auto' and r2_poly > best_model['r2']):
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
        except (np.linalg.LinAlgError, ValueError):
            poly_info = None

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
    residuals = peaks - best_predictions
    rmse_best = float(np.sqrt(np.mean(residuals ** 2)))
    if abs(slope_for_lod) < eps:
        lod = float('inf')
        loq = float('inf')
    else:
        lod = 3.0 * rmse_best / abs(slope_for_lod)
        loq = 10.0 * rmse_best / abs(slope_for_lod)

    roi_center = float(np.median(peaks))

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
        'robust_model': robust_info,
        'bootstrap': bootstrap_info,
        'feature_candidates': feature_summary,
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
    return 'transmittance' if 'transmittance' in df.columns else 'intensity'


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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for conc, arr in sorted(avg_by_conc.items(), key=lambda kv: kv[0]):
        ax1.plot(wl, arr, label=f"{conc:g} ppm")
    ax1.axvspan(roi_start, roi_end, color='orange', alpha=0.2, label='Max response ROI')
    ax1.set_ylabel('Transmittance')
    ax1.set_title('Average Transmittance per Concentration')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(wl, slopes, label='Slope (ΔT / Δppm)')
    ax2.axvspan(roi_start, roi_end, color='orange', alpha=0.2)
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
        f"- **Slope**: {calib['slope']:.4f} nm/ppm",
        f"- **Intercept**: {calib['intercept']:.4f} nm",
        f"- **R²**: {calib['r2']:.4f}",
        f"- **RMSE**: {calib['rmse']:.4f} nm",
        f"- **LOD**: {calib['lod']:.4f} ppm",
        f"- **LOQ**: {calib['loq']:.4f} ppm",
        f"- **ROI Center**: {calib['roi_center']:.4f} nm",
        '',
    ]

    lines.extend([
        '## Aggregated Spectra',
        '',
        f'- **Noise metrics**: `{noise_metrics_path}`',
        f'- **Aggregated summary CSV**: `{summary_csv_path}`',
        f'- **Concentration response metrics**: `{response_metrics_path}`',
    ])

    if canonical_plot_path:
        lines.append(f'- **Canonical overlay plot**: `{canonical_plot_path}`')
    if response_plot_path:
        lines.append(f'- **Concentration response plot**: `{response_plot_path}`')
    if repeatability_plot_path:
        lines.append(f'- **ROI repeatability plot**: `{repeatability_plot_path}`')
    if performance_metrics_path:
        lines.append(f'- **ROI performance metrics**: `{performance_metrics_path}`')
    if dynamics_summary_path:
        lines.append(f'- **Dynamics summary**: `{dynamics_summary_path}`')
    if dynamics_plot_path:
        lines.append(f'- **Dynamics plot**: `{dynamics_plot_path}`')
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

    lines.extend(['', '### Files per Concentration', ''])

    for conc, trials in sorted(aggregated_paths.items(), key=lambda kv: kv[0]):
        lines.append(f'- **{conc:g}**')
        for trial, path in trials.items():
            lines.append(f'  - `{trial}`: `{path}`')
        lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    return out_path


def save_calibration_outputs(calib: Dict[str, object], out_root: str):
    metrics_dir = os.path.join(out_root, 'metrics')
    plots_dir = os.path.join(out_root, 'plots')
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    # calibration.csv
    cal_path = os.path.join(metrics_dir, 'calibration.csv')
    data = pd.DataFrame({
        'concentration': calib['concentrations'],
        'peak_wavelength': calib['peak_wavelengths'],
    })
    data.to_csv(cal_path, index=False)

    # metrics.json
    meta = calib.copy()
    meta_path = os.path.join(metrics_dir, 'calibration_metrics.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(calib['concentrations'])
    y = np.array(calib['peak_wavelengths'])
    slope = calib['slope']
    intercept = calib['intercept']
    ax.scatter(x, y, label='Data')
    xx = np.linspace(x.min(), x.max(), 100)
    yy = intercept + slope * xx
    ax.plot(xx, yy, 'r-', label=f"Fit (R^2={calib['r2']:.3f})")
    ax.set_xlabel('Concentration (ppm)')
    ax.set_ylabel('Peak Wavelength (nm)')
    ax.set_title('Calibration: Wavelength Shift vs Concentration')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, 'calibration.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

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


# ----------------------
# High-level run helper (optional for CLI)
# ----------------------

def run_full_pipeline(root_dir: str, ref_path: str, out_root: str,
                      diff_threshold: float = 0.01,
                      avg_top_n: Optional[int] = None,
                      scan_full: bool = False,
                      top_k_candidates: int = 5) -> Dict[str, object]:
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

    stable_by_conc: Dict[float, Dict[str, pd.DataFrame]] = {}
    top_path: Dict[str, Dict[float, Dict[str, pd.DataFrame]]] = {}
    for conc, trials in mapping.items():
        conc_key = float(conc)
        conc_entry: Dict[str, object] = {
            'raw_trial_count': len(trials),
            'retained_trial_count': 0,
        }
        metadata['trials'][str(conc_key)] = conc_entry

        processed_trials: Dict[str, pd.DataFrame] = {}
        averaged_intensity_trials: Dict[str, pd.DataFrame] = {}
        averaged_trans_trials: Dict[str, pd.DataFrame] = {}
        spectral_arrays: List[np.ndarray] = []
        trial_names: List[str] = []
        base_wavelengths: Optional[np.ndarray] = None
        wavelengths_consistent = True

        for trial, frames in trials.items():
            frames_sorted = sorted(frames, key=lambda p: os.path.getmtime(p))
            dfs = [_read_csv_spectrum(p) for p in frames_sorted]
            dfs = [df for df in dfs if not df.empty]
            if not dfs:
                continue

            if apply_frames:
                dfs = [_preprocess_dataframe(df, stage='frame') for df in dfs]

            stability_cfg = CONFIG.get('stability', {})
            weight_mode = stability_cfg.get('weight_mode', 'uniform')
            top_k = stability_cfg.get('top_k', 0)
            s, e, weights = find_stable_block(
                dfs,
                diff_threshold=diff_threshold,
                weight_mode=weight_mode,
                top_k=int(top_k) if top_k else None,
            )
            avg_df = average_stable_block(dfs, s, e, weights=weights)
            if ref_df is not None and use_trans:
                avg_df = compute_transmittance(avg_df, ref_df)
            if apply_trans:
                avg_df = _preprocess_dataframe(avg_df, stage='transmittance')

            processed_trials[trial] = avg_df

            if avg_top_n:
                top_avg_int = average_top_frames(dfs, top_k=avg_top_n)
                averaged_intensity_trials[trial] = top_avg_int
                if ref_df is not None and use_trans:
                    top_avg_trans = compute_transmittance(top_avg_int, ref_df)
                else:
                    top_avg_trans = top_avg_int.copy()
                averaged_trans_trials[trial] = top_avg_trans

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

        final_trials = {trial: df for trial, df in processed_trials.items() if trial not in flagged_trials}
        final_intensity_top = {trial: averaged_intensity_trials[trial]
                               for trial in processed_trials
                               if trial not in flagged_trials and trial in averaged_intensity_trials}
        final_trans_top = {trial: averaged_trans_trials[trial]
                           for trial in processed_trials
                           if trial not in flagged_trials and trial in averaged_trans_trials}
        conc_entry['retained_trial_count'] = len(final_trials)
        if final_trials:
            stable_by_conc[conc] = final_trials
            if avg_top_n:
                if final_intensity_top:
                    top_path.setdefault('intensity', {})[conc] = final_intensity_top
                if final_trans_top:
                    top_path.setdefault('transmittance', {})[conc] = final_trans_top

    if not stable_by_conc:
        raise RuntimeError("No stable blocks found across trials")

    aggregated_paths = save_aggregated_spectra(stable_by_conc, out_root)

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
            response_stats_subset, avg_by_conc_subset = compute_concentration_response(data_map, top_k_candidates=top_k_candidates)
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
            }

    canonical = select_canonical_per_concentration(stable_by_conc)
    save_canonical_spectra(canonical, out_root)

    noise_metrics = compute_noise_metrics_map(stable_by_conc)
    noise_metrics_path = save_noise_metrics(noise_metrics, out_root)
    summary_csv_path = save_aggregated_summary(stable_by_conc, noise_metrics, out_root)
    aggregated_plot_paths = save_aggregated_plots(stable_by_conc, out_root)
    canonical_plot_path = save_canonical_overlay(canonical, out_root)

    response_stats, avg_by_conc = compute_concentration_response(stable_by_conc, top_k_candidates=top_k_candidates)
    repeatability = compute_roi_repeatability(stable_by_conc, response_stats)
    performance = compute_roi_performance(repeatability)
    response_metrics_path = save_concentration_response_metrics(response_stats, repeatability, out_root, name='concentration_response')
    response_plot_path = save_concentration_response_plot(response_stats, avg_by_conc, out_root, name='concentration_response', clamp_to_roi=True)
    repeatability_plot_path = save_roi_repeatability_plot(stable_by_conc, response_stats, out_root)
    performance_metrics_path = save_roi_performance_metrics(performance, out_root)

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

    calib = find_roi_and_calibration(canonical)
    save_calibration_outputs(calib, out_root)

    report_results = _invoke_report_generation(out_root, metadata)
    trend_plots = generate_trend_plots(out_root)

    archive_dir = Path(out_root) / 'archives' / run_timestamp

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
        'run_metadata': metadata_path,
        'archive_path': str(archive_path) if archive_path else None,
        'report_artifacts': report_results,
        'top_avg_results': top_results,
        'top_avg_summary': summarize_top_comparison(top_results) if top_results else [],
        'trend_plots': trend_plots,
        'report': report_path,
    }
