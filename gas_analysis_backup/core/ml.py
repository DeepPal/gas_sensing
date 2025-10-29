import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

from .pipeline import scan_experiment_root, _read_csv_spectrum, _find_peak_wavelength


def build_ml_dataset(
    root_dir: str,
    ref_path: str = None,
    out_root: str = None
) -> pd.DataFrame:
    """Build a tidy dataset for ML: gas, concentration, peak_wavelength, transmittance_at_peak.

    Args:
        root_dir: Experiment root with concentration/trial/frame structure.
        ref_path: Optional reference CSV for transmittance.
        out_root: Optional output path for dataset CSV.

    Returns:
        DataFrame with columns: gas, concentration, peak_wavelength, transmittance.
    """
    from .pipeline import load_reference_csv, compute_transmittance_on_frames, select_canonical_per_concentration

    mapping = scan_experiment_root(root_dir)
    ref_df = load_reference_csv(ref_path) if ref_path else None

    # Build canonical spectra per concentration
    stable_by_conc = {}
    for conc, trials in mapping.items():
        trial_out = {}
        for trial, frames in trials.items():
            frames_sorted = sorted(frames, key=lambda p: os.path.getmtime(p))
            dfs = [_read_csv_spectrum(p) for p in frames_sorted]
            if ref_df is not None:
                dfs = compute_transmittance_on_frames(dfs, ref_df)
            if not dfs:
                continue
            from .pipeline import find_stable_block, average_stable_block, CONFIG
            stability_cfg = CONFIG.get('stability', {})
            top_k = stability_cfg.get('top_k', 0)
            s, e, weights = find_stable_block(
                dfs,
                diff_threshold=stability_cfg.get('diff_threshold', 0.01),
                weight_mode=stability_cfg.get('weight_mode', 'uniform'),
                top_k=int(top_k) if top_k else None,
            )
            avg = average_stable_block(dfs, s, e, weights=weights)
            trial_out[trial] = avg
        if trial_out:
            stable_by_conc[conc] = trial_out

    canonical = select_canonical_per_concentration(stable_by_conc)

    # Extract features
    rows = []
    for conc, df in canonical.items():
        peak_wl = _find_peak_wavelength(df)
        ycol = 'transmittance' if 'transmittance' in df.columns else 'intensity'
        # Interpolate transmittance at peak_wl
        interp_val = np.interp(peak_wl, df['wavelength'].values, df[ycol].values)
        wl = df['wavelength'].values
        y = df[ycol].values
        mask = (wl >= 500.0) & (wl <= 900.0)
        band_mean = float(np.mean(y[mask])) if np.any(mask) else float(np.mean(y))

        rows.append({
            'gas': 'unknown',  # Placeholder; override if multi-gas
            'concentration': conc,
            'peak_wavelength': peak_wl,
            'transmittance': interp_val,
            'band_mean_500_900': band_mean,
        })

    df_out = pd.DataFrame(rows)
    if out_root:
        out_dir = os.path.join(out_root, 'ml')
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'ml_dataset.csv')
        df_out.to_csv(csv_path, index=False)
    return df_out


def train_gas_predictor(
    df: pd.DataFrame,
    target_col: str = 'concentration',
    feature_cols: list = None,
    out_root: str = None
) -> Dict[str, object]:
    """Train and evaluate ML models to predict gas concentration.

    Args:
        df: ML dataset from build_ml_dataset().
        target_col: Target variable (default: 'concentration').
        feature_cols: Features to use (default: ['peak_wavelength', 'transmittance']).
        out_root: Optional output path for model and scores.

    Returns:
        Dict with keys: 'models', 'scores', 'scaler'.
    """
    if feature_cols is None:
        feature_cols = ['peak_wavelength', 'transmittance']

    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        'Ridge': Ridge(alpha=1.0),
        'SVR': SVR(kernel='rbf', C=100, gamma='scale'),
    }

    scores = {}
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        train_r2 = r2_score(y, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y, y_pred))
        scores[name] = {
            'cv_r2_mean': float(cv_scores.mean()),
            'cv_r2_std': float(cv_scores.std()),
            'train_r2': float(train_r2),
            'train_rmse': float(train_rmse),
        }

    if out_root:
        out_dir = os.path.join(out_root, 'ml')
        os.makedirs(out_dir, exist_ok=True)
        scores_path = os.path.join(out_dir, 'model_scores.json')
        with open(scores_path, 'w') as f:
            json.dump(scores, f, indent=2)
        scaler_path = os.path.join(out_dir, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        for name, model in models.items():
            model_path = os.path.join(out_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)

    return {
        'models': models,
        'scores': scores,
        'scaler': scaler,
    }
