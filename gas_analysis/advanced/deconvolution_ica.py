import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _signal_column(df: pd.DataFrame) -> str:
    return 'transmittance' if 'transmittance' in df.columns else 'intensity'


def _build_feature_matrix_from_canonical(canonical: Dict[float, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not canonical:
        raise ValueError('Empty canonical')
    items = sorted(canonical.items(), key=lambda kv: kv[0])
    y = np.array([float(k) for k, _ in items], dtype=float)
    base_wl = items[0][1]['wavelength'].to_numpy()
    X_list: List[np.ndarray] = []
    for _, df in items:
        wl = df['wavelength'].to_numpy()
        col = _signal_column(df)
        vec = df[col].to_numpy()
        if not np.array_equal(wl, base_wl):
            vec = np.interp(base_wl, wl, vec)
        X_list.append(vec)
    X = np.vstack(X_list)
    return X, y, base_wl


def fit_ica_from_canonical(canonical: Dict[float, pd.DataFrame], cfg: Dict[str, object]) -> Optional[Dict[str, object]]:
    try:
        X, y, wl = _build_feature_matrix_from_canonical(canonical)
    except ValueError:
        return None

    n_samples, n_features = X.shape
    if n_samples < 3 or n_features < 5:
        return None

    n_components = int(cfg.get('n_components', min(n_samples, 3)))
    n_components = max(1, min(n_components, n_samples))
    max_iter = int(cfg.get('max_iter', 500))
    tol = float(cfg.get('tol', 1e-4))
    random_state = cfg.get('random_state', 0)

    try:
        ica = FastICA(n_components=n_components, whiten=True, max_iter=max_iter, tol=tol, random_state=random_state)
        S = ica.fit_transform(X)  # (n_samples, n_components)
        A = ica.mixing_            # (n_features, n_components)
        K = ica.components_        # (n_components, n_features)
    except Exception:
        return None

    # Orient components so scores correlate positively with concentration
    S_adj = S.copy()
    K_adj = K.copy()
    for i in range(n_components):
        s = S[:, i]
        corr = np.corrcoef(s, y)[0, 1] if np.std(s) > 0 and np.std(y) > 0 else 0.0
        if corr < 0:
            S_adj[:, i] = -S_adj[:, i]
            K_adj[i, :] = -K_adj[i, :]

    # Evaluate component predictive ability (LOOCV)
    comp_metrics: List[Dict[str, float]] = []

    def _loocv_scores_1d(x: np.ndarray, y_true: np.ndarray) -> Tuple[float, float]:
        n = len(y_true)
        if n < 3:
            return float('nan'), float('nan')
        y_pred = np.empty(n, dtype=float)
        for i in range(n):
            xi = np.delete(x, i)
            yi = np.delete(y_true, i)
            try:
                k, b = np.polyfit(xi, yi, 1)
                y_pred[i] = b + k * x[i]
            except Exception:
                return float('nan'), float('nan')
        try:
            ss_res = float(np.sum((y_true - y_pred) ** 2))
            ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
            r2cv = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmsecv = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            return float(r2cv), float(rmsecv)
        except Exception:
            return float('nan'), float('nan')

    for i in range(n_components):
        s = S_adj[:, i]
        try:
            k_in, b_in = np.polyfit(s, y, 1)
            yhat_in = b_in + k_in * s
            ss_res = float(np.sum((y - yhat_in) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2_in = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse_in = float(np.sqrt(np.mean((y - yhat_in) ** 2)))
        except Exception:
            r2_in, rmse_in = float('nan'), float('nan')
        r2_cv, rmse_cv = _loocv_scores_1d(s, y)
        comp_metrics.append({'r2_in': r2_in, 'rmse_in': rmse_in, 'r2_cv': r2_cv, 'rmse_cv': rmse_cv})

    best_idx = int(np.nanargmax([m['r2_cv'] for m in comp_metrics])) if comp_metrics else 0
    best = comp_metrics[best_idx] if comp_metrics else {'r2_cv': float('nan'), 'rmse_cv': float('nan'), 'r2_in': float('nan'), 'rmse_in': float('nan')}

    return {
        'method': 'ica_fastica',
        'n_components': n_components,
        'wavelengths': wl.tolist(),
        'concentrations': y.tolist(),
        'components': K_adj.tolist(),
        'scores': S_adj.tolist(),
        'mixing': A.tolist() if A is not None else None,
        'component_metrics': comp_metrics,
        'best_component': best_idx,
        'r2_in': float(best.get('r2_in', float('nan'))),
        'rmse_in': float(best.get('rmse_in', float('nan'))),
        'r2_cv': float(best.get('r2_cv', float('nan'))),
        'rmse_cv': float(best.get('rmse_cv', float('nan'))),
    }


def save_ica_outputs(res: Dict[str, object], out_root: str) -> Dict[str, str]:
    metrics_path = None
    comp_plot_path = None
    pred_plot_path = None
    try:
        import os
        from pathlib import Path
        run_dir = Path(out_root)
        metrics_dir = run_dir / 'metrics'
        plots_dir = run_dir / 'plots'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Add linear mapping for best component (y ≈ b + k*s)
        try:
            y = np.array(res.get('concentrations', []), dtype=float)
            S = np.array(res.get('scores', []), dtype=float)
            bi = int(res.get('best_component', 0))
            if S.size and y.size and S.shape[0] == y.size and bi < S.shape[1]:
                s = S[:, bi]
                k_lin, b_lin = np.polyfit(s, y, 1)
                res['best_linear_k'] = float(k_lin)
                res['best_linear_b'] = float(b_lin)
        except Exception:
            pass

        metrics_path = str(metrics_dir / 'deconvolution_ica.json')
        with open(metrics_path, 'w') as f:
            json.dump(res, f, indent=2)

        # Component spectra
        try:
            wl = np.array(res.get('wavelengths', []), dtype=float)
            comps = np.array(res.get('components', []), dtype=float)
            if comps.size and wl.size and comps.shape[1] == wl.size:
                plt.figure(figsize=(8, 4))
                for i in range(comps.shape[0]):
                    plt.plot(wl, comps[i], label=f'C{i}')
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('ICA Component (arb.)')
                plt.title('ICA Component Spectra')
                plt.grid(True, alpha=0.3)
                plt.legend(ncol=2, fontsize=8)
                plt.tight_layout()
                comp_plot_path = str(plots_dir / 'ica_components.png')
                plt.savefig(comp_plot_path, dpi=200)
                plt.close()
        except Exception:
            pass

        # Predicted vs actual for best component
        try:
            y = np.array(res.get('concentrations', []), dtype=float)
            S = np.array(res.get('scores', []), dtype=float)
            bi = int(res.get('best_component', 0))
            if S.size and y.size and S.shape[0] == y.size and bi < S.shape[1]:
                s = S[:, bi]
                k, b = np.polyfit(s, y, 1)
                yhat = b + k * s
                plt.figure(figsize=(5, 4))
                plt.scatter(y, yhat, s=20)
                mn, mx = float(min(y.min(), yhat.min())), float(max(y.max(), yhat.max()))
                plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
                r2cv = res.get('r2_cv', float('nan'))
                plt.xlabel('Actual (ppm)')
                plt.ylabel('Predicted (ppm)')
                plt.title(f'ICA Best Component (R²_CV={r2cv:.3f})')
                plt.tight_layout()
                pred_plot_path = str(plots_dir / 'ica_pred_vs_actual.png')
                plt.savefig(pred_plot_path, dpi=200)
                plt.close()
        except Exception:
            pass

    except Exception:
        pass

    return {
        'metrics': metrics_path,
        'components_plot': comp_plot_path,
        'pred_vs_actual_plot': pred_plot_path,
    }
