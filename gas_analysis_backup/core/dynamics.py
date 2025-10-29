import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .pipeline import _read_csv_spectrum


def compute_response_recovery_times(
    root_dir: str,
    out_root: str,
    signal_column: str = 'intensity',
    exposure_start_time: float = None,
    purge_start_time: float = None,
    saturation_fraction: float = 0.9,
    baseline_fraction: float = 0.1
) -> Dict[str, object]:
    """Compute response (T90) and recovery (T10) times from time-series CSVs.

    Expected structure:
      root/<concentration>/<trial>/frame_*.csv
    Each frame is a time-series: time,intensity.

    Args:
        root_dir: Experiment root with concentration/trial/frame structure.
        out_root: Output directory for results.
        signal_column: Column to analyze ('intensity' or 'transmittance').
        exposure_start_time: Time when gas exposure begins (default: first timestamp).
        purge_start_time: Time when purge begins (default: midpoint).
        saturation_fraction: Fraction of max signal to define T90.
        baseline_fraction: Fraction of max signal to define T10.

    Returns:
        Dict with keys: 'results' (DataFrame), 'plot_path'.
    """
    results = []

    # Handle both layouts:
    # 1. root/<concentration>/<trial>/frame_*.csv
    # 2. root/<concentration>/frame_*.csv
    if not os.path.isdir(root_dir):
        raise ValueError(f"Not a directory: {root_dir}")

    conc_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, d))]
    if not conc_dirs:
        raise ValueError(f"No concentration directories found in: {root_dir}")

    for conc_dir in conc_dirs:
        conc_name = os.path.basename(conc_dir)
        try:
            conc_val = float(''.join(filter(str.isdigit, conc_name))) or 0.0
        except (ValueError, TypeError, AttributeError):
            conc_val = 0.0

        # Check for direct frames or trial subfolders
        direct_frames = [os.path.join(conc_dir, f) for f in os.listdir(conc_dir)
                         if f.lower().endswith('.csv')]
        trial_dirs = [os.path.join(conc_dir, d) for d in os.listdir(conc_dir)
                      if os.path.isdir(os.path.join(conc_dir, d))]

        all_trial_frames = []
        if direct_frames:
            all_trial_frames.append(('direct', direct_frames))
        for trial_dir in trial_dirs:
            trial_name = os.path.basename(trial_dir)
            trial_frames = [os.path.join(trial_dir, f) for f in os.listdir(trial_dir)
                            if f.lower().endswith('.csv')]
            if trial_frames:
                all_trial_frames.append((trial_name, trial_frames))

        for trial_name, frame_paths in all_trial_frames:
            dfs = []
            for path in sorted(frame_paths, key=lambda p: os.path.getmtime(p)):
                df = _read_csv_spectrum(path)
                if 'time' not in df.columns:
                    df = df.rename(columns={'wavelength': 'time'})
                if 'time' in df.columns and signal_column in df.columns:
                    dfs.append(df)
            if not dfs:
                continue
            trial_df = pd.concat(dfs, ignore_index=True).sort_values('time')
            t = trial_df['time'].values
            y = trial_df[signal_column].values

            if len(y) < 2:
                continue

            # Normalize signal to [0,1]
            y_min = y.min()
            y_max = y.max()
            if y_max - y_min < 1e-9:
                continue
            y_norm = (y - y_min) / (y_max - y_min)

            # Define time windows
            t_start = exposure_start_time if exposure_start_time is not None else t[0]
            t_purge = purge_start_time if purge_start_time is not None else 0.5 * (t[0] + t[-1])

            # Response time (T90): time to reach 90% of max after exposure
            y90 = saturation_fraction
            idx_exposure = np.where(t >= t_start)[0]
            if len(idx_exposure) == 0:
                continue
            idx_sat = np.where(y_norm[idx_exposure] >= y90)[0]
            t90 = float(t[idx_exposure[idx_sat[0]]]) if len(idx_sat) > 0 else float('nan')

            # Recovery time (T10): time to drop to 10% after purge
            y10 = baseline_fraction
            idx_purge = np.where(t >= t_purge)[0]
            if len(idx_purge) == 0:
                continue
            idx_rec = np.where(y_norm[idx_purge] <= y10)[0]
            t10 = float(t[idx_purge[idx_rec[0]]]) if len(idx_rec) > 0 else float('nan')

            results.append({
                'concentration': conc_val,
                'trial': trial_name,
                'response_time_T90': t90,
                'recovery_time_T10': t10,
            })

    df_results = pd.DataFrame(results)
    out_dir = os.path.join(out_root, 'dynamics')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'response_recovery.csv')
    df_results.to_csv(csv_path, index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    for conc in df_results['concentration'].unique():
        sub = df_results[df_results['concentration'] == conc]
        ax.scatter(sub['response_time_T90'], sub['recovery_time_T10'], label=f'{conc} ppm')
    ax.set_xlabel('Response Time T90 (s)')
    ax.set_ylabel('Recovery Time T10 (s)')
    ax.set_title('Response vs Recovery Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'response_recovery.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {
        'results': df_results,
        'plot_path': plot_path,
    }
