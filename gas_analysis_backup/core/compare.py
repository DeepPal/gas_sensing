import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def merge_calibration_metrics(
    metrics_paths: List[str],
    gas_labels: List[str],
    out_root: str
) -> Dict[str, object]:
    """Merge calibration metrics from multiple gases and plot sensitivity comparison.

    Args:
        metrics_paths: List of paths to calibration_metrics.json files.
        gas_labels: Corresponding gas names (e.g., ['acetone', 'ethanol']).
        out_root: Output directory for merged results.

    Returns:
        Dict with keys: 'merged_df', 'plot_path'.
    """
    if len(metrics_paths) != len(gas_labels):
        raise ValueError("metrics_paths and gas_labels must have same length")

    rows = []
    for path, gas in zip(metrics_paths, gas_labels):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Metrics file not found: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        rows.append({
            'gas': gas,
            'slope': data.get('slope', float('nan')),
            'r2': data.get('r2', float('nan')),
            'lod': data.get('lod', float('nan')),
            'loq': data.get('loq', float('nan')),
        })

    df = pd.DataFrame(rows)
    out_dir = os.path.join(out_root, 'comparison')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'sensitivity_comparison.csv')
    df.to_csv(csv_path, index=False)

    # Bar plot of sensitivity (slope)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df['gas'], df['slope'], yerr=df['slope'] * 0.05, capsize=5)
    ax.set_ylabel('Sensitivity (nm/ppm)')
    ax.set_title('Sensitivity Comparison Across Gases')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'sensitivity_comparison.png')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return {
        'merged_df': df,
        'plot_path': plot_path,
    }
