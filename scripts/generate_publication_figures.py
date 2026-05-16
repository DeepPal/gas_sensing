#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for Manuscript

This script creates all figures required for the publication:
1. Multi-gas calibration comparison (6-panel)
2. Selectivity bar chart
3. ROI discovery heatmap
4. Feature engineering demonstration
5. Cross-sensitivity matrix
6. Clinical validation ROC curve

Target: Sensors & Actuators: B. Chemical (Tier-1)
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
matplotlib.use('Agg')

# Configure publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "output" / "scientific"
DEFAULT_WORLD_CLASS_PATH = PROJECT_ROOT / "output" / "world_class_analysis" / "comprehensive_results.json"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / 'output' / 'publication_figures'
DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR: Path = DEFAULT_FIGURES_DIR

DEFAULT_GASES = [
    "Acetone",
    "Ethanol",
    "Methanol",
    "Isopropanol",
    "Toluene",
    "Xylene",
]


@dataclass
class FigureArtifact:
    name: str
    path: Path
    sources: List[str]
    description: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig, save_path: Path, sources: List[str], description: str) -> FigureArtifact:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[SAVED] {save_path}")
    return FigureArtifact(name=save_path.name, path=save_path, sources=sources, description=description)


def normal_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def detection_probability(slope: float, noise_std: float, lod_ppm: float, fp_rate: float = 0.05) -> Optional[float]:
    if noise_std in (0, None) or slope in (0, None) or lod_ppm in (0, None):
        return None
    snr = abs(slope) * lod_ppm / noise_std
    threshold_z = 1.6448536269514722  # 95th percentile (approx) for FP=5%
    return max(0.0, min(1.0, 1 - normal_cdf(threshold_z - snr)))


def write_manifest(artifacts: List[FigureArtifact], manifest_path: Path) -> None:
    manifest = []
    for artifact in artifacts:
        png_hash = sha256_file(artifact.path)
        pdf_hash = sha256_file(artifact.path.with_suffix('.pdf')) if artifact.path.with_suffix('.pdf').exists() else None
        manifest.append({
            "name": artifact.name,
            "png_path": str(artifact.path.resolve()),
            "pdf_path": str(artifact.path.with_suffix('.pdf').resolve()),
            "sha256_png": png_hash,
            "sha256_pdf": pdf_hash,
            "sources": artifact.sources,
            "description": artifact.description,
        })

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "artifacts": manifest,
        }, f, indent=2)
    print(f"[SAVED] {manifest_path}")


def aggregate_sources(source_map: Dict[str, str], gases: List[str]) -> List[str]:
    sources = [source_map[gas] for gas in gases if gas in source_map]
    return sorted(set(sources))

# Gas colors for consistent styling
GAS_COLORS = {
    'Acetone': '#E74C3C',      # Red - primary target
    'Methanol': '#3498DB',     # Blue
    'Ethanol': '#2ECC71',      # Green
    'Isopropanol': '#9B59B6',  # Purple
    'Toluene': '#F39C12',      # Orange
    'Xylene': '#1ABC9C',       # Teal
}

def load_scientific_results(results_root: Path, gases: List[str]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """Load all scientific pipeline results from the standardized layout."""
    results: Dict[str, dict] = {}
    sources: Dict[str, str] = {}

    for gas in gases:
        metrics_path = results_root / gas / "metrics" / "calibration_metrics.json"
        if not metrics_path.exists():
            # Fallback to legacy naming if user still has old outputs
            legacy_path = PROJECT_ROOT / "output" / f"{gas.lower()}_scientific" / "metrics" / "calibration_metrics.json"
            metrics_path = legacy_path if legacy_path.exists() else None

        if metrics_path and metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as f:
                results[gas] = json.load(f)
            sources[gas] = str(metrics_path)
        else:
            print(f"[WARN] Missing metrics for {gas} at {results_root / gas}")

    return results, sources


def load_world_class_results(path: Path) -> Tuple[Dict, Optional[str]]:
    """Load world-class analysis results."""
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), str(path)
    return {}, None


def figure1_multigas_calibration(results: Dict[str, dict], source_map: Dict[str, str]) -> FigureArtifact:
    """
    Figure 1: Multi-gas calibration curves (6-panel grid)
    Shows Δλ vs concentration for all gases with linear fits.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    gases = DEFAULT_GASES
    
    for idx, gas in enumerate(gases):
        ax = axes[idx]
        
        if gas not in results:
            ax.text(0.5, 0.5, f'No data for {gas}', ha='center', va='center', transform=ax.transAxes)
            continue
        
        data = results[gas]
        cal = data.get('calibration_wavelength_shift', {}).get('centroid', {})
        
        if not cal:
            continue
        
        concs = np.array(cal['concentrations'])
        delta_lambda = np.array(cal['delta_lambda'])
        slope = cal['slope']
        intercept = cal['intercept']
        r2 = cal['r2']
        lod = cal['lod_ppm']
        slope_ci = cal.get('slope_ci_95', [None, None])
        loocv_r2 = cal.get('r2_cv', results[gas].get('loocv_validation', {}).get('wavelength_shift', {}).get('r2_cv'))
        noise_std = cal.get('noise_std', cal.get('rmse'))
        
        # Plot data points
        ax.scatter(concs, delta_lambda, s=80, c=GAS_COLORS.get(gas, 'gray'), 
                   edgecolors='black', linewidth=1, zorder=5, label='Data')
        
        # Plot fit line
        x_fit = np.linspace(0, max(concs) * 1.1, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color=GAS_COLORS.get(gas, 'gray'), 
                linewidth=2, alpha=0.8, label='Linear fit')

        # Confidence band using slope CI propagated through x
        if slope_ci[0] is not None and slope_ci[1] is not None:
            y_low = slope_ci[0] * x_fit + intercept
            y_high = slope_ci[1] * x_fit + intercept
            ax.fill_between(
                x_fit,
                y_low,
                y_high,
                color=GAS_COLORS.get(gas, 'gray'),
                alpha=0.15,
                label='Slope 95% CI'
            )

        # Add equation and metrics
        sign = '+' if intercept >= 0 else ''
        eq_text = f'Δλ = {slope:.4f}C {sign}{intercept:.4f}'
        metrics_lines = [
            f'R² = {r2:.4f}',
            f'LoD = {lod:.2f} ppm',
        ]
        if loocv_r2 is not None:
            metrics_lines.append(f'LOOCV R² = {loocv_r2:.3f}')
        if noise_std is not None:
            metrics_lines.append(f'σ = {noise_std:.3f} nm')
        metrics_text = "\n".join(metrics_lines)
        
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold')
        ax.text(0.05, 0.80, metrics_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top')
        
        ax.set_xlabel('Concentration (ppm)')
        ax.set_ylabel('Δλ (nm)')
        ax.set_title(f'{gas}', fontweight='bold', fontsize=12)
        ax.set_xlim(0, max(concs) * 1.15)
        
        # Highlight acetone as primary target
        if gas == 'Acetone':
            ax.set_facecolor('#FFF5F5')
    
    plt.suptitle('Multi-Gas Calibration Curves (Wavelength Shift Method)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'Figure1_multigas_calibration.png'
    fig_artifact = save_figure(
        fig,
        save_path,
        sources=aggregate_sources(source_map, gases),
        description="Multi-gas calibration curves with 95% CI bands and LOOCV annotations",
    )
    return fig_artifact


def figure2_selectivity_comparison(results: Dict[str, dict], source_map: Dict[str, str]) -> FigureArtifact:
    """
    Figure 2: Selectivity bar chart comparing sensitivity and LoD across gases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gases = DEFAULT_GASES
    sensitivities = []
    lods = []
    r2_values = []
    detection_probs = []
    colors = []
    
    for gas in gases:
        if gas in results:
            cal = results[gas].get('calibration_wavelength_shift', {}).get('centroid', {})
            sensitivities.append(abs(cal.get('slope', 0)))
            lods.append(cal.get('lod_ppm', 0))
            r2_values.append(cal.get('r2', 0))
            detection_probs.append(detection_probability(cal.get('slope'), cal.get('noise_std', cal.get('rmse')), cal.get('lod_ppm')))
            colors.append(GAS_COLORS.get(gas, 'gray'))
        else:
            sensitivities.append(0)
            lods.append(0)
            r2_values.append(0)
            detection_probs.append(None)
            colors.append('gray')
    
    x = np.arange(len(gases))
    width = 0.6
    
    # Panel A: Sensitivity
    ax1 = axes[0]
    bars1 = ax1.bar(x, sensitivities, width, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Sensitivity (nm/ppm)', fontsize=11)
    ax1.set_xlabel('Gas Species', fontsize=11)
    ax1.set_title('(a) Sensitivity Comparison', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gases, rotation=45, ha='right')
    ax1.axhline(y=0.116, color='red', linestyle='--', linewidth=1.5, 
                label='Paper benchmark (0.116 nm/ppm)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Add value labels
    for bar, val in zip(bars1, sensitivities):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Panel B: Detection Limit
    ax2 = axes[1]
    lod_bars = ax2.bar(x, lods, width, color=colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Detection Limit (ppm)', fontsize=11)
    ax2.set_xlabel('Gas Species', fontsize=11)
    ax2.set_title('(b) Detection Limit Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gases, rotation=45, ha='right')
    ax2.axhline(y=3.26, color='red', linestyle='--', linewidth=1.5,
                label='Paper benchmark (3.26 ppm)')
    ax2.axhline(y=1.8, color='green', linestyle=':', linewidth=1.5,
                label='Clinical threshold (1.8 ppm)')
    ax2.legend(loc='upper right', fontsize=8)
    for bar, prob in zip(lod_bars, detection_probs):
        if prob is not None:
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() * 0.95,
                f"Pdet@LoD={prob*100:.1f}%",
                ha='center',
                va='top',
                fontsize=7,
                rotation=90,
            )
    
    # Add value labels
    for bar, val in zip(lod_bars, lods):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Multi-Gas Selectivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'Figure2_selectivity_comparison.png'
    artifact = save_figure(
        fig,
        save_path,
        sources=aggregate_sources(source_map, gases),
        description="Selectivity comparison with detection probabilities annotated",
    )
    return artifact


def figure3_roi_discovery(results: Dict[str, dict], source_map: Dict[str, str]) -> FigureArtifact:
    """
    Figure 3: ROI discovery showing optimal wavelength regions for each gas.
    """
    gases = DEFAULT_GASES
    present_gases = [g for g in gases if g in results]
    if not present_gases:
        raise RuntimeError("No scientific results found for ROI discovery figure")

    fig, ax = plt.subplots(figsize=(12, 6))

    lod_values = [results[g].get('calibration_wavelength_shift', {}).get('centroid', {}).get('lod_ppm', np.nan) for g in present_gases]
    lod_values = [v for v in lod_values if not np.isnan(v) and v is not None]
    vmin = min(lod_values) if lod_values else 0.1
    vmax = max(lod_values) if lod_values else 5.0
    cmap = cm.get_cmap('viridis_r')
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    y_positions = np.arange(len(present_gases))

    for idx, gas in enumerate(present_gases):
        data = results[gas]
        roi = data.get('roi_range', [0, 0])
        centroid = data.get('calibration_wavelength_shift', {}).get('centroid', {})
        lod = centroid.get('lod_ppm')
        color = cmap(norm(lod)) if lod is not None else GAS_COLORS.get(gas, 'gray')
        r2 = centroid.get('r2', np.nan)

        width = roi[1] - roi[0]
        ax.barh(idx, width, left=roi[0], height=0.5, color=color, edgecolor='black', linewidth=1)
        ax.text(roi[1] + 5, idx, f'R²={r2:.3f}\nLoD={lod:.2f} ppm', va='center', fontsize=8)

        # Overlay top candidate ROIs as semi-transparent line segments
        top_candidates = data.get('roi_scan', {}).get('top_10_candidates', [])
        for cand in top_candidates[:5]:
            cand_range = cand.get('roi_range', [])
            cand_lod = cand.get('lod_ppm')
            cand_color = cmap(norm(cand_lod)) if cand_lod else '#888888'
            ax.plot([cand_range[0], cand_range[1]], [idx + 0.25, idx + 0.25],
                    color=cand_color, linewidth=4, alpha=0.35)

    ax.axvspan(675, 689, alpha=0.15, color='red', label='Paper ROI (675–689 nm)')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(present_gases)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Gas Species', fontsize=11)
    ax.set_title('ROI Discovery Landscape (color ∝ LoD)', fontweight='bold', fontsize=12)
    ax.set_xlim(500, 920)
    ax.grid(True, alpha=0.3, axis='x')
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Detection Limit (ppm)', fontsize=10)

    plt.tight_layout()

    save_path = FIGURES_DIR / 'Figure3_roi_discovery.png'
    artifact = save_figure(
        fig,
        save_path,
        sources=aggregate_sources(source_map, present_gases),
        description='ROI discovery summary with LoD-coded bars and top candidate overlays',
    )
    return artifact


def figure4_performance_summary(results: Dict[str, dict], source_map: Dict[str, str]) -> FigureArtifact:
    """
    Figure 4: Comprehensive performance summary table as a figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    gases = DEFAULT_GASES
    headers = ['Gas', 'ROI (nm)', 'Sensitivity\n(nm/ppm)', 'R²', 'LoD\n(ppm)', 'Spearman ρ', 'LOOCV R²']
    
    table_data = []
    highlight_row_idx = None
    for gas in gases:
        if gas in results:
            cal = results[gas].get('calibration_wavelength_shift', {}).get('centroid', {})
            roi = results[gas].get('roi_range', [0, 0])
            loocv = results[gas].get('loocv_validation', {}).get('wavelength_shift', {}).get('r2_cv', 'N/A')
            
            row = [
                gas,
                f'{roi[0]:.0f}-{roi[1]:.0f}',
                f'{abs(cal.get("slope", 0)):.4f}',
                f'{cal.get("r2", 0):.4f}',
                f'{cal.get("lod_ppm", 0):.2f}',
                f'{cal.get("spearman_r", 0):.2f}',
                f'{loocv:.4f}' if isinstance(loocv, float) else loocv
            ]
            if gas == 'Acetone':
                highlight_row_idx = len(table_data) + 1  # +1 for header row in table indexing
            table_data.append(row)
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#E8E8E8'] * len(headers)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    if highlight_row_idx is not None:
        for j in range(len(headers)):
            table[(highlight_row_idx, j)].set_facecolor('#FFE4E4')
    
    plt.title('Table: Comprehensive Multi-Gas Sensing Performance', 
              fontsize=14, fontweight='bold', pad=20)
    
    save_path = FIGURES_DIR / 'Figure4_performance_table.png'
    artifact = save_figure(
        fig,
        save_path,
        sources=aggregate_sources(source_map, gases),
        description='Performance summary table covering ROI, sensitivity, R², LoD, Spearman, LOOCV',
    )

    # Export machine-readable CSV
    csv_path = FIGURES_DIR / 'Figure4_performance_table.csv'
    df = pd.DataFrame(table_data, columns=headers)
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path}")

    return artifact


def figure5_improvement_comparison(world_class_results: Dict, source_path: Optional[str]) -> Optional[FigureArtifact]:
    """Figure 5: Standard vs ML-enhanced comparison from world-class analysis."""
    if not world_class_results:
        print("[SKIP] No world-class results for Figure 5")
        return None

    gases = list(world_class_results.keys())
    if not gases:
        print("[SKIP] Empty world-class results set")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    std_sens, ml_sens, std_r2, ml_r2, std_lod, ml_lod = [], [], [], [], [], []
    lod_improvements = []

    for gas in gases:
        data = world_class_results.get(gas, {})
        std_cal = data.get('standard', {}).get('calibration', {})
        ml_cal = data.get('enhanced', {}).get('calibration', {})

        std_s = std_cal.get('sensitivity_nm_per_ppm', 0)
        ml_s = ml_cal.get('sensitivity_nm_per_ppm', 0)
        std_l = std_cal.get('lod_ppm', np.nan)
        ml_l = ml_cal.get('lod_ppm', np.nan)

        std_sens.append(std_s)
        ml_sens.append(ml_s)
        std_r2.append(std_cal.get('r_squared', 0))
        ml_r2.append(ml_cal.get('r_squared', 0))
        std_lod.append(min(std_l, 50) if std_l is not None else np.nan)
        ml_lod.append(min(ml_l, 50) if ml_l is not None else np.nan)
        if std_l and ml_l:
            lod_improvements.append((std_l - ml_l) / std_l)
        else:
            lod_improvements.append(None)

    x = np.arange(len(gases))
    width = 0.35

    def annotate_percent(ax, std_vals, ml_vals, ylabel: str, unit: str = ""):
        for idx, (std_val, ml_val) in enumerate(zip(std_vals, ml_vals)):
            if std_val in (0, None) or ml_val in (0, None):
                continue
            delta = ((ml_val - std_val) / std_val) * 100 if std_val else 0
            ax.text(idx, max(std_val, ml_val) * 1.02, f"Δ={delta:+.1f}%{unit}",
                    ha='center', va='bottom', fontsize=8)

    ax1 = axes[0]
    ax1.bar(x - width/2, std_sens, width, label='Standard', color='#3498DB', edgecolor='black')
    ax1.bar(x + width/2, ml_sens, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax1.set_ylabel('Sensitivity (nm/ppm)')
    ax1.set_title('(a) Sensitivity', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gases, rotation=45, ha='right')
    ax1.legend()
    annotate_percent(ax1, std_sens, ml_sens, 'Sensitivity')

    ax2 = axes[1]
    ax2.bar(x - width/2, std_r2, width, label='Standard', color='#3498DB', edgecolor='black')
    ax2.bar(x + width/2, ml_r2, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax2.set_ylabel('R²')
    ax2.set_title('(b) Coefficient of Determination', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gases, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    annotate_percent(ax2, std_r2, ml_r2, 'R²')

    ax3 = axes[2]
    ax3.bar(x - width/2, std_lod, width, label='Standard', color='#3498DB', edgecolor='black')
    ax3.bar(x + width/2, ml_lod, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax3.set_ylabel('Detection Limit (ppm)')
    ax3.set_title('(c) Detection Limit', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gases, rotation=45, ha='right')
    ax3.axhline(y=3.26, color='green', linestyle='--', linewidth=1.5, label='Paper benchmark')
    ax3.legend()
    for idx, imp in enumerate(lod_improvements):
        if imp is not None:
            ax3.text(idx, min(std_lod[idx], ml_lod[idx]) * 0.9, f"ΔLoD={imp*100:+.1f}%",
                     ha='center', va='top', fontsize=8)

    plt.suptitle('Standard vs ML-Enhanced Analysis Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = FIGURES_DIR / 'Figure5_ml_comparison.png'
    artifact = save_figure(
        fig,
        save_path,
        sources=[source_path] if source_path else [],
        description='Standard vs ML-enhanced comparison across sensitivity, R², and LoD',
    )
    return artifact


def generate_summary_markdown(results, world_class_results):
    """Generate a markdown summary of all figures."""
    md = f"""# Publication Figures Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Figures Generated

### Figure 1: Multi-Gas Calibration Curves
- 6-panel grid showing Δλ vs concentration for all gases
- Linear fits with R² and LoD annotations
- Acetone highlighted as primary target

### Figure 2: Selectivity + Detectability
- Bar charts comparing sensitivity and LoD across gases
- Detection probability at the computed LoD annotated for each gas
- Reference lines for paper benchmark and clinical threshold

### Figure 3: ROI Discovery Landscape
- LoD-colored horizontal bars showing the optimal ROI per gas
- Top-ranked candidate ROIs (semi-transparent overlays) reveal nearby solutions
- Paper ROI (675–689 nm) highlighted for comparison

### Figure 4: Performance Summary Table
- Comprehensive metrics + CSV export for reproducibility
- Includes sensitivity, R², LoD, Spearman ρ, LOOCV R²

### Figure 5: ML Enhancement Comparison
- Standard vs ML-enhanced analysis (Sensitivity, R², LoD)
- Percentage change annotations highlight gains/losses

## Key Results Summary

| Gas | ROI (nm) | Sensitivity | R² | LoD (ppm) |
|-----|----------|-------------|-----|-----------|
"""
    
    for gas, data in results.items():
        cal = data.get('calibration_wavelength_shift', {}).get('centroid', {})
        roi = data.get('roi_range', [0, 0])
        md += f"| {gas} | {roi[0]:.0f}-{roi[1]:.0f} | {abs(cal.get('slope', 0)):.4f} | {cal.get('r2', 0):.4f} | {cal.get('lod_ppm', 0):.2f} |\n"
    
    md += """
## Files Generated

All figures saved in both PNG (300 DPI) and PDF formats:
- `Figure1_multigas_calibration.png/pdf`
- `Figure2_selectivity_comparison.png/pdf`
- `Figure3_roi_discovery.png/pdf`
- `Figure4_performance_table.png/pdf`
- `Figure5_ml_comparison.png/pdf`

## Usage in Manuscript

These figures are designed for direct insertion into the manuscript draft.
Recommended figure sizes:
- Single column: 8.5 cm width
- Double column: 17.5 cm width
- Full page: 17.5 x 23 cm
"""
    
    summary_path = FIGURES_DIR / 'figures_summary.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"[SAVED] {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures from scientific outputs")
    parser.add_argument('--results-root', default=str(DEFAULT_RESULTS_ROOT), help='Path to output/scientific directory')
    parser.add_argument('--world-class-path', default=str(DEFAULT_WORLD_CLASS_PATH), help='Path to world-class JSON results')
    parser.add_argument('--output-dir', default=str(DEFAULT_FIGURES_DIR), help='Directory to store publication figures')
    parser.add_argument('--gases', nargs='*', default=DEFAULT_GASES, help='Subset of gases to include')
    parser.add_argument('--manifest-path', default=None, help='Optional manifest JSON output path')
    return parser.parse_args()


def main():
    args = parse_args()

    global FIGURES_DIR
    FIGURES_DIR = Path(args.output_dir).resolve()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gases = args.gases if args.gases else DEFAULT_GASES

    print("\n" + "="*60)
    print(" PUBLICATION FIGURE GENERATION")
    print("="*60)
    print(f"Output dir: {FIGURES_DIR}")

    # Load results
    print("\nLoading results...")
    scientific_results, source_map = load_scientific_results(Path(args.results_root), gases)
    world_class_results, world_class_source = load_world_class_results(Path(args.world_class_path))

    print(f"Loaded {len(scientific_results)} scientific results")
    print(f"Loaded {len(world_class_results)} world-class results")

    # Generate figures
    print("\nGenerating figures...")
    artifacts: List[FigureArtifact] = []

    artifacts.append(figure1_multigas_calibration(scientific_results, source_map))
    artifacts.append(figure2_selectivity_comparison(scientific_results, source_map))
    artifacts.append(figure3_roi_discovery(scientific_results, source_map))
    artifacts.append(figure4_performance_summary(scientific_results, source_map))
    fig5 = figure5_improvement_comparison(world_class_results, world_class_source)
    if fig5:
        artifacts.append(fig5)

    # Generate summary + manifest
    generate_summary_markdown(scientific_results, world_class_results)
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else FIGURES_DIR / 'publication_figures_manifest.json'
    write_manifest(artifacts, manifest_path)

    print("\n" + "="*60)
    print(" FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
