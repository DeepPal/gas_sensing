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

import json
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
OUTPUT_DIR = PROJECT_ROOT / 'output'
FIGURES_DIR = OUTPUT_DIR / 'publication_figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Gas colors for consistent styling
GAS_COLORS = {
    'Acetone': '#E74C3C',      # Red - primary target
    'Methanol': '#3498DB',     # Blue
    'Ethanol': '#2ECC71',      # Green
    'Isopropanol': '#9B59B6',  # Purple
    'Toluene': '#F39C12',      # Orange
    'Xylene': '#1ABC9C',       # Teal
}

def load_scientific_results():
    """Load all scientific pipeline results."""
    results = {}
    gases = ['acetone', 'ethanol', 'methanol', 'isopropanol', 'toluene', 'xylene']
    
    for gas in gases:
        metrics_path = OUTPUT_DIR / f'{gas}_scientific' / 'metrics' / 'calibration_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                results[gas.capitalize()] = json.load(f)
    
    return results


def load_world_class_results():
    """Load world-class analysis results."""
    results_path = OUTPUT_DIR / 'world_class_analysis' / 'comprehensive_results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def figure1_multigas_calibration(results):
    """
    Figure 1: Multi-gas calibration curves (6-panel grid)
    Shows Δλ vs concentration for all gases with linear fits.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    gases = ['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 'Toluene', 'Xylene']
    
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
        
        # Plot data points
        ax.scatter(concs, delta_lambda, s=80, c=GAS_COLORS.get(gas, 'gray'), 
                   edgecolors='black', linewidth=1, zorder=5, label='Data')
        
        # Plot fit line
        x_fit = np.linspace(0, max(concs) * 1.1, 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color=GAS_COLORS.get(gas, 'gray'), 
                linewidth=2, alpha=0.8, label='Linear fit')
        
        # Add equation and metrics
        sign = '+' if intercept >= 0 else ''
        eq_text = f'Δλ = {slope:.4f}C {sign}{intercept:.4f}'
        metrics_text = f'R² = {r2:.4f}\nLoD = {lod:.2f} ppm'
        
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")
    return save_path


def figure2_selectivity_comparison(results):
    """
    Figure 2: Selectivity bar chart comparing sensitivity and LoD across gases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gases = ['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 'Toluene', 'Xylene']
    sensitivities = []
    lods = []
    r2_values = []
    colors = []
    
    for gas in gases:
        if gas in results:
            cal = results[gas].get('calibration_wavelength_shift', {}).get('centroid', {})
            sensitivities.append(abs(cal.get('slope', 0)))
            lods.append(cal.get('lod_ppm', 0))
            r2_values.append(cal.get('r2', 0))
            colors.append(GAS_COLORS.get(gas, 'gray'))
        else:
            sensitivities.append(0)
            lods.append(0)
            r2_values.append(0)
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
    bars2 = ax2.bar(x, lods, width, color=colors, edgecolor='black', linewidth=1)
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
    
    # Add value labels
    for bar, val in zip(bars2, lods):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Multi-Gas Selectivity Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'Figure2_selectivity_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")
    return save_path


def figure3_roi_discovery(results):
    """
    Figure 3: ROI discovery showing optimal wavelength regions for each gas.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gases = ['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 'Toluene', 'Xylene']
    
    y_positions = np.arange(len(gases))
    
    for idx, gas in enumerate(gases):
        if gas not in results:
            continue
        
        data = results[gas]
        roi = data.get('roi_range', [0, 0])
        r2 = data.get('calibration_wavelength_shift', {}).get('centroid', {}).get('r2', 0)
        
        # Draw ROI bar
        width = roi[1] - roi[0]
        bar = ax.barh(idx, width, left=roi[0], height=0.6, 
                      color=GAS_COLORS.get(gas, 'gray'), alpha=0.8,
                      edgecolor='black', linewidth=1)
        
        # Add R² annotation
        ax.text(roi[1] + 5, idx, f'R² = {r2:.4f}', va='center', fontsize=9)
    
    # Add reference paper ROI
    ax.axvspan(675, 689, alpha=0.2, color='red', label='Paper ROI (675-689 nm)')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(gases)
    ax.set_xlabel('Wavelength (nm)', fontsize=11)
    ax.set_ylabel('Gas Species', fontsize=11)
    ax.set_title('Optimal ROI Discovery for Each Gas', fontweight='bold', fontsize=12)
    ax.set_xlim(500, 900)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'Figure3_roi_discovery.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")
    return save_path


def figure4_performance_summary(results):
    """
    Figure 4: Comprehensive performance summary table as a figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Prepare table data
    gases = ['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 'Toluene', 'Xylene']
    headers = ['Gas', 'ROI (nm)', 'Sensitivity\n(nm/ppm)', 'R²', 'LoD\n(ppm)', 'Spearman ρ', 'LOOCV R²']
    
    table_data = []
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
    
    # Highlight acetone row
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#FFE4E4')
    
    plt.title('Table: Comprehensive Multi-Gas Sensing Performance', 
              fontsize=14, fontweight='bold', pad=20)
    
    save_path = FIGURES_DIR / 'Figure4_performance_table.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")
    return save_path


def figure5_improvement_comparison(world_class_results):
    """
    Figure 5: Standard vs ML-enhanced comparison from world-class analysis.
    """
    if not world_class_results:
        print("[SKIP] No world-class results for Figure 5")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    gases = list(world_class_results.keys())
    
    std_sens = []
    ml_sens = []
    std_r2 = []
    ml_r2 = []
    std_lod = []
    ml_lod = []
    
    for gas in gases:
        data = world_class_results[gas]
        std_cal = data.get('standard', {}).get('calibration', {})
        ml_cal = data.get('enhanced', {}).get('calibration', {})
        
        std_sens.append(std_cal.get('sensitivity_nm_per_ppm', 0))
        ml_sens.append(ml_cal.get('sensitivity_nm_per_ppm', 0))
        std_r2.append(std_cal.get('r_squared', 0))
        ml_r2.append(ml_cal.get('r_squared', 0))
        std_lod.append(min(std_cal.get('lod_ppm', 100), 50))  # Cap for visualization
        ml_lod.append(min(ml_cal.get('lod_ppm', 100), 50))
    
    x = np.arange(len(gases))
    width = 0.35
    
    # Panel A: Sensitivity
    ax1 = axes[0]
    ax1.bar(x - width/2, std_sens, width, label='Standard', color='#3498DB', edgecolor='black')
    ax1.bar(x + width/2, ml_sens, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax1.set_ylabel('Sensitivity (nm/ppm)')
    ax1.set_title('(a) Sensitivity', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(gases, rotation=45, ha='right')
    ax1.legend()
    
    # Panel B: R²
    ax2 = axes[1]
    ax2.bar(x - width/2, std_r2, width, label='Standard', color='#3498DB', edgecolor='black')
    ax2.bar(x + width/2, ml_r2, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax2.set_ylabel('R²')
    ax2.set_title('(b) Coefficient of Determination', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(gases, rotation=45, ha='right')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    
    # Panel C: LoD
    ax3 = axes[2]
    ax3.bar(x - width/2, std_lod, width, label='Standard', color='#3498DB', edgecolor='black')
    ax3.bar(x + width/2, ml_lod, width, label='ML-Enhanced', color='#E74C3C', edgecolor='black')
    ax3.set_ylabel('Detection Limit (ppm)')
    ax3.set_title('(c) Detection Limit', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(gases, rotation=45, ha='right')
    ax3.axhline(y=3.26, color='green', linestyle='--', linewidth=1.5, label='Paper benchmark')
    ax3.legend()
    
    plt.suptitle('Standard vs ML-Enhanced Analysis Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / 'Figure5_ml_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[SAVED] {save_path}")
    return save_path


def generate_summary_markdown(results, world_class_results):
    """Generate a markdown summary of all figures."""
    md = f"""# Publication Figures Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Figures Generated

### Figure 1: Multi-Gas Calibration Curves
- 6-panel grid showing Δλ vs concentration for all gases
- Linear fits with R² and LoD annotations
- Acetone highlighted as primary target

### Figure 2: Selectivity Comparison
- Bar charts comparing sensitivity and LoD across gases
- Reference lines for paper benchmark and clinical threshold

### Figure 3: ROI Discovery
- Horizontal bar chart showing optimal wavelength regions
- Paper ROI (675-689 nm) highlighted for comparison

### Figure 4: Performance Summary Table
- Comprehensive metrics for all gases
- Includes sensitivity, R², LoD, Spearman ρ, LOOCV R²

### Figure 5: ML Enhancement Comparison
- Standard vs ML-enhanced analysis
- Three panels: Sensitivity, R², Detection Limit

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


def main():
    print("\n" + "="*60)
    print(" PUBLICATION FIGURE GENERATION")
    print("="*60)
    
    # Load results
    print("\nLoading results...")
    scientific_results = load_scientific_results()
    world_class_results = load_world_class_results()
    
    print(f"Loaded {len(scientific_results)} scientific results")
    print(f"Loaded {len(world_class_results)} world-class results")
    
    # Generate figures
    print("\nGenerating figures...")
    
    figure1_multigas_calibration(scientific_results)
    figure2_selectivity_comparison(scientific_results)
    figure3_roi_discovery(scientific_results)
    figure4_performance_summary(scientific_results)
    figure5_improvement_comparison(world_class_results)
    
    # Generate summary
    generate_summary_markdown(scientific_results, world_class_results)
    
    print("\n" + "="*60)
    print(" FIGURE GENERATION COMPLETE")
    print("="*60)
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("\nGenerated files:")
    for f in sorted(FIGURES_DIR.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
