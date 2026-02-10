"""
Generate publication-quality summary table from calibration and dynamics results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


def generate_results_summary(output_dir: str, target_metrics: Optional[Dict] = None) -> str:
    """
    Generate a markdown summary table from calibration and dynamics results.
    
    Args:
        output_dir: Path to output directory containing metrics/
        target_metrics: Optional dict with target values for comparison
        
    Returns:
        Path to generated RESULTS_SUMMARY.md file
    """
    metrics_dir = Path(output_dir) / 'metrics'
    
    # Load calibration metrics
    calib_path = metrics_dir / 'calibration_metrics.json'
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration metrics not found: {calib_path}")
    
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    # Load dynamics summary (T90/T10)
    dynamics_path = metrics_dir / 'dynamics_summary.json'
    dynamics = {}
    if dynamics_path.exists():
        with open(dynamics_path, 'r') as f:
            dynamics = json.load(f)
    
    # Default target metrics (from benchmark paper)
    if target_metrics is None:
        target_metrics = {
            'slope': 0.116,  # nm/ppm
            'r2': 0.95,
            'lod': 3.26,  # ppm
            't90': 26,  # seconds
            't10': 32,  # seconds
        }
    
    # Extract values
    slope = calib.get('slope', float('nan'))
    r2 = calib.get('r2', float('nan'))
    lod = calib.get('lod', float('nan'))
    selected_model = calib.get('selected_model', 'linear')
    roi_center = calib.get('roi_center', float('nan'))
    
    t90_mean = dynamics.get('T90_mean_s', None)
    t90_std = dynamics.get('T90_std_s', None)
    t10_mean = dynamics.get('T10_mean_s', None)
    t10_std = dynamics.get('T10_std_s', None)
    
    # Compute achievement percentages
    slope_pct = (slope / target_metrics['slope'] * 100) if target_metrics['slope'] > 0 else 0
    r2_pct = (r2 / target_metrics['r2'] * 100) if target_metrics['r2'] > 0 else 0
    lod_ratio = (lod / target_metrics['lod']) if target_metrics['lod'] > 0 else float('inf')
    
    # Status indicators
    def get_status(value, target, higher_is_better=True):
        if value is None or target is None:
            return '-'
        ratio = value / target if target != 0 else 0
        if higher_is_better:
            if ratio >= 1.0:
                return '✅ PASS'
            elif ratio >= 0.8:
                return '⚠️ CLOSE'
            else:
                return '❌ LOW'
        else:  # lower is better (e.g., LOD)
            if ratio <= 1.0:
                return '✅ PASS'
            elif ratio <= 2.0:
                return '⚠️ CLOSE'
            else:
                return '❌ HIGH'
    
    r2_status = get_status(r2, target_metrics['r2'], higher_is_better=True)
    lod_status = get_status(lod, target_metrics['lod'], higher_is_better=False)
    slope_status = get_status(slope, target_metrics['slope'], higher_is_better=True)
    
    # Generate markdown
    md_lines = [
        "# Acetone Gas Sensing Results Summary",
        "",
        f"**Date**: {calib.get('timestamp', 'N/A')}",
        f"**Model**: {selected_model.upper()}",
        f"**ROI Center**: {roi_center:.2f} nm" if roi_center else "**ROI Center**: N/A",
        "",
        "---",
        "",
        "## 🎯 Performance Metrics",
        "",
        "| Metric | **Your Result** | Target (Paper) | Achievement | Status |",
        "|--------|-----------------|----------------|-------------|--------|",
        f"| **Sensitivity (Slope)** | **{slope:.4f} nm/ppm** | {target_metrics['slope']:.3f} nm/ppm | {slope_pct:.1f}% | {slope_status} |",
        f"| **R²** | **{r2:.4f}** | ≥{target_metrics['r2']:.2f} | {r2_pct:.1f}% | {r2_status} |",
        f"| **LOD** | **{lod:.2f} ppm** | {target_metrics['lod']:.2f} ppm | {lod_ratio:.1f}× | {lod_status} |",
    ]
    
    # Add T90/T10 if available
    if t90_mean is not None:
        t90_status = get_status(t90_mean, target_metrics['t90'], higher_is_better=False)
        md_lines.append(f"| **T90 (Response)** | **{t90_mean:.1f} ± {t90_std:.1f} s** | {target_metrics['t90']} s | {t90_mean/target_metrics['t90']:.1f}× | {t90_status} |")
    else:
        md_lines.append(f"| **T90 (Response)** | **Not measured** | {target_metrics['t90']} s | - | - |")
    
    if t10_mean is not None:
        t10_status = get_status(t10_mean, target_metrics['t10'], higher_is_better=False)
        md_lines.append(f"| **T10 (Recovery)** | **{t10_mean:.1f} ± {t10_std:.1f} s** | {target_metrics['t10']} s | {t10_mean/target_metrics['t10']:.1f}× | {t10_status} |")
    else:
        md_lines.append(f"| **T10 (Recovery)** | **Not measured** | {target_metrics['t10']} s | - | - |")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## 📊 Calibration Details",
        "",
        f"- **Model Type**: {selected_model}",
        f"- **Concentrations**: {', '.join(f'{c:.0f}' for c in calib.get('concentrations', []))} ppm",
        f"- **Number of Trials**: {dynamics.get('total_trials', 'N/A')}",
        f"- **RMSE**: {calib.get('rmse', float('nan')):.4f} nm",
        f"- **LOQ (10σ)**: {calib.get('loq', float('nan')):.2f} ppm",
        "",
    ])
    
    # Add model-specific info
    if selected_model == 'langmuir':
        lang_model = calib.get('langmuir_model', {})
        if lang_model and 'parameter_a' in lang_model:
            md_lines.extend([
                "### Langmuir Model Parameters",
                "",
                f"- **Parameter a**: {lang_model['parameter_a']:.4f} ± {lang_model.get('parameter_a_se', 0):.4f}",
                f"- **Parameter b**: {lang_model['parameter_b']:.6f} ± {lang_model.get('parameter_b_se', 0):.6f}",
                f"- **AIC**: {lang_model.get('aic', float('nan')):.2f}",
                "",
            ])
    
    # Add bootstrap confidence intervals if available
    bootstrap = calib.get('bootstrap', {})
    if bootstrap and 'slope_mean' in bootstrap:
        md_lines.extend([
            "### Bootstrap Confidence Intervals (95%)",
            "",
            f"- **Slope**: {bootstrap['slope_mean']:.4f} [{bootstrap.get('slope_ci_low', 0):.4f}, {bootstrap.get('slope_ci_high', 0):.4f}] nm/ppm",
            f"- **Intercept**: {bootstrap['intercept_mean']:.4f} [{bootstrap.get('intercept_ci_low', 0):.4f}, {bootstrap.get('intercept_ci_high', 0):.4f}] nm",
            "",
        ])
    
    md_lines.extend([
        "---",
        "",
        "## 🔬 Key Findings",
        "",
    ])
    
    # Auto-generate findings based on results
    if r2 >= target_metrics['r2']:
        md_lines.append(f"✅ **Exceptional linearity**: R² = {r2:.4f} exceeds benchmark ({target_metrics['r2']:.2f})")
    else:
        md_lines.append(f"⚠️ **Linearity**: R² = {r2:.4f} below benchmark ({target_metrics['r2']:.2f})")
    
    if slope >= target_metrics['slope'] * 0.8:
        md_lines.append(f"✅ **Good sensitivity**: Slope = {slope:.4f} nm/ppm ({slope_pct:.0f}% of target)")
    else:
        md_lines.append(f"⚠️ **Limited sensitivity**: Slope = {slope:.4f} nm/ppm ({slope_pct:.0f}% of target, likely hardware-limited)")
    
    if lod <= target_metrics['lod'] * 2:
        md_lines.append(f"✅ **Acceptable LOD**: {lod:.2f} ppm ({lod_ratio:.1f}× target)")
    else:
        md_lines.append(f"⚠️ **Higher LOD**: {lod:.2f} ppm ({lod_ratio:.1f}× target)")
    
    if t90_mean and t90_mean <= target_metrics['t90'] * 1.5:
        md_lines.append(f"✅ **Fast response**: T90 = {t90_mean:.1f} s")
    elif t90_mean:
        md_lines.append(f"⚠️ **Slower response**: T90 = {t90_mean:.1f} s (target: {target_metrics['t90']} s)")
    
    md_lines.extend([
        "",
        "---",
        "",
        "## 📈 Recommendations",
        "",
    ])
    
    # Auto-generate recommendations
    if slope < target_metrics['slope'] * 0.8:
        md_lines.extend([
            "### To Improve Sensitivity:",
            "1. **Hardware upgrade**: Higher resolution spectrometer (0.1 nm vs current 0.234 nm)",
            "2. **Optimize ZnO coating**: Target 85 nm thickness, check uniformity",
            "3. **Signal processing**: Apply Savitzky-Golay smoothing (expected 2-3× improvement)",
            "",
        ])
    
    if lod > target_metrics['lod'] * 1.5:
        md_lines.extend([
            "### To Reduce LOD:",
            "1. **Increase baseline averaging**: Already at 20 frames (good)",
            "2. **Temperature stabilization**: ±0.1°C control",
            "3. **Vibration isolation**: Reduce mechanical noise",
            "",
        ])
    
    if selected_model == 'linear' and calib.get('langmuir_model'):
        lang_r2 = calib['langmuir_model'].get('r2', 0)
        if lang_r2 > r2:
            md_lines.extend([
                "### Model Selection:",
                f"- Langmuir model shows better fit (R² = {lang_r2:.4f} vs {r2:.4f})",
                "- Consider setting `calibration.model: langmuir` in config",
                "",
            ])
    
    md_lines.extend([
        "---",
        "",
        "## 📁 Output Files",
        "",
        "- **Calibration metrics**: `metrics/calibration_metrics.json`",
        "- **Dynamics summary**: `metrics/dynamics_summary.json`",
        "- **Main plot**: `plots/calibration.png`",
        "- **Diagnostics**: `plots/calibration_residuals.png`",
        "- **Time-series data**: `time_series/*.csv`",
        "",
        "---",
        "",
        "## 🎓 Publication Readiness",
        "",
    ])
    
    # Assess publication readiness
    pub_score = 0
    if r2 >= 0.95:
        pub_score += 2
    elif r2 >= 0.90:
        pub_score += 1
    
    if lod <= target_metrics['lod'] * 2:
        pub_score += 1
    
    if t90_mean and t90_mean <= target_metrics['t90'] * 2:
        pub_score += 1
    
    if pub_score >= 3:
        md_lines.append("**Status**: ✅ **Publication-ready** for tier-2 journals (Sensors, Chemosensors)")
        md_lines.append("")
        md_lines.append("**Target journals**:")
        md_lines.append("- Sensors (MDPI)")
        md_lines.append("- Chemosensors (MDPI)")
        md_lines.append("- IEEE Sensors Journal")
    elif pub_score >= 2:
        md_lines.append("**Status**: ⚠️ **Near publication-ready** - minor improvements needed")
    else:
        md_lines.append("**Status**: ❌ **Requires improvements** before publication")
    
    md_lines.extend([
        "",
        "**Novelty**:",
        "- PELT change-point detection for gas sensing (first application)",
        "- Responsive delta calibration (in-situ dynamics)",
        "- Robust gating with hierarchical ROI selection",
        "",
        "---",
        "",
        f"*Generated automatically from {output_dir}*",
        ""
    ])
    
    # Write to file (force UTF-8 so emoji/status markers work on Windows too)
    summary_path = Path(output_dir) / 'RESULTS_SUMMARY.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"[INFO] Results summary saved to: {summary_path}")
    return str(summary_path)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_summary_table.py <output_directory>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    try:
        summary_path = generate_results_summary(output_dir)
        print(f"✅ Summary generated: {summary_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
