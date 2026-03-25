"""
Research-Grade Report Generator for Gas Sensing Calibration
Generates comprehensive, publication-quality reports with methodology explanation.
"""

from datetime import datetime
import json
import os
from typing import Any

import numpy as np
from scipy import stats


def generate_methodology_markdown(out_root: str, dataset_label: str, calib_result: dict) -> str:
    """Generate detailed methodology documentation in Markdown format."""

    if calib_result is None:
        return None

    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    md_path = os.path.join(reports_dir, "methodology_report.md")

    r2 = calib_result.get("r2", 0) or 0
    slope = calib_result.get("slope", 0) or 0
    lod = calib_result.get("lod", 0) or 0
    loq = calib_result.get("loq", 0) or 0
    rmse = calib_result.get("rmse", 0) or 0
    selected_feature = calib_result.get("selected_feature") or {}
    feature_type = (
        selected_feature.get("feature_type", "unknown") if selected_feature else "unknown"
    )
    feature_center = selected_feature.get("center_wavelength", 0) if selected_feature else 0

    content = f"""# {dataset_label} - Gas Sensing Calibration Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Executive Summary

This report presents the calibration analysis for **{dataset_label}** gas detection using
wavelength-shift spectroscopy. The analysis achieved:

| Metric | Value | Rating |
|--------|-------|--------|
| R² | {r2:.4f} | {"Excellent" if r2 >= 0.99 else "Very Good" if r2 >= 0.95 else "Good" if r2 >= 0.90 else "Acceptable"} |
| Sensitivity | {abs(slope) * 1000:.2f} pm/ppm | - |
| LOD | {lod:.2f} ppm | {"Excellent" if lod < 1 else "Good" if lod < 5 else "Acceptable"} |
| LOQ | {loq:.2f} ppm | - |
| RMSE | {rmse * 1000:.2f} pm | - |

---

## 2. Methodology

### 2.1 Theoretical Background

#### Beer-Lambert Law
The fundamental principle governing optical gas sensing:

```
A = ε × l × c
```

Where:
- **A** = Absorbance (dimensionless)
- **ε** = Molar absorptivity (L·mol⁻¹·cm⁻¹)
- **l** = Optical path length (cm)
- **c** = Concentration (mol·L⁻¹)

#### Wavelength Shift Mechanism
Gas molecules interact with the sensing layer, modifying the local refractive index:

```
Δn → Δλ_resonance → Calibration curve
```

The wavelength shift (Δλ) is proportional to concentration within the linear range.

### 2.2 Signal Processing Pipeline

```
Raw Spectrum → Baseline Correction → Transmittance → Absorbance → Feature Detection → Calibration
```

#### Step 1: Baseline Correction
- Reference spectrum acquired in clean air/N₂
- Sample spectrum normalized: T = I_sample / I_reference

#### Step 2: Absorbance Conversion
```
A = -log₁₀(T)
```

**Rationale:** Absorbance is directly proportional to concentration (Beer-Lambert law),
making it the preferred signal type for quantitative analysis.

#### Step 3: Region of Interest (ROI) Selection
- Scan wavelength range for responsive regions
- Criteria: Monotonic response, high R², sufficient SNR

#### Step 4: Feature Detection

**Selected Feature Type:** {feature_type.upper()}
**Center Wavelength:** {feature_center:.2f} nm

**Why {feature_type.upper()} for Absorbance Data:**
{"- Valleys represent transmission windows between absorption bands" if feature_type == "valley" else "- Peaks represent absorption maxima"}
{"- More stable baseline at minima" if feature_type == "valley" else "- Strong signal at absorption bands"}
{"- Less susceptible to saturation effects" if feature_type == "valley" else "- Clear spectral signature"}
- Consistent shift direction with concentration changes

#### Step 5: Weighted Centroid Calculation
Sub-pixel feature position estimation:

```
λ_centroid = Σ(λᵢ × wᵢ) / Σ(wᵢ)
```

For valleys: w = (max_signal - signal_i)
For peaks: w = (signal_i - min_signal)

---

## 3. Calibration Model

### 3.1 Model Selection
**Selected Model:** {calib_result.get("selected_model", "linear")}

### 3.2 Calibration Equation
```
λ = {slope:.6f} × C + {calib_result.get("intercept", 0):.4f}
```

Where:
- λ = Feature wavelength (nm)
- C = Gas concentration (ppm)

### 3.3 Model Validation
- **R²:** {r2:.4f} (coefficient of determination)
- **RMSE:** {rmse * 1000:.2f} pm (root mean square error)
- **Residual Analysis:** See plots for residual distribution

---

## 4. Performance Metrics

### 4.1 Limit of Detection (LOD)
```
LOD = 3.3 × σ / S = {lod:.2f} ppm
```
Where σ = standard deviation of blank, S = sensitivity (slope)

### 4.2 Limit of Quantification (LOQ)
```
LOQ = 10 × σ / S = {loq:.2f} ppm
```

### 4.3 Sensitivity
```
S = dλ/dC = {slope * 1000:.2f} pm/ppm
```

### 4.4 Dynamic Range
- Minimum tested: {min(calib_result.get("concentrations", [0])):.1f} ppm
- Maximum tested: {max(calib_result.get("concentrations", [0])):.1f} ppm

---

## 5. Quality Assessment

### 5.1 Correlation Quality
| Criterion | Threshold | Achieved | Status |
|-----------|-----------|----------|--------|
| R² | > 0.95 | {r2:.4f} | {"✓ PASS" if r2 > 0.95 else "○ MARGINAL" if r2 > 0.90 else "✗ FAIL"} |
| Monotonicity | Spearman ρ > 0.9 | - | - |
| Residual normality | p > 0.05 | - | - |

### 5.2 Recommendations
{"- Excellent calibration quality. Results are publication-ready." if r2 >= 0.95 else "- Consider expanding ROI search or adjusting parameters for improved correlation."}
- Include error bars and confidence intervals in final publication
- Perform cross-validation to demonstrate model robustness
- Report measurement uncertainty for all metrics

---

## 6. References

1. Beer, A. (1852). "Bestimmung der Absorption des rothen Lichts in farbigen Flüssigkeiten"
2. IUPAC Guidelines for Calibration in Analytical Chemistry
3. ISO 11843 - Capability of Detection

---

*Report generated by Gas Sensing Analysis Pipeline v2.0*
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(content)

    return md_path


def generate_analysis_json(
    out_root: str, dataset_label: str, calib_result: dict, canonical: dict[float, Any]
) -> str:
    """Generate comprehensive JSON analysis report."""

    if calib_result is None:
        return None

    reports_dir = os.path.join(out_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "analysis_summary.json")

    concentrations = np.array(calib_result.get("concentrations", []))
    peak_wavelengths = np.array(calib_result.get("peak_wavelengths", []))

    # Calculate additional statistics
    if len(concentrations) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            concentrations, peak_wavelengths
        )
        predictions = slope * concentrations + intercept
        residuals = peak_wavelengths - predictions
    else:
        slope = intercept = p_value = std_err = 0
        predictions = residuals = np.array([])

    report = {
        "metadata": {
            "dataset": dataset_label,
            "generated_at": datetime.now().isoformat(),
            "pipeline_version": "2.0",
        },
        "calibration": {
            "model": calib_result.get("selected_model", "linear"),
            "equation": f"λ = {slope:.6f} × C + {intercept:.4f}",
            "parameters": {
                "slope_nm_per_ppm": float(slope),
                "slope_pm_per_ppm": float(slope * 1000),
                "intercept_nm": float(intercept),
                "slope_se": float(std_err),
                "p_value": float(p_value),
            },
            "quality": {
                "r_squared": float(calib_result.get("r2", 0)),
                "rmse_nm": float(calib_result.get("rmse", 0)),
                "rmse_pm": float(calib_result.get("rmse", 0) * 1000),
            },
        },
        "detection_limits": {
            "lod_ppm": float(calib_result.get("lod", 0)),
            "loq_ppm": float(calib_result.get("loq", 0)),
            "lod_calculation": "3.3 × σ / S",
            "loq_calculation": "10 × σ / S",
        },
        "feature_selection": {
            "type": (calib_result.get("selected_feature") or {}).get("feature_type", "unknown"),
            "center_wavelength_nm": (calib_result.get("selected_feature") or {}).get(
                "center_wavelength", 0
            ),
            "rationale": "Valley preferred for absorbance data (transmission window)",
        },
        "data_summary": {
            "n_points": len(concentrations),
            "concentration_range_ppm": [
                float(min(concentrations)),
                float(max(concentrations)),
            ]
            if len(concentrations) > 0
            else [0, 0],
            "wavelength_shift_nm": float(max(peak_wavelengths) - min(peak_wavelengths))
            if len(peak_wavelengths) > 0
            else 0,
            "wavelength_shift_pm": float((max(peak_wavelengths) - min(peak_wavelengths)) * 1000)
            if len(peak_wavelengths) > 0
            else 0,
        },
        "residual_analysis": {
            "mean_pm": float(np.mean(residuals) * 1000) if len(residuals) > 0 else 0,
            "std_pm": float(np.std(residuals) * 1000) if len(residuals) > 0 else 0,
            "max_abs_pm": float(np.max(np.abs(residuals)) * 1000) if len(residuals) > 0 else 0,
        },
    }

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    return json_path
