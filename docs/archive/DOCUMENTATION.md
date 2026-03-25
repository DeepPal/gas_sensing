# Spectral Gas Sensing Calibration System
## Complete Scientific and Technical Documentation

**Version**: 2.0.0 | **Updated**: 2025-11-25

---

## 1. Executive Summary

This system provides a complete pipeline for **optical gas sensing calibration** using spectral analysis. It processes UV-Vis-NIR spectral data from fiber optic sensors to establish quantitative relationships between gas concentration and spectral response.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Dual Calibration** | Wavelength shift (Δλ) and Absorbance amplitude (ΔA) |
| **Auto Feature Detection** | Monotonic trends, centroid, cross-correlation |
| **Statistical Validation** | LOOCV, Bootstrap CI, LOD/LOQ |
| **Response Dynamics** | T90/T10 response and recovery times |

### Performance (Typical)

| Metric | Value | Description |
|--------|-------|-------------|
| R² | 0.85-0.99 | Calibration linearity |
| LOD | 0.05-5 ppm | Limit of Detection |
| T90 | 5-10 s | Response time |
| T10 | 25-40 s | Recovery time |

---

## 2. Scientific Background

### 2.1 Beer-Lambert Law

```
A = ε × c × l

Where:
  A = Absorbance
  ε = Molar absorptivity (L·mol⁻¹·cm⁻¹)
  c = Concentration (mol·L⁻¹)
  l = Path length (cm)
```

### 2.2 Transmittance & Absorbance

```
T = I / I₀           (Transmittance)
A = -log₁₀(T)        (Absorbance)
```

### 2.3 Calibration Metrics

**Sensitivity**: `S = Δy / Δc`

**LOD**: `LOD = 3.3 × σ / S`

**LOQ**: `LOQ = 10 × σ / S`

**R²**: `R² = 1 - SS_res / SS_tot`

---

## 3. System Architecture

### 3.1 Directory Structure

```
Joy_Code_1/
├── config/config.yaml           # Configuration
├── gas_analysis/core/
│   ├── pipeline.py              # Main engine (10,308 lines)
│   ├── dynamics.py              # T90/T10 analysis
│   ├── run_each_gas.py          # CLI entry point
│   └── research_report.py       # Report generation
├── Joy_Data/                    # Raw data
└── output/                      # Results
```

### 3.2 Processing Pipeline

```
RAW DATA → PREPROCESSING → FRAME SELECTION → CANONICAL SPECTRA
    ↓
FEATURE DETECTION → CALIBRATION (Δλ + ΔA) → VALIDATION → OUTPUTS
```

**Stages**:
1. Data Loading & Preprocessing
2. Frame Selection & Averaging
3. Canonical Spectrum Generation
4. Feature Detection (monotonic, centroid, xcorr)
5. Dual Calibration (Δλ and ΔA methods)
6. Validation (LOOCV, Bootstrap CI)
7. Dynamics Analysis (T90/T10)
8. Output Generation (plots, JSON, reports)

---

## 4. Calibration Methods

### 4.1 Wavelength Shift (Δλ)

Tracks spectral feature position vs concentration:

```
Δλ = λ(C) - λ(C=0)
```

**Feature Types**: centroid, xcorr, gaussian, valley, monotonic_peak

### 4.2 Absorbance Amplitude (ΔA)

Measures absorbance intensity change at optimal wavelength:

```
ΔA = A(C, λ_opt) - A(C=0, λ_opt)
```

**Enhancement Methods**:
- `raw`: Direct absorbance
- `window_avg`: ±2 nm average
- `derivative`: First derivative (dA/dλ)
- `differential`: Reference subtraction

### 4.3 Method Comparison

Pipeline automatically compares both methods and recommends the best based on R²_CV.

---

## 5. Validation

### 5.1 LOOCV (Leave-One-Out Cross-Validation)

```
R²_CV = 1 - Σ(yᵢ - ŷᵢ_CV)² / Σ(yᵢ - ȳ)²
```

### 5.2 Bootstrap Confidence Intervals

1000 resamples → 95% CI from percentiles

### 5.3 Monotonicity (Spearman ρ)

```
|ρ| > 0.9: Strong monotonic
|ρ| < 0.7: Flagged for review
```

---

## 6. Output Files

### 6.1 Directory Structure

```
output/{gas}_topavg/
├── plots/
│   ├── calibration_research_grade.png
│   ├── wavelength_shift_visualization.png
│   ├── absorbance_amplitude_calibration.png
│   └── method_comparison.png
├── metrics/
│   ├── calibration_metrics.json
│   ├── dynamics_summary.json
│   └── method_comparison.json
└── reports/
    └── summary.md
```

### 6.2 Key JSON Fields

```json
{
  "r2": 0.8814,
  "slope": -0.0361,
  "lod": 4.15,
  "loq": 13.83,
  "uncertainty": {
    "r2_cv": 0.6655,
    "slope_ci_95": [-0.0412, -0.0310]
  }
}
```

---

## 7. Usage

### 7.1 Command Line

```bash
# Single gas
python -m gas_analysis.core.run_each_gas --gas Ethanol --avg-top-n 6

# All gases
python -m gas_analysis.core.run_each_gas --avg-top-n 6 --top-k 6
```

### 7.2 Python API

```python
from gas_analysis.core.pipeline import run_full_pipeline

result = run_full_pipeline(
    root_dir="Joy_Data/Ethanol",
    ref_path="Joy_Data/Ethanol/ref.csv",
    out_root="output/ethanol_test"
)
print(f"R²: {result['calibration']['r2']:.4f}")
```

---

## 8. Configuration

### Key Parameters (config/config.yaml)

```yaml
preprocessing:
  smooth:
    window: 11
    polyorder: 2

roi:
  min_wavelength: 500
  max_wavelength: 900
  r2_weight: 0.55

calibration:
  min_r2: 0.7
  use_wls: true

dynamics:
  enabled: true
  frame_rate: 1.0
```

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| Low R² | Increase smoothing, expand ROI range |
| Missing dynamics | Check data structure format |
| Slow execution | Disable polynomial fitting |

---

## 10. References

1. Beer-Lambert Law: Swinehart, D.F. (1962) J. Chem. Educ.
2. Savitzky-Golay: Savitzky & Golay (1964) Anal. Chem.
3. LOOCV: Stone, M. (1974) J. Royal Stat. Soc.

---

*See CODEMAP.md for detailed function reference.*
