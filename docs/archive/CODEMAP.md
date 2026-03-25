# Gas Sensing Calibration Pipeline - Code Map

## Project Overview

This project implements a comprehensive spectral gas sensing calibration pipeline for VOC (Volatile Organic Compound) detection using optical fiber sensors. The pipeline processes spectral data to extract wavelength shift (Δλ) and absorbance amplitude (ΔA) features for gas concentration calibration.

---

## Directory Structure

```
Joy_Code_1/
├── config/
│   ├── config.yaml              # Main configuration file
│   └── config_loader.py         # Configuration loading utilities
│
├── gas_analysis/
│   ├── core/                    # Core pipeline modules
│   │   ├── pipeline.py          # Main pipeline (10,308 lines, 113 functions)
│   │   ├── dynamics.py          # T90/T10 response time computation
│   │   ├── run_each_gas.py      # CLI entry point for running pipeline
│   │   ├── research_report.py   # Markdown/JSON report generation
│   │   ├── preprocessing.py     # Signal preprocessing utilities
│   │   └── responsive_frame_selector.py  # Frame selection algorithms
│   │
│   ├── advanced/                # Advanced analysis methods
│   │   ├── deconvolution_ica.py # ICA-based spectral deconvolution
│   │   └── mcr_als.py           # MCR-ALS multivariate analysis
│   │
│   ├── tools/
│   │   └── build_report.py      # Report building utilities
│   │
│   └── utils/
│       └── generate_summary_table.py  # Summary table generation
│
├── scripts/                     # Standalone analysis scripts
│   ├── cross_gas_selectivity.py
│   ├── generate_selectivity_report.py
│   ├── multi_roi_analysis.py
│   └── pipeline_cli.py
│
├── tests/                       # Unit tests
│   ├── test_deconvolution.py
│   └── test_environment.py
│
├── Joy_Data/                    # Raw experimental data
│   └── [Gas folders with CSV spectra]
│
└── output/                      # Pipeline outputs
    ├── ethanol_topavg/
    ├── isopropanol_topavg/
    ├── methanol_topavg/
    └── mixvoc_topavg/
```

---

## Core Pipeline Architecture

### Entry Point: `run_each_gas.py`

```
CLI Arguments:
  --gas [Ethanol|Isopropanol|Methanol|MixVOC]  # Specific gas or all
  --avg-top-n N                                 # Top N frames to average
  --top-k K                                     # Top K candidates to track
```

### Main Pipeline Flow (`pipeline.py`)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           run_full_pipeline()                                │
│                              (Line 9500+)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING & PREPROCESSING                                             │
│     ├── scan_experiment_root()      - Discover CSV files                     │
│     ├── _read_csv_spectrum()        - Load individual spectra                │
│     ├── compute_transmittance()     - Calculate T = I/I_ref                  │
│     └── _append_absorbance_column() - Calculate A = -log10(T)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. FRAME SELECTION & AVERAGING                                              │
│     ├── find_stable_block()         - Detect stable measurement region       │
│     ├── average_stable_block()      - Average frames in stable region        │
│     └── average_top_frames()        - Select top-N most responsive frames    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. CANONICAL SPECTRUM GENERATION                                            │
│     ├── select_canonical_per_concentration()  - One spectrum per conc.       │
│     ├── _baseline_correct_canonical()         - Baseline correction          │
│     └── save_canonical_spectra()              - Save to CSV                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. ROI DISCOVERY & FEATURE DETECTION                                        │
│     ├── _discover_roi_in_band()              - Find optimal ROI              │
│     ├── _find_monotonic_wavelengths()        - Detect monotonic features     │
│     ├── _refine_centroid_with_derivative()   - Sub-pixel refinement          │
│     └── _evaluate_roi_candidate()            - Score ROI candidates          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. CALIBRATION                                                              │
│     ├── find_roi_and_calibration()           - Main calibration function     │
│     │   ├── Wavelength Shift (Δλ) Method                                     │
│     │   │   ├── _measure_peak_within_window()                                │
│     │   │   ├── _estimate_shift_crosscorr()                                  │
│     │   │   └── Linear/WLS/Polynomial/Langmuir fitting                       │
│     │   │                                                                    │
│     │   └── Feature Selection                                                │
│     │       ├── centroid, xcorr, gaussian, valley, monotonic_peak            │
│     │       └── Best feature selected by R² and CV                           │
│     │                                                                        │
│     └── perform_absorbance_amplitude_calibration()  - ΔA Method              │
│         ├── Raw absorbance                                                   │
│         ├── Window-averaged (±2 nm)                                          │
│         ├── First derivative (dA/dλ)                                         │
│         └── Differential absorbance                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. VALIDATION & UNCERTAINTY                                                 │
│     ├── LOOCV (Leave-One-Out Cross-Validation)                               │
│     ├── Bootstrap confidence intervals (95% CI)                              │
│     ├── LOD/LOQ calculation (3σ/10σ method)                                  │
│     └── Spearman correlation for monotonicity                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  7. DYNAMICS ANALYSIS                                                        │
│     ├── compute_response_recovery_times()    - From frame structure          │
│     └── compute_t90_t10_from_timeseries()    - From time-series CSV          │
│         ├── T90: Time to reach 90% of steady-state                           │
│         └── T10: Time to recover to 10% of response                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  8. VISUALIZATION & REPORTING                                                │
│     ├── save_wavelength_shift_visualization()                                │
│     ├── save_research_grade_calibration_plot()                               │
│     ├── save_spectral_response_diagnostic()                                  │
│     ├── generate_method_comparison_report()                                  │
│     ├── write_run_summary()                  - summary.md                    │
│     └── generate_methodology_markdown()      - methodology_report.md         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Functions Reference

### Data Loading (Lines 1-1000)

| Function | Line | Description |
|----------|------|-------------|
| `_read_csv_spectrum()` | ~800 | Load single CSV spectrum file |
| `scan_experiment_root()` | ~1200 | Discover all experiment files |
| `load_reference_csv()` | ~1100 | Load reference spectrum |

### Signal Processing (Lines 1000-3000)

| Function | Line | Description |
|----------|------|-------------|
| `compute_transmittance()` | ~1500 | Calculate T = I/I_ref |
| `_append_absorbance_column()` | ~1600 | Calculate A = -log10(T) |
| `_smooth_vector()` | ~900 | Savitzky-Golay smoothing |
| `find_stable_block()` | ~2000 | Detect stable measurement region |
| `average_top_frames()` | ~2200 | Select top-N responsive frames |

### ROI & Feature Detection (Lines 3000-5000)

| Function | Line | Description |
|----------|------|-------------|
| `_discover_roi_in_band()` | ~3500 | Find optimal ROI in wavelength band |
| `_find_monotonic_wavelengths()` | ~3800 | Detect monotonic spectral features |
| `_refine_centroid_with_derivative()` | ~3200 | Sub-pixel centroid refinement |
| `_evaluate_roi_candidate()` | ~3600 | Score ROI candidate quality |
| `_measure_peak_within_window()` | ~4000 | Measure peak position in window |

### Calibration (Lines 5000-7500)

| Function | Line | Description |
|----------|------|-------------|
| `find_roi_and_calibration()` | ~5200 | Main wavelength shift calibration |
| `perform_absorbance_amplitude_calibration()` | ~7200 | Enhanced ΔA calibration |
| `_weighted_linear()` | ~1000 | Weighted least squares regression |
| `_theil_sen()` | ~1050 | Robust Theil-Sen regression |
| `_ransac()` | ~1080 | RANSAC outlier-robust regression |

### Validation (Lines 5000-6000)

| Function | Line | Description |
|----------|------|-------------|
| LOOCV | ~5500 | Leave-One-Out Cross-Validation |
| Bootstrap CI | ~5600 | 95% confidence intervals |
| LOD/LOQ | ~5700 | Detection/Quantification limits |

### Dynamics (dynamics.py)

| Function | Line | Description |
|----------|------|-------------|
| `compute_response_recovery_times()` | 12 | T90/T10 from frame structure |
| `compute_t90_t10_from_timeseries()` | 153 | T90/T10 from time-series CSV |

### Visualization (Lines 6500-8000)

| Function | Line | Description |
|----------|------|-------------|
| `save_wavelength_shift_visualization()` | ~6500 | 6-panel Δλ visualization |
| `save_research_grade_calibration_plot()` | ~6800 | Publication-quality plot |
| `save_spectral_response_diagnostic()` | ~7000 | Δλ vs ΔA comparison |
| `generate_method_comparison_report()` | ~7600 | Method comparison plot |

### Reporting (Lines 8000-8800)

| Function | Line | Description |
|----------|------|-------------|
| `write_run_summary()` | ~8000 | Generate summary.md |
| `save_calibration_outputs()` | ~8700 | Save calibration JSON/CSV |

---

## Configuration Reference (`config.yaml`)

```yaml
# Key configuration sections:

preprocessing:
  smooth:
    window: 11          # Savitzky-Golay window
    polyorder: 2        # Polynomial order

roi:
  min_wavelength: 500   # ROI search range
  max_wavelength: 900
  band_half_width: 12   # Half-width for ROI band
  r2_weight: 0.55       # Weight for R² in scoring

calibration:
  min_r2: 0.7           # Minimum R² threshold
  min_points: 3         # Minimum data points
  use_wls: true         # Weighted least squares

dynamics:
  baseline_frames: 20   # Frames for baseline
  frame_rate: 1.0       # Frames per second
```

---

## Output Structure

```
output/{gas}_topavg/
├── aggregated/                    # Aggregated spectra per concentration
│   └── {conc}/
│       └── *.csv
│
├── plots/
│   ├── canonical_overlay.png      # All concentrations overlaid
│   ├── concentration_response.png # Response vs concentration
│   ├── calibration_research_grade.png  # Publication-quality calibration
│   ├── wavelength_shift_visualization.png  # 6-panel Δλ analysis
│   ├── absorbance_amplitude_calibration.png  # ΔA calibration
│   ├── spectral_response_diagnostic.png  # Δλ vs ΔA comparison
│   └── method_comparison.png      # Side-by-side method comparison
│
├── metrics/
│   ├── calibration_metrics.json   # Main calibration results
│   ├── absorbance_amplitude_calibration.json  # ΔA results
│   ├── method_comparison.json     # Method comparison metrics
│   ├── dynamics_summary.json      # T90/T10 times
│   ├── noise_metrics.json         # Signal noise analysis
│   └── qc_summary.json            # Quality control metrics
│
├── reports/
│   ├── summary.md                 # Human-readable summary
│   ├── methodology_report.md      # Detailed methodology
│   └── analysis_summary.json      # Machine-readable summary
│
├── dynamics/
│   └── response_recovery.png      # T90/T10 visualization
│
└── time_series/                   # Time-series data for dynamics
    └── *.csv
```

---

## Calibration Methods

### 1. Wavelength Shift (Δλ) Method

Tracks spectral feature position shift with concentration:

```
Δλ = λ(C) - λ(0)

Features tracked:
- centroid: Intensity-weighted center
- xcorr: Cross-correlation peak
- gaussian: Gaussian fit center
- valley: Absorption valley minimum
- monotonic_peak: Monotonic trend peak
```

### 2. Absorbance Amplitude (ΔA) Method

Measures absorbance intensity change at optimal wavelength:

```
ΔA = A(C) - A(0)

Enhancement methods:
- raw: Direct absorbance value
- window_avg: ±2 nm window average
- derivative: First derivative (dA/dλ)
- differential: Subtract reference wavelength
```

---

## Key Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| R² | Coefficient of determination | 1 - SS_res/SS_tot |
| R²_CV | Cross-validated R² | LOOCV mean |
| Slope | Sensitivity | Δλ/ΔC or ΔA/ΔC |
| LOD | Limit of Detection | 3σ/slope |
| LOQ | Limit of Quantification | 10σ/slope |
| T90 | Response time | Time to 90% of steady-state |
| T10 | Recovery time | Time to decay to 10% |

---

## Data Flow Diagram

```
Raw CSV Files (Joy_Data/)
        │
        ▼
┌───────────────────┐
│  Load & Preprocess │
│  - Read spectra    │
│  - Compute T, A    │
│  - Smooth signals  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Frame Selection   │
│  - Stable block    │
│  - Top-N averaging │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  Canonical Spectra │
│  - One per conc.   │
│  - Baseline corr.  │
└───────────────────┘
        │
        ├──────────────────────┐
        ▼                      ▼
┌───────────────────┐  ┌───────────────────┐
│  Δλ Calibration   │  │  ΔA Calibration   │
│  - Feature detect │  │  - Multi-method   │
│  - Linear fit     │  │  - Best selection │
│  - LOOCV          │  │  - Bootstrap CI   │
└───────────────────┘  └───────────────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
┌───────────────────────────────┐
│  Method Comparison & Reports   │
│  - summary.md                  │
│  - methodology_report.md       │
│  - calibration_metrics.json    │
│  - Publication-quality plots   │
└───────────────────────────────┘
```

---

## Usage Examples

### Run for single gas:
```bash
python -m gas_analysis.core.run_each_gas --gas Ethanol --avg-top-n 6 --top-k 6
```

### Run for all gases:
```bash
python -m gas_analysis.core.run_each_gas --avg-top-n 6 --top-k 6
```

### Generate summary table:
```bash
python -m gas_analysis.utils.generate_summary_table
```

---

## Version History

- **Current**: Enhanced absorbance amplitude with multi-method selection
- **Features**: Wavelength shift + Absorbance amplitude dual calibration
- **Validation**: LOOCV, Bootstrap CI, LOD/LOQ
- **Dynamics**: T90/T10 response/recovery times
- **Reporting**: Publication-quality plots and comprehensive reports

---

*Generated: 2025-11-25*
