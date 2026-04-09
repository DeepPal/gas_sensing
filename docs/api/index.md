# API Reference

Auto-generated from the typed `src/` package (mypy-clean, PEP 561 typed).

All public functions and classes include full type signatures, Google-style
docstrings, and source links. The stable commercial facade is `src/public_api.py`.

## Package layout

```
src/
├── preprocessing/          Spectrum preprocessing
│   ├── baseline.py         ALS asymmetric least-squares baseline correction
│   ├── denoising.py        Savitzky-Golay + wavelet denoising
│   └── quality.py          SNR estimation, saturation detection
│
├── calibration/            Calibration and uncertainty quantification
│   ├── gpr.py              Gaussian Process Regressor (scikit-learn wrapper)
│   ├── physics_kernel.py   Physics-informed GPR — Langmuir isotherm mean function
│   ├── pls.py              PLS regression with LOOCV and VIP scores
│   ├── conformal.py        Split conformal prediction (normalised scores, coverage guarantee)
│   ├── active_learning.py  BayesianExperimentDesigner — logspace max-variance acquisition
│   ├── isotherms.py        Langmuir / Freundlich / Hill isotherm fitting
│   ├── roi_scan.py         ROI scan and concentration-response computation
│   ├── multi_roi.py        Multi-ROI calibration fusion
│   ├── transforms.py       Signal transforms (log, power, derivative)
│   ├── batch_reproducibility.py  Batch sensor QC — pooled LOD, RSD, accept/reject
│   └── selectivity.py      Selectivity report — cross-reactivity K values (IUPAC)
│
├── features/               Feature extraction
│   └── lspr_features.py    LSPR: [Δλ, ΔI_peak, ΔI_area, ΔI_std]; fallback raw features
│
├── models/                 ML models
│   ├── cnn.py              GasCNN (1D conv) + CNNGasClassifier with MC Dropout
│   └── onnx_export.py      ONNX export with numerical validation
│
├── scientific/             IUPAC-compliant analytical figures of merit
│   ├── lod.py              LOD/LOQ/LOB triad — bootstrap CI, blank-based and residual-based
│   ├── regression.py       Weighted linear, Theil-Sen, RANSAC
│   └── selectivity.py      Selectivity matrix + from_calibration_data helper
│
├── inference/              Real-time inference
│   ├── realtime_pipeline.py  RealTimePipeline (4 stages) + ConformalCalibrator wiring
│   └── session_analyzer.py   SessionAnalyzer — post-session LOD/LOQ/T90/drift/linearity
│
├── io/                     Data I/O
│   └── hdf5.py             HDF5 archival — write/read spectral datasets
│
├── spectrometer/           Hardware abstraction layer (research-facing)
│   ├── base.py             AbstractSpectrometer + SpectralFrame dataclass
│   ├── registry.py         SpectrometerRegistry — register / discover / create
│   ├── simulated.py        SimulatedSpectrometer (LSPR, fluorescence, absorbance modes)
│   └── ccs200_adapter.py   CCS200Adapter — wraps native DLL / gas_analysis driver
│
├── batch/                  Batch multi-frame processing
│   ├── preprocessing.py    Multi-frame preprocessing pipeline
│   ├── response.py         Concentration-response aggregation
│   ├── aggregation.py      Stable-plateau detection, canonical spectrum builder
│   └── time_series.py      Response time-series extraction (T90/T10)
│
├── reporting/              Reports and figures
│   ├── metrics.py          LOD/SNR/QC metric computation
│   ├── plots.py            Calibration curves, spectral overlays, ROI diagnostics
│   ├── publication.py      Publication-quality figure generation (Nature/ACS style)
│   ├── environment.py      Environment metadata capture
│   └── io.py               Save canonical spectra, JSON reports, CSV outputs
│
├── agents/                 Signal-path agents
│   ├── quality.py          QualityAssuranceAgent — SNR + saturation gating
│   └── training.py         TrainingAgent — auto-retrain on drift / R² decay / volume
│
├── training/               Training scripts
│   ├── train_gpr.py        GPR training with MLflow tracking
│   ├── train_cnn.py        CNN training with ablation config
│   ├── ablation.py         Preprocessing ablation study runner
│   └── cross_gas_eval.py   Cross-analyte sensitivity evaluation (LOGO CV)
│
└── public_api.py           Stable commercial facade — re-exports all key public classes
```

## Stable public API

Import from `src.public_api` for stable, versioned access:

```python
from src.public_api import (
    # Inference
    RealTimePipeline,
    PipelineConfig,
    PipelineResult,

    # Models
    CNNGasClassifier,

    # Calibration
    GPRCalibration,

    # Schemas
    SpectrumReading,
    PredictionResult,

    # Reporting
    compute_roi_performance,
    save_calibration_outputs,
)
```

## Quick examples

### Calibrate with conformal intervals

```python
from src.calibration.gpr import GPRCalibration
from src.calibration.conformal import ConformalCalibrator
import numpy as np

# Fit GPR
concs = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
shifts = np.array([-1.2, -3.8, -6.5, -10.1, -18.4])
gpr = GPRCalibration()
gpr.fit(concs, shifts)

# Calibrate conformal predictor on hold-out set
cal = ConformalCalibrator()
cal.calibrate(gpr, concs_cal, shifts_cal)

# Predict with guaranteed coverage
lo, hi = cal.predict_interval(shift_new, alpha=0.10)  # 90% coverage
```

### Compute IUPAC LOD/LOQ/LOB

```python
from src.scientific.lod import compute_lod_loq

result = compute_lod_loq(
    concentrations=concs,
    signals=shifts,
    blank_signals=blank_shifts,   # optional
    bootstrap_n=2000,
)
print(result["lob"], result["lod"], result["loq"])
print(result["lod_ci_low"], result["lod_ci_high"])  # bootstrap 95% CI
```

### Use the spectrometer registry

```python
from src.spectrometer.registry import SpectrometerRegistry

with SpectrometerRegistry.create("simulated") as spec:
    spec.open()
    frame = spec.acquire()
    print(f"Peak: {frame.peak_wavelength:.2f} nm  SNR: {frame.snr:.1f}")
```

### Run the full real-time pipeline

```python
from src.inference.realtime_pipeline import RealTimePipeline, PipelineConfig
import numpy as np

cfg = PipelineConfig(
    integration_time_ms=50.0,
    peak_search_min_nm=650.0,
    peak_search_max_nm=780.0,
    reference_wavelength=717.9,
)
pipeline = RealTimePipeline(cfg)
result = pipeline.process_spectrum(wavelengths, intensities)
if result.success:
    print(result.spectrum.concentration_ppm, result.spectrum.ci_low, result.spectrum.ci_high)
```
