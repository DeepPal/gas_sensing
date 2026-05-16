# PIPELINE.md — Processing Architecture

## System Architecture (5-Layer Model)

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: Deployment + Agent                                        │
│  dashboard/app.py (Streamlit)  │  src/api/main.py (FastAPI)        │
│  src/agents/drift.py           │  src/agents/quality.py            │
└────────────────────────┬────────────────────────┬───────────────────┘
                         │                        │
┌────────────────────────▼────────────────────────▼───────────────────┐
│  Layer 4: ML + Learning                                             │
│  src/models/cnn_classifier.py  │  src/models/gpr_calibration.py    │
│  src/training/train_cnn.py     │  src/training/train_gpr.py        │
│  src/training/mlflow_tracker.py                                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│  Layer 3: Feature / Representation                                  │
│  src/features/peak_detection.py    src/features/roi_discovery.py   │
│  src/features/wavelength_shift.py  src/features/lspr_features.py   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│  Layer 2: Processing                                                │
│  src/preprocessing/denoising.py    src/preprocessing/baseline.py   │
│  src/preprocessing/normalization.py  src/preprocessing/quality.py  │
└────────────────────────┬────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────┐
│  Layer 1: Data                                                      │
│  src/schemas/spectrum.py (Pydantic contracts)                       │
│  src/batch/data_loader.py         src/acquisition/ccs200_realtime  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Real-Time Pipeline (4 Stages)

Entry point: `src/inference/realtime_pipeline.py::RealTimePipeline.process_spectrum()`

```
Raw spectrum (wavelengths[], intensities[])
        │
        ▼
┌───────────────────────────────────┐
│ Stage 1: Preprocessing            │
│  1. Validate input (shape, finite)│
│  2. Savitzky-Golay smoothing      │
│  3. ALS baseline correction       │
│  4. Optional wavelet denoising    │
│  Output: clean_intensities[]      │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Stage 2: Feature Extraction       │
│  1. Peak wavelength detection     │
│  2. Cross-correlation shift (Δλ)  │
│  3. SNR estimation                │
│  4. ROI-based sub-band features   │
│  Output: SpectrumData populated   │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Stage 3: Calibration / Inference  │
│  1. Heuristic: C = Δλ / slope     │
│  2. GPR (if model loaded)         │
│  3. CNN classification (if loaded)│
│  Output: concentration_ppm ± σ    │
│          gas_type + confidence    │
└────────────────┬──────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Stage 4: Quality Control          │
│  1. SNR gate (min_snr=4.0)       │
│  2. Saturation check (<60000 ct) │
│  3. Compute quality_score [0,1]  │
│  Output: PipelineResult           │
└───────────────────────────────────┘
```

### Preprocessing Functions (pure, stateless)

```python
# All functions: (wavelengths, intensities, **params) -> np.ndarray
smooth_spectrum(intensities, window=11, poly=3)          # Savitzky-Golay
als_baseline(intensities, lam=1e5, p=0.01, niter=10)   # Asymmetric Least Squares
wavelet_denoise(intensities, wavelet='db4', level=3)     # Discrete wavelet
normalize_spectrum(intensities, method='area')            # Area or peak normalization
```

### Feature Extraction Functions (pure, stateless)

```python
# Peak detection
detect_lspr_peak(wavelengths, intensities, search_range=(500, 600)) -> float

# Wavelength shift (primary signal)
estimate_shift_xcorr(wavelengths, intensities, reference_intensities,
                     window_nm=20.0, upsample=10) -> float  # returns Δλ in nm

# LSPR feature vector (for model input)
extract_lspr_features(wavelengths, intensities, reference_intensities,
                      roi_center, roi_width) -> dict:
  {delta_lambda: float, delta_intensity_peak: float,
   delta_intensity_area: float, delta_intensity_std: float}
```

---

## Batch Pipeline (Offline Analysis)

Entry point: `src/batch/runner.py::run_full_pipeline(data_path, config, output_dir)`

```
Experiment folder (data/raw/Joy_Data/Ethanol/)
        │
        ▼
[data_loader]   Scan folder → list of (concentration, session_path) pairs
        │
        ▼
[aggregation]   For each session: detect plateau → select top-K frames → average
        │
        ▼
[canonical]     Select single canonical spectrum per concentration level
        │
        ▼
[preprocessing] Apply full preprocessing pipeline to each canonical spectrum
        │
        ▼
[features]      Extract Δλ, peak, ROI features for each concentration
        │
        ▼
[calibration]   Fit calibration curve (polynomial / Langmuir / PLSR / GPR)
                Compute R², RMSE, LOD; log to MLflow
        │
        ▼
[dynamics]      Compute T90/T10 response + recovery times (from raw frames)
        │
        ▼
[reporting]     Generate plots, CSV summary, HTML report
        │
        ▼
reports/
├── plots/calibration_curve.png
├── plots/spectral_heatmap.png
├── metrics/calibration_summary.csv
└── run_summary.html
```

---

## Inference API Pipeline

Entry point: `POST http://localhost:8000/predict`

```
HTTP POST /predict
{spectrum_id, wavelengths[], intensities[], gas_type, concentration_ppm=0.0}
        │
        ▼
[api/dependencies]   Load ModelRegistry (singleton, loaded at startup)
        │
        ▼
[api/routes/predict] Validate SpectrumReading via Pydantic
        │
        ▼
[inference/realtime_pipeline]  Run 4-stage pipeline
        │
        ▼
HTTP 200 PredictionResult
{peak_wavelength, wavelength_shift_nm, concentration_ppm ± std,
 gas_type_predicted, confidence, snr, quality_score, processing_time_ms}
```

---

## Training Pipeline

Entry point: `python -m src.training.train_cnn` or `python -m src.training.train_gpr`

```
data/raw/Joy_Data/
        │
        ▼
[batch/data_loader]   Load all CSV files → SpectrumReading objects
[batch/aggregation]   Temporal gating → plateau spectra only
        │
        ▼
[preprocessing]       Full pipeline on each spectrum
[features]            Extract LSPR features (Δλ, ΔI_peak, ΔI_area, ΔI_std)
        │
        ▼
[training/mlflow_tracker]  Start MLflow run, log params + dataset fingerprint
        │
        ├── CNN branch:
        │   [models/cnn_classifier]  Train 1D CNN (gas classification)
        │   Log: accuracy, F1, confusion matrix
        │   Save: models/registry/cnn_classifier.pt
        │
        └── GPR branch:
            [models/gpr_calibration]  Train GPR per gas type (concentration)
            Log: R², RMSE, LOD, calibration curve plot
            Save: models/registry/gpr_calibration_{gas}.joblib
```

---

## Configuration Flow

All pipeline stages read from `configs/config.yaml` via `configs.config_loader.CONFIG`:

```yaml
preprocessing:
  savgol_window: 11
  savgol_poly: 3
  als_lambda: 1.0e5
  als_p: 0.01

roi:
  shift:
    step_nm: 0.1
    window_nm: [10, 15, 20, 25]
  discovery:
    enabled: true
    search_start_nm: 480
    search_end_nm: 600

calibration:
  model: "ensemble"          # polynomial | langmuir | plsr | gpr | ensemble
  cv_folds: 5

quality:
  min_snr: 4.0
  max_rsd_pct: 7.5
  saturation_threshold: 60000

mlflow:
  tracking_uri: "experiments/mlruns"
  experiment_name: "AuMIP_LSPR_Gas_Sensing"
```

---

## Performance Targets

| Stage | Latency Target | Measured (CPU) |
|-------|---------------|----------------|
| Stage 1: Preprocessing | < 5 ms | ~2 ms |
| Stage 2: Feature Extraction | < 5 ms | ~3 ms |
| Stage 3: GPR inference | < 20 ms | ~8 ms |
| Stage 3: CNN inference | < 20 ms | ~15 ms |
| Stage 4: QC | < 1 ms | ~0.5 ms |
| **Total (CPU, no GPU)** | **< 50 ms** | **~28 ms** |
| API overhead (FastAPI) | < 5 ms | ~3 ms |

---

## Extending the Pipeline

To add a new preprocessing step:
1. Add a pure function `src/preprocessing/new_step.py` with signature `(wavelengths, intensities, **params) -> np.ndarray`
2. Add the config key in `configs/PIPELINE.yaml`
3. Wire it into `PreprocessingStage.process()` in `src/inference/realtime_pipeline.py`
4. Add a test in `tests/test_preprocessing.py`

To add a new gas type:
1. Add the gas name to `KNOWN_GAS_TYPES` in `src/schemas/spectrum.py`
2. Add per-gas config override in `configs/config.yaml`
3. Add training data to `data/raw/{GasType}/`
4. Retrain CNN + GPR with `python -m src.training.train_cnn --gas {GasType}`
