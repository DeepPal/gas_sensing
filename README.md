# Au-MIP LSPR Gas Sensing Research Platform

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![Pytest](https://img.shields.io/badge/tests-pytest-green)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade platform for **Au nanoparticle Molecularly Imprinted Polymer (Au-MIP) Localized Surface Plasmon Resonance (LSPR)** gas sensor characterization. Provides real-time spectral acquisition, automated signal processing, physics-based calibration, and an interactive web dashboard — from raw photons to calibrated concentration in a single unified pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Sensor Physics](#sensor-physics)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Dashboard (Recommended)](#1-interactive-dashboard)
  - [CLI — Live Sensor](#2-cli--live-sensor-mode)
  - [CLI — Batch Analysis](#3-cli--batch-analysis)
  - [CLI — Simulation](#4-cli--simulation-mode)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing & Quality](#testing--quality)
- [Engineering Standards](#engineering-standards)
- [Troubleshooting](#troubleshooting)
- [Authors](#authors)

---

## Overview

This platform targets **VOC (Volatile Organic Compound) detection** using an optical sensor whose primary signal is the **shift in the LSPR peak wavelength** (Δλ, nanometers) of Au nanoparticles embedded in a molecularly imprinted polymer film.

Key capabilities:

| Feature | Details |
|---|---|
| **Real-time acquisition** | ThorLabs CCS200 spectrometer via DLL, VISA, or Serial |
| **Signal processing** | Savitzky-Golay smoothing, wavelet denoising, ALS / airPLS baseline correction |
| **Calibration** | Polynomial, Langmuir / Freundlich / Hill isotherms (AIC selection), GPR with uncertainty bounds |
| **Multi-ROI fusion** | Automated spectral region discovery with hybrid R²/slope-to-noise metric |
| **AI classification** | 1D CNN for gas-type identification; GPR for concentration estimation |
| **Dashboard** | Streamlit: 4 tabs (Automation, Experiment, Batch Analysis, Live Sensor) |
| **Session persistence** | Thread-safe CSV/Parquet streaming + per-session JSON metadata |
| **Simulation fallback** | Full pipeline runs without hardware for development/testing |

---

## Sensor Physics

The primary signal is the **peak wavelength shift** (Δλ) of the LSPR band:

```
Δλ = λ_gas − λ_reference      [nm]
```

- **Reference peak**: ~531.5 nm (Au nanoparticles, green region)
- **Physical sensitivity**: ~0.116 nm/ppm (literature value for ethanol)
- **Supported analytes**: Ethanol (EtOH), Isopropanol (IPA), Methanol (MeOH), mixed VOCs
- **Signal types**: Absorbance (primary), Transmittance, Raw Intensity

A **negative shift** (Δλ < 0) indicates analyte adsorption on the Au-MIP surface. The pipeline extracts Δλ via cross-correlation between the analyte spectrum and the reference spectrum, optionally averaged over multiple spectral ROIs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Entry Points                         │
│  run.py (CLI)          dashboard/app.py (Streamlit)     │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
┌────────────▼────────────┐  ┌────────────▼──────────────┐
│  SensorOrchestrator     │  │  Agentic Pipeline Tab      │
│  sensor_app/            │  │  dashboard/agentic_*       │
└────────────┬────────────┘  └────────────┬──────────────┘
             │                            │
┌────────────▼────────────────────────────▼──────────────┐
│               RealTimePipeline                          │
│         gas_analysis/core/realtime_pipeline.py          │
│  Stage 1: Preprocessing  (smooth, baseline, denoise)   │
│  Stage 2: Feature Extrac (peak find, ROI, Δλ)          │
│  Stage 3: Calibration    (polynomial / GPR)             │
│  Stage 4: Quality Ctrl   (SNR, saturation, confidence) │
└────────────┬────────────────────────────────────────────┘
             │
┌────────────▼──────────────────────┐   ┌────────────────┐
│  RealtimeAcquisitionService       │   │  ModelRegistry │
│  gas_analysis/acquisition/        │   │  CNN + GPR     │
│  (CCS200 DLL / VISA / Serial)     │   │  (optional)    │
└────────────┬──────────────────────┘   └────────────────┘
             │
┌────────────▼──────────────────────┐
│  LiveDataStore (singleton)        │   Thread-safe deque;
│  sensor_app/live_state.py         │   shared acq ↔ dashboard
└───────────────────────────────────┘
             │
     output/sessions/{YYYYMMDD_HHMMSS}/
     ├── pipeline_results.csv
     ├── session_meta.json
     └── raw_spectra.parquet   (optional)
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- Windows 10/11 (for CCS200 DLL) or Linux (VISA/serial modes)
- ThorLabs CCS200 spectrometer *(optional — simulation mode works without hardware)*

### 1. Clone the repository

```bash
git clone <repo-url>
cd Main_Research_Chula
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: `torch` (~2 GB) is required for the CNN classifier. If you only need
> calibration and signal processing, comment out `torch` and `torchvision` in
> `requirements.txt` — the platform degrades gracefully without them.

### 4. (Optional) Install hardware drivers

For real CCS200 acquisition:
```bash
pip install pyvisa pyvisa-py   # VISA backend
# Then install NI-VISA or libusb per your OS
```

---

## Usage

### Runtime Paths

This repository currently contains both a newer `spectraagent` runtime and older
research/legacy entrypoints.

Use these paths intentionally:

- `python -m spectraagent start --simulate --no-browser`
  Recommended for the newer FastAPI + agentic runtime. This is the primary
  production-oriented path for current runtime hardening, session persistence,
  WebSocket streaming, and Claude/agent integration.
- `.venv/Scripts/python.exe -m streamlit run dashboard/app.py`
  Recommended for the existing researcher-facing Streamlit dashboard workflow.
- `python run.py --mode ...`
  Legacy/compatibility CLI for older batch and sensor flows. Still useful, but
  not the main target of current runtime modernization work.

When adding new runtime features, prefer the `spectraagent` FastAPI/CLI stack
unless the change is explicitly for the legacy dashboard or historical pipeline.

### 1. Interactive Dashboard

The recommended entry point for researchers:

```bash
# From the project root
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
# or via helper script:
run_dashboard.bat
```

Open `http://localhost:8501` in your browser. Four tabs are available:

| Tab | Purpose |
|---|---|
| **Automation Pipeline** | 5-agent workflow: reference → acquire → train → predict → export |
| **Experiment (Guided)** | Step-by-step guided acquisition and calibration |
| **Batch Analysis** | Load Joy_Data/, visualize spectra, heatmaps, calibration curves |
| **Live Sensor** | Real-time CCS200 monitoring, concentration readout, SNR |

### 2. CLI — Live Sensor Mode

Acquire continuously from the CCS200 spectrometer:

```bash
python run.py --mode sensor --gas Ethanol --duration 3600
```

| Argument | Description | Default |
|---|---|---|
| `--gas` | Analyte label saved in session metadata | `unknown` |
| `--duration` | Acquisition duration in seconds | `60` |
| `--resource` | VISA resource string (e.g., `USB0::...`) | auto-detect |
| `--target-wavelength` | Expected LSPR peak (nm) | `532.0` |
| `--calibration-slope` | Sensitivity (nm/ppm) | `0.116` |

Outputs are written to `output/sessions/{YYYYMMDD_HHMMSS}/`.

### 3. CLI — Batch Analysis

Analyse a folder of experimental CSV files:

```bash
python run.py --mode batch --data data/JOY_Data/Ethanol
```

Expected folder structure:

```
data/JOY_Data/
└── Ethanol/
    ├── 0.5 ppm-1/
    │   ├── spectrum_001.csv
    │   └── ...
    ├── 1 ppm-1/
    └── ref_EtOH.csv        ← reference (baseline) spectrum
```

Each CSV must contain `wavelength` and `intensity` columns (or two unnamed columns in that order).

### 4. CLI — Simulation Mode

Run the full pipeline with synthetic spectra (no hardware needed):

```bash
python run.py --mode simulate --duration 30
```

---

## Project Structure

```
Main_Research_Chula/
├── run.py                      ← Unified CLI entry point
├── pyproject.toml              ← Build config, ruff & mypy settings, CLI scripts
├── requirements.txt            ← Runtime + dev dependencies
├── run_dashboard.bat           ← Windows shortcut for Streamlit
│
├── config/
│   ├── config.yaml             ← Full pipeline configuration
│   └── config_loader.py        ← YAML loader with duplicate-key detection
│
├── src/                        ← Primary Python package (strangler-fig migration)
│   ├── acquisition/            ← Re-exports CCS200Spectrometer & RealtimeAcquisitionService
│   ├── agents/
│   │   ├── drift.py            ← DriftDetectionAgent (rolling trend + offset alerts)
│   │   ├── quality.py          ← QualityAssuranceAgent (SNR / saturation gating)
│   │   └── training.py         ← TrainingAgent (auto-retrain on drift / R² decay / volume)
│   ├── api/                    ← FastAPI REST endpoints
│   ├── batch/
│   │   ├── aggregation.py      ← Stable-block detection, canonical spectrum selection
│   │   ├── data_loader.py      ← Joy-data CSV ingestion & scan-root discovery
│   │   └── pipeline.py         ← End-to-end batch analysis facade
│   ├── calibration/
│   │   ├── gpr.py              ← GPRCalibration: Gaussian Process Δλ → ppm
│   │   └── isotherms.py        ← Langmuir / Freundlich / Hill fitting + AIC model selection
│   ├── features/               ← Peak finding, ROI discovery, Lorentzian Δλ extraction
│   ├── inference/
│   │   └── orchestrator.py     ← SensorOrchestrator with drift + training agents wired in
│   ├── models/
│   │   ├── cnn.py              ← GasCNN (nn.Module) + CNNGasClassifier with MC Dropout
│   │   ├── registry.py         ← ModelRegistry: unified CNN/GPR/calibration loader
│   │   └── onnx_export.py      ← ONNX export, validation, OnnxInferenceWrapper
│   ├── preprocessing/
│   │   ├── baseline.py         ← ALS + airPLS (Zhang 2010) baseline correction
│   │   ├── normalization.py    ← Area/peak normalisation (NumPy 2.0 compatible)
│   │   └── smoothing.py        ← Savitzky-Golay + wavelet denoising
│   ├── schemas/
│   │   └── spectrum.py         ← Pydantic SpectrumReading + PredictionResult contracts
│   ├── scientific/
│   │   ├── lod.py              ← LOD/LOQ (ICH Q2(R1) bootstrap CI), robust regression, Mandel test
│   │   └── selectivity.py      ← Cross-sensitivity matrix + IUPAC selectivity coefficients
│   └── training/
│       ├── ablation.py         ← Preprocessing ablation study (6 configs, GPR CV)
│       ├── cross_gas_eval.py   ← Leave-one-gas-out (LOGO) cross-validation + MLflow
│       ├── mlflow_tracker.py   ← ExperimentTracker wrapper
│       ├── train_cnn.py        ← CNN training pipeline (LOOCV, MLflow logging)
│       └── train_gpr.py        ← GPR training pipeline (CV, calibration curves)
│
├── gas_analysis/               ← Legacy package (kept for backward compatibility)
│   ├── acquisition/            ← CCS200 hardware drivers (DLL / VISA / Serial)
│   ├── core/                   ← RealTimePipeline, preprocessing, calibration, CNN/GPR
│   ├── advanced/               ← ICA spectral decomposition, MCR-ALS
│   └── ...
│
├── sensor_app/                 ← Legacy orchestrator + LiveDataStore (still used by dashboard)
│
├── dashboard/
│   ├── app.py                  ← Streamlit app (4 tabs)
│   ├── agentic_pipeline_tab.py ← 5-agent automation workflow
│   ├── sensor_dashboard.py     ← Live sensor tab (real-time CCS200 feed)
│   ├── experiment_tab.py       ← Guided acquisition & calibration workflow
│   └── realtime_monitor.py     ← Performance metrics overlay
│
├── scripts/
│   ├── quality_gate.py         ← Local CI gate (ruff + pytest + mypy)
│   ├── train_realtime_models.py← CNN/GPR training helper
│   └── compare_sessions.py     ← Session comparison analysis
│
├── tests/                      ← 430 tests, 19 files (0 failures)
│   ├── conftest.py             ← Shared pytest fixtures & synthetic data builders
│   ├── test_acquisition.py     ← src.acquisition import contract
│   ├── test_agents.py          ← DriftDetectionAgent, QualityAssuranceAgent
│   ├── test_api.py             ← FastAPI endpoints
│   ├── test_batch.py           ← Batch pipeline end-to-end
│   ├── test_calibration.py     ← GPRCalibration fit/predict/persist
│   ├── test_cnn.py             ← GasCNN, CNNGasClassifier, MC Dropout (torch-skipped)
│   ├── test_config.py          ← Config loader
│   ├── test_deconvolution.py   ← ICA/MCR-ALS
│   ├── test_environment.py     ← Environment coefficients
│   ├── test_live_state.py      ← LiveDataStore thread-safety
│   ├── test_lod.py             ← LOD/LOQ/sensitivity/Mandel/robust_sensitivity
│   ├── test_isotherms.py       ← Langmuir/Freundlich/Hill/select_isotherm
│   ├── test_selectivity.py     ← Cross-sensitivity matrix & IUPAC coefficients
│   ├── test_models_registry.py ← ModelRegistry
│   ├── test_onnx_export.py     ← ONNX export/validate/wrapper (onnx-skipped)
│   ├── test_preprocessing.py   ← Baseline, smoothing, normalization
│   ├── test_realtime_pipeline.py ← RealTimePipeline 4-stage
│   ├── test_training_agent.py  ← TrainingAgent triggers & retrain cycle
│   └── test_training_scripts.py← train_gpr, train_cnn, ablation, cross_gas_eval CLIs
│
├── docs/
│   ├── ENGINEERING_STANDARDS.md
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── PAPER_METHODS_TEMPLATE.md
│   └── adr/                    ← Architecture Decision Records
│
└── output/                     ← Generated artefacts (git-ignored)
    ├── sessions/               ← Per-session pipeline_results.csv + session_meta.json
    └── models/                 ← Trained CNN (.pt), GPR (.joblib), calibration_params.json
```

---

## Configuration

All pipeline parameters live in [`config/config.yaml`](config/config.yaml). Key sections:

| Section | Controls |
|---|---|
| `preprocessing` | ALS smoothness (λ, p), Savitzky-Golay window |
| `roi.shift` | Cross-correlation step_nm, upsample factor, window widths |
| `roi.discovery` | Automated band search range and weighting |
| `calibration` | Model selection (polynomial / Langmuir / PLSR / ensemble), CV folds |
| `quality` | Minimum SNR (default 4.0), max RSD (7.5%), saturation threshold |
| `sensor` | Integration time (ms), target wavelength, warm-up frames |
| `response_series` | T90/T10 activation thresholds, changepoint method |

Per-gas overrides (`Ethanol`, `IPA`, `MeOH`, `MixVOC`) can be placed under the gas name key to override any base setting.

---

## Testing & Quality

### Run tests

```bash
pytest                        # all 430 tests
pytest tests/test_config.py   # specific file
pytest -v --tb=short          # verbose with short tracebacks
```

> **Note**: 18 tests are skipped when `onnx`/`onnxruntime` are not installed — this is intentional.
> Install with `pip install onnx onnxruntime` to activate them.

> **Important (pytest import mode)**: this project uses `--import-mode=importlib`.
> Do **not** add `__init__.py` files under `tests/` package paths that mirror
> real source package names (e.g. `tests/spectraagent/__init__.py`), because
> they can shadow runtime packages during collection.

### Local quality gate (mirrors CI)

```bash
python scripts/quality_gate.py           # ruff + pytest
python scripts/quality_gate.py --strict  # + mypy type checking
```

### Individual tools

```bash
ruff check src/            # linting (zero errors)
ruff format src/           # auto-format
mypy src/ gas_analysis/    # type checking
```

### CI

GitHub Actions runs ruff, pytest, and mypy on every push/PR via [`.github/workflows/quality.yml`](.github/workflows/quality.yml).

### Advanced CLI tools

```bash
# Leave-one-gas-out cross-validation (requires data + torch)
python -m src.training.cross_gas_eval --data-dir data/JOY_Data

# Preprocessing ablation study
python -m src.training.ablation --data-dir data/JOY_Data/Ethanol

# Export trained CNN to ONNX for edge deployment
gas-export-onnx --checkpoint output/models/cnn_classifier.pt --output output/models/cnn.onnx --validate
```

---

## Engineering Standards

This repository follows a research-to-production standards baseline:

- **Governance**: [`docs/ENGINEERING_STANDARDS.md`](docs/ENGINEERING_STANDARDS.md)
- **Architecture decisions**: [`docs/adr/`](docs/adr/)
- **Contribution workflow**: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Scientific methods**: [`docs/PAPER_METHODS_TEMPLATE.md`](docs/PAPER_METHODS_TEMPLATE.md)

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `DLL error -1073807343` | CCS200 connected but not powered | Power on the spectrometer before starting |
| `VI_ERROR_TMO (-1073807339)` | Stale VISA handle from ungraceful shutdown | Unplug/replug USB; call `close()` in finally block |
| `pyvisa not found` | VISA backend missing | `pip install pyvisa pyvisa-py` |
| Dashboard won't start | Wrong working directory | Run from project root; use `run_dashboard.bat` |
| `torch` import error | PyTorch not installed | `pip install torch torchvision`; platform degrades gracefully without it |
| UTF-8 console errors on Windows | Windows default cp1252 | Handled automatically in `run.py`; set `PYTHONIOENCODING=utf-8` as fallback |

---

## Authors

- **Chula Research Team** — Au-MIP LSPR sensor design and experimental data
- **Engineering contributions** — Pipeline architecture, dashboard, CI/CD

For bug reports and feature requests, please open an issue in this repository.
