# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added — Commercial-grade analysis quality (2026-03)
- `config/MODEL.yaml` — versioned ML hyperparameter registry (CNN architecture, GPR kernel,
  multi-task future path, production model paths); merged from stale `configs/` directory
- `tests/test_smoke.py` — 7 end-to-end smoke tests covering config load, preprocessing,
  peak detection, GPR fit/predict round-trip, stable-plateau aggregation, LiveDataStore
  thread safety, and ONNX numerical validation; run with `pytest -m smoke`
- `config/config.yaml` — added `mlflow`, `api`, and `agents` sections previously only in
  the now-deleted `configs/` directory; single source of truth for all configuration
- Dashboard: stable-plateau representative spectra via `build_canonical_from_scan()` —
  replaces naive last-10-frame average; uses temporal tail-gating + longest stable sub-block
- Dashboard: LSPR-correct Δλ response variable — when a reference spectrum is loaded,
  all calibration, sensitivity, LOD and isotherm calculations use Δλ (nm) not peak
  intensity (a.u.), matching IUPAC definitions for optical sensors
- Dashboard: residual-based LOD noise (σ from fit residuals, ddof=2) — more defensible
  than spectral baseline std per IUPAC 1995
- Dashboard: MCR-ALS spectral deconvolution panel — resolves mixture spectra into
  pure-component spectra (S) and abundance profiles (C) with non-negativity constraints
- Dashboard: HTML calibration report export — includes gas type, date, sensitivity, R²,
  LOD, LOQ, isotherm model, and instrument metadata
- Dashboard: CNN → ONNX export button with numerical validation (max delta check)

### Changed — Commercial-grade analysis quality (2026-03)
- Selectivity K ratios now use peak wavelength (nm/ppm) sensitivities — dimensionless
  ratios as per IUPAC definition; previously used peak intensity (a.u./ppm) which made
  cross-gas K values unit-dependent and physically incorrect
- `sensor_app/` directory removed — canonical code consolidated in `src/inference/`;
  all fallback `try/except ImportError` shims removed from `run.py`,
  `dashboard/sensor_dashboard.py`, and `src/api/main.py`
- `configs/` directory removed — merged into `config/` (canonical); `MODEL.yaml` moved,
  `mlflow`/`api`/`agents` sections merged into `config/config.yaml`
- `dashboard/integrated_dashboard.py` and `dashboard/realtime_monitor.py` removed —
  superseded by `dashboard/app.py` and `dashboard/sensor_dashboard.py`
- `pyproject.toml` — package discovery, ruff isort `known-first-party`, and coverage
  source updated to remove stale `configs*` and `sensor_app` entries; `gas_analysis*`
  added to setuptools package discovery; `smoke` pytest marker registered
- `.github/workflows/quality.yml` and `scripts/quality_gate.py` — `sensor_app` removed
  from mypy target paths
- `.pre-commit-config.yaml` — `sensor_app` removed from mypy paths; `src` added
- `.gitignore` — `output/batch/`, `output/ethanol_ui/`, `output/ml_dataset/`,
  `output/summary_*.json` added to prevent generated outputs from being committed

### Fixed — Commercial-grade analysis quality (2026-03)
- `src/api/main.py` — `from configs.config_loader` → `from config.config_loader`
  (would have caused a silent `ModuleNotFoundError` at FastAPI server startup)
- `run.py` sensor mode — removed dead `except ImportError` fallback to deleted
  `sensor_app.orchestrator` that would crash on `--mode sensor`

### Added
- `src/` package — strangler-fig migration of core pipeline into typed, tested modules
- `src/agents/drift.py` — DriftDetectionAgent with rolling trend + offset threshold alerts
- `src/agents/training.py` — TrainingAgent with auto-retrain on drift / R² decay / volume triggers
- `src/agents/quality.py` — QualityAssuranceAgent (SNR, saturation gating)
- `src/models/cnn.py` — GasCNN (1D conv) + CNNGasClassifier with MC Dropout uncertainty
- `src/models/onnx_export.py` — ONNX export, numerical validation, OnnxInferenceWrapper
- `src/calibration/gpr.py` — GPRCalibration: Gaussian Process Δλ → ppm with uncertainty
- `src/scientific/lod.py` — IUPAC LOD (3σ), LOQ (10σ), sensitivity, sensor_performance_summary
- `src/training/cross_gas_eval.py` — Leave-one-gas-out (LOGO) cross-validation + MLflow
- `src/training/ablation.py` — Preprocessing ablation study (6 configs, 5-fold GPR CV)
- `src/inference/orchestrator.py` — SensorOrchestrator wiring drift + training agents into live pipeline
- `src/schemas/spectrum.py` — Pydantic `SpectrumReading` + `PredictionResult` data contracts
- `src/batch/aggregation.py` — Stable-block detection, canonical spectrum selection
- 339-test suite across 17 test files (0 failures)
- `Makefile` for common developer commands (`make test`, `make lint`, `make dashboard`, etc.)
- `LICENSE` (MIT)
- `CHANGELOG.md`
- `.github/ISSUE_TEMPLATE/` — bug report + feature request templates
- `.github/PULL_REQUEST_TEMPLATE.md`

### Changed
- MLflow: all scripts now use `ExperimentTracker` wrapper — no more raw `mlflow.*` calls scattered across files
- MLflow tracking URI unified to `experiments/mlruns/` across all scripts
- MLflow run names now include timestamps (no duplicate/overwrite runs)
- MLflow now logs result JSON artifacts alongside metrics
- `TrainingAgent` MLflow: adds `gpr_r2_delta` / `cnn_acc_delta` metrics; logs model files when promoted
- `log_figure()` failure level raised from DEBUG → WARNING
- Coverage threshold raised from 40% → 60%
- Style: full `src/` package upgraded to modern Python 3.9+ annotations (`list[x]`, `x | None`)
- `scripts/quality_gate.py` and CI workflow now include `src/` in mypy targets

### Fixed
- Real-time hardware pipeline: `connect()` not called before `start()`, `.wavelengths` attribute used correctly, unix float timestamps normalised to `datetime`
- GPR ConvergenceWarning noise in test output suppressed via `filterwarnings`
- `asyncio_mode = "auto"` removed from pytest config (pytest-asyncio not installed)
- NumPy 2.0 `np.trapz` deprecation → `getattr(np, "trapezoid", np.trapz)`

---

## [2.0.0] — 2025-02

### Added
- Agentic Pipeline dashboard tab (5-agent automation workflow)
- Reference spectrum subtraction for Δλ LSPR signal extraction
- Multi-ROI fusion with hybrid R²/slope-to-noise metric
- ICA spectral decomposition + MCR-ALS
- FastAPI REST inference server (`serve.py`)
- Docker + docker-compose support

### Changed
- Pipeline restructured into 4 explicit stages (preprocess → feature → calibrate → QC)
- LiveDataStore made thread-safe singleton
- Dashboard expanded to 4 tabs

---

## [1.0.0] — 2024-11

### Added
- Initial release: batch CSV ingestion, ALS baseline, Savitzky-Golay, GPR calibration
- Streamlit dashboard (Batch Analysis tab)
- ThorLabs CCS200 DLL + VISA acquisition drivers
- T90/T10 kinetics analysis
