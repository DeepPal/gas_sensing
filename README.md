# SpectraAgent

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-1305%20passing-brightgreen)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue)](https://mypy.readthedocs.io/)

**Universal agentic spectroscopy platform — from raw photons to calibrated results with AI-native analysis.**

SpectraAgent provides a complete, hardware-agnostic runtime for optical spectroscopy research: real-time acquisition from any spectrometer, physics-informed signal processing, conformal prediction calibration, and autonomous AI agents that explain anomalies, narrate experiments, and plan the next measurement. A plugin architecture makes it straightforward to add new hardware drivers and sensor physics models.

> **Reference deployment**: Localized Surface Plasmon Resonance (LSPR) sensing with ThorLabs CCS200, but the platform supports any spectrometer producing wavelength–intensity arrays.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [SpectraAgent (primary runtime)](#spectraagent-primary-runtime)
  - [Streamlit Dashboard (scientific analysis)](#streamlit-dashboard-scientific-analysis)
- [Plugin System](#plugin-system)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing & Quality](#testing--quality)
- [Scientific Capabilities](#scientific-capabilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Canonical Project Tracking

To keep status consistent across contributors and AI agents, treat the files
below as the canonical tracking set and update them together when state changes:

- `REMAINING_WORK.md` (backlog and open gaps)
- `PRODUCTION_READINESS.md` (deployment/operations readiness)
- `CHANGELOG.md` (auditable change history)
- `.github/workflows/security.yml` (enforced security gates)

---

## Key Features

| Category | Capabilities |
| --- | --- |
| **Acquisition** | ThorLabs CCS200 (DLL/VISA/Serial), simulated driver, plugin-extensible hardware |
| **Signal processing** | Savitzky-Golay, wavelet denoising, ALS/airPLS baseline, Lorentzian peak fit |
| **Calibration** | Physics-informed GPR (Langmuir mean function), PLS, conformal prediction CI |
| **AI agents** | Anomaly explainer, experiment narrator, diagnostics, report writer (Claude API) |
| **Quality agents** | SNR/saturation gate, rolling drift detection, Bayesian experiment designer |
| **IUPAC analytics** | LOD/LOQ/LOB triad with bootstrap CI (2000 iterations), Mandel F-test |
| **Session analysis** | T90/T10 response times, drift rate, linearity, selectivity matrix |
| **Runtimes** | FastAPI + React (live acquisition) · Streamlit (batch/scientific analysis) |
| **Reproducibility** | HDF5 session archives, MLflow experiment tracking, ONNX model export |
| **Plugin system** | `spectraagent.hardware` and `spectraagent.sensor_physics` entry-points |

---

## Architecture

SpectraAgent has two complementary runtimes:

```text
┌─────────────────────────────────────────────────────────────────┐
│                    SpectraAgent Runtime                         │
│  spectraagent start [--simulate] [--host] [--port]              │
│                                                                  │
│  ┌─────────────────────────────┐   ┌──────────────────────────┐ │
│  │  React Frontend             │   │  Daemon Acquisition      │ │
│  │  (WebSocket + REST)         │   │  Thread  (~2–20 Hz)      │ │
│  │  • Live spectrum chart      │   │  CCS200 DLL / VISA /     │ │
│  │  • Agent log panel          │   │  Simulation              │ │
│  │  • Session controls         │   └──────────┬───────────────┘ │
│  └──────────────┬──────────────┘              │                 │
│                 │                   ┌──────────▼───────────────┐ │
│  ┌──────────────▼──────────────┐   │  AgentBus                │ │
│  │  FastAPI Server             │   │  (thread-safe bridge)    │ │
│  │  /api/sessions              │   │  call_soon_threadsafe    │ │
│  │  /api/agents                │◄──│  fans out to WS + JSONL  │ │
│  │  /ws/live                   │   └──────────┬───────────────┘ │
│  └──────────────────────────── ┘              │                 │
│                                   ┌──────────▼───────────────┐ │
│                                   │  Per-frame Hot Path       │ │
│                                   │  QualityAgent (SNR gate)  │ │
│                                   │  DriftAgent (peak shift)  │ │
│                                   │  CalibrationAgent (GPR)   │ │
│                                   └──────────────────────────┘ │
│                                                                  │
│  Claude API agents (async, never in hot path):                   │
│  AnomalyExplainer · ExperimentNarrator · DiagnosticsAgent        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                           │
│  streamlit run dashboard/app.py                                 │
│                                                                  │
│  Tab 1: Guided Calibration   — step-by-step calibration workflow  │
│  Tab 2: Experiments          — session browser + cross-session   │
│  Tab 3: Batch Analysis       — load data, heatmaps, curves       │
│  Tab 4: Live Sensor          — real-time CCS200 feed             │
│  Tab 5: Data-Driven Science  — ML training, figures, publishing  │
└─────────────────────────────────────────────────────────────────┘
```

### Scientific pipeline (shared by both runtimes)

```text
Raw Spectrum
    │
    ▼  Stage 1: Preprocessing
    │  Savitzky-Golay smooth → ALS/airPLS baseline → wavelet denoise
    │
    ▼  Stage 2: Feature Extraction
    │  Lorentzian peak fit → LSPR Δλ, ΔI_peak, ΔI_area, ΔI_std
    │
    ▼  Stage 3: Calibration
    │  Physics-informed GPR (Langmuir) + conformal prediction CI
    │  → concentration [ppm] ± coverage-guaranteed bounds
    │
    ▼  Stage 4: Quality Control
       SNR gate · saturation check · drift alert
       → PipelineResult{concentration, ci_low, ci_high, snr, flags}
```

---

## Installation

### Prerequisites

- Python 3.9 or later
- Windows 10/11, macOS, or Linux
- ThorLabs CCS200 spectrometer *(optional — simulation mode works without hardware)*

### 1. Clone

```bash
git clone <repo-url>
cd spectraagent
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install

```bash
# Core install (all scientific and API dependencies)
pip install -r requirements.txt

# Or as an editable package (recommended for development)
pip install -e ".[dev]"

# With PyTorch (CNN gas classifier)
pip install -e ".[dev,ml]"

# With hardware VISA support
pip install -e ".[dev,ml,hardware]"
```

> **PyTorch note**: ~2 GB download. If you only need calibration and signal processing, the platform degrades gracefully without it — CNN classification is skipped, all other features remain active.

### 4. Verify installation

```bash
spectraagent --version
spectraagent plugins list      # shows discovered hardware + physics plugins
```

---

## Quick Start

### SpectraAgent (primary runtime)

Start the server with simulated hardware (no spectrometer needed):

```bash
spectraagent start --simulate
# → FastAPI server at http://localhost:8765
# → React frontend at http://localhost:8765/app
# → API docs at http://localhost:8765/docs
```

Start with real hardware:

```bash
spectraagent start
# Auto-discovers ThorLabs CCS200 via plugin registry
```

Key CLI options:

| Flag | Default | Description |
| --- | --- | --- |
| `--simulate` | off | Use simulated spectrometer driver |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |
| `--no-browser` | off | Suppress auto-open browser |
| `--integration-time` | `50` | CCS200 integration time (ms) |

#### Session management

```bash
# List recorded sessions
spectraagent sessions list

# Export a session to HDF5
spectraagent sessions export <session-id> --format hdf5

# Generate PDF report
spectraagent sessions report <session-id>
```

### Streamlit Dashboard (scientific analysis)

```bash
# Windows
.venv\Scripts\python.exe -m streamlit run dashboard/app.py

# macOS / Linux
python -m streamlit run dashboard/app.py

# Or via helper script (Windows, includes authentication)
run_dashboard_secure.bat
```

Open `http://localhost:8501`.

### Legacy CLI (batch pipelines)

```bash
python run.py --mode simulate --duration 30
python run.py --mode batch --data data/JOY_Data/Ethanol
python run.py --mode sensor --gas Ethanol --duration 3600
```

---

## Plugin System

SpectraAgent discovers hardware drivers and sensor physics models at runtime via Python entry-points:

```toml
# pyproject.toml
[project.entry-points."spectraagent.hardware"]
thorlabs_ccs = "spectraagent.drivers.thorlabs:ThorlabsCCSDriver"
simulation   = "spectraagent.drivers.simulation:SimulationDriver"

[project.entry-points."spectraagent.sensor_physics"]
lspr = "spectraagent.physics.lspr:LSPRPlugin"
```

To add a new spectrometer (e.g., Ocean Optics):

```python
# ocean_driver/driver.py
from spectraagent.drivers.base import AbstractSpectrometerDriver

class OceanOpticsDriver(AbstractSpectrometerDriver):
    name = "ocean_optics"

    def acquire(self) -> tuple[np.ndarray, np.ndarray]:
        ...  # return (wavelengths, intensities)
```

```toml
# your package's pyproject.toml
[project.entry-points."spectraagent.hardware"]
ocean_optics = "ocean_driver.driver:OceanOpticsDriver"
```

After `pip install`, it appears in `spectraagent plugins list` automatically.

---

## Project Structure

```text
spectraagent/
├── spectraagent/               ← Primary runtime package
│   ├── __main__.py             ← CLI entry point (spectraagent start/sessions/plugins)
│   ├── drivers/
│   │   ├── thorlabs.py         ← ThorLabs CCS200 driver (DLL + VISA)
│   │   ├── simulation.py       ← SimulationDriver (Gaussian peak + noise model)
│   │   └── validation.py       ← Driver contract validation
│   ├── physics/
│   │   └── lspr.py             ← LSPRPlugin: Δλ extraction, Langmuir isotherm
│   └── webapp/
│       ├── server.py           ← FastAPI application + WebSocket /ws/live
│       ├── agent_bus.py        ← AgentBus: thread-safe acquisition → async bridge
│       ├── session_writer.py   ← Per-session JSONL + CSV streaming
│       ├── agents/
│       │   ├── quality.py      ← QualityAgent (SNR/saturation gate)
│       │   ├── drift.py        ← DriftAgent (rolling peak-shift monitor)
│       │   ├── planner.py      ← ExperimentPlannerAgent (Bayesian designer)
│       │   └── claude_agents.py← AnomalyExplainer, ExperimentNarrator, DiagnosticsAgent
│       └── frontend/           ← React + TypeScript frontend (Vite)
│
├── src/                        ← Scientific library (hardware-agnostic)
│   ├── public_api.py           ← Stable public façade
│   ├── spectrometer/           ← SpectrometerRegistry + AbstractSpectrometer
│   ├── calibration/            ← GPR, PLS, conformal prediction, physics kernel
│   ├── inference/              ← RealTimePipeline, SessionAnalyzer
│   ├── features/               ← LSPR peak extraction, multi-ROI fusion
│   ├── preprocessing/          ← Smoothing, baseline, denoising
│   ├── models/                 ← CNN classifier, ONNX export
│   ├── scientific/             ← LOD/LOQ/LOB (IUPAC), selectivity matrix
│   ├── reporting/              ← Metrics, plots, publication figures, PDF
│   ├── io/                     ← HDF5 session archives
│   └── experiment_tracking.py  ← MLflow wrapper
│
├── dashboard/                  ← Streamlit dashboard (4 tabs)
│   ├── app.py
│   ├── agentic_pipeline_tab.py
│   ├── auth.py                 ← Token-based access control
│   ├── security.py             ← Rate limiting, audit log
│   └── health.py               ← Health check endpoint
│
├── gas_analysis/               ← Hardware acquisition layer
│   └── acquisition/            ← CCS200 DLL/VISA/Serial drivers
│
├── tests/                      ← 1187 tests, 0 failures
│   ├── spectraagent/           ← SpectraAgent runtime tests
│   └── src/                    ← Scientific library tests
│
├── docs/                       ← MkDocs documentation
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── ENGINEERING_STANDARDS.md
│   └── guides/
│
├── spectraagent.toml           ← Platform configuration
├── pyproject.toml              ← Build config, entry-points, tool config
├── requirements.txt            ← Pinned runtime dependencies
└── Makefile                    ← Developer targets
```

---

## Configuration

Platform configuration lives in [`spectraagent.toml`](spectraagent.toml):

```toml
[hardware]
driver = "thorlabs_ccs"     # or "simulation"
integration_time_ms = 50
warmup_frames = 3

[physics]
plugin = "lspr"
reference_wavelength_nm = 532.0

[agents]
enable_claude = true        # requires ANTHROPIC_API_KEY
drift_window = 50           # frames for rolling drift detection
snr_threshold = 3.0

[session]
output_dir = "output/sessions"
hdf5_archive = true
```

Pipeline parameters (preprocessing, calibration, quality) live in [`config/config.yaml`](config/config.yaml).

---

## Testing & Quality

### Run tests

```bash
make test                   # full suite (1187 tests)
make test-fast              # fast lane (exclude reliability tests)
make test-reliability       # lifecycle/stability tests
make coverage               # with HTML coverage report
```

Or directly:

```bash
pytest                                      # all tests
pytest tests/spectraagent/                  # SpectraAgent runtime only
pytest tests/src/                           # scientific library only
pytest -m "not reliability" -x --tb=short  # fast lane
```

### Quality gate (mirrors CI)

```bash
make quality-gate           # ruff + mypy + pytest + reliability report
make check                  # lint + test (quick pre-commit check)
make lint                   # ruff linting only
```

### CI

GitHub Actions runs ruff, mypy, and pytest (two lanes) on every push/PR:

- **Fast lane**: `pytest -m "not reliability"` — quick regression feedback
- **Reliability lane**: `pytest -m "reliability"` — subprocess/lifecycle/soak stability
- **Nightly**: extended reliability with runtime budget enforcement

---

## Scientific Capabilities

### IUPAC LOD/LOQ/LOB (automatic, per session)

```python
from src.public_api import RealTimePipeline, PipelineConfig

pipeline = RealTimePipeline(PipelineConfig())
# ... acquire spectra ...

analyzer = pipeline.get_session_analyzer()
results = analyzer.analyze()

print(f"LOD  = {results.lod:.4f} ppm  (95% CI: {results.lod_ci})")
print(f"LOQ  = {results.loq:.4f} ppm")
print(f"LOB  = {results.lob:.4f} ppm")
print(f"T90  = {results.t90:.1f} s")
print(f"Drift rate = {results.drift_rate:.4f} nm/frame")
```

### Conformal prediction (coverage-guaranteed CI)

```python
from src.calibration.conformal import ConformalCalibrator

cal = ConformalCalibrator(base_model=gpr_model, coverage=0.95)
cal.calibrate(X_cal, y_cal)

pred = cal.predict(spectrum)
# pred.concentration, pred.ci_low, pred.ci_high
# Provable 95% coverage on held-out data
```

### Bayesian experiment designer

```python
from spectraagent.webapp.agents.planner import ExperimentPlannerAgent

planner = ExperimentPlannerAgent()
next_conc = planner.suggest_next(measured_concentrations, measured_responses)
# Logspace max-variance acquisition for optimal calibration curve coverage
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `DLL error -1073807343` | CCS200 connected but not powered | Power on spectrometer before starting |
| `VI_ERROR_TMO (-1073807339)` | Stale VISA handle from crash | Unplug/replug USB; ensure `close()` in finally block |
| `spectraagent plugins list` shows no hardware | Package not editable-installed | `pip install -e .` from repo root |
| React frontend blank | Frontend not built | `cd spectraagent/webapp/frontend && npm install && npm run build` |
| `torch` import error | PyTorch not installed | `pip install -e ".[ml]"`; platform degrades gracefully |
| `pyvisa not found` | VISA backend missing | `pip install pyvisa pyvisa-py` |
| UTF-8 console errors on Windows | Default cp1252 encoding | Set `PYTHONIOENCODING=utf-8` or use `run_spectraagent.bat` |
| Coverage low warnings | Legacy modules excluded | Check `[tool.coverage.run]` omit list in pyproject.toml |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ENGINEERING_STANDARDS.md](docs/ENGINEERING_STANDARDS.md).

- Run `make check` before opening a PR
- New hardware drivers: implement `AbstractSpectrometerDriver`, register via entry-point
- New physics plugins: implement `AbstractSensorPhysicsPlugin`, register via entry-point
- All new scientific methods require a corresponding test in `tests/`

---

## License

MIT — see [LICENSE](LICENSE).
