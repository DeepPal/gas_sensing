# System Architecture

SpectraAgent — Universal Agentic Spectroscopy Platform

Version 2.0 | Last updated 2026-03

---

## Overview

SpectraAgent is a **hardware-agnostic, physics-plugin-based spectroscopy platform**. Any spectrometer and any sensor physics model can be plugged in via Python entry-points — the core platform has no hard dependency on a specific instrument or sensing modality.

The current reference deployment uses a Thorlabs CCS200 spectrometer with an LSPR sensor, but this is one configuration among many. The platform supports fluorescence, absorbance, and Raman sensing through the same plugin interface.

The platform has two complementary runtime paths that serve different audiences:

| Path | Entry point | Audience |
|------|------------|----------|
| **SpectraAgent** (primary) | `python -m spectraagent start` | Live acquisition, real-time inference, Claude AI agents |
| **Research Dashboard** | `streamlit run dashboard/app.py` | Scientific analysis, batch calibration, publication figures |

Both paths share the same `src/` science library. Neither depends on the other.

---

## 1. SpectraAgent Runtime

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         SPECTRAAGENT RUNTIME                               │
│                  python -m spectraagent start [--simulate]                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Hardware Layer              Acquisition Thread (daemon, ~2–20 Hz)        │
│  ┌──────────────────┐        ┌─────────────────────────────────────────┐  │
│  │ ThorlabsDriver   │──────▶ │ _acquisition_loop()                     │  │
│  │ SimulationDriver │        │  ├─ driver.read_spectrum()               │  │
│  └──────────────────┘        │  ├─ QualityAgent.process()  (SNR/sat.)  │  │
│  (loaded via entry-point     │  ├─ DriftAgent.update()    (peak shift) │  │
│   spectraagent.hardware)     │  ├─ RealTimePipeline.process_spectrum() │  │
│                              │  └─ AgentBus.emit() + WS broadcast      │  │
│                              └─────────────────────────────────────────┘  │
│                                           │                                │
│                                           ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    AgentBus  (thread-safe bridge)                   │  │
│  │  call_soon_threadsafe → asyncio queue per subscriber               │  │
│  │  Also writes every event to agent_events.jsonl (session log)       │  │
│  └──────────┬───────────────────────────────────────────┬─────────────┘  │
│             │                                           │                  │
│             ▼ (asyncio event loop)                      ▼                  │
│  ┌────────────────────────┐               ┌─────────────────────────────┐ │
│  │  Deterministic Agents  │               │     Claude API Agents       │ │
│  │  ─────────────────     │               │  ────────────────────────   │ │
│  │  QualityAgent          │               │  AnomalyExplainer           │ │
│  │  DriftAgent            │               │  ExperimentNarrator         │ │
│  │  CalibrationAgent      │               │  DiagnosticsAgent           │ │
│  │  ExperimentPlannerAgent│               │  ReportWriter               │ │
│  └────────────────────────┘               └─────────────────────────────┘ │
│                                                                            │
│  FastAPI + WebSocket Server (spectraagent/webapp/server.py)                │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  GET  /health                   — hardware status, version, mode    │ │
│  │  POST /acquisition/config       — set integration time, gas label   │ │
│  │  POST /acquisition/start        — begin session, returns session_id │ │
│  │  POST /acquisition/stop         — end session, runs SessionAnalyzer │ │
│  │  POST /acquisition/reference    — capture reference spectrum        │ │
│  │  GET  /calibration/suggest      — next concentration (BED)         │ │
│  │  POST /calibration/add_point    — add (concentration, shift) point  │ │
│  │  POST /agents/ask               — SSE streaming Claude query        │ │
│  │  POST /agents/settings          — toggle auto_explain               │ │
│  │  POST /reports/generate         — trigger ReportWriter              │ │
│  │  GET  /sessions                 — list past sessions                │ │
│  │  GET  /sessions/{id}            — fetch session events + metadata   │ │
│  │  WS   /ws/spectrum              — live spectrum frames (JSON)       │ │
│  │  WS   /ws/agent-events          — live AgentBus events (JSON)       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  React Frontend (spectraagent/webapp/frontend/ — Vite + TypeScript)        │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Live spectrum chart  │  Agent event feed  │  Calibration wizard    │ │
│  │  HW badge (Live/Sim)  │  Session controls  │  Ask Claude panel      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 Startup sequence

```
spectraagent start
  1. Load config (spectraagent.toml)
  2. Load hardware driver (entry-point spectraagent.hardware)
       → ThorlabsDriver if CCS200 found, else SimulationDriver
  3. Load physics plugin (entry-point spectraagent.sensor_physics)
       → LSPRPhysicsPlugin (default)
  4. Build FastAPI app, wire AgentBus
  5. Instantiate all agents
  6. Wire RealTimePipeline
  7. Register startup callback → acquisition daemon thread starts
  8. Open browser (http://localhost:8765 by default)
  9. uvicorn.run() — blocking
```

### 1.2 Per-frame hot path (20 Hz)

```
driver.read_spectrum(wl, I)
  → QualityAgent.process(frame, wl, I)
      ├─ PASS → continue
      └─ FAIL (saturation) → drop frame, emit quality_error
  → plugin.detect_peak(wl, I)  → direct_peak_wl
  → DriftAgent.update(frame, peak_wl) → may emit drift_warn
  → RealTimePipeline.process_spectrum(wl, I)
      Stage 1: preprocessing (ALS baseline, S-G denoising)
      Stage 2: LSPR feature extraction (Δλ, ΔI_peak, ΔI_area, ΔI_std)
      Stage 3: GPR calibration → concentration_ppm + CI [low, high]
      Stage 4: CNN classification → gas_type + confidence_score
  → SessionWriter.append_frame_result(row)  [crash-safe CSV]
  → Broadcaster.broadcast(JSON payload)  [WebSocket fan-out]
```

### 1.3 Session lifecycle

```
POST /acquisition/start
  → session_id = YYYYMMDD_HHMMSS
  → session_running = True
  → output/sessions/{session_id}/pipeline_results.csv created

POST /acquisition/reference
  → captures current spectrum as reference
  → RealTimePipeline receives reference for Δλ calculation

POST /acquisition/stop
  → session_running = False
  → SessionAnalyzer.analyze(events) runs automatically
  → LOD/LOQ/drift/T90/T10 computed, stored in session_meta.json
  → session_complete event emitted to AgentBus
```

---

## 2. Research Dashboard Runtime

```
streamlit run dashboard/app.py
```

Four tabs:

| Tab | File | Purpose |
|-----|------|---------|
| Batch Analysis | `dashboard/app.py` (tabs 1–2) | Load Joy_Data CSVs, run calibration pipeline |
| Agentic Pipeline | `dashboard/agentic_pipeline_tab.py` | Step-by-step guided workflow with agents |
| Live Sensor | `dashboard/sensor_dashboard.py` | Legacy live view (pre-SpectraAgent) |
| Reports | `dashboard/app.py` (tab 4) | Export publication-quality figures |

Key features of the Agentic Pipeline tab:
- Step 1: Load raw spectra from Joy_Data folders
- Step 2: Load reference spectrum → compute `diff_signal = raw − ref_interp`
- Step 3: LSPR feature extraction using Δλ when reference available
- Step 4: GPR calibration fit → LOD/LOQ/LOB + bootstrap CI
- Step 5: Isotherm model selection (Langmuir / Freundlich / Hill, Mandel gate)
- Step 6: Session summary + reproducibility manifest

---

## 3. Shared Science Library (`src/`)

```
src/
├── preprocessing/          Spectrum preprocessing
│   ├── baseline.py         ALS asymmetric least-squares baseline correction
│   ├── denoising.py        Savitzky-Golay + wavelet denoising
│   └── quality.py          SNR estimation, saturation detection
│
├── calibration/            Calibration and uncertainty
│   ├── gpr.py              Gaussian Process Regressor (scikit-learn wrapper)
│   ├── physics_kernel.py   Physics-informed GPR — Langmuir isotherm mean function
│   ├── pls.py              PLS calibration with LOOCV and VIP scores
│   ├── conformal.py        Split conformal prediction (normalised scores, coverage guarantee)
│   ├── active_learning.py  BayesianExperimentDesigner — logspace max-variance acquisition
│   ├── isotherms.py        Langmuir / Freundlich / Hill isotherm fitting
│   ├── roi_scan.py         ROI scan and concentration-response computation
│   ├── multi_roi.py        Multi-ROI calibration
│   ├── transforms.py       Signal transforms for calibration
│   ├── batch_reproducibility.py  Batch sensor QC (pooled LOD, RSD, accept/reject)
│   └── selectivity.py      Cross-reactivity coefficients (IUPAC K values)
│
├── features/               Feature extraction
│   └── lspr_features.py    LSPR features: [Δλ, ΔI_peak, ΔI_area, ΔI_std]
│
├── models/                 ML models
│   ├── cnn.py              CNN gas classifier (PyTorch)
│   └── onnx_export.py      ONNX export with numerical validation
│
├── scientific/             Scientific metrics (IUPAC-compliant)
│   ├── lod.py              LOD/LOQ/LOB triad — bootstrap CI, blank-based and residual-based
│   ├── regression.py       Weighted linear, Theil-Sen, RANSAC
│   └── selectivity.py      Selectivity matrix and from-calibration-data helper
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
│   ├── registry.py         SpectrometerRegistry — register/discover/create drivers
│   ├── simulated.py        SimulatedSpectrometer (LSPR, fluorescence, absorbance modes)
│   └── ccs200_adapter.py   CCS200Adapter — wraps native DLL driver
│
├── batch/                  Batch processing
│   ├── preprocessing.py    Multi-frame preprocessing pipeline
│   ├── response.py         Concentration-response aggregation
│   ├── aggregation.py      Stable-plateau detection, canonical spectrum builder
│   └── time_series.py      Response time-series extraction
│
├── reporting/              Reporting and figures
│   ├── metrics.py          LOD/SNR/QC metric computation
│   ├── plots.py            Calibration curves, spectral overlays, ROI diagnostics
│   ├── publication.py      Publication-quality figure generation
│   ├── environment.py      Environment metadata capture
│   └── io.py               Save canonical spectra, JSON reports, CSV outputs
│
├── agents/                 Signal-path agents
│   ├── quality.py          QualityAgent (used by RealTimePipeline)
│   └── training.py         TrainingAgent — auto-retrain on drift / R² decay
│
├── training/               Training scripts
│   ├── train_gpr.py        GPR training with MLflow tracking
│   ├── train_cnn.py        CNN training with ablation config
│   ├── ablation.py         Ablation study runner
│   └── cross_gas_eval.py   Cross-gas sensitivity evaluation
│
└── public_api.py           Stable commercial facade — re-exports key classes
```

---

## 4. Plugin Architecture

SpectraAgent uses Python entry-points for hardware and physics plugins.

### 4.1 Hardware drivers (`spectraagent.hardware`)

```toml
# pyproject.toml
[project.entry-points."spectraagent.hardware"]
thorlabs_ccs200 = "spectraagent.drivers.thorlabs:ThorlabsDriver"
```

Implementing a new driver:

```python
from spectraagent.drivers.base import AbstractHardwareDriver
import numpy as np

class MyDriver(AbstractHardwareDriver):
    @property
    def name(self) -> str: return "MyInstrument"
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_wavelengths(self) -> np.ndarray: ...
    def read_spectrum(self) -> np.ndarray: ...
    def set_integration_time_ms(self, ms: float) -> None: ...
    def is_connected(self) -> bool: ...
```

Register it in `pyproject.toml` and it will be auto-discovered by `spectraagent start`.

### 4.2 Physics plugins (`spectraagent.sensor_physics`)

```toml
[project.entry-points."spectraagent.sensor_physics"]
lspr = "spectraagent.physics.lspr:LSPRPhysicsPlugin"
```

The physics plugin provides `detect_peak()`, `extract_features()`, and `calibration_priors()`.

### 4.3 Research-facing driver registry (`src/spectrometer`)

For scripts and notebooks that don't use the full SpectraAgent runtime:

```python
from src.spectrometer.registry import SpectrometerRegistry

# List available drivers
SpectrometerRegistry.available()  # ['ccs200', 'sim', 'simulated', 'thorlabs_ccs200']

# Create and use as context manager
with SpectrometerRegistry.create("simulated") as spec:
    spec.open()
    frame = spec.acquire()
    print(frame.peak_wavelength, frame.snr)

# Register a custom driver
@SpectrometerRegistry.register("usb2000")
class USB2000Driver(AbstractSpectrometer):
    ...
```

---

## 5. Data Flow Summary

```
Physical sensor (LSPR sensor on CCS200)
          │
          │  USB (TLCCS DLL)
          ▼
  ThorlabsDriver.read_spectrum()          ~2.4 Hz (50 ms integration)
          │
          ▼
  _acquisition_loop() in daemon thread
     ├── QualityAgent    → SNR check, saturation gate
     ├── DriftAgent      → peak shift trend (rolling window)
     ├── RealTimePipeline→ Stage1: ALS+S-G preprocessing
     │                     Stage2: LSPR Δλ feature extraction
     │                     Stage3: GPR → [concentration_ppm, CI_low, CI_high]
     │                     Stage4: CNN → [gas_type, confidence_score]
     ├── SessionWriter   → per-frame CSV (crash-safe)
     └── Broadcaster     → WebSocket JSON to React frontend

  AgentBus (thread-safe)
     ├── QualityAgent    → emits quality_ok / quality_warn / quality_error
     ├── DriftAgent      → emits drift_ok / drift_warn
     ├── CalibrationAgent→ emits model_selected (on sufficient data)
     ├── ExperimentPlannerAgent → emits experiment_suggestion (BED)
     └── ClaudeAgentRunner (asyncio)
           ├── AnomalyExplainer    → reacts to drift_warn (opt-in)
           ├── ExperimentNarrator  → reacts to model_selected (opt-in)
           └── DiagnosticsAgent    → reacts to hardware_error (always)

  Session stop
     └── SessionAnalyzer → LOD/LOQ/LOB, bootstrap CI, T90/T10,
                           drift rate, linearity, selectivity
                         → session_meta.json
```

---

## 6. Session Storage

```
output/sessions/
└── YYYYMMDD_HHMMSS/
    ├── pipeline_results.csv      per-frame: timestamp, peak_wl, shift, conc, CI, SNR, gas_type
    ├── session_meta.json         LOD/LOQ/LOB, T90/T10, drift_rate, linearity, bootstrap CI
    └── agent_events.jsonl        every AgentBus event (quality, drift, claude, calibration)
```

---

## 7. Configuration

`spectraagent.toml` (auto-created at first run):

```toml
[server]
host = "127.0.0.1"
port = 8765
open_browser = true

[hardware]
default_driver = "thorlabs_ccs200"
integration_time_ms = 50.0

[physics]
default_plugin = "lspr"

[claude]
model = "claude-sonnet-4-6"
timeout_s = 30.0

[agents]
auto_explain = false
anomaly_explainer_cooldown_s = 60.0
diagnostics_cooldown_s = 300.0
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Daemon acquisition thread, not asyncio | Hardware DLL calls block; asyncio can't yield inside ctypes |
| AgentBus as thread bridge | `call_soon_threadsafe` is the safe pattern for thread→asyncio hand-off |
| Claude agents in asyncio, not signal thread | LLM calls are I/O bound and should never block 20 Hz acquisition |
| SpectraAgent and Streamlit as separate runtimes | Different user models: live acquisition vs. scientific batch analysis |
| entry-points for hardware plugins | Allows third-party drivers without modifying the core package |
| Split conformal prediction for CI | Coverage guarantee is provably correct; GPR posterior CI alone is not calibrated |
| Physics-informed GPR with Langmuir | Prevents physically impossible extrapolation; better LOD at sparse calibration points |

See `docs/adr/` for full Architectural Decision Records.
