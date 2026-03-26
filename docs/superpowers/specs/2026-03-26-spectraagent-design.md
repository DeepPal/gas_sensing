# SpectraAgent — Universal Agentic Spectroscopy Platform
**Design Document · 2026-03-26**

---

## 1. Vision and Scope

SpectraAgent is a universal, AI-native spectrometer analysis platform. It is not an LSPR tool upgrade — it is a new product category: the first open-source platform that combines deterministic signal science with autonomous LLM-based reasoning for any spectroscopic sensor.

**Target user:** A research scientist (physics/chemistry PhD) running spectroscopic experiments who needs real-time AI guidance, calibration, and publication-ready reports without learning a commercial software stack.

**Not in scope (v1.0):** Multi-user auth, cloud hosting, Raman-specific deconvolution, FTIR.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER'S BROWSER  (localhost:8765)                                   │
│                                                                     │
│  React SPA — 5 tabs                                                 │
│  ┌──────────────┬──────────────┬──────────────┬──────────┬──────┐  │
│  │  Live Sensor │  Experiment  │ Batch Analysis│ Agentic  │Agent│  │
│  │  uPlot 20Hz  │  Acq Config  │ File/Folder  │ Pipeline │Consol│  │
│  │  WebSocket   │  + Results   │ + Reports    │ Steps 1-4│ Log │  │
│  └──────────────┴──────────────┴──────────────┴──────────┴──────┘  │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP REST + WebSocket
┌────────────────────────▼────────────────────────────────────────────┐
│  FastAPI Backend  (webapp/server.py)                                │
│                                                                     │
│  /api/acquisition/*    /api/pipeline/*    /api/sessions/*          │
│  /api/calibration/*    /api/reports/*     /api/health              │
│  /ws/spectrum          /ws/trend          /ws/agent-events         │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ SensorOrchest.  │  │  ReportBuilder   │  │  SessionRegistry │  │
│  │ (existing)      │  │  (new)           │  │  (new)           │  │
│  └────────┬────────┘  └──────────────────┘  └──────────────────┘  │
│           │                                                         │
│  ┌────────▼────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ RealTimePipeline│  │  AgentBus        │  │  StructuredLogger│  │
│  │ + LSPRReference │  │  (det. + Claude) │  │  → agent_events  │  │
│  │ (existing+fixed)│  │  (new)           │  │  .jsonl (new)    │  │
│  └────────┬────────┘  └──────────────────┘  └──────────────────┘  │
└───────────┼─────────────────────────────────────────────────────────┘
            │ DLL / VISA / Serial
┌───────────▼───────────┐      ┌────────────────────────────────────┐
│  Hardware Plugin       │      │  output/sessions/YYYYMMDD_HHMMSS/ │
│  (ThorLabs CCS200 /   │      │  ├── raw_spectra.parquet           │
│   Ocean Optics /      │      │  ├── pipeline_results.csv          │
│   Simulation)         │      │  ├── session_meta.json             │
└───────────────────────┘      │  ├── agent_events.jsonl            │
                               │  ├── report.html                   │
                               │  └── report.pdf                    │
                               └────────────────────────────────────┘
```

### Key structural decisions

- `webapp/` replaces `dashboard/` — same Python process, one port (8765), browser auto-opens on `spectraagent start`
- React SPA built with Vite; static files served by FastAPI under `/` in production
- All existing `src/` code is preserved — FastAPI routes call into existing `SensorOrchestrator`, `RealTimePipeline`, `src/reporting/` modules
- WebSocket endpoints from `live_server.py` are moved into the unified FastAPI app
- New `/ws/agent-events` stream pushes deterministic agent events and Claude API responses to the Agent Console tab in real time

---

## 3. Frontend — React SPA (5 Tabs)

### Tab 1: Live Sensor
- uPlot chart at 20 Hz via `/ws/spectrum` (existing, already working)
- Trend chart: Δλ vs time at 200 ms updates via `/ws/trend`
- Hardware status badge (connected / simulation / error)
- Reference capture button

### Tab 2: Experiment
- Acquisition config: integration time, gas label, target concentration
- Run/stop controls
- Live result cards: Δλ, SNR, quality gate status, estimated concentration
- Session name auto-assigned (datetime + gas + conc)

### Tab 3: Batch Analysis
- Folder picker → processes all CSV/parquet files in directory
- Lorentzian sub-pixel peak detection on each file
- Results table with export to CSV
- Session comparison overlay

### Tab 4: Agentic Pipeline (4 steps)
- Step 1: Data ingestion (Joy_Data loader or session file picker)
- Step 2: Reference subtraction (diff_signal = raw − ref_interp)
- Step 3: Feature extraction + isotherm fitting (Langmuir/Freundlich/Hill, AIC-selected)
- Step 4: LOD/LOQ, selectivity matrix, publication report export

### Tab 5: Agent Console
- Real-time event log streamed from `/ws/agent-events`
- Color-coded by source: QualityAgent (green), DriftAgent (amber), CalibrationAgent (blue), Claude (purple)
- "Ask Claude" button — opens free-text query box, sends structured session context to Claude API, streams response into log
- Filter by agent type; search events

---

## 4. Agentic AI Architecture (Hybrid)

### Layer 1 — Deterministic Agents (always running, <1 ms, no API cost)

| Agent | Trigger | Action |
|---|---|---|
| **QualityAgent** | Every frame | SNR gate, saturation check, outlier flag. Hard-blocks on saturation only. |
| **DriftAgent** | Rolling 60-frame window | CUSUM peak-shift trend. Emits WARN when drift rate > threshold. |
| **CalibrationAgent** | After each calibration point | AIC model selection: Langmuir vs Freundlich vs Hill vs Linear. Triggers retraining when R² drops. |
| **ExperimentPlannerAgent** | On user request or CalibrationAgent trigger | Bayesian optimization (BoTorch) to suggest next analyte concentration for maximum information gain. |

### Layer 2 — Claude API Agents (on-demand, async, never in signal path)

| Agent | Trigger | Output |
|---|---|---|
| **AnomalyExplainer** | DriftAgent WARN event | Plain-English explanation of drift pattern with likely cause and recommended action. |
| **ExperimentNarrator** | ExperimentPlannerAgent suggestion | Explanation of *why* that concentration was chosen and what uncertainty it reduces. |
| **ReportWriter** | User clicks "Generate Report" | Methods + Results sections in journal style (Sensors and Actuators B format) with actual session numbers filled in. |
| **DiagnosticsAgent** | Hardware error event | Explanation of error code and step-by-step fix suggestion. |

### Hard rule: Claude never receives raw signal data

```
Hardware → Signal Engine → Det. Agents → Claude API
(3648 px)   (Δλ, SNR)     (quality=OK)   (structured JSON only)
```

Claude receives structured results only:
```json
{"delta_lambda": -0.71, "snr": 62.4, "drift_rate": 0.002, "quality": "ok", "sensor": "LSPR", "gas": "Ethanol", "concentration": 0.1}
```

### Agent Bus

`AgentBus` is the internal pub/sub broker:
- Deterministic agents emit typed `AgentEvent` objects
- `AgentBus` routes events to: (a) `/ws/agent-events` WebSocket, (b) `agent_events.jsonl` log, (c) Claude API trigger conditions
- Claude responses are emitted back as `ClaudeEvent` objects through the same bus
- This keeps Claude fully decoupled from the signal pipeline

---

## 5. Plugin Architecture

### Hardware Drivers
Registered via `pyproject.toml` entry point group `spectraagent.hardware`.

| Plugin | Status |
|---|---|
| ThorLabs CCS200 | existing ✓ |
| Simulation (built-in) | existing ✓ |
| Ocean Optics USB2000+/Flame | v1.1 |
| Avantes AvaSpec | v1.2 |
| Custom user plugin | entry-point API |

### Sensor Physics Models
Registered via entry point group `spectraagent.sensor_physics`.

Each plugin must implement: `detect_peak()`, `extract_features()`, `calibration_priors()`, `expected_signal_shape()`.

| Plugin | Status |
|---|---|
| LSPR (Au nanoparticles) | existing ✓ |
| SPR (prism/fiber) | v1.1 |
| UV-Vis Absorption (Beer-Lambert) | v1.1 |
| NIR / NIRS | v1.2 |
| Raman / fluorescence | community |

Third-party plugins are auto-discovered at startup via `importlib.metadata.entry_points()`.

---

## 6. Packaging and CLI

### Entry point
```
spectraagent/__main__.py  →  cli() function (Typer)
```

### Commands
```bash
spectraagent start                              # start server, open browser
spectraagent start --simulate                  # force simulation mode
spectraagent start --host 0.0.0.0 --port 8765 # LAN-accessible
spectraagent report <session_dir> --format pdf # offline report generation
spectraagent recover <session_dir>             # merge flush partitions after crash
spectraagent plugins list                       # show installed plugins
```

### pyproject.toml wiring
```toml
[project.scripts]
spectraagent = "spectraagent.__main__:cli"

[project.entry-points."spectraagent.hardware"]
thorlabs_ccs = "spectraagent.drivers.thorlabs:ThorlabsCCSDriver"
simulation   = "spectraagent.drivers.simulation:SimulationDriver"

[project.entry-points."spectraagent.sensor_physics"]
lspr         = "spectraagent.physics.lspr:LSPRPlugin"
```

### Startup sequence (`spectraagent start`)
1. Discover hardware plugins, attempt connection, fall back to simulation
2. Load sensor physics plugin (default: LSPR)
3. Start `AgentBus` and deterministic agents
4. Start uvicorn (FastAPI) on configured port
5. Open browser at `http://localhost:8765`
6. Start watchdog thread (hardware reconnect on disconnect)

---

## 7. Session Output Structure

```
output/sessions/20260326_143201_EtOH_0.1ppm/
├── raw_spectra.parquet        # full 3648-px spectra, merged from flush partitions
├── pipeline_results.csv       # Δλ, SNR, quality, gas_conc per frame
├── session_meta.json          # gas, conc, sensor, timestamps, model version, hardware info
├── agent_events.jsonl         # all QualityAgent/DriftAgent/Claude events with timestamps
├── report.html                # self-contained HTML (no internet required to view)
└── report.pdf                 # PDF via WeasyPrint
```

---

## 8. Publication-Ready Report

The report is generated by `ReportBuilder` (deterministic template) optionally enhanced by `ReportWriter` (Claude API).

**Deterministic layer** (always, offline):
- Fills in all measured values: Δλ, SNR, LOD, LOQ, R², isotherm type, integration time, sampling rate, peak wavelength, session timestamps
- Uses Jinja2 HTML template styled for journal submission

**Claude API layer** (optional, when `ANTHROPIC_API_KEY` is set):
- Writes Methods and Results prose in natural journal style
- Marks all AI-generated text with a visual badge ("review before submitting")
- Never fabricates numbers — all values come from the deterministic layer

**Output formats:** HTML (self-contained, embeds charts as base64), PDF (WeasyPrint).

---

## 9. Existing Code Preservation

The following existing modules are kept exactly as-is and called through FastAPI routes:

- `src/` — all signal processing, Lorentzian fitting, GPR, calibration, reporting modules
- `src/features/lspr_features.py` — `LSPRReference`, `extract_lspr_features()` (already optimised)
- `src/inference/realtime_pipeline.py` — `RealTimePipeline` with `CalibrationStage` caching
- `src/inference/orchestrator.py` — `SensorOrchestrator`, `_SessionWriter` (partition-flush Parquet)
- `gas_analysis/acquisition/` — CCS200 hardware interface

The `dashboard/` directory is replaced by `webapp/`. The `streamlit` dependency is removed.

---

## 10. Error Handling and Reliability

- **Hardware disconnect:** Watchdog thread polls at 5 s. On disconnect, UI shows amber banner; reconnect attempted silently. Acquisition resumes automatically.
- **Claude API failure:** All Claude API calls have a 30 s timeout. On failure, the Agent Console shows a grey "Claude unavailable" event. Deterministic agents continue unaffected.
- **Session crash recovery:** `_SessionWriter` flushes every 0.5 s. On crash, all flushed parts are intact in `_raw_parts_dir/`. `spectraagent recover <session_dir>` merges parts.
- **Invalid spectrum:** QualityAgent hard-blocks on saturation (>60 000 counts). SNR < 3 is a warning, not a block.

---

## 11. Directory Layout (Target State)

```
spectraagent/              # new top-level package
├── __main__.py            # CLI entry point
├── webapp/
│   ├── server.py          # FastAPI app, all routes
│   ├── agent_bus.py       # AgentBus pub/sub
│   ├── agents/
│   │   ├── quality.py
│   │   ├── drift.py
│   │   ├── calibration.py
│   │   ├── planner.py
│   │   └── claude_agents.py
│   └── reports/
│       ├── builder.py     # deterministic ReportBuilder
│       └── templates/     # Jinja2 HTML templates
├── drivers/
│   ├── base.py            # AbstractHardwareDriver
│   ├── thorlabs.py        # existing CCS200 code, wrapped
│   └── simulation.py
├── physics/
│   ├── base.py            # AbstractSensorPhysicsPlugin
│   └── lspr.py            # existing LSPR logic, wrapped
src/                       # untouched (signal engine)
gas_analysis/              # untouched (hardware acquisition)
```

---

## 12. Out of Scope (v1.0)

- Multi-user authentication
- Cloud / remote hosting
- Raman-specific deconvolution
- Mobile / tablet UI
- Paid licensing or SaaS
- FTIR or mass spectrometry
