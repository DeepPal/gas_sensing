# SpectraAgent — Universal Agentic Spectroscopy Platform
**Design Document · 2026-03-26 (rev 2)**

---

## 1. Vision and Scope

SpectraAgent is a universal, AI-native spectrometer analysis platform. It is not an LSPR tool upgrade — it is a new product category: the first open-source platform that combines deterministic signal science with autonomous LLM-based reasoning for any spectroscopic sensor.

**Target user:** A research scientist (physics/chemistry PhD) running spectroscopic experiments who needs real-time AI guidance, calibration, and publication-ready reports without learning a commercial software stack.

**Not in scope (v1.0):** Multi-user auth, cloud hosting, Raman-specific deconvolution, FTIR, mobile UI, paid licensing.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER'S BROWSER  (localhost:8765)                                   │
│                                                                     │
│  React SPA — 5 tabs (pre-built, served as static files)            │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────────┐   │
│  │  Live    │Experiment│  Batch   │ Agentic  │ Agent Console  │   │
│  │  Sensor  │  Config  │ Analysis │ Pipeline │ /ws/agent-evts │   │
│  │  uPlot   │ +Results │ +Reports │ Steps 1-4│ + Ask Claude   │   │
│  └──────────┴──────────┴──────────┴──────────┴────────────────┘   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │ HTTP REST + WebSocket
┌──────────────────────▼──────────────────────────────────────────────┐
│  FastAPI  (spectraagent/webapp/server.py)  port 8765               │
│                                                                     │
│  /api/acquisition/*    /api/pipeline/*    /api/sessions/*          │
│  /api/calibration/*    /api/reports/*     /api/health              │
│  /ws/spectrum          /ws/trend          /ws/agent-events         │
│                                                                     │
│  ┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ SensorOrchest.   │  │ ReportBuilder  │  │ SessionRegistry  │   │
│  │ (existing src/)  │  │ (new)          │  │ (new)            │   │
│  └────────┬─────────┘  └────────────────┘  └──────────────────┘   │
│           │                                                         │
│  ┌────────▼─────────┐  ┌────────────────┐  ┌──────────────────┐   │
│  │ RealTimePipeline │  │   AgentBus     │  │ StructuredLogger │   │
│  │ + LSPRReference  │  │ (asyncio.Queue)│  │ → agent_events   │   │
│  │ (existing+fixed) │  │ (new)          │  │   .jsonl  (new)  │   │
│  └────────┬─────────┘  └────────────────┘  └──────────────────┘   │
└───────────┼─────────────────────────────────────────────────────────┘
            │ DLL / VISA / Serial
┌───────────▼────────────┐     ┌──────────────────────────────────────┐
│  Hardware Plugin        │     │  output/sessions/YYYYMMDD_HHMMSS/   │
│  (ThorLabs / OceanOpt  │     │  ├── raw_spectra.parquet             │
│   / Simulation)         │     │  ├── pipeline_results.csv           │
└────────────────────────┘     │  ├── session_meta.json               │
                               │  ├── agent_events.jsonl              │
                               └──────────────────────────────────────┘
```

### Key structural decisions

- `spectraagent/` is a **new top-level Python package** added alongside the existing `src/`, `gas_analysis/`, and `dashboard/` directories. It does not rename or remove them — it wraps them via adapter classes (see Section 13).
- React SPA is **pre-built** (by the developer using Vite) with output committed to `spectraagent/webapp/static/dist/`. End users need no Node.js. FastAPI mounts that directory under `/`.
- `dashboard/` is **removed at project end** (Phase 9 of migration). Streamlit is removed from dependencies.
- All existing `src/` signal math is called through FastAPI routes — no rewrites.
- New `/ws/agent-events` WebSocket streams both deterministic agent events and Claude API responses to the Agent Console tab.
- FastAPI adds `CORSMiddleware` with `allow_origins=["*"]` so LAN access works without configuration.

---

## 3. Frontend — React SPA (5 Tabs)

**State management:** Zustand — a single global store holds `{ session, hardwareStatus, latestSpectrum, latestResult, agentEvents[] }`. All tabs share this store. WebSocket messages call `store.setSpectrum(data)`, etc. Individual tabs own their local UI state (form values, selected files).

**React build:** `spectraagent/webapp/frontend/` contains the Vite project. `npm run build` outputs to `spectraagent/webapp/static/dist/`. This directory is committed to git and included in the wheel via `pyproject.toml` package-data. Developers modify the frontend and run `npm run build` before committing. End users never invoke npm.

### Tab 1: Live Sensor
- uPlot chart at 20 Hz via `/ws/spectrum`
- Trend chart: Δλ vs time at 200 ms via `/ws/trend`
- Hardware status badge (connected / simulation / error + error message)
- Reference capture button → POST `/api/acquisition/reference`

### Tab 2: Experiment
- Acquisition config: integration time (ms), gas label (string), target concentration (float, optional)
- Config is POST'd to `/api/acquisition/config` and held in server-side session state
- Run/stop → POST `/api/acquisition/start` / `/api/acquisition/stop`
- Live result cards: Δλ, SNR, quality gate status, estimated concentration
- Session name auto-assigned as `YYYYMMDD_HHMMSS_{gas}_{conc}ppm`; falls back to `YYYYMMDD_HHMMSS_unnamed` if gas/conc not set

### Tab 3: Batch Analysis
- Folder picker → GET `/api/batch/run?path=<dir>` processes all CSV/parquet files
- Lorentzian sub-pixel peak detection on each file (existing `src/` code)
- Results table with export to CSV
- Session comparison overlay

### Tab 4: Agentic Pipeline (4 steps)
- Step 1: Data ingestion (Joy_Data loader or session file picker)
- Step 2: Reference subtraction (`diff_signal = raw − ref_interp`)
- Step 3: Feature extraction + isotherm fitting (Langmuir/Freundlich/Hill, AIC-selected)
- Step 4: LOD/LOQ (regression method: LOD = 3s/m, LOQ = 10s/m where s = residual std of calibration fit, m = sensitivity at low-concentration linear region — per ICH Q2(R1) and standard *Sensors and Actuators B* practice), selectivity matrix, publication report export

### Tab 5: Agent Console
- Real-time event log from `/ws/agent-events`; server streams `AgentEvent` JSON objects (schema in Section 4.1)
- Color-coded by source: QualityAgent (green), DriftAgent (amber), CalibrationAgent (blue), Claude (purple)
- "Ask Claude" button: opens free-text input; client POSTs `{ "query": "...", "context": <current session snapshot> }` to `/api/agents/ask`; response streams back via Server-Sent Events and is inserted as a Claude event in the log
- Filter by agent source; search events; auto-scroll with pause-on-hover

---

## 4. Agentic AI Architecture (Hybrid)

### Layer 1 — Deterministic Agents (always running, <1 ms per frame, no API cost)

| Agent | Trigger | Action |
|---|---|---|
| **QualityAgent** | Every frame from `RealTimePipeline` | SNR gate, saturation check (>60 000 counts = hard block; SNR <3 = warning event, frame still processed). Emits `quality` event per frame. |
| **DriftAgent** | Rolling 60-frame window | CUSUM on peak wavelength shift. Emits `drift_warn` event when rate > 0.05 nm/min. Stores history. |
| **CalibrationAgent** | After each calibration point added | AIC model selection (Langmuir / Freundlich / Hill / Linear) over existing GPR fit. Emits `model_selected` event. Triggers Claude `ExperimentNarrator` when suggestion is ready. |
| **ExperimentPlannerAgent** | On user request (button) or CalibrationAgent trigger | Queries existing `GPRCalibration.predict()` at a grid of candidate concentrations; suggests the concentration with highest posterior standard deviation. Returns via `/api/calibration/suggest`. No BoTorch — the existing GPR already provides uncertainty. |

### Layer 2 — Claude API Agents (on-demand, async, never in the 20 Hz signal path)

| Agent | Trigger | Context sent to Claude | Output |
|---|---|---|---|
| **AnomalyExplainer** | `drift_warn` event from DriftAgent; cooldown 5 min per-agent | `{ delta_lambda, snr, drift_rate_nm_per_min, drift_history_60pt, sensor, gas, quality }` | Plain-English drift explanation + recommended action. |
| **ExperimentNarrator** | `model_selected` + new suggestion from CalibrationAgent; fires at most once per calibration point | `{ current_concentrations[], r2, aic_winner, suggested_conc, posterior_std }` | Why this concentration was chosen and what uncertainty it reduces. |
| **ReportWriter** | User clicks "Generate Report"; explicit user action only, never automatic | `{ session_meta, lod, loq, r2, isotherm_type, peak_wl_ref, delta_lambda_mean, snr_mean, n_frames, integration_time_ms }` | Methods + Results prose in journal style. Never called in batch loops. |
| **DiagnosticsAgent** | Hardware error event; cooldown 1 min per error code | `{ error_code, error_message, hardware_model, last_successful_frame_ago_s }` | Likely cause + step-by-step fix. |

**Claude model:** `claude-sonnet-4-6` for all agents (best cost/quality balance for structured scientific reasoning). All calls have a 30 s timeout. On failure or missing API key, agents emit a grey `claude_unavailable` event — deterministic agents are unaffected.

**Rate limiting:** Each Claude agent enforces its own per-instance cooldown via a `_last_called: float` timestamp. Cooldown durations are read from `spectraagent.toml` (user-configurable). Defaults: AnomalyExplainer 300 s, ExperimentNarrator once per calibration point, DiagnosticsAgent 60 s per error code. This prevents unbounded API spend during long sessions.

**Auto-explain is opt-in, default off.** `AnomalyExplainer` and `ExperimentNarrator` do not fire automatically unless the user enables "Auto-explain anomalies" in the Agent Console settings panel. By default, drift events appear in the log as deterministic events only — the user clicks "Ask Claude" to request an explanation. This prevents unexpected API charges mid-experiment.

**"Ask Claude" context:** The `/api/agents/ask` endpoint builds context as `{ query, session_meta, last_20_agent_events, latest_result }`. Raw spectrum arrays are never included.

### 4.1 AgentBus — Threading Model

The signal pipeline runs on a background OS thread (20 Hz). FastAPI runs on asyncio (single thread). The bridge:

```python
# In startup: store the running event loop
_loop = asyncio.get_event_loop()
_queue: asyncio.Queue[AgentEvent] = asyncio.Queue()

# From pipeline thread (sync) — zero-blocking push:
def emit(event: AgentEvent) -> None:
    _loop.call_soon_threadsafe(_queue.put_nowait, event)

# In FastAPI WebSocket handler (async) — one handler per connected client:
async def agent_events_ws(ws: WebSocket):
    await ws.accept()
    while True:
        event = await _queue.get()  # suspends without blocking the loop
        await ws.send_json(event.to_dict())
```

`AgentBus` also writes every event to `agent_events.jsonl` (appended synchronously from the async handler, since file I/O is fast relative to frame rate).

**AgentEvent JSON schema:**
```json
{
  "ts": "2026-03-26T14:32:11.042Z",
  "source": "QualityAgent",
  "level": "ok",
  "type": "quality",
  "data": {
    "frame": 1847,
    "snr": 62.4,
    "saturation_pct": 0.0,
    "quality": "ok"
  },
  "text": "Frame 1847 — SNR=62.4, saturation=0.0%, quality=OK"
}
```

`level` is one of: `ok`, `warn`, `error`, `info`, `claude`. This is the sole field the frontend uses for color coding.

---

## 5. Plugin Architecture

### Hardware Driver Interface (`spectraagent/drivers/base.py`)

```python
class AbstractHardwareDriver:
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_wavelengths(self) -> np.ndarray: ...       # shape (N,) — call once at startup
    def read_spectrum(self) -> np.ndarray: ...          # shape (N,) intensities, blocking
    def get_integration_time_ms(self) -> float: ...
    def set_integration_time_ms(self, ms: float) -> None: ...
    @property
    def name(self) -> str: ...                         # "ThorLabs CCS200", "Simulation", etc.
    @property
    def is_connected(self) -> bool: ...
```

### Sensor Physics Plugin Interface (`spectraagent/physics/base.py`)

```python
class AbstractSensorPhysicsPlugin:
    def detect_peak(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> float | None: ...                             # returns peak wavelength in nm, or None

    def extract_features(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        reference: np.ndarray | None = None,
        cached_ref: object | None = None,             # plugin-specific cache (e.g. LSPRReference)
    ) -> dict[str, float]: ...                        # e.g. {"delta_lambda": -0.71, "snr": 62.4}

    def compute_reference_cache(
        self, wavelengths: np.ndarray, reference: np.ndarray
    ) -> object: ...                                  # returns plugin-specific cache object

    def calibration_priors(self) -> dict: ...         # {"models": ["Langmuir", "Linear"], "bounds": {...}}

    @property
    def name(self) -> str: ...                        # "LSPR", "SPR", "UV-Vis"
```

### Plugin Registration

Registered via `pyproject.toml` entry points:

```toml
[project.entry-points."spectraagent.hardware"]
thorlabs_ccs = "spectraagent.drivers.thorlabs:ThorlabsCCSDriver"
simulation   = "spectraagent.drivers.simulation:SimulationDriver"

[project.entry-points."spectraagent.sensor_physics"]
lspr         = "spectraagent.physics.lspr:LSPRPlugin"
```

Discovered at startup via `importlib.metadata.entry_points(group="spectraagent.hardware")`. Third-party plugins install via `pip install spectraagent-ocean-optics` and are auto-discovered.

| Hardware Plugin | Status |
|---|---|
| ThorLabs CCS200 | existing ✓ (wraps `gas_analysis/acquisition/`) |
| Simulation | existing ✓ (wraps `gas_analysis/acquisition/simulation.py`) |
| Ocean Optics USB2000+/Flame | v1.1 |
| Avantes AvaSpec | v1.2 |

| Sensor Physics Plugin | Status |
|---|---|
| LSPR (Au nanoparticles) | existing ✓ (wraps `src/features/lspr_features.py`) |
| SPR (prism/fiber) | v1.1 |
| UV-Vis Absorption | v1.1 |
| NIR / NIRS | v1.2 |
| Raman / fluorescence | community |

---

## 6. Packaging and CLI

### Entry point (Typer)
```
spectraagent/__main__.py  →  cli() Typer app
```

### Commands
```bash
spectraagent start                              # start server, open browser
spectraagent start --simulate                  # force simulation mode
spectraagent start --no-browser                # headless / server deployment
spectraagent start --host 0.0.0.0 --port 8765 # LAN-accessible
spectraagent report <session_dir>              # generate HTML report (always works)
spectraagent report <session_dir> --format pdf # PDF via Playwright (optional extra)
spectraagent recover <session_dir>             # merge flush partitions after crash
spectraagent plugins list                      # show discovered plugins + status
```

### pyproject.toml additions
```toml
[project]
name = "spectraagent"               # NEW package name; old gas-analysis becomes internal

[project.scripts]
spectraagent = "spectraagent.__main__:cli"

[project.dependencies]
# ... existing deps ...
typer = ">=0.9"
anthropic = ">=0.25"
# streamlit REMOVED

[project.optional-dependencies]
pdf = ["playwright>=1.40"]

[tool.setuptools.package-data]
"spectraagent" = ["webapp/static/dist/**/*"]   # pre-built React app included in wheel

[project.entry-points."spectraagent.hardware"]
thorlabs_ccs = "spectraagent.drivers.thorlabs:ThorlabsCCSDriver"
simulation   = "spectraagent.drivers.simulation:SimulationDriver"

[project.entry-points."spectraagent.sensor_physics"]
lspr         = "spectraagent.physics.lspr:LSPRPlugin"
```

### Startup sequence (`spectraagent start`)
1. Discover hardware plugins via entry points; attempt connect; fall back to simulation; log result to console and browser status badge
2. Load sensor physics plugin (default: `lspr`; overridable via `--physics` flag)
3. Check `ANTHROPIC_API_KEY` env var; if missing, print warning "Claude agents disabled — set ANTHROPIC_API_KEY to enable"; continue without error
4. Start `AgentBus` (asyncio event loop + `asyncio.Queue`) and register deterministic agents
5. Start uvicorn (FastAPI) on configured host/port; add `CORSMiddleware(allow_origins=["*"])`
6. Start watchdog thread (polls hardware at 5 s interval; reconnects silently)
7. Unless `--no-browser`: open `http://localhost:{port}` in system browser (always localhost, never the bind host)

### React packaging (no Node.js required by end users)
- Source: `spectraagent/webapp/frontend/` (Vite + React + TypeScript project)
- Build output: `spectraagent/webapp/static/dist/` — committed to git
- Included in Python wheel via `package-data` (above)
- FastAPI mounts: `app.mount("/", StaticFiles(directory=dist_path, html=True))`
- End users: `pip install spectraagent` → React UI included. No npm, no Node.js.
- Developers: `cd spectraagent/webapp/frontend && npm run build` before committing frontend changes

---

## 7. Session Output Structure

`save_raw=True` by default — raw spectra are always saved. (The current `_SessionWriter` default of `save_raw=False` will be changed.)

```
output/sessions/20260326_143201_EtOH_0.1ppm/
├── raw_spectra.parquet        # 3648-px spectra per frame, merged from flush partitions
├── pipeline_results.csv       # Δλ, SNR, quality, gas_conc per frame
├── session_meta.json          # gas, conc, sensor, timestamps, model version, hardware info
├── agent_events.jsonl         # written by AgentBus; one JSON object per line
├── report.html                # working report — regenerated freely, may vary with Claude
├── report_final.html          # locked on "Finalize" — never overwritten; this is the submission copy
└── report.pdf                 # generated only if `pip install spectraagent[pdf]`
```

`agent_events.jsonl` is written by `AgentBus` as events are emitted — same objects sent to `/ws/agent-events`.

---

## 8. Publication-Ready Report

### Generation flow

```
Session ends → ReportBuilder.build(session_dir)
    ↓ always (offline)
Jinja2 template + measured values → report.html
    ↓ optional (if ANTHROPIC_API_KEY set and user requests)
ReportWriter Claude agent → rewrites Methods + Results sections in journal prose
    ↓ optional (if spectraagent[pdf] installed)
Playwright headless Chromium → report.pdf
```

### Deterministic layer (always, offline, no API key required)
- Template fills all measured values: Δλ, SNR, LOD, LOQ, R², isotherm type, integration time, sampling rate, peak wavelength, session timestamps, model version
- Charts embedded as base64 PNG (matplotlib, generated offline)
- Report is valid HTML with no external dependencies — opens in any browser

### Claude API layer (optional)
- `ReportWriter` rewrites the Methods and Results sections into journal-style prose
- All AI-generated text is marked with `<span class="ai-generated">` and a visible badge "AI-generated — review before submitting"
- Claude never invents numbers — all values come from the deterministic layer's context dict
- Called only when user explicitly clicks "Generate Report with AI" — never automatic
- **Finalization:** Once the user clicks "Finalize Report", the current `report.html` is copied to `report_final.html` and locked (never overwritten). Subsequent "Generate Report" calls write to `report.html` only. This ensures the submission version is reproducible — regenerating Claude prose does not alter the finalized document.

### PDF export
- `spectraagent report <dir> --format pdf` launches headless Chromium via Playwright on the generated `report.html`
- If Playwright not installed: command prints "Install spectraagent[pdf] for PDF export" and opens the HTML file in the system browser instead
- This avoids WeasyPrint's broken Windows native dependency chain (Pango/Cairo not pip-installable)

---

## 9. Existing Code Preservation

The adapter pattern — new `spectraagent/` package wraps existing code, never copies it:

| New module | Wraps |
|---|---|
| `spectraagent/drivers/thorlabs.py` | `gas_analysis/acquisition/ccs200_realtime.py` |
| `spectraagent/drivers/simulation.py` | `gas_analysis/acquisition/simulation.py` |
| `spectraagent/physics/lspr.py` | `src/features/lspr_features.py` (LSPRPlugin implements AbstractSensorPhysicsPlugin) |
| `spectraagent/webapp/server.py` | Imports and calls `SensorOrchestrator`, `RealTimePipeline`, `src/reporting/` directly |

The existing `live_server.py` WebSocket broadcaster logic (`_Broadcaster`, `_spectrum_loop`, `_trend_loop`) is moved into `webapp/server.py` and adapted to the unified FastAPI app. The `LiveDataStore` singleton is retained as-is.

Nothing in `src/` is modified for this migration. `gas_analysis/acquisition/` is also unchanged.

---

## 10. Error Handling and Reliability

- **Hardware disconnect:** Watchdog polls at 5 s. On disconnect, Zustand `hardwareStatus` → `"error"`, UI shows amber banner. Reconnect attempted silently; banner clears on success.
- **Claude API failure / timeout (30 s):** Agent emits `claude_unavailable` event (grey in console). Deterministic agents unaffected. No retry — user can click "Ask Claude" manually.
- **Missing API key:** Startup prints warning; Claude agents emit `claude_unavailable` immediately on any trigger. Not an error.
- **Session crash recovery:** `_SessionWriter` flushes every 0.5 s to `_raw_parts_dir/part_NNNNNN.parquet`. On crash, parts are intact. `spectraagent recover <session_dir>` calls `_merge_raw_parts()` and exits.
- **Saturation:** QualityAgent hard-blocks the frame (not written to CSV, not sent to Claude). SNR < 3 → warning event emitted, frame still processed.
- **`spectraagent start --host 0.0.0.0`:** Browser always opens `http://localhost:{port}`, never the bind host (which would be `http://0.0.0.0:8765` — invalid).
- **CORS:** `CORSMiddleware(allow_origins=["*"])` applied at startup. Required for LAN use (client browser origin differs from server host).

---

## 11. Directory Layout (Target State)

```
spectraagent/                  # new installable package
├── __main__.py                # Typer CLI: start, report, recover, plugins
├── webapp/
│   ├── server.py              # FastAPI app, all routes, CORS, static mount
│   ├── agent_bus.py           # AgentBus: asyncio.Queue bridge, event schema
│   ├── agents/
│   │   ├── quality.py         # QualityAgent
│   │   ├── drift.py           # DriftAgent (CUSUM, 60-frame window)
│   │   ├── calibration.py     # CalibrationAgent (AIC model selection)
│   │   ├── planner.py         # ExperimentPlannerAgent (GPR uncertainty)
│   │   └── claude_agents.py   # AnomalyExplainer, ExperimentNarrator, ReportWriter, DiagnosticsAgent
│   ├── reports/
│   │   ├── builder.py         # ReportBuilder (deterministic, Jinja2)
│   │   └── templates/         # report.html.j2
│   └── frontend/              # Vite + React source (not installed, dev-only)
│       ├── src/
│       └── package.json
│   └── static/
│       └── dist/              # pre-built React output — committed to git, included in wheel
├── drivers/
│   ├── base.py                # AbstractHardwareDriver
│   ├── thorlabs.py            # wraps gas_analysis/acquisition/ccs200_realtime.py
│   └── simulation.py          # wraps gas_analysis/acquisition/simulation.py
└── physics/
    ├── base.py                # AbstractSensorPhysicsPlugin
    └── lspr.py                # wraps src/features/lspr_features.py
src/                           # untouched (signal engine)
gas_analysis/                  # untouched (hardware acquisition)
dashboard/                     # deleted at Phase 9
```

---

## 12. Configuration File (`spectraagent.toml`)

User-editable config at the project root. Created with defaults on first `spectraagent start` if absent.

```toml
[hardware]
default_driver = "thorlabs_ccs"       # or "simulation"
integration_time_ms = 50.0

[physics]
default_plugin = "lspr"
search_min_nm = 500.0
search_max_nm = 900.0

[agents]
auto_explain = false                  # opt-in: AnomalyExplainer/ExperimentNarrator auto-fire
anomaly_explainer_cooldown_s = 300
diagnostics_cooldown_s = 60

[claude]
model = "claude-sonnet-4-6"
timeout_s = 30

[server]
host = "127.0.0.1"
port = 8765
open_browser = true
```

All CLI flags override the config file. Config file overrides built-in defaults.

---

## 13. Out of Scope (v1.0)

- Multi-user authentication
- Cloud / remote hosting
- Raman-specific deconvolution
- Mobile / tablet UI
- Paid licensing or SaaS
- FTIR or mass spectrometry
- BoTorch / multi-objective Bayesian optimization (simple GPR uncertainty is sufficient for 1D calibration)
- WeasyPrint PDF (Playwright is the v1.0 approach; WeasyPrint is explicitly excluded due to Windows native dep issues)

---

## 14. Migration Path (Current → Target)

Phases are sequential. Each phase leaves the codebase in a working state.

### Phase 1 — Skeleton
- Create `spectraagent/__init__.py`, `spectraagent/__main__.py` (stub CLI)
- Update `pyproject.toml`: new package name `spectraagent`, add Typer + anthropic deps, remove streamlit, add entry points, add package-data for `webapp/static/dist/`
- Verify: `pip install -e .` works; `spectraagent --help` runs

### Phase 2 — Hardware Adapters
- Create `spectraagent/drivers/base.py` (AbstractHardwareDriver)
- Create `spectraagent/drivers/thorlabs.py` wrapping `gas_analysis/acquisition/ccs200_realtime.py`
- Create `spectraagent/drivers/simulation.py`
- Test: instantiate both drivers in a Python REPL; `read_spectrum()` returns shape `(3648,)`

### Phase 3 — Physics Plugin
- Create `spectraagent/physics/base.py` (AbstractSensorPhysicsPlugin)
- Create `spectraagent/physics/lspr.py` wrapping `src/features/lspr_features.py` — `LSPRPlugin.extract_features()` calls `extract_lspr_features()` with the cached `LSPRReference`
- Test: `LSPRPlugin().detect_peak(wl, intensities)` returns a float

### Phase 4 — FastAPI Backend
- Create `spectraagent/webapp/server.py`
- Move WebSocket broadcaster from `dashboard/live_server.py` into server.py; adapt to unified app on port 8765
- Add `CORSMiddleware`; mount `webapp/static/dist/` under `/` (can be empty dir for now)
- Add routes: `/api/acquisition/*`, `/api/health`, `/ws/spectrum`, `/ws/trend`
- Wire to existing `SensorOrchestrator` and `RealTimePipeline`
- Update `spectraagent start` CLI to start uvicorn here
- Test: `spectraagent start --simulate --no-browser` → server runs; `/api/health` returns 200

### Phase 5 — React Frontend (5 tabs)
- Scaffold Vite project in `spectraagent/webapp/frontend/`
- Build 5-tab layout with Zustand store; wire WebSocket for Live Sensor tab
- Implement remaining tabs using REST API calls
- `npm run build` → `webapp/static/dist/`; verify FastAPI serves it
- Test: `spectraagent start --simulate` → browser opens, Live Sensor tab shows simulated spectrum at 20 Hz

### Phase 6 — AgentBus + Deterministic Agents
- Create `spectraagent/webapp/agent_bus.py`; implement threading bridge (Section 4.1)
- Implement QualityAgent, DriftAgent, CalibrationAgent, ExperimentPlannerAgent
- Wire QualityAgent + DriftAgent to `RealTimePipeline.process_spectrum()` callback
- Add `/ws/agent-events` WebSocket; add Agent Console tab to React frontend
- Test: `spectraagent start --simulate` → drift events appear in Agent Console within 60 frames

### Phase 7 — Claude API Agents
- Implement `claude_agents.py`: AnomalyExplainer, ExperimentNarrator, ReportWriter, DiagnosticsAgent
- Wire to AgentBus triggers; add cooldown logic
- Add `/api/agents/ask` endpoint (streaming SSE response)
- Test: set `ANTHROPIC_API_KEY`; induce drift in simulation; verify AnomalyExplainer fires once, explains in Agent Console

### Phase 8 — Reports
- Create `spectraagent/webapp/reports/builder.py` and Jinja2 template
- Wire to session end: `ReportBuilder.build(session_dir)` called in `_SessionWriter.stop()`
- Add `spectraagent report` CLI command
- Add optional Playwright PDF path
- Test: `spectraagent report output/sessions/test_session/` → `report.html` opens in browser with correct values

### Phase 9 — Cleanup
- Delete `dashboard/` directory
- Remove `streamlit` from dependencies
- Update `run_dashboard.bat` → `spectraagent start`
- Run full test suite (`pytest`); verify 669+ tests pass
- Commit: "feat: SpectraAgent v1.0 — full migration complete"
