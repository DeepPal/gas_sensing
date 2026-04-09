# SpectraAgent

**Universal Agentic Spectroscopy Platform**

SpectraAgent is a hardware-agnostic, AI-augmented spectroscopy platform for real-time
acquisition, calibration, and scientific characterisation. Any spectrometer (Thorlabs,
Ocean Insight, Avantes, simulated) and any sensing modality (LSPR, fluorescence,
absorbance, Raman) can be plugged in via Python entry-points.

---

## Quick Start

=== "Live acquisition (SpectraAgent)"

    ```bash
    python -m spectraagent start
    # Opens http://localhost:8765 automatically
    # Force simulation:
    python -m spectraagent start --simulate
    ```

=== "Research dashboard (Streamlit)"

    ```bash
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py
    ```

=== "Batch analysis"

    ```bash
    python run.py --mode batch --data data/JOY_Data/Ethanol
    ```

=== "List plugins"

    ```bash
    python -m spectraagent plugins list
    ```

---

## Platform Overview

```
┌───────────────────────────────────────────────────────────────┐
│             Any spectrometer (USB/VISA/serial)                │
│             + any sensor physics (LSPR/fluor/abs/Raman)       │
└──────────────────────────┬────────────────────────────────────┘
                           │  entry-point plugin
                           ▼
┌───────────────────────────────────────────────────────────────┐
│            SpectraAgent Acquisition Layer                     │
│  • Daemon thread at hardware frame rate (~2–20 Hz)            │
│  • QualityAgent (SNR, saturation gate)                        │
│  • DriftAgent (peak shift trend detection)                    │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│            RealTimePipeline  (src/inference)                  │
│  Stage 1: Preprocessing  (ALS baseline, Savitzky-Golay)       │
│  Stage 2: Feature extraction  (physics-plugin specific)       │
│  Stage 3: GPR calibration  → concentration + conformal CI     │
│  Stage 4: CNN classifier   → gas/analyte type + confidence    │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│            AgentBus + Claude AI Agents                        │
│  • AnomalyExplainer  — explains drift events via Claude API   │
│  • ExperimentNarrator — narrates calibration milestones       │
│  • DiagnosticsAgent  — reacts to hardware errors             │
│  • ExperimentPlannerAgent — suggests next concentration (BED) │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌───────────────────────────────────────────────────────────────┐
│            React Frontend + FastAPI + WebSocket               │
│  • Live spectrum chart    • Agent event feed                  │
│  • Session controls       • Calibration panel                 │
│  • Ask Claude panel       • Hardware badge (Live / Sim)       │
└───────────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

| Capability | Details |
|---|---|
| **Hardware** | Thorlabs CCS200 (built-in), any spectrometer via plugin |
| **Physics** | LSPR (built-in), any modality via plugin |
| **Calibration** | GPR, PLS, physics-informed GPR (Langmuir), conformal CI |
| **LOD/LOQ** | IUPAC-correct triad — LOB/LOD/LOQ with bootstrap CI |
| **Active learning** | Bayesian experiment designer — logspace max-variance |
| **Uncertainty** | Split conformal prediction — provable coverage guarantee |
| **AI agents** | Claude-powered: anomaly explanation, diagnostics, narration |
| **Session storage** | Per-frame CSV + metadata JSON + agent event log |
| **ONNX export** | CNN → ONNX with numerical validation |
| **Test coverage** | 1 187 tests passing (0 failures) |

---

## Two Runtime Paths

The platform offers two complementary entry points:

### SpectraAgent (primary — live acquisition)

```bash
python -m spectraagent start [--simulate] [--no-browser] [--port 8765]
```

- **Audience**: lab operator, live experiment
- **Interface**: React web app + FastAPI WebSocket server
- **What it does**: Real-time spectrum streaming, per-frame ML inference,
  AI agent event feed, session recording, calibration wizard

### Research Dashboard (Streamlit — scientific analysis)

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

- **Audience**: scientist, data analyst
- **Interface**: Streamlit multi-tab dashboard
- **What it does**: Batch calibration, publication figures,
  multi-analyte selectivity, MCR-ALS spectral deconvolution,
  HTML calibration report export

Both paths share the `src/` science library. They are independent and can
run simultaneously.

---

## Navigation

## Canonical status tracking

To avoid roadmap/readiness drift between documents and automation, update these
files together whenever project status changes:

- `REMAINING_WORK.md`
- `PRODUCTION_READINESS.md`
- `CHANGELOG.md`
- `.github/workflows/security.yml`

- [System Architecture](SYSTEM_ARCHITECTURE.md) — runtime design, data flow, plugin system
- [Engineering Standards](ENGINEERING_STANDARDS.md) — code quality, testing, CI gates
- [Validation (ICH Q2)](ICH_Q2_VALIDATION.md) — analytical method validation
- [Validation Strategy](VALIDATION_STRATEGY.md) — broader validation approach
- [API Reference](api/index.md) — auto-generated from typed `src/` package
- [ADRs](adr/README.md) — architectural decision records
- [Hardware Setup](hardware/CCS200_SETUP.md) — CCS200 DLL, VISA, troubleshooting
- [Calibration Workflow](guides/CALIBRATION_WORKFLOW.md) — step-by-step calibration guide

---

## Session Output

Every acquisition session writes to `output/sessions/YYYYMMDD_HHMMSS/`:

| File | Contents |
|---|---|
| `pipeline_results.csv` | Per-frame: timestamp, peak_wl, shift, conc, CI, SNR, gas_type |
| `session_meta.json` | LOD/LOQ/LOB, T90/T10, drift rate, linearity, bootstrap CI |
| `agent_events.jsonl` | Every AgentBus event (quality, drift, claude, calibration) |
