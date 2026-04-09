# AGENTS.md — Agent Architecture & Automation

## Agent Philosophy

Agents in this system are **simple, deterministic rule-based processes** — not LLM agents.
They watch for specific conditions, then trigger defined actions.

Rule: Start simple (if/else), add intelligence only when the rule fails in practice.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Layer Overview                         │
│                                                                 │
│  [Drift Detection Agent]  ──triggers──►  [Training Agent]      │
│                                                                 │
│  [Data Quality Agent]     ──flags──────► [Human Review Queue]  │
│                                                                 │
│  [Agentic Pipeline]       ──orchestrates─► [5-step workflow]   │
│  (dashboard/agentic_pipeline_tab.py)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent 1: Drift Detection Agent

**File:** `src/agents/drift.py`

**What it does:** Monitors the reference spectrum for baseline drift over time.
If the peak wavelength shifts by more than `drift_threshold_nm` relative to the
stored reference, it flags a calibration event.

**Trigger conditions:**
```python
# Check every N frames (default: every 100 frames = ~50 seconds at 2 Hz)
if abs(current_peak_wl - reference_peak_wl) > drift_threshold_nm:
    trigger: DriftEvent(magnitude_nm, timestamp, recommendation)
```

**Actions:**
1. Log drift event to MLflow with timestamp + magnitude
2. Emit `DriftEvent` to LiveDataStore so dashboard shows a warning
3. If `auto_recalibrate=True` in config: trigger reference spectrum update
4. If `auto_retrain=True` in config: add to training queue

**Key parameters (from config.yaml):**
```yaml
agents:
  drift:
    enabled: true
    check_every_n_frames: 100
    drift_threshold_nm: 0.5     # Alert if reference shifts > 0.5 nm
    auto_recalibrate: false     # Require human confirmation by default
    auto_retrain: false
```

---

## Agent 2: Data Quality Agent

**File:** `src/agents/quality.py`

**What it does:** Validates incoming spectra in real-time. Rejects bad frames
before they contaminate training data or calibration curves.

**Checks performed (in order):**

```python
def validate_spectrum(reading: SpectrumReading) -> QualityReport:
    checks = [
        check_saturation(intensities, threshold=60000),     # Hard block
        check_snr(intensities, wavelengths, min_snr=4.0),   # Warning if < 4
        check_finite(intensities),                           # Hard block
        check_baseline_stability(intensities),               # Warning
        check_peak_in_range(peak_wl, expected=(480, 600)),  # Warning
    ]
    return QualityReport(passed=all_hard_checks_passed, warnings=warnings)
```

**Severity levels:**
- **Hard block** (fails `success=False`): saturation, NaN/Inf intensities
- **Warning** (passes but flags): low SNR, peak outside expected range
- **Info**: RSD above threshold (normal during gas transitions)

---

## Agent 3: Training Agent

**File:** `src/agents/training.py` (future)

**What it does:** Watches the `data/raw/` directory for new experimental files.
When enough new data accumulates, triggers an incremental model retrain.

**Trigger conditions:**
```python
# Check on schedule (every 24 hours by default)
if new_samples_since_last_train >= min_new_samples:
    trigger: TrainingRun(gas_type, new_sample_count)
```

**Actions:**
1. Merge new data with existing training set
2. Launch `src/training/train_cnn.py` + `train_gpr.py`
3. Compare new model metrics vs production model
4. If new model is better (R² improvement > 0.01): update model registry
5. Log entire run to MLflow

---

## Agent 4: Agentic Pipeline (Dashboard)

**File:** `dashboard/agentic_pipeline_tab.py`

This is the **interactive 5-step automation workflow** in the Streamlit dashboard.
It guides the researcher through a full experiment session.

```
Step 1: Reference Acquisition
        → Record baseline spectrum (clean air / nitrogen purge)
        → Store as session reference

Step 2: Data Acquisition
        → Start CCS200 acquisition
        → Load existing Joy_Data OR record new session
        → Apply temporal gating (plateau detection)
        → Compute diff_signal = raw − reference

Step 3: Feature Extraction + Calibration Fit
        → Extract LSPR features (Δλ, ΔI_peak, ΔI_area, ΔI_std)
        → Fit calibration curve with selected model
        → Plot calibration curve (Y-axis = Δλ in nm)
        → Log to MLflow

Step 4: Model Training (optional)
        → Train CNN classifier on current session data
        → Train GPR calibration model
        → Cross-validate + report metrics

Step 5: Export + Report
        → Generate HTML report
        → Save plots to reports/
        → Export session CSV
```

**Session state variables (Streamlit `st.session_state`):**

```python
ap_ref_spectrum       # Reference intensities array
ap_ref_wl             # Reference wavelengths array
ap_ref_peak_wl        # Reference peak wavelength (for Δλ calculation)
ap_preprocessed       # List of preprocessed spectrum dicts
ap_features           # Extracted feature matrix
ap_calibration_model  # Fitted calibration object
ap_mlflow_run_id      # Active MLflow run ID
```

---

## Agent Interaction Pattern

```
                    CCS200 Acquisition
                          │
                          ▼
              ┌───────────────────────┐
              │  Data Quality Agent   │ ◄── validates each frame
              │  (runs inline)        │
              └───────────┬───────────┘
                          │ valid frames only
                          ▼
              ┌───────────────────────┐
              │  LiveDataStore        │
              │  (thread-safe deque)  │
              └───────────┬───────────┘
                          │
              ┌───────────┴────────────────────┐
              │                                │
              ▼                                ▼
   ┌──────────────────┐           ┌───────────────────────┐
   │  Streamlit       │           │  Drift Detection      │
   │  Dashboard       │           │  Agent (periodic)     │
   │  (reads state)   │           │                       │
   └──────────────────┘           └──────────┬────────────┘
                                             │ if drift detected
                                             ▼
                                  ┌──────────────────────┐
                                  │  Training Agent      │
                                  │  (async, background) │
                                  └──────────────────────┘
```

---

## Adding a New Agent

1. Create `src/agents/my_agent.py` with a class that follows this pattern:

```python
class MyAgent:
    def __init__(self, config: dict):
        self.config = config
        self.enabled = config.get("enabled", False)

    def check(self, state: LiveDataStore) -> AgentEvent | None:
        """Return an AgentEvent if action needed, else None."""
        if not self.enabled:
            return None
        # ... check logic ...
        return AgentEvent(agent="my_agent", action="...", data={...})

    def act(self, event: AgentEvent) -> None:
        """Execute the triggered action."""
        ...
```

2. Add config section in `configs/config.yaml` under `agents:`
3. Register in `src/inference/orchestrator.py` agent list
4. Add test in `tests/test_agents.py`

---

## Future Agent Ideas (Roadmap)

| Agent | Trigger | Action |
|-------|---------|--------|
| Humidity Compensation | `humidity > threshold` | Apply humidity correction factor to Δλ |
| Temperature Compensation | `|ΔT| > 2°C` | Apply temp correction to calibration |
| Anomaly Detection | Spectrum shape deviates from training distribution | Flag + request human label |
| Auto-Export | Session ends | Auto-generate PDF report + upload to shared drive |
| Cross-Sensor Calibration | Second sensor connected | Transfer calibration to new sensor |
