# Agent 06 — Master Dashboard Orchestrator
**Stage A+B · Step 6 of 6 · Entry point: `dashboard/app.py`**

---

## Purpose
Integrate Agents 01–05 into a single, seamless sequential Streamlit dashboard. Each agent maps to a clearly labelled step with a visual progress tracker. Session state persists across all steps so users never lose data mid-experiment.

---

## Source Files
| File | Role |
|------|------|
| `dashboard/app.py` | Tab router — 4 top-level tabs |
| `dashboard/agentic_pipeline_tab.py` | `render()` — full 4-step automation pipeline |
| `dashboard/experiment_tab.py` | `render()` — guided single-experiment flow |
| `dashboard/sensor_dashboard.py` | `render()` — live CCS200 sensor monitoring |

---

## Top-Level Tab Layout (`dashboard/app.py`)

```
┌──────────────────────┬──────────────────────┬──────────────────────┬──────────────┐
│  🤖 Automation       │  🧪 Experiment        │  📊 Batch Analysis   │  📡 Live     │
│  Pipeline            │  (Guided)             │                      │  Sensor      │
│  (Agents 01–05)      │  experiment_tab.py    │  app.py inline       │  sensor_     │
│  agentic_pipeline_   │                       │                      │  dashboard   │
│  tab.py              │                       │                      │              │
└──────────────────────┴──────────────────────┴──────────────────────┴──────────────┘
```

---

## Automation Pipeline Structure (`agentic_pipeline_tab.py`)

### Progress Banner
```python
badges = ["1  Acquisition & Logging", "2  Preprocessing",
          "3  Training & Insights",   "4  Deployment & Testing"]
# Each badge: green if completed, blue if active, grey if pending
```

### Step 1 — Acquisition & Logging  (Agents 01 + 02)

**Section A — 📡 Live Data View**
- `👁️ Preview Spectrum` button → `_acquire_frames(ms, 5, 0, chart_ph)` — not recorded
- Renders live frame in `chart_preview_ph = prev_left.empty()`
- Displays SNR and RMS noise quality cards in real time

**Section B — 📋 Data Logging & Acquisition**
- `st.form("ap_meta_form")` — gas type, concentration, trial, integration time, n_frames, comments
- `▶️ Record Snapshot` → acquires, QC-gates, saves CSV + JSON sidecar to `data/automation_dataset/`
- Session recording log table (all snapshots this session)
- `➡️ Proceed to Preprocessing` (disabled until ≥1 recording in buffer)

### Step 2 — Preprocessing  (Agent 03)
- Source selector: session buffer ↔ disk CSVs from `data/automation_dataset/`
- Dropdowns: Denoising · Baseline Removal · Normalization
- `⚙️ Run Preprocessing` → fills `ss["ap_preprocessed"]`
- QC improvement table (Raw SNR vs Proc SNR per spectrum)
- Raw vs preprocessed Plotly overlay chart

### Step 3 — Insights, Feature Extraction & Model Training  (Agent 04)
- Feature table: Peak (nm) · Peak Intensity · Integrated Area · Spectral Std
- **Scientific Metrics**: Sensitivity · R² · LOD · LOQ (requires ≥3 concentration points)
- **Spectral Overlay** — rainbow plot, all concentrations colour-coded by HSL gradient
- **Calibration Curve** — peak intensity vs. concentration with linear fit + R² (requires ≥3 points)
- **Sensitivity Heatmap** — `go.Heatmap(z=Z, x=wl_common, y=concs)` — wavelength × concentration
- **3D Response Surface** — `go.Surface(z=Z, x=wl_common, y=concs)` — interactive
- **Confusion Matrix** — NearestCentroid 3-fold cross-val (requires ≥2 gas classes, ≥4 samples)
- **Model Training** — GPR concentration regression (sklearn always available) or CNN gas classifier (requires torch)
- `🚀 Train Model` → saves to `models/`; sets `ss["ap_model_trained"] = True`

### Step 4 — Deployment & Real-Time Inference  (Agent 05)
- **Model loading**: from session OR `📂 Load Model from Disk` (auto-scans `models/*.pkl`)
- Status cards: Model Status · LOD · Training R²
- `🔮 Run Single Prediction` → acquire → preprocess → features → GPR → concentration gauge
- Prediction history with ±1σ confidence band time series
- MAE vs known concentration (when ≥3 predictions logged)
- **Test CSV Upload** — batch accuracy evaluation against labeled test files (filename must contain `{conc}ppm`)
- `📄 Generate Session Report` → saves `reports/session_{ts}.md` + `⬇️ Download` button

---

## Shared Session State Schema

| Key | Type | Written By | Read By |
|-----|------|------------|---------|
| `ap_step` | `int` 1–4 | Navigation buttons | `render()` router |
| `ap_preview_wl` | `ndarray` | Step 1 preview | Step 1 display |
| `ap_preview_frame` | `ndarray` | Step 1 preview | Step 1 display |
| `ap_meta` | `dict` | Step 1 form | Steps 2–4 |
| `ap_buffer` | `list[dict]` | Step 1 record | Step 2 |
| `ap_wl` | `ndarray` | Step 1 record | Steps 2–4 |
| `ap_preprocessed` | `list[dict]` | Step 2 | Step 3 |
| `ap_model_trained` | `bool` | Steps 3/4 load | Step 4 gate |
| `ap_gpr_sklearn` | sklearn estimator | Steps 3/4 | Step 4 inference |
| `ap_X_train` | `ndarray` | Step 3 | Step 4 context |
| `ap_y_concs` | `ndarray` | Step 3 | Step 4 fallback |
| `ap_class_names` | `list[str]` | Step 3 | Step 4 display |
| `ap_lod` | `float` | Step 3 | Step 4 card |
| `ap_r2` | `float` | Step 3 | Step 4 card |
| `ap_pred_history` | `list[tuple]` | Step 4 | Step 4 chart + report |

---

## CLI Launch

```bash
# Full dashboard (4 tabs)
streamlit run dashboard/app.py

# Headless modes
python run.py --mode batch  --data Joy_Data/Ethanol
python run.py --mode sensor --gas Ethanol --duration 3600
python run.py --mode simulate --duration 5
```

---

## End-to-End Research Flow

```
[CCS200 Spectrometer]
       │
       ▼
  Step 1A ─ 📡 Live Data View         ← real-time spectral preview, SNR/noise QC
       │
       ▼
  Step 1B ─ 📋 Data Logging            ← metadata form → record snapshots → auto-save CSV
       │                                  (multiple gases × concentrations × trials)
       ▼
  Step 2  ─ Preprocessing              ← denoise → baseline remove → normalize → QC table
       │
       ▼
  Step 3  ─ Insights & Training        ← features → calibration curve → heatmap →
       │     3D surface → confusion matrix → train GPR/CNN
       ▼
  Step 4  ─ Deployment & Testing       ← live predictions → ±1σ confidence → test CSV eval
                                          → session report (Markdown, downloadable)
```

---

## Done When
- All 4 steps render without error in a single browser session.
- Full pipeline executes end-to-end: raw hardware → final prediction.
- Session report generated, downloaded, and saved to `reports/`.
- Dashboard navigable without page reload or state loss between steps.
