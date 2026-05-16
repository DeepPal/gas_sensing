# Agent 02 — Pre-Acquisition Logging & Labeled Data Recording
**Stage A · Step 2 of 6 · Dashboard: Step 1 (bottom section)**

---

## Purpose
Before recording begins, capture structured experimental metadata (gas type, concentration, trial number, comments). Provide recording controls that automatically save labeled, timestamped CSV files tied to the confirmed metadata.

---

## Source Files
| File | Role |
|------|------|
| `dashboard/agentic_pipeline_tab.py` | Step 1 → "📋 Data Logging & Acquisition" section |
| `data/automation_dataset/` | Auto-created output directory tree |

---

## Dashboard Location
`dashboard/agentic_pipeline_tab.py` → Step 1 → **"📋 Data Logging & Acquisition"** section (below Live Preview)

---

## Behaviour

### 1. Metadata Form (locked after submission)
```python
with st.form("ap_meta_form"):
    gas      = st.selectbox("Gas Type", ["Ethanol","Methanol","Isopropanol","MixVOC","Air","Custom"])
    conc     = st.number_input("Concentration (ppm)", min_value=0.0, value=100.0, step=10.0)
    trial    = st.number_input("Trial #", min_value=1, step=1, value=1)
    int_ms   = st.slider("Integration time (ms)", 10, 5000, 30, 10)
    n_frames = st.slider("Frames to average", 5, 200, 30)
    comments = st.text_input("Comments (optional)")
    submitted = st.form_submit_button("✅ Confirm Metadata & Arm Recording", type="primary")
```
- Metadata is stored in `st.session_state["ap_meta"]`.
- Buffer is only reset when gas/concentration changes, not on every resubmit.

### 2. Recording (Snapshot Mode)
```python
start = st.button("▶️ Record Snapshot", type="primary",
                  disabled=not st.session_state.get("ap_meta"))
```
- Calls `_acquire_frames(integration_ms, n_frames, concentration_ppm, chart_ph)`.
- Runs QC gates — rejects if SNR < 10 or RMS noise ≥ 0.1.
- On pass: auto-saves CSV and JSON metadata sidecar.

### 3. Auto-Save CSV + Metadata Sidecar
```
data/automation_dataset/{gas}/{conc}ppm/trial_{n}/
  ├── {gas}_{conc}ppm_T{n}_{YYYYMMDD_HHMMSS}.csv   ← wavelength, intensity columns
  └── metadata_{YYYYMMDD_HHMMSS}.json                ← full metadata dict
```
```python
pd.DataFrame({"wavelength": wl, "intensity": mean_int}).to_csv(csv_path, index=False)
json.dump(meta, open(meta_path, "w"), indent=2)
```

### 4. Session Recording Log
Displayed as a live-updating table:
| Label | Path | QC |
|-------|------|-----|
| Ethanol_100.0ppm | Ethanol_100.0ppm_T1_20260226_103000.csv | ✅ |

### 5. Quality Requirements Before Saving
| Check | Threshold | Action if fails |
|-------|-----------|-----------------|
| SNR | ≥ 10.0 | Discard frame, show error |
| RMS Noise | < 0.1 | Discard frame |
| Saturation | frame.max() < 60 000 | Flag as ⚠️ |
| Min frames | ≥ 10 averaged | Reject recording |

---

## Session State Outputs
| Key | Type | Consumed By |
|-----|------|------------|
| `ap_meta` | `dict` | Agents 03–05, Step 4 inference params |
| `ap_buffer` | `list[dict]` | Agent 03 (in-memory source) |
| `ap_wl` | `np.ndarray` | Agents 03–05 |

Buffer entry schema:
```python
{"wl": np.ndarray, "intensity": np.ndarray, "label": "Gas_conc_ppm", "path": "/abs/path/to.csv"}
```

---

## Output Directory
```
data/automation_dataset/{gas_type}/{conc}ppm/trial_{n}/
```
Consumed by Agent 03 via `_scan_dataset_dir()`.

---

## Done When
- At least one CSV per concentration level saved with QC metadata.
- Session log shows all recordings in the experiment.
- → Pass control to **Agent 03** (Preprocessing Pipeline).
