# Agent 05 — Model Deployment & Real-Time Inference
**Stage B · Step 5 of 6 · Dashboard: Step 4**

---

## Purpose
Deploy the trained GPR/CNN model for real-time inference on new spectra. Run live predictions with ±1σ confidence bands, evaluate accuracy against uploaded test CSVs, and generate a downloadable session report.

---

## Source Files
| File | Role |
|------|------|
| `dashboard/agentic_pipeline_tab.py` | Step 4 — full deployment UI |
| `models/gpr_sklearn.pkl` | Sklearn GPR (primary fallback model) |
| `models/gpr_calibration.pkl` | Custom GPR (if trained) |
| `models/gas_cnn.pt` | CNN classifier (if trained) |
| `gas_analysis/core/intelligence/classifier.py` | `CNNGasClassifier.load()` |
| `gas_analysis/core/intelligence/gpr.py` | `GPRCalibration.load()` |

---

## Dashboard Location
`dashboard/agentic_pipeline_tab.py` → **Step 4 — Deployment & Real-Time Inference**

---

## Model Loading

### From Session (primary path)
If `ap_model_trained = True` in `st.session_state`, the trained sklearn GPR is already in `ss["ap_gpr_sklearn"]`.

### From Disk (fallback / independent deployment)
If no model was trained in the current session:
```python
model_dir = _REPO / "models"
pkl_files = sorted(model_dir.glob("*.pkl"))
# UI: st.selectbox("Select model file", pkl_files)
import pickle
with open(chosen_pkl, "rb") as f:
    model = pickle.load(f)
ss["ap_gpr_sklearn"] = model
ss["ap_model_trained"] = True
```
This allows deploying a previously saved model without re-training.

---

## Inference Pipeline (per prediction)

### Step 1: Acquire Frame
```python
wl_live, frame = _acquire_frames(integration_ms, n_frames, concentration_ppm, chart_ph)
```

### Step 2: Preprocess (must match training pipeline)
```python
proc = _preprocess(wl_live, frame, "Savitzky-Golay", "ALS", "Min-Max [0,1]")
```

### Step 3: Feature Extraction
```python
peak_idx = int(np.argmax(proc))
feat = np.array([[
    float(proc[peak_idx]),            # peak intensity
    float(wl_live[peak_idx]),         # peak wavelength
    float(np.trapz(proc, wl_live)),   # spectral area
    float(np.std(proc)),              # spectral variance
]])
```

### Step 4: Predict
```python
pred_conc, pred_std = gpr_model.predict(feat, return_std=True)
confidence = 1.0 - min(1.0, pred_std[0] / (abs(pred_conc[0]) + 1e-6))
```

---

## Live Display

### Prediction Gauge
```python
go.Indicator(
    mode="gauge+number+delta",
    value=conc_val,
    delta={"reference": meta.get("concentration_ppm", conc_val)},  # vs. known value
    gauge={"axis": {"range": [0, max(600, conc_val*1.5)]}, ...}
)
```

### Prediction History with ±1σ Band
```python
fig_ts.add_trace(go.Scatter(y=means, mode="lines+markers", name="Predicted"))
fig_ts.add_trace(go.Scatter(y=[m+s for m,s in zip(means,stds)], fill="tonexty",
                            fillcolor="rgba(255,165,0,0.2)", name="±1σ"))
```
When ≥3 predictions accumulated, shows MAE vs. true concentration.

---

## Accuracy Testing (Test CSV Upload)

Upload labeled CSVs with concentration encoded in the filename (e.g. `Ethanol_100ppm_test.csv`):
```python
test_files = st.file_uploader(..., type=["csv"], accept_multiple_files=True)
# For each file:
#   1. Read wavelength/intensity
#   2. Run same _preprocess() chain
#   3. Extract features, call gpr.predict()
#   4. Parse true concentration from filename via regex
#   5. Compute absolute error
# Show results table + Mean Absolute Error metric
```

---

## Session Report Generation
```python
report = f"""# Gas Sensing Session Report
**Gas:** {meta.get('gas')}
**Concentration:** {meta.get('concentration_ppm')} ppm
**LOD:** {ss.get('ap_lod')} ppm
**R²:** {ss.get('ap_r2')}
**Total Predictions:** {len(pred_history)}
**Mean Predicted Conc.:** {np.mean([h[0] for h in pred_history]):.2f} ppm
"""
# saved to: reports/session_{YYYYMMDD_HHMMSS}.md
st.download_button("⬇️ Download Report", report, ...)
```

---

## Session State Consumed
| Key | Written By | Usage |
|-----|-----------|-------|
| `ap_model_trained` | Agent 04 | Gate check |
| `ap_gpr_sklearn` | Agent 04 | Prediction engine |
| `ap_meta` | Agent 02 | Inference params + report |
| `ap_lod` | Agent 04 | Status card |
| `ap_r2` | Agent 04 | Status card |
| `ap_pred_history` | Agent 05 | Time series + report |

---

## Done When
- Model loaded (from session or disk).
- Single-shot predictions working with gauge + confidence metrics.
- Test CSV evaluation producing MAE table.
- Session report generated and downloadable.
- → Pipeline complete. Return to **Step 1** to run a new experiment or retrain.
