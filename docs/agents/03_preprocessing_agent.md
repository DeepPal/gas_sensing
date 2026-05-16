# Agent 03 — Preprocessing Pipeline
**Stage A · Step 3 of 6 · Dashboard: Step 2**

---

## Purpose
Apply a configurable, research-grade preprocessing pipeline on raw recorded CSVs to maximise signal quality before feature extraction. Produces clean, normalised spectra and a QC improvement report.

---

## Source Files
| File | Role |
|------|------|
| `dashboard/agentic_pipeline_tab.py` | Step 2 — full preprocessing UI |
| `gas_analysis/core/signal_proc.py` | `smooth_spectrum()`, `als_baseline()` |
| `gas_analysis/core/preprocessing.py` | `baseline_correction()`, `normalize_spectrum()`, `compute_snr()`, `estimate_noise_metrics()`, `detect_outliers()` |

---

## Dashboard Location
`dashboard/agentic_pipeline_tab.py` → **Step 2 — Preprocessing Pipeline**

---

## Data Sources (selectable in UI)
| Option | Source |
|--------|--------|
| Session recordings (in memory) | `ss["ap_buffer"]` from Agent 02 |
| Load from disk | CSVs under `data/automation_dataset/` via `_scan_dataset_dir()` |

---

## Pipeline Stages

### 1. Load Raw Data
```python
# From session buffer
items = [(r["label"], r["wl"], r["intensity"]) for r in ss["ap_buffer"]]
# From disk (multiselect)
df = pd.read_csv(path)
wl, intensity = df["wavelength"].values, df["intensity"].values
```

### 2. Outlier Rejection
Applied automatically on disk-loaded batches:
```python
from gas_analysis.core.preprocessing import detect_outliers
flags = detect_outliers(spectra, threshold=3.0)   # MAD z-score
clean = [s for s, f in zip(spectra, flags) if not f]
```

### 3. Denoising (user choice)
| Method | Call |
|--------|------|
| Savitzky-Golay | `smooth_spectrum(sig, window=11, poly_order=2)` |
| Wavelet DWT-db4 | `smooth_spectrum(sig, method="wavelet")` |
| None | skip |

### 4. Baseline Removal (user choice)
| Method | Call |
|--------|------|
| ALS | `sig - als_baseline(sig, lam=1e5, p=0.01)` |
| Polynomial | `baseline_correction(wl, sig, method="polynomial", poly_order=2)` |
| None | skip |

### 5. Normalization (user choice)
| Method | Call |
|--------|------|
| Min-Max [0,1] | `normalize_spectrum(sig, method="minmax")` |
| Z-score | `normalize_spectrum(sig, method="standard")` |
| None | skip |

The full chain is implemented in `_preprocess(wl, intensity, denoise, baseline, norm)`.

---

## QC Report Table
Displayed after processing:
| Label | Raw SNR | Proc SNR | Noise (raw) | Noise (proc) | QC |
|-------|---------|----------|------------|-------------|-----|
| Ethanol_100.0ppm | 12.4 | 48.7 | 0.0082 | 0.0021 | ✅ |

```python
snr_raw  = compute_snr(raw)
snr_proc = compute_snr(proc)
noise    = estimate_noise_metrics(wl, proc).rms
```

---

## Visualisation
Raw vs. Preprocessed overlay using Plotly:
```python
fig.add_trace(go.Scatter(x=wl, y=raw,  name="Raw",          line=dict(color="gray", dash="dot")))
fig.add_trace(go.Scatter(x=wl, y=proc, name="Preprocessed", line=dict(color="royalblue", width=2)))
```

---

## Session State Outputs
| Key | Type | Consumed By |
|-----|------|------------|
| `ap_preprocessed` | `list[dict]` | Agent 04 |

Each item:
```python
{"label": str, "wl": np.ndarray, "raw": np.ndarray, "processed": np.ndarray}
```

---

## Done When
- All selected spectra preprocessed with SNR improvement documented.
- `ap_preprocessed` populated with ≥1 clean spectrum.
- → Pass control to **Agent 04** (Feature Extraction & Model Training).
