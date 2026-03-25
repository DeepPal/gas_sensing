# Agent 04 — Feature Extraction, Insights & Model Training
**Stage A · Step 4 of 6 · Dashboard: Step 3**

---

## Purpose
Analyse preprocessed spectra to extract scientifically meaningful features. Visualise calibration curves, sensitivity heatmaps, and 3D response surfaces. Train and evaluate ML models (GPR for concentration, CNN for gas type) with publication-grade performance metrics.

---

## Source Files
| File | Role |
|------|------|
| `dashboard/agentic_pipeline_tab.py` | Step 3 — full insights + training UI |
| `gas_analysis/core/intelligence/classifier.py` | `CNNGasClassifier` |
| `gas_analysis/core/intelligence/gpr.py` | `GPRCalibration` |
| `gas_analysis/core/scientific/lod.py` | `calculate_lod_3sigma`, `calculate_sensitivity` |
| `gas_analysis/core/scientific/kinetics.py` | `calculate_t90_t10` |

---

## Dashboard Location
`dashboard/agentic_pipeline_tab.py` → **Step 3 — Insights, Feature Extraction & Model Training**

---

## Feature Extraction

### Per-Spectrum Features
```python
peak_idx       = int(np.argmax(processed))
peak_intensity = float(processed[peak_idx])       # Peak amplitude
peak_wavelength= float(wl[peak_idx])              # Peak position (nm)
area           = float(np.trapz(processed, wl))   # Integrated area under curve
spectral_std   = float(np.std(processed))         # Spectral variance
```

Feature matrix shape: `(n_samples, 4)` — used for both GPR and sklearn fallback.

### Label Parsing
- Concentration extracted from label string via regex: `re.search(r"([\d.]+)ppm", label)`
- Gas class extracted as the first underscore-delimited token in the label string.

---

## Scientific Metrics

### LOD / LOQ / Sensitivity / R²
```python
from gas_analysis.core.scientific.lod import calculate_lod_3sigma, calculate_sensitivity
slope, intercept, r2 = calculate_sensitivity(concentrations, peak_responses)
noise_floor = np.std(peak_arr[:max(1, len(peak_arr)//5)])
lod = calculate_lod_3sigma(noise_floor, slope)    # 3σ / slope
loq = 10 * noise_floor / abs(slope)              # 10σ / slope
```
Displayed as 4-column metric cards: Sensitivity · R² · LOD · LOQ.

---

## Visualisations (all rendered as interactive Plotly figures)

### 1. Spectral Overlay (Rainbow Plot)
All preprocessed spectra plotted together, colour-coded by concentration using HSL gradient.

### 2. Calibration Curve
Peak intensity vs. concentration with linear fit and R² annotation. Uses `scipy.stats.linregress`.
- Auto-selects the global peak wavelength (argmax of mean spectrum).

### 3. Sensitivity Heatmap
`go.Heatmap(z=Z, x=wl_common, y=concs_plot, colorscale="Viridis")`
Reveals which wavelength bands are most responsive to concentration changes.

### 4. 3D Response Surface
`go.Surface(z=Z, x=wl_common, y=concs_plot)` — interactive 3D view of spectral-concentration landscape.

### 5. Confusion Matrix (multi-class only)
`sklearn.neighbors.NearestCentroid` with 3-fold cross-validation.
Plotted as annotated heatmap via `plotly.figure_factory.create_annotated_heatmap`.
Shown only when ≥2 gas classes and ≥4 samples are present.

---

## Model Training

### Option A: Gaussian Process Regression (GPR — Concentration)
```python
# Custom module (preferred):
from gas_analysis.core.intelligence.gpr import GPRCalibration
gpr = GPRCalibration()
gpr.fit(X, y_concs)
gpr.save("models/gpr_calibration.pkl")

# sklearn fallback (always available):
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
gpr_sk = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, normalize_y=True)
gpr_sk.fit(X, y_concs)
# saved to: models/gpr_sklearn.pkl
```

### Option B: 1D CNN Classifier (Gas Type)
```python
from gas_analysis.core.intelligence.classifier import CNNGasClassifier
# Resamples all spectra to 1000 points on [200, 1000] nm
X_raw = np.array([np.interp(np.linspace(200,1000,1000), it["wl"], it["processed"]) for it in pp])
clf = CNNGasClassifier(input_length=1000, num_classes=len(class_names))
history = clf.fit(X_raw, y_labels, y_concs, class_names=class_names, epochs=20)
clf.save("models/gas_cnn.pt")
```

---

## Model Output Paths
| Model | Path |
|-------|------|
| sklearn GPR (fallback) | `models/gpr_sklearn.pkl` |
| Custom GPR | `models/gpr_calibration.pkl` |
| CNN Classifier | `models/gas_cnn.pt` |

---

## Performance Targets
| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 95% |
| Calibration R² | ≥ 0.99 |
| LOD | < 1 ppm |
| T90 response time | < 60 s |

---

## Session State Outputs
| Key | Type | Consumed By |
|-----|------|------------|
| `ap_model_trained` | `bool` | Agent 05 gate |
| `ap_gpr_sklearn` | sklearn estimator | Agent 05 inference |
| `ap_X_train` | `np.ndarray` | Agent 05 context |
| `ap_y_concs` | `np.ndarray` | Agent 05 context |
| `ap_class_names` | `list[str]` | Agent 05 display |
| `ap_lod` | `float` | Agent 05 status card |
| `ap_r2` | `float` | Agent 05 status card |

---

## Done When
- Model trained and saved to `models/`.
- LOD, LOQ, Sensitivity, R², calibration curve, heatmap, and 3D surface all displayed.
- `ap_model_trained = True` in session state.
- → Pass control to **Agent 05** (Deployment & Real-Time Inference).
