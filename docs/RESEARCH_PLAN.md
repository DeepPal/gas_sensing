# SpectraAgent — Complete Research & Implementation Plan

> **Vision**: A universal, sensor-agnostic, data-driven spectroscopic sensing platform that
> automatically discovers discriminative features from ANY spectrometer dataset — regardless
> of sensor type, sensing material, or analyte physics — enabling accurate identification and
> quantification of analytes in real-world conditions.

---

## The Core Scientific Problem

Traditional spectroscopic sensing is locked to specific sensor physics:
- One sensor type → one physics model → one analyte → linear calibration
- Cross-interference is treated as noise to suppress
- Switching sensor or analyte requires rebuilding everything from scratch

**Our approach is fundamentally different**: given ANY spectrometer time-series dataset,
the platform automatically discovers the hidden structure that discriminates analytes —
without being told the sensor material, chemistry, or physical model.

### What Makes This Novel

1. **Truly sensor-agnostic**: no assumptions about peak shape, sensor material, or adsorption physics
2. **Data-driven feature discovery**: algorithms learn what discriminates analytes directly from data
3. **Real-world generalisation**: transfers across sensor configurations with minimal fine-tuning
4. **Physics-agnostic kinetic encoding**: captures response dynamics (fast/slow, transient shape)
   without fitting an explicit physical model — the data reveals its own temporal structure
5. **Multi-analyte from single sensor**: resolves mixtures without a sensor array
6. **Simulation as diversity engine**: generates training data across configurations,
   not to model specific physics but to maximise generalisation

### Target Accuracy
Algorithms must achieve high accuracy on real-world data from diverse sensors and analytes.
The benchmark is: train on sensor config A, deploy on sensor config B with ≤50 fine-tuning spectra.

---

## Sensor Configuration Space

The platform must handle all four configurations:

| Config | Description | Feature Dimensionality |
|--------|-------------|------------------------|
| **1** | Single sensor, 1 peak, 1 analyte | 6D per peak |
| **2** | Single sensor, M peaks, 1 analyte | 6D × M (richer feature vector) |
| **3** | Single sensor, M peaks, N analytes (mixtures) | 6D × M per analyte + cross-terms |
| **4** | Single sensor, M peaks, N analytes + cross-interference | Full sensitivity matrix S[N×M] |

---

## Phase 1: Physics-Complete Simulation Engine

**Status**: ✅ DONE — `src/simulation/gas_response.py` implements full Langmuir kinetics with multi-analyte mixtures, Lorentzian/Gaussian/Fano peak shapes, environmental drift, and realistic spectrometer noise. `spectraagent/drivers/simulation.py` uses this engine for live acquisition simulation.

### 1A: Gas Response Simulation

**Langmuir adsorption kinetics** (1:1 binding model):
```
Δλ(t) = Δλ_eq × (1 − exp(−t / τ))       # association phase
Δλ(t) = Δλ_eq × exp(−t / τ_off)          # dissociation phase
Δλ_eq = S × c / (1 + c / K_d)             # Langmuir isotherm (non-linear at high conc)
```

Where:
- `S` = sensitivity (nm/ppm) — sensor+analyte specific, must be calibrated
- `c` = concentration (ppm)
- `K_d` = dissociation constant (ppm) — determines saturation
- `τ` = response time constant — ANALYTE-SPECIFIC discriminator
- `k_on = 1/τ`, `k_off = 1/τ_off`

**Multi-peak simulation**: each peak has independent sensitivity to each analyte:
```
Δλ_j(t) = Σᵢ S_ij × f_Langmuir(cᵢ, K_d_ij) × kinetics(t, τᵢⱼ)
```
where S_ij = sensitivity of peak j to analyte i.

**Multi-analyte mixture**: superposition (linear regime) or competitive binding (high conc):
```
Δλ_j_observed = Σᵢ S_ij × cᵢ × (1 − exp(−t/τᵢⱼ)) + ε_thermal + ε_noise
```

### 1B: Noise Model

Realistic spectrometer noise has three components:
```
I_measured(λ) = I_true(λ) + η_shot(λ) + η_dark + η_readout
```
- `η_shot ~ Poisson(I_true)` — photon shot noise (dominant at high signal)
- `η_dark ~ N(0, σ_dark)` — dark current thermal noise (increases with T, integration time)
- `η_readout ~ N(0, σ_readout)` — fixed electronics noise floor
- `η_speckle` — optional: coherence noise for laser sources

### 1C: Environmental Effects

Temperature and humidity shift the peak INDEPENDENTLY of analyte:
```
Δλ_T = α_T × ΔT      # thermal drift coefficient (nm/°C)
Δλ_RH = α_RH × ΔRH   # humidity sensitivity (nm/%RH)
Δλ_observed = Δλ_gas + Δλ_T + Δλ_RH + noise
```
These must be corrected before concentration estimation.

### 1D: Sensor-to-Sensor Variation

Manufacturing variation causes:
- Peak position scatter: ± δλ_0 (fixed offset per chip)
- Sensitivity scatter: ± δS (scaling factor per chip)
- FWHM variation: ± δγ (broadening per chip)

Used for domain randomization in training.

---

## Phase 2: Feature Engineering — Kinetic-Spectral Fusion

### 2A: Per-Peak Spectral Features (per analyte-peak pair)

For each detected spectral peak j:
```
f_spectral_j = [Δλⱼ, ΔFWHMⱼ, ΔAⱼ, ΔI_area_j, Δasymmetry_j]
```
5 physically orthogonal features per peak.

### 2B: Kinetic Features (the discriminative power)

From the Δλ(t) transient curve:
```
f_kinetic_j = [τ₆₃_j, τ₉₅_j, k_on_j, Δλ_eq_j, R²_fit_j]
```
- `τ₆₃` and `τ₉₅` are analyte-specific even when `Δλ_eq` is identical
- Two analytes with same `Δλ_eq` but different `τ` are PERFECTLY discriminable by kinetics

### 2C: Cross-Peak Ratio Features

When multiple peaks are present:
```
f_cross = [Δλⱼ / Δλₖ for all pairs j,k]
          [τⱼ / τₖ for all pairs j,k]
```
These ratios are analyte-specific signatures independent of concentration magnitude.

### 2D: Full Feature Vector

For M peaks and a session with kinetic data:
```
F = [f_spectral_0, f_kinetic_0, f_spectral_1, f_kinetic_1, ..., f_cross]
```
Dimensionality: M × 10 + M(M-1)/2 cross-terms.

For M=3 peaks: 30 + 3 = 33 features per analyte per session.

### 2E: Temperature Compensation

Before feature extraction:
```
Δλ_compensated_j = Δλ_raw_j − α_T × ΔT − α_RH × ΔRH
```
α_T and α_RH are calibrated per sensor type from blank runs at varying T, RH.

---

## Phase 3: Calibration System Redesign

### 3A: Sensitivity Matrix

For N analytes and M peaks, the sensitivity matrix S ∈ ℝ^{N×M}:
```
[Δλ₁]   [S₁₁  S₁₂ ... S₁ₙ] [c₁]   [ε₁]
[Δλ₂] = [S₂₁  S₂₂ ... S₂ₙ] [c₂] + [ε₂]
[...]    [...]               [...]   [...]
```

**Calibration protocol**:
1. Run each analyte ALONE at 3+ concentrations → estimate column of S
2. Run binary mixtures to validate cross-terms
3. Fit S via ordinary least squares (or regularized if under-determined)

**Concentration estimation** (inference):
```
ĉ = S⁺ × Δλ_observed    # Moore-Penrose pseudo-inverse
```
For non-linear (Langmuir), solve:
```
minimize ||Δλ_observed − Σᵢ S_col_i × cᵢ / (1 + cᵢ/K_d_i)||²
```

### 3B: Multi-Output GPR

Replace scalar GPR with multi-output GPR:
- Input: feature vector F (spectral + kinetic)
- Output: [c₁, c₂, ..., cₙ] — concentrations of all analytes simultaneously
- Kernel: Matérn-5/2 with separate lengthscales per feature dimension
- Uncertainty: full covariance matrix → correlated concentration uncertainties

### 3C: LOD/LOQ in Mixture Context

Classic LOD = 3σ_blank / slope is only valid for single analyte.

For multi-analyte:
```
LOD_i = 3 × sqrt(σ_blank^T × (S S^T)^{-1} × σ_blank) for analyte i
```
accounts for cross-interference contribution to uncertainty.

### 3D: Selectivity Matrix (IUPAC)

Modified selectivity coefficient:
```
K_ij = S_ij / S_ii    # interference of analyte j on analyte i measurement
```
If K_ij >> 1: analyte j strongly interferes with i measurement at that peak.
Multi-peak helps: choose features where K_ij is minimized.

---

## Phase 4: ML Architecture

### 4A: Physics-Informed Feature Extraction Network

Not a black-box CNN — a **physics-guided architecture**:
```
Input: raw differential spectrum (Δ_spectrum = I_gas − I_ref)
Layer 1: Learnable peak detection (attention over wavelength axis)
Layer 2: Per-peak Lorentzian parameter extraction (center, FWHM, amplitude)
Layer 3: Kinetic model fitting head (τ, k_on from temporal sequence)
Output: [Δλ_j, ΔFWHM_j, τ_j, k_on_j, ...] — interpretable features
```

This is more powerful than a vanilla CNN because the intermediate representations
have physical meaning and the network can be pre-trained on simulation.

### 4B: Multi-Task Learning

Single model, multiple heads:
```
Shared encoder (spectral + kinetic features)
├── Classification head: P(analyte_type | features) — multi-label
├── Regression head: [c₁, c₂, ..., cₙ] — multi-output concentration
└── Quality head: [LOD_flag, saturation_flag, drift_flag]
```

### 4C: Simulation-to-Real Transfer

**Domain randomization** during training:
- Randomize peak position (± 20 nm from any center)
- Randomize FWHM, amplitude, noise level
- Randomize thermal drift rate, temperature offset
- Randomize sensitivity (± 30% of nominal)

**Fine-tuning protocol**:
1. Pre-train on large synthetic dataset (10⁵ spectra from simulation)
2. Fine-tune on real sensor data (N=50-200 real calibration sessions)
3. Validate on held-out real sessions

### 4D: Active Learning for Calibration Efficiency

The planner agent suggests next concentration to measure:
- Choose c that maximally reduces uncertainty in S matrix
- Formally: Bayesian optimal experimental design (maximize information gain)
- Implementation: Expected improvement over current S estimate uncertainty

---

## Phase 5: Platform Architecture

### 5A: FastAPI Backend Additions

New endpoints:
- `POST /api/calibration/sensitivity-matrix` — fit S from calibration sessions
- `POST /api/inference/mixture` — estimate all concentrations simultaneously
- `GET /api/analytes` — list registered analytes with S matrix entries
- `POST /api/simulation/generate` — generate synthetic training spectra

### 5B: React Frontend (Live Acquisition)

New panels:
- **Multi-analyte concentration display**: real-time bar chart of [c₁, c₂, ..., cₙ]
- **Sensitivity matrix viewer**: heatmap of S matrix
- **Kinetic phase indicator**: association / equilibrium / dissociation
- **Uncertainty bands**: per-analyte 95% CI from GPR

### 5C: Streamlit Analysis Workbench

New tabs:
- **Mixture Analysis**: input multiple analyte concentrations, view predicted vs actual
- **Sensitivity Matrix**: calibrate and visualize S[N×M]
- **Selectivity**: compute and visualize K_ij cross-interference map
- **Training Data Generator**: configure simulation parameters, generate dataset
- **Model Training**: train CNN/GPR from within the UI

---

## Phase 6: Validation & Publication

### 6A: Benchmark Experiments

For each sensor configuration:
1. Single analyte calibration (LOD, LOQ, linearity, reproducibility)
2. Binary mixture — known concentrations, predict both
3. Ternary mixture (if M peaks ≥ 3)
4. Cross-interference matrix (K_ij for all pairs)
5. Session-to-session reproducibility (N=5 sessions, Bland-Altman)
6. Temperature stability (T = 20, 25, 30°C)

### 6B: Figures for Publication

1. System architecture schematic
2. Multi-peak spectral response (reference + gas exposure)
3. Langmuir kinetic fit (Δλ vs t, with τ annotation)
4. Sensitivity matrix heatmap S[N×M]
5. Selectivity matrix heatmap K[N×N] per peak
6. LOD comparison: single-peak vs multi-peak vs kinetic fusion
7. Mixture deconvolution: predicted vs actual for binary/ternary mixtures
8. Transfer learning curve: sim-pretrained vs from-scratch vs fine-tuned

### 6C: Target Journals

- *Analytical Chemistry* (ACS) — methods paper
- *Biosensors and Bioelectronics* — if bio application included
- *Sensors and Actuators B* — sensor-focused

---

## Implementation Order (Critical Path)

```
Week 1-2:  Phase 1 — Physics simulation (gas response + noise + multi-analyte)
Week 3-4:  Phase 2 — Feature engineering (kinetic features + cross-peak ratios)
Week 5-6:  Phase 3 — Calibration redesign (sensitivity matrix + multi-output GPR)
Week 7-8:  Phase 4A-B — ML architecture (physics-informed + multi-task)
Week 9-10: Phase 4C-D — Transfer learning + active learning
Week 11:   Phase 5 — Platform UI (React + Streamlit)
Week 12:   Phase 6 — Validation + publication pipeline
```

---

## File Structure (New Modules)

```
src/
├── simulation/
│   ├── __init__.py
│   ├── gas_response.py          # Langmuir kinetics, multi-analyte mixture
│   ├── noise_model.py           # Shot/dark/readout noise, realistic spectra
│   ├── sensor_variation.py      # Chip-to-chip manufacturing variation
│   └── dataset_generator.py    # Generate large training datasets
│
├── calibration/
│   ├── sensitivity_matrix.py    # S[N×M] estimation and inversion
│   ├── mixture_deconvolution.py # Non-linear mixture separation
│   ├── multi_output_gpr.py      # Multi-analyte GPR
│   └── lod_mixture.py          # LOD/LOQ accounting for cross-interference
│
├── features/
│   ├── lspr_features.py         # (existing, extended)
│   ├── kinetic_features.py      # (existing B1, extended)
│   ├── cross_peak_features.py   # NEW: ratio features between peaks
│   └── compensation.py          # NEW: T/RH compensation
│
├── models/
│   ├── physics_cnn.py           # Physics-informed CNN architecture
│   ├── multi_task.py            # Multi-task learning (classify + regress + QC)
│   └── transfer.py              # Sim→real transfer learning
│
└── training/
    ├── synthetic_dataset.py     # Simulation-based dataset creation
    ├── trainer.py               # Training loop
    └── active_learning.py       # Bayesian optimal experiment design
```

---

## Key Scientific Claims (Testable Hypotheses)

1. **H1**: Kinetic features (τ, k_on) discriminate analytes with ≤15% Δλ overlap better than
   spectral features alone (target: F1 improvement ≥ 20%).

2. **H2**: Multi-output GPR with sensitivity matrix achieves lower mixture quantification error
   than single-analyte GPR applied independently (target: RMSE reduction ≥ 30%).

3. **H3**: Physics-informed CNN pre-trained on simulation data requires ≤50 real calibration
   spectra to match accuracy of a from-scratch model trained on ≥500 real spectra.

4. **H4**: The M-peak feature vector reduces LOD by a factor of √M compared to single-peak
   measurement (theoretical: noise averages out across peaks, signal adds coherently).

---

## Current Status

> **Last updated: 2026-04-01** — All phases 1–5 complete. Phase 6 (validation with real experimental data) in progress.

| Component | Status |
|-----------|--------|
| Sensor-agnostic framework | ✅ Done |
| Multi-peak detection | ✅ Done |
| Single-peak kinetic features (B1) | ✅ Done — `src/features/lspr_features.py` (`estimate_response_kinetics`, `KineticFeatures`) |
| Gas response simulation | ✅ Done — `src/simulation/gas_response.py` (Langmuir kinetics, multi-analyte, Fano/Lorentzian/Gaussian peaks) |
| Multi-analyte noise model | ✅ Done — `src/simulation/noise_model.py` (shot/dark/readout/PRNU, domain randomization) |
| Training data pipeline | ✅ Done — `src/simulation/dataset_generator.py` (calibration, mixture, kinetic datasets) |
| Sensitivity matrix calibration | ✅ Done — `src/calibration/sensitivity_matrix.py` (OLS fit, pseudoinverse, LOD, condition number) |
| Mixture deconvolution | ✅ Done — `src/calibration/mixture_deconvolution.py` (linear lstsq + Langmuir L-BFGS-B) |
| Multi-output GPR | ✅ Done — `src/calibration/multi_output_gpr.py` (independent + joint, kinetic-spectral fusion) |
| Cross-peak ratio features | ✅ Done — `src/features/cross_peak_features.py` (SAM, ratios, PCA, pattern matching) |
| T/RH compensation | ✅ Done — `src/features/compensation.py` (OLS environmental, differential, EMA adaptive, polynomial detrend) |
| Physics-complete simulation driver | ✅ Done — `spectraagent/drivers/simulation.py` rewritten using SpectralSimulator |
| Multi-analyte FastAPI endpoints | ✅ Done — `/api/analytes`, `/api/inference/mixture`, `/api/calibration/sensitivity-matrix/fit`, `/api/simulation/generate` |
| Multi-analyte React UI | ✅ Done — kinetic phase badge, analyte bar chart, S-matrix table, residual display |
| Cross-session analysis (B2) | ✅ Done — `src/scientific/cross_session.py` (paired t-test, Bland-Altman, F-test, Mann-Whitney) |
| Selectivity tracking (B3) | ✅ Done — `SensorMemory.get_sensitivities_by_analyte()`, `CalibrationValidationOrchestrator._update_selectivity()` |
| Reference FWHM measurement (B4) | ✅ Done — Lorentzian fit on reference spectrum in `server.py acq_reference` |
| Environmental metadata (B5) | ✅ Done — `temperature_c` / `humidity_pct` in `AcquisitionConfig` + `SessionWriter` |
| Multi-task learning (Phase 4B) | ✅ Done — `src/models/multi_task.py` (`MultiTaskModel`: shared encoder + classification + regression + QC heads, 3 backbone choices) |
| Sim→real transfer / domain adapt (Phase 4C) | ✅ Done — `src/models/transfer.py` (`DomainAdaptModel` with GRL, `fine_tune()` for few-shot adaptation) |
| Spectral autoencoder (Phase 5C feature discovery) | ✅ Done — `src/models/spectral_autoencoder.py` (encoder–decoder, `train_autoencoder` with epoch callback) |
| Contrastive learning | ✅ Done — `src/models/contrastive.py` (SupCon + triplet, `train_contrastive`) |
| Feature attribution (gradient, IG, SHAP) | ✅ Done — `src/analysis/feature_importance.py` |
| Cross-dataset benchmark | ✅ Done — `src/analysis/cross_dataset_eval.py` (LOO, PCA or custom encoder, sklearn classifiers) |
| Embedding visualisation | ✅ Done — `src/analysis/embedding_viz.py` (UMAP/t-SNE/PCA, Plotly) |
| Model versioning | ✅ Done — `src/models/versioning.py` (`ModelVersionStore`: save/promote/rollback/compare) |
| Data-Driven Science dashboard (Phase 5C) | ✅ Done — `dashboard/science_tab.py` (5 sub-tabs: Dataset Explorer, Feature Discovery, Model Training, Cross-Dataset, Publication Figures) |
| Robustness CLI (B7) | ✅ Done — `spectraagent robustness` command (`spectraagent/commands/robustness.py`) |
| Physics-informed CNN (Phase 4A) | ⏳ Future — learnable peak detection + Lorentzian parameter extraction as network layers |
| Mixture UI (binary/ternary display) | ⏳ Future — mixture deconvolution code exists; UI not yet wired |
| Benchmark experiments with real data | ⏳ In progress — requires multi-config experimental datasets |
| Publication figures (final) | ⏳ In progress — `dashboard/science_tab.py` Tab 5 exports figures; awaiting real data |

### Known Workflow Gaps (see REMAINING_WORK.md Part C)

- **C1 (HIGH):** `spectraagent start` session CSVs are processed results, not raw spectra. Tab 5 Dataset Explorer needs a "load from session" converter.
- **C2 (MEDIUM):** Models trained in Tab 5 must be manually copied to `models/registry/` to activate in live acquisition — no UI button yet.
- **C3 (MEDIUM):** Cross-dataset benchmark requires ≥3 sensor configs. Use `spectraagent robustness` to generate config variants if only 1 sensor is available.
