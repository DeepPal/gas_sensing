# SOUL.md — System Principles & Research Philosophy

## What This System Is

An **spectrometer-based sensing platform** for detecting and quantifying Volatile Organic
Compounds (VOCs) using the localized surface plasmon resonance shift of gold nanoparticles
embedded in a molecularly imprinted polymer film.

The system is NOT:
- A general-purpose spectrometer controller
- A replacement for GC-MS
- A black-box ML pipeline

The system IS:
- A physics-grounded, AI-assisted gas sensing instrument
- A reproducible research platform designed for publication
- A generalized architecture that works with any gas/spectral dataset

---

## The Physics Constraint (NEVER Violate This)

The **primary signal is peak wavelength SHIFT (Δλ), not peak intensity**:

```
Δλ = λ_gas − λ_reference      [nm]
```

| Quantity            | Value                       | Notes                                 |
|---------------------|-----------------------------|---------------------------------------|
| Reference peak      | ~531.5 nm                   | Au nanoparticles, green region        |
| Physical sensitivity | −0.116 nm/ppm              | Literature value for ethanol on the sensor|
| Signal direction    | Negative Δλ on adsorption   | Analyte adsorbs → redshift            |
| Detection range     | 0.1 – 10 ppm typical        | Hardware-limited by SNR               |
| Spectral resolution | 3648 points, 400–700 nm     | ThorLabs CCS200 spectrometer          |

Every ML model, calibration curve, and feature extractor MUST respect this physics:
- Feature vectors must include Δλ as the primary feature
- Concentration estimates are computed as `C = Δλ / sensitivity`
- A model that ignores wavelength shift and uses only intensity has failed to learn LSPR physics

---

## Research Hypotheses

1. **ROI Hypothesis**: Not all spectral regions carry equal information. The optimal ROI
   (Region of Interest) is a 10–25 nm window centered near the LSPR peak that maximizes
   the ratio of slope-to-noise in the calibration curve.

2. **Multi-task Hypothesis**: A single model trained jointly on gas classification +
   concentration regression will outperform two separate models, because the shared
   representation captures gas-specific spectral shape.

3. **Uncertainty Hypothesis**: GPR-based concentration estimation provides calibrated
   uncertainty bounds (±σ ppm) that enable confidence-aware predictions — essential for
   safety-critical VOC detection.

4. **Temporal Gating Hypothesis**: Selecting the stable plateau of a concentration step
   (last 10 frames before gas switch) gives a cleaner training signal than using all frames.

---

## Success Metrics (Paper-Grade)

| Metric                        | Target            | Method                             |
|-------------------------------|-------------------|------------------------------------|
| Calibration R²                | > 0.99            | Leave-one-out cross-validation     |
| Concentration RMSE            | < 0.05 ppm        | Test set, unseen concentrations    |
| Gas classification accuracy   | > 95%             | 5-fold cross-validation            |
| Limit of Detection (LOD)      | < 0.1 ppm         | 3σ/slope method                    |
| Inference latency             | < 50 ms/spectrum  | FastAPI benchmark, CPU only        |
| SNR at 0.5 ppm EtOH           | > 10 dB           | Signal / noise floor               |

---

## Design Principles

### 1. Physics Before Statistics
Before deploying any ML model, verify the model's predictions are physically reasonable.
A concentration estimate of −5 ppm is a bug, not a prediction.

### 2. Reproducibility as a First-Class Requirement
Every experiment is logged in MLflow with:
- Exact preprocessing parameters
- Dataset fingerprint (hash of input files)
- Model architecture + training hyperparameters
- Evaluation metrics
If an experiment cannot be reproduced, it cannot be published.

### 3. Modular by Contract
Each system layer communicates via Pydantic schemas (`src/schemas/`).
Functions accept and return typed dataclasses — never raw `dict` passed between layers.

### 4. Graceful Degradation
The system runs at four capability levels:
- **Hardware + trained models** → full real-time prediction with uncertainty
- **Hardware, no models** → real-time acquisition + heuristic concentration estimate
- **No hardware, data files** → full batch analysis and training
- **No hardware, no data** → simulation mode for development/testing

### 5. The Sensor is the Ground Truth
Model predictions are guidance, not measurement. The physical sensor reading (peak shift)
is always logged alongside any model estimate. Discrepancies > 3σ trigger a quality alert.

---

## Scope Boundaries

**In scope:**
- VOC gases measurable by LSPR sensor (ethanol, IPA, methanol, mixed VOCs)
- Spectral data from CCS200 or any CSV with `wavelength` + `intensity` columns
- Offline batch analysis and online real-time inference
- Single-sensor deployment

**Out of scope (explicitly):**
- Multi-sensor array fusion (future work)
- Sub-ppb detection (requires different sensor chemistry)
- Non-optical sensing modalities
- Production IoT deployment (this is a research platform)

---

## Glossary

| Term | Definition |
|------|------------|
| LSPR | Localized Surface Plasmon Resonance — optical phenomenon in Au nanoparticles |
| Sensor material | Application-specific — platform is sensor-agnostic |
| Δλ | Peak wavelength shift (nm) — the primary physical signal |
| ROI | Region of Interest — spectral window used for shift calculation |
| CCS200 | ThorLabs CCS200 spectrometer — 3648-point, 400-700 nm, USB |
| Plateau | Stable region of a concentration step response (typically last 10 frames) |
| ALS | Asymmetric Least Squares — baseline correction algorithm |
| GPR | Gaussian Process Regression — probabilistic calibration model |
| SNR | Signal-to-Noise Ratio — quality metric for each spectrum |
| LOD | Limit of Detection — minimum detectable concentration (3σ/slope) |
| MLflow | Open-source ML experiment tracking (self-hosted) |
