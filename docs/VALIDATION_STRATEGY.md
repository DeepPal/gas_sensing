# Validation Strategy — Au-MIP LSPR Gas Sensing Pipeline

**Version:** 1.0
**Date:** 2026-03
**Standard:** ICH Q2(R1) / IUPAC 1995 / ISO 11843

---

## 1. Overview

This document describes the train/validation/test split strategy for all
models in the Au-MIP LSPR gas sensing pipeline. It must be cited in any
paper or report that uses metrics from this codebase.

---

## 2. Dataset Structure (Joy-Data Layout)

```
Joy_Data/
  Ethanol/
    Multi mix vary-EtOH-{conc}-{run}/   ← conc ∈ {0.1, 0.5, 1, 5, 10} ppm
      *.csv                              ← ~60 frames per trial at 2.4 Hz
  Multi mix vary-IPA/
    ...
  Multi mix vary-MeOH/
    ...
  Mixed gas/
    {conc} ppm EtOH IPA MeOH-{run}/
    ...
```

**Plateau extraction:** The last 10 CSVs of each trial are averaged as the
steady-state representative spectrum. This eliminates transient adsorption
response and reduces within-trial noise.

---

## 3. Model-Specific Validation Strategies

### 3.1 GPR Calibration Model (`src/training/train_gpr.py`)

| Property | Value |
|----------|-------|
| CV method | **Leave-One-Out CV (LOOCV)** when n < 20; k-fold (k=5) otherwise |
| n_typical | 4–5 concentration levels × 1 averaged spectrum = 4–5 points |
| Stratification | Not applicable (1 point per concentration) |
| Reported metric | CV R² (mean ± std), RMSE on full training set |
| LOD method | GPR-aware: find c where E[ŷ(c)] = 3 × Std[ŷ(c)] via grid search |
| LOD CI | Bootstrap 95% CI (n=1000 resamples) per ICH Q2(R1) Appendix |

**Why LOOCV?** With n=5 samples and 5-fold CV, each fold has 4 train / 1 test
— identical to LOOCV but with random split variance added. LOOCV is the
canonical method for n < 20 in analytical chemistry (see ISO 11843-3).

**Known limitation:** Full-fit R² = 1.0 for n=5 is expected GPR overfitting.
Do not report training R²; report only LOOCV R².

### 3.2 CNN Gas Classifier (`src/training/train_cnn.py`)

| Property | Value |
|----------|-------|
| CV method | **Leave-One-Gas-Out (LOGO)** cross-validation |
| Train set | All spectra from all gases *except* the held-out gas |
| Test set | All spectra of the held-out gas |
| Reported metrics | Per-gas: classification accuracy (%), concentration RMSE (ppm), MC dropout uncertainty (ppm) |
| Overall metrics | Macro accuracy (%), overall RMSE, R² |
| MC Dropout | 30 forward passes with Dropout(p=0.3) active for uncertainty |

**Why LOGO?** LOGO measures true generalisation — can the model detect a gas
it has never seen during training? This is the operationally relevant question
for a deployable gas sensor (e.g. detecting an unknown VOC).

**Temporal leakage warning:** Do not mix frames from the same trial across
train/test boundaries. Each trial (one folder) should be treated as one unit.
If using individual frames: all frames from trial `{gas}-{conc}-{run}` must
be entirely in train OR entirely in test.

### 3.3 Preprocessing Ablation (`src/training/ablation.py`)

| Property | Value |
|----------|-------|
| CV method | 5-fold CV within the single gas dataset |
| Purpose | Quantify ΔR² when each preprocessing step is removed |
| Baseline config | `all_on` (ALS + Savitzky-Golay + SNV) |
| Ablation configs | See `ABLATION_CONFIGS` in `ablation.py` |
| Reported metric | ΔR² and ΔRMSE vs. `all_on` baseline |

---

## 4. Temporal Autocorrelation Considerations

Consecutive frames within a trial are temporally correlated (sensor drift,
adsorption kinetics). This creates **data leakage** if adjacent frames end up
in different folds.

**Mitigation:**
- Train on averaged plateau spectra (last 10 frames per trial) — reduces temporal dependency to one representative point per trial.
- For frame-level evaluation: enforce a gap of ≥ 5 frames between any train and test frame from the same trial.
- LOGO cross-validation eliminates within-gas temporal leakage by design.

---

## 5. External Validation

The current validation is **internal** (same instrument, same measurement
session). For journal publication, the following external validations are
recommended:

1. **Cross-instrument**: Validate trained models on spectra from a second
   CCS200 unit or Ocean Optics equivalent.
2. **Cross-session**: Validate across measurement sessions separated by ≥ 24 h
   (captures thermal drift and sensor aging).
3. **Cross-operator**: Validate with samples prepared by a different operator
   to assess preparation-related variability.

---

## 6. Performance Reporting Requirements (ICH Q2(R1))

All performance metrics reported in papers must include:

| Metric | Required | Units | Method |
|--------|----------|-------|--------|
| LOD | Yes | ppm | ICH Q2(R1): 3.3σ/S + bootstrap 95% CI |
| LOQ | Yes | ppm | ICH Q2(R1): 10σ/S + bootstrap 95% CI |
| Sensitivity (slope) | Yes | nm/ppm | OLS ± SE |
| R² | Yes | — | LOOCV R² (not training R²) |
| RMSE | Yes | ppm | LOOCV or LOGO |
| Linearity range | Yes | ppm | Concentration range where R² ≥ 0.99 |
| Repeatability | Yes | % RSD | Triplicate measurements at one concentration |
| SNR definition | Yes | — | `\|Δλ\| / σ_noise` (see config.yaml) |

---

## 7. Reproducibility Checklist

Before submitting to a journal, verify:

- [ ] Random seeds set in `pyproject.toml` (`random_state = 42` in all models)
- [ ] MLflow run ID recorded for all training runs (`experiments/mlruns/`)
- [ ] Dataset SHA256 logged to MLflow (via `ExperimentTracker.log_dataset_info`)
- [ ] `config.yaml` committed with exact ALS λ, p, SGF window, ROI bounds
- [ ] LOOCV R² used in paper (not training R²)
- [ ] Bootstrap CI reported alongside LOD/LOQ point estimates
- [ ] Temporal leakage documented in Methods section
