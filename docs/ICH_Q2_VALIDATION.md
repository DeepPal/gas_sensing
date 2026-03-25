# ICH Q2(R1) Analytical Method Validation — Au-MIP LSPR Gas Sensor

**Standard:** ICH Q2(R1) *Validation of Analytical Procedures*, 2005
**Sensor:** Gold Molecularly Imprinted Polymer (Au-MIP) LSPR fiber sensor
**Analytes:** Ethanol (EtOH), Isopropanol (IPA), Methanol (MeOH), Mixed VOC
**Matrix:** Gas phase (ambient air)
**Signal:** Peak wavelength shift Δλ (nm), measured by CCS200 spectrometer

---

## 1. Validation Parameters (ICH Q2(R1) §4)

### 1.1 Specificity / Selectivity

**Definition:** Ability to assess unequivocally the analyte in the presence of
components that may be expected to be present (other VOCs, humidity, CO₂).

| Test | Method | Status |
|------|--------|--------|
| Cross-gas interference | LOGO cross-validation (CNN classifier) | Implemented (`cross_gas_eval.py`) |
| Humidity interference | Environmental compensation model | Defined in `config.yaml:environment` |
| Temperature drift | Theil-Sen drift monitoring | Implemented (`performance_monitor.py`) |
| Selectivity matrix | Response to each analyte vs. interferents | **TODO: experimental** |

**CNN selectivity:** Leave-one-gas-out accuracy > X% is required. Run
`python -m src.training.cross_gas_eval --data-dir Joy_Data` for current values.

---

### 1.2 Linearity

**Definition:** Ability to obtain test results directly proportional to
concentration over a given range.

**Method:** OLS linear regression of Δλ (nm) vs. concentration (ppm).

| Gas | Expected R² | Linear range | Notes |
|-----|-------------|--------------|-------|
| Ethanol | ≥ 0.95 | 0.1–10 ppm | Validated range |
| IPA | ≥ 0.95 | 0.1–10 ppm | Validate |
| Methanol | ≥ 0.95 | 0.1–10 ppm | Validate |
| Mixed VOC | ≥ 0.90 | 0.5–5 ppm | May require Langmuir at high conc |

**Linearity test:** Mandel's fitting test (F-test comparing linear vs.
quadratic fit) should be applied. If F-statistic significant at p < 0.05,
consider nonlinear calibration.

**Code:** `src/scientific/lod.py::calculate_sensitivity()` returns slope ± SE.

---

### 1.3 Range

**Definition:** Interval between upper and lower concentration levels within
which the analytical procedure has suitable precision, accuracy, and linearity.

| Gas | Minimum validated range | Maximum validated range |
|-----|------------------------|------------------------|
| Ethanol | 0.1 ppm | 10 ppm |
| IPA | 0.1 ppm | 10 ppm |
| Methanol | 0.1 ppm | 10 ppm |
| Mixed VOC | 0.5 ppm | 5 ppm (each component) |

The validated range must encompass the LOQ at the lower end and the upper
linearity limit at the upper end.

---

### 1.4 Limit of Detection (LOD)

**Method:** ICH Q2(R1) §5.2, approach based on standard deviation of response
and slope of calibration curve:

```
LOD = 3.3 × σ_noise / |S|
```

where:
- σ_noise = std of blank signal in noise region [690–720 nm] (see `config.yaml`)
- S = calibration slope (nm/ppm) from OLS regression
- Factor 3.3 ≈ 3σ (IUPAC 1995, Pure Appl. Chem. 67(10):1699)

**Uncertainty:** 95% bootstrap CI (n=1000 resamples) per ICH Q2(R2) Appendix B.

**Code:** `src/scientific/lod.py::lod_bootstrap_ci()`

**Current estimates** (run `python -m src.training.train_gpr --gas all --data Joy_Data`):

| Gas | LOD (ppm) | 95% CI |
|-----|-----------|--------|
| Ethanol | — | — |
| IPA | — | — |
| Methanol | — | — |

*Fill in from MLflow run after training.*

---

### 1.5 Limit of Quantification (LOQ)

**Method:** ICH Q2(R1) §5.3:

```
LOQ = 10 × σ_noise / |S|
```

**Uncertainty:** Bootstrap 95% CI, same resamples as LOD.

**Code:** `src/scientific/lod.py::calculate_loq_10sigma()` + CI from `sensor_performance_summary()`

---

### 1.6 Accuracy / Trueness

**Definition:** Closeness of agreement between the true value and the measured
value. Expressed as percent recovery or RMSE (ppm).

**Method:** Predict concentration on LOOCV held-out samples; compare to
gravimetrically prepared concentrations.

| Acceptable criteria | Value |
|--------------------|-------|
| RMSE | ≤ 15% of target concentration |
| R² (LOOCV) | ≥ 0.90 |
| Mean bias | ≤ ±10% at any calibration level |

**Code:** LOOCV R² and RMSE from `src/training/train_gpr.py` logged to MLflow.

---

### 1.7 Precision

**Definition:** Closeness of agreement among a series of measurements under
prescribed conditions. Assessed at three levels:

#### Repeatability (Intra-day, same operator, same instrument)

| Level | n replicates | Acceptable RSD |
|-------|-------------|----------------|
| Concentration 0.5 ppm | ≥ 3 | ≤ 5% |
| Concentration 2.0 ppm | ≥ 3 | ≤ 5% |
| Concentration 5.0 ppm | ≥ 3 | ≤ 5% |

**Code:** `src/scientific/lod.py::sensor_performance_summary()` reports noise_std.
RSD = (noise_std / mean_response) × 100%.

#### Intermediate Precision (Inter-day)

Repeat measurements on 3 different days. Report RSD across days.

| Acceptable RSD | ≤ 10% |
|---------------|-------|

#### Reproducibility

Not required for a method used in a single laboratory. Required if submitted
to regulatory agency for multi-site use.

---

### 1.8 Robustness

**Definition:** Measure of capacity to remain unaffected by small, deliberate
variations in method parameters.

**Ablation study** (`src/training/ablation.py`) quantifies robustness to
preprocessing choices:

| Parameter varied | ΔR² acceptable | Method |
|-----------------|----------------|--------|
| ALS λ (±1 order magnitude) | ≤ 0.05 | Ablation: `no_baseline` |
| SGF window (5 vs 11 vs 21) | ≤ 0.03 | Ablation: `no_smoothing` |
| SNV normalization on/off | ≤ 0.05 | Ablation: `no_normalization` |
| ROI center ±5 nm | ≤ 0.05 | Grid scan in `config.yaml:roi.shift.window_nm` |

**Run:** `python -m src.training.ablation --data-dir Joy_Data/Ethanol`

---

## 2. System Suitability Tests (SST)

Before each measurement session, verify:

| Test | Criterion | Frequency |
|------|-----------|-----------|
| Dark current stability | Max dark counts < 0.01 AU | Each session |
| Reference spectrum reproducibility | RSD < 0.1% at peak wavelength | Each session |
| Peak wavelength stability | Drift < 0.1 nm from last session | Each session |
| SNR of reference | SNR > 50 in [520–560 nm] | Each session |

**Code:** `gas_analysis/core/performance_monitor.py` logs all drift/quality metrics.

---

## 3. Calibration Model Selection Criteria

| Criterion | GPR | Linear OLS |
|-----------|-----|-----------|
| n calibration points | 4–20 | ≥ 5 |
| Preferred for | Uncertainty propagation, nonlinear response | Publication-ready equation |
| LOD method | GPR-aware (grid search) | ICH Q2(R1) 3.3σ/S |
| Cross-validation | LOOCV | LOOCV |
| When to use | Real-time concentration prediction | Calibration curve figure |

---

## 4. Documentation Requirements

Each MLflow training run automatically logs:
- `r2`, `rmse_ppm` — model accuracy
- `lod_ppm`, `lod_ppm_ci_lower`, `lod_ppm_ci_upper` — ICH Q2(R1) LOD
- `loq_ppm` — ICH Q2(R1) LOQ
- `calibration_slope`, `calibration_slope_se` — sensitivity ± uncertainty
- `noise_std` — blank noise used for LOD calculation
- `cv_strategy` — LOOCV or k-fold flag

To view: `mlflow ui --backend-store-uri experiments/mlruns`

---

## 5. References

1. ICH Q2(R1). *Validation of Analytical Procedures: Text and Methodology.* 2005.
2. IUPAC (1995). *Nomenclature in Evaluation of Analytical Methods.* Pure Appl. Chem. 67(10):1699–1723.
3. ISO 11843-3 (2003). *Capability of detection. Part 3: Methodology for determination of the critical value.* ISO.
4. Eilers & Boelens (2005). *Baseline correction with asymmetric least squares smoothing.* Leiden University.
5. Sen (1968). *Estimates of the regression coefficient based on Kendall's tau.* JASA 63(324):1379–1389.
6. Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.* MIT Press.
7. Ebrahimi et al. (2025). *SERS biosensing with AI/ML.* Adv. Sensor Res. [doi:10.1002/adsr.202400155]
