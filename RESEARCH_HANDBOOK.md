# SpectraAgent Research Handbook

**Living document — update after each new finding**
_Chulalongkorn University LSPR Chemical Gas Sensing Lab_

---

## Purpose

This handbook is the scientific backbone of the SpectraAgent platform.
It explains _why_ every algorithmic choice was made, what the outputs
mean physically, how to interpret sensor health metrics, and what experiments
are needed to publish the work. Read this before running any calibration session.

Document map:

- `RESEARCH_HANDBOOK.md`: scientific policy, validation criteria, novelty claims.
- `docs/RESEARCHER_USER_GUIDE.md`: operating instructions and day-to-day workflows.

---

## 1. Sensor Physics — What We Are Actually Measuring

### 1.1 The Primary Signal: Δλ (Peak Wavelength Shift)

The CCS200 spectrometer measures the LSPR (Localized Surface Plasmon Resonance)
extinction spectrum of the sensor chip. The resonance peak position λ₀ is set
by the local refractive index around the nanostructure:

```text
Δλ = m · Δn · (1 − exp(−2d/l_d))
```

where:

- **m** = bulk refractive index sensitivity (nm/RIU), a material property
- **Δn** = refractive index change caused by the adsorbed analyte layer
- **d** = thickness of the adsorbed layer (nm)
- **l_d** = characteristic plasmon evanescent field decay length (nm)

When analyte molecules adsorb onto the sensor surface, they increase the
local refractive index → the resonance shifts. **Δλ is NOT the concentration
directly. It is the optical transduction of molecular adsorption.**

**Sign convention used in this codebase:**

- Δλ < 0 (blue-shift): analyte lowers local RI (unusual; possible for volatile
  organics replacing air in a porous MIP)
- Δλ > 0 (red-shift): analyte increases local RI (most common)
- The sign in your calibration slope (`calibration_slope` in config or fitted
  model metadata) encodes this — **confirm it experimentally for your specific
  sensor before training any model.**

### 1.2 The 6-Feature Orthogonal Basis (implemented in `src/features/lspr_features.py`)

Each feature captures a physically independent aspect of the analyte–surface
interaction. Together they form a 6-dimensional fingerprint:

| Feature          | Symbol         | Physical Origin                     | What it discriminates                        |
| ---------------- | -------------- | ----------------------------------- | -------------------------------------------- |
| Peak shift       | **Δλ**         | Local RI change (polarizability)    | Analyte concentration, RI                    |
| FWHM change      | **ΔFWHM**      | Plasmon dephasing rate (damping)    | Molecular size, mass, scattering             |
| Amplitude change | **ΔA**         | Oscillator coupling strength        | Surface coverage density                     |
| Integrated area  | **ΔI_area**    | Combined RI + scattering extinction | Total optical cross-section change           |
| Spectral std     | **ΔI_std**     | Binding heterogeneity / noise       | Measurement quality, multi-site binding      |
| Asymmetry        | **Δasymmetry** | Peak shape directionality           | Binding mechanism (physi- vs. chemisorption) |

**Why these 6 and not just Δλ?**
Different gases can produce the same Δλ but different ΔFWHM and Δasymmetry.
Ethanol and methanol, for example, have similar RI contributions per ppm but
different molecular masses (binding kinetics differ → different ΔFWHM).
The asymmetry feature discriminates whether the interaction is predominantly
physisorption (symmetric broadening) or chemisorption (asymmetric red-tail
growth). The 6D fingerprint enables **single-sensor multi-gas discrimination**
without a sensor array.

### 1.3 The Reference Spectrum

Every measurement is relative to a **reference spectrum** captured in clean
carrier gas (typically dry air or N₂). The reference defines the zero-shift
baseline. Reference quality directly sets the noise floor:

```text
σ_Δλ ≈ √(σ_gas² + σ_ref²)   [from Lorentzian fit covariances]
LOD = 3σ_Δλ / |m|            [IUPAC 2012]
```

A noisy reference means a high LOD regardless of analyte concentration.
**Always capture the reference after at least 10 minutes of sensor equilibration.**

---

## 2. Performance Metrics — What to Track and Why

### 2.1 The 7 Mandatory Metrics for Publication

Monitoring LOD alone is **not sufficient** for sensor science publications.
Every peer-reviewed journal requires at least:

| #   | Metric                  | Symbol | Definition                 | Threshold                               |
| --- | ----------------------- | ------ | -------------------------- | --------------------------------------- |
| 1   | Limit of Detection      | LOD    | 3σ_blank / m (IUPAC)       | Must include 95% bootstrap CI           |
| 2   | Limit of Quantification | LOQ    | 10σ_blank / m              | Must be ≥ 3.3× LOD                      |
| 3   | Sensitivity             | m      | Calibration slope (nm/ppm) | Report ± std over N sessions            |
| 4   | Calibration R²          | R²     | 1 − SS_res/SS_tot          | Must be >0.9954 for linearity claim     |
| 5   | Dynamic Range           | LDR    | [LOQ, C_sat] (ppm)         | Width ≥ 2 orders of magnitude preferred |
| 6   | Reproducibility         | RSD%   | σ/μ × 100 at C_target      | <5% (intra-day), <10% (inter-day)       |
| 7   | Selectivity             | K_B,A  | Response(B) / Response(A)  | <0.05 for claimed specificity           |

**Why LOD alone is insufficient:**
A sensor with surface fouling shows:

1. FWHM broadening → **early warning** (1–3 sessions before LOD degrades)
2. Sensitivity decrease → **mid warning** (same session as FWHM starts broadening)
3. LOD increase → **late warning** (surface already significantly fouled)

SensorHealthAgent monitors all 7 dimensions and alerts at the earliest sign.

### 2.2 The SensorHealthAgent 5-Metric Scorecard

Each dimension is scored 0–100 based on the sensor's own history:

```text
LOD score       = min(best_ever_LOD / current_LOD, 1.0) × 100
Sensitivity score = min(current_|sensitivity| / best_ever_|sensitivity|, 1.0) × 100
R² score        = max(0, (R² - 0.90) / 0.10) × 100
Drift score     = min(typical_drift / current_drift, 1.0) × 100
SNR score       = max(0, (SNR - 3) / 27) × 100

Overall health  = LOD×30% + Sensitivity×30% + R²×25% + Drift×10% + SNR×5%
```

**Recalibration is required when:**

- LOD score < 67 (LOD has degraded >50% from best)
- Sensitivity score < 70 (sensitivity dropped >30%)
- R² < 0.95 for 2+ consecutive sessions
- Overall health < 55

### 2.3 Noise Equivalent Concentration (NEC)

The fundamental lower bound on detection, independent of calibration model:

```text
NEC = σ_blank_nm / |sensitivity_nm_per_ppm|
```

NEC is the concentration at which the signal equals one standard deviation of
the blank noise. LOD = 3 × NEC. When NEC is stable but LOD increases,
the sensitivity is degrading, not the noise floor. This decomposition is
critical for diagnosing whether to re-clean the surface or re-collect the
reference spectrum.

---

## 3. ICH Q2(R1) Validation Roadmap

The CalibrationValidationOrchestrator tracks your progress through ICH Q2(R1)
automatically. Here is what each test requires experimentally:

### 3.1 §4.1 Specificity

**What to do:** Expose the sensor to each known interferent at 10× its OSHA-PEL
concentration and to the target analyte at LOQ. Calculate:

```text
selectivity coefficient K = Δλ(interferent at C_B) / Δλ(analyte at C_A)
```

**Pass criterion:** K < 0.05 for all interferents, OR document which interferents
affect the sensor and at what concentration.

### 3.2 §4.2 Linearity

**What to do:** Measure at ≥5 concentration levels spanning the entire claimed
range (C_min to C_max). Perform linear regression and record R².
**Pass criterion:** R² > 0.9954, residuals show no systematic pattern.

### 3.3 §4.3 Range

**What to do:** Demonstrate that the method gives acceptable precision, accuracy,
and linearity at the extremes of the claimed range. Min = LOQ, Max = C_sat / 2.
**Pass criterion:** R² > 0.99 at both extremes simultaneously with ≥3 replicates.

### 3.4 §4.4 Precision

**Repeatability (intra-day):** ≥6 measurements at one concentration level within
one day by one analyst. Report RSD%.
**Intermediate precision (inter-day):** Same measurement on ≥3 different days.

**Pass criteria:** RSD < 2% (repeatability), < 3% (intermediate precision).

### 3.5 §4.5 Accuracy (Recovery)

**What to do:** Prepare reference standards at 3 concentration levels (80%,
100%, 120% of target) independently of the calibration standards. Calculate:

```text
Recovery% = (measured − blank) / (spiked − blank) × 100
```

**Pass criterion:** Recovery = 98–102% at each level.

### 3.6 §4.6/4.7 LOD and LOQ

**What to do:** Already automated by `SessionAnalyzer`. Requires blank measurements
(carrier gas only) interleaved with calibration points.
**Pass criterion:** LOD and LOQ must be reported with 95% bootstrap CI (already computed).

### 3.7 §4.8 Robustness

**What to do:** Deliberately vary one parameter at a time by ±10%:

- Integration time ±10%
- Temperature ±5°C (log room temperature each session)
- Reference age (1 h, 4 h, 8 h after capture)
- Flow rate ±10% (if using flow cell)

For each variation, measure LOD and R². Report the change relative to nominal.
**Pass criterion:** LOD changes < 20%, R² remains > 0.99.

---

## 4. Calibration Workflow — Step by Step

### 4.1 Pre-Session Checklist

- [ ] Chip is freshly functionalized (or document the functionalization age)
- [ ] Carrier gas is flowing for ≥10 minutes before reference capture
- [ ] Temperature logged (affects LSPR sensitivity by ~0.02 nm/°C)
- [ ] Integration time set consistently with previous sessions
- [ ] Reference spectrum captured and baseline stable (drift < 0.05 nm/min)

### 4.2 Concentration Schedule (Optimal for ICH Q2(R1))

For maximum information per experiment, use logarithmically spaced concentrations:

```text
Suggested levels: LOQ/2, LOQ, 2×LOQ, 5×LOQ, 10×LOQ, C_max
Replicates per level: ≥3 (for precision) or ≥6 (for repeatability claim)
```

The ExperimentPlannerAgent (Bayesian design) will suggest the optimal next
concentration if auto_explain is enabled.

### 4.3 Session-End Actions (automated by SpectraAgent)

At `/api/acquisition/stop`, the platform automatically:

1. Runs `SessionAnalyzer`: computes LOD, LOQ, LOB, R², RMSE, drift
2. Records outcomes to `SensorMemory` (`output/memory/sensor_memory.json`)
3. `SensorHealthAgent` scores the session against historical performance
4. `CalibrationValidationOrchestrator` updates ICH Q2(R1) tracker and suggests next test
5. `ExperimentNarrator` narrates the calibration result (if auto_explain=True)

---

## 5. Model Selection Guide

The `CalibrationAgent` tests four models and selects by corrected AIC (AICc):

| Model            | Best when                                  | Physical interpretation                  |
| ---------------- | ------------------------------------------ | ---------------------------------------- |
| Linear           | Small Δλ range (<2 nm), Henry's law regime | Ideal dilute solution                    |
| Langmuir         | Saturation visible at high concentration   | Single-site monolayer adsorption         |
| Freundlich       | Log-linear response, no saturation         | Heterogeneous multi-site adsorption      |
| GPR (Matérn 5/2) | Any                                        | Non-parametric; use when physics unknown |

**For publication:** Always report which model was selected, its AICc score,
and the evidence ratio vs. the second-best model (ΔAICc > 2 = substantial evidence).

---

## 6. Sensor Aging and Recalibration Policy

### 6.1 Signs That Recalibration Is Needed

| Observation                                             | Likely cause                               | Action                                                  |
| ------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------- |
| FWHM broadening (ΔFWHM trending positive over sessions) | Surface fouling; non-specific adsorption   | Clean chip with EtOH/IPA, re-functionalize              |
| Sensitivity decrease (slope \|m\| shrinking)            | Active site blockage or MIP collapse       | Re-functionalize; consider fresh chip                   |
| LOD increase without sensitivity change                 | Reference noise floor has grown            | Recapture reference; check light source stability       |
| R² below 0.95                                           | Calibration curve nonlinearity or outliers | Check concentration preparation; repeat linearity range |
| Baseline drift > 0.1 nm/min                             | Temperature instability or outgassing      | Extend equilibration time; control lab temperature      |

### 6.2 Recalibration vs. Re-functionalization vs. New Chip

```text
SensorHealthAgent overall score < 55  → Recalibrate (new session, same chip)
Sensitivity score < 50 + FWHM trend degrading  → Re-functionalize chip
3+ consecutive recalibration events with no improvement  → New chip
```

### 6.3 Chip Lifetime Estimation

Track `reference_peak_nm` across sessions in `SensorMemory`. A red-shift of
the reference peak (in clean carrier gas) over multiple sessions indicates
nanostructure or surface chemistry degradation — this is a chip-level signal,
not a measurement artefact.

---

## 7. Publication Checklist

When you are ready to submit, verify all of the following:

### Required Data (ICH Q2(R1) / Eurachem)

- [ ] LOD with 95% bootstrap CI (N_bootstrap ≥ 2000)
- [ ] LOQ with 95% bootstrap CI
- [ ] LOB (Limit of Blank) from dedicated blank measurements
- [ ] Sensitivity m (nm/ppm) with 95% CI from regression
- [ ] R² and residual plot (Supplementary)
- [ ] Repeatability RSD% (≥6 replicates, one day)
- [ ] Intermediate precision RSD% (≥3 days)
- [ ] Recovery% at 80%, 100%, 120% of target concentration
- [ ] Selectivity coefficients for major interferents
- [ ] Linearity range [LOQ, C_sat/2] with confidence band plot
- [ ] Robustness: effect of ±10% variation in 3+ parameters

### Required Metadata (Reproducibility)

- [ ] Hardware model and serial number (ThorLabs CCS200 + chip serial)
- [ ] Integration time, averaging, scan rate
- [ ] Functionalization protocol (batch number, date, operator)
- [ ] Room temperature and humidity (log every session)
- [ ] Git commit hash of the analysis software used
- [ ] Python version and key library versions (numpy, scipy, sklearn)
- [ ] Raw data available in repository (`output/sessions/*/`)

### What Makes This Work Novel

The core novelty claim for publication is:

1. **Data-driven adaptive agents**: The AI agents reason from the sensor's
   _actual experimental history_ stored in `SensorMemory`, not from hardcoded
   literature values. Each session makes the agents smarter about _this specific
   sensor_, not sensors of this type in general.

2. **6-feature physically orthogonal basis**: The Δasymmetry feature enables
   binding mechanism discrimination (physisorption vs. chemisorption) from a
   single spectrometer without additional sensors.

3. **ICH Q2(R1) automation**: No other open-source optical sensor platform
   automates regulatory validation state tracking and experiment design.

4. **Cross-session sensor health monitoring**: The multi-metric scorecard
   provides early warning of surface degradation 1–3 sessions before LOD
   is affected.

5. **Generic sensor platform**: Works with any spectrometer sensor whose
   response is a wavelength shift (LSPR, SPR, fluorescence shift). Demonstrated
   on ThorLabs CCS200 with LSPR chips used in the Chulalongkorn workflow.

---

## 8. Current Platform Status (updated 2026-04-01)

The platform has completed all Part A (production), Part B (science gaps), and Phase 5C
(data-driven science layer) items.

Test totals evolve continuously. Treat CI and local test runs as authoritative
status instead of fixed numbers in this document.

### Phase 5C Additions

| Feature                                          | Location                             | Status  |
| ------------------------------------------------ | ------------------------------------ | ------- |
| Data-Driven Science tab (Tab 5)                  | `dashboard/science_tab.py`           | ✅ Done |
| ModelVersionStore (timestamped, git-hash IDs)    | `src/models/versioning.py`           | ✅ Done |
| Multi-task CNN (class + regr + QC heads)         | `src/models/multi_task.py`           | ✅ Done |
| Domain adaptation (GRL transfer learning)        | `src/models/transfer.py`             | ✅ Done |
| Feature attribution (SHAP, integrated gradients) | `src/analysis/feature_importance.py` | ✅ Done |
| Leave-one-config-out benchmark                   | `src/analysis/cross_dataset_eval.py` | ✅ Done |

### Known Workflow Gaps — see `REMAINING_WORK.md` Part C

| #   | Gap                                                          | Priority | Status                                                         |
| --- | ------------------------------------------------------------ | -------- | -------------------------------------------------------------- |
| C1  | Session CSV → Tab 5 spectral dataset bridge                  | HIGH     | ✅ Done — `load_session_csv()` in `src/io/universal_loader.py` |
| C2  | "Promote to live system" button for trained model            | MEDIUM   | ✅ Done — Tab 5 Model Training                                 |
| C3  | Multi-config data collection guidance for single-sensor labs | MEDIUM   | ✅ Done — expander in Tab 5 Cross-Dataset + §8.1 below         |
| C5  | ONNX export button in Tab 5                                  | LOW      | Open                                                           |
| C6  | MLflow wiring into Tab 5 training                            | LOW      | Open                                                           |

### 8.1 Generating Multi-Config Data for the Cross-Dataset Benchmark

The cross-dataset benchmark (Tab 5 → Cross-Dataset Analysis) requires data from
**≥ 3 distinct sensor configurations** to produce a publishable leave-one-config-out
accuracy score. If you only have one CCS200 spectrometer, use one of these approaches:

#### Option A — Robustness sweep (automated)

```bash
spectraagent robustness --param integration_time --range 45:55 --steps 3 --gas Ethanol
```

This sweeps integration time from 45 → 55 ms in 3 steps (standard ICH §4.8 robustness
range), runs a full calibration session at each setting, and saves results to
`output/robustness/`. Load each sub-directory as a separate config in Tab 5.

#### Option B — Manual conditions

Collect calibration sessions under these three conditions:

| Config   | Change                                                                              |
| -------- | ----------------------------------------------------------------------------------- |
| Config A | Nominal (50 ms integration, fresh reference)                                        |
| Config B | Integration time 100 ms (×2 slower, lower noise floor)                              |
| Config C | Reference recaptured after 4 h (aged baseline, tests robustness to reference drift) |

**Why this matters for publication:** The benchmark answer — _"Does the model
generalise to an unseen sensor configuration without retraining?"_ — is the core
novelty claim. An accuracy > 90% on the held-out config, reported with bootstrap CI,
is the sentence that earns the paper its journal acceptance.

---

## 9. Open Scientific Questions (Future Work)

These gaps represent the boundary of current knowledge and the most promising
directions for follow-on publication:

| Gap                                            | Why it matters                                                                   | Effort                  | Status                                                            |
| ---------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------- |
| Kinetic features (τ₆₃, τ₉₅)                    | Binding rate constant discriminates analytes better than steady-state Δλ         | Medium                  | ✅ Implemented — `src/features/lspr_features.py`                  |
| Reference FWHM as chip age predictor           | FWHM(reference, clean gas) tracks nanostructure degradation across chip lifetime | Low                     | ✅ Implemented — `reference_fwhm_nm` in CalibrationObservation    |
| Transfer learning across analytes              | Train on Ethanol → fine-tune on Methanol with 5 calibration points               | High                    | ✅ Implemented — `src/models/transfer.py` (GRL domain adaptation) |
| Cross-lab validation                           | Run same protocol at ≥2 Chulalongkorn labs, compare LOD/sensitivity              | Low effort, high impact | Open — requires data collection                                   |
| Multi-analyte mixture discrimination           | Binary/ternary gas mixtures                                                      | High                    | Open — future paper                                               |
| In-situ drift correction using temperature     | Δλ_corrected = Δλ_raw − α·ΔT                                                     | Medium                  | Open                                                              |
| Conformal prediction for non-exchangeable data | Current conformal PI assumes exchangeability; time-series violates this          | High                    | Open                                                              |

---

## 10. Direction Lock (April 2026)

This section prevents scope drift and keeps the project aligned with the publication mission.

### 10.1 What counts as "on-direction" work

Work is on-direction when it strengthens at least one of these pillars:

1. **Scientific validity**: better calibration quality, uncertainty quantification, selectivity evidence, and reproducibility statistics.
2. **Research integrity**: tamper-evident artifacts, replay verification, provenance, and auditable CI checks.
3. **Operational reliability**: stable deployment, secure defaults, deterministic session outputs, and documented runbooks.
4. **Publishability**: clearer ICH Q2(R1) completion path, stronger figures/tables, and reproducible methods text.

### 10.2 Scope policy for novel modules

Large additions (new simulation/modeling pipelines) are encouraged, but must be staged:

1. **Milestone A (production and integrity)** must remain green.
2. **Milestone B (new science module)** should be merged behind clear boundaries:
   - explicit tests,
   - clear API surface,
   - no regressions to acquisition/deployment workflows.
3. Novel modules should not change live deployment behavior by default until validated on real lab data.

### 10.3 Groundbreaking novelty statement (current)

The strongest novelty claim remains the integration of:

1. Physics-grounded spectral features,
2. session-aware AI scientific agents,
3. automated ICH Q2(R1) progress tracking,
4. and executable integrity controls (manifest checksums + replay verification + CI gate).

This combined stack is the central research contribution; individual model additions should reinforce this claim, not fragment it.

### 10.4 Handbook update protocol (mandatory)

Update this handbook whenever any of the following changes:

1. Calibration math, thresholds, or acceptance criteria,
2. integrity workflow or release gates,
3. live acquisition behavior,
4. researcher-facing UI workflow,
5. novelty claim wording used in drafts/papers.

For every update, include:

1. Date,
2. changed file(s),
3. scientific reason,
4. impact on ongoing experiments.

Use this compact changelog format:

```text
[YYYY-MM-DD] <topic>
- Files: <paths>
- Reason: <scientific or integrity rationale>
- Research impact: <what researchers must do differently>
```

---

_This handbook is updated automatically when `RESEARCH_HANDBOOK.md` is edited.
The authoritative source of truth for all numerical thresholds is the codebase —
if a number in this document conflicts with the code, trust the code._
