# Calibration Workflow

Step-by-step guide for performing a full sensor calibration session with SpectraAgent.
This covers both the **live acquisition path** (React + FastAPI) and the
**research dashboard path** (Streamlit Agentic Pipeline tab).

---

## Overview

A complete calibration produces:

- Sensitivity (slope, nm/ppm or signal/ppm)
- R² (linearity)
- LOB / LOD / LOQ with bootstrap 95% CI (IUPAC-compliant)
- Conformal prediction intervals for future measurements
- Isotherm model selection (Langmuir / Freundlich / Hill vs. linear)
- Session report (JSON + optional HTML)

Minimum data required: **5 calibration points** at known concentrations.
For LOL (Limit of Linearity), you need **6+ points**.

---

## Path A — Live Acquisition (SpectraAgent React UI)

### Step 1: Start the server

```bash
python -m spectraagent start
# Opens http://localhost:8765 automatically
```

Confirm the hardware badge shows **Live – CCS200** (not Simulation).
If it shows Simulation, check [CCS200 Setup](../hardware/CCS200_SETUP.md).

### Step 2: Configure acquisition

In the **Calibration** panel on the right side of the UI:

1. Set **Integration time** (default 50 ms — adjust for your light source).
2. Enter **Gas / analyte label** (e.g. `Ethanol`).
3. Click **Apply Config**.

### Step 3: Capture a blank / reference

1. Expose the sensor to **zero analyte** (clean carrier gas or ambient air).
2. Wait until the spectrum is stable (live chart shows a flat, repeatable peak).
3. Click **Capture Reference**.
   - The server stores the current spectrum as the reference.
   - All subsequent Δλ measurements are computed relative to this reference.
   - The peak wavelength shown in the UI resets to ~0 nm shift.

### Step 4: Start a session

Click **Start Session**. The server begins:
- Recording frames to `output/sessions/YYYYMMDD_HHMMSS/pipeline_results.csv`
- Accumulating events for post-session `SessionAnalyzer`

### Step 5: Run concentration series

For each calibration concentration (use **Suggest Next** to get the Bayesian
optimal next point — this minimises the number of experiments needed):

1. Set the analyte concentration in the exposure cell.
2. Wait for the signal to **stabilise** (agent event feed shows no `drift_warn`).
3. In the Calibration panel, enter the **concentration value** and click
   **Add Calibration Point**.
   - The `CalibrationAgent` records the current peak shift alongside the concentration.
   - Once you have ≥ 3 points, a preliminary GPR fit is shown.

Recommended starting grid if not using Suggest: 0.1, 0.5, 1.0, 2.0, 5.0 ppm.

### Step 6: Stop session and review

Click **Stop Session**. The server automatically:
1. Runs `SessionAnalyzer` on all accumulated frame events.
2. Computes LOB / LOD / LOQ (with bootstrap CI), T90, T10, drift rate, linearity.
3. Writes results to `output/sessions/{id}/session_meta.json`.
4. Emits `session_complete` event to the agent feed.

### Step 7: Generate report (optional)

With `ANTHROPIC_API_KEY` set, click **Generate Report** in the UI.
`ReportWriter` (Claude agent) produces a narrative summary in the agent event feed.

---

## Path B — Research Dashboard (Streamlit Agentic Pipeline tab)

For batch analysis of pre-recorded spectra from `Joy_Data/` directories.

```bash
.venv/Scripts/python.exe -m streamlit run dashboard/app.py
```

Navigate to the **Agentic Pipeline** tab.

### Step 1: Load spectra

Click **Load Joy_Data spectra**. The loader auto-parses gas name and concentration
from folder names (e.g. `Ethanol_1ppm`, `Ethanol_0.5ppm`), then averages the last
10 CSVs per group as the steady-state representative spectrum.

### Step 2: Load reference spectrum

In the **Reference** section:
- Upload a reference CSV, or
- Select a folder labelled `blank` / `zero` / `reference`, or
- Use the first loaded spectrum group as reference.

When a reference is loaded:
- `diff_signal = raw − ref_interp` is computed for every spectrum.
- All downstream features use **Δλ** (peak shift) — the physically correct LSPR signal.
- The calibration curve Y-axis switches to Δλ (nm).

### Step 3: Feature extraction

SpectraAgent automatically selects features based on reference availability:
- **With reference** → LSPR features: `[Δλ, ΔI_peak, ΔI_area, ΔI_std]`
- **Without reference** → Raw features: `[peak_int, peak_wl, area, std]`

### Step 4: GPR calibration fit

Click **Run Calibration**. The pipeline:
1. Fits `GPRCalibration` to (concentration, Δλ) pairs.
2. Runs `PhysicsInformedGPR` with Langmuir mean function.
3. Mandel F-test determines whether Langmuir or linear model is preferred.
4. Calibrates `ConformalCalibrator` on residuals — produces guaranteed CI.
5. Computes LOB / LOD / LOQ / LOL with bootstrap CI (2 000 iterations).

Results shown:
- Calibration curve with conformal CI bands
- LOD / LOQ / LOL values with 95% CI
- R², RMSE, residual plot
- Selected isotherm model

### Step 5: Selectivity (optional)

If spectra for multiple analytes are loaded, click **Selectivity Matrix**.
Computes IUPAC cross-reactivity coefficients K_{A,B} for all analyte pairs.

### Step 6: Export

- **HTML Report** — includes gas type, date, sensitivity, R², LOD/LOQ, isotherm model, instrument metadata.
- **Reproducibility Manifest** — JSON with git commit, package versions, config hash.
- **Publication Figures** — `src/reporting/publication.py` generates Nature/ACS-style plots.

---

## Understanding the LOD/LOQ Output

| Field | Definition | Source |
|---|---|---|
| `lob` | Limit of Blank — highest signal expected from blank | IUPAC 1995 |
| `lod` | Limit of Detection — 3σ/slope above LOB | IUPAC 1995 |
| `loq` | Limit of Quantification — 10σ/slope above LOB | ICH Q2(R1) |
| `lod_ci_low/high` | Bootstrap 95% CI on LOD | 2 000 iterations |
| `loq_ci_low/high` | Bootstrap 95% CI on LOQ | 2 000 iterations |
| `sigma_source` | `"blank_events"` or `"calibration_residuals"` | Audit trail |
| `lol` | Limit of Linearity — highest linear concentration | Mandel F-test |

The σ source is chosen automatically:
- If blank measurements exist → `sigma = std(blank signals)` (preferred, per IUPAC)
- Otherwise → `sigma = std(calibration residuals)`, ddof=2

Both are recorded in `session_meta.json` under `methods_audit`.

---

## Active Learning — Suggest Next Concentration

Instead of a fixed grid, use **Bayesian Experiment Design** to minimise experiments:

```python
from src.calibration.active_learning import BayesianExperimentDesigner

designer = BayesianExperimentDesigner(conc_min=0.05, conc_max=10.0)
# After each measurement:
designer.record_measured(conc, shift)
next_conc = designer.suggest()   # maximises variance in log-space
```

In the UI: click **Suggest Next** after each calibration point. The designer recommends
the concentration with highest GPR uncertainty — typically 5–7 points are sufficient
for a complete calibration instead of 10–15 on a fixed grid.

---

## Programmatic Calibration (scripts / notebooks)

```python
import numpy as np
from src.calibration.gpr import GPRCalibration
from src.calibration.conformal import ConformalCalibrator
from src.scientific.lod import compute_lod_loq

# Calibration data
concs  = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
shifts = np.array([-1.2, -3.8, -6.5, -10.1, -18.4])

# Fit GPR
gpr = GPRCalibration()
gpr.fit(concs, shifts)

# Calibrate conformal predictor on same data (use hold-out in production)
cal = ConformalCalibrator()
cal.calibrate(gpr, concs, shifts)

# Predict with guaranteed 90% coverage
shift_new = -7.3
lo, hi = cal.predict_interval(shift_new, alpha=0.10)
print(f"Concentration CI: [{lo:.3f}, {hi:.3f}] ppm (90% coverage)")

# IUPAC LOD/LOQ
result = compute_lod_loq(concentrations=concs, signals=shifts, bootstrap_n=2000)
print(f"LOD: {result['lod']:.4f} ppm  [{result['lod_ci_low']:.4f}, {result['lod_ci_high']:.4f}]")
print(f"LOQ: {result['loq']:.4f} ppm  [{result['loq_ci_low']:.4f}, {result['loq_ci_high']:.4f}]")
```

---

## Common Issues

**Signal not stabilising (DriftAgent keeps firing `drift_warn`)**

- The analyte concentration in the exposure cell has not equilibrated.
- Increase purge time between concentrations.
- Check for leaks in the gas delivery system.

**LOD is unrealistically low**

- The reference spectrum was taken during analyte exposure (not blank).
- Retake reference with confirmed zero-analyte conditions.
- Check `sigma_source` in `session_meta.json` — if `"calibration_residuals"`, the fit
  is too perfect (R² ≈ 1.0), which drives σ toward 0. Add blank measurements.

**Calibration R² < 0.90**

- Ensure all spectra in the same session used the same integration time.
- Check that the sensor was not saturating at high concentrations
  (`QualityAgent` would have emitted `quality_error` events).
- Try physics-informed GPR — it is more stable at sparse data than a plain RBF kernel.

**`ConformalCalibrator` warns "fewer than 10 calibration samples"**

- Conformal coverage guarantees degrade with small calibration sets.
- Add more calibration points before relying on the CI for quantitative decisions.
