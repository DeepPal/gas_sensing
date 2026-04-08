# SpectraAgent — Researcher User Guide

**For:** Lab researchers at Chulalongkorn University LSPR Chemical Gas Sensing Lab  
**Assumes:** You know how to run commands in a terminal. No software engineering experience required.  
**Goal:** Take you from first install to a publication-ready calibration result.

---

## Contents

1. [What this software does — in plain English](#1-what-this-software-does)
2. [Two tools, two purposes](#2-two-tools-two-purposes)
3. [First-time setup (20 minutes)](#3-first-time-setup)
4. [Day 1 — Simulation walkthrough (no hardware needed)](#4-day-1--simulation-walkthrough)
5. [Real hardware workflow — step by step](#5-real-hardware-workflow)
6. [SpectraAgent live UI — panel by panel](#6-spectraagent-live-ui-guide)
7. [Streamlit Dashboard — tab by tab](#7-streamlit-dashboard-guide)
8. [Understanding the AI agents](#8-understanding-the-ai-agents)
9. [From data to publication figures](#9-from-data-to-publication-figures)
10. [Common errors and fixes](#10-common-errors-and-fixes)
11. [Daily lab checklist](#11-daily-lab-checklist)
12. [Frequently asked questions](#12-frequently-asked-questions)
13. [Research integrity workflow (required)](#13-research-integrity-workflow-required)
14. [Research use-case playbook (step-by-step)](#14-research-use-case-playbook-step-by-step)
15. [World-class readiness gap audit](#15-world-class-readiness-gap-audit)

---

## 1. What This Software Does

SpectraAgent is a research platform that:

1. **Reads your spectrometer** (ThorLabs CCS200) in real time
2. **Automatically computes** the 6 LSPR features from every spectrum — including the peak wavelength shift Δλ that tells you the gas concentration
3. **Fits a calibration curve** (LOD, LOQ, R², sensitivity) automatically using your experimental data
4. **Tracks ICH Q2(R1) regulatory validation** across all your sessions — tells you exactly which tests still need to be done before publication
5. **Generates AI-written summaries** of each calibration session (suitable for a Methods section draft)
6. **Produces publication-ready figures** — calibration curves, uncertainty bands, feature importance plots

What it does NOT do: it does not replace your judgment as a scientist. Every result it produces includes an uncertainty estimate, and every AI interpretation is flagged as AI-generated. You verify; the software accelerates.

Document role clarity:

- `docs/RESEARCHER_USER_GUIDE.md` is the operator guide for researchers.
- `RESEARCH_HANDBOOK.md` is the scientific policy and publication reference.

Use this guide when running sessions; use the handbook when deciding scientific claims and acceptance thresholds.

### Handbook commitment (living documentation)

Yes, the researcher handbook is maintained continuously as the platform evolves.
When workflows, validation criteria, integrity gates, or novelty claims change,
the handbook and this guide are updated in the same development cycle so lab
practice stays aligned with software behavior.

If you are unsure whether a workflow changed, check:

1. `RESEARCH_HANDBOOK.md` for scientific policy and acceptance criteria.
2. `CHANGELOG.md` for implementation-level changes.
3. the release workflow integrity step status in CI.

---

## ⚠️ Critical Methodological Limitations — Read Before Using

These are not bugs — they are known limitations you must understand before trusting any result from this platform.

### L1 — Hardcoded default sensitivity (most critical)

`config/config.yaml` ships with `calibration_slope: 0.116 nm/ppm`. This is a literature value for ethanol on a generic Au nanoparticle chip. **If you have not run a real calibration session, every concentration estimate produced by the platform is computed from this fixed number.**

The problem is deeper than just the config file: 0.116 is also hardcoded as a fallback in the live pipeline source files (`src/inference/realtime_pipeline.py`, `src/models/registry.py`, `gas_analysis/core/realtime_pipeline.py`, and others). Changing only the config YAML may not be sufficient — a full calibration session that writes a fitted slope to `output/models/calibration_params.json` is the only reliable way to replace this default throughout the system.

If your chip's true sensitivity differs from 0.116 nm/ppm, all your concentration readings will be proportionally wrong — and you will not see an error message. The pipeline will run normally and produce confident-looking numbers.

**Action required**: Run at minimum one linearity session with a traceable gas standard before trusting any concentration output. Check Tab 1 of the Streamlit Dashboard to confirm a calibration curve has been fitted from your own data and written to `output/models/calibration_params.json`.

### L2 — Simulation models cannot be used with real hardware

The simulation driver generates spectra with a peak at ~717.9 nm. Real Au nanoparticle LSPR chips typically have peaks at 520–560 nm. Any CNN or GPR model trained in simulation mode is **not valid for real hardware** — it will produce meaningless predictions.

Always retrain models on real hardware data before deployment. In Tab 5 → Model Training, clicking **Deploy to Live Acquisition** overwrites the live model checkpoint. Never deploy a simulation-trained model to a hardware session.

### L3 — Temperature correction is tracked but not applied

The platform records room temperature per session (if set in the UI), and the LSPR peak shifts approximately 0.02 nm/°C due to thermal expansion. However, **no temperature compensation is applied to the Δλ signal** in the current version. For sessions spanning a large temperature range (> 3°C variation), this introduces a systematic error that looks identical to analyte binding.

For publication: report room temperature per session, keep sessions within ±2°C, and note this limitation in your Methods section.

### L4 — Cross-config benchmark tests reproducibility, not multi-sensor generalization

The "Cross-Dataset Analysis" in Tab 5 evaluates whether your model generalizes across different acquisition configurations (integration time, etc.) on **the same physical sensor**. It does NOT demonstrate generalization to a different sensor chip or a different physical system. Do not make multi-sensor generalization claims based on this benchmark alone — that requires data from at least 2 independently fabricated chips.

### L5 — ICH Q2(R1) tracker is software-only

The ICH validation tracker in SpectraAgent marks sections as "passed" based on the data you upload and the statistical tests that pass. It does not substitute for regulatory review. If your goal is actual regulatory submission, have a regulatory affairs specialist review the full validation package — not just the tracker output.

---

## 2. Two Tools, Two Purposes

You will use two separate programs. This is intentional — they do different things.

| Tool             | SpectraAgent                                               | Streamlit Dashboard                                                                 |
| ---------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Start with**   | `run_spectraagent.bat`                                     | `run_dashboard_secure.bat`                                                          |
| **Opens at**     | [http://localhost:8765/app](http://localhost:8765/app)     | [http://localhost:8501](http://localhost:8501)                                      |
| **Use it for**   | Live acquisition + AI agents                               | Data analysis + ML training                                                         |
| **When to use**  | In the lab, hardware connected                             | At desk, after collecting data                                                      |
| **What it does** | Records spectra, monitors sensor health, narrates sessions | Loads Joy_Data or sessions, fits calibration curves, trains models, exports figures |

**Rule of thumb:**

- Hardware in front of you → SpectraAgent
- Analyzing data you already collected → Streamlit Dashboard

---

## 3. First-Time Setup

### What you need

- Windows 10 or 11 computer
- Python 3.9 or later (check: open Command Prompt, type `python --version`)
- Git (check: type `git --version`)
- An Anthropic API key (for AI agent features — optional but recommended)
- ~5 GB free disk space

### Step 1 — Get the code

Open Command Prompt or PowerShell in your research folder, then:

```bash
git clone <repository-url>
cd Main_Research_Chula
```

### Step 2 — Create the Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ml]"
```

This takes 5–15 minutes the first time (PyTorch is ~2 GB).

### Step 3 — Set your API key (for AI agents)

Set `ANTHROPIC_API_KEY` as a real operating system environment variable. There is no `.env` file support — a `.env` file placed in the project folder will be ignored.

**Windows — current session only (Command Prompt):**

```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

Then launch SpectraAgent or the dashboard from the same window.

**Windows — permanent (survives restarts):**

1. Open Start → search "Environment Variables"
2. Click "Edit the system environment variables" → "Environment Variables..."
3. Under "User variables", click New
4. Variable name: `ANTHROPIC_API_KEY`, Value: your key
5. Click OK, then restart any open terminals

**Linux / macOS:**

```bash
export ANTHROPIC_API_KEY=your-api-key-here
```

If you do not have a key, the platform still works — AI narration features will be disabled, all scientific calculations remain active.

### Step 4 — Verify installation

```bash
.venv\Scripts\python.exe -m pytest tests/ -q --tb=no -x
```

You should see `1305 passed` (or more). If you see failures, see [Common Errors](#10-common-errors-and-fixes).

### Step 5 — Run simulation to confirm everything works

```bash
run_spectraagent.bat --simulate
```

A browser tab should open at `http://localhost:8765/app`. If it does, setup is complete.

---

## 4. Day 1 — Simulation Walkthrough

Before using real hardware, do this walkthrough once in simulation mode. It takes about 30 minutes and teaches you how everything connects.

### 4.1 Start SpectraAgent in simulation

```bash
run_spectraagent.bat --simulate
```

Browser opens at `http://localhost:8765/app`.

### 4.2 Capture a simulated reference spectrum

1. In the React UI, click **Sensor Setup**
2. Under "Reference spectrum", click **Capture Reference**
3. Wait 3 seconds — a spectrum plot appears showing the LSPR peak at ~717 nm (simulated)
4. The status bar should show `Reference captured — FWHM: X.X nm`

### 4.3 Run a simulated calibration session

1. Click **Start Session**
2. Set **Gas label**: `Ethanol`
3. Set **Target concentration**: `1.0` (ppm)
4. Watch the live spectrum chart — Δλ (wavelength shift) is the blue line
5. After 10 seconds, change the concentration to `2.0` by clicking **Set Concentration** in the experiment plan panel
6. Repeat for 3.0, 5.0, 8.0 ppm
7. Click **Stop Session**

### 4.4 Read the AI session summary

After stopping, look at the **Session Summary** panel (right side). You will see:

```text
Session complete: Ethanol, 5 concentration levels.
Sensitivity: −0.116 nm/ppm (simulation default).
LOD: 0.09 ppm [95% CI: 0.07–0.12 ppm].
R²: 0.997.
ICH Q2(R1) status: §4.2 Linearity — PASS. Next recommended: §4.4 Precision.
```

This summary is AI-generated from your actual experimental data. It is intended as a first draft for your Methods section — always verify the numbers against the raw plots.

### 4.5 Load the session in Streamlit for analysis

Open a second Command Prompt window:

```bash
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

Browser opens at `http://localhost:8501`.

1. Go to **Tab 5 — Data-Driven Science**
2. Under **Dataset Explorer**, choose **SpectraAgent session results**
3. Select the session you just ran (shows as `YYYYMMDD_HHMMSS — Ethanol (50 frames)`)
4. Click **Load Session**
5. You should see: `Loaded 50 frames. Features: [wavelength_shift, peak_wavelength, snr, confidence_score]`

You have now completed the full simulation workflow.

---

## 5. Real Hardware Workflow

### 5.1 Pre-session checklist

Before turning on the spectrometer, do these in order:

- [ ] Sensor chip is freshly functionalized (or note the functionalization age)
- [ ] CCS200 USB cable is connected
- [ ] ThorLabs CCS200 drivers are installed (VISA/IVI driver from ThorLabs website)
- [ ] Carrier gas (dry air or N₂) is flowing
- [ ] Wait **10 minutes** for sensor chip to reach thermal equilibrium
- [ ] Room temperature logged (affects sensitivity by ~0.02 nm/°C)

### 5.2 Start SpectraAgent with hardware

```bash
run_spectraagent.bat --hardware
```

If the CCS200 is detected, the header bar shows `Hardware: CCS200 connected`.

If you see `Hardware not found`, see [CCS200 errors](#ccs200-errors).

### 5.3 Capture dark spectrum (do once per session start)

1. Block all light from the sensor chip (cover the optical path)
2. In SpectraAgent UI → **Sensor Setup** → **Capture Dark (20 frames)**
3. Wait 5 seconds
4. Uncover the optical path

This removes the detector noise floor from all subsequent measurements.

### 5.4 Capture reference spectrum

1. Confirm carrier gas is flowing steadily (baseline stable for 5+ minutes)
2. Click **Capture Reference**
3. The system fits a Lorentzian to find the precise LSPR peak position and FWHM
4. Reference peak and FWHM are stored in `SensorMemory` for cross-session comparison

**If the reference looks noisy** (jagged spectrum): increase integration time with the slider (try 100 ms) or average more frames.

### 5.5 Run a calibration session

For a basic linearity test (ICH §4.2), collect at minimum 5 concentration levels.

**Recommended schedule:**

| Step | Concentration (ppm)          | Wait time |
| ---- | ---------------------------- | --------- |
| 1    | Carrier gas only (blank)     | 5 min     |
| 2    | 0.5                          | 5 min     |
| 3    | 1.0                          | 5 min     |
| 4    | 2.0                          | 5 min     |
| 5    | 5.0                          | 5 min     |
| 6    | 10.0                         | 5 min     |
| 7    | Carrier gas (recovery check) | 5 min     |

For each step:

1. Set the gas concentration on your gas delivery system
2. In SpectraAgent UI, click **Set Concentration** and enter the value
3. Watch the Δλ trace — wait until it is stable (plateau, not still drifting)
4. Click **Mark plateau** or just wait — the system uses the last 10 frames automatically

### 5.6 Real-sensor acceptance gate (required before external claims)

Use this gate after every full calibration batch. Do not claim readiness unless
all required checks pass.

#### A. Pre-run gate

- [ ] Equilibration complete (>= 10 minutes in carrier gas)
- [ ] Dark spectrum captured in this session
- [ ] Reference captured in this session
- [ ] Temperature and humidity logged
- [ ] Gas standard IDs and expiry dates recorded
- [ ] Session operator name and timestamp recorded

#### B. In-run gate

- [ ] At least 5 concentration levels measured for linearity
- [ ] At least 3 replicates per level (6 for repeatability claim)
- [ ] Plateau stability reached before capture at each level
- [ ] Recovery leg recorded in carrier gas after highest concentration
- [ ] No unexplained discontinuity in live drift trace

#### C. Post-run gate

- [ ] Qualification dossier exported
- [ ] Research package ZIP exported
- [ ] Reproducibility summary reviewed
- [ ] Benchmark evidence attached in artifact package
- [ ] Blinded replication manifest attached in artifact package

#### D. Pass/fail thresholds (pilot readiness)

| Metric | Threshold | Decision |
| ------ | --------- | -------- |
| R2 | >= 0.90 | Required pass |
| RMSE | <= 1.5 ppm | Required pass |
| LOD RSD | <= 20% | Required pass |
| LOQ RSD | <= 20% | Required pass |
| Drift trend | No sustained growth across batch | Required pass |
| Critical qualification checks | 0 failures | Required pass |

If any required row fails, classify the run as research-only and do not use it
for external pilot claims.

#### E. Fast triage when gate fails

| Failure pattern | Likely cause | Immediate action |
| --------------- | ------------ | ---------------- |
| LOD and LOQ both degrade | Reference/noise issue | Recapture dark + reference, repeat blank |
| R2 drops, RMSE rises | Concentration prep or plateau timing | Re-run with stricter plateau hold |
| Drift grows during run | Thermal instability or flow instability | Extend equilibration, stabilize flow |
| Sensitivity decreases session-over-session | Surface aging/fouling | Clean or re-functionalize chip |

Keep failed runs. They are required evidence for robustness and root-cause logs.

### 5.6 Stop the session

Click **Stop Session**. The platform automatically:

- Computes LOD, LOQ, LOB, R², RMSE, sensitivity, drift
- Scores sensor health (0–100) against your historical best
- Updates ICH Q2(R1) validation tracker
- Saves everything to `output/sessions/YYYYMMDD_HHMMSS/`
- Generates AI session narrative

The session narrative appears in the **Report** modal (click the document icon). Copy this text into your lab notebook — it is a draft of your session Methods paragraph.

---

## 6. SpectraAgent Live UI Guide

Open `http://localhost:8765/app` after starting `run_spectraagent.bat`.

### Header bar

| Indicator                    | Meaning                                  |
| ---------------------------- | ---------------------------------------- |
| `Hardware: CCS200 connected` | Real spectrometer detected               |
| `Hardware: Simulation`       | Running in simulation mode               |
| `Session: ACTIVE`            | Acquisition running, data being recorded |
| `Health: 87/100`             | SensorHealthAgent score (see below)      |

### Live spectrum chart

- **Blue line**: current raw spectrum (intensity vs wavelength)
- **Red dashed line**: reference spectrum captured at session start
- **Shaded region**: ROI (Region of Interest) where LSPR peak is tracked
- **Green vertical line**: current peak position

If the blue and red lines overlap perfectly → no gas present or sensor not responding. If you see the blue line shift left/right → that is your Δλ signal.

### Δλ time trace (bottom panel)

This is the most important plot. It shows peak wavelength shift vs time.

- A flat line near 0 → carrier gas, no analyte
- A step down (negative shift) after introducing gas → analyte adsorbing
- A plateau → steady-state; this is when calibration data is valid
- Recovery back toward 0 after removing gas → desorption

### Agent log panel (right side)

Shows real-time messages from the AI agents. Messages are color-coded:

| Color  | Source             | What it means                               |
| ------ | ------------------ | ------------------------------------------- |
| Blue   | DriftMonitorAgent  | Drift rate, baseline warnings               |
| Green  | CalibrationAgent   | New calibration point accepted              |
| Orange | QualityAgent       | SNR warning, saturation flag                |
| Purple | ExperimentNarrator | Session summary text                        |
| Red    | AnomalyExplainer   | Unexpected spectral change — read carefully |

### Health panel

Scores your sensor on 5 dimensions. Click any bar for details.

| Score  | Meaning   | Action                           |
| ------ | --------- | -------------------------------- |
| 80–100 | Excellent | Continue normally                |
| 60–79  | Good      | Monitor trend                    |
| 40–59  | Degraded  | Recalibrate this session         |
| < 40   | Poor      | Re-functionalize chip or replace |

**If LOD score drops below 67**: your limit of detection has worsened by >50% from your historical best. This almost always means the chip surface needs cleaning or re-functionalization.

### Validation tracker

Shows your progress through ICH Q2(R1) regulatory validation:

- Grey = not started
- Yellow = in progress / partial data
- Green = passed
- Red = failed

Click any section (§4.1–§4.8) to see what experiment is needed next.

---

## 7. Streamlit Dashboard Guide

Open `http://localhost:8501`.

To start the dashboard — use `run_dashboard_secure.bat` for lab use (includes authentication):

```bat
run_dashboard_secure.bat
```

Or directly via Python:

```bash
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

### Tab 1 — Guided Calibration (📋)

A step-by-step batch workflow for analyzing Joy_Data directories or offline CSV files.

**Use this tab when**: you have existing data files (Joy_Data exports) and want to build a calibration curve without live hardware.

Step-by-step:

1. **Step 1 — Load Data**: enter the path to your data directory, e.g. `Joy_Data/Ethanol/stable_selected`
2. **Step 2 — Load Reference & Blanks**:
   - Upload a reference spectrum CSV recorded in clean carrier gas
   - The system auto-fits a Lorentzian to find the reference FWHM (stored for FOM calculation)
   - Load ≥6 blank spectra (clean carrier gas, no analyte) — these give σ_blank for IUPAC LOD/LOB/NEC
   - ⚠️ Without blank measurements the LOD uses OLS residual σ, which is less rigorous — load blanks for publication
3. **Step 3 — Feature Extraction & Scientific Metrics**:
   - The system computes Δλ, ΔFWHM, ΔI_peak, ΔI_area, ΔI_std for each spectrum
   - Full IUPAC metric suite is shown: NEC, LOB, LOD (±95% CI), LOQ, LOL, R², RMSE
   - **Detection-limit hierarchy check**: the dashboard verifies NEC ≤ LOB ≤ LOD ≤ LOQ — a violation (shown in red) means your blank mean is offset from your reference, requiring reference recapture
   - **Figure of Merit (FOM)**: shown when a reference spectrum FWHM was fitted — FOM = |S|/FWHM (ppm⁻¹), the standard LSPR comparison metric
   - **WLS auto-correction**: if Breusch-Pagan test detects heteroscedastic residuals, weighted least squares is applied automatically (1/c² weights) — an orange notice appears showing both OLS and WLS slopes; report the WLS slope in Methods
   - **Prediction interval at LOD**: expandable panel shows the prediction interval (wider than CI band) — use this in your Methods, not the CI band, per EURACHEM/CITAC CG 4
   - **Residual diagnostics**: Durbin-Watson, Shapiro-Wilk, Breusch-Pagan shown with pass/fail — include these values verbatim in your Supplementary
4. **Step 4 — Model Selection**: choose GPR (recommended) or linear; the system fits and reports R², LOD, LOQ
5. **Step 5 — Export**: download a self-contained HTML report (includes calibration curve, all metrics, auto-generated Methods paragraph)

### Tab 2 — Experiments (📊)

Shows all recorded sessions with metadata. Use this to:

- Browse past sessions by date and gas
- Compare LOD/sensitivity across sessions (cross-session trend)
- Re-export any session to PDF

### Tab 3 — Batch Analysis

For processing large data directories that don't fit the live workflow.

### Tab 4 — Live Sensor (📡)

Real-time feed from CCS200 (requires hardware + `spectraagent start` running).

### Tab 5 — Data-Driven Science (🔬)

This is the advanced analysis tab for ML training and publication figure generation. Use it after collecting data from multiple sessions.

#### Sub-tab: Dataset Explorer

Two modes:

- **Spectral CSV files**: load raw spectra from Joy_Data directories (enables spectral autoencoder)
- **SpectraAgent session results**: load processed features from a live session

After loading, click **Register current dataset** with a name like `Ethanol_config_A`. Register 3+ datasets to unlock the Cross-Dataset benchmark.

#### Sub-tab: Feature Discovery

Trains a spectral autoencoder, then shows a 2D scatter plot (UMAP or t-SNE) of all your spectra colored by concentration. If distinct clusters appear by gas type → the 6-feature basis is discriminating the analytes. This plot goes in your paper.

#### Sub-tab: Model Training

Trains a multi-task neural network (classification + concentration regression) on your loaded dataset.

1. Select model type: **Multi-Task** for most cases
2. Set hyperparameters (defaults are reasonable for small datasets: embed_dim=64, epochs=100)
3. Click **Train Model**
4. After training: click **Save to output/models/** to keep the weights
5. To use this model for live inference: click **Deploy to Live Acquisition**

MLflow automatically logs every training run. View history with:

```bash
.venv\Scripts/python.exe -m mlflow ui --port 5000
# Open http://localhost:5000
```

#### Sub-tab: Cross-Dataset Analysis

**Requires ≥ 3 registered datasets.** Runs leave-one-config-out benchmark — trains on all configs except one, tests on the held-out config. The accuracy score is your generalization claim.

If you only have 1 sensor, generate 3 configs by running:

```bash
spectraagent robustness --param integration_time --range 45:55 --steps 3 \
    --dataset-dir data/Joy_Data/Ethanol
```

This runs the same data through 3 different integration time settings and outputs a comparison table. Each condition becomes one registered dataset.

> **Scope reminder (see L4 above)**: this tests measurement robustness on the same chip, not generalization to a different sensor.

#### Sub-tab: Publication Figures

Generates all 7 required publication figures:

1. Calibration curve with 95% CI bands
2. LOD/LOQ bar chart with bootstrap CI
3. Sensitivity vs session number (reproducibility)
4. 2D embedding scatter (from Feature Discovery)
5. Feature importance heatmap (wavelength attribution)
6. Cross-dataset generalization bar chart
7. ICH Q2(R1) compliance checklist figure

Click **Export all figures (ZIP)** → download `publication_figures.zip` → ready for manuscript submission.

---

## 8. Understanding the AI Agents

The AI agents use Claude (Anthropic) to reason about your sensor's history stored in `output/memory/sensor_memory.json`. They do NOT hallucinate physics — they read your actual measurement data and interpret it.

### What each agent does

**ExperimentNarrator** — writes a prose summary of each session. Example output:

> _"Session 2026-03-31: Ethanol calibration at 5 concentration levels (0.5–10 ppm). Sensitivity: −0.113 nm/ppm (within 3% of historical mean −0.116 nm/ppm). LOD improved from 0.14 ppm (previous session) to 0.09 ppm following reference recapture. R² = 0.998. ICH §4.2 linearity: PASS (R² > 0.9954). Recommend §4.4 precision test (repeatability at 5 ppm, ≥6 replicates) as next step."_

Copy this text directly into your lab notebook. Verify the numbers match the plots before using in a manuscript.

**DriftMonitorAgent** — computes inter-session drift. If it says:

> _"Reference peak has shifted +0.08 nm over 3 sessions. Consistent with chip aging. No immediate action required."_

This is normal. If it says > 0.2 nm shift: recapture your reference spectrum.

**SensorHealthAgent** — compares your current session to your historical best. If it says:

> _"Sensitivity score: 62/100. Current sensitivity −0.081 nm/ppm is 30% below historical best −0.116 nm/ppm. Possible cause: surface fouling or active site blockage. Recommended action: clean chip with EtOH/IPA rinse, recapture reference, run verification step at known 5 ppm standard."_

Follow the recommendation literally.

**CalibrationValidatorAgent** — reads ICH Q2(R1) requirements and tells you what's missing. Example:

> _"§4.1 Specificity: NOT STARTED. Required: measure sensor response to IPA and methanol at 10× OSHA-PEL. §4.4 Repeatability: IN PROGRESS (4/6 replicates at 5 ppm). Run 2 more replicates to complete."_

This is your roadmap to publication readiness.

**AnomalyExplainer** — fires when the spectrum looks unexpected. Take these seriously. Example:

> _"WARNING: Peak intensity dropped 40% compared to previous frame. Possible causes: (1) fiber optic coupling disturbed, (2) light source warming up — wait 10 min, (3) sensor chip partially delaminated. Recommend: verify optical alignment before continuing."_

When you see a red anomaly message, pause and investigate before continuing the session.

### When the API key is not set

---

## 13. Research Integrity Workflow (Required)

Use this checklist for any dataset intended for publication or external sharing.

### 13.1 Session integrity before analysis

1. Acquire and stop session normally (do not edit raw outputs by hand).
2. Confirm session manifest exists in `output/sessions/<session_id>/`.
3. Run replay verification:

```bash
.venv\Scripts\python.exe scripts/replay_session.py --help
```

Use the command help to verify the exact replay arguments for your session path.

### 13.2 Integrity gate before release/reporting

Run local integrity gate:

```bash
.venv\Scripts\python.exe scripts/research_integrity_gate.py --allow-empty
```

For gate self-validation (tamper detection check):

```bash
.venv\Scripts\python.exe scripts/research_integrity_gate.py --self-check
```

Interpretation:

1. `self-check OK` means the verifier correctly detects tampering behavior.
2. `SKIP: No manifests found` is acceptable only for environments without session outputs.
3. Any `FAIL` must be resolved before publishing metrics.

### 13.3 What to include in manuscript supplements

1. Session manifest(s),
2. integrity verification output,
3. software commit hash,
4. calibration and uncertainty outputs used for final figures.

All scientific calculations (LOD, LOQ, calibration curves, feature extraction, model training) work without the API key. Only the prose narratives and AI explanations are disabled. You will see `[API key not configured — AI narration disabled]` in the agent panel.

---

## 9. From Data to Publication Figures

This is the complete path from "data collected" to "figures submitted to journal."

### Step 1 — Collect minimum required data

For an ICH Q2(R1)-compliant analytical paper you need:

| Experiment                   | Minimum                                         | What it gives you                      |
| ---------------------------- | ----------------------------------------------- | -------------------------------------- |
| Linearity sessions (Ethanol) | ≥3 sessions, 5+ concentrations each             | R², calibration curve, LOD/LOQ with CI |
| Repeatability                | 6 replicates at same conc, same day             | Intra-day RSD%                         |
| Intermediate precision       | Same as above, ≥3 different days                | Inter-day RSD%                         |
| Selectivity                  | 1 session each of IPA, methanol at high conc    | K_B,A selectivity coefficients         |
| Robustness                   | 3 sessions with ±10% integration time variation | Robustness claim                       |

### Step 2 — Check ICH compliance in SpectraAgent

In the live UI → Validation Tracker. Each green section is a completed test. All 8 sections (§4.1–§4.8) must be green before submission.

### Step 3 — Generate publication figures in Tab 5

1. Open Streamlit → Tab 5 → Dataset Explorer
2. Load your Ethanol session data (or Joy_Data/Ethanol directory)
3. Go to Publication Figures sub-tab
4. Click **Export all figures (ZIP)**

### Step 4 — Verify numbers against RESEARCH_HANDBOOK.md targets

Open `RESEARCH_HANDBOOK.md` §2.1 and check each metric against the threshold:

| Metric                    | Your result | Target                        | Pass? |
| ------------------------- | ----------- | ----------------------------- | ----- |
| Calibration R²            | —           | > 0.9954                      | —     |
| LOD                       | —           | < 0.1 ppm (with 95% bootstrap CI) | —  |
| Sensitivity ± SE(S)       | —           | Report ± SE                   | —     |
| FOM (ppm⁻¹)               | —           | Report (no fixed threshold)   | —     |
| NEC ≤ LOB ≤ LOD ≤ LOQ     | —           | All PASS (hierarchy check)    | —     |
| BP test (homoscedasticity)| —           | p ≥ 0.017 (OLS valid)         | —     |
| Repeatability RSD         | —           | < 2% intra-day                | —     |
| MK trend (LOD stability)  | —           | No significant trend (p ≥ 0.05) | —  |

### Step 5 — Write the Methods section

Use `docs/PAPER_METHODS_TEMPLATE.md` as the starting template — it contains fill-in-the-blank text for all statistical methods that are now implemented in the platform, including the equations reviewers will look for. The AI-generated session narratives from SpectraAgent provide the per-session metadata.

The Methods section must include (per journal requirements for *Analytical Chemistry* / *Sensors & Actuators B*):

**Experimental setup**
- Hardware model, serial number, integration time, averaging
- Functionalization protocol
- Room temperature and humidity (per session)

**Statistical methods** (copy from `docs/PAPER_METHODS_TEMPLATE.md` §2.3–2.6):
- LOD/LOQ derivation with IUPAC 2012 equations and σ_blank source
- Bootstrap CI parameters (n=1000, fix_noise_std flag — state whether σ_blank was held fixed)
- Prediction interval at LOD (state this was used, not just confidence band)
- Homoscedasticity check and whether WLS was applied
- Mandel linearity test F-statistic and p-value
- FOM with FWHM_ref source
- Cross-session Mann-Kendall τ and trend classification

**Reproducibility**
- Git commit hash: `git rev-parse --short HEAD`
- Data archived in `output/sessions/*/` (include in supplementary)
- Integrity verification output: `python scripts/research_integrity_gate.py --allow-empty`
- Temperature correction not applied (see L3 above — note in Methods)

---

## 10. Common Errors and Fixes

### CCS200 errors

**`Error -1073807343: Device connected but not powered/ready`**

The spectrometer USB is connected but the device is not initialized. Fix:

1. Unplug and replug the USB cable
2. Wait 5 seconds
3. Restart SpectraAgent

**`Error -1073807339 (VI_ERROR_TMO): Timeout on first read`**

A previous session crashed without closing the device handle. Fix:

1. Close all Command Prompt windows running SpectraAgent
2. Unplug CCS200 USB, wait 10 seconds, replug
3. Restart SpectraAgent

**`total_samples=0` — no data being recorded**

Rare after a system update. Fix:

```bash
.venv\Scripts\python.exe -c "from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService; s = RealtimeAcquisitionService(); print('OK')"
```

If this fails, the hardware driver is not loading. Check ThorLabs VISA driver installation.

**RSD ~12% on dark spectrum is normal** — this is the detector noise floor. Do not be alarmed.

**`research_preflight.py` reports WARN/FAIL**

For publication-grade runs, resolve warnings before acquisition whenever possible.
Common causes:

1. Missing or stale session manifest in latest run.
2. Default or missing calibration slope in config path.
3. Environment compensation still disabled in config.

### Streamlit dashboard errors

**`Error: No module named 'torch'`**

PyTorch was not installed. Fix:

```bash
.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cpu
```

**`Tab 5 unavailable: ImportError`**

Expand the error details in the tab. Usually a missing optional package. Fix:

```bash
.venv\Scripts\pip.exe install plotly scikit-learn umap-learn
```

**Dataset Explorer shows `No spectrum CSV files found`**

You are pointing at a Joy_Data directory that has no CSVs matching the expected format. Check:

1. The directory exists and contains `.csv` files
2. Each CSV has a `wavelength` column and an `intensity` column
3. The filenames or parent directory names contain a concentration number (e.g. `1.0_ppm_Ethanol.csv` or a directory named `1.0/`)

### Model training errors

**`Contrastive training requires analyte class labels`**

The loaded dataset has no concentration labels that map to distinct classes. Use **Multi-Task** model type instead, or ensure your data directory names contain the gas name.

**`Training failed: CUDA out of memory`**

Reduce batch size (slider in the UI) from 32 to 8.

### SpectraAgent / API errors

**`Agent panel: [API key not configured]`**

Set the `ANTHROPIC_API_KEY` environment variable before launching (see Section 3, Step 3). There is no `.env` file support — setting a key in a `.env` file will not work. No key = no AI narration. All scientific calculations still work.

**`Agent panel shows nothing after Stop Session`**

The Claude API call may have timed out. Click **Regenerate report** button in the Report modal. If it fails again, check your internet connection and API key.

---

## 11. Daily Lab Checklist

Print this and keep it next to the spectrometer.

### Before starting

- [ ] Run preflight readiness check:

  ```bash
   .venv\Scripts\python.exe scripts/research_preflight.py --fail-on-warning --require-manifest
  ```

- [ ] CCS200 plugged in and powered (green LED on device)
- [ ] Carrier gas flowing ≥10 minutes
- [ ] Start SpectraAgent: `run_spectraagent.bat --hardware`
- [ ] Verify `Hardware: CCS200 connected` in header bar
- [ ] Capture dark spectrum (block light first)
- [ ] Capture reference spectrum (confirm carrier gas only)
- [ ] Check SensorHealth score — if < 55, do not proceed; recalibrate or re-functionalize

### During session

- [ ] Set concentration in UI before each gas introduction
- [ ] Wait for Δλ trace to reach plateau before marking calibration point
- [ ] Log room temperature in the UI (temp slider or manually in session notes)
- [ ] Watch Agent log for red anomaly messages — investigate immediately

### After session

- [ ] Click Stop Session — wait for AI summary to appear (30–60 seconds)
- [ ] Copy session narrative to lab notebook
- [ ] Check ICH Validation Tracker — note which test was completed
- [ ] Close SpectraAgent cleanly (Ctrl+C in the terminal) — important to avoid CCS200 timeout errors next session

### Weekly

- [ ] Open Streamlit → Tab 2 (Experiments) → check sensitivity trend across sessions
- [ ] If sensitivity decreasing over 3+ sessions → re-functionalize chip
- [ ] Back up `output/sessions/` to external drive

---

## 12. Frequently Asked Questions

**Q: My calibration R² is 0.98 but the target is 0.9954. What should I do?**

Check the residual plot (shown in Tab 1 Step 4). If residuals show a curve (not random scatter), your response is nonlinear — try Langmuir model instead of linear. If residuals are random but R² is still low, increase the number of replicates per concentration level.

**Q: The Δλ trace drifts even in carrier gas. Is this normal?**

Slow drift (< 0.05 nm/min) is normal and due to temperature. Log the room temperature and use the drift correction option in the calibration workflow. Fast drift (> 0.1 nm/min) usually means the reference spectrum is stale — recapture it.

**Q: How do I know if my chip needs replacing vs just recalibrating?**

The SensorHealthAgent tells you. If Sensitivity score < 50 AND FWHM of the reference peak is broadening across 3+ sessions → re-functionalize. If 3+ recalibration events in a row show no improvement → new chip.

**Q: Can I use this with a different spectrometer (not CCS200)?**

Yes. Any spectrometer that outputs a CSV with `wavelength` and `intensity` columns works with the batch analysis and Streamlit Dashboard. For live acquisition with a different spectrometer, a hardware driver plugin is needed (see `spectraagent/drivers/`).

**Q: What does "session_features mode" mean in the Dataset Explorer?**

When you load a SpectraAgent session (not a raw Joy_Data directory), Tab 5 loads processed features (`wavelength_shift`, `peak_wavelength`, `snr`, `confidence_score`) rather than raw spectra. The spectral autoencoder sub-tab will not give physically meaningful results in this mode. Model Training and Cross-Dataset Analysis work normally.

**Q: I ran `spectraagent robustness` but the output is empty. Why?**

The robustness command requires a `--dataset-dir` pointing to labelled CSV spectra (not a `--gas` flag). Example:

```bash
spectraagent robustness --param integration_time --range 45:55 --steps 3 \
    --dataset-dir data/Joy_Data/Ethanol --output-csv robustness_result.csv
```

**Q: The ICH Validation Tracker shows §4.1 Specificity as "Not started." What exactly do I need to do?**

Expose the sensor to each interferent (IPA, methanol) at 10× their OSHA-PEL concentration while running a full calibration session. Set the gas label to the interferent name (e.g. `IPA_interferent`). The system automatically detects this is an interferent session and computes the selectivity coefficient K = Δλ(interferent) / Δλ(target). You need K < 0.05 to pass specificity.

**Q: Where are my results saved?**

- Session raw results: `output/sessions/YYYYMMDD_HHMMSS/pipeline_results.csv`
- Session metadata: `output/sessions/YYYYMMDD_HHMMSS/session_meta.json`
- Sensor memory (cross-session): `output/memory/sensor_memory.json`
- Trained models: `output/model_versions/` and `models/registry/cnn_classifier.pt`
- MLflow experiment history: `mlruns.db` (view with `mlflow ui`)

**Q: How do I cite this software in my paper?**

Include in the Methods section:

> _"Data acquisition and analysis were performed using SpectraAgent vX.X (commit `<git hash>`, available at `<repository url>`). Python X.X, NumPy X.X, SciPy X.X, scikit-learn X.X, PyTorch X.X."_

To get the exact git hash at the time of your experiment:

```bash
git rev-parse --short HEAD
```

Note: the git hash is **not** stored automatically in `session_meta.json`. The fields actually present in that file are: `gas_label`, `target_concentration`, `hardware`, `temperature_c`, `humidity_pct`, `session_id`, `started_at`, `stopped_at`, `frame_count`. Record the hash manually in your lab notebook alongside the session ID.

---

_For scientific reference (physics, calibration theory, ICH validation requirements): see `RESEARCH_HANDBOOK.md`_  
_For system architecture and developer docs: see `docs/SYSTEM_ARCHITECTURE.md`_  
_For deployment and security setup: see `DEPLOY_RESEARCH_LAB.md`_

---

## 14. Research Use-Case Playbook (Step-by-Step)

This section maps real researcher goals to executable workflows.

### Use Case 1 — First Week Onboarding (No Hardware)

Goal: Learn the full pipeline safely in simulation.

Steps:

1. Run setup from Section 3.
2. Start simulation:

   ```bash
   run_spectraagent.bat --simulate
   ```

3. Capture reference and run one 5-level simulated calibration session.
4. Stop session and read the AI summary.
5. Open Streamlit and load the session in Tab 5.

Deliverable: one complete session folder under `output/sessions/` and one exported figure ZIP.

### Use Case 2 — Daily Pre-Experiment Health Check

Goal: Decide whether the chip is ready before consuming standards and lab time.

Steps:

1. Run preflight checks:

   ```bash
   .venv\Scripts\python.exe scripts/research_preflight.py --fail-on-warning --require-manifest
   ```

2. Start hardware mode.
3. Capture dark and reference spectra.
4. Observe SensorHealth score and DriftMonitor messages.
5. If score < 55, stop and recalibrate before running expensive experiments.

Deliverable: go/no-go decision recorded in notebook with session ID.

### Use Case 3 — Single-Analyte Calibration (Publication Core)

Goal: Generate valid calibration for one analyte (e.g., ethanol).

Steps:

1. Use concentration ladder: blank, 0.5, 1, 2, 5, 10 ppm, blank recovery.
2. At each level, wait for plateau in Δλ trace.
3. Stop session and review LOD, LOQ, R², sensitivity.
4. Export calibration report and plots.

Acceptance criteria:

1. R² meets your target.
2. CI widths are acceptable.
3. No unresolved anomaly alerts.

### Use Case 4 — Repeatability Study (ICH §4.4 Intra-day)

Goal: Demonstrate repeatability using same-day replicates.

Steps:

1. Choose one concentration near target operating point.
2. Run at least 6 replicate measurements under same conditions.
3. Compute RSD% from session outputs.
4. Record pass/fail against acceptance threshold.

Deliverable: repeatability table for manuscript supplementary material.

### Use Case 5 — Intermediate Precision (Multi-day)

Goal: Demonstrate day-to-day robustness.

Steps:

1. Repeat the same repeatability protocol on at least 3 different days.
2. Keep protocol fixed (operator, flow, integration settings if possible).
3. Compare cross-session variance with statistical tests.

Deliverable: inter-day precision figure and summary stats.

### Use Case 6 — Selectivity / Interferent Validation (ICH §4.1)

Goal: Quantify interference risk from non-target gases.

Steps:

1. Run target analyte session at reference concentration.
2. Run separate sessions for interferents (e.g., IPA, methanol) at defined challenge levels.
3. Compute selectivity coefficient K values.
4. Document whether thresholds are met.

Deliverable: selectivity matrix and specificity statement.

### Use Case 7 — Robustness Sweep (ICH §4.8)

Goal: Show method remains valid under controlled parameter variation.

Steps:

1. Run automated robustness command:

   ```bash
   spectraagent robustness --param integration_time --range 45:55 --steps 3 \
       --dataset-dir data/Joy_Data/Ethanol --output-csv robustness_result.csv
   ```

2. Compare LOD and R² shifts across settings.
3. Keep results in manuscript appendix.

Deliverable: robustness table and pass/fail conclusion.

### Use Case 8 — Drift Investigation and Recovery

Goal: Resolve unstable baselines before data quality collapses.

Steps:

1. Observe drift trend in live UI and experiments tab.
2. Recapture dark/reference.
3. Re-run a short verification session at known concentration.
4. If still degraded, clean/re-functionalize chip.

Deliverable: corrective action record tied to session IDs.

### Use Case 9 — Model Training for Research Analysis

Goal: Train model for analysis and comparison experiments.

Steps:

1. Load curated datasets in Tab 5.
2. Train Multi-Task model with tracked hyperparameters.
3. Save model artifact.
4. Evaluate with held-out data and confidence intervals.

Deliverable: model card with metrics and reproducible settings.

### Use Case 10 — Cross-Config Generalization Benchmark

Goal: Evaluate transfer across acquisition configurations.

Steps:

1. Register at least 3 datasets/configs in Tab 5.
2. Run leave-one-config-out evaluation.
3. Export benchmark chart with CI.

Deliverable: generalization claim scoped to same-sensor config transfer.

### Use Case 11 — Manuscript Figure and Evidence Package

Goal: Generate publication package from validated sessions.

Steps:

1. Export all publication figures (Tab 5).
2. Run integrity gate and include outputs.
3. Add methods metadata and git hash.
4. Attach session manifests and core tables.

Deliverable: complete submission-ready evidence bundle.

### Use Case 12 — External Review / Audit Readiness

Goal: Prepare for collaborator, supervisor, or reviewer audit.

Steps:

1. Provide session IDs and manifest files.
2. Provide integrity gate outputs.
3. Provide calibration source data and scripts used.
4. Provide exact software commit hash and environment summary.

Deliverable: traceable, replayable audit package.

---

## 15. World-Class Readiness Gap Audit

This is the current gap list to satisfy world-class researcher expectations.

### Critical gaps (close first)

1. **Default slope dependency risk**
   - Current risk: uncalibrated runs may still rely on default slope.
   - Required improvement: enforce fail-closed concentration reporting until experimental calibration is confirmed.

2. **Temperature compensation not yet active in primary inference path**
   - Current risk: thermal drift can masquerade as analyte signal.
   - Required improvement: apply validated temperature/humidity correction in production inference, not only metadata tracking.

3. **Automatic provenance capture completeness**
   - Current risk: git hash and full environment are partly manual.
   - Required improvement: auto-write commit hash, dependency snapshot, and runtime config checksum into each session manifest.

### High-impact scientific upgrades

1. **Independent multi-chip validation**
   - Needed for stronger external validity claims beyond single-chip robustness.

2. **Mixture discrimination benchmark (binary/ternary gases)**
   - Needed for next novelty tier and broader publication impact.

3. **Blinded external dataset evaluation**
   - Needed to reduce internal bias and strengthen reviewer confidence.

### Operational excellence upgrades

1. **Formal SOP pack**
   - Printable SOPs for calibration, drift response, and integrity validation.

2. **Automated release evidence bundle**
   - CI artifact containing figures, manifests, integrity logs, and validation status in one package.

3. **Long-term stability dashboard**
   - Weekly/monthly health trend reports for preventative maintenance planning.

### Practical priority order

1. Enforce fail-closed calibration requirement for concentration output.
2. Activate and validate temperature compensation in live inference.
3. Automate full provenance capture in session artifacts.
4. Run multi-chip validation campaign.
5. Add mixture discrimination study and blinded benchmark.

Completion of these items will materially strengthen claims for world-class research quality, reproducibility, and publishability.
