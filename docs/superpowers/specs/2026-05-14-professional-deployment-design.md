# SpectraAgent — Professional Deployment Design

**Date:** 2026-05-14
**Status:** Approved — proceeding to implementation plan
**Branch:** feat/research-evidence-pack → main (via milestoned PRs)

---

## 1. Objective

Make SpectraAgent genuinely world-class for three audiences simultaneously:

- **Research scientists** — calibration sessions, reproducible results, citable software
- **Industrial integrators** — Docker single-command start, REST API contract, hardware plugin guide
- **Open-source contributors** — one coherent architecture, no dead code, clear extension points

The platform must meet the engineering discipline bar of Tier 1 analytical instrumentation
software: every scientific output numerically reproducible, every scientific decision
documented, every refactor parity-verified before old code is deleted.

---

## 2. Critical Finding

**The two runtimes currently use different GPR calibration kernels.** This is a live
scientific inconsistency affecting every session today.

| Runtime | File | Kernel | Restarts | Uncertainty model |
|---------|------|--------|----------|-------------------|
| SpectraAgent (live) | `src/calibration/gpr.py` | Matérn ν=5/2 | 10 | epistemic + aleatoric in quadrature |
| Streamlit dashboard | `gas_analysis/core/intelligence/gpr.py` | RBF | 2 | posterior std only |

The Matérn ν=5/2 kernel is scientifically correct for LSPR calibration (Langmuir isotherms
are twice-differentiable, not infinitely differentiable — the exact regime Matérn 5/2
models). The RBF kernel oversmooths. Phase 1 fixes this before any cleanup proceeds.

Two secondary findings:
- `src/calibration/gpr.py` docstring falsely states "RBF kernel" — documentation bug.
- `detect_outliers` has no equivalent in `src/` — must be added before migration.

---

## 3. Target Architecture

```
spectraagent/webapp/          ← Live acquisition: FastAPI + React (WebSocket)
  routes/                     ← NEW: acquisition / sessions / reports / agents
  server.py                   ← create_app() factory only (~150 lines post-split)

dashboard/                    ← Scientific analysis: Streamlit
  agentic/                    ← NEW: split from 4165-line monolith
    steps.py                  ← Step 1-5 business logic
    visualizations.py         ← Plotly / Matplotlib chart builders
    lspr_physics.py           ← Lorentzian fit, FOM, WLS, FWHM helpers
    tab.py                    ← Streamlit UI wiring only (~200 lines)

src/                          ← ALL business logic (typed, tested, stable API)
  inference/                  ← Real-time pipeline (canonical)
  calibration/                ← GPR (Matérn), PLS, conformal, selectivity
  scientific/                 ← LOD/LOQ, Allan deviation, residual diagnostics,
  signal/                     ←   publication tables, cross-session stats, quality
  models/                     ← CNN, ONNX export, versioning
  reporting/                  ← Publication-quality figures and tables
  public_api.py               ← Stable external import surface

gas_analysis/                 ← Hardware + advanced signal ONLY (no business logic)
  acquisition/                ← CCS200 drivers — unique, not in src/
  advanced/                   ← ICA, MCR-ALS — unique, used by Streamlit
  logging_setup.py            ← Logging config

tests/
  science_regression/         ← NEW: numerical output contracts (CI gate)
  migration_parity/           ← NEW: old vs new equivalence before deletion

CHARTER.md                    ← NEW: north star — what correct means
docs/adr/science/             ← NEW: immutable scientific decision log
docs/quickstart/              ← NEW: per-audience getting-started guides
```

Everything in `gas_analysis/core/` is deleted by end of Phase 3.
`spectrometer_webapp/`, `n8n/`, `run.py`, phase migration scripts are deleted in Phase 2.

---

## 4. Phased Execution Plan

### Phase 0 — Scientific Foundation (Week 1)
*Nothing else starts until this is done.*

**Goal:** Establish what "correct" means before any code moves.

**Deliverables:**

**0a. `CHARTER.md`** (root level, ~200 lines)
- What the system does and explicitly does not do
- Correct definition for each scientific output:
  - LOD: IUPAC 2011, 3σ blank method, bootstrap n=2000
  - GPR: Matérn ν=5/2, 10 restarts, aleatoric tracked separately
  - FOM: |S| / FWHM (Willets & Van Duyne 2007)
  - Signal: Δλ (peak shift, nm), redshift = negative convention
  - Conformal prediction: split-CP, not Bayesian credible intervals
- Success criteria table (per-milestone)
- Scientific invariants list — items requiring an ADR to change

**0b. `docs/adr/science/` — five ADR files**

| File | Decision |
|------|----------|
| `001-lod-definition.md` | IUPAC 2011 (3σ blank), not 3× SNR, not DIN 32645 |
| `002-gpr-kernel.md` | Matérn ν=5/2; why not RBF; why 10 restarts not 2 |
| `003-lspr-signal-convention.md` | Δλ primary; redshift negative; ΔI secondary only |
| `004-fom-definition.md` | Willets & Van Duyne 2007; |S|/FWHM units ppm⁻¹ |
| `005-conformal-prediction.md` | Split-CP; coverage guarantee; why not Bayesian |

Format: immutable once merged. Science change = new ADR, never edit existing.

**0c. `tests/science_regression/`**

Fixed pinned fixture: synthetic 6-point LSPR calibration curve (stored as
`tests/fixtures/lspr_calibration_fixture.npz`). Assertions:

```python
# LOD
assert abs(lod - BASELINE_LOD) / BASELINE_LOD < 0.02        # ±2%

# Sensitivity
assert abs(sensitivity - BASELINE_SENSITIVITY) / abs(BASELINE_SENSITIVITY) < 0.01  # ±1%

# GPR posterior std at 1.0 ppm
assert abs(gpr_std - BASELINE_GPR_STD) / BASELINE_GPR_STD < 0.05  # ±5%

# Peak wavelength shift
assert abs(delta_lambda - BASELINE_DELTA_LAMBDA) < 0.01     # ±0.01 nm

# Allan deviation τ_opt
assert abs(tau_opt - BASELINE_TAU_OPT) / BASELINE_TAU_OPT < 0.05  # ±5%
```

CI job: `science-regression` — runs on every PR, blocks merge on failure.

**0d. `tests/migration_parity/` framework**
Empty directory with `conftest.py` establishing the pattern:
load old implementation + new implementation → same input → `assert_allclose(rtol=1e-6)`.
Populated progressively during Phase 3.

---

### Phase 1 — Fix Live Scientific Inconsistency (Week 2)
*Single PR. Highest correctness priority.*

**1a. Fix `src/calibration/gpr.py` docstring**
- Line 13: remove "RBF kernel" — replace with "Matérn ν=5/2 kernel"
- Add physics justification comment citing the Matérn choice

**1b. Add `src/signal/quality.py`**
New file. Moves `detect_outliers` into `src/` with identical implementation,
verified by a migration parity test before the dashboard import is re-routed.

Functions to add:
- `detect_outliers(spectra, threshold)` — Z-score multi-metric outlier detection
- `compute_snr(signal, noise_region)` — if not already in src/ (verify first)

**1c. Migrate dashboard GPR and CNN imports**

In `dashboard/agentic_pipeline_tab.py` lines 88–89:
```python
# Delete:
from gas_analysis.core.intelligence.classifier import CNNGasClassifier
from gas_analysis.core.intelligence.gpr import GPRCalibration

# Add:
from src.models.cnn import CNNGasClassifier
from src.calibration.gpr import GPRCalibration
```

**1d. Migrate dashboard signal/outlier imports**

Lines 63, 72–76:
```python
# Delete:
from gas_analysis.core.signal_proc import detect_outliers
from gas_analysis.core.preprocessing import (...)
from gas_analysis.core.signal_proc import (...)

# Add:
from src.signal.quality import detect_outliers
from src.signal.transforms import (baseline_correction, smooth_spectrum,
                                   normalize_spectrum)
from src.preprocessing.denoising import wavelet_denoise, als_baseline
```

**Verification gate:** Run `tests/science_regression/` before and after. If LOD shifts
>5%, this is a scientific finding — document correction in ADR-002, update baseline.

---

### Phase 2 — Zero-Risk Dead Code Deletion (Days 8–12)
*Can overlap Phase 1. Zero regression risk.*

**Deletions (single PR per group):**

PR-A: Legacy standalone prototype
```
spectrometer_webapp/           # no imports anywhere in main codebase
```

PR-B: Unintegrated tooling
```
n8n/                           # only in docs/archive reference
scripts/phase5_replace_bodies.py
scripts/phase6_replace_bodies.py
scripts/phase7a_batch_aggregation.py
scripts/phase7b_extract_bodies.py
experiments/mlruns/            # duplicate of root mlruns/
```

PR-C: Superseded entry point
```
run.py                         # replaced by launcher.py + spectraagent CLI
```
Replace with a one-line stub that prints:
`"run.py is retired. Use: python launcher.py (or: spectraagent start)"`
then exits with code 1. Gives clear error rather than silent removal.

**pyproject.toml:** Remove `"spectrometer_webapp"` from linting paths (line 166).

**Verification:** All tests + science regression suite pass after each PR.

---

### Phase 3 — Complete Migration with Parity Gates (Weeks 3–4)

**3a. Write parity tests for remaining imports**

For each remaining `gas_analysis.core` import in `dashboard/`:
- `dashboard/app.py` line 126: `gas_analysis.core.signal_proc.*`
- `dashboard/app.py` line 1301: `gas_analysis.advanced.mcr_als._als_nnls` (stays — unique)
- `dashboard/experiment_tab.py` line 56: `gas_analysis.core.signal_proc.*`
- `dashboard/predict_tab.py` line 119: `gas_analysis.core.signal_proc.baseline_correction`

Write parity test → confirm numerical equivalence → re-route → delete.

**3b. Re-route all remaining dashboard imports to `src/`**

`dashboard/app.py` line 47: `gas_analysis.logging_setup` → keep (logging_setup stays in
gas_analysis/ — it is not `gas_analysis/core/`).

**3c. Add CI grep gate to `quality.yml`**
```yaml
- name: No gas_analysis.core imports remain
  run: |
    if grep -rn "from gas_analysis\.core\|import gas_analysis\.core" \
         --include="*.py" dashboard/ src/ spectraagent/ tests/ 2>/dev/null; then
      echo "FAIL: gas_analysis.core import found outside gas_analysis/"
      exit 1
    fi
    echo "OK: no gas_analysis.core imports outside gas_analysis/"
```

**3d. Delete `gas_analysis/core/` subtree (~7,000 lines)**

Files removed:
- `pipeline.py` (4,881 lines)
- `realtime_pipeline.py` (1,061 lines)
- `signal_proc/` (advanced.py, basic.py)
- `preprocessing.py`
- `intelligence/` (classifier.py, gpr.py, model_registry.py)
- `calibration/` (methods.py + contents)
- `scientific/` (kinetics.py, lod.py)
- `tools/build_report.py` (1,738 lines) — at `gas_analysis/tools/`, deleted separately
- `dynamics.py`, `calibration_memory.py`, `performance_monitor.py`,
  `research_report.py`, `responsive_frame_selector.py`, `run_each_gas.py`

**Remaining `gas_analysis/` after Phase 3:**
```
gas_analysis/
  acquisition/     ← CCS200 hardware drivers
  advanced/        ← ICA, MCR-ALS (deconvolution_ica.py, mcr_als.py)
  logging_setup.py
  __init__.py
```

**Verification:** All tests + science regression + grep gate pass.

---

### Phase 4 — File Splitting (Week 5)
*Readability. One PR per split.*

**4a. Split `dashboard/agentic_pipeline_tab.py` (4,165 lines)**

```
dashboard/agentic/
  __init__.py           # re-exports for app.py compatibility
  steps.py              # Step 1-5 logic: preprocessing, features, calibration, QC
  visualizations.py     # all Plotly / Matplotlib chart builders
  lspr_physics.py       # Lorentzian fit, FOM, WLS correction, FWHM helpers
  tab.py                # Streamlit st.* calls only; imports from above
```

Target: `tab.py` ≤ 250 lines. Each module ≤ 600 lines.

**4b. Split `spectraagent/webapp/server.py` (1,896 lines)**

```
spectraagent/webapp/
  routes/
    __init__.py
    acquisition.py      # /api/acq/* — start, stop, dark, reference
    sessions.py         # /api/sessions/* — list, get, export, archive
    reports.py          # /api/reports/* — generate, fallback, download
    agents.py           # /api/agents/* — anomaly, narrator, ask, plan
  server.py             # create_app() factory + lifespan + middleware only
```

Target: `server.py` ≤ 150 lines post-split. Route modules ≤ 400 lines each.

`contracts/openapi_baseline.json` must be regenerated after the split and verified
identical (route handlers move, not route signatures).

---

### Phase 5 — GitHub and Documentation (Week 6)

**5a. `CHARTER.md`** already written in Phase 0 — verify it's accurate now.

**5b. README restructure**

Three-section landing page:
```markdown
# SpectraAgent

Hardware-agnostic optical spectroscopy platform.
Raw photons → calibrated, uncertainty-quantified results with AI-native analysis.

[badges]

## Start in 60 seconds
    docker compose up

Then open:
- http://localhost:8765/app    ← live acquisition (SpectraAgent)
- http://localhost:8501        ← scientific analysis (Streamlit)

## Guides
→ Research labs              docs/quickstart/research.md
→ Industrial integrators     docs/quickstart/integration.md
→ Contributors               CONTRIBUTING.md + docs/guides/plugin-walkthrough.md
```

**5c. `docs/quickstart/research.md`**
- Prerequisites (Python 3.10+, Docker, or pip install)
- Running a calibration session end-to-end
- Streamlit tabs walkthrough with screenshots
- Exporting results (HDF5, CSV, HTML report)
- Citing the software (CITATION.cff, Zenodo DOI)

**5d. `docs/quickstart/integration.md`**
- Hardware plugin guide (entry-points, driver interface)
- REST API contract (`contracts/openapi_baseline.json` walkthrough)
- ONNX model export and inference
- Docker Compose in production (resource limits, health checks, log rotation)

**5e. Docker Compose verification**
- Run `docker compose up` end-to-end in CI (the existing `deploy-smoke.yml` workflow)
- Confirm both services are reachable at their ports within the health-check timeout
- Document the verified startup sequence in `docs/quickstart/integration.md`

---

## 5. Quality Gates (all enforced in CI)

| Gate | When | Blocks |
|------|------|--------|
| `science-regression` (new) | Every PR | Merge if numerical output shifts beyond tolerance |
| `migration-parity` (new) | Phase 3 PRs | Deletion if old/new differ numerically |
| `no-gas-analysis-core` (new) | Post-Phase 3 | Any re-introduction of deprecated imports |
| `mypy src/` (existing) | Every PR | 0 errors required |
| `ruff` (existing) | Every PR | Linting violations |
| `pytest` (existing) | Every PR | All tests must pass |
| `coverage ≥ 75%` (existing) | Every PR | Below threshold |
| `openapi-compat` (existing) | Every PR | Contract breakage |

---

## 6. What Must Never Change Without an ADR

- LOD/LOQ formula or detection limit convention
- GPR kernel family or number of optimizer restarts
- LSPR signal primary quantity (Δλ) or sign convention
- FOM definition or reference
- Conformal prediction coverage level (currently 95%)
- Allan deviation estimator (currently OADEV)
- Bootstrap iteration count for LOD CI (currently n=2000)

---

## 7. Done Criteria for the Entire Project

The project is done when all of the following are true simultaneously:

1. `docker compose up` starts both services, both health-checks pass, no manual steps required
2. `git clone` + `pip install -e ".[all]"` works on a clean machine (Python 3.10–3.11)
3. A new contributor can identify the canonical pipeline entry point in under 2 minutes of reading
4. Zero imports of `gas_analysis.core` outside `gas_analysis/` (enforced by CI grep gate)
5. All tests pass + science regression suite passes + mypy 0 errors
6. `CHARTER.md` accurately describes the system as it actually works
7. `docs/quickstart/research.md` allows a scientist to complete a calibration session without asking for help

---

## 8. Decisions Deferred

- Multi-tenant / cloud deployment: not in scope for this release; storage and transport
  interfaces are designed to be swappable but not activated
- React frontend scientific tabs: not in scope; Streamlit remains the scientific analysis
  runtime; unification would require 3–6 months of frontend porting work
- Additional hardware drivers beyond CCS200: out of scope; plugin system is the mechanism
