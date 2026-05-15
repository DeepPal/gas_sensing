# SpectraAgent — Developer Contract for Claude

> Read this before touching any code. Every section is load-bearing.
> Derived from CHARTER.md, pyproject.toml, and five science ADRs.
> The authoritative north star is CHARTER.md — this file operationalises it.

---

## What This Codebase Is

A **hardware-agnostic optical spectroscopy platform**: raw wavelength–intensity arrays in,
calibrated uncertainty-quantified concentration measurements out. AI agents assist with
anomaly explanation, experiment narration, and report writing.

**Reference deployment:** LSPR sensing with ThorLabs CCS200, detecting VOCs in air.
The platform is not limited to this configuration.

**Two complementary runtimes — keep them separate:**

| Runtime | Entry point | Audience | Purpose |
|---------|-------------|----------|---------|
| SpectraAgent (FastAPI + React) | `spectraagent start` | Instrument operators | Live acquisition, WebSocket streaming, REST API |
| Streamlit dashboard | `streamlit run dashboard/app.py` | Research scientists | Batch analysis, calibration, publication-ready figures |

Both import from `src/`. They do NOT import from each other.

---

## Environment (Windows)

```powershell
# Always use the project venv — system Python lacks torch and scientific deps
.venv\Scripts\python.exe -m pytest tests/ -q

# OR activate first
.venv\Scripts\Activate.ps1
python -m pytest tests/ -q
```

**System Python (3.10.11) does not have torch.** Any import that touches
`src.models.*`, `gas_analysis.acquisition.*`, or `spectraagent.*` will fail under system Python.

---

## Common Commands

```powershell
# Run full test suite
.venv\Scripts\python.exe -m pytest tests/ -q

# Run only the science regression suite (MUST pass on every PR)
.venv\Scripts\python.exe -m pytest tests/science_regression/ -v

# Run fast CI subset (excludes reliability/integration markers)
.venv\Scripts\python.exe -m pytest tests/ -q -m "not reliability and not integration"

# Lint
.venv\Scripts\python.exe -m ruff check .

# Format
.venv\Scripts\python.exe -m ruff format .

# Type check (advisory — not blocking CI, but keep clean)
.venv\Scripts\python.exe -m mypy src/ --ignore-missing-imports

# Run Streamlit dashboard
.venv\Scripts\python.exe -m streamlit run dashboard/app.py

# Run SpectraAgent API (simulated hardware)
.venv\Scripts\python.exe -m spectraagent start --simulate

# Docker (both runtimes)
docker compose up
```

---

## Architecture Invariants

**Violating these requires a design review before proceeding.**

1. **`src/` is the single source of truth** for all business logic — signal processing,
   calibration, reporting, scientific computations. Both runtimes import from `src/`.

2. **`gas_analysis/` contains only:** hardware acquisition drivers (`gas_analysis/acquisition/`)
   and advanced signal processing (ICA, MCR-ALS). It does NOT contain calibration or pipeline
   logic. `gas_analysis/core/` was deleted in 2026-05 — do not re-create it.

3. **`src/public_api.py` is the stable external import surface.** No breaking changes without
   a semver bump in `pyproject.toml`.

4. **OpenAPI contract** (`contracts/openapi_baseline.json`) must be explicitly updated for any
   route change. The `update_openapi_baseline.py` script generates the new baseline.

5. **`spectraagent/webapp/routes/`** contains the FastAPI route modules. Add new routes there,
   not in `spectraagent/webapp/server.py`.

---

## Scientific Invariants

**These 9 definitions are fixed. Changing any requires a new ADR in `docs/adr/science/`
AND an update to `tests/science_regression/baselines.json`.**

| # | Invariant | Correct value | Wrong value |
|---|-----------|---------------|-------------|
| 1 | LOD formula | IUPAC 2011: `3σ_blank / \|S\|` | 3×SNR, DIN 32645 |
| 2 | LOD CI method | Parametric bootstrap, n=2000 | Analytical CI |
| 3 | GPR kernel | Matérn ν=5/2 | RBF (oversmooths Langmuir) |
| 4 | GPR restarts | 10 | 2 or fewer |
| 5 | Primary sensor signal | Δλ (peak wavelength shift, nm) | ΔIntensity |
| 6 | Signal sign convention | Redshift = negative Δλ | Redshift = positive |
| 7 | FOM | `\|S\| / FWHM` (Willets & Van Duyne 2007) | Any other formula |
| 8 | Prediction intervals | Conformal (split-CP, 95% marginal coverage) | Bayesian credible intervals |
| 9 | Temporal stability | Overlapping Allan Deviation (OADEV) | Standard deviation over time |

**Key implementations:**
- LOD: `src/scientific/lod.py → calculate_lod_3sigma(noise_std, sensitivity_slope)`
- GPR: `src/calibration/gpr.py → GPRCalibration` (NOT `gas_analysis.core.intelligence.gpr`)
- Sensitivity: `src/calibration/sensitivity.py → calculate_sensitivity(concentrations, responses)` → 4-tuple `(slope, intercept, r2, slope_se)`
- Conformal: `src/calibration/conformal.py → ConformalCalibrator`

---

## What NOT to Do

### Never import from deleted modules
```python
# DELETED — will cause ImportError
from gas_analysis.core import ...
from gas_analysis.core.intelligence import ...
from gas_analysis.core.pipeline import ...
from spectrometer_webapp import ...
```

### Never use the RBF kernel for GPR calibration
```python
# WRONG — was used in old gas_analysis.core, do not copy it
from sklearn.gaussian_process.kernels import RBF
gpr = GaussianProcessRegressor(kernel=RBF())

# CORRECT — Matérn ν=5/2, 10 restarts
from src.calibration.gpr import GPRCalibration
gpr = GPRCalibration()
```

### Never treat this platform as sensor-specific
The platform is hardware-agnostic. Do NOT hardcode:
- Gold nanoparticles (Au-MIP, AuNP)
- Any specific substrate chemistry
- Wavelength ranges specific to one sensor model
- "Ethanol" or any specific analyte in non-test code

### Never use `run.py` — it is a retired stub
The CLI entry point is `spectraagent` (pyproject.toml `[project.scripts]`).
`run.py` prints a redirect message and exits 1.

### Never write `gas_analysis.core` imports in new code
The CI grep gate will block the PR:
```yaml
# .github/workflows/quality.yml — workflow-hygiene job
# Blocks any file outside gas_analysis/ that imports gas_analysis.core
```

---

## File Size Budget

Per CHARTER.md Phase 4 success criterion: **no single file > 600 lines** in
`dashboard/` or `spectraagent/webapp/`. If you are growing a file past 400 lines,
stop and assess whether responsibilities should be split.

Exceptions (already exist, do not grow further):
- `dashboard/agentic/tab.py` — Streamlit render monolith; ~3778 lines; all extracted helpers in `steps.py`, `visualizations.py`, `lspr_physics.py`
- `spectraagent/webapp/server.py` — bootstrap + WebSocket; ~1323 lines; route logic in `routes/`

---

## Testing

### Hierarchy
```
tests/
├── science_regression/   # MUST pass on every PR — numerical output guards
├── migration_parity/     # Empty after gas_analysis/core/ deletion — add new parity tests here
├── src/                  # Unit tests for src/ modules
├── spectraagent/         # Unit + integration tests for SpectraAgent
└── dashboard/            # Streamlit tab unit tests
```

### Science regression baselines
`tests/science_regression/baselines.json` is the ground truth. **Do not edit it manually.**
To update baselines after a deliberate scientific change:
1. Write a science ADR in `docs/adr/science/`
2. Re-run `tests/fixtures/generate_fixture.py` to regenerate the fixture
3. Update `baselines.json` with the new values
4. Commit all three files together

### Tolerances
| Metric | Tolerance |
|--------|-----------|
| LOD | ±2% |
| Sensitivity | ±1% |
| GPR posterior std | ±5% |
| Δλ predicted at 1 ppm | ±0.3 nm |

### pytest markers
- `smoke` — fast end-to-end pipeline tests
- `integration` — cross-component or subprocess tests
- `reliability` — long-running or lifecycle stability (nightly only)

---

## Code Style

- **Line length:** 100 (ruff enforces)
- **Formatter:** ruff format (not black, though black is installed as backup)
- **Comments:** Default to none. Only add when the WHY is non-obvious — a hidden constraint, a workaround, a subtle invariant. Never explain WHAT the code does.
- **Type annotations:** Required for all new `src/` code. mypy is advisory but keep it clean.
- **No bare `except:`** — use `except Exception:` or a specific exception type.
- **No `print()` in library code** — use `logging.getLogger(__name__)`.

---

## CI / Workflow Overview

```
quality.yml
├── quality-fast         — ruff, mypy, fast tests (no reliability marker)
├── workflow-hygiene     — grep gate (blocks gas_analysis.core re-introduction)
│                          ← excludes tests/migration_parity/
└── science-regression   — needs quality-fast; runs tests/science_regression/
                           ← blocks reliability-nightly

reliability-nightly.yml  — long-running lifecycle tests (nightly only)
qualification-artifacts.yml — signed evidence pack on version tags or manual dispatch
security.yml             — dependency audit
secret-scan.yml          — trufflehog secret detection
```

### The grep gate rule
Any file **outside `gas_analysis/`** that imports `gas_analysis.core` will block CI.
This prevents regression to the deleted monolith. The check runs in `workflow-hygiene`.

---

## Key Module Map

```
src/
├── calibration/         gpr.py, conformal.py, sensitivity.py, roi_scan.py, multi_roi.py
├── signal/              roi.py, peak.py, transforms.py (ALS/airPLS, SG, wavelet)
├── preprocessing/       quality.py (detect_outliers, SNR, saturation gate)
├── scientific/          lod.py, allan_deviation.py, cross_session.py, lspr_features.py
├── reporting/           metrics.py, environment.py, io.py, plots.py
├── models/              versioning.py, multi_task.py, transfer.py, onnx_export.py
├── features/            lspr_features.py (KineticFeatures), cross_peak_features.py
├── batch/               preprocessing.py, response.py, aggregation.py
├── public_api.py        stable external import surface

dashboard/
├── agentic/             tab.py, steps.py, visualizations.py, lspr_physics.py
├── predict_tab.py       Predict Unknown (Tab 2)
├── science_tab.py       Data-Driven Science (Tab 5)
├── calibration_library.py  persistent validated-model store
├── report_generator.py  HTML report builder

spectraagent/webapp/
├── server.py            FastAPI app bootstrap, WebSocket, Anthropic streaming
├── routes/              acquisition.py, sessions.py, reports.py, agents.py, _models.py

gas_analysis/
├── acquisition/         CCS200 hardware driver (DLL/VISA/Serial)
│   └── ccs200lib/       ctypes wrapper — never modify without hardware testing
└── (everything else was deleted in 2026-05)
```

---

## Hardware Notes (CCS200)

- Calling convention for `getScanData`: pass `buf = (c_double * 3648)()` directly, NOT `byref(buf)`
- Always close with `try/finally` — unclosed handles leave device in bad state
- Error `-1074001152` = `TLCCS_ERROR_SCAN_PENDING`; `-1073807339` = `VI_ERROR_TMO` (stale state)
- Dark noise floor: max ~0.008, min ~−0.004 (no light); RSD ~12% at dark noise is normal
- `pyvisa` is optional — add with `pip install pyvisa` for VISA interface

---

## Release Process

1. All tests pass (including science regression)
2. `scripts/research_preflight.py --self-check` passes
3. `scripts/research_integrity_gate.py --self-check` passes
4. Tag with `vX.Y.Z` → `qualification-artifacts.yml` runs automatically
5. Signed evidence pack uploaded as CI artifact

---

## ADR Log (Science)

| ADR | Topic | Status |
|-----|-------|--------|
| [S001](docs/adr/science/001-lod-definition.md) | LOD — IUPAC 2011 (3σ blank) | Accepted |
| [S002](docs/adr/science/002-gpr-kernel.md) | GPR kernel — Matérn ν=5/2 | Accepted |
| [S003](docs/adr/science/003-primary-signal.md) | Primary signal — Δλ (not ΔI) | Accepted |
| [S004](docs/adr/science/004-conformal-prediction.md) | Prediction intervals — split-CP 95% | Accepted |
| [S005](docs/adr/science/005-conformal-prediction.md) | FOM — \|S\|/FWHM (Willets & Van Duyne) | Accepted |

Next ADR number: **S006**

---

## Contacts & References

- Project charter: [CHARTER.md](CHARTER.md)
- Science ADRs: [docs/adr/science/](docs/adr/science/)
- Research quickstart: [docs/quickstart/research.md](docs/quickstart/research.md)
- Integration quickstart: [docs/quickstart/integration.md](docs/quickstart/integration.md)
- Sensor physics background: [dashboard/agentic/lspr_physics.py](dashboard/agentic/lspr_physics.py)
