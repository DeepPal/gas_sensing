# SpectraAgent Project Charter

> This document is the authoritative north star for all development decisions.
> Every contributor should read it before opening a pull request.
> Every reviewer should check against it before approving.

---

## What This System Does

SpectraAgent is a hardware-agnostic optical spectroscopy platform. It receives
raw wavelength–intensity arrays from any spectrometer, processes them through a
physics-informed signal pipeline, fits calibration models, and returns
uncertainty-quantified concentration measurements.

**Reference deployment:** Localized Surface Plasmon Resonance (LSPR) sensing
with ThorLabs CCS200, detecting volatile organic compounds (VOCs) in air.
The platform is not limited to this configuration.

## What This System Explicitly Does Not Do

- Provide cloud/multi-tenant infrastructure (single-machine deployment only)
- Replace laboratory safety procedures or regulatory submissions
- Certify measurement results without human validation
- Support non-optical sensor modalities (pressure, temperature, etc.)

---

## Correct Definitions for Each Scientific Output

These definitions are fixed. Changing any of them requires a new entry in
`docs/adr/science/` and an update to `tests/science_regression/baselines.json`.

### Limit of Detection (LOD)
- **Method:** IUPAC 2011 — 3σ of the blank signal divided by the calibration slope
- **Formula:** `LOD = 3 × σ_blank / |S|` where S = sensitivity (nm/ppm)
- **CI method:** Parametric bootstrap, n=2000 iterations
- **Reference:** IUPAC Pure Appl. Chem. 2011, 83(5), 1129–1143
- **Not used:** 3× SNR (signal-domain), DIN 32645, signal-to-noise definition

### Calibration Model (GPR)
- **Kernel:** Matérn ν=5/2 (twice-differentiable; physically correct for Langmuir-type response)
- **Optimizer restarts:** 10 (robust hyperparameter search)
- **Uncertainty:** Total std = √(epistemic² + aleatoric²)
- **Feature scaling:** StandardScaler on input features
- **Implementation:** `src/calibration/gpr.py → GPRCalibration`
- **Not used:** RBF kernel (oversmooths Langmuir isotherms)

### Primary Sensor Signal (LSPR)
- **Quantity:** Δλ = λ_analyte − λ_reference (nm)
- **Sign convention:** Redshift (analyte adsorption) = negative Δλ
- **Measurement:** Cross-correlation peak tracking vs reference spectrum
- **Not used:** ΔIntensity as primary signal (secondary only)

### Figure of Merit (FOM)
- **Formula:** FOM = |S| / FWHM (units: ppm⁻¹)
- **S:** Sensitivity from linear region of calibration curve (nm/ppm)
- **FWHM:** Full-width at half-maximum of reference spectrum Lorentzian fit (nm)
- **Reference:** Willets & Van Duyne, Annu. Rev. Phys. Chem. 2007, 58, 267–297

### Prediction Intervals
- **Method:** Conformal prediction (split-CP)
- **Coverage guarantee:** 95% marginal coverage
- **Implementation:** `src/calibration/conformal.py → ConformalCalibrator`
- **Not used:** Bayesian credible intervals (no coverage guarantee for finite samples)

### Temporal Stability (Allan Deviation)
- **Estimator:** Overlapping Allan deviation (OADEV)
- **Output:** τ_opt = integration time minimising OADEV (drift-noise crossover)
- **Implementation:** `src/scientific/allan_deviation.py`

---

## Success Criteria Per Release

| Milestone | Criterion |
|-----------|-----------|
| Phase 1 complete | Both runtimes use identical GPR kernel; science regression passes |
| Phase 2 complete | Zero imports of `spectrometer_webapp` or `n8n` anywhere |
| Phase 3 complete | Zero imports of `gas_analysis.core` outside `gas_analysis/`; all tests pass |
| Phase 4 complete | No single file > 600 lines in dashboard/ or spectraagent/webapp/ |
| Phase 5 complete | `docker compose up` reaches both service health-checks without manual steps |

---

## Scientific Invariants

The following must not change without a new ADR in `docs/adr/science/`:

1. LOD formula (IUPAC 2011, 3σ blank)
2. GPR kernel family (Matérn ν=5/2)
3. Number of GPR optimizer restarts (10)
4. Primary signal quantity (Δλ, not ΔI)
5. Signal sign convention (redshift = negative)
6. FOM definition (|S|/FWHM, Willets & Van Duyne 2007)
7. Conformal prediction coverage level (95%)
8. Allan deviation estimator (OADEV)
9. Bootstrap iterations for LOD CI (n=2000)

---

## Architecture Invariants

These must not change without a design review:

1. `src/` is the single source of truth for all business logic
2. `gas_analysis/` contains only: hardware acquisition drivers and advanced signal (ICA, MCR-ALS)
3. Both runtimes (spectraagent webapp and Streamlit dashboard) import from `src/`
4. `src/public_api.py` is the stable external import surface — no breaking changes without semver bump
5. OpenAPI contract (`contracts/openapi_baseline.json`) requires explicit update for any route change
