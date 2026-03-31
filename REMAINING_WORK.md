# Remaining Work

**Status:** Platform core complete — production tasks + research science gaps remain

---

## Part A: Production Tasks (original 5 — unchanged)

All 5 are scaffolded and ready to implement.  See original implementation
blueprints below.  Estimated effort: 6–8 hours total.

| # | Task | Priority | Hours |
| --- | --- | --- | --- |
| 1 | HTTPS Certificate Generation | Critical | 1 |
| 2 | Reproducibility Manifest | Critical | 1.5 |
| 3 | Backup Manager Integration | High | 1.5 |
| 4 | Uncertainty UI Display | High | 1 |
| 5 | Quality Gates Before Export | Medium | 1 |

Full implementation blueprints for Tasks 1–5 retained below.

---

## Part B: Research Science Gaps (new — identified 2026-03-31)

These are the gaps between "working sensor platform" and "publishable
benchmark tool."  Prioritized by scientific impact.

### B1. Kinetic Features — τ₆₃ and τ₉₅ (HIGH IMPACT)

**What is missing:** The binding rate constant from the transient response curve.
τ₆₃ (63% equilibration time) discriminates analytes better than steady-state
Δλ alone because different molecules bind at different rates even if their
refractive index contributions are similar.

**Why it matters for publication:** Binding kinetics is a standard characterization
in every SPR/LSPR paper.  Not having it is a reviewer comment waiting to happen.

**How to implement:**

1. In `src/features/lspr_features.py`, add `estimate_response_kinetics(delta_lambda_series, timestamps)` → `KineticFeatures(tau_63, tau_95, k_on, k_off)`
2. Use scipy.optimize.curve_fit to fit `f(t) = A·(1 − exp(−t/τ))` to the Δλ time series
3. Store τ₆₃ in `LSPRFeatures.tau_63_s` and add to extended feature vector
4. Record to SensorMemory for cross-session kinetics tracking
5. Wire into SessionAnalyzer output

Estimated effort: 1.5 days

---

### B2. Cross-Session Statistical Comparison (HIGH IMPACT for publication)

**What is missing:** Formal statistical tests comparing sessions to each other.
"This session's LOD is 0.015 ppm vs. last session's 0.018 ppm" is not
publishable without a test of significance.

**Needed tests:**

- Paired t-test for Δλ means across sessions at the same concentration
- Bland-Altman plot for method agreement between sessions
- F-test for variance equality (reproducibility)
- Mann-Whitney U for non-parametric comparison

**How to implement:** New module `src/scientific/cross_session.py`
with `compare_sessions(session_a, session_b) → CrossSessionComparison` dataclass.

**Estimated effort:** 1 day

---

### B3. Selectivity Coefficient Estimation (HIGH IMPACT)

**What is missing:** Quantitative selectivity against known interferents.
Required for §4.1 Specificity in ICH Q2(R1).

**How to implement:**

1. Add `InterferentSession` concept: session where interferent (not target) is the analyte
2. In CalibrationValidationOrchestrator, detect when `gas_label` is an interferent name
3. Compute K = Δλ(interferent at C_B) / Δλ(target at C_A) using SensorMemory history
4. Store selectivity matrix in SensorMemory
5. Report in RESEARCH_HANDBOOK calibration workflow

**Estimated effort:** 1 day

---

### B4. Reference FWHM as Chip Age Predictor (MEDIUM IMPACT)

**What is missing:** Absolute FWHM of the reference peak (in clean carrier gas)
as a chip-level health indicator.  Tracks nanostructure degradation over chip lifetime.

**Current state:** `SensorMemory` stores `reference_peak_nm` (wavelength) but NOT
FWHM of the reference peak.

**How to implement:**

1. In `spectraagent/webapp/server.py` `acq_reference` route, extract FWHM from
   Lorentzian fit and store on `app.state.ref_fwhm_nm`
2. Pass to SensorMemory in `CalibrationObservation` (add `reference_fwhm_nm` field)
3. In SensorHealthAgent, track FWHM trend of reference peak across sessions
4. Add to scorecard as 6th dimension: "Reference FWHM score"
5. Update RESEARCH_HANDBOOK §6.3 Chip Lifetime Estimation

**Estimated effort:** 0.5 days

---

### B5. Temperature / Humidity Co-Registration (MEDIUM IMPACT)

**What is missing:** Environmental metadata per session.  Every LSPR paper
reports room temperature because the sensitivity changes by ~0.02 nm/°C.

**How to implement:**

1. Add `temperature_c` and `humidity_pct` optional fields to SessionWriter metadata
2. In `AcquisitionConfig` Pydantic model, add optional `temperature_c` and `humidity_pct`
3. Include in `session_complete` event data
4. In SensorHealthAgent, flag if sessions were run at different temperatures
   (confounds LOD comparison)
5. In ReportWriter context, include environmental conditions

**Estimated effort:** 0.5 days

---

### B6. Tests for Knowledge Modules (TECHNICAL DEBT)

**What is missing:** Unit tests for:

- `spectraagent/knowledge/sensor_memory.py` — record/read/trend
- `spectraagent/knowledge/analytes.py` — lookup
- `spectraagent/knowledge/protocols.py` — ValidationTracker state machine
- `spectraagent/knowledge/context_builders.py` — output format
- `spectraagent/webapp/agents/sensor_health.py` — scorecard logic
- `spectraagent/webapp/agents/calibration_validator.py` — ICH gap detection
- `src/features/lspr_features.py` — new `_compute_peak_asymmetry`

**Estimated effort:** 1 day

---

### B7. Robustness Experiments (REQUIRED FOR PUBLICATION §4.8)

**What is missing:** Automated robustness testing protocol.  Currently the
researcher must manually vary parameters.  Should be scripted.

**How to implement:**

1. New CLI command: `spectraagent robustness --param integration_time --range 45:55 --steps 3`
2. Runs N sessions at each parameter value
3. Compares LOD and R² across conditions
4. Reports table suitable for publication Methods section

**Estimated effort:** 2 days

---

### B8. Multi-Analyte Mixture Discrimination (FUTURE — HIGH NOVELTY)

**What is missing:** Binary/ternary gas mixture sensing.  The 6-feature
vector is designed to enable this but no training data or model exists yet.

**Approach:** Mixture fraction estimation using NMF (Non-negative Matrix
Factorization) on the differential spectrum.  Each pure analyte's spectral
signature (extracted from SensorMemory) contributes a basis vector.

**Estimated effort:** 3–4 days (plus 2–3 days data collection)

---

## Summary: Science Gap Priority Matrix

| Gap | Impact on publication | Effort | Recommended order |
| --- | --- | --- | --- |
| B1: Kinetic features (τ₆₃) | Very High | 1.5 days | **1st** |
| B6: Tests | Required for merge | 1 day | **2nd** |
| B3: Selectivity coefficient | High | 1 day | **3rd** |
| B2: Cross-session statistics | High | 1 day | **4th** |
| B4: Reference FWHM tracker | Medium | 0.5 days | **5th** |
| B5: Temperature co-registration | Medium | 0.5 days | **6th** |
| B7: Robustness automation | Required for ICH | 2 days | **7th** |
| B8: Mixture discrimination | Novel; future paper | 4+ days | Future |

---

## Part A: Original Production Tasks (Implementation Blueprints)

### Task 1: HTTPS Certificate Generation (1 hour)

**File:** `dashboard/security.py`

```python
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime

def generate_self_signed_cert(cert_path: Path, key_path: Path, days: int = 365) -> bool:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Chula Research"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(u"localhost")]), critical=False)
        .sign(private_key, hashes.SHA256(), default_backend())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()
    ))
    return True
```

Test: `streamlit run dashboard/app.py` → browser HTTPS lock (self-signed warning expected).

---

### Task 2: Reproducibility Manifest (1.5 hours)

**File:** `dashboard/reproducibility.py`

Hook into export pipeline: `log_experiment(session_id, {"analyst": name, "hardware": {...}})` after each CSV export.  Output goes to `experiments/{date}/{session_id}_manifest.json`.

Captures: git commit SHA, Python version, config snapshot, hardware model, analyst name.

---

### Task 3: Backup Manager (1.5 hours)

**File:** `dashboard/backups.py`

`BackupManager.backup_session(session_id, data_dir)` → tar.gz + SHA256 checksum.
`verify_backup_integrity(archive_path)` → bool.
Schedule via Windows Task Scheduler or cron (see `DEPLOY_RESEARCH_LAB.md`).

---

### Task 4: Uncertainty UI (1 hour)

**Files:** `dashboard/experiment_tab.py`, `dashboard/agentic_pipeline_tab.py`

Display GPR predictions as `f"{pred_mean:.1f} ± {pred_std:.1f} ppm"`.
Add `fill_between` CI band to all calibration curves.
Display LOD with bootstrap CI from `SessionAnalysis.lod_ci_lower/upper`.

---

### Task 5: Quality Gates (1 hour)

**Files:** `dashboard/agentic_pipeline_tab.py`

Block export when SNR < 5, R² < 0.99, drift > 2 nm.
Add "Export anyway" checkbox that appends `quality_flags` to metadata JSON.

---

## Validation Checklist (Before Merging Each Item)

```bash
pytest tests/ -v                    # All tests pass
python -m mypy src/ spectraagent/   # 0 errors on typed modules
python run.py --mode simulate       # Smoke test
```
