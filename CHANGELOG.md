# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed — Coverage gate policy alignment (2026-04-10)
- `.github/workflows/quality.yml` — aligned fast-lane coverage enforcement to `--cov-fail-under=75` to match project coverage policy in `pyproject.toml`
- `scripts/quality_gate.py` — aligned local `--coverage` behavior with CI by using `src` coverage scope and the same 75% threshold
- `tests/test_quality_gate.py` — added assertions to prevent future drift in coverage threshold/scope

### Fixed — LOD typing robustness and full `src` mypy clean pass (2026-04-10)
- `src/scientific/lod.py` — hardened WLS field extraction and Breusch-Pagan p-value handling with explicit numeric guards, eliminating optional/`object` typing hazards while preserving scientific behavior
- Validation: `mypy src --no-site-packages --ignore-missing-imports --disable-error-code import-untyped` now reports clean over 109 source files; `pytest -q tests/test_lod.py` passed

### Added — Industry evaluation bundle regression tests (2026-04-10)
- `tests/test_build_industry_evaluation_bundle.py` — added focused tests for `scripts/build_industry_evaluation_bundle.py` covering success and failure paths, manifest status semantics, summary content, and ZIP artifact composition

### Added — Deterministic scientific reporting fallback (2026-04-09)
- `src/reporting/scientific_summary.py` — new deterministic session-summary formatter that turns session metadata and computed analysis into a scientist-facing markdown report with publication-readiness checks, audit trail, and recommended next experiments
- `spectraagent/webapp/server.py` — `/api/reports/generate` now falls back to the deterministic scientific summary when Claude/ReportWriter is unavailable, instead of failing with 503
- `spectraagent/webapp/server.py` — completed sessions now automatically write `{session_id}_scientific_summary.md` and `{session_id}_scientific_summary.json` beside the reproducibility manifest, and research package ZIP exports now include them
- `tests/test_scientific_summary.py` and `tests/spectraagent/webapp/test_server.py` — added coverage for deterministic report content and fallback behavior

### Added — Research evidence pack orchestration (2026-04-09)
- `scripts/build_research_evidence_pack.py` — new single-command builder that orchestrates benchmark evidence, blinded replication manifest, and qualification dossier generation, then emits a checksummed evidence index (`research_evidence_index_*.json/.md`)
- `.github/workflows/qualification-artifacts.yml` — switched artifact generation to the unified evidence-pack command for reproducible CI behavior
- `Makefile` — added `make evidence-pack` for local lab workflows
- `docs/guides/RELEASE_RUNBOOK.md` — added local evidence-pack preflight step before tagging releases

### Added — Dependency automation and release runbook (2026-04-09)
- `.github/dependabot.yml` — enabled weekly Dependabot updates for `pip`, GitHub Actions, and frontend `npm` dependencies
- `docs/guides/RELEASE_RUNBOOK.md` — added a practical release checklist aligned with tag-based release workflow and artifact verification requirements

### Changed — Governance ownership and quality ratchet (2026-04-09)
- `.github/CODEOWNERS` — tightened ownership routing for release/governance-critical files (`pyproject.toml`, `Dockerfile`, `docker-compose.yml`, status docs, scripts, dependabot config)
- `pyproject.toml` — raised coverage threshold from 70 to 75
- `CONTRIBUTING.md` — linked release process to the release runbook

### Added — Repository hygiene + governance hardening (2026-04-09)
- `.github/workflows/secret-scan.yml` — added Gitleaks-based secret scanning on PRs and pushes to `main`
- `scripts/check_repo_hygiene.py` — added CI guard that fails on flattened absolute-path artifacts and forbidden generated runtime files tracked in git
- `scripts/check_status_sync.py` — added CI guard enforcing canonical status-tracking references across `README.md`, `CONTRIBUTING.md`, `REMAINING_WORK.md`, and `PRODUCTION_READINESS.md`

### Changed — Quality gate strictness (2026-04-09)
- `.github/workflows/quality.yml` — `workflow-hygiene` now runs repository-hygiene and status-sync checks
- `.github/workflows/quality.yml` — promoted legacy mypy lane to required (removed advisory `continue-on-error`)

### Fixed — Repo index contamination (2026-04-09)
- Removed malformed duplicate frontend tree tracked under flattened absolute-path prefix (`cUsersdeeppDesktop...`)
- Removed accidental flattened test-results artifact tracked under `UsersdeeppDesktop...`
- Stopped tracking generated runtime artifacts under `output/memory/` and `output/test-results/` and tightened ignore rules in `.gitignore`

### Added — CI reliability, release integrity, and diagnostics (2026-04-02)
- `.github/workflows/release.yml` — added Sigstore signing/verification coverage for release artifacts and hardened provenance checks in the release lane
- `.github/workflows/quality.yml` — added flaky-test detection/reporting lane and surfaced reliability diagnostics in CI outputs
- `scripts/detect_flaky_tests.py` — new reliability utility to detect repeated unstable tests from JUnit history and emit markdown/JUnit-friendly summaries
- `scripts/generate_pr_comment.py` — new PR diagnostics helper that turns CI outcomes into actionable troubleshooting comments
- `MANIFEST.in`, `pyproject.toml` — strengthened packaging hygiene to keep generated/non-distribution artifacts out of release wheels/sdists
- `docs/guides/` and CI troubleshooting docs — expanded guidance for recurring workflow failures, release checks, and deployment smoke validation

### Fixed — Deployment contract and dashboard auth hardening (2026-04-01)
- `Dockerfile` — fixed editable-install build path by copying the application tree before `pip install -e`, added an explicit `api` target, and kept `spectraagent` as the live-platform runtime target
- `docker-compose.yml` — aligned the default deployment with the actual product architecture: `spectraagent` now runs on port `8765` with `/api/health`, while `dashboard` points users at the live platform URL via `SPECTRAAGENT_BASE_URL`
- `dashboard/auth.py` — removed the hardcoded fallback password, added PBKDF2-SHA256 verifier support, added a password-management CLI, and kept plaintext env/file support only for compatibility
- `run_dashboard_secure.bat`, `run_dashboard_secure.sh` — fail closed when no password is configured and surface the hashed-password setup path
- `DEPLOY_RESEARCH_LAB.md`, `PRODUCTION_READINESS.md` — synced deployment and security docs to the actual implementation

### Added — Research integrity gate and replay verification (2026-04-01)
- `dashboard/reproducibility.py` — manifests now include per-artifact SHA256 checksums for tamper-evident replay
- `scripts/replay_session.py` — verifies a session manifest against on-disk artifact checksums
- `scripts/research_integrity_gate.py` — validates manifest schema + checksums and includes a built-in tamper-detection self-check mode
- `tests/test_reproducibility_manifest.py` — added checksum generation and tamper-detection coverage
- `.github/workflows/release.yml` — release pipeline now executes integrity self-check after test lane

### Added — Uncertainty UI, export quality gates, and robustness CLI (2026-04-01)
- `dashboard/agentic_pipeline_tab.py`:
  - Added calibration-curve 95% CI band (GPR-based with linear fallback)
  - Updated live inference metrics to display concentration estimate as
    `mean ± 1σ` in ppm
  - Added hard export quality gates for `R² >= 0.95`, `SNR >= 3`, and
    replicate drift `<= 2 nm`, with explicit override checkbox
  - Added export metadata sidecar JSON including `quality_flags` and
    override traceability for report generation
- `spectraagent/commands/robustness.py`: new `RobustnessRunner` command module
  for scripted robustness sweeps and publication-ready LOD/R² comparison tables
- `spectraagent/__main__.py`: new `spectraagent robustness` CLI command with
  `--param`, `--range`, `--steps`, `--runs`, `--dataset-dir`, and `--output-csv`

### Added — Codemap audit & B1 status correction (2026-04-01)
- Confirmed `estimate_response_kinetics` / `KineticFeatures` (τ₆₃, τ₉₅) already
  implemented in `src/features/lspr_features.py` and wired into
  `src/inference/session_analyzer.py` + `SensorMemory`; updated
  `REMAINING_WORK.md` B1 status to ✅ IMPLEMENTED and priority matrix accordingly

### Added — Governance & tracking consistency (2026-04-01)
- `.github/workflows/security.yml` — dedicated Security Gates workflow:
  CodeQL (Python + JavaScript), `pip-audit` dependency vulnerability scan,
  Bandit source scan, and PR dependency diff review (`dependency-review-action`)
- Status-document truth sync to reduce planning drift between docs and code:
  `PRODUCTION_READINESS.md` and `REMAINING_WORK.md` updated to reflect
  completed production/security/science work items
- Canonical status-tracking references documented in `README.md` and
  `CONTRIBUTING.md` so future contributors/agents update the same sources

### Added — Production hardening & scientific completeness (2026-03-30)
- `src/calibration/batch_reproducibility.py` — batch sensor QC: pooled LOD, inter-sensor
  RSD, accept/reject verdict with configurable thresholds
- `src/calibration/selectivity.py` — selectivity report from calibration data: IUPAC K values,
  cross-reactivity coefficients, selectivity flag (excellent/good/poor)
- `src/reporting/publication.py` — publication-quality figure generation (Nature/ACS journal style)
- `tests/integration/test_production_compliance.py` — 37 integration tests: full IUPAC triad,
  response kinetics, selectivity, batch reproducibility, public API completeness, audit trail
- LOB (Limit of Blank) added to IUPAC triad — `lob`, `lob_ci_low`, `lob_ci_high`
- Bootstrap CI on LOD/LOQ (2 000 iterations) — `lod_ci_low/high`, `loq_ci_low/high`
- Limit of Linearity (LOL) computed from 5+ calibration points using Mandel F-test
- Methods audit trail in `SessionAnalyzer` — records sigma source, method name, timestamp,
  references (IUPAC 1995, ICH Q2(R1)), git commit hash
- Lorentzian peak fit in `LSPRPhysicsPlugin` — replaces centroid estimate; result cached
  across frames via `compute_reference_cache()`
- Leave-one-out coverage check for conformal predictor
- `src/io/hdf5.py` — HDF5 archival for spectral datasets (write + read round-trip)
- `src/spectrometer/` — hardware abstraction layer: `AbstractSpectrometer`, `SpectralFrame`,
  `SpectrometerRegistry`, `SimulatedSpectrometer` (LSPR/fluorescence/absorbance modes),
  `CCS200Adapter`
- `src/calibration/pls.py` — PLS calibration with LOOCV and VIP scores
- `tests/test_spectrometer.py`, `tests/test_hdf5.py`, `tests/test_pls_calibration.py`,
  `tests/test_publication_figures.py`, `tests/test_reproducibility_manifest.py` — 60+ new tests

### Fixed — Production hardening (2026-03-30)
- `gas_analysis/acquisition/ccs200_realtime.py` — Unicode symbols (`✓`, `✗`, `⚠`) in print
  statements raised `UnicodeEncodeError` on Windows cp1252 console, silently triggering
  simulation fallback even when hardware was connected; replaced with ASCII equivalents
- `gas_analysis/acquisition/ccs200_realtime.py` — `_last_sample_time` was never updated in
  acquisition loop; health watchdog silence-check was permanently disabled
- `spectraagent/drivers/thorlabs.py` — `set_integration_time_ms()` only updated the sleep
  timing attribute; never propagated to the hardware DLL via `spec.set_integration_time()`
- `spectraagent/webapp/frontend/src/App.tsx` — WebSocket `onclose` callback fired
  asynchronously after `ws.close()` in cleanup, scheduling a reconnect timer on an unmounted
  component; fixed with `unmounted` flag + `clearTimeout` in both WebSocket effects
- `spectraagent/webapp/frontend/src/App.tsx` — simulation badge checked only `health.simulate`
  (CLI flag); silent hardware fallback sets `hardware="Simulation"` with `simulate=false`;
  badge now checks both
- `tests/spectraagent/webapp/agents/test_claude_agents.py` — all `messages.create` assertions
  updated to `messages.stream` after production code switched to streaming API

### Added — SpectraAgent live platform (2026-03-26 to 2026-03-29)
- `spectraagent/` package — full agentic spectroscopy server:
  - `__main__.py` — Typer CLI (`spectraagent start`, `spectraagent plugins list`)
  - `webapp/server.py` — FastAPI app factory with all HTTP + WebSocket routes
  - `webapp/agent_bus.py` — `AgentBus`: thread-safe bridge (call_soon_threadsafe + asyncio queues)
  - `webapp/session_writer.py` — `SessionWriter`: crash-safe per-frame CSV + metadata JSON
  - `webapp/agents/quality.py` — `QualityAgent`: SNR check + saturation hard-gate
  - `webapp/agents/drift.py` — `DriftAgent`: rolling peak-shift trend detection
  - `webapp/agents/calibration.py` — `CalibrationAgent`: accumulates (conc, shift) points
  - `webapp/agents/planner.py` — `ExperimentPlannerAgent`: wraps `BayesianExperimentDesigner`
  - `webapp/agents/claude_agents.py` — `AnomalyExplainer`, `ExperimentNarrator`,
    `DiagnosticsAgent`, `ReportWriter`, `ClaudeAgentRunner`
  - `drivers/` — `AbstractHardwareDriver`, `ThorlabsDriver`, `SimulationDriver`, `validation.py`
  - `physics/` — `AbstractSensorPhysicsPlugin`, `LSPRPhysicsPlugin`
  - `config.py` — TOML config loader (`spectraagent.toml`)
- `spectraagent/webapp/frontend/` — React + TypeScript + Vite frontend:
  - Live spectrum chart (WebSocket), agent event feed, hardware badge
  - Session start/stop controls, reference capture, calibration panel
  - Ask Claude (SSE streaming), auto-explain toggle
- `spectraagent.toml` — default configuration file
- Entry-points in `pyproject.toml`: `spectraagent.hardware` and `spectraagent.sensor_physics`

### Added — Scientific hardening (2026-03-20 to 2026-03-25)
- `src/calibration/conformal.py` — split conformal prediction with normalised scores;
  provable coverage guarantee; `ConformalCalibrator` wired into `RealTimePipeline`
- `src/calibration/physics_kernel.py` — `PhysicsInformedGPR` with Langmuir isotherm mean
  function; Mandel F-test gate suppresses Langmuir on linear data
- `src/calibration/active_learning.py` — `BayesianExperimentDesigner` with logspace
  max-variance acquisition; replaces fixed concentration grids
- `src/inference/session_analyzer.py` — `SessionAnalyzer`: post-session LOD/LOQ/T90/T10/
  drift rate/linearity; runs automatically on `POST /acquisition/stop`
- `src/inference/realtime_pipeline.py` — `RealTimePipeline`: 4-stage pipeline wired into
  per-frame hot path; emits `concentration_ppm`, `ci_low`, `ci_high`, `peak_shift_nm`,
  `gas_type`, `confidence_score` to WebSocket broadcast
- `src/public_api.py` — stable commercial import facade (PEP 561 typed package)
- `src/py.typed` — PEP 561 marker

### Added — Commercial-grade analysis quality (2026-03)
- `config/MODEL.yaml` — versioned ML hyperparameter registry (CNN architecture, GPR kernel,
  multi-task future path, production model paths); merged from stale `configs/` directory
- `tests/test_smoke.py` — 7 end-to-end smoke tests covering config load, preprocessing,
  peak detection, GPR fit/predict round-trip, stable-plateau aggregation, LiveDataStore
  thread safety, and ONNX numerical validation; run with `pytest -m smoke`
- `config/config.yaml` — added `mlflow`, `api`, and `agents` sections previously only in
  the now-deleted `configs/` directory; single source of truth for all configuration
- Dashboard: stable-plateau representative spectra via `build_canonical_from_scan()` —
  replaces naive last-10-frame average; uses temporal tail-gating + longest stable sub-block
- Dashboard: LSPR-correct Δλ response variable — when a reference spectrum is loaded,
  all calibration, sensitivity, LOD and isotherm calculations use Δλ (nm) not peak
  intensity (a.u.), matching IUPAC definitions for optical sensors
- Dashboard: residual-based LOD noise (σ from fit residuals, ddof=2) — more defensible
  than spectral baseline std per IUPAC 1995
- Dashboard: MCR-ALS spectral deconvolution panel — resolves mixture spectra into
  pure-component spectra (S) and abundance profiles (C) with non-negativity constraints
- Dashboard: HTML calibration report export — includes gas type, date, sensitivity, R²,
  LOD, LOQ, isotherm model, and instrument metadata
- Dashboard: CNN → ONNX export button with numerical validation (max delta check)

### Changed — Commercial-grade analysis quality (2026-03)
- Selectivity K ratios now use peak wavelength (nm/ppm) sensitivities — dimensionless
  ratios as per IUPAC definition; previously used peak intensity (a.u./ppm) which made
  cross-gas K values unit-dependent and physically incorrect
- `sensor_app/` directory removed — canonical code consolidated in `src/inference/`;
  all fallback `try/except ImportError` shims removed from `run.py`,
  `dashboard/sensor_dashboard.py`, and `src/api/main.py`
- `configs/` directory removed — merged into `config/` (canonical); `MODEL.yaml` moved,
  `mlflow`/`api`/`agents` sections merged into `config/config.yaml`
- `dashboard/integrated_dashboard.py` and `dashboard/realtime_monitor.py` removed —
  superseded by `dashboard/app.py` and `dashboard/sensor_dashboard.py`
- `pyproject.toml` — package discovery, ruff isort `known-first-party`, and coverage
  source updated to remove stale `configs*` and `sensor_app` entries; `gas_analysis*`
  added to setuptools package discovery; `smoke` pytest marker registered
- `.github/workflows/quality.yml` and `scripts/quality_gate.py` — `sensor_app` removed
  from mypy target paths
- `.pre-commit-config.yaml` — `sensor_app` removed from mypy paths; `src` added
- `.gitignore` — `output/batch/`, `output/ethanol_ui/`, `output/ml_dataset/`,
  `output/summary_*.json` added to prevent generated outputs from being committed

### Fixed — Commercial-grade analysis quality (2026-03)
- `src/api/main.py` — `from configs.config_loader` → `from config.config_loader`
  (would have caused a silent `ModuleNotFoundError` at FastAPI server startup)
- `run.py` sensor mode — removed dead `except ImportError` fallback to deleted
  `sensor_app.orchestrator` that would crash on `--mode sensor`

### Added
- `src/` package — strangler-fig migration of core pipeline into typed, tested modules
- `src/agents/drift.py` — DriftDetectionAgent with rolling trend + offset threshold alerts
- `src/agents/training.py` — TrainingAgent with auto-retrain on drift / R² decay / volume triggers
- `src/agents/quality.py` — QualityAssuranceAgent (SNR, saturation gating)
- `src/models/cnn.py` — GasCNN (1D conv) + CNNGasClassifier with MC Dropout uncertainty
- `src/models/onnx_export.py` — ONNX export, numerical validation, OnnxInferenceWrapper
- `src/calibration/gpr.py` — GPRCalibration: Gaussian Process Δλ → ppm with uncertainty
- `src/scientific/lod.py` — IUPAC LOD (3σ), LOQ (10σ), sensitivity, sensor_performance_summary
- `src/training/cross_gas_eval.py` — Leave-one-gas-out (LOGO) cross-validation + MLflow
- `src/training/ablation.py` — Preprocessing ablation study (6 configs, 5-fold GPR CV)
- `src/inference/orchestrator.py` — SensorOrchestrator wiring drift + training agents into live pipeline
- `src/schemas/spectrum.py` — Pydantic `SpectrumReading` + `PredictionResult` data contracts
- `src/batch/aggregation.py` — Stable-block detection, canonical spectrum selection
- 339-test suite across 17 test files (0 failures)
- `Makefile` for common developer commands (`make test`, `make lint`, `make dashboard`, etc.)
- `LICENSE` (MIT)
- `CHANGELOG.md`
- `.github/ISSUE_TEMPLATE/` — bug report + feature request templates
- `.github/PULL_REQUEST_TEMPLATE.md`

### Changed
- MLflow: all scripts now use `ExperimentTracker` wrapper — no more raw `mlflow.*` calls scattered across files
- MLflow tracking URI unified to `experiments/mlruns/` across all scripts
- MLflow run names now include timestamps (no duplicate/overwrite runs)
- MLflow now logs result JSON artifacts alongside metrics
- `TrainingAgent` MLflow: adds `gpr_r2_delta` / `cnn_acc_delta` metrics; logs model files when promoted
- `log_figure()` failure level raised from DEBUG → WARNING
- Coverage threshold raised from 40% → 60%
- Style: full `src/` package upgraded to modern Python 3.9+ annotations (`list[x]`, `x | None`)
- `scripts/quality_gate.py` and CI workflow now include `src/` in mypy targets

### Fixed
- Real-time hardware pipeline: `connect()` not called before `start()`, `.wavelengths` attribute used correctly, unix float timestamps normalised to `datetime`
- GPR ConvergenceWarning noise in test output suppressed via `filterwarnings`
- `asyncio_mode = "auto"` removed from pytest config (pytest-asyncio not installed)
- NumPy 2.0 `np.trapz` deprecation → `getattr(np, "trapezoid", np.trapz)`

---

## [2.0.0] — 2025-02

### Added
- Agentic Pipeline dashboard tab (5-agent automation workflow)
- Reference spectrum subtraction for Δλ LSPR signal extraction
- Multi-ROI fusion with hybrid R²/slope-to-noise metric
- ICA spectral decomposition + MCR-ALS
- FastAPI REST inference server (`serve.py`)
- Docker + docker-compose support

### Changed
- Pipeline restructured into 4 explicit stages (preprocess → feature → calibrate → QC)
- LiveDataStore made thread-safe singleton
- Dashboard expanded to 4 tabs

---

## [1.0.0] — 2024-11

### Added
- Initial release: batch CSV ingestion, ALS baseline, Savitzky-Golay, GPR calibration
- Streamlit dashboard (Batch Analysis tab)
- ThorLabs CCS200 DLL + VISA acquisition drivers
- T90/T10 kinetics analysis
