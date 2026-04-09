# Engineering Standards Baseline

This document defines the minimum engineering standards for the gas sensing research platform.

## 1. Goals

- Reproducibility: every run is traceable to code, config, and data inputs.
- Reliability: quality gates catch regressions before merge.
- Scientific integrity: uncertainty, calibration, and quality metrics are first-class outputs.
- Deployability: one command should validate the system in CI and locally.

## 2. Required Repository Practices

- Use Architecture Decision Records (ADRs) for all non-trivial technical decisions.
- Use pull requests for all changes to main.
- Pass CI quality checks before merging.
- Keep changes scoped and documented.
- Preserve backward compatibility for data outputs unless an ADR approves a schema change.

## 3. Data and Experiment Reproducibility

Every experiment result should include:

- code version (git commit hash)
- configuration snapshot
- dataset identity (path + timestamp or manifest)
- calibration version
- quality metrics (SNR, drift, confidence, validity rate)
- uncertainty outputs where available

## 4. Definition of Done for Research Features

A feature is considered done when all conditions below are met:

- unit or integration tests added/updated
- lint checks pass
- mypy check run on touched modules (strictness may be phased)
- documentation updated (README or docs)
- ADR created if the change alters architecture, data schema, or workflow

## 5. Quality Gates

Current baseline gates:

- Ruff lint: required
- Pytest: required
- Mypy: advisory during phase 1, required in phase 2 for touched modules

## 6. ADR Requirement

Create an ADR when changing:

- pipeline structure
- calibration logic or formulas
- quality control thresholds
- data schemas and file formats
- deployment architecture and runtime contracts

Store ADR files in docs/adr.

## 7. Versioning and Releases

- Use semantic versioning for the platform.
- Document notable changes in release notes or changelog.
- Tag major research milestones.

## 8. Security and Safety

- Do not commit secrets or credentials.
- Validate external inputs and file paths.
- Keep dependencies current and pinned where practical.

## 9. Phase Plan

- Phase 1: establish CI, ADRs, local quality gate script.
- Phase 2: raise type-check coverage and add schema validation checks.
- Phase 3: enforce reproducibility manifests for all experiment runs.

## 10. CI Lanes and Required Status Checks

The CI pipeline is split into two lanes with a strict dependency order:

```
quality-fast  →  reliability  →  docker-build  →  docs
```

### Lane definitions

| Job name       | Trigger       | What it runs                                      | Time budget |
|----------------|---------------|---------------------------------------------------|-------------|
| `quality-fast` | Every push/PR | Ruff lint, mypy (src/), fast pytest suite         | ~2–3 min    |
| `reliability`  | After fast ✅  | Reliability-marked pytest suite, JUnit XML        | ≤45 s total |
| `docker-build` | After both ✅  | Docker image build                                | ~3–5 min    |
| `docs`         | After fast ✅  | MkDocs build                                      | ~1 min      |

A nightly workflow (`reliability-nightly.yml`) re-runs the reliability suite with
**enforced** runtime budgets (45 s total, 12 s per test) and uploads a full
markdown report as a GitHub Actions artifact.

### Branch protection settings (recommended for `main`)

Navigate to **Settings → Branches → Branch protection rules** and configure:

1. **Require a pull request before merging** — avoid direct pushes to `main`.
2. **Require status checks to pass before merging** — add both required checks:
	- `quality-fast`
	- `reliability`
3. **Require branches to be up to date before merging** — ensures CI runs on
	the latest merge base (prevents stale-green PRs).
4. **Do not allow bypassing the above settings** — enforce even for admins to
	preserve scientific reproducibility guarantees.

```
Required checks (exact job names from .github/workflows/quality.yml):
  ✅  quality-fast
  ✅  reliability
```

### Local quality gate (mirrors CI)

Run the full gate before pushing:

```bash
# Fast lane only (lint + types + tests, ~1 min)
python scripts/quality_gate.py --lane fast

# Full gate including reliability suite + budget check
python scripts/quality_gate.py --lane all --reliability-report --enforce-reliability-budget

# Or via Make:
make quality-gate
```

The gate exits non-zero if any check fails, matching CI behaviour.

### Runtime budget thresholds

| Scope        | Budget   | Mode                  |
|--------------|----------|-----------------------|
| Total suite  | 45 s     | Enforced nightly, advisory on PR |
| Single test  | 12 s     | Enforced nightly, advisory on PR |

Budget reports are uploaded as GitHub Actions artifacts and appended to the
GitHub Step Summary for every reliability run.
