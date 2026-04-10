# Operational Reliability and SLO Guide

This guide defines practical reliability objectives for operating SpectraAgent
in advanced research labs and external technical evaluations.

## Purpose

Reliability targets provide a measurable standard for run quality and
operational confidence. They do not replace regulatory validation.

## Service Objectives

### SLO-1: Fast-Lane Test Health

Objective:
- Maintain green fast-lane quality checks for merge/release work.

Target:
- 100% pass rate for required fast-lane CI checks on protected branches.

Evidence source:
- `.github/workflows/quality.yml` fast-lane jobs.

### SLO-2: Reliability-Lane Stability

Objective:
- Keep reliability tests consistently passing within expected runtime envelope.

Target:
- Reliability lane pass rate >= 95% on active development windows.
- Runtime budget checks remain within configured advisory thresholds.

Evidence source:
- Reliability JUnit output and markdown summaries.
- `scripts/check_junit_budget.py` reports.

### SLO-3: API Contract Stability

Objective:
- Prevent accidental breaking changes to public integration endpoints.

Target:
- 0 unapproved contract regressions per release candidate.

Evidence source:
- `scripts/check_openapi_compat.py`.
- API contract tests in `tests/spectraagent/webapp/test_api_contract.py`.

### SLO-4: Integrator Compatibility

Objective:
- Ensure a new technical evaluator can run core compatibility checks quickly.

Target:
- Integrator smoke check passes in less than 5 minutes on supported environments.

Evidence source:
- `python scripts/integration_smoke_check.py`.

## Operational Error Budget Framing

Use an error-budget style view per release cycle:

- Budget unit: failed required CI reliability/contract/integrator checks.
- Budget depletion trigger: repeated failures of the same required gate.
- Action on depletion: block release tagging until remediation is merged.

## Incident Handling Model

Severity levels:

- SEV-1: Contract regression or data-integrity risk.
- SEV-2: Reliability lane instability that blocks release confidence.
- SEV-3: Non-blocking advisory degradations (for example style-only drift).

Minimum incident record:

1. Trigger condition and timestamp.
2. Impacted workflows/artifacts.
3. Root cause summary.
4. Corrective action and verification evidence.

## Release-Readiness Reliability Gate

Before tagging a release candidate, verify:

1. Quality fast-lane required checks are green.
2. Reliability lane required checks are green.
3. OpenAPI compatibility check is green.
4. Integrator smoke check is green.
5. Evidence artifacts are generated and checksummed.

## Recommended Review Cadence

- Weekly: review reliability summaries and flaky test report trends.
- Per release candidate: run full release checklist and evidence bundle generation.
- Monthly: review SLO adherence and adjust thresholds only via ADR.
