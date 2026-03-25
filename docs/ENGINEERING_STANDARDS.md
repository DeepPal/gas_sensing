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
