# Industry Readiness Guide

This guide defines how SpectraAgent is maintained so external engineering teams can evaluate, integrate, and operate it with confidence.

## Positioning

SpectraAgent is built as a research-to-industry bridge:

- Research-first scientific depth.
- Production-style software quality controls.
- Stable integration surface for external systems.

This is not a regulatory approval claim and not a medical or safety-certified product claim.

## Engineering Quality Baseline

Minimum required quality gates for merge:

- Workflow validation and repository hygiene checks.
- Security and secret scanning workflows.
- Fast lane and reliability lane automated tests.
- Type checks on primary source packages.
- Reproducible qualification artifact generation.

## API Stability Expectations

Public API routes under `/api/*` are treated as integration contracts.

- Backward-compatible additive changes are preferred.
- Breaking response-shape changes require an ADR and migration note.
- Contract tests should be updated alongside intentional API changes.

Current approach:

- OpenAPI route presence checks for critical endpoints.
- Response-shape checks for key health and report endpoints.

## Reproducibility and Auditability

For each session, preserve evidence that supports external review:

- Session metadata and event logs.
- Pipeline outputs and reproducibility manifest.
- Deterministic scientific summary artifacts.
- Qualification dossier exports and package bundles.

## Operational Expectations for External Labs

- Use pinned Python environments and documented setup commands.
- Run preflight and integrity checks before qualification runs.
- Keep immutable release notes and checksums for shared artifacts.
- Record environment metadata (temperature/humidity) for every benchmark session.

## What External Engineering Teams Usually Need

- Predictable API behavior and schema evolution discipline.
- Deterministic fallback behavior when optional AI services are unavailable.
- CI evidence demonstrating repeatable quality and reliability.
- Clear release runbook and rollback process.

## Recommended Review Package for Potential Adopters

When sharing with external teams, include:

1. The tagged release and changelog entry.
2. CI run summary (quality, reliability, security).
3. Qualification dossier export and package ZIP.
4. This guide plus the release runbook.
