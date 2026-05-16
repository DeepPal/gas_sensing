# ADR-0001: Governance and Quality Baseline

- Status: Accepted
- Date: 2026-03-16
- Deciders: DeepPal Research Team
- Technical Area: governance, quality, deployment

## Context

The repository has strong research logic, but lacked standardized engineering governance comparable to mature open repositories: no CI workflow, no ADR system, and no unified quality gate entrypoint.

## Decision

Adopt a baseline engineering standard composed of:

1. Engineering standards document in docs/ENGINEERING_STANDARDS.md.
2. ADR framework under docs/adr with template and index.
3. CI quality workflow that runs lint, tests, and advisory typing.
4. Local quality gate script for reproducible pre-merge checks.

## Alternatives Considered

1. Continue with ad-hoc scripts only.
2. Introduce strict full-repo type enforcement immediately.
3. Adopt phased standards rollout with CI plus advisory typing.

## Consequences

- Positive: higher reproducibility, clearer decisions, lower regression risk.
- Negative: slightly higher contribution overhead.
- Risk: initial CI failures on legacy code.
- Mitigation: phase typing gate from advisory to required over time.

## Migration / Implementation Notes

- Phase 1 (current): required lint + tests, advisory mypy.
- Phase 2: enforce mypy on touched modules.
- Phase 3: enforce experiment manifest and schema checks in CI.
