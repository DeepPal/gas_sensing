# Contributing

## Contribution Flow

1. Create a branch from main.
2. Make focused changes.
3. Run local quality gates.
4. Open pull request with summary, risks, and validation notes.

## Local Checks

Run baseline checks before opening a pull request:

- python scripts/quality_gate.py

Run strict checks including required typing:

- python scripts/quality_gate.py --strict

## ADR Policy

Create or update an ADR for any architectural change:

- pipeline structure
- calibration formulas
- quality thresholds
- data contract or schema changes
- deployment/runtime contract changes

Use docs/adr/ADR-TEMPLATE.md.

## Pull Request Checklist

- tests added or updated for behavior changes
- lint passes
- documentation updated
- ADR added when required
- risks and assumptions documented
