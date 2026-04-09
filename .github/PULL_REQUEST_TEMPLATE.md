## Summary

Describe what changed and why in 2-5 bullets.

-
-

## Scope

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor
- [ ] Documentation
- [ ] Tests only
- [ ] CI / workflow change

## Validation

List exactly what you ran locally and the result.

- [ ] `ruff check . --select E9,F63,F7,F82`
- [ ] `mypy src --no-site-packages --ignore-missing-imports --disable-error-code import-untyped`
- [ ] `pytest -q --tb=short -m "not reliability"`
- [ ] Hardware validation done (if hardware path changed)

## Risk Review

- [ ] No secrets, credentials, or tokens committed
- [ ] No breaking changes to public APIs, CLI flags, or config schema (or they are documented below)
- [ ] Migration notes included (if required)

## Documentation and Release Notes

- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Docs updated (README/guides/architecture where relevant)
- [ ] ADR added/updated when architecture or contracts changed

## Linked Context

- Issue/Task:
- Related PRs:
- Deployment impact:

## Reviewer Focus

Call out the 2-3 files/areas that deserve closest review.

-
-
