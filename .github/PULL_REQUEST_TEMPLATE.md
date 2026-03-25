## Summary
<!-- What does this PR do? 1-3 bullet points. -->

-

## Type of change
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Documentation
- [ ] Tests only

## Testing
<!-- How was this tested? -->

- [ ] `make test` passes (339+ tests, 0 failures)
- [ ] `make lint` passes (0 errors in `src/`)
- [ ] Tested on real CCS200 hardware *(if hardware-related)*
- [ ] New tests added for new behaviour

## Checklist
- [ ] Code follows existing patterns (strangler-fig imports, `ExperimentTracker` for MLflow)
- [ ] No hardcoded paths or credentials
- [ ] MLflow runs write to `experiments/mlruns/` (not `mlruns/`)
- [ ] CHANGELOG.md updated *(for user-facing changes)*
