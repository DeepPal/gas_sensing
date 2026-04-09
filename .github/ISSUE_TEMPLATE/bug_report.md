---
name: Bug report
about: Something broken in the pipeline, dashboard, or hardware interface
title: "[BUG] "
labels: bug
assignees: ''
---

## Describe the bug
A clear description of what is wrong.

## Steps to reproduce
1. Run `...`
2. Connect `...`
3. See error

## Expected behaviour
What you expected to happen.

## Actual behaviour
What actually happened. Include the full traceback.

```
paste traceback here
```

## Impact

- [ ] Blocks core workflow
- [ ] Incorrect scientific result risk
- [ ] CI failure only
- [ ] Documentation inconsistency

## Reproducibility

- Frequency: [always / intermittent / once]
- First bad version/commit: [if known]
- Last known good version/commit: [if known]

## Environment
- OS: [e.g. Windows 11]
- Python version: [e.g. 3.11.5]
- Hardware: [CCS200 connected / simulation mode]
- Run mode: [dashboard / CLI sensor / CLI batch]

## Logs and artifacts

Attach any relevant files:

- `output/sessions/<session_id>/pipeline_results.csv`
- `output/sessions/<session_id>/session_meta.json`
- failing CI job link (if applicable)

## Additional context
Any other context (config.yaml snippets, spectrum files, etc.)
