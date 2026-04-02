# CI Troubleshooting Guide

This runbook is for the GitHub Actions workflows in `.github/workflows`.
Use it when a gate fails and you need a fast, deterministic recovery path.

## 1) Quality Gates Failed

### Symptoms
- `quality-fast` fails on Ruff, mypy, or pytest
- `workflow-hygiene` fails before tests start

### Local Reproduction
```bash
.venv/Scripts/python.exe scripts/validate_workflows.py
.venv/Scripts/python.exe -m ruff check . --select E9,F63,F7,F82
.venv/Scripts/python.exe -m mypy src --no-site-packages --ignore-missing-imports --disable-error-code import-untyped
.venv/Scripts/python.exe -m pytest -q --tb=short -m "not reliability"
```

### Typical Root Causes
- New workflow `if:` expression wrapped as `${{ ... }}`
- `secrets.*` used directly in `if:` expressions
- Runtime-only imports lacking type narrowing
- Assertions added without handling empty arrays or optional values

### Fix Pattern
1. Fix the first required failure only (Ruff required subset first).
2. Re-run the same command locally until clean.
3. Re-run the full sequence above before pushing.

## 2) Security Gates Failed

### Symptoms
- `dependency-review` fails on pull requests
- `dependency-audit` fails due to vulnerable dependency
- `bandit` flags a source module

### Local Reproduction
```bash
.venv/Scripts/python.exe -m pip install --upgrade pip pip-audit bandit
.venv/Scripts/python.exe -m pip_audit -r requirements.txt
.venv/Scripts/python.exe -m bandit -r src spectraagent dashboard gas_analysis -x tests -lll
```

### Typical Root Causes
- Added package introduces high-severity advisory
- Insecure temporary file or subprocess usage in source code
- Secrets accidentally committed in code/config

### Fix Pattern
1. Prefer upgrading or replacing vulnerable package.
2. If risk is accepted, document justification in PR and track remediation date.
3. Keep suppressions minimal and scoped to exact lines.

## 3) Release Workflow Failed

### Symptoms
- Release build passes but publish step is skipped or fails

### Checks
1. Confirm tag format is valid for release automation.
2. Confirm release token/secret exists in repository settings.
3. Ensure publish logic uses shell-guard checks, not `if:` secret expressions.

## 4) Deploy Smoke Failed

### Symptoms
- `deploy-smoke` fails in smoke tests or API startup checks

### Local Reproduction
```bash
.venv/Scripts/python.exe -m pytest -q --tb=short -m smoke --maxfail=1
.venv/Scripts/python.exe -c "from fastapi.testclient import TestClient; from src.api.main import app; c=TestClient(app); print(c.get('/health').status_code, c.get('/status').status_code)"
```

### Typical Root Causes
- Config loading changed required defaults unexpectedly
- API route import error after refactor
- Startup dependency changed without adding package to project dependencies

## 5) Environment Mismatch (Most Common)

### Symptoms
- Local failures not matching CI
- Import errors referencing unexpected global packages

### Fix Pattern
1. Always run checks with project virtual environment interpreter.
2. Avoid system `python` in diagnostics for this repo.
3. Verify with:
```bash
.venv/Scripts/python.exe --version
```

## 6) Fast Recovery Checklist

1. Run workflow validation first.
2. Run required Ruff subset.
3. Run required mypy for `src` only.
4. Run pytest fast lane.
5. Run preflight and integrity self-checks.
6. Push only after all required gates pass locally.
