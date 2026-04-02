# Contributing to SpectraAgent

## Local development setup

```bash
git clone <repo-url>
cd spectraagent
python -m venv .venv && .venv\Scripts\activate  # Windows
# source .venv/bin/activate                     # macOS/Linux

pip install -e ".[dev]"          # core + dev tools
pip install -e ".[dev,ml]"       # + PyTorch (CNN classifier)
pip install -e ".[dev,hardware]" # + VISA hardware drivers
```

Verify everything works:

```bash
spectraagent --version
make test-fast
```

## Branch naming

| Type | Pattern | Example |
| --- | --- | --- |
| Feature | `feat/<short-desc>` | `feat/lorentzian-peak-fit` |
| Bug fix | `fix/<short-desc>` | `fix/ws-reconnect-leak` |
| Refactor | `refactor/<short-desc>` | `refactor/session-writer` |
| Docs | `docs/<short-desc>` | `docs/hardware-setup-guide` |
| Tests | `test/<short-desc>` | `test/cli-version-flag` |

Branch from `main`. Keep branches short-lived and focused on one concern.

## Commit message format

```text
<type>: <short imperative summary>  (≤72 chars)

Optional body explaining why (not what — the diff shows what).
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `perf`

Examples:

```text
feat: add --version flag to CLI
fix: seed integration_time_ms from /api/health on first connect
test: add CLI sessions subcommand integration tests
```

## Quality gates

Run before opening a PR:

```bash
make check                # ruff lint + full test suite
make quality-gate         # lint + tests (both lanes) + reliability report
```

Individual tools:

```bash
pytest -q -m "not reliability"         # fast lane only
pytest tests/spectraagent/             # SpectraAgent runtime only
ruff check src/                        # linting
mypy src --no-site-packages            # type checking (required: 0 errors)
```

## Adding a hardware driver

1. Create `your_package/driver.py` implementing `AbstractHardwareDriver`
2. Register it in your package's `pyproject.toml`:

   ```toml
   [project.entry-points."spectraagent.hardware"]
   my_device = "your_package.driver:MyDriver"
   ```

3. `pip install -e your_package/` → it appears in `spectraagent plugins list`
4. Add tests under `tests/spectraagent/drivers/`

## Adding a physics plugin

Same pattern using the `spectraagent.sensor_physics` entry-point group and implementing `AbstractSensorPhysicsPlugin`.

## ADR policy

Create or update an Architecture Decision Record for:

- Changes to the plugin contract (`AbstractHardwareDriver`, `AbstractSensorPhysicsPlugin`)
- Changes to session storage format (CSV columns, JSONL schema)
- Calibration model changes (physics kernel, conformal coverage)
- Quality gate threshold changes (SNR, saturation limits)
- Data contract or Pydantic schema changes

Use `docs/adr/ADR-TEMPLATE.md`.

## Status tracking policy

When a PR changes roadmap progress, deployment readiness, or CI security gates,
update the canonical tracking files in the same PR to prevent status drift:

- `REMAINING_WORK.md`
- `PRODUCTION_READINESS.md`
- `CHANGELOG.md`
- `.github/workflows/security.yml` (if gate logic changed)

## Pull request checklist

- [ ] Tests added or updated for all behaviour changes
- [ ] `make check` passes locally
- [ ] Documentation updated (docstrings, guides, API stubs)
- [ ] ADR added when required (see above)
- [ ] No new `the sensor`/sensor-specific language in generic platform code
- [ ] Risks and assumptions noted in PR description
