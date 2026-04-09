# Release Runbook

This runbook defines the mandatory release process for tag-based releases.
It is designed to match `.github/workflows/release.yml` and reduce operator error.

## Scope

Use this runbook for all `vX.Y.Z` and pre-release tags (`rc`, `alpha`, `beta`).

## Preconditions

- Branch protection enabled on `main` with required checks.
- Local branch is synced with `origin/main`.
- Canonical status files are updated together when release-relevant changes were made:
  - `REMAINING_WORK.md`
  - `PRODUCTION_READINESS.md`
  - `CHANGELOG.md`
  - `.github/workflows/security.yml`

## Release Checklist

1. Confirm working tree is clean.

```bash
git status --short --branch
```

2. Ensure quality and security lanes are green on `main`:

- `Quality Gates`
- `Security Gates`
- `Secret Scan`
- `Deploy Smoke`

3. Verify package version and changelog are aligned.

- `spectraagent.__version__` in package code
- `CHANGELOG.md` has a release section matching the version

4. Run local release preflight (recommended):

```bash
python scripts/validate_workflows.py
python scripts/research_preflight.py --self-check
python scripts/research_integrity_gate.py --self-check
```

5. Create and push an annotated tag:

```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

6. Monitor release workflow:

- Build wheel/sdist
- Artifact hygiene checks
- Checksum generation and verification
- Sigstore signing
- GitHub Release creation
- Optional PyPI publish (if token configured)

7. Verify release assets:

- `dist/*.whl`
- `dist/*.tar.gz`
- `dist/sha256sums.txt`
- `dist/*.sigstore.json`

8. Post-release sanity checks:

- Confirm GitHub Release notes are correct.
- Confirm `pip install` from release artifacts works in a clean venv.
- Record any follow-up items in `REMAINING_WORK.md`.

## Rollback Guidance

If release workflow fails after tag push:

1. Fix the issue in a new commit on `main`.
2. Create a new patch tag (for example `vX.Y.(Z+1)`).
3. Avoid reusing or force-moving existing release tags.

## Ownership

- Primary owner: `@DeepPal`
- Critical files owner routing: `.github/CODEOWNERS`
