#!/usr/bin/env python3
"""Enforce scope-tracking consistency from git diff.

Rule:
- If any canonical tracking file changes, all canonical tracking files must be
  updated in the same change set to prevent status drift.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_SCOPE_FILES = {
    "REMAINING_WORK.md",
    "PRODUCTION_READINESS.md",
    ".github/workflows/security.yml",
}
CHANGELOG_FILE = "CHANGELOG.md"


def _git_changed_files(base_ref: str | None) -> list[str]:
    candidates: list[list[str]] = []
    if base_ref:
        candidates.append(["git", "diff", "--name-only", f"{base_ref}...HEAD"])
    candidates.append(["git", "diff", "--name-only", "HEAD~1..HEAD"])

    for cmd in candidates:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    raise RuntimeError("Unable to determine changed files from git diff.")


def _evaluate(changed_files: set[str]) -> tuple[bool, list[str]]:
    changed_core = changed_files & CORE_SCOPE_FILES
    # Changelog-only edits are allowed; full sync is required only when core
    # scope/readiness/security state is modified.
    if not changed_core:
        return True, []

    required = set(CORE_SCOPE_FILES)
    required.add(CHANGELOG_FILE)
    changed_required = changed_files & required
    missing = sorted(required - changed_required)
    if not missing:
        return True, []

    errors = [
        "core scope/readiness/security files were partially updated.",
        f"changed core files: {sorted(changed_core)}",
        f"missing required synchronized files: {missing}",
    ]
    return False, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate scope-tracking consistency")
    parser.add_argument(
        "--base-ref",
        default=None,
        help="Git base ref for diff (for example: origin/main).",
    )
    args = parser.parse_args()

    base_ref = args.base_ref
    if base_ref is None:
        env_base = os.environ.get("GITHUB_BASE_REF")
        if env_base:
            base_ref = f"origin/{env_base}"
        else:
            base_ref = "origin/main"

    changed = set(_git_changed_files(base_ref=base_ref))
    ok, errors = _evaluate(changed)

    if ok:
        print("[scope-compliance] PASS")
        return 0

    print("[scope-compliance] FAIL")
    for err in errors:
        print(f"- {err}")
    print(
        "\nWhen core scope/readiness/security state changes, update together: "
        "REMAINING_WORK.md, PRODUCTION_READINESS.md, .github/workflows/security.yml, CHANGELOG.md"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
