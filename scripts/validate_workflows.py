#!/usr/bin/env python3
"""Validate GitHub workflow hygiene.

Checks for patterns that frequently cause CI failures or editor false positives:
1) Disallow `secrets.` usage inside `if:` expressions.
2) Disallow wrapped `if: ${{ ... }}` style in favor of plain expression style.

This script is intentionally conservative and fast so it can run in pre-commit and CI.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

RE_IF_LINE = re.compile(r"^\s*if:\s*(.+?)\s*$")


def _validate_security_dependency_review(path: Path, content: str) -> list[str]:
    """Enforce critical invariants for the Security Gates dependency review job."""
    if path.name != "security.yml":
        return []

    errors: list[str] = []
    required_snippets = [
        "dependency-review:",
        "if: github.event_name == 'pull_request'",
        "base-ref: ${{ github.event.pull_request.base.sha }}",
        "head-ref: ${{ github.event.pull_request.head.sha }}",
    ]
    for snippet in required_snippets:
        if snippet not in content:
            errors.append(
                f"{path.relative_to(ROOT)}: missing required security dependency-review setting: {snippet}"
            )
    return errors


def _validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    for idx, line in enumerate(lines, start=1):
        m = RE_IF_LINE.match(line)
        if not m:
            continue

        expr = m.group(1).strip()

        # Standardize on native expression style: if: github.ref == 'refs/heads/main'
        if expr.startswith("${{") and expr.endswith("}}"):
            errors.append(
                f"{path.relative_to(ROOT)}:{idx}: avoid wrapped 'if: ${{{{ ... }}}}' style; "
                "use plain expression style instead"
            )

        # `secrets` in if-expressions is brittle and often blocked by linters/tooling.
        if "secrets." in expr:
            errors.append(
                f"{path.relative_to(ROOT)}:{idx}: avoid using secrets in 'if:' expressions; "
                "use a guarded shell step with env vars instead"
            )

    errors.extend(_validate_security_dependency_review(path, content))

    return errors


def main() -> int:
    if not WORKFLOWS.exists():
        print("[workflow-validate] No workflow directory found, skipping.")
        return 0

    files = sorted(list(WORKFLOWS.glob("*.yml")) + list(WORKFLOWS.glob("*.yaml")))
    all_errors: list[str] = []
    for f in files:
        all_errors.extend(_validate_file(f))

    if all_errors:
        print("[workflow-validate] FAILED")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print(f"[workflow-validate] OK: validated {len(files)} workflow file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
