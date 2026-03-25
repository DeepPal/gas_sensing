#!/usr/bin/env python3
"""Run project quality checks in a consistent local order.

This script provides a single command to run the same baseline checks used in CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(command: str, required: bool = True) -> int:
    print(f"\n[quality] Running: {command}")
    # Substitute 'pytest' and 'ruff'/'mypy' with venv-relative calls so the
    # correct interpreter is used regardless of PATH on Windows.
    parts = shlex.split(command)
    if parts[0] == "pytest":
        parts = [sys.executable, "-m", "pytest", *parts[1:]]
    elif parts[0] in {"ruff", "mypy"}:
        parts = [sys.executable, "-m", parts[0], *parts[1:]]
    completed = subprocess.run(
        parts,
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        state = "REQUIRED" if required else "ADVISORY"
        print(f"[quality] {state} check failed: {command}")
    else:
        print(f"[quality] Passed: {command}")
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local quality gates")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat mypy as required (default is advisory)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run the smoke test suite (pytest -m smoke)",
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage threshold check (faster for quick local runs)",
    )
    args = parser.parse_args()

    pytest_cmd = (
        "pytest -q --cov=src --cov=gas_analysis --cov-fail-under=60"
        if not args.no_coverage
        else "pytest -q"
    )

    checks = [
        # Style: format first (cheapest), then lint
        ("ruff format --check .", True),
        ("ruff check .", True),
        # Tests
        (pytest_cmd, True),
        # Type checking: src/ is required (clean package); legacy is advisory
        (
            "mypy src --ignore-missing-imports --follow-imports=skip",
            True,  # always required — src/ is the clean typed package
        ),
        (
            "mypy run.py gas_analysis config dashboard tests --ignore-missing-imports",
            args.strict,  # legacy dirs: required only with --strict
        ),
    ]

    if args.smoke:
        checks.append(("pytest -m smoke -v", True))

    failed_required = False
    for cmd, required in checks:
        rc = run_cmd(cmd, required=required)
        if rc != 0 and required:
            failed_required = True

    if failed_required:
        print("\n[quality] FAILED required gates")
        return 1

    print("\n[quality] All quality gates passed")
    if not args.strict:
        print("[quality] Note: mypy is advisory — run with --strict to enforce")
    if not args.smoke:
        print("[quality] Note: smoke tests skipped — run with --smoke to include")
    return 0


if __name__ == "__main__":
    sys.exit(main())
