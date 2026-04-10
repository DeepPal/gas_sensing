#!/usr/bin/env python3
"""Run project quality checks in a consistent local order.

This script provides a single command to run the same baseline checks used in CI.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TEST_RESULTS_DIR = ROOT / "output" / "test-results"
RUFF_REQUIRED_CHECK = "ruff check . --select E9,F63,F7,F82"
RUFF_ADVISORY_CHECK = "ruff check ."
MYPY_SRC_REQUIRED_CHECK = (
    "mypy src --no-site-packages --ignore-missing-imports "
    "--disable-error-code import-untyped --python-version 3.11"
)
MYPY_LEGACY_ADVISORY_CHECK = (
    "mypy run.py gas_analysis config dashboard tests "
    "--ignore-missing-imports --no-site-packages --disable-error-code import-untyped "
    "--namespace-packages --explicit-package-bases --python-version 3.11"
)
FAST_LANE_COVERAGE_THRESHOLD = 75


def _pytest_base_command(*, with_coverage: bool, marker: str) -> str:
    parts = ["pytest", "-q", "--tb=short", "-m", f'"{marker}"']
    if with_coverage:
        parts.extend(
            [
                "--cov=src",
                f"--cov-fail-under={FAST_LANE_COVERAGE_THRESHOLD}",
            ]
        )
    return " ".join(parts)


def build_checks(args: argparse.Namespace) -> list[tuple[str, bool]]:
    checks: list[tuple[str, bool]] = [
        ("python scripts/validate_workflows.py", True),
        (RUFF_REQUIRED_CHECK, True),
        (RUFF_ADVISORY_CHECK, False),
    ]

    if args.format_check:
        checks.append(("ruff format --check .", True))

    if args.lane in {"fast", "all"}:
        checks.append(
            (
                _pytest_base_command(
                    with_coverage=args.coverage,
                    marker="not reliability",
                ),
                True,
            )
        )

    if args.lane in {"reliability", "all"}:
        if args.reliability_report or args.enforce_reliability_budget:
            checks.append((f"python -c \"from pathlib import Path; Path(r'{TEST_RESULTS_DIR}').mkdir(parents=True, exist_ok=True)\"", True))
        reliability_cmd = _pytest_base_command(with_coverage=False, marker="reliability")
        if args.reliability_report or args.enforce_reliability_budget:
            reliability_cmd += (
                " --durations=20"
                f" --junitxml={TEST_RESULTS_DIR / 'reliability-junit.xml'}"
            )
        checks.append((reliability_cmd, True))

        if args.reliability_report or args.enforce_reliability_budget:
            checks.append(
                (
                    f"python scripts/summarize_junit.py --junit \"{TEST_RESULTS_DIR / 'reliability-junit.xml'}\" "
                    f"--output \"{TEST_RESULTS_DIR / 'reliability-summary.md'}\" "
                    "--title \"Reliability Local Summary\" --top-n 10",
                    True,
                )
            )

        if args.enforce_reliability_budget:
            checks.append(
                (
                    f"python scripts/check_junit_budget.py --junit \"{TEST_RESULTS_DIR / 'reliability-junit.xml'}\" "
                    f"--output \"{TEST_RESULTS_DIR / 'reliability-budget.md'}\" "
                    "--title \"Reliability Local Budget\" "
                    f"--max-total-seconds {args.max_total_seconds} "
                    f"--max-case-seconds {args.max_case_seconds}",
                    True,
                )
            )

    checks.extend(
        [
            (
                MYPY_SRC_REQUIRED_CHECK,
                True,
            ),
            (
                MYPY_LEGACY_ADVISORY_CHECK,
                args.strict,
            ),
        ]
    )

    if args.smoke:
        checks.append(("pytest -m smoke -v", True))

    return checks


def run_cmd(command: str, required: bool = True) -> int:
    print(f"\n[quality] Running: {command}")
    # Substitute 'pytest' and 'ruff'/'mypy' with venv-relative calls so the
    # correct interpreter is used regardless of PATH on Windows.
    parts = shlex.split(command)
    if parts[0] == "pytest":
        parts = [sys.executable, "-m", "pytest", *parts[1:]]
    elif parts[0] == "python":
        parts = [sys.executable, *parts[1:]]
    elif parts[0] in {"ruff", "mypy"}:
        parts = [sys.executable, "-m", parts[0], *parts[1:]]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        parts,
        cwd=ROOT,
        env=env,
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
        "--lane",
        choices=["all", "fast", "reliability"],
        default="all",
        help="Which pytest lane to run (default: all)",
    )
    parser.add_argument(
        "--format-check",
        action="store_true",
        help="Also require ruff format --check . locally (stricter than CI)",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help=(
            "Also require the fast-lane coverage threshold locally "
            f"({FAST_LANE_COVERAGE_THRESHOLD}%)"
        ),
    )
    parser.add_argument(
        "--reliability-report",
        action="store_true",
        help="Generate reliability JUnit and markdown summary outputs when running the reliability lane",
    )
    parser.add_argument(
        "--enforce-reliability-budget",
        action="store_true",
        help="Fail the local quality gate if reliability runtime budgets are exceeded",
    )
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=45.0,
        help="Maximum allowed total reliability runtime when budget enforcement is enabled",
    )
    parser.add_argument(
        "--max-case-seconds",
        type=float,
        default=12.0,
        help="Maximum allowed single reliability test runtime when budget enforcement is enabled",
    )
    parser.add_argument("--no-coverage", dest="coverage", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(coverage=False)
    args = parser.parse_args()

    checks = build_checks(args)

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
        print("[quality] Note: mypy is advisory - run with --strict to enforce")
    if not args.smoke:
        print("[quality] Note: smoke tests skipped - run with --smoke to include")
    if not args.coverage:
        print("[quality] Note: fast-lane coverage threshold skipped - run with --coverage to enforce")
    if not args.format_check:
        print("[quality] Note: formatting gate skipped - run with --format-check to enforce")
    if args.lane == "fast":
        print("[quality] Reliability lane skipped - use --lane reliability or --lane all to include it")
    elif args.lane == "reliability":
        print("[quality] Fast lane skipped - use --lane fast or --lane all to include it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
