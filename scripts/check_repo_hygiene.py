#!/usr/bin/env python3
"""Fail CI if repository contains known hygiene anti-patterns.

Checks:
- Flattened absolute-path artifact filenames accidentally committed.
- Generated runtime artifacts tracked under output/ that should remain local.
"""

from __future__ import annotations

from pathlib import PurePosixPath
import subprocess
import sys

BAD_PREFIXES = (
    "UsersdeeppDesktopChula_WorkPRojectsMain_Research_Chula",
    "cUsersdeeppDesktopChula_WorkPRojectsMain_Research_Chula",
)

# Keep this list focused on regenerable runtime artifacts only.
FORBIDDEN_TRACKED_PATHS = {
    "output/memory/sensor_memory.json",
    "output/test-results/reliability-junit.xml",
    "output/test-results/reliability-nightly-junit-local.xml",
    "output/test-results/reliability-summary.md",
    "output/test-results/reliability-budget.md",
    "UsersdeeppDesktopChula_WorkPRojectsMain_Research_Chulaoutputtest-resultsreliability-junit.xml",
}


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    files = tracked_files()
    violations: list[str] = []

    for path in files:
        pure = PurePosixPath(path)
        path_text = pure.as_posix()

        if path_text.startswith(BAD_PREFIXES):
            violations.append(
                f"flattened-absolute-path artifact is tracked: {path_text}"
            )

        if path_text in FORBIDDEN_TRACKED_PATHS:
            violations.append(
                f"generated runtime artifact is tracked: {path_text}"
            )

    if violations:
        print("[repo-hygiene] FAIL")
        for item in violations:
            print(f"- {item}")
        print(
            "\nFix: remove from git index (git rm --cached ...) and keep ignored via .gitignore."
        )
        return 1

    print("[repo-hygiene] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
