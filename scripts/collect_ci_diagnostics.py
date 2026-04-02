#!/usr/bin/env python3
"""Collect compact environment diagnostics for CI failures.

Designed to run only on failure paths so maintainers can triage quickly without
rerunning jobs just to inspect interpreter and dependency context.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import sys

INTERESTING_ENV = [
    "GITHUB_WORKFLOW",
    "GITHUB_JOB",
    "GITHUB_REF",
    "GITHUB_SHA",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "RUNNER_OS",
    "RUNNER_ARCH",
    "PYTHONUTF8",
    "PYTHONIOENCODING",
]


def _safe_run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output.strip()
    except Exception as exc:  # pragma: no cover - defensive path
        return f"<failed to execute {' '.join(command)}: {exc}>"


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect CI diagnostics markdown")
    parser.add_argument(
        "--output",
        default="output/test-results/ci-diagnostics.md",
        help="Markdown output path",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    pip_freeze = _safe_run([sys.executable, "-m", "pip", "freeze"])
    pip_freeze_lines = pip_freeze.splitlines()
    if len(pip_freeze_lines) > 300:
        pip_freeze_lines = pip_freeze_lines[:300] + ["... (truncated)"]

    sections: list[str] = [
        "# CI Diagnostics",
        "",
        f"Generated: {now}",
        "",
        "## Runtime",
        f"- Python executable: {sys.executable}",
        f"- Python version: {sys.version.splitlines()[0]}",
        f"- Platform: {platform.platform()}",
        f"- Working directory: {os.getcwd()}",
        "",
        "## GitHub Context",
    ]

    for key in INTERESTING_ENV:
        sections.append(f"- {key}: {os.environ.get(key, '<unset>')}")

    sections.extend(
        [
            "",
            "## pip freeze (top 300 lines)",
            "```text",
            *pip_freeze_lines,
            "```",
            "",
        ]
    )

    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[ci-diagnostics] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
