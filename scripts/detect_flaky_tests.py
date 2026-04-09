#!/usr/bin/env python3
"""Detect flaky tests by analyzing JUnit XML history across CI runs.

Compares test outcomes across multiple JUnit XML files to identify tests
that have inconsistent results (sometimes pass, sometimes fail). Helps
surface non-deterministic tests before they erode CI confidence.

Usage — local review of accumulated history:
    python scripts/detect_flaky_tests.py --history-dir output/test-history

Usage — CI (with history cache restored via actions/cache):
    python scripts/detect_flaky_tests.py \\
        --history-dir output/test-history \\
        --ingest output/test-results/reliability-junit.xml \\
        --output output/test-results/flaky-report.md \\
        --advisory

The --ingest flag copies the given XML into --history-dir with a timestamp
suffix before analysis, so history accumulates across CI runs when the cache
key is restored each run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


@dataclass
class TestRun:
    file: str
    status: str  # passed | failed | error | skipped


@dataclass
class TestHistory:
    classname: str
    name: str
    runs: list[TestRun] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.classname}::{self.name}"

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.runs if r.status == "passed")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.runs if r.status in ("failed", "error"))

    @property
    def is_flaky(self) -> bool:
        return self.pass_count > 0 and self.fail_count > 0

    @property
    def flakiness_rate(self) -> float:
        total = len(self.runs)
        return 0.0 if total == 0 else self.fail_count / total


def _parse_status(node: ET.Element) -> str:
    if node.find("failure") is not None:
        return "failed"
    if node.find("error") is not None:
        return "error"
    if node.find("skipped") is not None:
        return "skipped"
    return "passed"


def _parse_junit_xml(path: Path) -> list[tuple[str, str, str]]:
    """Return list of (classname, name, status) tuples from a JUnit XML file."""
    try:
        tree = ET.parse(path)  # noqa: S314  (trusted internal CI artifacts)
    except ET.ParseError:
        print(f"  [flaky-detect] WARNING: could not parse {path}, skipping")
        return []
    root = tree.getroot()
    results = []
    for tc in root.iter("testcase"):
        classname = tc.attrib.get("classname", "")
        name = tc.attrib.get("name", "")
        results.append((classname, name, _parse_status(tc)))
    return results


def _ingest_xml(source: Path, history_dir: Path) -> Path:
    """Copy source XML into history_dir using a UTC timestamp suffix."""
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = history_dir / f"{source.stem}--{ts}.xml"
    shutil.copy2(source, dest)
    return dest


def _build_history(history_dir: Path) -> dict[str, TestHistory]:
    """Scan all XMLs in history_dir and build per-test outcome history."""
    history: dict[str, TestHistory] = {}
    for xml_file in sorted(history_dir.glob("*.xml")):
        for classname, name, status in _parse_junit_xml(xml_file):
            key = f"{classname}::{name}"
            if key not in history:
                history[key] = TestHistory(classname=classname, name=name)
            history[key].runs.append(TestRun(file=xml_file.name, status=status))
    return history


def _render_report(
    flaky: list[TestHistory],
    total_tests: int,
    total_files: int,
) -> str:
    lines = [
        "# Flaky Test Report",
        "",
        f"Analyzed **{total_files}** JUnit XML run(s) covering **{total_tests}** unique test(s).",
        "",
    ]
    if not flaky:
        lines += [
            "**No flaky tests detected.** All tests have consistent outcomes across runs.",
            "",
        ]
        return "\n".join(lines)

    lines += [
        f"**{len(flaky)} flaky test(s) detected** — inconsistent pass/fail across runs:",
        "",
        "| Test | Runs | Pass | Fail | Flakiness |",
        "|------|------|------|------|-----------|",
    ]
    for th in sorted(flaky, key=lambda t: -t.flakiness_rate):
        rate = f"{th.flakiness_rate * 100:.0f}%"
        lines.append(
            f"| `{th.key}` | {len(th.runs)} | {th.pass_count} | {th.fail_count} | {rate} |"
        )

    lines += ["", "## Per-test run history", ""]
    for th in sorted(flaky, key=lambda t: -t.flakiness_rate):
        lines.append(f"### `{th.key}`")
        for run in th.runs:
            symbol = "✓" if run.status == "passed" else "✗"
            lines.append(f"- {symbol} `{run.file}` → **{run.status}**")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--history-dir",
        default="output/test-history",
        metavar="DIR",
        help="Directory containing historical JUnit XML files (default: output/test-history)",
    )
    parser.add_argument(
        "--ingest",
        metavar="XML",
        help="Copy this JUnit XML into --history-dir before analysis",
    )
    parser.add_argument(
        "--output",
        metavar="MD",
        help="Write markdown report to this path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        metavar="N",
        help="Report at most N flaky tests (default: 20)",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=2,
        metavar="N",
        help="Minimum runs required before a test can be flagged as flaky (default: 2)",
    )
    parser.add_argument(
        "--advisory",
        action="store_true",
        help="Always exit 0 — report flakiness but never block CI",
    )
    args = parser.parse_args()

    history_dir = Path(args.history_dir)

    if args.ingest:
        src = Path(args.ingest)
        if not src.exists():
            print(f"[flaky-detect] ERROR: --ingest file not found: {src}")
            return 1
        dest = _ingest_xml(src, history_dir)
        print(f"[flaky-detect] Ingested {src.name} → {dest.name}")

    history = _build_history(history_dir)

    if not history:
        print(
            f"[flaky-detect] No history found in {history_dir}. "
            "Run with --ingest to accumulate run results."
        )
        return 0

    xml_files = sorted(history_dir.glob("*.xml"))

    # Only flag tests with enough runs to be meaningful
    flaky = [
        th for th in history.values()
        if th.is_flaky and len(th.runs) >= args.min_runs
    ]
    flaky = flaky[: args.top_n]

    report = _render_report(
        flaky,
        total_tests=len(history),
        total_files=len(xml_files),
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"[flaky-detect] Report written to {out_path}")

    print(report)

    if flaky:
        verdict = "ADVISORY" if args.advisory else "FAIL"
        print(f"[flaky-detect] {verdict}: {len(flaky)} flaky test(s) detected.")
        return 0 if args.advisory else 1

    print(f"[flaky-detect] OK: no flaky tests detected across {len(xml_files)} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
