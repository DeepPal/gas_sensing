#!/usr/bin/env python3
"""Generate a markdown summary from a pytest JUnit XML report."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass
class TestCaseResult:
    classname: str
    name: str
    time_s: float
    status: str
    message: str
    detail: str


def _iter_testcase_nodes(root: ET.Element):
    yield from root.iter("testcase")


def _status_for_case(node: ET.Element) -> str:
    if node.find("failure") is not None:
        return "failed"
    if node.find("error") is not None:
        return "error"
    if node.find("skipped") is not None:
        return "skipped"
    return "passed"


def _parse_float(value: str | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _text_preview(value: str, max_len: int = 120) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _extract_case_detail(node: ET.Element) -> tuple[str, str]:
    child = node.find("failure")
    if child is None:
        child = node.find("error")
    if child is None:
        child = node.find("skipped")
    if child is None:
        return "", ""
    message = child.attrib.get("message", "").strip()
    detail = (child.text or "").strip()
    return message, detail


def classify_case(case: TestCaseResult) -> str:
    if case.status == "skipped":
        return "skip"

    text = f"{case.message} {case.detail}".lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "assert" in text or "assertionerror" in text:
        return "assertion"
    if "importerror" in text or "modulenotfounderror" in text or "cannot import" in text:
        return "import"
    if "fixture" in text or "setup" in text:
        return "setup"
    if case.status == "error":
        return "runtime-error"
    if case.status == "failed":
        return "test-failure"
    return "pass"


def parse_junit(path: Path) -> tuple[list[TestCaseResult], dict[str, float]]:
    tree = ET.parse(path)
    root = tree.getroot()

    cases: list[TestCaseResult] = []
    for node in _iter_testcase_nodes(root):
        message, detail = _extract_case_detail(node)
        cases.append(
            TestCaseResult(
                classname=node.attrib.get("classname", ""),
                name=node.attrib.get("name", ""),
                time_s=_parse_float(node.attrib.get("time")),
                status=_status_for_case(node),
                message=message,
                detail=detail,
            )
        )

    totals = {
        "tests": float(len(cases)),
        "passed": float(sum(1 for c in cases if c.status == "passed")),
        "failed": float(sum(1 for c in cases if c.status == "failed")),
        "errors": float(sum(1 for c in cases if c.status == "error")),
        "skipped": float(sum(1 for c in cases if c.status == "skipped")),
        "time_s": float(sum(c.time_s for c in cases)),
    }
    return cases, totals


def build_markdown(
    title: str,
    cases: list[TestCaseResult],
    totals: dict[str, float],
    top_n: int,
) -> str:
    sorted_cases = sorted(cases, key=lambda c: c.time_s, reverse=True)
    slowest = sorted_cases[:top_n]
    failed_cases = [case for case in cases if case.status == "failed"]
    error_cases = [case for case in cases if case.status == "error"]
    skipped_cases = [case for case in cases if case.status == "skipped"]

    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "- Tests: "
        f"{int(totals['tests'])} "
        f"(passed={int(totals['passed'])}, failed={int(totals['failed'])}, "
        f"errors={int(totals['errors'])}, skipped={int(totals['skipped'])})"
    )
    lines.append(f"- Total runtime (sum of test case durations): {totals['time_s']:.2f}s")
    lines.append("")

    if not slowest:
        lines.append("No test cases found in report.")
        return "\n".join(lines) + "\n"

    lines.append(f"### Slowest {len(slowest)} Tests")
    lines.append("")
    lines.append("| Test | Status | Time (s) |")
    lines.append("|---|---:|---:|")

    for case in slowest:
        test_name = f"{case.classname}::{case.name}".strip(":")
        lines.append(f"| {test_name} | {case.status} | {case.time_s:.3f} |")

    lines.append("")

    if failed_cases or error_cases:
        lines.append("### Failure Triage")
        lines.append("")
        lines.append("| Test | Status | Category | Time (s) | Preview |")
        lines.append("|---|---:|---:|---:|---|")
        for case in [*failed_cases, *error_cases]:
            test_name = f"{case.classname}::{case.name}".strip(":")
            preview = _text_preview(case.message or case.detail or case.status)
            lines.append(
                f"| {test_name} | {case.status} | {classify_case(case)} | {case.time_s:.3f} | {preview} |"
            )
        lines.append("")

    if skipped_cases:
        lines.append("### Skipped Tests")
        lines.append("")
        lines.append(f"- Skipped count: {len(skipped_cases)}")
        preview = skipped_cases[: min(5, len(skipped_cases))]
        for case in preview:
            test_name = f"{case.classname}::{case.name}".strip(":")
            lines.append(f"- {test_name}")
        if len(skipped_cases) > len(preview):
            lines.append(f"- ... and {len(skipped_cases) - len(preview)} more")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize pytest JUnit XML into markdown")
    parser.add_argument("--junit", required=True, help="Path to JUnit XML file")
    parser.add_argument("--output", required=True, help="Path to markdown output file")
    parser.add_argument("--title", default="Test Summary", help="Markdown heading title")
    parser.add_argument("--top-n", type=int, default=10, help="Number of slow tests to include")
    args = parser.parse_args()

    junit_path = Path(args.junit)
    output_path = Path(args.output)

    cases, totals = parse_junit(junit_path)
    content = build_markdown(args.title, cases, totals, top_n=max(1, args.top_n))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
