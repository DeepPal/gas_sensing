#!/usr/bin/env python3
"""Check pytest JUnit runtime budgets and emit a markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize_junit import TestCaseResult, parse_junit


def build_budget_markdown(
    title: str,
    cases: list[TestCaseResult],
    total_time_s: float,
    max_total_s: float,
    max_case_s: float,
) -> tuple[str, bool]:
    sorted_cases = sorted(cases, key=lambda case: case.time_s, reverse=True)
    slowest = sorted_cases[:10]

    total_ok = total_time_s <= max_total_s
    slowest_case = slowest[0] if slowest else None
    case_ok = slowest_case is None or slowest_case.time_s <= max_case_s
    within_budget = total_ok and case_ok

    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"- Total runtime budget: {total_time_s:.2f}s / {max_total_s:.2f}s")
    if slowest_case is None:
        lines.append(f"- Slowest test budget: no tests found / {max_case_s:.2f}s")
    else:
        slow_name = f"{slowest_case.classname}::{slowest_case.name}".strip(":")
        lines.append(
            f"- Slowest test budget: {slowest_case.time_s:.3f}s / {max_case_s:.2f}s"
            f" ({slow_name})"
        )
    lines.append(f"- Budget status: {'PASS' if within_budget else 'FAIL'}")
    lines.append("")

    if slowest:
        lines.append(f"### Slowest {len(slowest)} Tests")
        lines.append("")
        lines.append("| Test | Time (s) | Budget Status |")
        lines.append("|---|---:|---:|")
        for case in slowest:
            case_name = f"{case.classname}::{case.name}".strip(":")
            status = "PASS" if case.time_s <= max_case_s else "FAIL"
            lines.append(f"| {case_name} | {case.time_s:.3f} | {status} |")
        lines.append("")

    return "\n".join(lines), within_budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Check pytest JUnit runtime budgets")
    parser.add_argument("--junit", required=True, help="Path to JUnit XML file")
    parser.add_argument("--output", required=True, help="Path to markdown output file")
    parser.add_argument("--title", default="Reliability Budget Check", help="Markdown heading title")
    parser.add_argument("--max-total-seconds", type=float, required=True, help="Max allowed total runtime")
    parser.add_argument("--max-case-seconds", type=float, required=True, help="Max allowed single test runtime")
    parser.add_argument("--advisory", action="store_true", help="Always exit 0 even if budgets are exceeded")
    args = parser.parse_args()

    junit_path = Path(args.junit)
    output_path = Path(args.output)

    cases, totals = parse_junit(junit_path)
    content, within_budget = build_budget_markdown(
        title=args.title,
        cases=cases,
        total_time_s=float(totals["time_s"]),
        max_total_s=args.max_total_seconds,
        max_case_s=args.max_case_seconds,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")

    if within_budget or args.advisory:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
