from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_script_module(name: str):
    module_path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


summarize_junit = _load_script_module("summarize_junit")
check_junit_budget = _load_script_module("check_junit_budget")


def _write_junit(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "report.xml"
    path.write_text(textwrap.dedent(content).strip(), encoding="utf-8")
    return path


def test_parse_junit_counts_all_statuses(tmp_path: Path) -> None:
    junit_path = _write_junit(
        tmp_path,
        """
        <testsuite tests="4" failures="1" errors="1" skipped="1">
          <testcase classname="pkg.mod" name="test_pass" time="1.1" />
          <testcase classname="pkg.mod" name="test_fail" time="2.2"><failure message="boom" /></testcase>
          <testcase classname="pkg.mod" name="test_error" time="3.3"><error message="oops" /></testcase>
          <testcase classname="pkg.mod" name="test_skip" time="0.4"><skipped /></testcase>
        </testsuite>
        """,
    )

    cases, totals = summarize_junit.parse_junit(junit_path)

    assert len(cases) == 4
    assert totals["passed"] == 1.0
    assert totals["failed"] == 1.0
    assert totals["errors"] == 1.0
    assert totals["skipped"] == 1.0
    assert totals["time_s"] == 7.0


def test_build_markdown_includes_failure_and_skip_sections(tmp_path: Path) -> None:
    junit_path = _write_junit(
        tmp_path,
        """
        <testsuite tests="3" failures="1" skipped="1">
          <testcase classname="pkg.mod" name="test_pass" time="1.0" />
                    <testcase classname="pkg.mod" name="test_fail" time="2.0"><failure message="AssertionError: expected x == y">assert x == y</failure></testcase>
          <testcase classname="pkg.mod" name="test_skip" time="0.5"><skipped /></testcase>
        </testsuite>
        """,
    )

    cases, totals = summarize_junit.parse_junit(junit_path)
    content = summarize_junit.build_markdown("Example Summary", cases, totals, top_n=5)

    assert "## Example Summary" in content
    assert "### Failure Triage" in content
    assert "pkg.mod::test_fail" in content
    assert "assertion" in content
    assert "AssertionError: expected x == y" in content
    assert "### Skipped Tests" in content
    assert "pkg.mod::test_skip" in content


def test_classify_case_distinguishes_common_failure_types() -> None:
    timeout_case = summarize_junit.TestCaseResult(
        classname="pkg.mod",
        name="test_timeout",
        time_s=3.0,
        status="failed",
        message="operation timed out after 5s",
        detail="",
    )
    import_case = summarize_junit.TestCaseResult(
        classname="pkg.mod",
        name="test_import",
        time_s=0.1,
        status="error",
        message="ModuleNotFoundError: No module named demo",
        detail="",
    )

    assert summarize_junit.classify_case(timeout_case) == "timeout"
    assert summarize_junit.classify_case(import_case) == "import"


def test_budget_markdown_passes_within_thresholds(tmp_path: Path) -> None:
    junit_path = _write_junit(
        tmp_path,
        """
        <testsuite tests="2">
          <testcase classname="pkg.mod" name="test_fast" time="1.0" />
          <testcase classname="pkg.mod" name="test_medium" time="2.5" />
        </testsuite>
        """,
    )

    cases, totals = summarize_junit.parse_junit(junit_path)
    content, within_budget = check_junit_budget.build_budget_markdown(
        title="Budget",
        cases=cases,
        total_time_s=float(totals["time_s"]),
        max_total_s=10.0,
        max_case_s=5.0,
    )

    assert within_budget is True
    assert "Budget status: PASS" in content
    assert "test_medium" in content


def test_budget_markdown_fails_when_runtime_exceeds_thresholds(tmp_path: Path) -> None:
    junit_path = _write_junit(
        tmp_path,
        """
        <testsuite tests="2">
          <testcase classname="pkg.mod" name="test_fast" time="1.0" />
          <testcase classname="pkg.mod" name="test_slow" time="9.5" />
        </testsuite>
        """,
    )

    cases, totals = summarize_junit.parse_junit(junit_path)
    content, within_budget = check_junit_budget.build_budget_markdown(
        title="Budget",
        cases=cases,
        total_time_s=float(totals["time_s"]),
        max_total_s=8.0,
        max_case_s=8.0,
    )

    assert within_budget is False
    assert "Budget status: FAIL" in content
    assert "pkg.mod::test_slow" in content
