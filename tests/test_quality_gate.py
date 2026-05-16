from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

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


quality_gate = _load_script_module("quality_gate")


def _args(**overrides):
    base = dict(
        strict=False,
        smoke=False,
        lane="all",
        format_check=False,
        coverage=False,
        reliability_report=False,
        enforce_reliability_budget=False,
        max_total_seconds=45.0,
        max_case_seconds=12.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_checks_all_lane_includes_fast_and_reliability() -> None:
    checks = quality_gate.build_checks(_args())
    commands = [command for command, _ in checks]
    required = {command: is_required for command, is_required in checks}

    assert "python scripts/validate_workflows.py" in commands
    assert required["python scripts/validate_workflows.py"] is True
    assert any('pytest -q --tb=short -m "not reliability"' in command for command in commands)
    assert any('pytest -q --tb=short -m "reliability"' in command for command in commands)
    assert quality_gate.RUFF_REQUIRED_CHECK in required
    assert required[quality_gate.RUFF_REQUIRED_CHECK] is True
    assert quality_gate.RUFF_ADVISORY_CHECK in required
    assert required[quality_gate.RUFF_ADVISORY_CHECK] is False
    assert any(command.startswith("mypy src") for command in commands)



def test_build_checks_fast_lane_excludes_reliability_commands() -> None:
    checks = quality_gate.build_checks(_args(lane="fast"))
    commands = [command for command, _ in checks]

    assert any('pytest -q --tb=short -m "not reliability"' in command for command in commands)
    assert not any("--cov=src" in command for command in commands)
    assert not any('--junitxml=' in command for command in commands)
    assert not any('pytest -q --tb=short -m "reliability"' in command for command in commands)



def test_build_checks_fast_lane_with_coverage_adds_cov_threshold() -> None:
    checks = quality_gate.build_checks(_args(lane="fast", coverage=True))
    commands = [command for command, _ in checks]

    assert any("--cov=src" in command and "--cov-fail-under=75" in command for command in commands)
    assert not any("--cov=gas_analysis" in command for command in commands)



def test_build_checks_reliability_budget_flow_adds_report_commands() -> None:
    checks = quality_gate.build_checks(
        _args(lane="reliability", reliability_report=True, enforce_reliability_budget=True)
    )
    commands = [command for command, _ in checks]

    assert any('mkdir' not in command and '--junitxml=' in command for command in commands)
    assert any('scripts/summarize_junit.py' in command for command in commands)
    assert any('scripts/check_junit_budget.py' in command for command in commands)
    assert any('Path(r' in command for command in commands)



def test_build_checks_smoke_appends_smoke_pytest() -> None:
    checks = quality_gate.build_checks(_args(smoke=True))
    commands = [command for command, _ in checks]

    assert commands[-1] == 'pytest -m smoke -v'


def test_coverage_preflight_message_none_without_coverage() -> None:
    msg = quality_gate._coverage_preflight_message(_args(lane="fast", coverage=False))
    assert msg is None


def test_coverage_preflight_message_none_for_reliability_lane(monkeypatch) -> None:
    monkeypatch.setattr(
        quality_gate,
        "_missing_optional_coverage_dependencies",
        lambda: ["mlflow"],
    )
    msg = quality_gate._coverage_preflight_message(_args(lane="reliability", coverage=True))
    assert msg is None


def test_coverage_preflight_message_lists_missing_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(
        quality_gate,
        "_missing_optional_coverage_dependencies",
        lambda: ["mlflow", "onnxruntime"],
    )
    msg = quality_gate._coverage_preflight_message(_args(lane="fast", coverage=True))
    assert msg is not None
    assert "mlflow, onnxruntime" in msg
    assert 'pip install -e ".[dev,ml,tracking,all]"' in msg
