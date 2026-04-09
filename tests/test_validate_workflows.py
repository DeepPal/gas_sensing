from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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


validate_workflows = _load_script_module("validate_workflows")


def test_security_workflow_dependency_review_rules_pass(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    security = workflows / "security.yml"
    security.write_text(
        """
name: Security Gates
jobs:
  dependency-review:
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/dependency-review-action@v4
        with:
          base-ref: ${{ github.event.pull_request.base.sha }}
          head-ref: ${{ github.event.pull_request.head.sha }}
""".strip(),
        encoding="utf-8",
    )

    old_root = validate_workflows.ROOT
    try:
        validate_workflows.ROOT = tmp_path
        errors = validate_workflows._validate_file(security)
    finally:
        validate_workflows.ROOT = old_root

    assert errors == []


def test_security_workflow_dependency_review_rules_fail_when_missing_refs(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    security = workflows / "security.yml"
    security.write_text(
        """
name: Security Gates
jobs:
  dependency-review:
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/dependency-review-action@v4
""".strip(),
        encoding="utf-8",
    )

    old_root = validate_workflows.ROOT
    try:
        validate_workflows.ROOT = tmp_path
        errors = validate_workflows._validate_file(security)
    finally:
        validate_workflows.ROOT = old_root

    assert any("base-ref" in e for e in errors)
    assert any("head-ref" in e for e in errors)
