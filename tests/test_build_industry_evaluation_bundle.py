from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import zipfile

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


industry_bundle = _load_script_module("build_industry_evaluation_bundle")


def test_build_bundle_success_writes_manifest_summary_and_zip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(industry_bundle, "ROOT", tmp_path)

    contracts_dir = tmp_path / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    (contracts_dir / "openapi_baseline.json").write_text('{"routes": []}', encoding="utf-8")

    steps = [
        {
            "name": "openapi_compat",
            "command": "python scripts/check_openapi_compat.py",
            "returncode": 0,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "ended_at_utc": "2026-01-01T00:00:01+00:00",
            "stdout": "ok",
            "stderr": "",
            "passed": True,
        },
        {
            "name": "integrator_smoke",
            "command": "python scripts/integration_smoke_check.py",
            "returncode": 0,
            "started_at_utc": "2026-01-01T00:00:02+00:00",
            "ended_at_utc": "2026-01-01T00:00:03+00:00",
            "stdout": "ok",
            "stderr": "",
            "passed": True,
        },
    ]
    call_idx = {"value": 0}

    def _fake_run_step(name: str, args: list[str], cwd: Path):
        assert cwd == tmp_path
        step = steps[call_idx["value"]]
        call_idx["value"] += 1
        assert step["name"] == name
        return step

    monkeypatch.setattr(industry_bundle, "_run_step", _fake_run_step)
    monkeypatch.setattr(industry_bundle, "_git_sha", lambda: "abc123")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_industry_evaluation_bundle.py",
            "--output-dir",
            "output/industry-eval/test",
            "--session-id",
            "ci-session",
        ],
    )

    industry_bundle.main()

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "ok"

    manifest_path = Path(payload["manifest"])
    summary_path = Path(payload["summary"])
    bundle_zip = Path(payload["bundle_zip"])
    baseline_files = sorted(manifest_path.parent.glob("openapi_baseline_*.json"))

    assert manifest_path.exists()
    assert summary_path.exists()
    assert bundle_zip.exists()
    assert len(baseline_files) == 1
    baseline_copy = baseline_files[0]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["session_id"] == "ci-session"
    assert [step["name"] for step in manifest["steps"]] == ["openapi_compat", "integrator_smoke"]

    summary = summary_path.read_text(encoding="utf-8")
    assert "Status: **ok**" in summary
    assert "openapi_compat: **PASS**" in summary
    assert "integrator_smoke: **PASS**" in summary

    with zipfile.ZipFile(bundle_zip) as zf:
        names = set(zf.namelist())
    assert manifest_path.name in names
    assert summary_path.name in names
    assert baseline_copy.name in names


def test_build_bundle_sets_failed_status_when_any_step_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(industry_bundle, "ROOT", tmp_path)
    (tmp_path / "contracts").mkdir(parents=True, exist_ok=True)

    steps = [
        {
            "name": "openapi_compat",
            "command": "python scripts/check_openapi_compat.py",
            "returncode": 1,
            "started_at_utc": "2026-01-01T00:00:00+00:00",
            "ended_at_utc": "2026-01-01T00:00:01+00:00",
            "stdout": "",
            "stderr": "compat failed",
            "passed": False,
        },
        {
            "name": "integrator_smoke",
            "command": "python scripts/integration_smoke_check.py",
            "returncode": 0,
            "started_at_utc": "2026-01-01T00:00:02+00:00",
            "ended_at_utc": "2026-01-01T00:00:03+00:00",
            "stdout": "ok",
            "stderr": "",
            "passed": True,
        },
    ]
    call_idx = {"value": 0}

    def _fake_run_step(name: str, args: list[str], cwd: Path):
        _ = args
        assert cwd == tmp_path
        step = steps[call_idx["value"]]
        call_idx["value"] += 1
        assert step["name"] == name
        return step

    monkeypatch.setattr(industry_bundle, "_run_step", _fake_run_step)
    monkeypatch.setattr(industry_bundle, "_git_sha", lambda: "deadbeef")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_industry_evaluation_bundle.py",
            "--output-dir",
            "output/industry-eval/fail-case",
            "--session-id",
            "fail-session",
        ],
    )

    import pytest

    with pytest.raises(SystemExit) as exc_info:
        industry_bundle.main()
    assert exc_info.value.code == 1

    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "failed"

    manifest_path = Path(payload["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert any(step["passed"] is False for step in manifest["steps"])
