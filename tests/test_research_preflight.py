from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

from dashboard.reproducibility import ReproducibilityManifest

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


preflight = _load_script_module("research_preflight")


def test_parse_iso_handles_z_suffix() -> None:
    dt = preflight._parse_iso("2026-04-01T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_invalid_returns_none() -> None:
    assert preflight._parse_iso("not-a-date") is None


def test_check_config_default_slope_warns(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
calibration:
  calibration_slope: 0.116
environment:
  enabled: false
  coefficients:
    temperature: 0.0
    humidity: 0.0
""".strip(),
        encoding="utf-8",
    )

    findings = preflight._check_config(cfg, require_calibrated_slope=False)
    levels = {(f.title, f.level) for f in findings}
    assert ("Calibration slope", "WARN") in levels


def test_check_config_default_slope_fails_when_required(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
calibration:
  calibration_slope: 0.116
environment:
  enabled: false
  coefficients:
    temperature: 0.0
    humidity: 0.0
""".strip(),
        encoding="utf-8",
    )

    findings = preflight._check_config(cfg, require_calibrated_slope=True)
    levels = {(f.title, f.level) for f in findings}
    assert ("Calibration slope", "FAIL") in levels


def test_check_config_non_numeric_environment_coeffs_warn(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
calibration:
  calibration_slope: 0.2
environment:
  enabled: true
  coefficients:
    temperature: abc
    humidity: def
""".strip(),
        encoding="utf-8",
    )

    findings = preflight._check_config(cfg, require_calibrated_slope=False)
    env = [f for f in findings if f.title == "Environment compensation"]
    assert env
    assert env[0].level == "WARN"
    assert "Non-numeric" in env[0].detail


def test_check_latest_session_warns_when_empty(tmp_path: Path) -> None:
    findings = preflight._check_latest_session(tmp_path / "sessions", max_age_h=24)
    assert any(f.title == "Latest session" and f.level == "WARN" for f in findings)


def test_check_latest_session_missing_manifest_warns_by_default(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    s = sessions_root / "20260401_120000"
    s.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    meta = {"started_at": now.isoformat(), "stopped_at": now.isoformat()}
    (s / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    findings = preflight._check_latest_session(sessions_root, max_age_h=24)
    assert any(f.title == "Manifest presence" and f.level == "WARN" for f in findings)


def test_check_latest_session_missing_manifest_fails_when_required(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    s = sessions_root / "20260401_120000"
    s.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    meta = {"started_at": now.isoformat(), "stopped_at": now.isoformat()}
    (s / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    findings = preflight._check_latest_session(sessions_root, max_age_h=24, require_manifest=True)
    assert any(f.title == "Manifest presence" and f.level == "FAIL" for f in findings)


def test_check_latest_session_manifest_pass(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    s = sessions_root / "20260401_120000"
    s.mkdir(parents=True)

    now = datetime.now(timezone.utc)
    meta = {"started_at": now.isoformat(), "stopped_at": now.isoformat()}
    (s / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Create one artifact and corresponding manifest with checksums.
    (s / "pipeline_results.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    manifest = ReproducibilityManifest("20260401_120000", app_root=ROOT, operator="test")
    manifest.save(s)

    findings = preflight._check_latest_session(sessions_root, max_age_h=24)
    assert any(f.title == "Manifest integrity" and f.level == "PASS" for f in findings)


def test_check_latest_session_recency_warn(tmp_path: Path) -> None:
    sessions_root = tmp_path / "sessions"
    s = sessions_root / "20260401_120000"
    s.mkdir(parents=True)

    old = datetime.now(timezone.utc) - timedelta(hours=200)
    meta = {"started_at": old.isoformat(), "stopped_at": old.isoformat()}
    (s / "session_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    findings = preflight._check_latest_session(sessions_root, max_age_h=24)
    assert any(f.title == "Calibration recency" and f.level == "WARN" for f in findings)


def test_summary_counts() -> None:
    findings = [
        preflight.Finding("PASS", "a", "a"),
        preflight.Finding("WARN", "b", "b"),
        preflight.Finding("FAIL", "c", "c"),
        preflight.Finding("PASS", "d", "d"),
    ]
    assert preflight._summary(findings) == (2, 1, 1)


def test_self_check_passes() -> None:
    assert preflight._run_self_check() == 0
