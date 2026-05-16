#!/usr/bin/env python3
"""Research preflight checks for lab sessions.

Runs fast checks before starting a high-value experiment so researchers can
catch avoidable issues (integrity mismatches, stale calibration assumptions,
missing manifests, and environment gaps) early.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from dashboard.reproducibility import ReproducibilityManifest, verify_manifest_artifacts

DEFAULT_SLOPE = 0.116


@dataclass
class Finding:
    level: str  # PASS | WARN | FAIL
    title: str
    detail: str


def _load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        import yaml
    except Exception:
        return None

    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _find_latest_session(sessions_root: Path) -> Path | None:
    if not sessions_root.exists():
        return None
    session_dirs = [p for p in sessions_root.iterdir() if p.is_dir()]
    if not session_dirs:
        return None

    def _mtime(p: Path) -> float:
        meta = p / "session_meta.json"
        if meta.exists():
            return meta.stat().st_mtime
        return p.stat().st_mtime

    return max(session_dirs, key=_mtime)


def _check_python() -> Finding:
    ok = sys.version_info >= (3, 9)
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if ok:
        return Finding("PASS", "Python version", f"Detected Python {version}")
    return Finding("FAIL", "Python version", f"Detected Python {version}; require >= 3.9")


def _check_config(config_path: Path, require_calibrated_slope: bool) -> list[Finding]:
    findings: list[Finding] = []
    cfg = _load_yaml(config_path)
    if cfg is None:
        findings.append(Finding("WARN", "Config load", f"Could not read {config_path}"))
        return findings

    cal = cfg.get("calibration") or {}
    slope = cal.get("calibration_slope")
    if slope is None:
        findings.append(
            Finding(
                "WARN",
                "Calibration slope",
                "No explicit calibration.calibration_slope found in config; verify live slope source before quantification",
            )
        )
    else:
        try:
            slope_val = float(slope)
            if abs(slope_val - DEFAULT_SLOPE) < 1e-9:
                level = "FAIL" if require_calibrated_slope else "WARN"
                findings.append(
                    Finding(
                        level,
                        "Calibration slope",
                        f"Using default literature slope {slope_val:.3f} nm/ppm; replace with experimentally validated slope",
                    )
                )
            else:
                findings.append(
                    Finding("PASS", "Calibration slope", f"Configured slope is {slope_val:.6f} nm/ppm")
                )
        except Exception:
            findings.append(Finding("WARN", "Calibration slope", f"Non-numeric slope value: {slope!r}"))

    env = cfg.get("environment") or {}
    enabled = bool(env.get("enabled", False))
    temp_coeff_raw = (env.get("coefficients") or {}).get("temperature", 0.0)
    hum_coeff_raw = (env.get("coefficients") or {}).get("humidity", 0.0)
    try:
        temp_coeff = float(temp_coeff_raw)
        hum_coeff = float(hum_coeff_raw)
        if enabled and (abs(temp_coeff) > 0 or abs(hum_coeff) > 0):
            findings.append(
                Finding("PASS", "Environment compensation", "Environment compensation is configured and enabled")
            )
        else:
            findings.append(
                Finding(
                    "WARN",
                    "Environment compensation",
                    "Temperature/humidity compensation appears disabled or zeroed; verify thermal drift controls for publication runs",
                )
            )
    except Exception:
        findings.append(
            Finding(
                "WARN",
                "Environment compensation",
                f"Non-numeric environment coefficients: temperature={temp_coeff_raw!r}, humidity={hum_coeff_raw!r}",
            )
        )

    return findings


def _check_latest_session(
    sessions_root: Path,
    max_age_h: float,
    require_manifest: bool = False,
) -> list[Finding]:
    findings: list[Finding] = []
    latest = _find_latest_session(sessions_root)
    if latest is None:
        findings.append(
            Finding(
                "WARN",
                "Latest session",
                f"No sessions found under {sessions_root}; preflight cannot verify recent calibration integrity",
            )
        )
        return findings

    findings.append(Finding("PASS", "Latest session", f"Found latest session directory: {latest.name}"))

    meta_path = latest / "session_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ts = _parse_iso(meta.get("stopped_at") or meta.get("started_at"))
            if ts is not None:
                age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
                if age_h > max_age_h:
                    findings.append(
                        Finding(
                            "WARN",
                            "Calibration recency",
                            f"Latest session age is {age_h:.1f}h (threshold {max_age_h:.1f}h); consider a fresh verification run",
                        )
                    )
                else:
                    findings.append(
                        Finding("PASS", "Calibration recency", f"Latest session age is {age_h:.1f}h")
                    )
        except Exception as exc:
            findings.append(Finding("WARN", "Session metadata", f"Could not parse session_meta.json: {exc}"))
    else:
        findings.append(Finding("WARN", "Session metadata", "session_meta.json is missing in latest session"))

    manifests = sorted(latest.glob("*_manifest.json"))
    if not manifests:
        level = "FAIL" if require_manifest else "WARN"
        findings.append(
            Finding(
                level,
                "Manifest presence",
                "No reproducibility manifest found in latest session",
            )
        )
        return findings

    manifest = manifests[0]
    ok, errs = verify_manifest_artifacts(manifest)
    if ok:
        findings.append(Finding("PASS", "Manifest integrity", f"Artifact checksum verification passed for {manifest.name}"))
    else:
        detail = "; ".join(errs[:3])
        findings.append(Finding("FAIL", "Manifest integrity", f"Artifact verification failed: {detail}"))

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        commit = ((payload.get("version_control") or {}).get("commit_hash") or "unknown").strip()
        if commit and commit != "unknown":
            findings.append(Finding("PASS", "Manifest provenance", f"Commit hash captured: {commit[:12]}"))
        else:
            findings.append(Finding("WARN", "Manifest provenance", "Manifest commit hash is unknown"))
    except Exception:
        findings.append(Finding("WARN", "Manifest provenance", "Could not parse manifest provenance fields"))

    return findings


def _check_writable(path: Path) -> Finding:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".preflight_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return Finding("PASS", "Output directory write", f"Writable: {path}")
    except Exception as exc:
        return Finding("FAIL", "Output directory write", f"Cannot write to {path}: {exc}")


def _print_findings(findings: list[Finding]) -> None:
    for f in findings:
        tag = {
            "PASS": "[PASS]",
            "WARN": "[WARN]",
            "FAIL": "[FAIL]",
        }.get(f.level, "[INFO]")
        print(f"{tag} {f.title}: {f.detail}")


def _summary(findings: list[Finding]) -> tuple[int, int, int]:
    p = sum(1 for f in findings if f.level == "PASS")
    w = sum(1 for f in findings if f.level == "WARN")
    f = sum(1 for f in findings if f.level == "FAIL")
    return p, w, f


def _run_self_check() -> int:
    """Run a deterministic strict-mode preflight over synthetic artifacts.

    This validates the preflight gate itself end-to-end for CI and release checks.
    """
    with tempfile.TemporaryDirectory(prefix="preflight_self_check_") as tmp:
        root = Path(tmp)
        config_path = root / "config.yaml"
        sessions_root = root / "sessions"
        output_root = root / "output"
        session_dir = sessions_root / "20990101_000000"
        session_dir.mkdir(parents=True, exist_ok=True)

        config_path.write_text(
            """
calibration:
  calibration_slope: 0.200
environment:
  enabled: true
  coefficients:
    temperature: 0.001
    humidity: 0.001
""".strip(),
            encoding="utf-8",
        )

        now = datetime.now(timezone.utc).isoformat()
        (session_dir / "session_meta.json").write_text(
            json.dumps({"started_at": now, "stopped_at": now}),
            encoding="utf-8",
        )
        (session_dir / "pipeline_results.csv").write_text("x,y\n1,2\n", encoding="utf-8")

        manifest = ReproducibilityManifest("20990101_000000", app_root=Path.cwd(), operator="preflight-self-check")
        manifest.save(session_dir)

        findings: list[Finding] = []
        findings.append(_check_python())
        findings.extend(_check_config(config_path, require_calibrated_slope=True))
        findings.extend(
            _check_latest_session(
                sessions_root,
                max_age_h=24,
                require_manifest=True,
            )
        )
        findings.append(_check_writable(output_root))

        _print_findings(findings)
        passed, warned, failed = _summary(findings)
        print(f"\n[self-check summary] PASS={passed} WARN={warned} FAIL={failed}")

        if failed > 0:
            return 2
        if warned > 0:
            return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run researcher preflight readiness checks before experiments.")
    parser.add_argument("--config", type=Path, default=Path("config/config.yaml"), help="Path to calibration config YAML")
    parser.add_argument("--sessions-root", type=Path, default=Path("output/sessions"), help="Root directory containing recorded sessions")
    parser.add_argument("--output-root", type=Path, default=Path("output"), help="Output directory to test write access")
    parser.add_argument("--max-calibration-age-hours", type=float, default=72.0, help="Warn when latest session is older than this")
    parser.add_argument("--require-calibrated-slope", action="store_true", help="Fail if default slope appears to be in use")
    parser.add_argument("--require-manifest", action="store_true", help="Fail when latest session has no reproducibility manifest")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return non-zero exit code when warnings exist")
    parser.add_argument("--self-check", action="store_true", help="Run deterministic strict-mode self-check for CI gate validation")
    args = parser.parse_args()

    if args.self_check:
        return _run_self_check()

    findings: list[Finding] = []
    findings.append(_check_python())
    findings.extend(_check_config(args.config, args.require_calibrated_slope))
    findings.extend(
        _check_latest_session(
            args.sessions_root,
            args.max_calibration_age_hours,
            require_manifest=args.require_manifest,
        )
    )
    findings.append(_check_writable(args.output_root))

    _print_findings(findings)
    passed, warned, failed = _summary(findings)
    print(f"\n[summary] PASS={passed} WARN={warned} FAIL={failed}")

    if failed > 0:
        return 2
    if warned > 0 and args.fail_on_warning:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
