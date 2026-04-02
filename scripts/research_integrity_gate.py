#!/usr/bin/env python3
"""Research integrity gate checks.

Validates reproducibility manifests and artifact checksums for session outputs.
Can also run a self-check to ensure tamper detection is functional.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from dashboard.reproducibility import ReproducibilityManifest, verify_manifest_artifacts


REQUIRED_TOP_LEVEL_KEYS = {
    "session_id",
    "timestamp",
    "operator",
    "application",
    "version_control",
    "environment",
    "configuration",
}


def _find_manifests(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*_manifest.json") if p.is_file())


def _validate_manifest_schema(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Could not parse {path}: {exc}"]

    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(payload.keys()))
    if missing:
        errors.append(f"Missing required keys: {', '.join(missing)}")

    cfg = payload.get("configuration")
    if not isinstance(cfg, dict) or "hash" not in cfg:
        errors.append("configuration.hash missing")

    env = payload.get("environment")
    if not isinstance(env, dict) or "python_version" not in env:
        errors.append("environment.python_version missing")

    vc = payload.get("version_control")
    if not isinstance(vc, dict) or "commit_hash" not in vc:
        errors.append("version_control.commit_hash missing")

    return errors


def _self_check() -> tuple[bool, list[str]]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        session_dir = root / "session_demo"
        session_dir.mkdir(parents=True)
        data_file = session_dir / "frame.csv"
        data_file.write_text("frame,peak_wavelength\n1,531.5\n", encoding="utf-8")

        manifest = ReproducibilityManifest("session_demo", app_root=root, operator="ci")
        manifest_path = manifest.save(session_dir)

        ok, verify_errors = verify_manifest_artifacts(manifest_path)
        if not ok:
            errors.extend(verify_errors)
            return False, errors

        data_file.write_text("tampered\n", encoding="utf-8")
        ok_after, verify_errors_after = verify_manifest_artifacts(manifest_path)
        if ok_after:
            errors.append("Self-check failed: tampering was not detected")
        elif not any("Checksum mismatch" in e for e in verify_errors_after):
            errors.append("Self-check failed: mismatch reason was not surfaced")

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run research integrity gate checks.")
    parser.add_argument("--root", type=Path, default=Path("output/sessions"), help="Directory to scan for manifests")
    parser.add_argument("--allow-empty", action="store_true", help="Do not fail when no manifests are found")
    parser.add_argument("--require-checksums", action="store_true", help="Fail when manifests have no artifact_checksums")
    parser.add_argument("--self-check", action="store_true", help="Run built-in tamper detection self-check")
    args = parser.parse_args()

    if args.self_check:
        ok, errors = _self_check()
        if ok:
            print("[integrity] self-check OK")
            return 0
        print("[integrity] self-check FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1

    manifests = _find_manifests(args.root)
    if not manifests:
        msg = f"No manifests found under {args.root}"
        if args.allow_empty:
            print(f"[integrity] SKIP: {msg}")
            return 0
        print(f"[integrity] FAILED: {msg}")
        return 1

    failures: list[str] = []
    for manifest_path in manifests:
        schema_errors = _validate_manifest_schema(manifest_path)
        failures.extend([f"{manifest_path}: {e}" for e in schema_errors])

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = payload.get("artifact_checksums", [])
        if args.require_checksums and not checksums:
            failures.append(f"{manifest_path}: artifact_checksums missing or empty")

        ok, verify_errors = verify_manifest_artifacts(manifest_path)
        if not ok:
            failures.extend([f"{manifest_path}: {e}" for e in verify_errors])

    if failures:
        print("[integrity] FAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"[integrity] OK: verified {len(manifests)} manifest(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
