#!/usr/bin/env python3
"""Build a compact industry evaluation bundle.

The bundle is designed for external technical evaluators and contains:
- OpenAPI compatibility check output
- Integrator smoke check output
- Environment manifest (platform, python, git sha)
- OpenAPI baseline snapshot
- Optional research evidence pack outputs
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _run_step(name: str, args: list[str], cwd: Path) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    ended = datetime.now(timezone.utc)
    return {
        "name": name,
        "command": " ".join(args),
        "returncode": proc.returncode,
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0,
    }


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip() or "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/industry-eval/local")
    parser.add_argument("--session-id", default="industry-eval")
    parser.add_argument("--with-evidence-pack", action="store_true")
    args = parser.parse_args()

    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    steps: list[dict[str, Any]] = []
    steps.append(
        _run_step(
            "openapi_compat",
            [sys.executable, "scripts/check_openapi_compat.py"],
            cwd=ROOT,
        )
    )
    steps.append(
        _run_step(
            "integrator_smoke",
            [sys.executable, "scripts/integration_smoke_check.py"],
            cwd=ROOT,
        )
    )

    evidence_summary: dict[str, Any] | None = None
    if args.with_evidence_pack:
        evidence_dir = out_dir / "evidence_pack"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        evidence_step = _run_step(
            "research_evidence_pack",
            [
                sys.executable,
                "scripts/build_research_evidence_pack.py",
                "--output-dir",
                str(evidence_dir),
                "--session-id",
                args.session_id,
                "--git-sha",
                _git_sha(),
                "--run-id",
                f"industry-eval-{stamp}",
                "--ref-name",
                "local",
            ],
            cwd=ROOT,
        )
        steps.append(evidence_step)
        evidence_summary = {
            "enabled": True,
            "passed": evidence_step["passed"],
            "output_dir": str(evidence_dir),
        }
    else:
        evidence_summary = {
            "enabled": False,
            "passed": None,
            "output_dir": None,
        }

    manifest = {
        "status": "ok" if all(step["passed"] for step in steps) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "steps": steps,
        "evidence_pack": evidence_summary,
    }

    manifest_path = out_dir / f"industry_eval_manifest_{stamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline_src = ROOT / "contracts" / "openapi_baseline.json"
    baseline_copy = out_dir / f"openapi_baseline_{stamp}.json"
    if baseline_src.exists():
        baseline_copy.write_text(baseline_src.read_text(encoding="utf-8"), encoding="utf-8")

    summary_path = out_dir / f"industry_eval_summary_{stamp}.md"
    summary_lines = [
        "# Industry Evaluation Bundle Summary",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Session ID: `{args.session_id}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Step Results",
        "",
    ]
    for step in steps:
        status = "PASS" if step["passed"] else "FAIL"
        summary_lines.append(f"- {step['name']}: **{status}**")
    summary_lines.append("")
    summary_lines.append("## Files")
    summary_lines.append("")
    summary_lines.append(f"- `{manifest_path.name}`")
    summary_lines.append(f"- `{summary_path.name}`")
    if baseline_copy.exists():
        summary_lines.append(f"- `{baseline_copy.name}`")

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    zip_path = out_dir / f"industry_evaluation_bundle_{stamp}.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(manifest_path, arcname=manifest_path.name)
        zf.write(summary_path, arcname=summary_path.name)
        if baseline_copy.exists():
            zf.write(baseline_copy, arcname=baseline_copy.name)

    checksums = {
        manifest_path.name: _sha256(manifest_path),
        summary_path.name: _sha256(summary_path),
        zip_path.name: _sha256(zip_path),
    }
    if baseline_copy.exists():
        checksums[baseline_copy.name] = _sha256(baseline_copy)

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(out_dir),
                "manifest": str(manifest_path),
                "summary": str(summary_path),
                "bundle_zip": str(zip_path),
                "checksums": checksums,
            }
        )
    )


if __name__ == "__main__":
    main()
