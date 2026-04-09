#!/usr/bin/env python3
"""Build a publication-facing research evidence pack in one command.

This orchestrates benchmark evidence generation, blinded replication manifest
creation, and qualification dossier packaging into a single reproducible flow.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def _extract_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Expected JSON output from sub-command, but none was found.")


def _run_step(step: str, args: list[str]) -> dict[str, Any]:
    cmd = [sys.executable] + args
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    payload = _extract_last_json(result.stdout)
    payload["_step"] = step
    payload["_command"] = " ".join(cmd)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_optional_path(payload: dict[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/qualification/ci")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--profile", default="config/benchmark_profiles/external_blinded_profile.json")
    parser.add_argument("--session-id", default="ci-release")
    parser.add_argument("--git-sha", default="unknown")
    parser.add_argument("--ref-name", default="unknown")
    parser.add_argument("--run-id", default="local")
    parser.add_argument("--signing-key", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_cmd = ["scripts/generate_benchmark_evidence.py", "--output-dir", str(out_dir)]
    if args.data_dir:
        benchmark_cmd += ["--data-dir", args.data_dir]

    benchmark = _run_step("benchmark_evidence", benchmark_cmd)
    blinded = _run_step(
        "blinded_replication_manifest",
        [
            "scripts/generate_blinded_replication_manifest.py",
            "--output-dir",
            str(out_dir),
            "--profile",
            args.profile,
            "--run-id",
            args.run_id,
            "--git-sha",
            args.git_sha,
        ],
    )

    qualification = _run_step(
        "qualification_dossier",
        [
            "scripts/generate_qualification_artifacts.py",
            "--output-dir",
            str(out_dir),
            "--session-id",
            args.session_id,
            "--git-sha",
            args.git_sha,
            "--ref-name",
            args.ref_name,
            "--run-id",
            args.run_id,
            "--signing-key",
            args.signing_key,
        ],
    )

    tracked_paths: list[Path] = []
    for payload in (benchmark, blinded, qualification):
        for key in ("json", "markdown", "html", "signature", "package"):
            resolved = _resolve_optional_path(payload, key)
            if resolved is not None:
                tracked_paths.append(resolved)

    # Stable ordering for reproducibility and easy diffs.
    tracked_paths = sorted({path.resolve() for path in tracked_paths})

    checksums: dict[str, str] = {}
    for path in tracked_paths:
        rel = path.relative_to(out_dir.resolve()).as_posix()
        checksums[rel] = _sha256(path)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    index_payload = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": args.session_id,
        "git_sha": args.git_sha,
        "ref_name": args.ref_name,
        "run_id": args.run_id,
        "steps": {
            "benchmark": benchmark,
            "blinded_replication": blinded,
            "qualification": qualification,
        },
        "artifacts": {
            "count": len(tracked_paths),
            "sha256": checksums,
        },
    }

    index_json = out_dir / f"research_evidence_index_{stamp}.json"
    index_md = out_dir / f"research_evidence_index_{stamp}.md"

    index_json.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Research Evidence Pack",
        "",
        f"- Session ID: {args.session_id}",
        f"- Git SHA: {args.git_sha}",
        f"- Ref Name: {args.ref_name}",
        f"- Run ID: {args.run_id}",
        f"- Generated: {index_payload['generated_at_utc']}",
        "",
        "## Artifacts",
        "",
    ]
    for rel, digest in checksums.items():
        lines.append(f"- `{rel}`  ")
        lines.append(f"  - sha256: `{digest}`")
    index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out_dir),
                "index_json": str(index_json),
                "index_markdown": str(index_md),
                "artifact_count": len(tracked_paths),
            }
        )
    )


if __name__ == "__main__":
    main()
