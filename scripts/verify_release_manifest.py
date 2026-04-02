#!/usr/bin/env python3
"""Verify release artifact manifest completeness.

Ensures a checksum manifest exists and contains entries for all wheel and
source-distribution artifacts in the dist directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


LINE_RE = re.compile(r"^[0-9a-f]{64}\s{2}(.+)$")


def _parse_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        match = LINE_RE.match(line)
        if not match:
            raise ValueError(f"Malformed manifest line {idx}: {raw}")
        filename = match.group(1).strip()
        records[filename] = line[:64]
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify release checksum manifest")
    parser.add_argument("--dist-dir", default="dist", help="Distribution directory")
    parser.add_argument("--manifest", default="sha256sums.txt", help="Manifest filename")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.exists() or not dist_dir.is_dir():
        print(f"[release-manifest] ERROR: dist directory not found: {dist_dir}")
        return 1

    manifest_path = dist_dir / args.manifest
    if not manifest_path.exists():
        print(f"[release-manifest] ERROR: manifest not found: {manifest_path}")
        return 1

    expected = sorted([*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")], key=lambda p: p.name)
    if not expected:
        print(f"[release-manifest] ERROR: no release artifacts found under {dist_dir}")
        return 1

    try:
        records = _parse_manifest(manifest_path)
    except Exception as exc:
        print(f"[release-manifest] ERROR: {exc}")
        return 1

    missing: list[str] = []
    for artifact in expected:
        if artifact.name not in records:
            missing.append(artifact.name)

    if missing:
        print("[release-manifest] ERROR: missing checksum entries for:")
        for name in missing:
            print(f"  - {name}")
        return 1

    print(f"[release-manifest] OK: verified {len(expected)} artifact entries in {manifest_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
