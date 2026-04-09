#!/usr/bin/env python3
"""Generate SHA-256 checksums for release artifacts.

This script scans a distribution directory for wheel and source distribution files
and writes a deterministic checksums file suitable for release verification.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SHA-256 checksums for dist artifacts")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing built artifacts")
    parser.add_argument(
        "--output",
        default="sha256sums.txt",
        help="Output checksum filename (relative to dist dir if not absolute)",
    )
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.exists() or not dist_dir.is_dir():
        print(f"[checksums] ERROR: dist directory not found: {dist_dir}")
        return 1

    artifacts = sorted(
        [*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")],
        key=lambda p: p.name,
    )
    if not artifacts:
        print(f"[checksums] ERROR: no wheel/sdist artifacts found under {dist_dir}")
        return 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = dist_dir / output_path

    lines = [f"{_sha256(artifact)}  {artifact.name}" for artifact in artifacts]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[checksums] Wrote {len(lines)} entries to {output_path}")
    for line in lines:
        print(f"[checksums] {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
