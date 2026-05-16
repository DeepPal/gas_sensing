#!/usr/bin/env python3
"""Validate release artifacts for unexpected packaged content.

Checks wheel and source distribution files for known unwanted paths
(e.g., node_modules, caches, compiled bytecode) before publishing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import zipfile

FORBIDDEN_FRAGMENTS = [
    "node_modules/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
]

FORBIDDEN_SUFFIXES = [
    ".pyc",
    ".pyo",
]


def _is_forbidden(path: str) -> bool:
    lowered = path.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in lowered:
            return True
    return any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def _scan_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return [name for name in archive.namelist() if _is_forbidden(name)]


def _scan_sdist(path: Path) -> list[str]:
    with tarfile.open(path, mode="r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
    return [name for name in names if _is_forbidden(name)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check release artifacts for forbidden paths")
    parser.add_argument("--dist-dir", default="dist", help="Distribution directory")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.exists() or not dist_dir.is_dir():
        print(f"[artifact-hygiene] ERROR: dist directory not found: {dist_dir}")
        return 1

    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.name)
    sdists = sorted(dist_dir.glob("*.tar.gz"), key=lambda p: p.name)

    if not wheels and not sdists:
        print(f"[artifact-hygiene] ERROR: no .whl or .tar.gz artifacts found under {dist_dir}")
        return 1

    violations: list[tuple[str, str]] = []

    for wheel in wheels:
        for item in _scan_wheel(wheel):
            violations.append((wheel.name, item))

    for sdist in sdists:
        for item in _scan_sdist(sdist):
            violations.append((sdist.name, item))

    if violations:
        print("[artifact-hygiene] ERROR: forbidden paths detected in release artifacts:")
        for artifact, entry in violations[:100]:
            print(f"  - {artifact}: {entry}")
        if len(violations) > 100:
            print(f"  ... and {len(violations) - 100} more")
        return 1

    print(f"[artifact-hygiene] OK: checked {len(wheels)} wheel(s) and {len(sdists)} sdist(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
