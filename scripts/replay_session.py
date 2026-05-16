#!/usr/bin/env python3
"""Replay integrity verifier for reproducibility manifests.

This command verifies that a session directory still matches the checksums
captured in its manifest file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dashboard.reproducibility import verify_manifest_artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify reproducibility manifest artifacts.")
    parser.add_argument("manifest", type=Path, help="Path to <session_id>_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest

    if not manifest_path.exists():
        print(f"[replay] Manifest not found: {manifest_path}")
        return 2

    ok, errors = verify_manifest_artifacts(manifest_path)
    if ok:
        print(f"[replay] OK: {manifest_path}")
        return 0

    print(f"[replay] FAILED: {manifest_path}")
    for err in errors:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
