#!/usr/bin/env python3
"""Validate canonical status-tracking references stay aligned across docs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_FILES = [
    "REMAINING_WORK.md",
    "PRODUCTION_READINESS.md",
    "CHANGELOG.md",
    ".github/workflows/security.yml",
]

REQUIRED_DOCS = [
    "README.md",
    "CONTRIBUTING.md",
    "REMAINING_WORK.md",
    "PRODUCTION_READINESS.md",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> int:
    errors: list[str] = []

    # Ensure canonical files exist.
    for rel in CANONICAL_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            errors.append(f"missing canonical file: {rel}")

    # Ensure each required doc references every canonical file path.
    for rel_doc in REQUIRED_DOCS:
        path = REPO_ROOT / rel_doc
        if not path.exists():
            errors.append(f"missing required doc: {rel_doc}")
            continue

        text = read_text(path)
        for canonical in CANONICAL_FILES:
            if canonical not in text:
                errors.append(
                    f"{rel_doc} is missing canonical reference: {canonical}"
                )

    if errors:
        print("[status-sync] FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("[status-sync] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
