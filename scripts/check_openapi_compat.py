#!/usr/bin/env python3
"""Fail CI when public API contracts regress versus the committed OpenAPI baseline.

Checks are intentionally conservative:
- No baseline path may disappear.
- No baseline HTTP method may disappear from an existing path.
- No baseline response status code may disappear from an existing operation.

Additive changes are allowed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
import sys

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "contracts" / "openapi_baseline.json"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return cast(dict, json.loads(path.read_text(encoding="utf-8")))  # type: ignore[no-any-return]


def _current_openapi() -> dict:
    from spectraagent.webapp.server import create_app

    app = create_app(simulate=True)
    return app.openapi()


def _compare(baseline: dict, current: dict) -> list[str]:
    errors: list[str] = []
    baseline_paths = baseline.get("paths", {})
    current_paths = current.get("paths", {})

    for path, baseline_ops in baseline_paths.items():
        if path not in current_paths:
            errors.append(f"missing API path: {path}")
            continue

        current_ops = current_paths[path]
        for method, baseline_op in baseline_ops.items():
            method_key = method.lower()
            if method_key not in current_ops:
                errors.append(f"missing operation: {method_key.upper()} {path}")
                continue

            baseline_responses = (baseline_op or {}).get("responses", {})
            current_responses = (current_ops[method_key] or {}).get("responses", {})
            for status_code in baseline_responses:
                if status_code not in current_responses:
                    errors.append(
                        f"missing response status for {method_key.upper()} {path}: {status_code}"
                    )

    return errors


def main() -> int:
    if not BASELINE_PATH.exists():
        print(f"[openapi-compat] FAIL: baseline not found at {BASELINE_PATH.relative_to(ROOT)}")
        print("Run: python scripts/update_openapi_baseline.py")
        return 1

    baseline = _load_json(BASELINE_PATH)
    current = _current_openapi()
    errors = _compare(baseline, current)

    if errors:
        print("[openapi-compat] FAIL")
        for err in errors:
            print(f"- {err}")
        print("\nIf this change is intentional, regenerate and review baseline:")
        print("  python scripts/update_openapi_baseline.py")
        return 1

    print("[openapi-compat] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
