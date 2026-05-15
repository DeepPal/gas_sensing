#!/usr/bin/env python3
"""Regenerate the committed OpenAPI contract baseline.

Use this only when an intentional API contract change has been approved.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "contracts" / "openapi_baseline.json"


def _current_openapi() -> dict:
    from spectraagent.webapp.server import create_app

    app = create_app(simulate=True)
    return app.openapi()


def main() -> int:
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = _current_openapi()
    BASELINE_PATH.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[openapi-baseline] wrote {BASELINE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
