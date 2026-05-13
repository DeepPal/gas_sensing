#!/usr/bin/env python3
"""Fast external integrator smoke check.

This script verifies that a clean local install can construct the SpectraAgent
web application and interact with core public endpoints without hardware.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from spectraagent.webapp.server import create_app

REQUIRED_OPENAPI_PATHS = {
    "/api/health",
    "/api/research-flow",
    "/api/qualification/dossier",
    "/api/reports/generate",
    "/api/sessions",
}


def _assert_status(client: TestClient, path: str, expected: int) -> dict:
    resp = client.get(path)
    if resp.status_code != expected:
        raise SystemExit(f"[integrator-smoke] {path} expected {expected}, got {resp.status_code}")
    return resp.json()


def main() -> int:
    app = create_app(simulate=True)
    with TestClient(app) as client:
        health = _assert_status(client, "/api/health", 200)
        if health.get("status") != "ok":
            raise SystemExit(f"[integrator-smoke] unexpected health payload: {health}")

        _assert_status(client, "/api/research-flow", 200)
        _assert_status(client, "/api/qualification/dossier", 200)

        openapi = _assert_status(client, "/openapi.json", 200)
        paths = set((openapi.get("paths") or {}).keys())
        missing = sorted(REQUIRED_OPENAPI_PATHS - paths)
        if missing:
            raise SystemExit(f"[integrator-smoke] missing required OpenAPI paths: {missing}")

    print("[integrator-smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
