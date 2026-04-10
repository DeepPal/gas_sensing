import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from spectraagent.webapp.server import create_app


def _client() -> TestClient:
    app = create_app(simulate=True)
    return TestClient(app)


def test_openapi_includes_stable_core_routes() -> None:
    with _client() as client:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()

    paths = spec.get("paths", {})
    expected_paths = {
        "/api/health",
        "/api/research-flow",
        "/api/qualification/dossier",
        "/api/qualification/dossier/export",
        "/api/qualification/package",
        "/api/reports/generate",
        "/api/agents/ask",
        "/api/sessions",
        "/api/sessions/{session_id}",
    }
    missing = sorted(expected_paths - set(paths.keys()))
    assert not missing, f"OpenAPI missing expected contract routes: {missing}"


def test_health_contract_has_required_fields() -> None:
    with _client() as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        payload = resp.json()

    required_top = {
        "status",
        "version",
        "hardware",
        "simulate",
        "claude_api_key_configured",
        "knowledge_base_available",
        "knowledge_context_mode",
        "rate_limits",
    }
    assert required_top.issubset(payload.keys())
    assert payload["status"] == "ok"

    rate_limits = payload.get("rate_limits", {})
    assert "claude" in rate_limits
    assert "report" in rate_limits


def test_reports_generate_contract_exposes_source_and_notice() -> None:
    with _client() as client:
        client.app.state.last_session_analysis = SimpleNamespace(
            calibration_n_points=8,
            calibration_r2=0.985,
            mean_snr=4.2,
            lod_ppm=0.012,
            loq_ppm=0.040,
            drift_rate_nm_per_frame=0.001,
            summary_text="Contract test session.",
            audit={"checks": []},
        )
        client.app.state.last_session_id = "contract-session"

        resp = client.post(
            "/api/reports/generate",
            content=json.dumps({"session_id": "contract-session"}),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200
        payload = resp.json()

    assert "report" in payload
    assert "report_source" in payload
    assert payload["report_source"] in {"deterministic", "claude"}
    if "report_notice" in payload:
        assert isinstance(payload["report_notice"], str)
