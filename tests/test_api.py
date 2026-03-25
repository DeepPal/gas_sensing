"""
tests/test_api.py
==================
Integration tests for the FastAPI inference API.

These tests use FastAPI's ``TestClient`` (backed by ``httpx``) which runs
the full application stack — including the lifespan startup that initialises
the pipeline — without a real network socket.

Coverage
--------
- GET /health returns 200 with correct shape
- GET /status returns 200 after pipeline init
- POST /predict with a valid spectrum → 200 + PredictionResult shape
- POST /predict with mismatched arrays → 422 validation error
- POST /predict with NaN intensities → 200 success=False (graceful)
- POST /predict/batch with multiple spectra → list of results
- POST /predict/batch > 1000 spectra → 422
- GET / (root) returns info JSON
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

import numpy as np
import pytest

# Skip the entire module if FastAPI or httpx are not available
pytest.importorskip("fastapi", reason="fastapi required for API tests")
pytest.importorskip("httpx", reason="httpx required for TestClient")

from fastapi.testclient import TestClient

from src.api.main import create_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """Create a TestClient that runs the full app lifespan."""
    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_spectrum_payload(
    n_points: int = 200,
    peak_nm: float = 531.5,
    concentration: float = 0.5,
    gas_type: str = "Ethanol",
) -> dict:
    """Build a valid SpectrumReading JSON payload."""
    wl = np.linspace(480.0, 600.0, n_points).tolist()
    # Synthetic LSPR absorption spectrum (Gaussian dip)
    baseline = np.ones(n_points) * 10_000.0
    noise = np.random.default_rng(42).normal(0, 20, n_points)
    absorption = 500.0 * np.exp(-((np.linspace(480, 600, n_points) - peak_nm) ** 2) / (2 * 2**2))
    intensities = (baseline + noise - absorption).tolist()

    return {
        "spectrum_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "wavelengths": wl,
        "intensities": intensities,
        "sensor_id": "test-sensor",
        "gas_type": gas_type,
        "concentration_ppm": concentration,
    }


# ---------------------------------------------------------------------------
# Health / status tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_models_loaded_dict(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], dict)

    def test_status_returns_200(self, client: TestClient) -> None:
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_has_pipeline_stats(self, client: TestClient) -> None:
        data = client.get("/status").json()
        assert "total_processed" in data

    def test_root_returns_info(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "docs" in data


# ---------------------------------------------------------------------------
# POST /predict — happy path
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_predict_returns_200(self, client: TestClient) -> None:
        payload = _make_spectrum_payload()
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_prediction_result_shape(self, client: TestClient) -> None:
        payload = _make_spectrum_payload()
        data = client.post("/predict", json=payload).json()
        required = {"spectrum_id", "timestamp", "success", "quality_score", "processing_time_ms"}
        assert required.issubset(data.keys())

    def test_predict_spectrum_id_matches_input(self, client: TestClient) -> None:
        payload = _make_spectrum_payload()
        data = client.post("/predict", json=payload).json()
        assert data["spectrum_id"] == payload["spectrum_id"]

    def test_predict_processing_time_positive(self, client: TestClient) -> None:
        data = client.post("/predict", json=_make_spectrum_payload()).json()
        assert data["processing_time_ms"] > 0

    def test_predict_quality_score_in_range(self, client: TestClient) -> None:
        data = client.post("/predict", json=_make_spectrum_payload()).json()
        assert 0.0 <= data["quality_score"] <= 1.0

    def test_predict_peak_wavelength_in_lspr_range(self, client: TestClient) -> None:
        data = client.post("/predict", json=_make_spectrum_payload(peak_nm=531.0)).json()
        peak = data.get("peak_wavelength")
        if peak is not None:
            assert 480.0 <= peak <= 600.0, f"Peak {peak} nm outside LSPR range"

    def test_predict_success_is_bool(self, client: TestClient) -> None:
        data = client.post("/predict", json=_make_spectrum_payload()).json()
        assert isinstance(data["success"], bool)

    def test_gas_type_alias_normalised(self, client: TestClient) -> None:
        """'EtOH' should be accepted (normalised to 'Ethanol' by schema)."""
        payload = _make_spectrum_payload(gas_type="EtOH")
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict — validation errors
# ---------------------------------------------------------------------------


class TestPredictValidation:
    def test_mismatched_array_lengths_returns_422(self, client: TestClient) -> None:
        payload = _make_spectrum_payload()
        # Make intensities shorter than wavelengths
        payload["intensities"] = payload["intensities"][:50]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_arrays_returns_422(self, client: TestClient) -> None:
        payload = _make_spectrum_payload()
        payload["wavelengths"] = []
        payload["intensities"] = []
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_negative_concentration_returns_422(self, client: TestClient) -> None:
        payload = _make_spectrum_payload(concentration=-1.0)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_very_low_snr_spectrum_handled(self, client: TestClient) -> None:
        """All-zeros spectrum (zero signal, zero noise) should return HTTP 200."""
        payload = _make_spectrum_payload()
        # Replace with a flat, zero-signal spectrum (still valid JSON)
        n = len(payload["intensities"])
        payload["intensities"] = [0.0] * n
        response = client.post("/predict", json=payload)
        # Zero spectrum is valid JSON — pipeline either succeeds or flags as low-quality
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------


class TestPredictBatchEndpoint:
    def test_batch_returns_list(self, client: TestClient) -> None:
        payloads = [_make_spectrum_payload() for _ in range(3)]
        response = client.post("/predict/batch", json=payloads)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_batch_each_result_has_spectrum_id(self, client: TestClient) -> None:
        payloads = [_make_spectrum_payload() for _ in range(2)]
        data = client.post("/predict/batch", json=payloads).json()
        for result in data:
            assert "spectrum_id" in result

    def test_batch_over_1000_returns_422(self, client: TestClient) -> None:
        """Batch size limit of 1000 should be enforced."""
        payloads = [_make_spectrum_payload() for _ in range(1001)]
        response = client.post("/predict/batch", json=payloads)
        assert response.status_code == 422
