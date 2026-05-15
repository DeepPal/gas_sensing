# Getting Started — Industrial Integrators

## Architecture overview

```
┌─────────────────────┐     WebSocket      ┌──────────────────────┐
│  Spectrometer       │ ──────────────────► │  SpectraAgent        │
│  (CCS200 or plugin) │                     │  FastAPI + React     │
└─────────────────────┘                     │  :8765               │
                                            └────────┬─────────────┘
                                                     │ REST API
                                            ┌────────▼─────────────┐
                                            │  Your system         │
                                            │  (SCADA / LIMS / ERP)│
                                            └──────────────────────┘
```

## REST API

The OpenAPI specification is at `contracts/openapi_baseline.json`.
The interactive API docs are served at `http://localhost:8765/docs`.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Service health check |
| `POST` | `/api/acquisition/start` | Start acquisition session |
| `POST` | `/api/acquisition/stop` | Stop acquisition session |
| `GET` | `/api/sessions/{id}` | Retrieve session results |
| `POST` | `/api/reports/generate` | Generate session report |
| `WS` | `/ws/spectrum` | Live spectrum WebSocket stream |

## Docker deployment

```bash
# Production (background)
docker compose up -d spectraagent

# With custom data directory
DATA_DIR=/mnt/sensor-data docker compose up -d

# Health check
curl http://localhost:8765/api/health
```

## Adding a hardware driver (plugin)

1. Implement `spectraagent.drivers.BaseDriver` from `spectraagent/drivers/`
2. Register as an entry point in your `pyproject.toml`:

```toml
[project.entry-points."spectraagent.hardware"]
my_sensor = "my_package.driver:MyDriver"
```

3. Install your package — SpectraAgent auto-discovers it at startup.

## ONNX model export

```python
from src.public_api import CNNGasClassifier
model = CNNGasClassifier.load("output/models/cnn.pt")
model.export_onnx("output/models/cnn.onnx")
```

The exported model runs in any ONNX-compatible runtime (TensorRT, OpenVINO, ONNX Runtime).

## API contract stability

`contracts/openapi_baseline.json` is enforced in CI. Any breaking change
to a public route fails the `check_openapi_compat` gate. Pin your integration
to a tagged release for guaranteed API stability.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECTRAAGENT_PORT` | `8765` | HTTP port |
| `SPECTRAAGENT_SIMULATE` | `false` | Enable simulated hardware |
| `SPECTRAAGENT_DATA_DIR` | `output/` | Session data directory |
| `ANTHROPIC_API_KEY` | — | Required for AI agent features |
