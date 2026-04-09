# SCHEMA.md — Data Schema & Contracts

## Core Data Contract

Every spectrum flowing through this system is represented as a `SpectrumReading`.
This is the **canonical unit of data** — from acquisition to storage to model input.

```json
{
  "spectrum_id": "uuid4-string",
  "timestamp": "2026-03-19T10:23:45.123Z",

  "wavelengths": [399.8, 400.0, 400.2, "..."],
  "intensities": [10023.4, 10018.7, "..."],

  "sensor_id": "CCS200-LAB-01",
  "gas_type": "Ethanol",
  "concentration_ppm": 0.5,

  "temperature_c": null,
  "humidity_pct": null,
  "reference_spectrum": null,
  "dark_spectrum": null,
  "integration_time_ms": 50.0
}
```

### Field Rules

| Field | Required | Type | Constraints |
|-------|----------|------|-------------|
| `spectrum_id` | Yes | `str` (UUID4) | Unique per spectrum |
| `timestamp` | Yes | ISO-8601 datetime | UTC preferred |
| `wavelengths` | Yes | `list[float]` | Monotonically increasing; len ≥ 2 |
| `intensities` | Yes | `list[float]` | `len == len(wavelengths)`; finite values |
| `sensor_id` | Yes | `str` | e.g. `"CCS200-LAB-01"`, `"simulation"` |
| `gas_type` | Yes | `str` | See gas type vocabulary below |
| `concentration_ppm` | Yes | `float` | `≥ 0.0`; use `0.0` if unknown |
| `temperature_c` | No | `float \| null` | Physical range −40 to +150 |
| `humidity_pct` | No | `float \| null` | 0 to 100 |
| `reference_spectrum` | No | `list[float] \| null` | Same length as `intensities` |
| `dark_spectrum` | No | `list[float] \| null` | Same length as `intensities` |
| `integration_time_ms` | No | `float \| null` | CCS200: 1–60000 ms |

### Gas Type Vocabulary

```
"Ethanol"      → EtOH, C2H5OH
"IPA"          → Isopropanol, C3H7OH
"Methanol"     → MeOH, CH3OH
"MixVOC"       → Mixed gas (any combination)
"unknown"      → Label not available (inference mode)
```

**Normalization rules:**
- `"EtOH"` → normalize to `"Ethanol"` at ingest
- `"IPA"`, `"Isopropanol"` → normalize to `"IPA"`
- `"MeOH"`, `"Methanol"` → normalize to `"Methanol"`
- `"Mix"`, `"Mixed"` → normalize to `"MixVOC"`

---

## Prediction Contract

The `PredictionResult` is the system's output. It pairs with a `SpectrumReading` by `spectrum_id`.

```json
{
  "spectrum_id": "uuid4-string",
  "timestamp": "2026-03-19T10:23:45.234Z",

  "peak_wavelength": 531.24,
  "wavelength_shift_nm": -0.26,
  "concentration_ppm": 2.24,
  "concentration_std_ppm": 0.18,

  "gas_type_predicted": "Ethanol",
  "gas_type_confidence": 0.91,

  "snr": 18.4,
  "quality_score": 0.87,
  "success": true,
  "processing_time_ms": 12.3,
  "model_version": "v2.1.0",
  "pipeline_version": "3.0.0"
}
```

### Quality Score Interpretation

| `quality_score` | Interpretation | Action |
|-----------------|----------------|--------|
| ≥ 0.80 | High confidence | Accept |
| 0.50–0.79 | Marginal | Accept with warning |
| < 0.50 | Low confidence | Flag for review |
| `success=false` | Pipeline failure | Discard |

---

## Session Record

A session groups all spectra acquired in a single experimental run.

```json
{
  "session_id": "20260319_102345",
  "started_at": "2026-03-19T10:23:45Z",
  "stopped_at": "2026-03-19T11:23:45Z",
  "gas_type": "Ethanol",
  "sensor_id": "CCS200-LAB-01",
  "total_spectra": 7200,
  "valid_spectra": 7188,
  "model_version": "v2.1.0",
  "config_hash": "sha256-abc123...",
  "notes": ""
}
```

---

## File Formats

### Raw Experimental CSV (Joy_Data format)

Two-column CSV, no header or with header `wavelength,intensity`:

```csv
399.800,10023.4
400.012,10018.7
400.224,10022.1
...
```

- Folder name encodes gas type and concentration: `"0.5 ppm EtOH IPA MeOH-1/"`
- Each CSV = one spectrum acquisition (~5 seconds at 20 Hz)

### Processed Parquet (output format)

Columnar format for batch analysis results:

```
timestamp | gas_type | concentration_ppm | peak_wavelength | wavelength_shift_nm | snr | quality_score
```

Stored in `data/processed/{gas_type}/{YYYYMMDD_HHMMSS}.parquet`

### Session CSV (real-time output)

Per-frame CSV in `output/sessions/{YYYYMMDD_HHMMSS}/pipeline_results.csv`:

```csv
timestamp,sample_id,peak_wavelength,wavelength_shift_nm,concentration_ppm,
concentration_std_ppm,gas_type_predicted,gas_type_confidence,snr,quality_score,success
```

---

## YAML Config Schema

Top-level sections of `configs/config.yaml`:

```yaml
sensor:        # Hardware acquisition parameters
preprocessing: # Signal processing parameters
roi:           # Region-of-interest discovery + shift calculation
calibration:   # Calibration model selection + hyperparameters
quality:       # SNR thresholds, saturation limits
response_series: # T90/T10 temporal gating
mlflow:        # Experiment tracking URI + experiment name
api:           # FastAPI host, port, model paths
```

---

## Directory Layout for Data

```
data/
├── raw/                         # Original experimental files (gitignored)
│   ├── Joy_Data/
│   │   ├── Ethanol/
│   │   │   ├── 0.5 ppm-1/       # 5-min session at 0.5 ppm
│   │   │   │   └── *.csv
│   │   │   └── 1 ppm-1/
│   │   ├── IPA/
│   │   ├── MeOH/
│   │   └── Mixed gas/
│   └── ...
├── processed/                   # Processed + standardized (gitignored)
│   ├── Ethanol_0.5ppm.parquet
│   └── ...
└── metadata/
    ├── reference_spectra/        # Baseline spectra per gas per date
    │   └── EtOH_ref_20260319.csv
    └── calibration/
        └── calibration_params.json
```
