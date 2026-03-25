# Au-MIP LSPR Gas Sensing Platform

**Commercial-grade optical gas sensing platform** built on Au nanoparticle
molecularly-imprinted polymer (MIP) localised surface plasmon resonance (LSPR).

---

## Quick Start

=== "Batch analysis"

    ```bash
    python run.py --mode batch --data data/JOY_Data/Ethanol
    ```

=== "Live sensor"

    ```bash
    python run.py --mode sensor --gas Ethanol --duration 3600
    ```

=== "Dashboard"

    ```bash
    .venv/Scripts/python.exe -m streamlit run dashboard/app.py
    ```

=== "REST API"

    ```bash
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    # Interactive docs: http://localhost:8000/docs
    ```

---

## Platform Overview

```
┌──────────────────────────────────────────────────────────┐
│                 Au-MIP LSPR Gas Sensor                   │
│          Peak wavelength shift: Δλ ≈ −10 nm              │
│          at 0.1 ppm EtOH  (ref ~717.9 nm)                │
└───────────────────────┬──────────────────────────────────┘
                        │ CCS200 USB spectrometer
                        ▼
┌──────────────────────────────────────────────────────────┐
│               Acquisition Layer (src/acquisition)        │
│  • CCS200 real-time driver  • 2.4 Hz @ 50 ms integration │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│              Preprocessing  (src/preprocessing)          │
│  • Baseline correction (ALS)  • Denoising (Savitzky-Golay│
│  • Normalisation (area/min-max/z-score/SNV)              │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│            Feature Extraction  (src/features)            │
│  • LSPR: [Δλ, ΔI_peak, ΔI_area, ΔI_std]                 │
└───────────────────────┬──────────────────────────────────┘
                        ▼
┌───────────────────────────────────────────────────────────────────┐
│                    Inference  (src/inference, src/models)         │
│  • CNN gas classifier  • Gaussian Process Regressor (uncertainty) │
│  • Calibration isotherms (Langmuir / Freundlich / Hill)           │
└───────────────────────┬───────────────────────────────────────────┘
                        ▼
┌──────────────────────────────────────────────────────────┐
│         Scientific Reporting  (src/scientific)           │
│  • ICH Q2(R1) LoD/LoQ  • Selectivity matrix             │
└──────────────────────────────────────────────────────────┘
```

---

## Key Capabilities

| Capability | Details |
|---|---|
| **Gases** | Ethanol, Isopropanol, Methanol, mixed VOC |
| **LoD** | ~0.05 ppm (Ethanol, 3σ/slope method, ICH Q2) |
| **Concentration range** | 0.1 – 10 ppm |
| **Acquisition rate** | 2.4 Hz (50 ms integration) |
| **Uncertainty** | GPR posterior σ per prediction |
| **REST API** | FastAPI, OpenAPI 3.0 spec at `/docs` |
| **Export** | ONNX for edge deployment |

---

## Navigation

- **Architecture** — system design and agent pipeline
- **Engineering Standards** — code quality, testing, CI gates
- **Validation (ICH Q2)** — analytical method validation report
- **API Reference** — auto-generated from typed `src/` package
- **ADRs** — architectural decision records
