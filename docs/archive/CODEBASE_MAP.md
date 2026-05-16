# Codebase Structure Map

## Current State (Messy - Multiple Overlapping Systems)

```
Main_Research_Chula/
├── gas_analysis/                          # Core processing library
│   └── core/
│       ├── pipeline.py                    # 369 KB - BATCH processing (main)
│       ├── realtime_pipeline.py           # 33 KB - REAL-TIME processing
│       ├── preprocessing.py               # 12 KB - Signal processing
│       ├── calibration/
│       │   └── methods.py                 # 37 KB - ROI, peaks, calibration
│       └── ...
│
├── PDMS_Ring_Rectum Sensor/src/           # DUPLICATE implementations
│   ├── acquisition/
│   │   └── ccs200_realtime.py             # 28 KB - Spectrometer hardware
│   ├── analysis/
│   │   ├── realtime_analyzer.py           # 28 KB - DUPLICATE of realtime_pipeline
│   │   └── advanced_signal_processor.py    # 30 KB - Overlaps preprocessing
│   └── pipelines/
│       ├── realtime_monitoring.py         # 30 KB - Another real-time system
│       ├── unified_research_system.py     # 21 KB - Another "unified" system
│       ├── research_grade_app.py          # 28 KB - Another app
│       └── dual_fixed_wavelength_app.py   # 38 KB - Specific app
│
├── unified_research_platform/             # ANOTHER platform package
│   └── run_platform.py                    # 6 KB - Launcher
│
└── realtime_gas_sensing_system.py         # 40 KB - Main entry (uses gas_analysis)
```

## Core Logic Components (What Actually Matters)

| Component | File | Purpose | Size |
|-----------|------|---------|------|
| **Batch Pipeline** | `gas_analysis/core/pipeline.py` | Offline analysis, calibration training | 369 KB |
| **Real-time Pipeline** | `gas_analysis/core/realtime_pipeline.py` | Live processing | 33 KB |
| **Preprocessing** | `gas_analysis/core/preprocessing.py` | Denoise, baseline, normalize | 12 KB |
| **Calibration** | `gas_analysis/core/calibration/methods.py` | ROI, peaks, wavelength shift | 37 KB |
| **Acquisition** | `PDMS_Ring_Rectum Sensor/src/acquisition/ccs200_realtime.py` | Spectrometer hardware | 28 KB |

## Duplicates to Remove/Consolidate

| Duplicate | Keep | Remove |
|-----------|------|--------|
| Real-time pipeline | `gas_analysis/core/realtime_pipeline.py` | `PDMS_Ring_Rectum Sensor/src/analysis/realtime_analyzer.py` |
| Signal processing | `gas_analysis/core/preprocessing.py` | `PDMS_Ring_Rectum Sensor/src/analysis/advanced_signal_processor.py` |
| Multiple apps | `realtime_gas_sensing_system.py` | `unified_research_system.py`, `research_grade_app.py`, `realtime_monitoring.py` |

## Proposed Unified Structure

```
Main_Research_Chula/
├── src/
│   ├── acquisition/
│   │   ├── __init__.py
│   │   ├── ccs200.py              # Spectrometer interface (from PDMS_Ring_Rectum)
│   │   └── simulation.py          # Simulation fallback
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py       # Signal processing (from gas_analysis)
│   │   ├── calibration.py         # Calibration methods
│   │   └── quality.py             # Quality control
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── batch.py               # Batch processing (from pipeline.py)
│   │   └── realtime.py            # Real-time processing
│   │
│   └── analysis/
│       ├── __init__.py
│       ├── roi.py                 # ROI discovery
│       ├── peaks.py               # Peak detection
│       └── concentration.py       # Concentration estimation
│
├── config/
│   └── config.yaml                # Configuration
│
├── run.py                         # SINGLE entry point
└── requirements.txt
```

## Entry Point Usage

```bash
# Run real-time acquisition and processing
python run.py --mode realtime --duration 60

# Run batch analysis for calibration training
python run.py --mode batch --data Joy_Data/Acetone

# Run with specific spectrometer
python run.py --mode realtime --resource "USB0::0x1313::0x8089::M00505929::RAW"
```
