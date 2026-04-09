# Unified Gas Sensing Research Platform

## Single Professional Workflow

This is now a **single unified research platform** that integrates real-time spectrometer acquisition with advanced gas analysis into one cohesive system.

---

## Project Structure

```
Main_Research_Chula/
│
├── realtime_gas_sensing_system.py    # SINGLE ENTRY POINT
│
├── PDMS_Ring_Rectum Sensor/          # Real-time acquisition hardware
│   ├── src/acquisition/              # Spectrometer drivers
│   │   ├── ccs200_realtime.py        # Core acquisition service
│   │   └── device_discovery.py       # Hardware discovery
│   └── src/pipelines/                # Acquisition pipelines
│
├── gas_analysis/                     # Advanced analysis pipeline
│   ├── core/                         # Core analysis modules
│   │   ├── pipeline.py               # Main analysis pipeline
│   │   ├── run_each_gas.py           # Batch analysis runner
│   │   └── signal_proc.py            # Signal processing
│   ├── calibration/                  # Calibration methods
│   │   ├── wavelength_shift.py       # Δλ calibration
│   │   └── absorbance_amplitude.py   # ΔA calibration
│   └── utils/                        # Utilities
│
├── dashboard/                        # Web visualization
│   └── app.py                        # Streamlit dashboard
│
├── Joy_Data/                         # Experimental data
├── output/                           # Analysis results
├── logs/                             # System logs
├── realtime_data/                    # Real-time session data
└── reports/                          # Session reports
```

---

## Single Entry Point

**Launch the entire system with one command:**

```bash
python realtime_gas_sensing_system.py
```

---

## Unified Workflow

### Real-Time Acquisition + Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED RESEARCH PLATFORM                    │
│                                                                  │
│  1. Spectrometer Acquisition (PDMS_Ring_Rectum Sensor)         │
│     ↓                                                            │
│  2. Quality Control & Validation (Real-time System)            │
│     ↓                                                            │
│  3. Signal Processing (gas_analysis/core/signal_proc.py)        │
│     ↓                                                            │
│  4. Calibration Analysis (gas_analysis/calibration/)            │
│     ↓                                                            │
│  5. Results & Visualization (dashboard/app.py)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Research-Level Analysis Features

### 1. Real-Time Acquisition
- ThorLabs CCS200 spectrometer integration
- Dual spectrometer support for differential sensing
- Configurable integration time (default: 30 ms)
- Target wavelength monitoring (default: 532 nm)

### 2. Quality Control
- **SNR Threshold**: Minimum signal-to-noise ratio (10.0)
- **Noise Validation**: Maximum acceptable noise level (0.1)
- **Saturation Detection**: Automatic flagging
- **Confidence Scoring**: Real-time metrics (0-1)

### 3. Signal Processing
- Wavelet denoising
- ALS baseline correction
- Smoothing algorithms
- ROI discovery

### 4. Calibration Methods
- **Wavelength Shift (Δλ)**: Peak position tracking
- **Absorbance Amplitude (ΔA)**: Intensity change analysis
- **GPR Calibration**: Probabilistic uncertainty quantification
- **1D CNN**: Blind gas classification

### 5. Metrics
- R² correlation
- LOD (Limit of Detection)
- LOQ (Limit of Quantification)
- T90/T10 response times
- SNR (Signal-to-Noise Ratio)
- Confidence scores

---

## System Menu

```
📋 SYSTEM MENU
1. 🔍 Discover Spectrometers       - Auto-detect connected hardware
2. 📋 Load Calibration             - Load or auto-detect calibration
3. 🚀 Start Real-Time Acquisition  - Live data capture & analysis
4. 📊 Run Batch Analysis           - Process existing data
5. 🌐 Launch Dashboard             - Interactive visualization
6. 📊 Generate Report              - Session summary
7. 🛑 Stop System                  - Halt acquisition
8. 🚪 Exit                         - Quit system
```

---

## Typical Research Workflows

### Workflow 1: Real-Time Experiment

```bash
1. Launch system
   $ python realtime_gas_sensing_system.py

2. Discover spectrometers (Option 1)
   
3. Load calibration (Option 2)
   - Auto-detects latest calibration from output/

4. Start real-time acquisition (Option 3)
   - Enter spectrometer resource string
   - Set duration (optional)
   - System automatically:
     ✓ Captures spectra
     ✓ Validates quality
     ✓ Analyzes concentration
     ✓ Records to CSV
     ✓ Generates live metrics

5. Generate report (Option 6)
   - Comprehensive session summary
```

### Workflow 2: Batch Analysis

```bash
1. Launch system
   $ python realtime_gas_sensing_system.py

2. Run batch analysis (Option 4)
   - Select gas: Ethanol, Isopropanol, Methanol, MixVOC
   - System uses existing gas_analysis pipeline
   - Generates calibration files

3. Launch dashboard (Option 5)
   - Visualize results
   - Compare calibrations
   - Interactive exploration

4. Generate report (Option 6)
```

### Workflow 3: Calibration Development

```bash
1. Run batch analysis on known concentrations
   - Option 4 → Select gas type

2. Review calibration metrics
   - Check output/{gas}_topavg/calibration_metrics.json

3. Load calibration for real-time use
   - Option 2 → Auto-detect or specify file

4. Validate with real-time acquisition
   - Option 3 → Run with known gas concentrations
   - Compare estimated vs. known values
```

---

## Integration Points

### Real-Time ↔ Batch Analysis

```
Real-Time Acquisition
        ↓
   Save to CSV
        ↓
Batch Analysis Pipeline (can process later)
        ↓
Generate Calibration
        ↓
Load Calibration → Use in Real-Time
```

### Analysis ↔ Dashboard

```
Analysis Results (JSON/CSV)
        ↓
Dashboard Visualization
        ↓
Interactive Exploration
        ↓
Export Figures for Publication
```

---

## Data Flow

```
Spectrometer Hardware (PDMS_Ring_Rectum Sensor)
        ↓
RealtimeAcquisitionService (ccs200_realtime.py)
        ↓
Quality Control (realtime_gas_sensing_system.py)
        ↓
Signal Processing (gas_analysis/core/signal_proc.py)
        ↓
Calibration (gas_analysis/calibration/)
        ↓
Results (CSV + JSON)
        ↓
Dashboard (Streamlit)
```

---

## Output Files

### Real-Time Session

```
realtime_data/
├── spectrum_data_YYYYMMDD_HHMMSS.csv      # Raw spectra
└── analysis_results_YYYYMMDD_HHMMSS.csv   # Concentration estimates

logs/
└── system_YYYYMMDD_HHMMSS.log             # Complete audit trail

reports/
└── session_report_YYYYMMDD_HHMMSS.md      # Summary report
```

### Batch Analysis

```
output/
└── {gas}_topavg/
    ├── calibration_metrics.json           # Calibration parameters
    ├── roi_summary.json                   # ROI information
    ├── delta_lambda_plot.png              # Δλ vs concentration
    └── summary_table.csv                  # Results table
```

---

## Quality Assurance

### Automatic Validation

Every sample is validated against quality gates:

```python
✓ SNR > 10.0
✓ Noise level < 0.1
✓ No saturation
✓ Confidence > 0.5
```

### Audit Trail

All operations logged with:
- Timestamp
- Operation type
- Parameters used
- Quality metrics
- Error tracebacks (if any)

---

## Publication-Ready Features

### Defensible Methodology

- Quality gates documented
- Calibration parameters recorded
- Confidence intervals provided
- Complete audit trail maintained

### Reproducibility

- Configuration saved per session
- Version tracking
- Parameter logging
- Data provenance

### Statistical Rigor

- Confidence scoring
- Error tracking
- Quality filtering statistics
- LOD/LOQ calculations

---

## Configuration

### System Configuration

Located in `realtime_gas_sensing_system.py`:

```python
@dataclass
class SystemConfig:
    integration_time_ms: float = 30.0
    target_wavelength: float = 532.0
    buffer_size: int = 10000
    min_snr: float = 10.0
    max_noise_level: float = 0.1
    confidence_threshold: float = 0.5
```

### Analysis Configuration

Located in `config/config.yaml`:

```yaml
signal_processing:
  denoising:
    method: "wavelet"
    wavelet: "db4"
    level: 3
  
  baseline_correction:
    method: "als"
    lambda: 1e5
    p: 0.01

roi:
  discovery:
    method: "gradient"
    min_width_nm: 5.0
```

---

## Hardware Requirements

### Essential

- ThorLabs CCS200 Spectrometer
- USB 2.0/3.0 connection
- ThorLabs drivers installed

### Optional

- Second CCS200 for differential sensing
- Gas delivery system
- Temperature control

---

## Software Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
torch>=1.9.0
streamlit>=1.0.0
plotly>=5.0.0
PyWavelets>=1.1.0
pyyaml>=5.4.0
```

---

## Quick Reference

| Task | Command/Option |
|------|---------------|
| Launch system | `python realtime_gas_sensing_system.py` |
| Discover hardware | Option 1 |
| Load calibration | Option 2 |
| Start acquisition | Option 3 |
| Batch analysis | Option 4 |
| View dashboard | Option 5 |
| Generate report | Option 6 |
| Stop | Option 7 or Ctrl+C |

---

## Support Files

- **System Guide**: `REALTIME_SYSTEM_GUIDE.md`
- **Code Map**: `CODEMAP.md`
- **Documentation**: `DOCUMENTATION.md`
- **Main README**: `README.md`

---

## Version

**Version 2.0.0** - Unified Professional Research Platform

- Single entry point
- Integrated workflows
- Research-level analysis
- Publication-ready outputs
- Comprehensive quality control
