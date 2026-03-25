# Real-Time Gas Sensing System - Professional Research Solution

## Overview

This is a **professional-grade, publication-ready** real-time gas sensing system that integrates spectrometer data acquisition, recording, and processing into a single cohesive platform. Designed for high-quality research with robust error handling, comprehensive validation, and automatic quality control.

**Version:** 2.0.0  
**Author:** DeepPal Research Team, Chula University  
**Status:** Production Ready

---

## System Architecture

### Single Unified Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME GAS SENSING SYSTEM                 │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Spectrometer │───▶│  Processing  │───▶│   Analysis   │      │
│  │  Acquisition │    │  & Quality   │    │  & Recording │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│    ┌────────┐          ┌────────┐         ┌────────┐           │
│    │  Live  │          │ Quality│         │Results │           │
│    │  Data  │          │ Gates  │         │  CSV   │           │
│    └────────┘          └────────┘         └────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Real-Time Acquisition** - Continuous spectrometer data capture
2. **Quality Control** - Automatic data validation and filtering
3. **Live Processing** - Real-time signal processing and analysis
4. **Calibration Management** - Automatic calibration loading and validation
5. **Data Recording** - Professional CSV logging with metadata
6. **Batch Analysis** - Integration with existing gas analysis pipeline
7. **Dashboard** - Interactive web interface for visualization

---

## Quick Start

### 1. Launch System

```bash
cd Main_Research_Chula
python realtime_gas_sensing_system.py
```

### 2. System Menu

```
📋 SYSTEM MENU
1. 🔍 Discover Spectrometers
2. 📋 Load Calibration
3. 🚀 Start Real-Time Acquisition
4. 📊 Run Batch Analysis
5. 🌐 Launch Dashboard
6. 📊 Generate Report
7. 🛑 Stop System
8. 🚪 Exit
```

### 3. Typical Workflow

**For Real-Time Analysis:**
1. Discover spectrometers (Option 1)
2. Load calibration (Option 2)
3. Start real-time acquisition (Option 3)
4. Generate report (Option 6)

**For Batch Analysis:**
1. Run batch analysis (Option 4)
2. Launch dashboard (Option 5)
3. Generate report (Option 6)

---

## Professional Features

### 🔬 Quality Control System

**Automatic Data Validation:**
- **SNR Threshold** - Minimum signal-to-noise ratio (default: 10.0)
- **Noise Level Check** - Maximum acceptable noise (default: 0.1)
- **Saturation Detection** - Automatic flagging of saturated signals
- **Confidence Scoring** - Real-time confidence metrics

**Quality Gates:**
```python
# Data is only accepted if:
✓ SNR > 10.0
✓ Noise level < 0.1
✓ No saturation detected
✓ Confidence score > 0.5
```

### 📊 Real-Time Analysis

**Supported Methods:**
- Wavelength Shift (Δλ) Analysis
- Absorbance Amplitude (ΔA) Analysis
- Real-time concentration estimation
- Confidence-weighted results

**Live Metrics:**
- Signal-to-noise ratio (SNR)
- Concentration (ppm)
- Wavelength shift (nm)
- Confidence score (0-1)

### 📁 Professional Data Management

**Automatic File Organization:**
```
Main_Research_Chula/
├── realtime_data/
│   ├── spectrum_data_20240123_153000.csv
│   └── analysis_results_20240123_153000.csv
├── logs/
│   └── system_20240123_153000.log
└── reports/
    └── session_report_20240123_153000.md
```

**CSV Format:**
- Timestamped entries
- Quality metrics included
- Metadata preserved
- Publication-ready format

### 🔒 Robust Error Handling

**Comprehensive Validation:**
- Dependency checking at startup
- Calibration file validation
- Spectrometer connection verification
- Automatic fallback to simulation mode

**Professional Logging:**
- All events logged to file
- Error traceback captured
- Session tracking
- Audit trail maintained

---

## Configuration

### System Configuration

```python
@dataclass
class SystemConfig:
    # Acquisition settings
    integration_time_ms: float = 30.0      # Spectrometer integration time
    target_wavelength: float = 532.0       # Target wavelength (nm)
    buffer_size: int = 10000               # Data buffer size
    
    # Quality control
    min_snr: float = 10.0                  # Minimum SNR threshold
    max_noise_level: float = 0.1           # Maximum noise level
    confidence_threshold: float = 0.5      # Minimum confidence
    
    # Recording settings
    auto_save_interval: int = 100          # Save every N samples
    max_file_size_mb: int = 100            # Maximum file size
    
    # Analysis settings
    enable_realtime_analysis: bool = True  # Enable real-time analysis
    calibration_method: str = "wavelength_shift"  # Analysis method
```

### Customizing Configuration

```python
from realtime_gas_sensing_system import RealTimeGasSensingSystem, SystemConfig

# Create custom configuration
config = SystemConfig(
    integration_time_ms=50.0,
    target_wavelength=808.0,
    min_snr=15.0
)

# Initialize system with custom config
system = RealTimeGasSensingSystem(config=config)
```

---

## Calibration Management

### Loading Calibration

**Automatic Detection:**
```bash
# System automatically finds latest calibration
Option 2 → Press Enter (auto-detect)
```

**Manual Loading:**
```bash
# Specify calibration file
Option 2 → Enter path to calibration_metrics.json
```

### Calibration Validation

The system validates:
- ✓ JSON format correctness
- ✓ Required fields present
- ✓ Valid calibration method
- ✓ Proper ROI format
- ✓ Numerical values reasonable

### Required Calibration Fields

```json
{
  "calibration_method": "wavelength_shift",
  "roi_wavelengths": [520.0, 540.0],
  "baseline_wavelength": 532.0,
  "wavelength_shift_slope": 0.116,
  "wavelength_shift_intercept": 0.0
}
```

---

## Real-Time Acquisition

### Hardware Mode

**Requirements:**
- ThorLabs CCS200 spectrometer connected
- Proper drivers installed
- Valid resource string

**Steps:**
1. Discover spectrometers (Option 1)
2. Select spectrometer resource
3. Start acquisition (Option 3)
4. Monitor live data

### Simulation Mode

**For Testing/Demo:**
- Automatically activated if hardware unavailable
- Generates realistic spectral data
- Simulates gas absorption features
- Full quality control active

**Running Simulation:**
```bash
Option 3 → Press Enter (no resource string)
```

---

## Data Quality Metrics

### Quality Metrics Tracked

```python
@dataclass
class QualityMetrics:
    snr: float                    # Signal-to-noise ratio
    noise_level: float            # Noise standard deviation
    saturation_flag: bool         # Saturation detection
    valid_data: bool              # Overall validity
    confidence_score: float       # Confidence (0-1)
```

### Quality Statistics

**Session Summary Includes:**
- Total samples collected
- Valid samples (passed quality gates)
- Quality-filtered samples
- Analysis errors
- SNR statistics (mean, min, max)
- Concentration statistics

---

## Integration with Existing Pipeline

### Batch Analysis Integration

The system integrates seamlessly with your existing gas analysis pipeline:

```bash
Option 4 → Select gas type
- Ethanol
- Isopropanol
- Methanol
- MixVOC
```

**Automatically:**
- Runs existing batch pipeline
- Generates summary tables
- Updates calibration files
- Creates output directories

### Dashboard Integration

Launches your existing Streamlit dashboard:

```bash
Option 5 → Dashboard launches
```

**Features:**
- Real-time visualization
- Historical data analysis
- Interactive plots
- Model comparison

---

## Output Files

### Spectrum Data CSV

```csv
timestamp,sample_id,target_intensity,snr,noise_level,transmittance,absorbance,wavelength
2024-01-23 15:30:00,RT_000001,10234.5,45.2,123.4,0.98,0.009,532.0
2024-01-23 15:30:01,RT_000002,10198.3,44.8,125.6,0.97,0.013,532.0
```

### Analysis Results CSV

```csv
timestamp,sample_id,concentration_ppm,wavelength_shift_nm,peak_wavelength,snr,confidence_score
2024-01-23 15:30:05,RT_000005,2.34,0.27,532.27,48.3,0.87
2024-01-23 15:30:06,RT_000006,2.41,0.28,532.28,47.9,0.86
```

### Session Report (Markdown)

```markdown
# Real-Time Gas Sensing Session Report

**Session ID:** 20240123_153000
**Generated:** 2024-01-23 15:45:00

## Data Summary
- Total Samples: 1000
- Mean SNR: 45.3
- Min SNR: 12.1
- Max SNR: 67.8

## Analysis Results
- Total Analyses: 950
- Mean Concentration: 2.34 ppm
- Min Concentration: 1.89 ppm
- Max Concentration: 2.78 ppm
```

---

## Professional Best Practices

### 1. Pre-Session Checklist

- [ ] Verify spectrometer connection
- [ ] Check calibration file exists
- [ ] Confirm sufficient disk space
- [ ] Review quality thresholds
- [ ] Set appropriate duration

### 2. During Acquisition

- Monitor SNR values
- Watch quality filtering rate
- Check concentration trends
- Verify file sizes
- Monitor system logs

### 3. Post-Session

- Generate summary report
- Review quality statistics
- Validate output files
- Archive session data
- Document any issues

### 4. Data Management

- Regular backup of realtime_data/
- Archive old sessions
- Maintain calibration history
- Version control configurations
- Document parameter changes

---

## Troubleshooting

### Common Issues

**1. No Spectrometers Found**
```
Solution:
- Check USB connections
- Verify ThorLabs drivers installed
- Try manual resource entry
- Use simulation mode for testing
```

**2. Calibration Loading Failed**
```
Solution:
- Run batch analysis first
- Check JSON file format
- Verify required fields
- Check file permissions
```

**3. High Quality Filtering Rate**
```
Solution:
- Lower SNR threshold
- Increase integration time
- Check spectrometer settings
- Review noise sources
```

**4. Low Confidence Scores**
```
Solution:
- Improve signal quality
- Adjust integration time
- Check reference spectrum
- Verify calibration accuracy
```

---

## Publication-Ready Features

### Data Quality Assurance

- **Automatic validation** - Every sample checked
- **Quality metrics** - Comprehensive tracking
- **Audit trail** - Complete logging
- **Reproducibility** - Session tracking

### Statistical Rigor

- **Confidence intervals** - Real-time scoring
- **Quality gates** - Defensible thresholds
- **Error tracking** - Comprehensive logging
- **Metadata preservation** - Full context

### Documentation

- **Session reports** - Automatic generation
- **Method tracking** - Calibration logging
- **Parameter recording** - Configuration saved
- **Version tracking** - System versioned

---

## Advanced Usage

### Programmatic Control

```python
from realtime_gas_sensing_system import RealTimeGasSensingSystem, SystemConfig

# Create custom configuration
config = SystemConfig(
    integration_time_ms=50.0,
    target_wavelength=808.0,
    min_snr=15.0
)

# Initialize system
system = RealTimeGasSensingSystem(config=config)

# Load calibration
system.load_calibration("output/ethanol_topavg/calibration_metrics.json")
system.is_analyzing = True

# Run acquisition
system.start_realtime_acquisition(
    resource_string="USB0::0x1313::0x8089::M00505929::RAW",
    duration_seconds=300
)

# Generate report
system.generate_summary_report()
```

### Integration with Analysis Pipeline

```python
# Run batch analysis
system.run_batch_analysis("Ethanol")

# Access results
concentrations = [r['concentration_ppm'] for r in system.analysis_results]
snr_values = [d['snr'] for d in system.data_buffer]

# Export to DataFrame
import pandas as pd
df = pd.DataFrame(list(system.data_buffer))
```

---

## System Requirements

### Software

- Python 3.8+
- numpy
- pandas
- scipy
- streamlit (for dashboard)

### Hardware (Optional)

- ThorLabs CCS200 Spectrometer
- USB 2.0/3.0 connection
- ThorLabs drivers installed

### Operating System

- Windows 10/11 (tested)
- Linux (compatible)
- macOS (compatible)

---

## Version History

### Version 2.0.0 (Current)

- ✓ Professional error handling
- ✓ Comprehensive quality control
- ✓ Real-time analysis integration
- ✓ Automatic calibration management
- ✓ Professional logging system
- ✓ Session tracking and reporting
- ✓ Simulation mode for testing
- ✓ Publication-ready outputs

---

## Support

### Documentation

- System logs: `logs/system_YYYYMMDD_HHMMSS.log`
- Session reports: `reports/session_report_YYYYMMDD_HHMMSS.md`
- This guide: `REALTIME_SYSTEM_GUIDE.md`

### Log Analysis

```bash
# View recent logs
tail -n 50 logs/system_*.log

# Search for errors
grep "ERROR" logs/system_*.log

# Check quality filtering
grep "filtered by quality" logs/system_*.log
```

---

## Conclusion

This **Real-Time Gas Sensing System** provides a professional, publication-ready platform for gas sensing research. It integrates seamlessly with your existing analysis pipeline while adding robust quality control, comprehensive logging, and professional data management.

**Key Benefits:**
- Single unified system for all operations
- Automatic quality assurance
- Publication-ready outputs
- Comprehensive documentation
- Professional error handling
- Reproducible research workflows

**Ready for:**
- Real-time experiments
- Long-term monitoring
- Publication-quality data collection
- Integration with existing research
- Professional research workflows

---

**For questions or issues, check the system logs or refer to this guide.**
