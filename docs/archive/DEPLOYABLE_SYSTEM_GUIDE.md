# Deployable Real-Time Gas Sensing System

## Overview

A professional, deployable gas sensor system that acquires spectrometer data in real-time, processes it with auto-calibration, monitors performance metrics, and generates ML-ready datasets.

## Key Features

### 🚀 Auto-Calibration System
- **Memory-based**: Maintains rolling window of recent measurements
- **Intelligent Recalibration**: Triggers when confidence < 0.7 or drift > 0.1 nm/min
- **Calibration History**: Tracks calibration versions with R² values
- **Fresh Start**: No pre-existing calibration needed

### 📊 Real-Time Performance Monitoring
- **Wavelength Shift Tracking**: Δλ with confidence intervals
- **Absorbance Pattern Analysis**: Shape change detection for gas identification
- **Response Time Measurement**: T90 and recovery time tracking
- **Drift Detection**: Real-time drift rate calculation (nm/min)
- **LOD Estimation**: 3-sigma method for limit of detection

### 🤖 ML-Ready Data Pipeline
- **Feature Extraction**: Time-based, measurement, and statistical features
- **Auto-Labeling**: Based on calibration updates
- **Continuous Dataset**: Exports ML-ready CSV every 100 samples
- **Structured Format**: Ready for ML algorithm training

## Usage

### Basic Deployable Mode
```bash
python run.py --mode deployable --duration 60
```

### Advanced Configuration
```bash
python run.py --mode deployable \
  --duration 3600 \
  --target-wavelength 532.0 \
  --calibration-slope 0.116
```

## Real-Time Output

```
============================================================
DEPLOYABLE MODE
Duration: 60.0s | Target: 532.0nm
Features: Auto-calibration, Performance monitoring, ML-ready data
============================================================

[   5.1s] Sample    10 | Peak: 531.23nm | Conc:  -6.63ppm | SNR: 100.1 | FAIL
         Calibration: v10 | Confidence: 1.000 | Drift: -123.0242 nm/min
         Performance: LOD: 0.125nm | Response: 0.0s | Quality: 0.850

[  10.2s] Sample    20 | Peak: 531.53nm | Conc:  -4.04ppm | SNR: 105.4 | FAIL
         Calibration: v20 | Confidence: 1.000 | Drift: -58.7812 nm/min
         Performance: LOD: 0.125nm | Response: 0.0s | Quality: 0.850
```

## Generated Files

### Calibration Memory
- `output/calibration_memory_YYYYMMDD_HHMMSS.json`
- Contains calibration history, statistics, and version tracking

### Performance Summary
- `output/performance_summary_YYYYMMDD_HHMMSS.json`
- 30-minute window performance metrics
- Includes LOD, response time, drift rate averages

### ML Dataset
- `output/ml_dataset/ml_features_YYYYMMDD_HHMMSS_XXXXXX.csv`
- ML-ready feature vectors with timestamps
- Includes all performance and calibration features

## System Architecture

```
run.py (Single Entry)
├── GasSensingSystem
│   ├── RealTimePipeline (Enhanced)
│   │   ├── CalibrationMemory (Auto-calibration)
│   │   ├── PerformanceMonitor (Real-time metrics)
│   │   └── Processing Stages (Preprocess, Feature Extract, etc.)
│   └── SpectrometerInterface (Hardware + Simulation)
└── Data Export (CSV, JSON, ML-ready)
```

## Performance Metrics

### Calibration Quality
- **Confidence Score**: 0.0-1.0 (higher = better)
- **Drift Rate**: nm/minute (lower = better stability)
- **Calibration Version**: Tracks improvements over time

### Sensor Performance
- **LOD**: Limit of detection in nm (3-sigma method)
- **Response Time**: Time to reach 90% of final value
- **Signal Quality**: Composite score (SNR + pattern + confidence)
- **Validity Rate**: Percentage of samples passing quality gates

## Integration Points

### With Existing Systems
- **Uses**: `gas_analysis/core/realtime_pipeline.py` (enhanced)
- **Uses**: `PDMS_Ring_Rectum Sensor/src/acquisition/` (hardware)
- **Extends**: Batch analysis capabilities for real-time use
- **Replaces**: Multiple duplicate systems with unified approach

### For ML Training
- **Feature Types**: Temporal, spectral, statistical, performance
- **Label Source**: Auto-calibration confidence and version
- **Export Format**: CSV with comprehensive metadata
- **Update Frequency**: Every 100 samples (configurable)

## Deployment Benefits

1. **Professional**: Research-grade data quality and validation
2. **Self-Maintaining**: Auto-calibration reduces manual intervention
3. **Performance Aware**: Real-time monitoring of sensor health
4. **ML Ready**: Continuous dataset generation for algorithm development
5. **Deployable**: Single command starts complete system

This system provides a complete solution for real-time gas sensing with professional-grade monitoring and ML integration capabilities.
