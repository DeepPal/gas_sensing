# Duplicate and Unused Files Analysis

## Duplicates Found

### 1. config_loader.py (2 copies)
- **KEEP**: `config/config_loader.py` (4.7 KB) - Main config
- **REMOVE**: `gas_analysis/agent/config_loader.py` (14.2 KB) - Agent-specific, unused

### 2. realtime_pipeline.py (2 copies)
- **KEEP**: `gas_analysis/core/realtime_pipeline.py` (33.4 KB) - Core pipeline
- **REMOVE**: `unified_research_platform/core/realtime_pipeline.py` (17.4 KB) - Outdated duplicate

### 3. __init__.py (17 copies)
- **KEEP**: Essential ones for package structure
- **REMOVE**: Empty ones in unused subdirectories

## Large Files - Purpose Analysis

| Size (KB) | File | Purpose | Status |
|------------|------|---------|--------|
| 369.2 | `gas_analysis/core/pipeline.py` | **KEEP** - Batch processing |
| 116 | `dashboard/app.py` | **KEEP** - Web interface |
| 76.6 | `gas_analysis/tools/build_report.py` | **KEEP** - Reporting |
| 40.3 | `realtime_gas_sensing_system.py` | **REMOVE** - Superseded by run.py |
| 37.8 | `PDMS_Ring_Rectum Sensor/src/pipelines/dual_fixed_wavelength_app.py` | **REMOVE** - Specific app |
| 37.3 | `gas_analysis/core/calibration/methods.py` | **KEEP** - Calibration methods |
| 36.7 | `PDMS_Ring_Rectum Sensor/src/visualization/advanced_dashboard.py` | **REMOVE** - Duplicate of dashboard |
| 33.5 | `gas_analysis/agent/anomaly_detector.py` | **REMOVE** - Agent-specific |
| 33.4 | `gas_analysis/core/realtime_pipeline.py` | **KEEP** - Core pipeline |
| 31.9 | `PDMS_Ring_Rectum Sensor/src/analysis/ml_pressure_predictor.py` | **REMOVE** - Specific analysis |
| 30.4 | `PDMS_Ring_Rectum Sensor/src/pipelines/realtime_monitoring.py` | **REMOVE** - Duplicate of realtime system |
| 29.7 | `PDMS_Ring_Rectum Sensor/src/analysis/advanced_signal_processor.py` | **REMOVE** - Duplicate of preprocessing |
| 29 | `gas_analysis/agent/benchmarks.py` | **REMOVE** - Agent-specific |
| 28.9 | `gas_analysis/agent/n8n_integration.py` | **REMOVE** - Agent integration |
| 28.3 | `PDMS_Ring_Rectum Sensor/src/analysis/statistical_analysis.py` | **REMOVE** - Duplicate of analysis |
| 28.3 | `PDMS_Ring_Rectum Sensor/src/analysis/realtime_analyzer.py` | **REMOVE** - Duplicate of pipeline |

## Files to Remove (Safe)

### High Priority - Clear Duplicates
```
realtime_gas_sensing_system.py                    # 40KB - Superseded by run.py
unified_research_platform/core/realtime_pipeline.py # 17KB - Duplicate
PDMS_Ring_Rectum Sensor/src/analysis/realtime_analyzer.py # 28KB - Duplicate
PDMS_Ring_Rectum Sensor/src/analysis/advanced_signal_processor.py # 30KB - Duplicate
PDMS_Ring_Rectum Sensor/src/pipelines/realtime_monitoring.py # 30KB - Duplicate
```

### Medium Priority - Unused Apps
```
PDMS_Ring_Rectum Sensor/src/pipelines/dual_fixed_wavelength_app.py # 38KB
PDMS_Ring_Rectum Sensor/src/pipelines/research_grade_app.py # 28KB
PDMS_Ring_Rectum Sensor/src/pipelines/unified_research_system.py # 21KB
PDMS_Ring_Rectum Sensor/src/visualization/advanced_dashboard.py # 37KB
```

### Low Priority - Agent/Tools
```
gas_analysis/agent/ (entire folder - 200KB+)
PDMS_Ring_Rectum Sensor/src/analysis/ml_pressure_predictor.py # 32KB
PDMS_Ring_Rectum Sensor/src/analysis/statistical_analysis.py # 28KB
```

## Space Savings

**Total removable**: ~500KB+ of duplicate/unused code
**Core system after cleanup**: ~600KB (essential modules only)

## Recommended Action

```bash
# Remove clear duplicates
rm realtime_gas_sensing_system.py
rm -rf unified_research_platform/
rm PDMS_Ring_Rectum\ Sensor/src/analysis/realtime_analyzer.py
rm PDMS_Ring_Rectum\ Sensor/src/analysis/advanced_signal_processor.py

# Keep only essential structure:
run.py                    # Single entry point
gas_analysis/core/         # Core processing
PDMS_Ring_Rectum Sensor/src/acquisition/  # Hardware interface
```

This would result in a clean, non-duplicated codebase where:
- `run.py` is the single entry point
- `gas_analysis/core/` contains all processing logic
- `PDMS_Ring_Rectum Sensor/src/acquisition/` contains hardware interface
- No duplicate functionality
