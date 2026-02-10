# Pipeline Guide

## Quick Start

The **unified pipeline interface** provides access to all analysis modes through a single command:

```bash
python unified_pipeline.py --mode scientific --gas Acetone
```

## Available Pipeline Modes

### 1. **Scientific Mode** (Recommended for Publication)
```bash
python unified_pipeline.py --mode scientific --gas Acetone
```
- Uses validated scientific pipeline
- Publication-ready results (matches `output/publication_figures/` tables)
- Robust statistical validation
- Consistent with UNIFIED_RESULTS.md

### 2. **ML-Enhanced Mode**
```bash
python unified_pipeline.py --mode ml-enhanced --gas Acetone
```
- Machine learning feature engineering
- 1D-CNN analysis (if enabled)
- Enhanced detection limit calculation
- Advanced statistical validation

### 3. **World-Class Analysis**
```bash
python unified_pipeline.py --mode world-class --gas Acetone
```
- Comprehensive analysis suite
- Multi-metric validation
- Publication-quality figures
- Detailed reporting

### 4. **Comparative Analysis**
```bash
python unified_pipeline.py --mode comparative
```
- Analyzes all 6 gases
- Cross-gas performance comparison
- Unified reporting
- Benchmark comparisons

### 5. **Debug Mode**
```bash
python unified_pipeline.py --mode debug --gas Acetone --verbose
```
- Detailed logging
- Step-by-step execution
- Error diagnostics
- Performance profiling

### 6. **Validation Mode**
```bash
python unified_pipeline.py --mode validation
```
- System health check
- Dependency verification
- Configuration validation
- Data integrity tests

## Legacy Scripts

The following scripts are still available but deprecated in favor of the unified interface:

- `run_scientific_pipeline.py` → Use `--mode scientific`
- `run_ml_enhanced_pipeline.py` → Use `--mode ml-enhanced`
- `run_world_class_analysis.py` → Use `--mode world-class`
- `comparative_analysis.py` → Use `--mode comparative`
- `run_debug.py` → Use `--mode debug`

## Examples

### Basic Analysis
```bash
# Analyze acetone with scientific pipeline
python unified_pipeline.py --mode scientific --gas Acetone

# Analyze all gases
python unified_pipeline.py --mode comparative

# Run with verbose output
python unified_pipeline.py --mode scientific --gas Acetone --verbose
```

### Advanced Usage
```bash
# List all available modes
python unified_pipeline.py --list-modes

# Run system validation
python unified_pipeline.py --mode validation

# Debug specific gas
python unified_pipeline.py --mode debug --gas Ethanol --verbose
```

## Output Locations

All results are saved to the `output/` directory:

- `output/{gas}_scientific/` - Individual gas analysis
- `output/publication_figures/` - Publication-ready figures (Figure 4 metrics)
- `output/comparative_analysis.png` - Multi-gas comparison
- `UNIFIED_RESULTS.md` - Canonical results summary

## Troubleshooting

1. **Dependencies not found**: Run `python unified_pipeline.py --mode validation`
2. **Configuration errors**: Check `config/config.yaml`
3. **Data issues**: Verify `Kevin_Data/` directory structure
4. **Import errors**: Ensure all requirements are installed

## Getting Help

```bash
# See all options
python unified_pipeline.py --help

# List available modes
python unified_pipeline.py --list-modes
```

For detailed troubleshooting, see the test suite:
```bash
python test_suite.py
```
