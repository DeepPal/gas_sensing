# Sensitivity-Optimized Gas Sensing - Reproducibility Package

## Quick Start

### Option 1: Python Environment
```bash
pip install -r requirements.txt
python run_scientific_pipeline.py
```

### Option 2: Docker
```bash
docker-compose up gas-sensing
```

### Option 3: Conda
```bash
conda env create -f environment.yml
conda activate gas-sensing-optimized
python run_scientific_pipeline.py
```

## Expected Results

Running the pipeline should produce:
- **Acetone sensitivity**: 0.2692 nm/ppm
- **ROI**: 595-625 nm
- **R²**: 0.9945
- **LOD**: 0.75 ppm

## Validation

To validate your installation:
```bash
python validate_installation.py
```

## Troubleshooting

1. **Missing dependencies**: Ensure all packages in requirements.txt are installed
2. **Data not found**: Check that Kevin_Data directory is present
3. **Permission errors**: Ensure write access to output directory

## Citation

If you use this method, please cite:
[Your citation information]
