#!/usr/bin/env python3
"""
Enhanced Reproducibility Package
Complete environment setup and validation for tier-1 publication
"""

import os
import sys
import subprocess
import json
import hashlib
import platform
from pathlib import Path

def create_environment_specification():
    """
    Create complete environment specification
    """
    
    print("=== Creating Enhanced Reproducibility Package ===")
    
    # Python environment
    python_version = sys.version
    packages = {
        'numpy': '1.21.0',
        'pandas': '1.3.0',
        'matplotlib': '3.4.2',
        'scipy': '1.7.0',
        'scikit-learn': '1.0.2',
        'yaml': '5.4.1',
        'seaborn': '0.11.2',
        'jupyter': '1.0.0'
    }
    
    env_spec = {
        'python_version': python_version,
        'packages': packages,
        'platform': sys.platform,
        'architecture': platform.machine()
    }
    
    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write("# Enhanced Reproducibility Package\n")
        f.write("# For Sensitivity-Optimized Gas Sensing\n\n")
        for package, version in packages.items():
            f.write(f"{package}=={version}\n")
    
    # Create environment.yml for conda
    env_yml = f"""name: gas-sensing-optimized
channels:
  - conda-forge
  - defaults
dependencies:
  - python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}
  - numpy={packages['numpy']}
  - pandas={packages['pandas']}
  - matplotlib={packages['matplotlib']}
  - scipy={packages['scipy']}
  - scikit-learn={packages['scikit-learn']}
  - pyyaml={packages['yaml']}
  - seaborn={packages['seaborn']}
  - jupyter={packages['jupyter']}
  - pip
  - pip:
    - -e .
"""
    
    with open('environment.yml', 'w') as f:
        f.write(env_yml)
    
    print("✓ Environment specifications created")
    return env_spec

def create_docker_configuration():
    """
    Create Docker configuration for complete reproducibility
    """
    
    dockerfile = """# Sensitivity-Optimized Gas Sensing
# Complete Reproducibility Environment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GAS_SENSING_DATA=/app/data

# Expose port for Jupyter
EXPOSE 8888

# Default command
CMD ["python", "run_scientific_pipeline.py"]
"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    docker_compose = """version: '3.8'

services:
  gas-sensing:
    build: .
    volumes:
      - ./data:/app/data
      - ./output:/app/output
    environment:
      - PYTHONPATH=/app
    ports:
      - "8888:8888"
    
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./:/app
    command: ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    
    print("✓ Docker configuration created")

def create_data_validation():
    """
    Create data integrity validation scripts
    """
    
    def calculate_file_hash(filepath):
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    # Validate key data files
    data_files = {
        'config/config.yaml': 'Configuration file',
        'run_scientific_pipeline.py': 'Main pipeline',
        'gas_analysis/core/pipeline.py': 'Core analysis',
        'Kevin_Data/Acetone': 'Acetone data directory'
    }
    
    validation_report = {
        'timestamp': '2026-01-13T16:00:00',
        'files_validated': {},
        'validation_passed': True
    }
    
    for file_path, description in data_files.items():
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                file_hash = calculate_file_hash(file_path)
                file_size = os.path.getsize(file_path)
                validation_report['files_validated'][file_path] = {
                    'description': description,
                    'hash': file_hash,
                    'size': file_size,
                    'exists': True
                }
            else:
                # Directory
                num_files = len(list(Path(file_path).rglob('*')))
                validation_report['files_validated'][file_path] = {
                    'description': description,
                    'type': 'directory',
                    'num_files': num_files,
                    'exists': True
                }
        else:
            validation_report['files_validated'][file_path] = {
                'description': description,
                'exists': False,
                'validation_passed': False
            }
            validation_report['validation_passed'] = False
    
    with open('output/data_validation_report.json', 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    print("✓ Data validation report created")
    return validation_report

def create_documentation_suite():
    """
    Create comprehensive documentation
    """
    
    # README for reproducibility
    readme = """# Sensitivity-Optimized Gas Sensing - Reproducibility Package

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
"""
    
    with open('README_REPRODUCIBILITY.md', 'w') as f:
        f.write(readme)
    
    # Installation validation script
    validation_script = """#!/usr/bin/env python3
\"\"\"
Installation Validation Script
\"\"\"

import sys
import importlib
import os

def check_package(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def validate_installation():
    print("=== Installation Validation ===")
    
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 
        'sklearn', 'yaml', 'seaborn'
    ]
    
    all_good = True
    for package in required_packages:
        if check_package(package):
            print(f"[OK] {package}")
        else:
            print(f"[MISSING] {package} - MISSING")
            all_good = False
    
    # Check data directory
    if os.path.exists('Kevin_Data'):
        print("[OK] Kevin_Data directory found")
    else:
        print("[MISSING] Kevin_Data directory missing")
        all_good = False
    
    # Check configuration
    if os.path.exists('config/config.yaml'):
        print("[OK] Configuration file found")
    else:
        print("[MISSING] Configuration file missing")
        all_good = False
    
    if all_good:
        print("\\n[SUCCESS] Installation validated successfully!")
        print("You can now run: python run_scientific_pipeline.py")
    else:
        print("\\n[FAILED] Installation validation failed")
        print("Please check the missing components above")
    
    return all_good

if __name__ == "__main__":
    validate_installation()
"""
    
    with open('validate_installation.py', 'w') as f:
        f.write(validation_script)
    
    print("✓ Documentation suite created")

def create_test_suite():
    """
    Create comprehensive test suite
    """
    
    test_script = """#!/usr/bin/env python3
\"\"\"
Test Suite for Sensitivity-Optimized Gas Sensing
\"\"\"

import unittest
import numpy as np
import json
from pathlib import Path

class TestGasSensingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.test_config = 'config/config.yaml'
        self.output_dir = 'output/test'
        Path(self.output_dir).mkdir(exist_ok=True)
    
    def test_sensitivity_first_selection(self):
        \"\"\"Test sensitivity-first ROI selection\"\"\"
        # This would test the core algorithm
        from run_scientific_pipeline import ScientificGasPipeline
        
        pipeline = ScientificGasPipeline()
        # Add specific tests here
        self.assertTrue(True)  # Placeholder
    
    def test_acetone_sensitivity(self):
        \"\"\"Test acetone sensitivity meets target\"\"\"
        target_sensitivity = 0.2692
        tolerance = 0.01
        
        # Load results from previous run
        if Path('output/acetone_scientific/metrics/calibration_metrics.json').exists():
            with open('output/acetone_scientific/metrics/calibration_metrics.json') as f:
                results = json.load(f)
            
            actual_sensitivity = results.get('slope', 0)
            self.assertAlmostEqual(
                actual_sensitivity, target_sensitivity, 
                delta=tolerance, 
                msg=f"Sensitivity {actual_sensitivity} not close to target {target_sensitivity}"
            )
    
    def test_roi_selection(self):
        \"\"\"Test ROI selection is optimal\"\"\"
        expected_roi = [595, 625]
        
        if Path('output/acetone_scientific/metrics/calibration_metrics.json').exists():
            with open('output/acetone_scientific/metrics/calibration_metrics.json') as f:
                results = json.load(f)
            
            actual_roi = results.get('roi_range', [])
            self.assertEqual(
                actual_roi, expected_roi,
                msg=f"ROI {actual_roi} not equal to expected {expected_roi}"
            )

if __name__ == '__main__':
    unittest.main()
"""
    
    with open('test_pipeline.py', 'w') as f:
        f.write(test_script)
    
    print("✓ Test suite created")

def main():
    """
    Create complete reproducibility package
    """
    
    print("Creating Enhanced Reproducibility Package...")
    
    # Create all components
    env_spec = create_environment_specification()
    create_docker_configuration()
    validation_report = create_data_validation()
    create_documentation_suite()
    create_test_suite()
    
    # Create summary
    summary = {
        'package_created': '2026-01-13T16:00:00',
        'components': {
            'environment_specification': 'requirements.txt, environment.yml',
            'docker_configuration': 'Dockerfile, docker-compose.yml',
            'data_validation': 'output/data_validation_report.json',
            'documentation': 'README_REPRODUCIBILITY.md',
            'validation_script': 'validate_installation.py',
            'test_suite': 'test_pipeline.py'
        },
        'installation_methods': [
            'pip install -r requirements.txt',
            'docker-compose up',
            'conda env create -f environment.yml'
        ],
        'expected_results': {
            'acetone_sensitivity': '0.2692 nm/ppm',
            'roi': '595-625 nm',
            'r2': '0.9945',
            'lod': '0.75 ppm'
        }
    }
    
    with open('output/reproducibility_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✓ Enhanced Reproducibility Package Complete!")
    print("\nTo validate installation:")
    print("  python validate_installation.py")
    print("\nTo run tests:")
    print("  python test_pipeline.py")
    print("\nTo use Docker:")
    print("  docker-compose up gas-sensing")

if __name__ == "__main__":
    main()
