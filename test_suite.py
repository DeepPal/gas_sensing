#!/usr/bin/env python3
"""
Automated Testing Suite for ML-Enhanced Gas Sensing Pipeline
Tests core functionality, data integrity, and configuration validation
"""

import unittest
import numpy as np
import pandas as pd
import yaml
import os
import sys
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestDependencies(unittest.TestCase):
    """Test that all required dependencies are available and version-compatible"""
    
    def test_numpy_import(self):
        try:
            import numpy as np
            self.assertGreaterEqual(np.__version__.split('.')[0], '1')
        except ImportError:
            self.fail("numpy not available")
    
    def test_pandas_import(self):
        try:
            import pandas as pd
            self.assertGreaterEqual(pd.__version__.split('.')[0], '1')
        except ImportError:
            self.fail("pandas not available")
    
    def test_scipy_import(self):
        try:
            import scipy
            self.assertGreaterEqual(scipy.__version__.split('.')[0], '1')
        except ImportError:
            self.fail("scipy not available")
    
    def test_sklearn_import(self):
        try:
            import sklearn
            self.assertGreaterEqual(sklearn.__version__.split('.')[0], '1')
        except ImportError:
            self.fail("scikit-learn not available")

class TestConfiguration(unittest.TestCase):
    """Test configuration file integrity and consistency"""
    
    def setUp(self):
        self.config_path = project_root / 'config' / 'config.yaml'
        
    def test_config_exists(self):
        self.assertTrue(self.config_path.exists(), "config.yaml not found")
    
    def test_config_valid_yaml(self):
        try:
            with open(self.config_path, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.fail(f"Invalid YAML in config.yaml: {e}")
    
    def test_config_required_sections(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'analysis', 'calibration', 'roi', 'preprocessing']
        for section in required_sections:
            self.assertIn(section, config, f"Missing required section: {section}")
    
    def test_no_duplicate_sections(self):
        """Check for duplicate configuration sections"""
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Count occurrences of major section headers (must be at start of line)
        sections = ['preprocessing:', 'data:', 'analysis:', 'calibration:']
        for section in sections:
            # Only count if section appears at start of line (YAML section header)
            lines = content.split('\n')
            count = sum(1 for line in lines if line.strip().startswith(section))
            self.assertEqual(count, 1, f"Duplicate section found: {section}")

class TestDataIntegrity(unittest.TestCase):
    """Test data directory structure and sample files"""
    
    def setUp(self):
        self.data_dir = project_root / 'Kevin_Data'
        
    def test_data_directory_exists(self):
        self.assertTrue(self.data_dir.exists(), "Kevin_Data directory not found")
    
    def test_gas_directories_exist(self):
        required_gases = ['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 'Toluene', 'Xylene']
        for gas in required_gases:
            gas_dir = self.data_dir / gas
            self.assertTrue(gas_dir.exists(), f"Gas directory not found: {gas}")
    
    def test_reference_files_exist(self):
        """Check that reference files are present"""
        with open(project_root / 'config' / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        ref_files = config['data']['ref_files']
        for gas, ref_file in ref_files.items():
            ref_path = self.data_dir / ref_file
            self.assertTrue(ref_path.exists(), f"Reference file not found: {gas} -> {ref_file}")

class TestOutputStructure(unittest.TestCase):
    """Test output directory structure and file generation"""
    
    def setUp(self):
        self.output_dir = project_root / 'output'
        
    def test_output_directory_exists(self):
        self.assertTrue(self.output_dir.exists(), "output directory not found")
    
    def test_publication_figures_directory(self):
        pub_dir = self.output_dir / 'publication_figures'
        self.assertTrue(pub_dir.exists(), "publication_figures directory not found")
    
    def test_gas_scientific_directories(self):
        """Check that scientific analysis directories exist"""
        required_gases = ['acetone', 'ethanol', 'methanol', 'isopropanol', 'toluene', 'xylene']
        for gas in required_gases:
            gas_dir = self.output_dir / f"{gas}_scientific"
            self.assertTrue(gas_dir.exists(), f"Scientific analysis directory not found: {gas}")

class TestPipelineModules(unittest.TestCase):
    """Test that core pipeline modules can be imported"""
    
    def test_gas_analysis_import(self):
        try:
            import gas_analysis
        except ImportError as e:
            self.fail(f"Cannot import gas_analysis module: {e}")
    
    def test_config_module_import(self):
        try:
            from config import config_loader
        except ImportError as e:
            self.fail(f"Cannot import config loader: {e}")

class TestResultsConsistency(unittest.TestCase):
    """Test that results documents are consistent"""
    
    def test_unified_results_exists(self):
        unified_path = project_root / 'UNIFIED_RESULTS.md'
        self.assertTrue(unified_path.exists(), "UNIFIED_RESULTS.md not found")
    
    def test_no_conflicting_results(self):
        """Check that ANALYSIS_STATUS.md references UNIFIED_RESULTS.md"""
        status_path = project_root / 'ANALYSIS_STATUS.md'
        with open(status_path, 'r') as f:
            content = f.read()
        
        self.assertIn('UNIFIED_RESULTS.md', content, 
                     "ANALYSIS_STATUS.md should reference UNIFIED_RESULTS.md")

class TestReproducibility(unittest.TestCase):
    """Test reproducibility measures"""
    
    def test_requirements_exists(self):
        req_path = project_root / 'requirements.txt'
        self.assertTrue(req_path.exists(), "requirements.txt not found")
    
    def test_environment_exists(self):
        env_path = project_root / 'environment.yml'
        self.assertTrue(env_path.exists(), "environment.yml not found")
    
    def test_gitignore_exists(self):
        gitignore_path = project_root / '.gitignore'
        self.assertTrue(gitignore_path.exists(), ".gitignore not found")

def run_comprehensive_test():
    """Run all tests and return summary"""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Create test suite
    test_classes = [
        TestDependencies,
        TestConfiguration, 
        TestDataIntegrity,
        TestOutputStructure,
        TestPipelineModules,
        TestResultsConsistency,
        TestReproducibility
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return summary
    return {
        'total_tests': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    }

if __name__ == '__main__':
    print("=" * 60)
    print("ML-Enhanced Gas Sensing Pipeline - Automated Testing Suite")
    print("=" * 60)
    
    summary = run_comprehensive_test()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Failures: {summary['failures']}")
    print(f"Errors: {summary['errors']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    
    if summary['failures'] == 0 and summary['errors'] == 0:
        print("\n✅ ALL TESTS PASSED - Pipeline is ready for production!")
    else:
        print("\n⚠️  Some tests failed - Please review and fix issues")
    
    print("=" * 60)
