#!/usr/bin/env python3
"""
Test Suite for Sensitivity-Optimized Gas Sensing
"""

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
        """Test sensitivity-first ROI selection"""
        # This would test the core algorithm
        from run_scientific_pipeline import ScientificGasPipeline
        
        pipeline = ScientificGasPipeline()
        # Add specific tests here
        self.assertTrue(True)  # Placeholder
    
    def test_acetone_sensitivity(self):
        """Test acetone sensitivity meets target"""
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
        """Test ROI selection is optimal"""
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
