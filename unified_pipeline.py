#!/usr/bin/env python3
"""
Unified Pipeline Interface for ML-Enhanced Gas Sensing
Consolidates all pipeline modes into a single, user-friendly interface
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def run_scientific_mode(gas, **kwargs):
    """Run the validated scientific pipeline"""
    logging.info(f"Running scientific pipeline for {gas}")
    try:
        from run_scientific_pipeline import main as scientific_main
        return scientific_main(gas)
    except Exception as e:
        logging.error(f"Scientific pipeline failed: {e}")
        return False

def run_ml_enhanced_mode(gas, **kwargs):
    """Run the ML-enhanced pipeline"""
    logging.info(f"Running ML-enhanced pipeline for {gas}")
    try:
        from run_ml_enhanced_pipeline import main as ml_main
        return ml_main(gas)
    except Exception as e:
        logging.error(f"ML-enhanced pipeline failed: {e}")
        return False

def run_world_class_mode(gas, **kwargs):
    """Run world-class analysis"""
    logging.info(f"Running world-class analysis for {gas}")
    try:
        from run_world_class_analysis import main as world_main
        return world_main(gas)
    except Exception as e:
        logging.error(f"World-class analysis failed: {e}")
        return False

def run_comparative_mode(**kwargs):
    """Run comparative analysis across all gases"""
    logging.info("Running comparative analysis")
    try:
        from comparative_analysis import main as comp_main
        return comp_main()
    except Exception as e:
        logging.error(f"Comparative analysis failed: {e}")
        return False

def run_debug_mode(gas, **kwargs):
    """Run debug pipeline"""
    logging.info(f"Running debug pipeline for {gas}")
    try:
        from run_debug import main as debug_main
        return debug_main(gas)
    except Exception as e:
        logging.error(f"Debug pipeline failed: {e}")
        return False

def run_validation():
    """Run comprehensive validation"""
    logging.info("Running comprehensive validation")
    try:
        from validate_installation import main as validate_main
        return validate_main()
    except Exception as e:
        logging.error(f"Validation failed: {e}")
        return False

def main():
    """Main pipeline interface"""
    parser = argparse.ArgumentParser(
        description="Unified Gas Sensing Pipeline Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Modes:
  scientific    - Validated scientific pipeline (recommended for publication)
  ml-enhanced   - ML-enhanced analysis with feature engineering
  world-class   - Comprehensive world-class analysis
  comparative   - Multi-gas comparative analysis
  debug         - Debug mode with detailed logging
  validation    - Run comprehensive system validation

Examples:
  python unified_pipeline.py --mode scientific --gas Acetone
  python unified_pipeline.py --mode comparative
  python unified_pipeline.py --mode validation
        """
    )
    
    parser.add_argument('--mode', '-m', 
                       choices=['scientific', 'ml-enhanced', 'world-class', 
                               'comparative', 'debug', 'validation'],
                       default='scientific',
                       help='Pipeline mode to run (default: scientific)')
    
    parser.add_argument('--gas', '-g',
                       choices=['Acetone', 'Ethanol', 'Methanol', 'Isopropanol', 
                               'Toluene', 'Xylene', 'MixVOC'],
                       help='Gas to analyze (required for non-comparative modes)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--list-modes', action='store_true',
                       help='List all available pipeline modes')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # List modes if requested
    if args.list_modes:
        print("Available Pipeline Modes:")
        print("  scientific    - Validated scientific pipeline (publication-ready)")
        print("  ml-enhanced   - ML-enhanced analysis with feature engineering")
        print("  world-class   - Comprehensive world-class analysis")
        print("  comparative   - Multi-gas comparative analysis")
        print("  debug         - Debug mode with detailed logging")
        print("  validation    - Run comprehensive system validation")
        return
    
    # Validate arguments
    if args.mode != 'comparative' and args.mode != 'validation' and not args.gas:
        parser.error(f"--gas is required for mode '{args.mode}'")
    
    # Run selected pipeline
    logging.info(f"Starting unified pipeline in {args.mode} mode")
    
    success = False
    
    if args.mode == 'scientific':
        success = run_scientific_mode(args.gas)
    elif args.mode == 'ml-enhanced':
        success = run_ml_enhanced_mode(args.gas)
    elif args.mode == 'world-class':
        success = run_world_class_mode(args.gas)
    elif args.mode == 'comparative':
        success = run_comparative_mode()
    elif args.mode == 'debug':
        success = run_debug_mode(args.gas)
    elif args.mode == 'validation':
        success = run_validation()
    
    if success:
        logging.info("Pipeline completed successfully")
        return 0
    else:
        logging.error("Pipeline failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
