#!/usr/bin/env python3
"""
Installation Validation Script
"""

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
        print("\n[SUCCESS] Installation validated successfully!")
        print("You can now run: python run_scientific_pipeline.py")
    else:
        print("\n[FAILED] Installation validation failed")
        print("Please check the missing components above")
    
    return all_good

if __name__ == "__main__":
    validate_installation()
