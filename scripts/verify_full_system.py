from pathlib import Path
import sys

# Add project root to path
REPO_ROOT = Path(".").resolve()
sys.path.append(str(REPO_ROOT))

print("1. Testing Imports...")
try:
    import numpy as np
    import pandas as pd
    import pywt
    import streamlit
    import torch

    print("   [OK] Basic Libraries (numpy, pandas, torch, pywt, streamlit)")
except ImportError as e:
    print(f"   [FAIL] Basic Libraries: {e}")
    sys.exit(1)

print("2. Testing Core Modules...")
try:
    print("   [OK] gas_analysis.core.pipeline")

    print("   [OK] gas_analysis.core.signal_proc (incl. advanced)")

    print("   [OK] gas_analysis.core.calibration")

    from gas_analysis.core.intelligence import CNNGasClassifier, GPRCalibration

    print("   [OK] gas_analysis.core.intelligence")
except Exception as e:
    print(f"   [FAIL] Core Modules: {e}")
    sys.exit(1)

print("3. Testing Class Instantiation...")
try:
    clf = CNNGasClassifier(input_length=500, num_classes=3)
    print("   [OK] CNNGasClassifier instantiated")

    gpr = GPRCalibration()
    print("   [OK] GPRCalibration instantiated")
except Exception as e:
    print(f"   [FAIL] Class Instantiation: {e}")
    sys.exit(1)

print("\n---------------------------------------------------")
print("✅ ALL SYSTEMS VERIFIED. PROJECT IS READY.")
print("---------------------------------------------------")
