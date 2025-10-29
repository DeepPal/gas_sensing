"""
Minimal Gas Analysis Package
-------------------
A streamlined package for gas spectral analysis with essential processing capabilities.
"""

__version__ = "1.0.0"

# Import key components for easier access
from .core.pipeline import run_full_pipeline
from .core.preprocessing import baseline_correction, smooth_spectrum

__all__ = [
    'run_full_pipeline',
    'baseline_correction',
    'smooth_spectrum'
]
