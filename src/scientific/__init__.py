"""
src.scientific
==============
Scientific metrics for sensor characterisation.

Modules
-------
lod
    LOD / LOQ (IUPAC 3σ/slope), bootstrap confidence intervals,
    robust Theil-Sen sensitivity, Mandel's linearity F-test.
selectivity
    Cross-sensitivity matrix and selectivity coefficients (IUPAC).
"""

from src.scientific.lod import (
    calculate_lod_3sigma,
    calculate_loq_10sigma,
    calculate_sensitivity,
    lod_bootstrap_ci,
    mandel_linearity_test,
    robust_sensitivity,
    sensor_performance_summary,
)
from src.scientific.selectivity import (
    SelectivityResult,
    compute_cross_sensitivity,
    selectivity_from_calibration_data,
    selectivity_matrix,
)

__all__ = [
    # lod
    "calculate_lod_3sigma",
    "calculate_loq_10sigma",
    "calculate_sensitivity",
    "lod_bootstrap_ci",
    "mandel_linearity_test",
    "robust_sensitivity",
    "sensor_performance_summary",
    # selectivity
    "SelectivityResult",
    "compute_cross_sensitivity",
    "selectivity_from_calibration_data",
    "selectivity_matrix",
]
