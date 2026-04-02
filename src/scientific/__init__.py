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
allan_deviation
    Allan deviation noise analysis: optimal averaging time, noise floor,
    white/flicker/random-walk noise classification.  Required for
    publication in Sensors & Actuators B and IEEE Sensors Journal.
ruggedness
    Youden 8-run ruggedness test (Youden & Steiner 1975, AOAC) and
    spike recovery protocol (ICH Q2(R1)).  Both are mandatory for
    analytical method validation in chemistry / sensor journals.
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
from src.scientific.allan_deviation import (
    AllanDeviationResult,
    NoiseType,
    adev_noise_fractions,
    allan_deviation,
)
from src.scientific.ruggedness import (
    YoudensDesign,
    RuggednessResult,
    SpikeRecoveryResult,
    SpikeRecoveryPoint,
    youden_ruggedness,
    spike_recovery,
    recovery_acceptance,
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
    # allan deviation
    "AllanDeviationResult",
    "NoiseType",
    "adev_noise_fractions",
    "allan_deviation",
    # ruggedness / spike recovery
    "YoudensDesign",
    "RuggednessResult",
    "SpikeRecoveryResult",
    "SpikeRecoveryPoint",
    "youden_ruggedness",
    "spike_recovery",
    "recovery_acceptance",
]
