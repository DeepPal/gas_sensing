"""
src.scientific
==============
Scientific metrics for sensor characterisation.

Modules
-------
lod
    LOD / LOQ (IUPAC 3σ/slope), bootstrap confidence intervals,
    robust Theil-Sen sensitivity, Mandel's linearity F-test.
cross_session
    Cross-session statistical comparison: paired t-test, Bland-Altman,
    F-test for variance equality, Mann-Whitney U.  Required for ISO 5725-2
    reproducibility claims.
residual_diagnostics
    OLS residual validation suite: Durbin-Watson (autocorrelation),
    Shapiro-Wilk (normality), Breusch-Pagan (heteroscedasticity),
    and lack-of-fit F-test.  Mandatory for Analytical Chemistry /
    Sensors & Actuators B submissions.
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

from src.scientific.cross_session import (
    CrossSessionComparison,
    SessionData,
    compare_lod_series,
    compare_sessions,
)
from src.scientific.publication_tables import (
    BatchReproducibilityRow,
    DiagnosticsRow,
    SensorPerformanceRow,
    build_supplementary_s1,
    build_supplementary_s2,
    build_table1,
    format_supplementary_s1_text,
    format_supplementary_s2_text,
    format_table1_csv,
    format_table1_latex,
    format_table1_text,
)
from src.scientific.residual_diagnostics import (
    ResidualDiagnostics,
    format_diagnostics_report,
    residual_diagnostics,
)
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
    # cross-session comparison
    "CrossSessionComparison",
    "SessionData",
    "compare_lod_series",
    "compare_sessions",
    # publication tables
    "BatchReproducibilityRow",
    "DiagnosticsRow",
    "SensorPerformanceRow",
    "build_supplementary_s1",
    "build_supplementary_s2",
    "build_table1",
    "format_supplementary_s1_text",
    "format_supplementary_s2_text",
    "format_table1_csv",
    "format_table1_latex",
    "format_table1_text",
    # residual diagnostics
    "ResidualDiagnostics",
    "format_diagnostics_report",
    "residual_diagnostics",
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
