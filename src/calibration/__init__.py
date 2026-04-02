"""
src.calibration
===============
Calibration models for optical gas sensing pipelines.

Modules
-------
gpr
    Gaussian Process Regression for concentration estimation with uncertainty.
    ``GPRCalibration.predict_single(delta_lambda)`` → ``(ppm, uncertainty_ppm)``.
sensitivity_matrix
    N×M sensitivity matrix calibration for multi-analyte multi-peak sensors.
    ``SensitivityMatrix.estimate_concentrations(delta_lambda)`` → concentrations dict.
multi_output_gpr
    Multi-output GPR for simultaneous multi-analyte quantification.
    ``IndependentMultiOutputGPR``, ``JointMultiOutputGPR``, ``build_feature_vector``.
transforms
    Concentration-axis transforms (log10, log, sqrt, linear) for curve fitting.
    ``transform_concentrations(concs, mode)`` → ``(transformed, meta)``.
multi_roi
    Multi-ROI fusion: candidate selection and multivariate linear calibration.
roi_scan
    Per-wavelength linear regression scan for ROI discovery.
"""

from src.calibration.active_learning import BayesianExperimentDesigner
from src.calibration.batch_reproducibility import (
    BatchReproducibilityAnalyzer,
    BatchReproducibilityReport,
)
from src.calibration.conformal import ConformalCalibrator
from src.calibration.gpr import GPRCalibration
from src.calibration.multi_output_gpr import (
    IndependentMultiOutputGPR,
    JointMultiOutputGPR,
    build_feature_vector,
)
from src.calibration.multi_roi import fit_multi_roi_fusion, select_multi_roi_candidates
from src.calibration.physics_kernel import (
    LangmuirMeanFunction,
    PhysicsInformedGPR,
    fit_langmuir_params,
)
from src.calibration.pls import PLSCalibration, PLSFitResult
from src.calibration.roi_scan import (
    RoiScanConfig,
    compute_concentration_response,
    stack_trials_for_response,
)
from src.calibration.selectivity import SelectivityAnalyzer, SelectivityReport
from src.calibration.sensitivity_matrix import SensitivityEntry, SensitivityMatrix
from src.calibration.transforms import transform_concentrations

__all__ = [
    "BayesianExperimentDesigner",
    "ConformalCalibrator",
    "GPRCalibration",
    "LangmuirMeanFunction",
    "PhysicsInformedGPR",
    "fit_langmuir_params",
    "PLSCalibration",
    "PLSFitResult",
    "SelectivityAnalyzer",
    "SelectivityReport",
    "BatchReproducibilityAnalyzer",
    "BatchReproducibilityReport",
    "transform_concentrations",
    "select_multi_roi_candidates",
    "fit_multi_roi_fusion",
    "RoiScanConfig",
    "compute_concentration_response",
    "stack_trials_for_response",
    "SensitivityMatrix",
    "SensitivityEntry",
    "IndependentMultiOutputGPR",
    "JointMultiOutputGPR",
    "build_feature_vector",
]
