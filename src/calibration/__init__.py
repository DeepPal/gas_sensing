"""
src.calibration
===============
Calibration models for the Au-MIP LSPR gas sensing pipeline.

Modules
-------
gpr
    Gaussian Process Regression for concentration estimation with uncertainty.
    ``GPRCalibration.predict_single(delta_lambda)`` → ``(ppm, uncertainty_ppm)``.
transforms
    Concentration-axis transforms (log10, log, sqrt, linear) for curve fitting.
    ``transform_concentrations(concs, mode)`` → ``(transformed, meta)``.
multi_roi
    Multi-ROI fusion: candidate selection and multivariate linear calibration.
    ``select_multi_roi_candidates(discovered_roi)`` → list of candidates.
    ``fit_multi_roi_fusion(discovered_roi, concentrations)`` → metrics dict.
roi_scan
    Per-wavelength linear regression scan for ROI discovery.
    ``compute_concentration_response(stable_by_conc, cfg)`` → ``(response, avg_by_conc)``.
"""

from src.calibration.active_learning import BayesianExperimentDesigner
from src.calibration.conformal import ConformalCalibrator
from src.calibration.gpr import GPRCalibration
from src.calibration.multi_roi import fit_multi_roi_fusion, select_multi_roi_candidates
from src.calibration.physics_kernel import (
    LangmuirMeanFunction,
    PhysicsInformedGPR,
    fit_langmuir_params,
)
from src.calibration.roi_scan import (
    RoiScanConfig,
    compute_concentration_response,
    stack_trials_for_response,
)
from src.calibration.selectivity import SelectivityAnalyzer, SelectivityReport
from src.calibration.batch_reproducibility import (
    BatchReproducibilityAnalyzer,
    BatchReproducibilityReport,
)
from src.calibration.pls import PLSCalibration, PLSFitResult
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
]
