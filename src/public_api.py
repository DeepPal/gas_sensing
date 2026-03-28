"""
Au-MIP LSPR Gas Sensing Platform — Public API
===============================================

This module is the **stable import surface** for library consumers.
Import from here, not from internal submodules, so that internal
refactors do not break your code.

Quick start
-----------
::

    from src.public_api import RealTimePipeline, PipelineConfig

    pipeline = RealTimePipeline(PipelineConfig(snr_threshold=5.0))
    pipeline.set_calibration(slope=0.116, intercept=0.0)

    result = pipeline.process_spectrum(wavelengths, intensities)
    if result.success:
        print(f"{result.spectrum.concentration_ppm:.3f} ppm")

Stability contract
------------------
Names exported here are **stable across minor versions**.  Names in
internal ``src.*`` submodules may change without notice.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
from src import __version__

# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
from src.calibration.gpr import GPRCalibration
from src.calibration.roi_scan import RoiScanConfig, compute_concentration_response

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
from src.inference.realtime_pipeline import (
    PipelineConfig,
    PipelineResult,
    RealTimePipeline,
)

# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
from src.models.cnn import CNNGasClassifier

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
from src.reporting import (
    compute_noise_metrics_map,
    compute_roi_performance,
    save_concentration_response_metrics,
    save_concentration_response_plot,
    save_research_grade_calibration_plot,
    save_roi_performance_metrics,
    select_signal_column,
)

# ---------------------------------------------------------------------------
# Schemas / data types
# ---------------------------------------------------------------------------
from src.schemas import (
    KNOWN_GAS_TYPES,
    PredictionResult,
    SessionMeta,
    SpectrumReading,
    normalise_gas_type,
)

__all__ = [
    # version
    "__version__",
    # pipeline
    "RealTimePipeline",
    "PipelineConfig",
    "PipelineResult",
    # models
    "CNNGasClassifier",
    # calibration
    "GPRCalibration",
    "RoiScanConfig",
    "compute_concentration_response",
    # schemas
    "SpectrumReading",
    "PredictionResult",
    "SessionMeta",
    "KNOWN_GAS_TYPES",
    "normalise_gas_type",
    # reporting
    "select_signal_column",
    "compute_noise_metrics_map",
    "compute_roi_performance",
    "save_concentration_response_metrics",
    "save_roi_performance_metrics",
    "save_concentration_response_plot",
    "save_research_grade_calibration_plot",
]
