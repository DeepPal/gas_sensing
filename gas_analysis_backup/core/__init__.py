"""Core analysis pipeline utilities for gas analysis.

Modules:
- pipeline: loading, transmittance, stability, ROI & calibration, exports
"""

from .pipeline import (
    load_reference_csv,
    scan_experiment_root,
    compute_transmittance,
    compute_transmittance_on_frames,
    select_canonical_per_concentration,
    find_roi_and_calibration,
    save_canonical_spectra,
    save_aggregated_spectra,
    save_calibration_outputs,
)
from .dynamics import compute_response_recovery_times
from .compare import merge_calibration_metrics
from .ml import build_ml_dataset, train_gas_predictor
from .synthetic import generate_synthetic_spectra
