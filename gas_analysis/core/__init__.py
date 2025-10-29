"""Core analysis pipeline utilities for gas analysis.

Modules:
- pipeline: loading, transmittance, stability, ROI & calibration, exports
- preprocessing: spectral preprocessing utilities
- dynamics: response and recovery time analysis
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
    run_full_pipeline,
)
from .preprocessing import (
    baseline_correction,
    smooth_spectrum,
    normalize_spectrum,
    estimate_noise_metrics,
    detect_outliers,
)
from .dynamics import compute_response_recovery_times
