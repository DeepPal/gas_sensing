"""Core analysis pipeline utilities for gas analysis.

Modules:
- pipeline: loading, transmittance, stability, ROI & calibration, exports
- preprocessing: spectral preprocessing utilities
- dynamics: response and recovery time analysis
"""

from .dynamics import compute_response_recovery_times
from .pipeline import (
    compute_transmittance,
    compute_transmittance_on_frames,
    find_roi_and_calibration,
    load_reference_csv,
    run_full_pipeline,
    save_aggregated_spectra,
    save_calibration_outputs,
    save_canonical_spectra,
    scan_experiment_root,
    select_canonical_per_concentration,
)
from .preprocessing import (
    baseline_correction,
    detect_outliers,
    estimate_noise_metrics,
    normalize_spectrum,
    smooth_spectrum,
)
