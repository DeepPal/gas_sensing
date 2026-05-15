"""
src.reporting
=============
Report generation, metric aggregation, and output formatting utilities.

Modules
-------
metrics
    Pure computation: noise maps, ROI repeatability/performance, dynamics
    summaries, QC summaries, and signal column selection helpers.
    All functions are CONFIG-free — all parameters passed explicitly.
environment
    Environment compensation metadata and coefficient estimation.
    All functions are CONFIG-free — all parameters passed explicitly.
io
    JSON/CSV serialisers for pipeline analysis outputs.
    All functions are CONFIG-free — all parameters passed explicitly.
plots
    Matplotlib visualisations for pipeline analysis outputs.
    All functions are CONFIG-free — all parameters passed explicitly.
publication
    Publication-quality figure export with journal presets (ACS, Nature,
    RSC, Elsevier).  Provides calibration, spectral overlay, and PLS
    diagnostic figures at journal-correct dimensions and DPI.
"""

from src.reporting.environment import (
    compute_environment_coefficients,
    compute_environment_summary,
)
from src.reporting.io import (
    save_aggregated_spectra,
    save_aggregated_summary,
    save_canonical_spectra,
    save_concentration_response_metrics,
    save_dynamics_error,
    save_dynamics_summary,
    save_environment_compensation_summary,
    save_noise_metrics,
    save_quality_summary,
    save_roi_performance_metrics,
)
from src.reporting.metrics import (
    common_signal_columns,
    compute_comprehensive_sensor_characterization,
    compute_noise_metrics_map,
    compute_roi_performance,
    compute_roi_repeatability,
    select_common_signal,
    select_signal_column,
    summarize_dynamics_metrics,
    summarize_quality_control,
    summarize_top_comparison,
)
from src.reporting.plots import (
    save_aggregated_plots,
    save_calibration_outputs,
    save_canonical_overlay,
    save_concentration_response_plot,
    save_research_grade_calibration_plot,
    save_roi_discovery_plot,
    save_roi_repeatability_plot,
    save_spectral_response_diagnostic,
    save_wavelength_shift_visualization,
)
from src.reporting.publication import (
    JOURNAL_PRESETS,
    journal_style,
    list_presets,
    preset_info,
    save_calibration_figure,
    save_pls_diagnostics_figure,
    save_spectral_overlay_figure,
)
from src.reporting.scientific_summary import (
    build_deterministic_scientific_report,
    save_deterministic_scientific_summary,
    session_analysis_to_dict,
)

__all__ = [
    # metrics
    "compute_comprehensive_sensor_characterization",
    "select_signal_column",
    "select_common_signal",
    "common_signal_columns",
    "compute_noise_metrics_map",
    "compute_roi_repeatability",
    "compute_roi_performance",
    "summarize_dynamics_metrics",
    "summarize_quality_control",
    "summarize_top_comparison",
    # environment
    "compute_environment_summary",
    "compute_environment_coefficients",
    # io
    "save_canonical_spectra",
    "save_aggregated_spectra",
    "save_noise_metrics",
    "save_quality_summary",
    "save_aggregated_summary",
    "save_roi_performance_metrics",
    "save_dynamics_summary",
    "save_dynamics_error",
    "save_concentration_response_metrics",
    "save_environment_compensation_summary",
    # plots
    "save_roi_discovery_plot",
    "save_concentration_response_plot",
    "save_wavelength_shift_visualization",
    "save_research_grade_calibration_plot",
    "save_spectral_response_diagnostic",
    "save_roi_repeatability_plot",
    "save_aggregated_plots",
    "save_canonical_overlay",
    "save_calibration_outputs",
    # publication
    "JOURNAL_PRESETS",
    "journal_style",
    "list_presets",
    "preset_info",
    "save_calibration_figure",
    "save_spectral_overlay_figure",
    "save_pls_diagnostics_figure",
    "build_deterministic_scientific_report",
    "save_deterministic_scientific_summary",
    "session_analysis_to_dict",
]
