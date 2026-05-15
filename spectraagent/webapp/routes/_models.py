"""Shared Pydantic request/response models for SpectraAgent route modules.

These are extracted from server.py so all route modules can import them
without circular dependencies.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AcquisitionConfig(BaseModel):
    integration_time_ms: float = 50.0
    gas_label: str = "unknown"
    target_concentration: float | None = None
    temperature_c: float | None = None   # room temperature at session start (°C)
    humidity_pct: float | None = None    # relative humidity (%) — LSPR sensitivity ~0.02 nm/°C


class CalibrationPoint(BaseModel):
    concentration: float
    delta_lambda: float


class AskRequest(BaseModel):
    query: str = Field(..., max_length=2000)


class ReportRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class AgentSettings(BaseModel):
    auto_explain: bool


class QualitySettings(BaseModel):
    saturation_threshold: float | None = None
    snr_warn_threshold: float | None = None


class DriftSettings(BaseModel):
    drift_threshold_nm_per_min: float | None = None
    window_frames: int | None = None


class SensitivityFitRequest(BaseModel):
    """Fit sensitivity matrix from single-analyte calibration data."""
    analytes: list[str]
    n_peaks: int
    calibration_data: list[dict]
    # Each entry: {analyte, peak_idx, conc_ppm: [..], shifts_nm: [..]}


class MixtureInferenceRequest(BaseModel):
    """Estimate analyte concentrations from observed peak shifts."""
    delta_lambda: list[float]          # observed peak shifts (nm), one per peak
    analytes: list[str]
    S_matrix: list[list[float]]        # [[S_00, S_01, ...], [S_10, ...]] (N×M)
    Kd_matrix: list[list[float]] | None = None   # K_d matrix (ppm), same shape; null = linear
    use_nonlinear: bool = False


class SimGenerateRequest(BaseModel):
    """Generate a batch of synthetic spectra from the physics simulation."""
    peak_nm: float = 700.0
    fwhm_nm: float = 20.0
    wl_start: float = 500.0
    wl_end: float = 900.0
    analyte_name: str = "Gas"
    sensitivity_nm_per_ppm: float = -0.5
    tau_s: float = 30.0
    kd_ppm: float = 100.0
    concentrations: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0]
    n_sessions: int = 5
    random_seed: int = 42
