"""Tests for ci_low/ci_high fields and CalibrationStage.set_gpr()."""
import numpy as np
import pytest
from src.inference.realtime_pipeline import (
    CalibrationStage,
    PipelineConfig,
    SpectrumData,
)


def _make_spectrum(wl: np.ndarray, intensities: np.ndarray) -> SpectrumData:
    sd = SpectrumData(wavelengths=wl, intensities=intensities)
    sd.processed_intensities = intensities.copy()
    sd.wavelength_shift = -2.0
    return sd


def test_spectrum_data_has_ci_fields():
    """SpectrumData must have ci_low and ci_high fields defaulting to None."""
    wl = np.linspace(300, 1000, 3648)
    sd = SpectrumData(wavelengths=wl, intensities=np.ones(3648))
    assert hasattr(sd, "ci_low")
    assert hasattr(sd, "ci_high")
    assert sd.ci_low is None
    assert sd.ci_high is None


def test_calibration_stage_has_set_gpr():
    """CalibrationStage must expose set_gpr(model, X_cal, y_cal)."""
    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    assert callable(getattr(stage, "set_gpr", None))


def test_set_gpr_populates_ci_on_process():
    """After set_gpr() with calibration data, process() populates ci_low/ci_high."""
    from src.calibration.gpr import GPRCalibration

    np.random.seed(0)
    shifts = np.linspace(-5, -0.2, 20)
    concs = -shifts * 2.5 + np.random.normal(0, 0.1, 20)
    gpr = GPRCalibration()
    gpr.fit(shifts.reshape(-1, 1), concs)

    X_cal = shifts[:10].reshape(-1, 1)
    y_cal = concs[:10]

    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    stage.set_gpr(gpr, X_cal, y_cal)

    wl = np.linspace(300, 1000, 3648)
    spectrum = _make_spectrum(wl, np.random.rand(3648))
    result_spectrum, _ = stage.process(spectrum)

    assert result_spectrum.ci_low is not None
    assert result_spectrum.ci_high is not None
    assert result_spectrum.ci_low < result_spectrum.ci_high


def test_no_gpr_no_ci():
    """Without set_gpr(), ci_low and ci_high remain None after process()."""
    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    wl = np.linspace(300, 1000, 3648)
    spectrum = _make_spectrum(wl, np.random.rand(3648))
    result_spectrum, _ = stage.process(spectrum)
    assert result_spectrum.ci_low is None
    assert result_spectrum.ci_high is None


def test_set_gpr_without_cal_data_skips_conformal():
    """set_gpr(model) without X_cal/y_cal must not raise and must skip CI."""
    from src.calibration.gpr import GPRCalibration

    np.random.seed(1)
    shifts = np.linspace(-4, -0.5, 10)
    concs = -shifts * 2.0
    gpr = GPRCalibration()
    gpr.fit(shifts.reshape(-1, 1), concs)

    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    stage.set_gpr(gpr)  # no calibration data

    wl = np.linspace(300, 1000, 3648)
    spectrum = _make_spectrum(wl, np.random.rand(3648))
    result_spectrum, _ = stage.process(spectrum)
    assert result_spectrum.ci_low is None
    assert result_spectrum.ci_high is None
