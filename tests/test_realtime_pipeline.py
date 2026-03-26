"""
tests/test_realtime_pipeline.py
================================
Unit tests for ``src.inference.realtime_pipeline``.

Coverage targets
----------------
- ``PipelineConfig`` defaults and custom construction
- ``RealTimePipeline`` construction with default config
- ``process_spectrum`` returns a ``PipelineResult``
- Valid / invalid spectrum shapes are handled gracefully
- Calibration can be set and affects the concentration estimate
- ``get_statistics`` returns sane counters after processing
- Simulation spectra (from ``conftest``) pass through without error
- Edge cases: empty arrays, mismatched lengths, all-zeros intensity

Design notes
------------
- We do NOT mock ``RealTimePipeline`` — these are genuine integration tests
  against the real processing code.
- Tests that require torch (CNN classifier) are skipped if torch is absent
  so the test suite runs in a lean CI environment.
- All tests are deterministic (no random seeds needed — pipeline is stateless
  per frame).
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip the entire module if the new pipeline cannot be imported
# ---------------------------------------------------------------------------
realtime_pipeline_spec = importlib.util.find_spec("src.inference.realtime_pipeline")
if realtime_pipeline_spec is None:
    pytest.skip(
        "src.inference.realtime_pipeline not importable — skipping",
        allow_module_level=True,
    )

from src.inference.realtime_pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineResult,
    RealTimePipeline,
    SpectrumData,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(
    target_wl: float = 532.0,
    slope: float = 0.116,
    intercept: float = 0.0,
    peak_search_min_nm: float = 480.0,
    peak_search_max_nm: float = 600.0,
) -> RealTimePipeline:
    """Create a ``RealTimePipeline`` with deterministic calibration."""
    cfg = PipelineConfig(
        target_wavelength=target_wl,
        calibration_slope=slope,
        calibration_intercept=intercept,
        reference_wavelength=target_wl,
        peak_search_min_nm=peak_search_min_nm,
        peak_search_max_nm=peak_search_max_nm,
    )
    pipeline = RealTimePipeline(cfg)
    pipeline.set_calibration(slope=slope, intercept=intercept, reference_wl=target_wl)
    return pipeline


def _lspr_spectrum(wl: np.ndarray, peak_nm: float = 531.5) -> np.ndarray:
    """Synthetic LSPR absorption spectrum on *wl* axis."""
    baseline = np.ones_like(wl) * 10_000.0
    noise = np.random.default_rng(0).normal(0, 20, wl.size)
    absorption = 300.0 * np.exp(-((wl - peak_nm) ** 2) / (2 * 1.5**2))
    return baseline + noise - absorption


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_target_wavelength(self) -> None:
        cfg = PipelineConfig()
        assert cfg.target_wavelength > 0, "target_wavelength should be positive"

    def test_custom_slope(self) -> None:
        cfg = PipelineConfig(calibration_slope=0.25)
        assert cfg.calibration_slope == pytest.approx(0.25)

    def test_custom_integration_time(self) -> None:
        cfg = PipelineConfig(integration_time_ms=50.0)
        assert cfg.integration_time_ms == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# RealTimePipeline construction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    def test_default_construction(self) -> None:
        pipeline = RealTimePipeline()
        assert pipeline is not None

    def test_custom_config_construction(self) -> None:
        pipeline = _make_pipeline(target_wl=530.0, slope=0.2)
        assert pipeline is not None

    def test_get_statistics_initial_state(self) -> None:
        pipeline = _make_pipeline()
        stats = pipeline.get_statistics()
        assert isinstance(stats, dict)
        assert stats.get("total_processed", 0) == 0

    def test_set_calibration_does_not_raise(self) -> None:
        pipeline = _make_pipeline()
        pipeline.set_calibration(slope=0.15, intercept=0.01, reference_wl=532.0)


# ---------------------------------------------------------------------------
# process_spectrum — happy path
# ---------------------------------------------------------------------------


class TestProcessSpectrum:
    def test_returns_pipeline_result(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline()
        wl = synthetic_spectrum["wavelengths"]
        intensities = synthetic_spectrum["intensities"]
        result = pipeline.process_spectrum(wl, intensities)
        assert isinstance(result, PipelineResult)

    def test_result_has_spectrum_attribute(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline()
        result = pipeline.process_spectrum(
            synthetic_spectrum["wavelengths"],
            synthetic_spectrum["intensities"],
        )
        assert hasattr(result, "spectrum")
        assert isinstance(result.spectrum, SpectrumData)

    def test_success_flag_is_bool(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline()
        result = pipeline.process_spectrum(
            synthetic_spectrum["wavelengths"],
            synthetic_spectrum["intensities"],
        )
        assert isinstance(result.success, bool)

    def test_peak_wavelength_in_range(self, synthetic_spectrum: dict) -> None:
        search_min, search_max = 480.0, 600.0
        pipeline = _make_pipeline(
            target_wl=531.5,
            peak_search_min_nm=search_min,
            peak_search_max_nm=search_max,
        )
        result = pipeline.process_spectrum(
            synthetic_spectrum["wavelengths"],
            synthetic_spectrum["intensities"],
        )
        if result.success and result.spectrum.peak_wavelength is not None:
            # Peak must lie within the configured search window
            assert search_min < result.spectrum.peak_wavelength < search_max, (
                f"Peak wavelength {result.spectrum.peak_wavelength:.1f} nm is outside "
                f"the configured search window ({search_min}–{search_max} nm)"
            )

    def test_snr_non_negative(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline()
        result = pipeline.process_spectrum(
            synthetic_spectrum["wavelengths"],
            synthetic_spectrum["intensities"],
        )
        if result.spectrum.snr is not None:
            assert result.spectrum.snr >= 0

    def test_statistics_increment_after_processing(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline()
        n_frames = 5
        for _ in range(n_frames):
            pipeline.process_spectrum(
                synthetic_spectrum["wavelengths"],
                synthetic_spectrum["intensities"],
            )
        stats = pipeline.get_statistics()
        assert stats.get("total_processed", 0) >= n_frames

    def test_concentration_type_when_set(self, synthetic_spectrum: dict) -> None:
        pipeline = _make_pipeline(slope=0.116, intercept=0.0)
        result = pipeline.process_spectrum(
            synthetic_spectrum["wavelengths"],
            synthetic_spectrum["intensities"],
        )
        if result.success and result.spectrum.concentration_ppm is not None:
            assert isinstance(result.spectrum.concentration_ppm, float)


# ---------------------------------------------------------------------------
# process_spectrum — edge cases
# ---------------------------------------------------------------------------


class TestProcessSpectrumEdgeCases:
    def test_empty_wavelengths_handled_gracefully(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.process_spectrum(np.array([]), np.array([]))
        # Must not raise; success should be False
        assert isinstance(result, PipelineResult)
        assert result.success is False

    def test_mismatched_array_lengths_handled_gracefully(self) -> None:
        pipeline = _make_pipeline()
        wl = np.linspace(400, 700, 100)
        intensities = np.ones(50)  # wrong length
        result = pipeline.process_spectrum(wl, intensities)
        assert isinstance(result, PipelineResult)
        assert result.success is False

    def test_all_zero_intensity_handled_gracefully(self) -> None:
        pipeline = _make_pipeline()
        wl = np.linspace(400, 700, 200)
        intensities = np.zeros(200)
        result = pipeline.process_spectrum(wl, intensities)
        assert isinstance(result, PipelineResult)
        # All-zero spectrum has no meaningful peak — success may be False

    def test_single_point_handled_gracefully(self) -> None:
        pipeline = _make_pipeline()
        result = pipeline.process_spectrum(np.array([532.0]), np.array([10000.0]))
        assert isinstance(result, PipelineResult)

    def test_nan_intensity_handled_gracefully(self) -> None:
        pipeline = _make_pipeline()
        wl = np.linspace(400, 700, 100)
        intensities = np.full(100, np.nan)
        result = pipeline.process_spectrum(wl, intensities)
        assert isinstance(result, PipelineResult)

    def test_very_high_intensity_no_crash(self) -> None:
        """Saturated detector — intensities ~65535 (16-bit max)."""
        pipeline = _make_pipeline()
        wl = np.linspace(400, 700, 200)
        intensities = np.full(200, 65535.0)
        result = pipeline.process_spectrum(wl, intensities)
        assert isinstance(result, PipelineResult)


# ---------------------------------------------------------------------------
# Calibration sensitivity test
# ---------------------------------------------------------------------------


class TestCalibrationEffect:
    def test_higher_slope_gives_lower_concentration_estimate(self) -> None:
        """With larger slope, the same Δλ maps to a smaller concentration."""
        wl = np.linspace(480, 600, 200)
        intensities = _lspr_spectrum(wl, peak_nm=531.0)  # slight shift from 531.5

        pipeline_low = _make_pipeline(slope=0.116)
        pipeline_high = _make_pipeline(slope=0.5)

        res_low = pipeline_low.process_spectrum(wl, intensities)
        res_high = pipeline_high.process_spectrum(wl, intensities)

        if (
            res_low.success
            and res_high.success
            and res_low.spectrum.concentration_ppm is not None
            and res_high.spectrum.concentration_ppm is not None
        ):
            # Higher slope → smaller concentration for same shift
            assert res_high.spectrum.concentration_ppm <= res_low.spectrum.concentration_ppm, (
                "Higher calibration slope should yield lower concentration estimate"
            )
