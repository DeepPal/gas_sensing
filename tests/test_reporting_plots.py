"""Smoke tests for src.reporting.plots — matplotlib visualisations.

Each test verifies that the function:
  1. Returns the expected path (or None for empty inputs)
  2. Writes a non-empty PNG when it returns a path
  3. Closes all figures (no matplotlib memory leak)

Plot correctness (pixel values) is not tested — that would be fragile across
matplotlib versions.  Functional correctness is covered by the src/ module's
design and the underlying pipeline integration tests.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_root(tmp_path):
    return str(tmp_path)


def _spectrum(wl_start=600.0, n=50, signal="transmittance") -> pd.DataFrame:
    wl = np.linspace(wl_start, wl_start + n - 1, n)
    return pd.DataFrame({"wavelength": wl, signal: np.random.rand(n) * 0.3 + 0.5})


def _canonical(concs=(0.5, 1.0, 2.0)):
    return {c: _spectrum() for c in concs}


def _assert_png(path: str | None) -> None:
    """Assert path exists and is a non-empty PNG."""
    assert path is not None
    assert os.path.isfile(path), f"Expected file at {path}"
    assert os.path.getsize(path) > 0
    with open(path, "rb") as f:
        header = f.read(8)
    assert header[:4] == b"\x89PNG", f"Not a PNG: {path}"


def _no_open_figures():
    return len(plt.get_fignums()) == 0


# ---------------------------------------------------------------------------
# save_roi_discovery_plot
# ---------------------------------------------------------------------------


class TestSaveRoiDiscoveryPlot:
    def test_returns_none_when_no_candidates(self, out_root):
        assert save_roi_discovery_plot({}, out_root) is None
        assert save_roi_discovery_plot({"candidates": []}, out_root) is None

    def test_creates_png(self, out_root):
        discovery = {
            "candidates": [
                {"center_nm": 650.0, "slope_nm_per_ppm": 0.05, "r2": 0.95},
                {"center_nm": 670.0, "slope_nm_per_ppm": 0.03, "r2": 0.88},
            ]
        }
        path = save_roi_discovery_plot(discovery, out_root)
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_concentration_response_plot
# ---------------------------------------------------------------------------


class TestSaveConcentrationResponsePlot:
    def test_returns_none_for_empty_response(self, out_root):
        assert save_concentration_response_plot({}, {}, out_root) is None

    def test_creates_png(self, out_root):
        wl = np.linspace(600, 700, 20)
        response = {
            "wavelengths": wl.tolist(),
            "slopes": (np.random.rand(20) * 0.01).tolist(),
            "r2_values": (np.random.rand(20) * 0.5 + 0.5).tolist(),
            "roi_start_wavelength": 630.0,
            "roi_end_wavelength": 670.0,
        }
        avg_by_conc = {0.5: np.random.rand(20), 1.0: np.random.rand(20)}
        path = save_concentration_response_plot(response, avg_by_conc, out_root)
        _assert_png(path)
        assert _no_open_figures()

    def test_x_min_x_max_accepted(self, out_root):
        wl = np.linspace(600, 700, 20)
        response = {
            "wavelengths": wl.tolist(),
            "slopes": [0.01] * 20,
            "r2_values": [0.9] * 20,
            "roi_start_wavelength": 630.0,
            "roi_end_wavelength": 670.0,
        }
        path = save_concentration_response_plot(
            response, {}, out_root, clamp_to_roi=True, x_min=620.0, x_max=680.0
        )
        _assert_png(path)


# ---------------------------------------------------------------------------
# save_wavelength_shift_visualization
# ---------------------------------------------------------------------------


class TestSaveWavelengthShiftVisualization:
    def test_returns_none_for_empty(self, out_root):
        assert save_wavelength_shift_visualization({}, {}, out_root) is None

    def test_creates_png(self, out_root):
        canonical = _canonical()
        calib_result = {
            "slope": -0.5,
            "intercept": 720.0,
            "r2": 0.97,
            "concentrations": [0.5, 1.0, 2.0],
            "peak_wavelengths": [719.75, 719.5, 719.0],
        }
        path = save_wavelength_shift_visualization(canonical, calib_result, out_root)
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_research_grade_calibration_plot
# ---------------------------------------------------------------------------


class TestSaveResearchGradeCalibrationPlot:
    def test_returns_none_for_empty_canonical(self, out_root):
        assert save_research_grade_calibration_plot({}, {}, out_root) is None

    def test_creates_png(self, out_root):
        canonical = _canonical()
        calib_result = {
            "slope": -0.5,
            "intercept": 720.0,
            "r2": 0.97,
            "concentrations": [0.5, 1.0, 2.0],
            "peak_wavelengths": [719.75, 719.5, 719.0],
            "residuals": [0.01, -0.01, 0.0],
            "predictions": [719.75, 719.5, 719.0],
        }
        path = save_research_grade_calibration_plot(canonical, calib_result, out_root)
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_spectral_response_diagnostic
# ---------------------------------------------------------------------------


class TestSaveSpectralResponseDiagnostic:
    def test_returns_none_for_empty(self, out_root):
        assert save_spectral_response_diagnostic({}, out_root) is None

    def test_returns_none_for_single_concentration(self, out_root):
        assert save_spectral_response_diagnostic({0.5: _spectrum()}, out_root) is None

    def test_creates_png(self, out_root):
        canonical = _canonical()
        path = save_spectral_response_diagnostic(
            canonical, out_root, step_nm=5.0, window_nm=10.0
        )
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_roi_repeatability_plot
# ---------------------------------------------------------------------------


class TestSaveRoiRepeatabilityPlot:
    def test_returns_none_when_roi_degenerate(self, out_root):
        response = {"roi_start_wavelength": 650.0, "roi_end_wavelength": 650.0}
        assert save_roi_repeatability_plot({0.5: {"t": _spectrum()}}, response, out_root) is None

    def test_creates_png(self, out_root):
        response = {"roi_start_wavelength": 620.0, "roi_end_wavelength": 660.0}
        stable = {0.5: {"t0": _spectrum(), "t1": _spectrum()}, 1.0: {"t0": _spectrum()}}
        path = save_roi_repeatability_plot(stable, response, out_root)
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_aggregated_plots
# ---------------------------------------------------------------------------


class TestSaveAggregatedPlots:
    def test_creates_one_png_per_trial(self, out_root):
        aggregated = {
            0.5: {"trial_0": _spectrum(), "trial_1": _spectrum()},
            1.0: {"trial_0": _spectrum()},
        }
        plot_paths = save_aggregated_plots(aggregated, out_root)
        assert len(plot_paths[0.5]) == 2
        assert len(plot_paths[1.0]) == 1
        for conc_paths in plot_paths.values():
            for p in conc_paths.values():
                _assert_png(p)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_canonical_overlay
# ---------------------------------------------------------------------------


class TestSaveCanonicalOverlay:
    def test_returns_none_for_empty(self, out_root):
        assert save_canonical_overlay({}, out_root) is None

    def test_creates_png(self, out_root):
        path = save_canonical_overlay(_canonical(), out_root)
        _assert_png(path)
        assert _no_open_figures()


# ---------------------------------------------------------------------------
# save_calibration_outputs
# ---------------------------------------------------------------------------


class TestSaveCalibrationOutputs:
    def test_creates_csv_and_metrics_json(self, out_root):
        calib = {
            "concentrations": [0.5, 1.0, 2.0],
            "peak_wavelengths": [719.75, 719.5, 719.0],
            "slope": -0.25,
            "intercept": 719.875,
            "r2": 0.99,
            "residuals": [0.005, -0.005, 0.0],
            "predictions": [719.75, 719.5, 719.0],
        }
        save_calibration_outputs(calib, out_root)
        assert os.path.isfile(os.path.join(out_root, "metrics", "calibration.csv"))
        assert os.path.isfile(os.path.join(out_root, "metrics", "calibration_metrics.json"))
        _assert_png(os.path.join(out_root, "plots", "calibration.png"))
        assert _no_open_figures()

    def test_name_suffix_applied(self, out_root):
        calib = {
            "concentrations": [0.5, 1.0],
            "peak_wavelengths": [719.75, 719.5],
            "slope": -0.25,
            "intercept": 719.875,
            "r2": 0.99,
        }
        save_calibration_outputs(calib, out_root, name_suffix="_mode1")
        assert os.path.isfile(os.path.join(out_root, "metrics", "calibration_mode1_metrics.json"))
        assert _no_open_figures()

    def test_mutates_plots_key(self, out_root):
        calib = {
            "concentrations": [0.5, 1.0],
            "peak_wavelengths": [719.75, 719.5],
            "slope": -0.25,
            "intercept": 719.875,
            "r2": 0.99,
        }
        save_calibration_outputs(calib, out_root)
        assert "plots" in calib
        assert "calibration" in calib["plots"]
