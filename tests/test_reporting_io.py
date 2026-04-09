"""Tests for src.reporting.io — pure JSON/CSV serialisers."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def out_root(tmp_path):
    return str(tmp_path)


def _make_spectrum(wl_start=600.0, n=10) -> pd.DataFrame:
    wl = np.linspace(wl_start, wl_start + n - 1, n)
    return pd.DataFrame({"wavelength": wl, "transmittance": np.random.rand(n)})


# ---------------------------------------------------------------------------
# save_canonical_spectra
# ---------------------------------------------------------------------------


class TestSaveCanonicalSpectra:
    def test_creates_one_csv_per_concentration(self, out_root):
        canonical = {0.5: _make_spectrum(), 1.0: _make_spectrum()}
        paths = save_canonical_spectra(canonical, out_root)
        assert len(paths) == 2
        for p in paths:
            assert os.path.isfile(p)
            assert p.endswith(".csv")

    def test_filename_uses_concentration(self, out_root):
        canonical = {2.0: _make_spectrum()}
        paths = save_canonical_spectra(canonical, out_root)
        assert os.path.basename(paths[0]) == "2_stable.csv"

    def test_csv_content_round_trips(self, out_root):
        df = _make_spectrum()
        save_canonical_spectra({0.5: df}, out_root)
        loaded = pd.read_csv(os.path.join(out_root, "stable_selected", "0.5_stable.csv"))
        assert list(loaded.columns) == list(df.columns)
        assert len(loaded) == len(df)


# ---------------------------------------------------------------------------
# save_aggregated_spectra
# ---------------------------------------------------------------------------


class TestSaveAggregatedSpectra:
    def test_creates_per_trial_csvs(self, out_root):
        aggregated = {0.5: {"trial_0": _make_spectrum(), "trial_1": _make_spectrum()}}
        saved = save_aggregated_spectra(aggregated, out_root)
        assert set(saved[0.5].keys()) == {"trial_0", "trial_1"}
        for path in saved[0.5].values():
            assert os.path.isfile(path)

    def test_stale_csvs_removed(self, out_root):
        # Write a stale file, then overwrite with new data
        conc_dir = os.path.join(out_root, "aggregated", "0.5")
        os.makedirs(conc_dir, exist_ok=True)
        stale = os.path.join(conc_dir, "stale.csv")
        open(stale, "w").close()
        save_aggregated_spectra({0.5: {"trial_0": _make_spectrum()}}, out_root)
        assert not os.path.isfile(stale)


# ---------------------------------------------------------------------------
# save_noise_metrics
# ---------------------------------------------------------------------------


class TestSaveNoiseMetrics:
    def test_creates_json(self, out_root):
        metrics = {0.5: {"trial_0": {"rms": 0.01, "snr": 20.0}}}
        path = save_noise_metrics(metrics, out_root)
        assert os.path.isfile(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "0.5" in data

    def test_numpy_scalars_serialised(self, out_root):
        metrics = {1.0: {"t": {"rms": np.float64(0.02)}}}
        path = save_noise_metrics(metrics, out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert isinstance(data["1.0"]["t"]["rms"], float)


# ---------------------------------------------------------------------------
# save_quality_summary
# ---------------------------------------------------------------------------


class TestSaveQualitySummary:
    def test_creates_json(self, out_root):
        qc = {"pass_rate": 0.95, "total": 20, "passed": 19}
        path = save_quality_summary(qc, out_root)
        assert os.path.isfile(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["pass_rate"] == 0.95


# ---------------------------------------------------------------------------
# save_aggregated_summary
# ---------------------------------------------------------------------------


class TestSaveAggregatedSummary:
    def test_creates_csv_with_expected_columns(self, out_root):
        aggregated = {0.5: {"t0": _make_spectrum()}}
        noise = {0.5: {"t0": {"rms": 0.01, "mad": 0.005, "spectral_entropy": 3.5, "snr": 25.0}}}
        path = save_aggregated_summary(aggregated, noise, out_root)
        df = pd.read_csv(path)
        assert "concentration" in df.columns
        assert "rms" in df.columns
        assert len(df) == 1


# ---------------------------------------------------------------------------
# save_roi_performance_metrics
# ---------------------------------------------------------------------------


class TestSaveRoiPerformanceMetrics:
    def test_returns_none_for_empty(self, out_root):
        assert save_roi_performance_metrics({}, out_root) is None

    def test_creates_json_when_nonempty(self, out_root):
        perf = {"lod": 0.1, "loq": 0.3, "r2": 0.99}
        path = save_roi_performance_metrics(perf, out_root)
        assert path is not None
        assert os.path.isfile(path)
        assert json.loads(Path(path).read_text(encoding="utf-8"))["lod"] == 0.1


# ---------------------------------------------------------------------------
# save_dynamics_summary / save_dynamics_error
# ---------------------------------------------------------------------------


class TestSaveDynamics:
    def test_summary_creates_json(self, out_root):
        summary = {"t90_response": 4.2, "t90_recovery": 8.1}
        path = save_dynamics_summary(summary, out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["t90_response"] == 4.2

    def test_error_creates_error_key(self, out_root):
        path = save_dynamics_error("not enough data", out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["error"] == "not enough data"


# ---------------------------------------------------------------------------
# save_concentration_response_metrics
# ---------------------------------------------------------------------------


class TestSaveConcentrationResponseMetrics:
    def test_creates_json_with_repeatability(self, out_root):
        response = {"roi_start_wavelength": 650.0, "roi_end_wavelength": 700.0}
        repeatability = {"global": {"std_transmittance": 0.003}}
        path = save_concentration_response_metrics(response, repeatability, out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert "roi_repeatability" in data
        assert data["roi_start_wavelength"] == 650.0

    def test_custom_name(self, out_root):
        path = save_concentration_response_metrics({}, {}, out_root, name="my_response")
        assert "my_response.json" in path


# ---------------------------------------------------------------------------
# save_environment_compensation_summary
# ---------------------------------------------------------------------------


class TestSaveEnvironmentCompensationSummary:
    def test_creates_json(self, out_root):
        info = {"temperature_coefficient": -0.002, "humidity_coefficient": 0.001}
        path = save_environment_compensation_summary(info, out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["temperature_coefficient"] == pytest.approx(-0.002)

    def test_nan_serialised_as_null(self, out_root):
        info = {"value": float("nan")}
        path = save_environment_compensation_summary(info, out_root)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assert data["value"] is None
