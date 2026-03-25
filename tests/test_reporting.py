"""Tests for src.reporting.metrics — pure metric aggregation utilities."""
import math

import numpy as np
import pandas as pd
import pytest

from src.reporting.metrics import (
    common_signal_columns,
    compute_noise_metrics_map,
    compute_roi_performance,
    compute_roi_repeatability,
    select_common_signal,
    select_signal_column,
    summarize_dynamics_metrics,
    summarize_quality_control,
    summarize_top_comparison,
)


# ---------------------------------------------------------------------------
# select_signal_column
# ---------------------------------------------------------------------------


class TestSelectSignalColumn:
    def test_absorbance_preferred(self):
        df = pd.DataFrame({"wavelength": [700.0], "absorbance": [0.3], "transmittance": [0.5]})
        assert select_signal_column(df) == "absorbance"

    def test_transmittance_second(self):
        df = pd.DataFrame({"wavelength": [700.0], "transmittance": [0.95]})
        assert select_signal_column(df) == "transmittance"

    def test_intensity_fallback(self):
        df = pd.DataFrame({"wavelength": [700.0], "intensity": [1234.0]})
        assert select_signal_column(df) == "intensity"

    def test_default_when_none_match(self):
        """If no preferred column exists, returns 'intensity' as default."""
        df = pd.DataFrame({"wavelength": [700.0], "custom": [0.5]})
        assert select_signal_column(df) == "intensity"


# ---------------------------------------------------------------------------
# select_common_signal
# ---------------------------------------------------------------------------


class TestSelectCommonSignal:
    def test_empty_frames_returns_none(self):
        assert select_common_signal([]) is None

    def test_common_transmittance(self):
        dfs = [pd.DataFrame({"wavelength": [700.0], "transmittance": [0.9]}) for _ in range(3)]
        assert select_common_signal(dfs) == "transmittance"

    def test_missing_in_one_frame_skipped(self):
        df1 = pd.DataFrame({"wavelength": [700.0], "transmittance": [0.9], "intensity": [500.0]})
        df2 = pd.DataFrame({"wavelength": [700.0], "intensity": [500.0]})
        # transmittance not in df2, falls through to intensity
        assert select_common_signal([df1, df2]) == "intensity"

    def test_no_common_returns_none(self):
        df1 = pd.DataFrame({"wavelength": [700.0], "other1": [0.9]})
        df2 = pd.DataFrame({"wavelength": [700.0], "other2": [0.5]})
        assert select_common_signal([df1, df2]) is None

    def test_custom_priority(self):
        dfs = [pd.DataFrame({"wavelength": [700.0], "absorbance": [0.1]}) for _ in range(2)]
        assert select_common_signal(dfs, priority=("absorbance",)) == "absorbance"


# ---------------------------------------------------------------------------
# common_signal_columns
# ---------------------------------------------------------------------------


class TestCommonSignalColumns:
    def test_empty_returns_empty(self):
        assert common_signal_columns([]) == []

    def test_wavelength_excluded(self):
        dfs = [pd.DataFrame({"wavelength": [700.0], "intensity": [0.5]}) for _ in range(2)]
        result = common_signal_columns(dfs)
        assert "wavelength" not in result
        assert "intensity" in result

    def test_returns_sorted(self):
        dfs = [pd.DataFrame({"wavelength": [700.0], "z_col": [0.5], "a_col": [0.5]})
               for _ in range(2)]
        result = common_signal_columns(dfs)
        assert result == sorted(result)

    def test_excludes_non_common_columns(self):
        df1 = pd.DataFrame({"wavelength": [700.0], "a": [0.5], "b": [0.5]})
        df2 = pd.DataFrame({"wavelength": [700.0], "b": [0.5], "c": [0.5]})
        assert common_signal_columns([df1, df2]) == ["b"]


# ---------------------------------------------------------------------------
# compute_noise_metrics_map
# ---------------------------------------------------------------------------


def _make_spectrum(seed: int = 0, signal: float = 0.5, noise: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    wl = np.linspace(600.0, 800.0, 100)
    intensity = rng.normal(signal, noise, 100)
    return pd.DataFrame({"wavelength": wl, "intensity": intensity})


class TestComputeNoiseMetricsMap:
    def test_keys_match_aggregated(self):
        aggregated = {
            0.5: {"t1": _make_spectrum(0), "t2": _make_spectrum(1)},
            1.0: {"t1": _make_spectrum(2)},
        }
        result = compute_noise_metrics_map(aggregated)
        assert set(result.keys()) == {0.5, 1.0}
        assert set(result[0.5].keys()) == {"t1", "t2"}

    def test_each_entry_has_snr(self):
        aggregated = {0.5: {"t1": _make_spectrum(0)}}
        result = compute_noise_metrics_map(aggregated)
        assert "snr" in result[0.5]["t1"]

    def test_snr_positive(self):
        aggregated = {0.5: {"t1": _make_spectrum(0, signal=1.0, noise=0.001)}}
        result = compute_noise_metrics_map(aggregated)
        assert result[0.5]["t1"]["snr"] > 0


# ---------------------------------------------------------------------------
# compute_roi_repeatability
# ---------------------------------------------------------------------------


def _make_aggregated(
    concs: list[float],
    n_trials: int = 3,
    roi_signal: float = 0.7,
    noise: float = 0.005,
    seed: int = 0,
) -> dict[float, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    agg = {}
    wl = np.linspace(680.0, 730.0, 80)
    for c in concs:
        agg[c] = {}
        for t in range(n_trials):
            sig = rng.normal(roi_signal + 0.01 * c, noise, 80)
            agg[c][f"t{t}"] = pd.DataFrame({"wavelength": wl, "transmittance": sig})
    return agg


class TestComputeRoiRepeatability:
    def test_per_concentration_keys(self):
        agg = _make_aggregated([0.5, 1.0, 2.0])
        response = {"roi_start_wavelength": 695.0, "roi_end_wavelength": 710.0}
        result = compute_roi_repeatability(agg, response)
        assert "per_concentration" in result
        assert "global" in result
        assert "0.5" in result["per_concentration"]

    def test_cv_non_negative(self):
        agg = _make_aggregated([0.5, 1.0])
        response = {"roi_start_wavelength": 695.0, "roi_end_wavelength": 710.0}
        result = compute_roi_repeatability(agg, response)
        for conc_stats in result["per_concentration"].values():
            assert float(conc_stats["cv_transmittance"]) >= 0

    def test_roi_width_correct(self):
        agg = _make_aggregated([0.5])
        response = {"roi_start_wavelength": 695.0, "roi_end_wavelength": 715.0}
        result = compute_roi_repeatability(agg, response)
        assert result["roi_width"] == pytest.approx(20.0)

    def test_out_of_range_fallback(self):
        """ROI outside wavelength range falls back to interpolation — no crash."""
        agg = _make_aggregated([0.5])
        response = {"roi_start_wavelength": 850.0, "roi_end_wavelength": 900.0}
        result = compute_roi_repeatability(agg, response)
        assert "per_concentration" in result


# ---------------------------------------------------------------------------
# compute_roi_performance
# ---------------------------------------------------------------------------


class TestComputeRoiPerformance:
    def _make_repeatability(self, concs: list[float]) -> dict:
        """Build a synthetic repeatability dict with known signal trend."""
        per_conc = {}
        for c in concs:
            # signal decreases with concentration → negative slope
            mean_val = 0.8 - 0.05 * c
            per_conc[str(c)] = {
                "mean_transmittance": mean_val,
                "std_transmittance": 0.002,
                "cv_transmittance": 0.002 / mean_val,
                "trial_count": 3,
            }
        return {
            "per_concentration": per_conc,
            "global": {"std_transmittance": 0.003, "mean_transmittance": 0.75, "count": 9},
        }

    def test_returns_empty_for_one_conc(self):
        rep = self._make_repeatability([0.5])
        assert compute_roi_performance(rep) == {}

    def test_regression_keys_present(self):
        rep = self._make_repeatability([0.5, 1.0, 2.0])
        perf = compute_roi_performance(rep)
        for key in ["regression_slope", "regression_r2", "lod_ppm", "loq_ppm"]:
            assert key in perf

    def test_lod_loq_ratio(self):
        """LOQ should be ~10/3 × LOD by default."""
        rep = self._make_repeatability([0.5, 1.0, 2.0])
        perf = compute_roi_performance(rep)
        if math.isfinite(perf["lod_ppm"]) and perf["lod_ppm"] > 0:
            ratio = perf["loq_ppm"] / perf["lod_ppm"]
            assert ratio == pytest.approx(10.0 / 3.0, rel=1e-3)

    def test_custom_sigma(self):
        rep = self._make_repeatability([0.5, 1.0, 2.0])
        perf = compute_roi_performance(rep, lod_sigma=5.0, loq_sigma=20.0)
        if math.isfinite(perf["lod_ppm"]) and perf["lod_ppm"] > 0:
            assert perf["loq_ppm"] / perf["lod_ppm"] == pytest.approx(4.0, rel=1e-3)

    def test_concentrations_sorted(self):
        # Feed unsorted concentrations — should be sorted in output
        rep = self._make_repeatability([2.0, 0.5, 1.0])
        perf = compute_roi_performance(rep)
        concs = perf["concentrations"]
        assert concs == sorted(concs)


# ---------------------------------------------------------------------------
# summarize_dynamics_metrics
# ---------------------------------------------------------------------------


class TestSummarizeDynamicsMetrics:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "concentration": [0.5, 0.5, 1.0, 1.0],
            "response_time_T90": [12.0, 13.0, 10.0, 11.0],
            "recovery_time_T10": [20.0, 21.0, 18.0, 19.0],
        })

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=["concentration", "response_time_T90", "recovery_time_T10"])
        assert summarize_dynamics_metrics(df) == {}

    def test_per_concentration_keys(self):
        result = summarize_dynamics_metrics(self._make_df())
        assert "per_concentration" in result
        assert "overall" in result
        assert "0.5" in result["per_concentration"]
        assert "1.0" in result["per_concentration"]

    def test_mean_values_correct(self):
        result = summarize_dynamics_metrics(self._make_df())
        assert result["per_concentration"]["0.5"]["mean_T90"] == pytest.approx(12.5)
        assert result["per_concentration"]["1.0"]["mean_T90"] == pytest.approx(10.5)

    def test_overall_count(self):
        result = summarize_dynamics_metrics(self._make_df())
        assert result["overall"]["count"] == 4

    def test_inf_replaced(self):
        """Infinite values should be treated as NaN — no inf in output."""
        df = pd.DataFrame({
            "concentration": [0.5, 0.5],
            "response_time_T90": [float("inf"), 12.0],
            "recovery_time_T10": [20.0, 20.0],
        })
        result = summarize_dynamics_metrics(df)
        # mean of [nan, 12.0] = 12.0
        assert math.isfinite(result["per_concentration"]["0.5"]["mean_T90"])


# ---------------------------------------------------------------------------
# summarize_top_comparison
# ---------------------------------------------------------------------------


class TestSummarizeTopComparison:
    def test_empty_returns_empty(self):
        assert summarize_top_comparison({}) == []

    def test_one_mode(self):
        results = {
            "transmittance": {
                "canonical_count": 5,
                "response_stats": {"max_r_squared": 0.98, "max_slope": -0.5, "max_slope_wavelength": 712.0},
                "performance": {"roi_performance": {"lod": 0.1, "loq": 0.3}},
                "metrics_path": "/tmp/m.json",
                "plot_path": "/tmp/p.png",
            }
        }
        summary = summarize_top_comparison(results)
        assert len(summary) == 1
        assert summary[0]["mode"] == "transmittance"
        assert summary[0]["roi_max_r2"] == pytest.approx(0.98)
        assert summary[0]["lod"] == pytest.approx(0.1)

    def test_missing_keys_graceful(self):
        """Partial payload — missing keys should return None, not raise."""
        results = {"absorbance": {}}
        summary = summarize_top_comparison(results)
        assert len(summary) == 1
        assert summary[0]["roi_max_r2"] is None
        assert summary[0]["lod"] is None


# ---------------------------------------------------------------------------
# summarize_quality_control
# ---------------------------------------------------------------------------


def _make_qc_data(
    concs: list = None,
    snr: float = 20.0,
    signal: float = 0.5,
    n_trials: int = 3,
    noise: float = 0.005,
    seed: int = 0,
):
    concs = concs or [0.5, 1.0]
    rng = np.random.default_rng(seed)
    wl = np.linspace(680, 750, 50)
    stable: dict = {}
    nm: dict = {}
    for c in concs:
        stable[c] = {}
        nm[c] = {}
        for t in range(n_trials):
            sig = rng.normal(signal, noise, 50)
            stable[c][f"t{t}"] = pd.DataFrame({"wavelength": wl, "intensity": sig})
            nm[c][f"t{t}"] = {"snr": snr, "rms": noise, "mad": noise, "spectral_entropy": 1.0}
    return stable, nm


class TestSummarizeQualityControl:
    def test_snr_pass_above_threshold(self):
        stable, nm = _make_qc_data(snr=20.0)
        qc = summarize_quality_control(stable, nm, min_snr=10.0)
        assert qc["snr_pass"] is True
        assert qc["overall_pass"] is True

    def test_snr_fail_below_threshold(self):
        stable, nm = _make_qc_data(snr=5.0)
        qc = summarize_quality_control(stable, nm, min_snr=10.0)
        assert qc["snr_pass"] is False
        assert qc["overall_pass"] is False

    def test_rsd_computed_per_concentration(self):
        stable, nm = _make_qc_data()
        qc = summarize_quality_control(stable, nm)
        assert isinstance(qc["rsd_by_concentration"], dict)

    def test_thresholds_stored_in_result(self):
        stable, nm = _make_qc_data()
        qc = summarize_quality_control(stable, nm, min_snr=15.0, max_rsd=3.0)
        assert qc["snr_threshold"] == pytest.approx(15.0)
        assert qc["rsd_threshold_percent"] == pytest.approx(3.0)

    def test_empty_inputs_no_crash(self):
        qc = summarize_quality_control({}, {})
        assert "overall_pass" in qc
        assert qc["overall_pass"] is False  # no data → can't pass

    def test_median_snr_correct(self):
        stable, nm = _make_qc_data(concs=[0.5], snr=30.0, n_trials=2)
        qc = summarize_quality_control(stable, nm)
        assert qc["median_snr"] == pytest.approx(30.0)
