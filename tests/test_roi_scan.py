"""Tests for src.calibration.roi_scan — CONFIG-free ROI discovery."""
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from src.calibration.roi_scan import (
    RoiScanConfig,
    compute_concentration_response,
    stack_trials_for_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stable_by_conc(
    concs: list[float],
    n_trials: int = 2,
    n_wl: int = 60,
    dip_at: int = 30,
    seed: int = 0,
) -> dict[float, dict[str, pd.DataFrame]]:
    """Synthetic dataset with a dip at wl[dip_at] proportional to concentration."""
    rng = np.random.default_rng(seed)
    wl = np.linspace(680.0, 750.0, n_wl)
    result: dict[float, dict[str, pd.DataFrame]] = {}
    for c in concs:
        trials = {}
        for t in range(n_trials):
            # transmittance dips at dip_at — negative correlation with conc
            sig = 0.9 - 0.02 * c * np.exp(-((np.arange(n_wl) - dip_at) ** 2) / 8.0)
            sig += rng.normal(0, 0.001, n_wl)
            trials[f"t{t}"] = pd.DataFrame({"wavelength": wl, "transmittance": sig})
        result[c] = trials
    return result


# ---------------------------------------------------------------------------
# stack_trials_for_response
# ---------------------------------------------------------------------------


class TestStackTrialsForResponse:
    def test_output_shapes(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0], n_trials=2, n_wl=50)
        wl, Y, concs, avg = stack_trials_for_response(agg)
        assert wl.shape == (50,)
        assert Y.shape == (6, 50)
        assert concs.shape == (6,)
        assert len(avg) == 3

    def test_concs_sorted_ascending(self):
        agg = _make_stable_by_conc([2.0, 0.5, 1.0], n_trials=1)
        _, _, concs, _ = stack_trials_for_response(agg)
        assert list(concs) == sorted(concs)

    def test_avg_by_conc_is_mean_of_trials(self):
        agg = _make_stable_by_conc([0.5], n_trials=2, n_wl=10)
        _, Y, _, avg = stack_trials_for_response(agg)
        np.testing.assert_allclose(avg[0.5], (Y[0] + Y[1]) / 2.0, rtol=1e-10)

    def test_grid_mismatch_interpolated(self):
        wl1 = np.linspace(680, 750, 50)
        wl2 = np.linspace(681, 749, 48)
        df1 = pd.DataFrame({"wavelength": wl1, "transmittance": np.ones(50) * 0.8})
        df2 = pd.DataFrame({"wavelength": wl2, "transmittance": np.ones(48) * 0.7})
        agg = {0.5: {"t0": df1}, 1.0: {"t0": df2}}
        _, Y, _, _ = stack_trials_for_response(agg)
        assert Y.shape[1] == 50  # aligned to first grid

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No spectra"):
            stack_trials_for_response({})


# ---------------------------------------------------------------------------
# RoiScanConfig
# ---------------------------------------------------------------------------


class TestRoiScanConfig:
    def test_defaults(self):
        cfg = RoiScanConfig()
        assert cfg.selection_metric == "r2"
        assert cfg.min_r2 == 0.0
        assert cfg.band_half_width is None
        assert cfg.global_std == 0.0

    def test_fields_set_correctly(self):
        cfg = RoiScanConfig(selection_metric="hybrid", r2_weight=0.7, min_r2=0.5)
        assert cfg.selection_metric == "hybrid"
        assert cfg.r2_weight == 0.7
        assert cfg.min_r2 == 0.5

    def test_none_cfg_uses_defaults_inside_function(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0], seed=1)
        r1, _ = compute_concentration_response(agg, cfg=None)
        r2, _ = compute_concentration_response(agg, cfg=RoiScanConfig())
        assert r1["max_slope_wavelength"] == r2["max_slope_wavelength"]


# ---------------------------------------------------------------------------
# compute_concentration_response
# ---------------------------------------------------------------------------


class TestComputeConcentrationResponse:
    def test_basic_smoke(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, avg_by_conc = compute_concentration_response(agg)
        assert "max_slope_wavelength" in response
        assert "roi_start_wavelength" in response
        assert "roi_end_wavelength" in response
        assert len(avg_by_conc) == 3

    def test_roi_within_spectrum_range(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, _ = compute_concentration_response(agg)
        wl_all = np.asarray(response["wavelengths"], dtype=float)
        roi_start = cast(float, response["roi_start_wavelength"])
        roi_end = cast(float, response["roi_end_wavelength"])
        assert roi_start >= wl_all[0]
        assert roi_end <= wl_all[-1]
        assert roi_start <= roi_end

    def test_decreasing_trend_gives_negative_slope(self):
        # Dip signal → negative correlation with concentration
        agg = _make_stable_by_conc([0.5, 1.0, 2.0], dip_at=30)
        cfg = RoiScanConfig(expected_trend="decreasing")
        response, _ = compute_concentration_response(agg, cfg)
        assert cast(float, response["max_slope"]) < 0

    def test_wavelength_range_filter_applied(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0], n_wl=60)
        cfg = RoiScanConfig(min_wavelength=700.0, max_wavelength=740.0)
        response, _ = compute_concentration_response(agg, cfg)
        for wl in np.asarray(response["wavelengths"], dtype=float):
            assert 700.0 <= wl <= 740.0

    def test_wavelength_filter_removes_all_raises(self):
        agg = _make_stable_by_conc([0.5, 1.0])
        cfg = RoiScanConfig(min_wavelength=900.0, max_wavelength=950.0)
        with pytest.raises(ValueError, match="No wavelengths"):
            compute_concentration_response(agg, cfg)

    def test_band_half_width_respected(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0], n_wl=60)
        cfg = RoiScanConfig(band_half_width=3)
        response, _ = compute_concentration_response(agg, cfg)
        center = cast(float, response["max_slope_wavelength"])
        roi_start = cast(float, response["roi_start_wavelength"])
        roi_end = cast(float, response["roi_end_wavelength"])
        assert roi_start <= center <= roi_end

    def test_top_k_candidates_count(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, _ = compute_concentration_response(agg, top_k_candidates=3)
        candidates = cast(list[Any], response["candidates"])
        assert len(candidates) <= 3
        assert len(candidates) > 0

    def test_zero_candidates_by_default(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, _ = compute_concentration_response(agg)
        assert response["candidates"] == []

    def test_validation_within_tolerance(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        r0, _ = compute_concentration_response(agg)
        center = cast(float, r0["max_slope_wavelength"])
        cfg = RoiScanConfig(expected_center=center, center_tolerance=10.0)
        response, _ = compute_concentration_response(agg, cfg)
        validation = cast(dict[str, object], response["validation"])
        assert validation["within_tolerance"] is True

    def test_validation_outside_tolerance(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        cfg = RoiScanConfig(expected_center=400.0, center_tolerance=1.0)
        response, _ = compute_concentration_response(agg, cfg)
        validation = cast(dict[str, object], response["validation"])
        assert validation["within_tolerance"] is False

    def test_poly_r_squared_none_when_disabled(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, _ = compute_concentration_response(agg, RoiScanConfig(alt_models_enabled=False))
        assert response["poly_r_squared"] is None

    def test_r_squared_in_unit_interval(self):
        agg = _make_stable_by_conc([0.5, 1.0, 2.0])
        response, _ = compute_concentration_response(agg)
        r2 = cast(float, response["max_r_squared"])
        assert 0.0 <= r2 <= 1.0

    def test_avg_by_conc_correct_keys(self):
        concs = [0.5, 1.0, 2.0]
        agg = _make_stable_by_conc(concs)
        _, avg_by_conc = compute_concentration_response(agg)
        assert set(avg_by_conc.keys()) == set(concs)
