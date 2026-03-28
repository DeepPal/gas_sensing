import math

import numpy as np
import pandas as pd
import pytest

from gas_analysis.core import pipeline as pl
from src.reporting.environment import (
    compute_environment_coefficients,
    compute_environment_summary,
)


def _make_df(wl, signal, T=None, H=None):
    d = {"wavelength": wl.astype(float), "intensity": signal.astype(float)}
    if T is not None:
        d["temperature"] = np.full_like(wl, float(T))
    if H is not None:
        d["humidity"] = np.full_like(wl, float(H))
    return pd.DataFrame(d)


def test_compute_environment_coefficients_estimates_ct_ch_close():
    # Configure reference values
    pl.CONFIG["environment"] = {
        "enabled": True,
        "apply_to_frames": False,
        "apply_to_transmittance": True,
        "reference": {"temperature": 25.0, "humidity": 50.0},
        "coefficients": {"temperature": 0.0, "humidity": 0.0},
        "override": {"temperature": None, "humidity": None},
    }

    rng = np.random.default_rng(0)
    # Synthetic design: concentrations and environment values per concentration
    concs = np.array([0.5, 1.0, 2.0, 3.0, 4.0], dtype=float)
    T_vals = np.array([22.0, 24.0, 26.0, 27.0, 23.0], dtype=float)  # around 25C
    H_vals = np.array([55.0, 50.0, 45.0, 52.0, 48.0], dtype=float)  # around 50%

    # True mapping for the response y (peak wavelengths)
    beta0 = 520.0
    beta_c = 1.50
    cT_true = 0.20
    cH_true = -0.10
    T_ref = 25.0
    H_ref = 50.0

    # y per concentration (mean), add tiny noise
    y = (
        beta0
        + beta_c * concs
        + cT_true * (T_vals - T_ref)
        + cH_true * (H_vals - H_ref)
        + rng.normal(0, 0.02, size=concs.size)
    )

    # Minimal canonical mapping needed by compute_environment_coefficients: stable_by_conc with T/H columns
    wl = np.linspace(500.0, 900.0, 50)
    stable_by_conc = {}
    for ci, c in enumerate(concs):
        # A trial dataframe per concentration with T/H columns; intensity is arbitrary
        signal = np.sin(wl / 50.0) * 0.01 + 1.0
        df = _make_df(wl, signal, T=T_vals[ci], H=H_vals[ci])
        stable_by_conc[float(c)] = {f"trial_{ci}": df}

    calib = {
        "concentrations": concs.tolist(),
        "transformed_concentrations": concs.tolist(),
        "peak_wavelengths": y.tolist(),
    }

    res = pl.compute_environment_coefficients(stable_by_conc, calib)
    assert isinstance(res, dict) and res, "Expected non-empty coefficients result"
    est = res.get("estimated_coefficients", {})
    cT_est = est.get("temperature", None)
    cH_est = est.get("humidity", None)
    assert cT_est is not None and cH_est is not None, "Missing estimated cT/cH"
    # Within tolerance ~0.05 due to noise and small sample size
    assert abs(cT_est - cT_true) < 0.05, f"cT estimate off: got {cT_est}, true {cT_true}"
    assert abs(cH_est - cH_true) < 0.05, f"cH estimate off: got {cH_est}, true {cH_true}"
    # Check improvement
    dr2 = res.get("delta_r2", None)
    assert dr2 is not None and dr2 >= 0.0, f"Expected non-negative ΔR², got {dr2}"


def test_compute_environment_coefficients_handles_missing_env():
    # No env columns available -> expect empty output
    pl.CONFIG["environment"] = {
        "enabled": True,
        "reference": {"temperature": 25.0, "humidity": 50.0},
        "coefficients": {},
        "override": {},
    }

    concs = np.array([0.5, 1.0, 2.0], dtype=float)
    wl = np.linspace(500.0, 900.0, 50)
    stable_by_conc = {}
    for ci, c in enumerate(concs):
        signal = np.cos(wl / 40.0) * 0.01 + 1.0
        df = pd.DataFrame({"wavelength": wl, "intensity": signal})
        stable_by_conc[float(c)] = {f"trial_{ci}": df}

    calib = {
        "concentrations": concs.tolist(),
        "transformed_concentrations": concs.tolist(),
        "peak_wavelengths": (520.0 + 1.0 * concs).tolist(),
    }

    res = pl.compute_environment_coefficients(stable_by_conc, calib)
    assert isinstance(res, dict)
    assert res == {} or (
        res.get("estimated_coefficients", {}) == {} and res.get("offset_count", 0) == 0
    )


# ---------------------------------------------------------------------------
# src.reporting.environment — CONFIG-free tests
# ---------------------------------------------------------------------------


def _make_env_agg(
    concs: list,
    temp: float = 25.0,
    humid: float = 50.0,
    add_env_cols: bool = True,
) -> dict:
    agg: dict = {}
    wl = np.linspace(680, 750, 20)
    for c in concs:
        d: dict = {"wavelength": wl, "transmittance": np.ones(20) * (0.9 - 0.01 * c)}
        if add_env_cols:
            d["temperature"] = np.full(20, temp)
            d["humidity"] = np.full(20, humid)
        agg[c] = {"t0": pd.DataFrame(d)}
    return agg


class TestComputeEnvironmentSummary:
    def test_no_env_columns_zero_offsets(self):
        agg = _make_env_agg([0.5], add_env_cols=False)
        info = compute_environment_summary(agg)
        assert info["offset_count"] == 0
        assert info["temperature_mean"] is None

    def test_reference_defaults(self):
        info = compute_environment_summary({})
        assert info["reference"]["temperature"] == 25.0
        assert info["reference"]["humidity"] == 50.0

    def test_temperature_mean_correct(self):
        agg = _make_env_agg([0.5, 1.0], temp=30.0)
        info = compute_environment_summary(agg)
        assert info["temperature_mean"] == pytest.approx(30.0)

    def test_ct_offset_computed(self):
        agg = _make_env_agg([0.5], temp=30.0)
        # offset = 0.002 * (30 - 25) = 0.01
        info = compute_environment_summary(agg, T_ref=25.0, cT=0.002)
        assert info["offset_mean"] == pytest.approx(0.01)
        assert info["offset_count"] == 1

    def test_both_coefficients(self):
        agg = _make_env_agg([0.5], temp=30.0, humid=60.0)
        # offset = 0.001 * 5 + 0.002 * 10 = 0.025
        info = compute_environment_summary(agg, T_ref=25.0, H_ref=50.0, cT=0.001, cH=0.002)
        assert info["offset_mean"] == pytest.approx(0.025)

    def test_override_temp_used_when_column_absent(self):
        agg = _make_env_agg([0.5], add_env_cols=False)
        info = compute_environment_summary(agg, T_ref=25.0, cT=0.01, override_temp=27.0)
        assert info["offset_mean"] == pytest.approx(0.02)

    def test_env_enabled_stored(self):
        info = compute_environment_summary({}, env_enabled=True)
        assert info["enabled"] is True

    def test_coefficients_none_by_default(self):
        info = compute_environment_summary({})
        assert info["coefficients"]["temperature"] is None
        assert info["coefficients"]["humidity"] is None


class TestComputeEnvironmentCoefficients:
    def test_empty_calib_returns_empty(self):
        assert compute_environment_coefficients({}, {}) == {}

    def test_no_env_data_returns_empty(self):
        agg = _make_env_agg([0.5, 1.0], add_env_cols=False)
        calib = {"concentrations": [0.5, 1.0], "peak_wavelengths": [717.0, 716.5]}
        assert compute_environment_coefficients(agg, calib) == {}

    def test_result_keys_present(self):
        agg = _make_env_agg([0.5, 1.0, 2.0])
        for i, c in enumerate([0.5, 1.0, 2.0]):
            agg[c]["t0"]["temperature"] = np.full(20, 24.0 + i * 2)
        calib = {"concentrations": [0.5, 1.0, 2.0], "peak_wavelengths": [717.5, 717.0, 716.5]}
        result = compute_environment_coefficients(agg, calib)
        for key in ["estimated_coefficients", "r2_conc_only", "r2_full", "delta_r2", "n_points"]:
            assert key in result

    def test_n_points_matches_input(self):
        agg = _make_env_agg([0.5, 1.0, 2.0])
        for i, c in enumerate([0.5, 1.0, 2.0]):
            agg[c]["t0"]["temperature"] = np.full(20, 24.0 + i * 2)
        calib = {"concentrations": [0.5, 1.0, 2.0], "peak_wavelengths": [717.5, 717.0, 716.5]}
        result = compute_environment_coefficients(agg, calib)
        assert result["n_points"] == 3

    def test_intercept_is_finite(self):
        agg = _make_env_agg([0.5, 1.0, 2.0])
        for i, c in enumerate([0.5, 1.0, 2.0]):
            agg[c]["t0"]["temperature"] = np.full(20, 24.0 + i)
        calib = {"concentrations": [0.5, 1.0, 2.0], "peak_wavelengths": [717.5, 717.0, 716.5]}
        result = compute_environment_coefficients(agg, calib)
        assert math.isfinite(result["estimated_coefficients"]["intercept"])

    def test_delta_r2_non_negative_with_covarying_env(self):
        concs = [0.5, 1.0, 1.5, 2.0]
        temps = [23.0, 24.5, 26.0, 27.5]
        peak_wl = [717.5 - 0.1 * c - 0.05 * (T - 25.0) for c, T in zip(concs, temps)]
        wl = np.linspace(680, 750, 20)
        agg = {c: {"t0": pd.DataFrame({"wavelength": wl, "transmittance": np.ones(20),
                                        "temperature": np.full(20, T)})}
               for c, T in zip(concs, temps)}
        calib = {"concentrations": concs, "peak_wavelengths": peak_wl}
        result = compute_environment_coefficients(agg, calib)
        assert result["delta_r2"] >= -1e-10
