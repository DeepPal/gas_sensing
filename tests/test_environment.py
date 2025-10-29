import numpy as np
import pandas as pd

from gas_analysis.core import pipeline as pl


def _make_df(wl, signal, T=None, H=None):
    d = {'wavelength': wl.astype(float), 'intensity': signal.astype(float)}
    if T is not None:
        d['temperature'] = np.full_like(wl, float(T))
    if H is not None:
        d['humidity'] = np.full_like(wl, float(H))
    return pd.DataFrame(d)


def test_compute_environment_coefficients_estimates_ct_ch_close():
    # Configure reference values
    pl.CONFIG['environment'] = {
        'enabled': True,
        'apply_to_frames': False,
        'apply_to_transmittance': True,
        'reference': {'temperature': 25.0, 'humidity': 50.0},
        'coefficients': {'temperature': 0.0, 'humidity': 0.0},
        'override': {'temperature': None, 'humidity': None},
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
    y = beta0 + beta_c * concs + cT_true * (T_vals - T_ref) + cH_true * (H_vals - H_ref) + rng.normal(0, 0.02, size=concs.size)

    # Minimal canonical mapping needed by compute_environment_coefficients: stable_by_conc with T/H columns
    wl = np.linspace(500.0, 900.0, 50)
    stable_by_conc = {}
    for ci, c in enumerate(concs):
        # A trial dataframe per concentration with T/H columns; intensity is arbitrary
        signal = np.sin(wl / 50.0) * 0.01 + 1.0
        df = _make_df(wl, signal, T=T_vals[ci], H=H_vals[ci])
        stable_by_conc[float(c)] = {f"trial_{ci}": df}

    calib = {
        'concentrations': concs.tolist(),
        'transformed_concentrations': concs.tolist(),
        'peak_wavelengths': y.tolist(),
    }

    res = pl.compute_environment_coefficients(stable_by_conc, calib)
    assert isinstance(res, dict) and res, "Expected non-empty coefficients result"
    est = res.get('estimated_coefficients', {})
    cT_est = est.get('temperature', None)
    cH_est = est.get('humidity', None)
    assert cT_est is not None and cH_est is not None, "Missing estimated cT/cH"
    # Within tolerance ~0.05 due to noise and small sample size
    assert abs(cT_est - cT_true) < 0.05, f"cT estimate off: got {cT_est}, true {cT_true}"
    assert abs(cH_est - cH_true) < 0.05, f"cH estimate off: got {cH_est}, true {cH_true}"
    # Check improvement
    dr2 = res.get('delta_r2', None)
    assert dr2 is not None and dr2 >= 0.0, f"Expected non-negative ΔR², got {dr2}"


def test_compute_environment_coefficients_handles_missing_env():
    # No env columns available -> expect empty output
    pl.CONFIG['environment'] = {
        'enabled': True,
        'reference': {'temperature': 25.0, 'humidity': 50.0},
        'coefficients': {},
        'override': {},
    }

    concs = np.array([0.5, 1.0, 2.0], dtype=float)
    wl = np.linspace(500.0, 900.0, 50)
    stable_by_conc = {}
    for ci, c in enumerate(concs):
        signal = np.cos(wl / 40.0) * 0.01 + 1.0
        df = pd.DataFrame({'wavelength': wl, 'intensity': signal})
        stable_by_conc[float(c)] = {f"trial_{ci}": df}

    calib = {
        'concentrations': concs.tolist(),
        'transformed_concentrations': concs.tolist(),
        'peak_wavelengths': (520.0 + 1.0 * concs).tolist(),
    }

    res = pl.compute_environment_coefficients(stable_by_conc, calib)
    assert isinstance(res, dict)
    assert res == {} or (res.get('estimated_coefficients', {}) == {} and res.get('offset_count', 0) == 0)
