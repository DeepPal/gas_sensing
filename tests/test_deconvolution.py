import os
import tempfile
import numpy as np
import pandas as pd

from gas_analysis.advanced.deconvolution_ica import fit_ica_from_canonical, save_ica_outputs
from gas_analysis.advanced.mcr_als import fit_mcrals_from_canonical, save_mcrals_outputs


def _gaussian(wl, center, width):
    return np.exp(-0.5 * ((wl - center) / width) ** 2)


def build_synthetic_canonical(n_samples=12, n_points=240, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(500.0, 900.0, n_points)
    # Two spectral components
    s1 = _gaussian(wl, center=650.0, width=25.0)
    s2 = 0.7 * _gaussian(wl, center=780.0, width=35.0)
    # Normalize shapes to similar scale
    s1 = s1 / (np.linalg.norm(s1) + 1e-12)
    s2 = s2 / (np.linalg.norm(s2) + 1e-12)

    y = np.linspace(0.5, 5.0, n_samples)  # concentration of component-1
    c2 = rng.uniform(0.2, 2.5, size=n_samples)

    canonical = {}
    for i in range(n_samples):
        mix = y[i] * s1 + c2[i] * s2
        mix += rng.normal(0.0, noise, size=mix.shape)
        df = pd.DataFrame({
            'wavelength': wl,
            'intensity': mix.astype(float),
        })
        canonical[float(y[i])] = df
    return canonical


def test_fastica_recovers_component_high_r2():
    canonical = build_synthetic_canonical(n_samples=14, noise=0.005, seed=42)
    cfg = {
        'n_components': 2,
        'max_iter': 1000,
        'tol': 1e-5,
        'random_state': 0,
    }
    res = fit_ica_from_canonical(canonical, cfg)
    assert res is not None, "ICA returned None"
    # Expect reasonably high CV R^2 for best component
    r2cv = res.get('r2_cv', float('nan'))
    assert np.isfinite(r2cv) and r2cv >= 0.85, f"ICA CV R^2 too low: {r2cv}"
    # Shapes
    comps = np.array(res.get('components', []), dtype=float)
    wl = np.array(res.get('wavelengths', []), dtype=float)
    assert comps.ndim == 2 and comps.shape[1] == wl.size


def test_mcrals_recovers_component_high_r2():
    canonical = build_synthetic_canonical(n_samples=14, noise=0.005, seed=7)
    cfg = {
        'n_components': 2,
        'max_iter': 300,
        'tol': 1e-6,
        'random_state': 0,
    }
    res = fit_mcrals_from_canonical(canonical, cfg)
    assert res is not None, "MCR-ALS returned None"
    r2cv = res.get('r2_cv', float('nan'))
    assert np.isfinite(r2cv) and r2cv >= 0.85, f"MCR-ALS CV R^2 too low: {r2cv}"
    comps = np.array(res.get('components', []), dtype=float)
    wl = np.array(res.get('wavelengths', []), dtype=float)
    assert comps.ndim == 2 and comps.shape[1] == wl.size


def test_artifacts_saved_for_ica_and_mcr():
    canonical = build_synthetic_canonical(n_samples=10, noise=0.01, seed=1)
    ica_res = fit_ica_from_canonical(canonical, {'n_components': 2, 'random_state': 0})
    mcr_res = fit_mcrals_from_canonical(canonical, {'n_components': 2, 'random_state': 0})
    assert ica_res is not None and mcr_res is not None

    with tempfile.TemporaryDirectory() as tmp:
        ica_paths = save_ica_outputs(ica_res, tmp)
        mcr_paths = save_mcrals_outputs(mcr_res, tmp)
        # JSON metrics should exist
        assert ica_paths.get('metrics') and os.path.exists(ica_paths['metrics'])
        assert mcr_paths.get('metrics') and os.path.exists(mcr_paths['metrics'])
        # Plots (components) should exist
        assert ica_paths.get('components_plot') and os.path.exists(ica_paths['components_plot'])
        assert mcr_paths.get('components_plot') and os.path.exists(mcr_paths['components_plot'])
