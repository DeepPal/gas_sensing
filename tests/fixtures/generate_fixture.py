"""
Generate the pinned LSPR calibration fixture for science regression tests.

Run once when establishing or resetting baselines:
    python tests/fixtures/generate_fixture.py

Outputs:
    tests/fixtures/lspr_calibration_fixture.npz
    tests/science_regression/baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _langmuir_shift(c: np.ndarray, S: float = 2.5, K: float = 0.08) -> np.ndarray:
    """Δλ = -S·c / (1 + K·c)  — Langmuir adsorption LSPR response."""
    return -(S * c) / (1.0 + K * c)


def main() -> None:
    rng = np.random.default_rng(42)

    concentrations = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])  # ppm
    delta_lambda_true = _langmuir_shift(concentrations)
    sigma_noise = 0.012  # nm — realistic shot noise
    delta_lambda_measured = delta_lambda_true + rng.normal(0, sigma_noise, len(concentrations))

    wavelengths = np.linspace(500, 900, 3648)
    ref_peak_wl = 717.9   # nm
    fwhm_nm = 12.5        # nm

    spectra: list[np.ndarray] = []
    for dl in delta_lambda_measured:
        wl_peak = ref_peak_wl + dl
        lorentzian = 1.0 / (1.0 + ((wavelengths - wl_peak) / (fwhm_nm / 2.0)) ** 2)
        spectra.append(lorentzian + rng.normal(0, 0.003, len(wavelengths)))
    reference_spectrum = 1.0 / (1.0 + ((wavelengths - ref_peak_wl) / (fwhm_nm / 2.0)) ** 2)

    fixture_path = ROOT / "tests" / "fixtures" / "lspr_calibration_fixture.npz"
    np.savez(
        fixture_path,
        concentrations=concentrations,
        delta_lambda_true=delta_lambda_true,
        delta_lambda_measured=delta_lambda_measured,
        wavelengths=wavelengths,
        spectra=np.array(spectra),
        reference_spectrum=reference_spectrum,
        ref_peak_wl=np.array([ref_peak_wl]),
        fwhm_nm=np.array([fwhm_nm]),
    )
    print(f"Fixture written: {fixture_path}")

    # Run current src/ pipeline to record baselines
    from src.calibration.gpr import GPRCalibration
    from src.scientific.lod import calculate_lod_3sigma, calculate_sensitivity

    gpr = GPRCalibration()
    X = delta_lambda_measured.reshape(-1, 1)
    gpr.fit(X, concentrations)
    mean_at_1nm, std_at_1nm = gpr.predict(np.array([[-1.0]]))

    # calculate_sensitivity(concentrations, responses) — concentrations first
    sensitivity_slope, intercept, r_squared, slope_se = calculate_sensitivity(
        concentrations, delta_lambda_measured
    )
    # calculate_lod_3sigma(noise_std, sensitivity_slope) — correct param names
    lod = float(calculate_lod_3sigma(noise_std=sigma_noise, sensitivity_slope=sensitivity_slope))

    baselines = {
        "lod_ppm": round(lod, 8),
        "sensitivity_nm_per_ppm": round(float(sensitivity_slope), 8),
        "gpr_std_at_neg1nm": round(float(std_at_1nm[0]), 8),
        "gpr_mean_at_neg1nm": round(float(mean_at_1nm[0]), 8),
        "delta_lambda_at_1ppm_true": round(float(delta_lambda_true[2]), 8),
        "kernel": "Matern-5/2",
        "generated_date": "2026-05-14",
    }

    baselines_path = ROOT / "tests" / "science_regression" / "baselines.json"
    baselines_path.parent.mkdir(parents=True, exist_ok=True)
    baselines_path.write_text(json.dumps(baselines, indent=2))
    print(f"Baselines written: {baselines_path}")
    print("\n=== BASELINES ===")
    for k, v in baselines.items():
        print(f"  {k}: {v}")
    print("\nCommit both files: tests/fixtures/lspr_calibration_fixture.npz and tests/science_regression/baselines.json")


if __name__ == "__main__":
    main()
