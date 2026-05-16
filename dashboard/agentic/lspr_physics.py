"""
LSPR physics helpers — pure functions, no Streamlit.

Covers:
- Lorentzian FWHM fitting on reference spectra
- FOM (Figure of Merit) and WLS correction helpers
- Simulated frame generation (Gaussian peak + noise)
- Signal preprocessing chain
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import numpy as np

# ── project root ───────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ── signal processing (src/ canonical) ───────────────────────────────────────
from src.preprocessing.baseline import als_baseline
from src.preprocessing.baseline import correct_baseline as baseline_correction
from src.preprocessing.denoising import smooth_spectrum
from src.preprocessing.normalization import normalize_spectrum

_SP_AVAILABLE = True


# ─────────────────────────────────────────────────────────────────────────────
# Simulated acquisition
# ─────────────────────────────────────────────────────────────────────────────


def _simulate_frame(
    wl: np.ndarray,
    concentration_ppm: float = 100.0,
    sim_peak_nm: float = 532.0,
) -> np.ndarray:
    """Produce a realistic simulated frame with a Gaussian peak + noise.

    Parameters
    ----------
    wl:
        Wavelength array (nm).
    concentration_ppm:
        Analyte concentration used to scale peak amplitude.
    sim_peak_nm:
        Centre wavelength of the simulated Gaussian peak (nm).
        Override to match your sensor's expected peak position.
    """
    peak = np.exp(-0.5 * ((wl - sim_peak_nm) / 18) ** 2) * (concentration_ppm / 100) * 4.5
    noise = np.random.normal(0, 0.04, len(wl))
    return peak + noise


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing chain
# ─────────────────────────────────────────────────────────────────────────────


def _preprocess(
    wl: np.ndarray, intensity: np.ndarray, denoise: str, baseline: str, norm: str
) -> np.ndarray:
    """Apply the chosen preprocessing chain."""
    sig = intensity.copy()

    if _SP_AVAILABLE:
        if denoise == "Savitzky-Golay":
            sig = smooth_spectrum(sig, window=11, poly_order=2)
        elif denoise == "Wavelet (DWT-db4)":
            sig = smooth_spectrum(sig, method="wavelet")

        if baseline == "ALS":
            try:
                sig = sig - als_baseline(sig, lam=1e5, p=0.01)
            except Exception:
                sig = baseline_correction(wl, sig, method="als")
        elif baseline == "Polynomial":
            sig = baseline_correction(wl, sig, method="polynomial", order=2)

        if norm == "Min-Max [0,1]":
            sig = normalize_spectrum(sig, method="minmax")
        elif norm == "Z-score":
            sig = normalize_spectrum(sig, method="standard")

    return cast(np.ndarray, sig)


# ─────────────────────────────────────────────────────────────────────────────
# Lorentzian FWHM / FOM helpers
# ─────────────────────────────────────────────────────────────────────────────


def _store_ref_fwhm(ss: dict, wl: np.ndarray, intensity: np.ndarray) -> None:
    """Compute Lorentzian FWHM of the reference spectrum and store in session state.

    Stores ``ss["ap_ref_fwhm_nm"]`` — used downstream for FOM = |S|/FWHM.
    Silently skips on any failure (FWHM is optional for core pipeline).
    """
    try:
        from src.features.spectral_features import fit_lorentzian_peak
        peak_idx = int(np.argmax(intensity))
        peak_wl = float(wl[peak_idx])
        fit = fit_lorentzian_peak(wl, intensity, peak_wl, half_width_nm=30.0)
        if fit is not None:
            _fwhm = float(fit[1])  # fit = (center_nm, fwhm_nm, amplitude, center_std_nm)
            if 1.0 < _fwhm < 200.0:  # sanity bounds for LSPR peak
                ss["ap_ref_fwhm_nm"] = round(_fwhm, 4)
    except Exception:
        pass
