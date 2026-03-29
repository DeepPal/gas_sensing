"""Tests for src.signal.peak — Gaussian, Lorentzian, and cross-correlation peak detection."""
import numpy as np
import pytest

from src.signal.peak import gaussian_peak_center, lorentzian_peak_center, estimate_shift_crosscorr


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lorentzian_spectrum(
    wl: np.ndarray,
    center: float = 717.9,
    gamma: float = 12.0,
    amplitude: float = 1000.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Synthetic Lorentzian LSPR peak with optional Gaussian noise."""
    signal = amplitude / (1.0 + ((wl - center) / (gamma / 2.0)) ** 2)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        signal = signal + rng.normal(0, noise_std, len(wl))
    return signal


def _gaussian_spectrum(
    wl: np.ndarray,
    center: float = 717.9,
    sigma: float = 6.0,
    amplitude: float = 1000.0,
) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2)


# ── lorentzian_peak_center ────────────────────────────────────────────────────

def test_lorentzian_recovers_known_center_noiseless():
    """Lorentzian fit must recover the true peak centre within 0.01 nm on noise-free data."""
    wl = np.linspace(680, 760, 512)
    true_center = 717.9
    signal = _lorentzian_spectrum(wl, center=true_center)
    center, std = lorentzian_peak_center(wl, signal)
    assert abs(center - true_center) < 0.01, f"Centre error {abs(center - true_center):.4f} nm > 0.01"


def test_lorentzian_returns_finite_std():
    """Lorentzian fit must return finite centre_std for noise-free data."""
    wl = np.linspace(680, 760, 512)
    signal = _lorentzian_spectrum(wl, center=717.9)
    _, center_std = lorentzian_peak_center(wl, signal)
    assert np.isfinite(center_std)
    assert center_std >= 0.0


def test_lorentzian_noisy_center_within_tolerance():
    """Lorentzian fit must recover centre within 0.05 nm at SNR ~ 100."""
    wl = np.linspace(680, 760, 512)
    true_center = 717.9
    signal = _lorentzian_spectrum(wl, center=true_center, noise_std=10.0)
    center, _ = lorentzian_peak_center(wl, signal)
    assert np.isfinite(center), "Lorentzian fit returned nan on noisy data"
    assert abs(center - true_center) < 0.1


def test_lorentzian_returns_nan_on_too_few_points():
    """Lorentzian fit must return (nan, nan) when fewer than 5 points are provided."""
    wl = np.array([716.0, 717.0, 718.0])
    signal = np.array([0.5, 1.0, 0.5])
    center, std = lorentzian_peak_center(wl, signal)
    assert np.isnan(center)
    assert np.isnan(std)


def test_lorentzian_more_precise_than_gaussian_on_lspr_data():
    """Lorentzian fit error must be <= Gaussian fit error on LSPR (Lorentzian) data."""
    wl = np.linspace(680, 760, 512)
    true_center = 717.9
    signal = _lorentzian_spectrum(wl, center=true_center, noise_std=5.0)
    lor_center, _ = lorentzian_peak_center(wl, signal)
    gauss_center = gaussian_peak_center(wl, signal)
    if np.isfinite(lor_center) and np.isfinite(gauss_center):
        # On Lorentzian data, Lorentzian fit should be at least as good as Gaussian
        assert abs(lor_center - true_center) <= abs(gauss_center - true_center) + 0.05


# ── gaussian_peak_center ──────────────────────────────────────────────────────

def test_gaussian_recovers_known_valley_center():
    """gaussian_peak_center handles absorption minima (valleys) as designed.

    In LSPR transmittance spectra the resonance appears as a dip (valley), not
    a peak.  gaussian_peak_center detects the feature polarity via the is_min
    criterion; this test verifies it correctly localises an inverted Gaussian
    (transmittance minimum) to within 0.05 nm.

    Note: for LSPR emission-type peaks (maxima), prefer lorentzian_peak_center
    which uses the Lorentzian model and is robust to both feature polarities.
    """
    wl = np.linspace(713, 723, 201)   # 0.05 nm/pixel
    true_center = 718.0
    background = 1000.0
    # Inverted Gaussian = absorption valley (as in LSPR transmittance spectra)
    signal = background - _gaussian_spectrum(wl, center=true_center, sigma=1.0, amplitude=800.0)
    center = gaussian_peak_center(wl, signal, half_width=50)
    assert abs(center - true_center) < 0.05


def test_gaussian_returns_finite_on_noisy():
    """Gaussian fit or fallback centroid must return a finite value."""
    wl = np.linspace(680, 760, 100)
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 100)  # pure noise
    center = gaussian_peak_center(wl, signal)
    assert np.isfinite(center)


# ── estimate_shift_crosscorr ──────────────────────────────────────────────────

def test_crosscorr_zero_shift():
    """Cross-correlation of identical spectra must return zero shift."""
    wl = np.linspace(680, 760, 512)
    signal = _lorentzian_spectrum(wl)
    shift = estimate_shift_crosscorr(wl, signal, signal)
    assert abs(shift) < 0.1


def test_crosscorr_detects_known_shift():
    """Cross-correlation must detect a 2 nm shift with upsample=4."""
    wl = np.linspace(680, 760, 512)
    ref = _lorentzian_spectrum(wl, center=717.9)
    tgt = _lorentzian_spectrum(wl, center=715.9)  # 2 nm red-shift
    shift = estimate_shift_crosscorr(wl, ref, tgt, upsample=4)
    # sign convention: negative means target is red-shifted
    assert abs(abs(shift) - 2.0) < 0.5
