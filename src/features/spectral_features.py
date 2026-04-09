"""
src.features.spectral_features
================================
Spectrometer-based sensor feature extraction: peak detection, wavelength
shift (Δλ), and the 4-component feature vector used as model input.

Sensor-agnostic: the peak-detection and feature-extraction logic works for
any spectrometer sensor whose response is a wavelength shift or intensity
change — LSPR, fluorescence, absorbance, Raman, or any other modality.

Physics note: the primary signal is Δλ = λ_gas − λ_reference (nm).
- **Negative Δλ** (blue-shift): analyte adsorption shortens the effective
  optical path or decreases the local refractive index.  Sensitivity slope < 0.
- **Positive Δλ** (red-shift): analyte increases the local refractive index.
  Sensitivity slope > 0.
``LSPR_SEARCH_MIN_NM`` and ``LSPR_SEARCH_MAX_NM`` are full-spectrum
defaults.  Narrow them at the call site (or via ``spectraagent.toml``)
for sensors that respond in a known sub-band.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search window defaults — sensor-agnostic, covers the full visible/NIR range.
# These are NOT sensor-specific constants. The actual peak position is
# discovered at runtime from the reference spectrum captured by the user.
# Narrow these in spectraagent.toml ([physics] search_min_nm / search_max_nm)
# if your sensor only has features in a sub-band.
# ---------------------------------------------------------------------------

LSPR_SEARCH_MIN_NM: float = 400.0   # full visible spectrum start (nm)
LSPR_SEARCH_MAX_NM: float = 900.0   # full visible/NIR spectrum end (nm)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LSPRFeatures:
    """LSPR feature vector for one spectrum relative to a reference.

    Seven physically orthogonal features extracted from the Lorentzian fit
    and differential spectrum — each encoding a different physical interaction
    mechanism between the analyte and the sensor surface.

    Feature orthogonality rationale
    --------------------------------
    Δλ    : refractive index of the adsorbed molecular layer (polarizability)
    ΔFWHM : plasmon dephasing / scattering damping (molecular size, mass)
    ΔA    : oscillator coupling strength (surface coverage density)
    ΔI_area: integrated extinction change (combined RI + scattering)
    σ_Δλ : Lorentzian fit quality (measurement uncertainty)

    Because these arise from mechanistically independent physical effects,
    different analytes produce distinct signatures in the full feature space
    even when Δλ alone cannot discriminate them.  This enables multi-gas
    sensing from a single sensor without a sensor array.
    """

    delta_lambda: float | None = None
    """Peak wavelength shift Δλ = λ_gas − λ_reference (nm). Primary signal."""

    delta_fwhm_nm: float | None = None
    """Change in Lorentzian linewidth: FWHM_gas − FWHM_ref (nm).
    Encodes plasmon dephasing rate change — sensitive to molecular size and
    surface scattering.  Orthogonal to Δλ for multi-gas discrimination."""

    delta_amplitude: float | None = None
    """Change in Lorentzian amplitude: A_gas − A_ref (a.u.).
    Encodes oscillator coupling / surface coverage density change."""

    delta_intensity_peak: float | None = None
    """Change in peak intensity: I_gas(λ_peak) − I_ref(λ_ref_peak)."""

    delta_intensity_area: float | None = None
    """Change in integrated area under the LSPR band (trapz difference)."""

    delta_intensity_std: float | None = None
    """Standard deviation of (I_gas − I_ref) within the ROI window."""

    peak_wavelength: float | None = None
    """Absolute peak wavelength of the gas spectrum (nm)."""

    snr: float | None = None
    """Signal-to-noise ratio of the gas spectrum within the ROI."""

    roi_center: float | None = None
    """Center of the ROI window used for feature extraction (nm)."""

    roi_width: float | None = None
    """Width of the ROI window used for feature extraction (nm)."""

    delta_lambda_std: float | None = None
    """1-σ uncertainty on Δλ from Lorentzian fit covariance (nm).
    Propagated from reference + gas peak fit uncertainties in quadrature:
    σ_Δλ = √(σ_ref² + σ_gas²).  ``None`` if Lorentzian fit was not used."""

    peak_fwhm_nm: float | None = None
    """Absolute FWHM of the gas spectrum Lorentzian peak (nm)."""

    peak_asymmetry: float | None = None
    """Asymmetry factor of the gas spectrum peak = left_half_width / right_half_width.
    1.0 = perfectly symmetric (ideal Lorentzian).
    >1.0 = left-heavy (blue-side broader): typical of physisorption where the
           short-wavelength scattering background is enhanced.
    <1.0 = right-heavy (red-side broader): typical of chemisorption and
           covalent surface functionalization extending the red tail.
    Computed from the half-maximum crossing points; ``None`` if insufficient
    points are found on one side of the peak."""

    delta_asymmetry: float | None = None
    """Change in peak asymmetry relative to reference: asymmetry_gas − asymmetry_ref.
    The 6th orthogonal feature in the feature vector.

    Physical interpretation
    -----------------------
    A positive Δasymmetry means the gas exposure has broadened the blue side
    more than the red side — consistent with dipolar near-field coupling where
    the adsorbed molecular layer preferentially damps the higher-energy (blue)
    side of the plasmon resonance.

    A negative Δasymmetry means the red tail has grown — consistent with
    chemisorption or formation of a new resonance mode at longer wavelengths
    (e.g., charge-transfer plasmon, or dimerization of surface complexes).

    This feature is orthogonal to Δλ (shift), ΔFWHM (total broadening), and
    ΔA (amplitude) because it captures the *direction* of asymmetric broadening
    rather than its magnitude.  Different analytes binding through different
    mechanisms produce distinct Δasymmetry signatures, enabling single-sensor
    multi-gas discrimination beyond what the Δλ channel alone provides."""

    @property
    def feature_vector(self) -> list[float]:
        """Return the 6-element physically orthogonal feature vector.

        ``[Δλ, ΔFWHM, ΔA, ΔI_area, ΔI_std, Δasymmetry]``

        - Δλ          — refractive index channel (RI sensing)
        - ΔFWHM       — plasmon damping / molecular size channel
        - ΔA          — surface coverage density (oscillator coupling)
        - ΔI_area     — integrated extinction change
        - ΔI_std      — spectral noise / binding heterogeneity
        - Δasymmetry  — binding mechanism directionality (physi- vs. chemisorption)

        Missing values are replaced with 0.0 for model compatibility.
        The extended orthogonal basis enables single-sensor multi-gas
        discrimination: different analytes produce distinct signatures in
        this 6D space even when Δλ alone cannot separate them.
        """
        def _f(v: float | None) -> float:
            return float(v) if v is not None else 0.0

        return [
            _f(self.delta_lambda),
            _f(self.delta_fwhm_nm),
            _f(self.delta_amplitude),
            _f(self.delta_intensity_area),
            _f(self.delta_intensity_std),
            _f(self.delta_asymmetry),
        ]

    @property
    def feature_vector_legacy(self) -> list[float]:
        """Legacy 4-element vector [Δλ, ΔI_peak, ΔI_area, ΔI_std] for backward compatibility."""
        def _f(v: float | None) -> float:
            return float(v) if v is not None else 0.0

        return [
            _f(self.delta_lambda),
            _f(self.delta_intensity_peak),
            _f(self.delta_intensity_area),
            _f(self.delta_intensity_std),
        ]


# ---------------------------------------------------------------------------
# Kinetic features
# ---------------------------------------------------------------------------


@dataclass
class KineticFeatures:
    """Binding kinetics extracted from the Δλ transient response curve.

    Computed by fitting ``Δλ(t) = ΔλEq · (1 − exp(−t/τ))`` to the rising
    portion of a gas-exposure transient.  This is the standard 1:1 Langmuir
    pseudo-first-order association model used in SPR/LSPR literature.

    Attributes
    ----------
    tau_63_s : time constant τ (s) — time to 63.2% of equilibrium response.
        τ = 1/k_obs where k_obs = k_on·[A] + k_off.
    tau_95_s : time to 95% equilibrium = −τ·ln(0.05) ≈ 3τ (s).
    k_on_per_s : pseudo-first-order association rate constant k_obs (s⁻¹).
        Note: this is k_obs = k_on·[A] + k_off, not the true bimolecular k_on,
        unless [A] and k_off are separately measured.
    delta_lambda_eq_nm : fitted equilibrium shift ΔλEq (nm).
    fit_r2 : goodness of fit R² of the exponential model (0–1).
    onset_idx : index in the input time series where gas introduction was
        detected.  ``None`` if onset detection failed.
    """

    tau_63_s: float | None = None
    tau_95_s: float | None = None
    k_on_per_s: float | None = None
    delta_lambda_eq_nm: float | None = None
    fit_r2: float | None = None
    onset_idx: int | None = None


def estimate_response_kinetics(
    delta_lambda_series: list[float] | np.ndarray,
    timestamps_s: list[float] | np.ndarray,
    *,
    baseline_frac: float = 0.2,
    onset_sigma_threshold: float = 3.0,
    min_fit_points: int = 8,
) -> KineticFeatures:
    """Estimate binding kinetics from a Δλ time series.

    Fits the 1:1 Langmuir pseudo-first-order association model:

        Δλ(t) = ΔλEq · (1 − exp(−k_obs · (t − t_onset)))

    Parameters
    ----------
    delta_lambda_series : Δλ values (nm) ordered by acquisition time.
    timestamps_s        : Corresponding elapsed time (s) from session start.
    baseline_frac       : Fraction of early points to use as baseline
                          for onset detection (default 0.20 = first 20%).
    onset_sigma_threshold : How many baseline σ above mean to call as onset
                          (default 3.0).
    min_fit_points      : Minimum points after onset required to attempt a fit.

    Returns
    -------
    KineticFeatures
        All fields are ``None`` if onset detection or fitting fails.

    Notes
    -----
    The onset detection uses a consecutive-threshold rule: onset is declared
    at the first index where Δλ exceeds ``baseline_mean + threshold·baseline_σ``
    for at least ``ceil(0.05·N)`` consecutive samples.  This is robust to
    slow baseline drift and isolated spikes.

    Negative Δλ (blue-shift sensors) are sign-flipped internally so the
    algorithm always works on a rising signal, then the sign is restored.
    """
    dl = np.asarray(delta_lambda_series, dtype=float)
    ts = np.asarray(timestamps_s, dtype=float)

    n = len(dl)
    if n < max(min_fit_points * 2, 10) or len(ts) != n:
        return KineticFeatures()

    # Handle blue-shift sensors: work on |Δλ| with sign restoration later
    sign = 1.0 if float(np.median(dl[n // 2 :])) >= 0 else -1.0
    dl_abs = dl * sign

    # Baseline: first baseline_frac of the series
    n_baseline = max(3, int(n * baseline_frac))
    baseline = dl_abs[:n_baseline]
    bl_mean = float(np.mean(baseline))
    bl_std = float(np.std(baseline))
    if bl_std < 1e-6:
        bl_std = 1e-6  # prevent division by zero for perfectly flat baseline

    threshold = bl_mean + onset_sigma_threshold * bl_std

    # Consecutive-threshold onset detection
    min_run = max(2, int(np.ceil(0.05 * n)))
    onset_idx: int | None = None
    run = 0
    for i in range(n_baseline, n):
        if dl_abs[i] > threshold:
            run += 1
            if run >= min_run:
                onset_idx = i - run + 1
                break
        else:
            run = 0

    if onset_idx is None or (n - onset_idx) < min_fit_points:
        return KineticFeatures(onset_idx=onset_idx)

    # Fit exponential to the transient from onset onwards.
    # Subtract the signal value at onset so the model always starts from 0 —
    # this handles late onset detection (onset_idx after true onset) correctly.
    t_fit = ts[onset_idx:] - ts[onset_idx]
    y0 = float(dl_abs[onset_idx])
    y_fit = dl_abs[onset_idx:] - y0

    # Initial guesses: ΔλEq ≈ remaining amplitude, τ ≈ t_fit midpoint
    a0 = float(np.max(y_fit))
    tau0 = float(t_fit[len(t_fit) // 2]) if len(t_fit) > 1 else 10.0
    if a0 <= 0:
        a0 = 0.1
    if tau0 <= 0:
        tau0 = 1.0

    def _model(t: np.ndarray, a: float, tau: float) -> np.ndarray:
        return a * (1.0 - np.exp(-t / tau))

    try:
        popt, _ = curve_fit(
            _model,
            t_fit,
            y_fit,
            p0=[a0, tau0],
            bounds=([0, 0.01], [a0 * 10, t_fit[-1] * 10 + 1.0]),
            maxfev=2000,
        )
        tau_fit = float(popt[1])
        # Total equilibrium shift = initial offset at onset + remaining fitted amplitude
        a_fit = (y0 + float(popt[0])) * sign  # restore sign

        # R² of fit
        y_pred = _model(t_fit, popt[0], popt[1])
        ss_res = float(np.sum((y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        return KineticFeatures(
            tau_63_s=tau_fit,
            tau_95_s=float(-tau_fit * np.log(0.05)),  # = ~2.996·τ
            k_on_per_s=1.0 / tau_fit,
            delta_lambda_eq_nm=a_fit,
            fit_r2=float(np.clip(r2, 0.0, 1.0)),
            onset_idx=onset_idx,
        )
    except Exception as exc:
        log.debug("Kinetics fit failed: %s", exc)
        return KineticFeatures(onset_idx=onset_idx)


# ---------------------------------------------------------------------------
# Reference cache
# ---------------------------------------------------------------------------


@dataclass
class LSPRReference:
    """Pre-computed Lorentzian fit of a reference (baseline) spectrum.

    Call :func:`compute_lspr_reference` **once** when the reference spectrum
    is loaded, then pass the result to :func:`extract_lspr_features` every
    frame.  This avoids re-fitting the (unchanging) reference at acquisition
    rate (~20 Hz), which accounts for roughly half the per-frame compute budget.

    Attributes
    ----------
    wavelengths, intensities:
        The reference spectral arrays — kept for xcorr fallback.
    peak_wl:
        Detected reference peak wavelength (nm), or ``None`` if detection failed.
    fit:
        Lorentzian fit result ``(center_nm, fwhm_nm, amplitude, center_std_nm)``
        from :func:`fit_lorentzian_peak`, or ``None`` if fitting failed.
    """

    wavelengths: np.ndarray
    intensities: np.ndarray
    peak_wl: float | None
    fit: tuple[float, float, float, float] | None


# ---------------------------------------------------------------------------
# Peak Detection
# ---------------------------------------------------------------------------


def fit_lorentzian_peak(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    peak_wl_init: float,
    half_width_nm: float = 10.0,
) -> tuple[float, float, float, float] | None:
    """Fit a Lorentzian profile to an LSPR peak for sub-pixel wavelength precision.

    The Lorentzian (Cauchy) profile is the physically correct model for LSPR
    absorption peaks: *I(λ) = A / [1 + ((λ − λ₀) / (Γ/2))²]*, driven by the
    Lorentzian frequency response of plasmonic oscillators.

    Provides ~0.01 nm precision vs. ~0.5 nm for argmax — essential for
    detecting small Δλ shifts at low concentrations.

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data.
    peak_wl_init:
        Initial guess for peak center (nm), e.g. from :func:`detect_lspr_peak`.
    half_width_nm:
        Window half-width around ``peak_wl_init`` used for fitting (nm).

    Returns
    -------
    tuple ``(center_nm, fwhm_nm, amplitude, center_std_nm)`` or ``None``
        - *center_nm*: Fitted peak wavelength (nm)
        - *fwhm_nm*: Full-width at half-maximum (nm) — sensor selectivity proxy
        - *amplitude*: Peak amplitude (same units as intensities)
        - *center_std_nm*: 1-σ uncertainty on center from covariance matrix (nm)

        Returns ``None`` if the fit fails or converges to unphysical values.
    """
    lo = peak_wl_init - half_width_nm
    hi = peak_wl_init + half_width_nm
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 5:
        return None

    wl = wavelengths[mask].astype(float)
    yi = intensities[mask].astype(float)

    def _lorentzian(
        x: np.ndarray, center: float, gamma: float, amp: float, offset: float
    ) -> np.ndarray:
        return np.asarray(amp / (1.0 + ((x - center) / (gamma / 2.0)) ** 2) + offset)

    p0 = [peak_wl_init, half_width_nm, float(yi.max() - yi.min()), float(yi.min())]
    bounds = (
        [wl[0], 0.5, 0.0, -np.inf],
        [wl[-1], half_width_nm * 2, float(yi.max() * 10), np.inf],
    )

    try:
        # Shot-noise weighting: σ_i = √max(I_i, 1).
        # Poisson photon statistics → variance ∝ signal.  Weighting by 1/σ_i
        # de-emphasises the bright peak apex (high shot noise) and gives more
        # influence to the stable wing regions, which carry the FWHM information.
        # This is the statistically optimal weight for photon-counting detectors.
        sigma_weights = np.sqrt(np.maximum(yi - yi.min() + 1.0, 1.0))
        popt, pcov = curve_fit(
            _lorentzian, wl, yi, p0=p0, bounds=bounds, maxfev=2000,
            sigma=sigma_weights, absolute_sigma=False,
        )
        center, gamma, amp, _ = popt
        perr = np.sqrt(np.diag(pcov))
        center_std = float(perr[0])

        # Sanity checks: center must be within window; FWHM must be positive
        if not (lo <= center <= hi) or gamma <= 0 or amp <= 0:
            return None

        return float(center), float(abs(gamma)), float(amp), center_std
    except Exception as exc:
        log.debug("Lorentzian fit failed (wl range %.1f–%.1f nm): %s", lo, hi, exc)
        return None


def detect_lspr_peak(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    search_min: float = LSPR_SEARCH_MIN_NM,
    search_max: float = LSPR_SEARCH_MAX_NM,
    prominence: float = 0.01,
) -> float | None:
    """Detect the LSPR absorption peak wavelength within a search window.

    For LSPR in absorbance mode the peak is a maximum.  In transmittance mode
    it is a minimum — pass ``-intensities`` in that case.

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data arrays.
    search_min, search_max:
        Wavelength window to search for the peak (nm).
    prominence:
        Minimum peak prominence (fraction of intensity range).

    Returns
    -------
    float | None
        Peak wavelength in nm, or ``None`` if no peak found.
    """
    # Mask to search window
    mask = (wavelengths >= search_min) & (wavelengths <= search_max)
    if mask.sum() < 5:
        log.debug("LSPR search window too narrow: only %d points.", mask.sum())
        return None

    wl_roi = wavelengths[mask]
    int_roi = intensities[mask]

    # Absolute prominence threshold
    int_range: float = float(int_roi.max() - int_roi.min())
    prom_abs = prominence * int_range if int_range > 0 else 0.0

    peaks, props = find_peaks(int_roi, prominence=prom_abs)
    if len(peaks) == 0:
        # Fallback: coarse argmax, then Lorentzian-refine
        idx = int(np.argmax(int_roi))
        coarse_wl = float(wl_roi[idx])
    else:
        # Most prominent peak → Lorentzian refinement
        best = peaks[int(np.argmax(props["prominences"]))]
        coarse_wl = float(wl_roi[best])

    # Attempt Lorentzian fit for sub-pixel precision
    fit = fit_lorentzian_peak(wavelengths, intensities, coarse_wl)
    if fit is not None:
        center_nm, fwhm_nm, _amp, center_std = fit
        log.debug(
            "Lorentzian fit: λ₀=%.4f nm, FWHM=%.3f nm, σ=%.4f nm",
            center_nm,
            fwhm_nm,
            center_std,
        )
        return center_nm

    return coarse_wl


def detect_all_peaks(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    search_min: float | None = None,
    search_max: float | None = None,
    min_prominence_fraction: float = 0.05,
    max_peaks: int = 10,
) -> list[float]:
    """Detect ALL significant spectral peaks within a search window.

    Supports multi-peak sensor configurations where a single sensor produces
    multiple resonance features (e.g., multi-mode photonic sensors, multiple
    nanoparticle populations, or multi-analyte cross-interference signatures).

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data arrays.
    search_min, search_max:
        Wavelength window (nm). Defaults to full array range.
    min_prominence_fraction:
        A peak must rise at least this fraction of the intensity range above
        its surrounding baseline to be counted (default 5%).
    max_peaks:
        Maximum number of peaks to return.

    Returns
    -------
    list[float]
        Peak wavelengths (nm) sorted by wavelength position (ascending).
        At least one value is always returned (argmax fallback).
    """
    smin = search_min if search_min is not None else float(wavelengths[0])
    smax = search_max if search_max is not None else float(wavelengths[-1])

    mask = (wavelengths >= smin) & (wavelengths <= smax)
    if mask.sum() < 5:
        log.debug("detect_all_peaks: search window too narrow (%d points).", mask.sum())
        return []

    wl_roi = wavelengths[mask]
    int_roi = intensities[mask]
    int_range = float(int_roi.max() - int_roi.min())
    prom_abs = min_prominence_fraction * int_range if int_range > 0 else 0.0

    peaks_idx, props = find_peaks(int_roi, prominence=prom_abs)

    if len(peaks_idx) == 0:
        # Fallback: single argmax peak
        idx = int(np.argmax(int_roi))
        coarse_wl = float(wl_roi[idx])
        fit = fit_lorentzian_peak(wavelengths, intensities, coarse_wl)
        return [fit[0] if fit is not None else coarse_wl]

    # Sort by prominence (highest first), take top max_peaks
    order = np.argsort(-props["prominences"])[:max_peaks]
    result: list[float] = []
    for i in order:
        coarse_wl = float(wl_roi[peaks_idx[i]])
        fit = fit_lorentzian_peak(wavelengths, intensities, coarse_wl)
        result.append(fit[0] if fit is not None else coarse_wl)

    return sorted(result)  # ascending wavelength order


def refine_peak_centroid(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    peak_wl: float,
    half_width_nm: float = 3.0,
) -> float:
    """Refine a peak location using intensity-weighted centroid.

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data.
    peak_wl:
        Initial peak estimate in nm.
    half_width_nm:
        Half-width of the centroid window (nm).

    Returns
    -------
    float
        Refined peak wavelength (nm).  Returns ``peak_wl`` if refinement fails.
    """
    mask = (wavelengths >= peak_wl - half_width_nm) & (wavelengths <= peak_wl + half_width_nm)
    if mask.sum() < 3:
        return peak_wl

    wl_win = wavelengths[mask]
    int_win = intensities[mask]
    int_win = np.clip(int_win - int_win.min(), 0, None)
    total = int_win.sum()
    if total < 1e-12:
        return peak_wl
    return float(np.dot(wl_win, int_win) / total)


# ---------------------------------------------------------------------------
# Wavelength Shift (Primary Signal)
# ---------------------------------------------------------------------------


def estimate_shift_xcorr(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    reference_intensities: np.ndarray,
    window_nm: float = 15.0,
    center_nm: float | None = None,
    upsample: int = 10,
) -> float | None:
    """Estimate Δλ via cross-correlation within a spectral window.

    Upsamples the ROI by ``upsample`` × before computing the cross-correlation
    peak position, giving sub-pixel wavelength resolution.

    Parameters
    ----------
    wavelengths:
        Wavelength axis (monotonically increasing).
    intensities:
        Gas spectrum intensities.
    reference_intensities:
        Reference (baseline) intensities, same axis as ``wavelengths``.
    window_nm:
        Width of the ROI window for shift estimation (nm).
    center_nm:
        Center of the ROI window (nm).  ``None`` → auto-detect from reference spectrum midpoint.
    upsample:
        Upsampling factor for sub-pixel resolution.

    Returns
    -------
    float | None
        Δλ in nm (negative = blue-shift, positive = red-shift); ``None`` if estimation fails.
    """
    # If no center given, use midpoint of the wavelength axis
    ctr = center_nm if center_nm is not None else float(wavelengths[len(wavelengths) // 2])
    lo, hi = ctr - window_nm / 2, ctr + window_nm / 2
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 5:
        return None

    wl_roi = wavelengths[mask]
    ref_roi = reference_intensities[mask]
    gas_roi = intensities[mask]

    # Upsample
    n_up = len(wl_roi) * upsample
    wl_up = np.linspace(wl_roi[0], wl_roi[-1], n_up)
    ref_up = np.interp(wl_up, wl_roi, ref_roi)
    gas_up = np.interp(wl_up, wl_roi, gas_roi)

    # Normalised cross-correlation
    ref_norm = ref_up - ref_up.mean()
    gas_norm = gas_up - gas_up.mean()
    if ref_norm.std() < 1e-12 or gas_norm.std() < 1e-12:
        return None

    xcorr = np.correlate(gas_norm, ref_norm, mode="full")
    shift_idx = int(np.argmax(xcorr)) - (len(ref_up) - 1)
    delta_wl_per_sample = (wl_up[-1] - wl_up[0]) / (n_up - 1)
    return float(shift_idx * delta_wl_per_sample)


def concentration_from_shift(
    delta_lambda: float,
    slope: float = 1.0,
    intercept: float = 0.0,
) -> float | None:
    """Compute concentration estimate from wavelength shift.

    Parameters
    ----------
    delta_lambda:
        Measured Δλ in nm.
    slope:
        Calibration slope (nm/ppm).  Must be determined experimentally for
        each sensor/analyte combination.  Sign: negative = blue-shift on
        adsorption; positive = red-shift.  Default 1.0 is a neutral placeholder.
    intercept:
        Calibration intercept (nm).

    Returns
    -------
    float | None
        Estimated concentration in ppm; ``None`` if ``slope`` is ~zero.
    """
    if abs(slope) < 1e-12:
        return None
    conc = (delta_lambda - intercept) / slope
    return float(max(0.0, conc))  # physical floor: 0 ppm


# ---------------------------------------------------------------------------
# Peak Asymmetry
# ---------------------------------------------------------------------------


def _compute_peak_asymmetry(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    peak_wl: float,
    half_width_nm: float = 15.0,
) -> float | None:
    """Compute the left/right half-width asymmetry factor of a spectral peak.

    Asymmetry = (λ_peak − λ_left_HM) / (λ_right_HM − λ_peak)

    where λ_left_HM and λ_right_HM are the wavelengths at which the spectrum
    crosses the half-maximum value on each side of the peak, determined by
    linear interpolation between adjacent points.

    Parameters
    ----------
    wavelengths, intensities:
        Spectral data (may be gas or reference).
    peak_wl:
        Peak center wavelength in nm (from Lorentzian fit or argmax).
    half_width_nm:
        Window half-width for the asymmetry measurement (nm).  Should be
        ≥ 1.5× the expected FWHM so that the half-maximum crossing is
        captured on both sides.

    Returns
    -------
    float | None
        Asymmetry factor (1.0 = symmetric), or ``None`` if the crossing
        cannot be found on one or both sides (e.g., peak near window edge).
    """
    lo = peak_wl - half_width_nm
    hi = peak_wl + half_width_nm
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 7:
        return None

    wl = wavelengths[mask]
    yi = intensities[mask]

    peak_idx = int(np.argmax(yi))
    peak_val = float(yi[peak_idx])
    baseline = float(yi.min())
    amplitude = peak_val - baseline
    if amplitude < 1e-12:
        return None
    half_max = baseline + amplitude / 2.0

    # Left half-width: scan from peak toward shorter wavelengths
    lambda_left: float | None = None
    for i in range(peak_idx - 1, -1, -1):
        if yi[i] <= half_max:
            # Linear interpolation between points i and i+1
            denom = float(yi[i + 1] - yi[i])
            if abs(denom) > 1e-15:
                frac = (half_max - float(yi[i])) / denom
                lambda_left = float(wl[i]) + frac * float(wl[i + 1] - wl[i])
            else:
                lambda_left = float(wl[i])
            break

    # Right half-width: scan from peak toward longer wavelengths
    lambda_right: float | None = None
    for i in range(peak_idx + 1, len(yi)):
        if yi[i] <= half_max:
            denom = float(yi[i] - yi[i - 1])
            if abs(denom) > 1e-15:
                frac = (half_max - float(yi[i - 1])) / denom
                lambda_right = float(wl[i - 1]) + frac * float(wl[i] - wl[i - 1])
            else:
                lambda_right = float(wl[i])
            break

    if lambda_left is None or lambda_right is None:
        return None

    peak_center = float(wl[peak_idx])
    left_hw = peak_center - lambda_left
    right_hw = lambda_right - peak_center

    if left_hw <= 0 or right_hw <= 0:
        return None

    return left_hw / right_hw


# ---------------------------------------------------------------------------
# Full LSPR Feature Vector
# ---------------------------------------------------------------------------


def compute_lspr_reference(
    wavelengths: np.ndarray,
    reference_intensities: np.ndarray,
    search_min: float = LSPR_SEARCH_MIN_NM,
    search_max: float = LSPR_SEARCH_MAX_NM,
) -> LSPRReference:
    """Pre-compute the Lorentzian fit of a reference spectrum.

    Call this **once** when the reference spectrum is captured or loaded.
    Pass the returned :class:`LSPRReference` to :func:`extract_lspr_features`
    on every subsequent frame to skip the redundant reference re-fit.

    Parameters
    ----------
    wavelengths, reference_intensities:
        Baseline (reference) spectrum arrays.
    search_min, search_max:
        Wavelength search window for the LSPR peak (nm).

    Returns
    -------
    LSPRReference
        Cached fit — valid until the reference spectrum changes.
    """
    peak_wl = detect_lspr_peak(wavelengths, reference_intensities, search_min, search_max)
    fit: tuple[float, float, float, float] | None = None
    if peak_wl is not None:
        # Use the same wide fit window as extract_lspr_features so that the
        # reference FWHM is measured with consistent bounds (FWHM ≤ 40 nm by
        # default), enabling meaningful delta_fwhm_nm differences.
        fit = fit_lorentzian_peak(wavelengths, reference_intensities, peak_wl, half_width_nm=20.0)
    log.debug(
        "compute_lspr_reference: peak_wl=%s nm, fit=%s",
        f"{peak_wl:.4f}" if peak_wl is not None else "None",
        "ok" if fit is not None else "failed — xcorr fallback will be used",
    )
    return LSPRReference(
        wavelengths=wavelengths,
        intensities=reference_intensities,
        peak_wl=peak_wl,
        fit=fit,
    )


def extract_lspr_features(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    reference_intensities: np.ndarray,
    roi_center: float | None = None,
    roi_width: float = 20.0,
    lspr_ref: LSPRReference | None = None,
) -> LSPRFeatures:
    """Extract the spectral feature vector for one spectrum relative to a reference.

    ``roi_center`` is the peak wavelength around which all features are
    extracted.  When ``None`` (default), it is resolved automatically:
    first from ``lspr_ref.peak_wl``, then by argmax of the reference
    spectrum — so callers never need to hardcode a sensor-specific wavelength.
    """
    """Extract the 4-component LSPR feature vector for one spectrum.

    Parameters
    ----------
    wavelengths, intensities:
        Gas spectrum.
    reference_intensities:
        Reference (baseline) spectrum, same wavelength axis.
    roi_center:
        Centre of the ROI window (nm).
    roi_width:
        Width of the ROI window (nm).
    lspr_ref:
        Pre-computed reference fit from :func:`compute_lspr_reference`.
        When provided, the (expensive) reference Lorentzian fit is skipped —
        halving the number of ``curve_fit`` calls per frame.  Pass ``None``
        to retain the original single-call behaviour (e.g. in training scripts).

    Returns
    -------
    LSPRFeatures
        Populated dataclass.
    """
    # ── Auto-resolve roi_center from reference if not provided ────────────
    if roi_center is None:
        if lspr_ref is not None and lspr_ref.peak_wl is not None:
            roi_center = lspr_ref.peak_wl
        else:
            # Full-spectrum argmax fallback — works for any sensor
            auto = detect_lspr_peak(wavelengths, reference_intensities,
                                    float(wavelengths[0]), float(wavelengths[-1]))
            roi_center = auto if auto is not None else float(wavelengths[len(wavelengths) // 2])

    feat = LSPRFeatures(roi_center=roi_center, roi_width=roi_width)

    # ── Peak detection with Lorentzian refinement ──────────────────────────
    # Derive the search window from roi_center ± 2×roi_width so the function
    # always looks near the sensor's resonance (discovered from reference).
    _search_half = max(roi_width * 2.0, 30.0)  # at least ±30 nm
    _smin = roi_center - _search_half
    _smax = roi_center + _search_half

    coarse_peak = detect_lspr_peak(wavelengths, intensities, _smin, _smax)
    feat.peak_wavelength = coarse_peak

    # Use roi_width as the Lorentzian fit half-window so that the FWHM upper
    # bound (= 2 × half_width_nm) scales with the expected peak width.
    # Default roi_width=20 nm gives a 40 nm FWHM bound, which is appropriate
    # for LSPR peaks (~35–60 nm FWHM).  Using the default 10 nm window
    # would clamp all FWHM values at 20 nm and make delta_fwhm_nm useless.
    _fit_half_width = max(roi_width, 15.0)

    gas_fit = None
    if coarse_peak is not None:
        gas_fit = fit_lorentzian_peak(wavelengths, intensities, coarse_peak, half_width_nm=_fit_half_width)
        if gas_fit is not None:
            feat.peak_wavelength = gas_fit[0]
            feat.peak_fwhm_nm = gas_fit[1]

    # Reference peak — use pre-computed LSPRReference if available.
    # This avoids re-fitting the Lorentzian on data that never changes,
    # saving ~2 curve_fit calls (~5–10 ms) on every acquisition frame.
    if lspr_ref is not None:
        ref_fit = lspr_ref.fit
    else:
        ref_fit = None
        ref_peak_wl = detect_lspr_peak(wavelengths, reference_intensities, _smin, _smax)
        if ref_peak_wl is not None:
            ref_fit = fit_lorentzian_peak(wavelengths, reference_intensities, ref_peak_wl, half_width_nm=_fit_half_width)

    if gas_fit is not None and ref_fit is not None:
        # High-precision Δλ = gas_center − ref_center
        feat.delta_lambda = float(gas_fit[0] - ref_fit[0])
        # σ_Δλ = √(σ_gas² + σ_ref²)  (error propagation in quadrature)
        feat.delta_lambda_std = float(np.hypot(gas_fit[3], ref_fit[3]))
        # ΔFWHM and ΔAmplitude — orthogonal physical channels for multi-gas sensing.
        # ΔFWHM encodes plasmon damping (molecular size/scattering).
        # ΔA encodes surface coverage density change (oscillator coupling).
        # Both are extracted from the same Lorentzian fit at zero extra cost.
        feat.delta_fwhm_nm = float(gas_fit[1] - ref_fit[1])
        feat.delta_amplitude = float(gas_fit[2] - ref_fit[2])

        # 6th feature: Δasymmetry = peak shape directionality change.
        # Uses the same half-width as the Lorentzian fit window so the
        # crossing scan always covers the full FWHM extent.
        _asym_hw = _fit_half_width
        gas_asym = _compute_peak_asymmetry(wavelengths, intensities, gas_fit[0], _asym_hw)
        ref_asym = _compute_peak_asymmetry(
            wavelengths,
            reference_intensities,
            ref_fit[0] if lspr_ref is None else (lspr_ref.peak_wl or ref_fit[0]),
            _asym_hw,
        )
        feat.peak_asymmetry = gas_asym
        if gas_asym is not None and ref_asym is not None:
            feat.delta_asymmetry = gas_asym - ref_asym
    else:
        # Fallback: cross-correlation shift (still sub-pixel, but no σ estimate)
        feat.delta_lambda = estimate_shift_xcorr(
            wavelengths,
            intensities,
            reference_intensities,
            window_nm=roi_width,
            center_nm=roi_center,
        )

    # ── ROI-based intensity differences ────────────────────────────────────
    lo, hi = roi_center - roi_width / 2, roi_center + roi_width / 2
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() >= 3:
        gas_roi = intensities[mask]
        ref_roi = reference_intensities[mask]
        diff = gas_roi - ref_roi

        ref_peak_idx = np.argmax(ref_roi)
        feat.delta_intensity_peak = float(gas_roi[ref_peak_idx] - ref_roi[ref_peak_idx])
        feat.delta_intensity_area = float(np.trapezoid(diff, x=wavelengths[mask]))
        feat.delta_intensity_std = float(np.std(diff))

        # SNR: dimensionally consistent estimate.
        # Primary path (Lorentzian fit available): |Δλ| / σ_Δλ (nm/nm, unitless).
        # σ_Δλ is propagated from both fit covariances so units cancel exactly.
        # Fallback (xcorr only): |ΔI_peak| / σ_ΔI (intensity/intensity, unitless).
        # The old formula |Δλ_nm| / σ_intensity_counts was dimensionally invalid.
        if (
            feat.delta_lambda_std is not None
            and feat.delta_lambda_std > 1e-12
            and feat.delta_lambda is not None
        ):
            feat.snr = float(abs(feat.delta_lambda) / feat.delta_lambda_std)
        elif (
            feat.delta_intensity_std is not None
            and feat.delta_intensity_std > 1e-12
            and feat.delta_intensity_peak is not None
        ):
            feat.snr = float(abs(feat.delta_intensity_peak) / feat.delta_intensity_std)

    return feat
