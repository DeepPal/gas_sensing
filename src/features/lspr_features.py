"""
src.features.lspr_features
============================
Spectrometer-based sensor feature extraction: peak detection, wavelength
shift (Δλ), and the 4-component feature vector used as model input.

Originally developed for Au-MIP LSPR sensors; the peak-detection and
feature-extraction logic is general and works for any spectrometer sensor
whose response is a wavelength shift or intensity change.

Physics note: the primary signal is Δλ = λ_gas − λ_reference (nm).
- **Negative Δλ** (blue-shift): analyte adsorption shortens the effective
  optical path or decreases the local refractive index.  Sensitivity slope < 0.
- **Positive Δλ** (red-shift): analyte increases the local refractive index.
  Sensitivity slope > 0.
The sign of ``LSPR_SENSITIVITY_NM_PER_PPM`` matches the sensor's response
direction and must be confirmed experimentally for each sensor type.

``LSPR_REFERENCE_PEAK_NM``, ``LSPR_SEARCH_MIN_NM``, and
``LSPR_SEARCH_MAX_NM`` are **defaults** that should be overridden at the
call site for sensors with a different spectral range.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LSPR Physical Constants
# ---------------------------------------------------------------------------

LSPR_REFERENCE_PEAK_NM: float = 531.5  # Au nanoparticles reference peak
LSPR_SENSITIVITY_NM_PER_PPM: float = -0.116  # negative: adsorption → blue-shift (shorter λ)
LSPR_SEARCH_MIN_NM: float = 480.0
LSPR_SEARCH_MAX_NM: float = 600.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LSPRFeatures:
    """LSPR feature vector for one spectrum relative to a reference.

    These four features capture the physically meaningful changes in the
    LSPR band upon analyte adsorption.
    """

    delta_lambda: float | None = None
    """Peak wavelength shift Δλ = λ_gas − λ_reference (nm). Primary signal."""

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
    """Full-width at half-maximum of the LSPR peak from Lorentzian fit (nm).
    Proxy for sensor selectivity — narrower peaks → better discrimination."""

    @property
    def feature_vector(self) -> list[float]:
        """Return the 4-element feature vector [Δλ, ΔI_peak, ΔI_area, ΔI_std].

        Missing values are replaced with 0.0 for model compatibility.
        """
        return [
            self.delta_lambda if self.delta_lambda is not None else 0.0,
            self.delta_intensity_peak if self.delta_intensity_peak is not None else 0.0,
            self.delta_intensity_area if self.delta_intensity_area is not None else 0.0,
            self.delta_intensity_std if self.delta_intensity_std is not None else 0.0,
        ]


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
        popt, pcov = curve_fit(_lorentzian, wl, yi, p0=p0, bounds=bounds, maxfev=2000)
        center, gamma, amp, _ = popt
        perr = np.sqrt(np.diag(pcov))
        center_std = float(perr[0])

        # Sanity checks: center must be within window; FWHM must be positive
        if not (lo <= center <= hi) or gamma <= 0 or amp <= 0:
            return None

        return float(center), float(abs(gamma)), float(amp), center_std
    except Exception:
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
        Center of the ROI window.  ``None`` → use ``LSPR_REFERENCE_PEAK_NM``.
    upsample:
        Upsampling factor for sub-pixel resolution.

    Returns
    -------
    float | None
        Δλ in nm (negative = blue-shift, positive = red-shift); ``None`` if estimation fails.
    """
    ctr = center_nm if center_nm is not None else LSPR_REFERENCE_PEAK_NM
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
    slope: float = LSPR_SENSITIVITY_NM_PER_PPM,
    intercept: float = 0.0,
) -> float | None:
    """Compute concentration estimate from wavelength shift.

    Parameters
    ----------
    delta_lambda:
        Measured Δλ in nm.
    slope:
        Calibration slope (nm/ppm).  Default: −0.116 nm/ppm (ethanol literature).
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
        fit = fit_lorentzian_peak(wavelengths, reference_intensities, peak_wl)
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
    roi_center: float = LSPR_REFERENCE_PEAK_NM,
    roi_width: float = 20.0,
    lspr_ref: LSPRReference | None = None,
) -> LSPRFeatures:
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
    feat = LSPRFeatures(roi_center=roi_center, roi_width=roi_width)

    # ── Peak detection with Lorentzian refinement ──────────────────────────
    # Primary: detect coarse peak, then fit Lorentzian for sub-nm precision.
    # The fit also provides FWHM (sensor selectivity) and center uncertainty
    # (σ_gas), which propagates into Δλ uncertainty.
    coarse_peak = detect_lspr_peak(wavelengths, intensities)
    feat.peak_wavelength = coarse_peak

    gas_fit = None
    if coarse_peak is not None:
        gas_fit = fit_lorentzian_peak(wavelengths, intensities, coarse_peak)
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
        ref_peak_wl = detect_lspr_peak(wavelengths, reference_intensities)
        if ref_peak_wl is not None:
            ref_fit = fit_lorentzian_peak(wavelengths, reference_intensities, ref_peak_wl)

    if gas_fit is not None and ref_fit is not None:
        # High-precision Δλ = gas_center − ref_center
        feat.delta_lambda = float(gas_fit[0] - ref_fit[0])
        # σ_Δλ = √(σ_gas² + σ_ref²)  (error propagation in quadrature)
        feat.delta_lambda_std = float(np.hypot(gas_fit[3], ref_fit[3]))
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
