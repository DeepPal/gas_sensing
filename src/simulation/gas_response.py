"""
src.simulation.gas_response
============================
Physics-complete spectroscopic sensor simulation.

Models the complete gas sensing measurement chain:
  1. Broadband source illuminates the sensor
  2. Sensor produces Lorentzian/Gaussian/Fano peaks at characteristic wavelengths
  3. Analyte adsorption shifts peak positions via Langmuir kinetics
  4. Multiple analytes shift multiple peaks independently (superposition, linear regime)
  5. Environmental drift (temperature, humidity) adds a slowly-varying baseline shift
  6. The spectrometer records the intensity vs wavelength

Sensor configurations supported
---------------------------------
- Single sensor, 1 peak, 1 analyte  → simplest case
- Single sensor, M peaks, 1 analyte  → richer feature vector per analyte
- Single sensor, M peaks, N analytes → simultaneous multi-analyte quantification
- Any wavelength range (400–2500 nm)

Key equations
-------------
Langmuir kinetics (association phase):
    Δλ_j(t) = Σᵢ  S_ij × f_Langmuir(cᵢ, Kd_ij) × (1 − exp(−t / τ_ij))

where:
    S_ij    = sensitivity of peak j to analyte i (nm per ppm)
    f_Langmuir(c, Kd) = c / (1 + c / Kd)  (saturation at high concentration)
    τ_ij    = 1 / (kon_ij × cᵢ + koff_ij)  (concentration-dependent time constant)
    kon_ij  = association rate constant (ppm⁻¹ s⁻¹)
    koff_ij = dissociation rate constant (s⁻¹)
    Kd_ij   = koff_ij / kon_ij  (dissociation constant, ppm)

Dissociation phase (after exposure ends):
    Δλ_j(t) = Δλ_j(t_end) × exp(−koff_ij × (t − t_end))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PeakProfile:
    """One spectral peak produced by the sensor (in clean air / reference state).

    Parameters
    ----------
    center_nm:
        Peak centre wavelength (nm). Sensor-specific — set from reference capture.
    fwhm_nm:
        Full width at half maximum (nm). Lorentzian linewidth.
    amplitude:
        Peak intensity (counts or a.u.) at the spectrometer detector.
    shape:
        Peak lineshape: 'lorentzian' (most optical sensors), 'gaussian'
        (fluorescence), or 'fano' (plasmonic sensors with dark mode coupling).
    fano_q:
        Fano asymmetry parameter q. Only used when shape='fano'.
        q → ∞ approaches Lorentzian; q = 0 gives a pure dip.
    """
    center_nm: float
    fwhm_nm: float = 20.0
    amplitude: float = 0.8
    shape: Literal["lorentzian", "gaussian", "fano"] = "lorentzian"
    fano_q: float = 3.0


@dataclass
class AnalyteProfile:
    """Physical properties of one analyte on a specific sensor.

    Each analyte can interact with EVERY peak on the sensor, but with
    different sensitivity and kinetics per peak.  The sensitivity matrix
    S[analyte_idx, peak_idx] is built from these profiles.

    Parameters
    ----------
    name:
        Analyte identifier (e.g. 'Ethanol', 'Acetone', 'Toluene').
    sensitivity_per_peak:
        Sensitivity S_ij (nm/ppm) for each sensor peak j.
        Negative = blue-shift; positive = red-shift.
        Length must match SensorConfig.peaks.
    kd_per_peak:
        Langmuir dissociation constant K_d (ppm) per peak.
        Controls the concentration at which the response saturates.
        For linear regime: cᵢ << K_d_ij.
    kon_per_peak:
        Association rate constant k_on (ppm⁻¹ s⁻¹) per peak.
        Determines how fast the response rises with concentration.
    koff_per_peak:
        Dissociation rate constant k_off (s⁻¹) per peak.
        Controls recovery rate after gas removal.
        Relationship: K_d = k_off / k_on.
    """
    name: str
    sensitivity_per_peak: list[float]       # nm/ppm, one per peak
    kd_per_peak: list[float]                # ppm, one per peak
    kon_per_peak: list[float]               # ppm⁻¹ s⁻¹, one per peak
    koff_per_peak: list[float]              # s⁻¹, one per peak

    def tau(self, peak_idx: int, concentration_ppm: float) -> float:
        """Effective time constant τ at given concentration (s).

        τ = 1 / (k_on × c + k_off)

        τ is CONCENTRATION-DEPENDENT — this is a key discriminator: two
        analytes with the same equilibrium Δλ will have different τ at
        the same concentration if their kon/koff differ.
        """
        kon = self.kon_per_peak[peak_idx]
        koff = self.koff_per_peak[peak_idx]
        return 1.0 / (kon * concentration_ppm + koff)

    def delta_lambda_eq(self, peak_idx: int, concentration_ppm: float) -> float:
        """Equilibrium peak shift Δλ_eq at given concentration (nm).

        Uses Langmuir isotherm: Δλ_eq = S × c / (1 + c / K_d)
        Linear regime (c << K_d): Δλ_eq ≈ S × c
        Saturation regime (c >> K_d): Δλ_eq → S × K_d
        """
        s = self.sensitivity_per_peak[peak_idx]
        kd = self.kd_per_peak[peak_idx]
        return s * concentration_ppm / (1.0 + concentration_ppm / kd)


@dataclass
class SensorConfig:
    """Complete physical description of one sensor chip.

    Parameters
    ----------
    peaks:
        List of PeakProfile — one per spectral peak.
    wl_start_nm, wl_end_nm:
        Spectrometer wavelength range (nm).
    n_pixels:
        Number of detector pixels.
    analytes:
        List of AnalyteProfile — one per analyte the sensor can detect.
        len(a.sensitivity_per_peak) must equal len(peaks) for each a.
    temp_sensitivity_nm_per_c:
        Thermal drift coefficient (nm/°C). Applies to ALL peaks uniformly.
        Typical LSPR: 0.02–0.10 nm/°C.
    humidity_sensitivity_nm_per_pct:
        Humidity drift coefficient (nm/%RH). Typical: 0.01–0.05 nm/%RH.
    """
    peaks: list[PeakProfile]
    wl_start_nm: float = 400.0
    wl_end_nm: float = 900.0
    n_pixels: int = 3648
    analytes: list[AnalyteProfile] = field(default_factory=list)
    temp_sensitivity_nm_per_c: float = 0.05
    humidity_sensitivity_nm_per_pct: float = 0.02

    def __post_init__(self) -> None:
        for a in self.analytes:
            if len(a.sensitivity_per_peak) != len(self.peaks):
                raise ValueError(
                    f"Analyte '{a.name}' has {len(a.sensitivity_per_peak)} sensitivities "
                    f"but sensor has {len(self.peaks)} peaks."
                )

    @property
    def wavelengths(self) -> np.ndarray:
        return np.linspace(self.wl_start_nm, self.wl_end_nm, self.n_pixels)

    @property
    def sensitivity_matrix(self) -> np.ndarray:
        """S matrix: shape (n_analytes, n_peaks). S[i,j] = sensitivity of peak j to analyte i."""
        return np.array([a.sensitivity_per_peak for a in self.analytes])

    @property
    def selectivity_matrix(self) -> np.ndarray | None:
        """K[i,j] = interference of analyte j on analyte i (via primary peak).

        K_ij = S_ij / S_ii (using primary peak, diagonal = 1.0).
        Returns None if fewer than 2 analytes.
        """
        if len(self.analytes) < 2:
            return None
        S = self.sensitivity_matrix
        # Use primary peak (peak 0) for each analyte's self-sensitivity
        diag = np.array([S[i, 0] for i in range(len(self.analytes))])
        if np.any(np.abs(diag) < 1e-9):
            return None
        K = S[:, 0:1] / diag[:, None]
        return cast(np.ndarray, K)


# ---------------------------------------------------------------------------
# Spectral synthesis
# ---------------------------------------------------------------------------


def _lorentzian(wl: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return amplitude / (1.0 + ((wl - center) / gamma) ** 2)


def _gaussian(wl: np.ndarray, center: float, fwhm: float, amplitude: float) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return cast(np.ndarray, amplitude * np.exp(-0.5 * ((wl - center) / sigma) ** 2))


def _fano(wl: np.ndarray, center: float, fwhm: float, amplitude: float, q: float) -> np.ndarray:
    """Fano resonance: asymmetric peak from interference of discrete mode with continuum."""
    gamma = fwhm / 2.0
    eps = (wl - center) / gamma
    return amplitude * (q + eps) ** 2 / (1.0 + eps ** 2)


def _render_peak(wl: np.ndarray, peak: PeakProfile, center_offset_nm: float = 0.0) -> np.ndarray:
    """Render one peak onto the wavelength axis with an optional centre offset."""
    c = peak.center_nm + center_offset_nm
    if peak.shape == "lorentzian":
        return _lorentzian(wl, c, peak.fwhm_nm, peak.amplitude)
    elif peak.shape == "gaussian":
        return _gaussian(wl, c, peak.fwhm_nm, peak.amplitude)
    else:  # fano
        return _fano(wl, c, peak.fwhm_nm, peak.amplitude, peak.fano_q)


def build_spectrum(
    wl: np.ndarray,
    peaks: list[PeakProfile],
    peak_offsets_nm: list[float] | None = None,
    baseline: float = 0.02,
) -> np.ndarray:
    """Render a complete spectrum from a list of peaks.

    Parameters
    ----------
    wl:
        Wavelength axis (nm).
    peaks:
        List of PeakProfile objects describing each resonance feature.
    peak_offsets_nm:
        Per-peak wavelength shift (Δλ) from the reference position.
        Length must match peaks. None → all zeros.
    baseline:
        Flat additive baseline (broadband scattering / dark level).

    Returns
    -------
    np.ndarray
        Intensity spectrum, same shape as wl.
    """
    offsets = peak_offsets_nm if peak_offsets_nm is not None else [0.0] * len(peaks)
    spectrum = np.full_like(wl, baseline)
    for peak, offset in zip(peaks, offsets):
        spectrum = spectrum + _render_peak(wl, peak, center_offset_nm=offset)
    return spectrum


# ---------------------------------------------------------------------------
# Simulated session
# ---------------------------------------------------------------------------


@dataclass
class ExposureEvent:
    """One analyte exposure step in a session."""
    analyte_name: str
    concentration_ppm: float
    start_time_s: float
    duration_s: float


@dataclass
class SimulatedFrame:
    """One acquisition frame from the simulation."""
    time_s: float
    wavelengths: np.ndarray
    intensities: np.ndarray          # raw spectrum (includes noise)
    intensities_clean: np.ndarray    # noise-free spectrum
    peak_shifts_nm: list[float]      # true Δλ per peak
    concentrations: dict[str, float] # true concentrations (ppm) per analyte


@dataclass
class SimulatedSession:
    """Complete simulated measurement session.

    Contains reference spectrum, all acquisition frames, and ground truth.
    Ready for feeding into the feature extraction / calibration pipeline.
    """
    sensor_config: SensorConfig
    reference_spectrum: np.ndarray
    wavelengths: np.ndarray
    frames: list[SimulatedFrame]
    exposure_events: list[ExposureEvent]
    ambient_temp_c: float
    ambient_humidity_pct: float

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def times(self) -> np.ndarray:
        return np.array([f.time_s for f in self.frames])

    @property
    def true_concentrations(self) -> dict[str, np.ndarray]:
        """Time series of true concentrations per analyte."""
        result: dict[str, list[float]] = {}
        for frame in self.frames:
            for name, conc in frame.concentrations.items():
                result.setdefault(name, []).append(conc)
        return {k: np.array(v) for k, v in result.items()}

    @property
    def true_peak_shifts(self) -> np.ndarray:
        """True Δλ per frame per peak. Shape: (n_frames, n_peaks)."""
        return np.array([f.peak_shifts_nm for f in self.frames])


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class SpectralSimulator:
    """Physics-complete spectroscopic sensor simulator.

    Generates realistic synthetic spectra for any sensor configuration,
    including multi-analyte mixtures with Langmuir kinetics and
    cross-interference between analytes.

    Parameters
    ----------
    config:
        Full sensor physical description (peaks, analytes, wavelength range).
    noise_model:
        Noise model to apply. If None, uses default SpectrometerNoise.
    rng:
        Random number generator for reproducibility.
    """

    def __init__(
        self,
        config: SensorConfig,
        noise_model: object | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._cfg = config
        self._wl = config.wavelengths
        self._rng = rng or np.random.default_rng()
        # Import here to avoid circular imports
        if noise_model is None:
            from src.simulation.noise_model import SpectrometerNoise
            self._noise = SpectrometerNoise()
        else:
            self._noise = noise_model  # type: ignore[assignment]

    # ── Reference spectrum ────────────────────────────────────────────────

    def reference_spectrum(
        self,
        temp_c: float = 25.0,
        humidity_pct: float = 50.0,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Generate the clean-air reference spectrum.

        The reference has all peaks at their nominal positions (no analyte).
        Environmental offsets from the calibration temperature/humidity are
        included if temp_c or humidity_pct differ from standard.
        """
        env_shift = (
            self._cfg.temp_sensitivity_nm_per_c * (temp_c - 25.0)
            + self._cfg.humidity_sensitivity_nm_per_pct * (humidity_pct - 50.0)
        )
        offsets = [env_shift] * len(self._cfg.peaks)
        clean = build_spectrum(self._wl, self._cfg.peaks, offsets)
        if add_noise:
            return self._noise.apply(clean, self._wl, self._rng)
        return clean

    # ── Single frame ──────────────────────────────────────────────────────

    def spectrum_at_state(
        self,
        analyte_concentrations: dict[str, float],
        elapsed_since_exposure_s: float,
        temp_c: float = 25.0,
        humidity_pct: float = 50.0,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, list[float]]:
        """Generate one spectrum at a given measurement state.

        Parameters
        ----------
        analyte_concentrations:
            Current concentration (ppm) of each analyte by name.
        elapsed_since_exposure_s:
            Time since the CURRENT concentration level was first applied.
            Used to compute the kinetic response (how far along the curve).
        temp_c, humidity_pct:
            Current environmental conditions.
        add_noise:
            Whether to add realistic spectrometer noise.

        Returns
        -------
        (spectrum, peak_shifts_nm)
            spectrum: intensity array, same shape as self.wavelengths
            peak_shifts_nm: true Δλ per peak (ground truth)
        """
        # Environmental baseline shift (same for all peaks)
        env_shift = (
            self._cfg.temp_sensitivity_nm_per_c * (temp_c - 25.0)
            + self._cfg.humidity_sensitivity_nm_per_pct * (humidity_pct - 50.0)
        )

        # Compute per-peak shift from all analytes (superposition)
        peak_shifts = [env_shift] * len(self._cfg.peaks)

        for analyte in self._cfg.analytes:
            c = analyte_concentrations.get(analyte.name, 0.0)
            if c <= 0.0:
                continue
            for j in range(len(self._cfg.peaks)):
                tau = analyte.tau(j, c)
                delta_eq = analyte.delta_lambda_eq(j, c)
                kinetic_factor = 1.0 - np.exp(-elapsed_since_exposure_s / tau)
                peak_shifts[j] += delta_eq * kinetic_factor

        clean = build_spectrum(self._wl, self._cfg.peaks, peak_shifts)
        if add_noise:
            spectrum = self._noise.apply(clean, self._wl, self._rng)
        else:
            spectrum = clean.copy()

        return spectrum, peak_shifts

    # ── Full session ──────────────────────────────────────────────────────

    def simulate_session(
        self,
        exposure_events: list[ExposureEvent],
        total_duration_s: float,
        frame_rate_hz: float = 2.0,
        temp_c: float = 25.0,
        humidity_pct: float = 50.0,
        temp_drift_rate_c_per_min: float = 0.0,
        add_noise: bool = True,
    ) -> SimulatedSession:
        """Simulate a complete measurement session.

        Generates frames at the specified frame rate, computing the correct
        kinetic state at each time point given the exposure schedule.

        Parameters
        ----------
        exposure_events:
            Ordered list of gas exposure events (analyte, concentration, start, duration).
            Events can overlap (mixture exposure) or be sequential (selectivity test).
        total_duration_s:
            Total session duration (s).
        frame_rate_hz:
            Acquisition frame rate (frames per second).
        temp_c:
            Starting ambient temperature (°C).
        temp_drift_rate_c_per_min:
            Slow temperature drift during session (°C/min). Models lab conditions.
        add_noise:
            Whether to add realistic spectrometer noise.

        Returns
        -------
        SimulatedSession
        """
        dt = 1.0 / frame_rate_hz
        times = np.arange(0.0, total_duration_s, dt)

        # Reference spectrum at session start
        ref = self.reference_spectrum(temp_c, humidity_pct, add_noise=add_noise)

        frames: list[SimulatedFrame] = []

        for t in times:
            # Current temperature (slow drift)
            current_temp = temp_c + temp_drift_rate_c_per_min * t / 60.0

            # Current concentrations from all active exposure events
            current_conc: dict[str, float] = {}
            elapsed_by_event: dict[str, float] = {}  # analyte → time since exposure start

            for ev in exposure_events:
                if ev.start_time_s <= t < ev.start_time_s + ev.duration_s:
                    # Active exposure
                    existing = current_conc.get(ev.analyte_name, 0.0)
                    current_conc[ev.analyte_name] = existing + ev.concentration_ppm
                    elapsed_by_event[ev.analyte_name] = float(t - ev.start_time_s)

            # Use the effective elapsed time for kinetics
            # For multiple simultaneous analytes, use the one with longest elapsed
            # (simplified model — proper treatment needs per-analyte tracking)
            elapsed = max(elapsed_by_event.values()) if elapsed_by_event else 0.0

            spectrum, shifts = self.spectrum_at_state(
                current_conc,
                elapsed_since_exposure_s=elapsed,
                temp_c=float(current_temp),
                humidity_pct=humidity_pct,
                add_noise=add_noise,
            )

            frames.append(SimulatedFrame(
                time_s=float(t),
                wavelengths=self._wl.copy(),
                intensities=spectrum,
                intensities_clean=build_spectrum(self._wl, self._cfg.peaks, shifts),
                peak_shifts_nm=list(shifts),
                concentrations=dict(current_conc),
            ))

        return SimulatedSession(
            sensor_config=self._cfg,
            reference_spectrum=ref,
            wavelengths=self._wl.copy(),
            frames=frames,
            exposure_events=exposure_events,
            ambient_temp_c=temp_c,
            ambient_humidity_pct=humidity_pct,
        )


# ---------------------------------------------------------------------------
# Factory helpers — quick sensor configs for common use cases
# ---------------------------------------------------------------------------


def make_single_peak_sensor(
    peak_nm: float = 700.0,
    fwhm_nm: float = 20.0,
    wl_start: float = 400.0,
    wl_end: float = 900.0,
    n_pixels: int = 3648,
) -> SensorConfig:
    """Convenience: one-peak sensor at any wavelength."""
    return SensorConfig(
        peaks=[PeakProfile(center_nm=peak_nm, fwhm_nm=fwhm_nm)],
        wl_start_nm=wl_start,
        wl_end_nm=wl_end,
        n_pixels=n_pixels,
    )


def make_multi_peak_sensor(
    peak_nms: list[float],
    fwhm_nms: list[float] | None = None,
    wl_start: float = 400.0,
    wl_end: float = 900.0,
    n_pixels: int = 3648,
) -> SensorConfig:
    """Convenience: multi-peak sensor at arbitrary peak positions."""
    fwhms = fwhm_nms or [20.0] * len(peak_nms)
    peaks = [PeakProfile(center_nm=c, fwhm_nm=f) for c, f in zip(peak_nms, fwhms)]
    return SensorConfig(peaks=peaks, wl_start_nm=wl_start, wl_end_nm=wl_end, n_pixels=n_pixels)


def make_analyte(
    name: str,
    n_peaks: int,
    sensitivity_nm_per_ppm: float,
    tau_s: float = 30.0,
    kd_ppm: float = 100.0,
    sensitivity_variation: float = 0.3,
    rng: np.random.Generator | None = None,
) -> AnalyteProfile:
    """Convenience: create an analyte with sensitivities for n_peaks.

    Each peak gets a slightly different sensitivity (modelling multi-mode
    binding — e.g., two different MIP binding sites with different affinities).

    Parameters
    ----------
    sensitivity_nm_per_ppm:
        Primary peak sensitivity (nm/ppm). Sign: negative = blue-shift.
    tau_s:
        Response time constant (s) at 1 ppm concentration.
    kd_ppm:
        Langmuir dissociation constant (ppm).
    sensitivity_variation:
        Fraction by which peak sensitivities vary (e.g. 0.3 → ±30%).
    """
    _rng = rng or np.random.default_rng()
    # Each peak has a slightly different sensitivity (physical: different binding modes)
    variations = 1.0 + _rng.uniform(-sensitivity_variation, sensitivity_variation, n_peaks)
    sensitivities = [sensitivity_nm_per_ppm * float(v) for v in variations]

    # Exact formula ensuring τ(c=1 ppm) = tau_s exactly.
    # τ = 1 / (kon × c + koff)  at c=1: kon + koff = 1/tau_s
    # Kd = koff / kon  →  koff = Kd × kon
    # kon × (1 + Kd) = 1/tau_s  →  kon = 1 / (tau_s × (1 + Kd))
    kon = 1.0 / (tau_s * (1.0 + kd_ppm))  # ppm⁻¹ s⁻¹
    koff = kon * kd_ppm                    # s⁻¹

    return AnalyteProfile(
        name=name,
        sensitivity_per_peak=sensitivities,
        kd_per_peak=[kd_ppm] * n_peaks,
        kon_per_peak=[kon] * n_peaks,
        koff_per_peak=[koff] * n_peaks,
    )
