"""
src.simulation.dataset_generator
==================================
Batch synthetic dataset generation for training and validation.

Generates large-scale datasets of (spectrum, label) pairs from the
physics-complete :class:`~src.simulation.gas_response.SpectralSimulator`.

Dataset types
-------------
- **calibration** — step-concentration protocol per analyte; steady-state
  spectra at each concentration level. Ground truth: concentration (ppm).
- **mixture** — two or more analytes simultaneously; ground truth is the
  full concentration vector.
- **kinetic** — time-series spectra from a single exposure step; features
  include τ₆₃/τ₉₅ extracted from the rising edge.
- **selectivity** — one analyte present at a time; used to build/validate the
  sensitivity matrix.

Domain randomisation
--------------------
Each sample (or each session) can randomise:
  - Peak positions (±σ_peak_nm)
  - FWHM (±σ_fwhm_fraction)
  - Analyte sensitivities (±sensitivity_variation)
  - Noise model parameters (via DomainRandomizedNoise)

This produces training diversity that improves sim→real transfer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.gas_response import (
    AnalyteProfile,
    ExposureEvent,
    PeakProfile,
    SensorConfig,
    SimulatedSession,
    SpectralSimulator,
    make_analyte,
    make_single_peak_sensor,
)
from src.simulation.noise_model import DomainRandomizedNoise, SpectrometerNoise

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Full configuration for one dataset generation run.

    Parameters
    ----------
    sensor_config:
        Base sensor physical description. If ``domain_randomize=True``,
        each session randomises peak positions and FWHM around these values.
    analyte_names:
        List of analyte names to include.  Must match analytes in
        ``sensor_config.analytes``.
    concentration_levels:
        List of concentration levels (ppm) for calibration sweeps.
        For mixtures, all combinations are generated.
    n_sessions:
        Number of independent sessions to generate. Each session uses a
        freshly randomised sensor if ``domain_randomize=True``.
    frames_per_level:
        Number of frames to collect at each concentration level (steady-state
        average suppresses noise for calibration data).
    frame_rate_hz:
        Acquisition rate (frames/second) during simulation.
    exposure_duration_s:
        Duration of each concentration step (s).
    pre_exposure_s:
        Clean-air baseline collected before first analyte exposure (s).
    post_exposure_s:
        Recovery period after final exposure (s).
    domain_randomize:
        If True, randomise sensor peak positions, FWHM, sensitivities, and
        noise model parameters per session.
    peak_position_sigma_nm:
        Std of per-session peak centre randomisation (nm).
    fwhm_variation_fraction:
        Fractional std of FWHM randomisation (e.g. 0.15 = ±15%).
    sensitivity_variation_fraction:
        Fractional std of per-session sensitivity randomisation.
    include_mixtures:
        If True, also generate two-analyte mixture frames.
    temp_range_c:
        (min, max) ambient temperature range (°C) across sessions.
    humidity_range_pct:
        (min, max) relative humidity range (%) across sessions.
    random_seed:
        Master seed for reproducibility.
    """

    sensor_config: SensorConfig
    analyte_names: list[str]
    concentration_levels: list[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 5.0])
    n_sessions: int = 10
    frames_per_level: int = 20
    frame_rate_hz: float = 2.0
    exposure_duration_s: float = 120.0
    pre_exposure_s: float = 30.0
    post_exposure_s: float = 60.0
    domain_randomize: bool = True
    peak_position_sigma_nm: float = 2.0
    fwhm_variation_fraction: float = 0.15
    sensitivity_variation_fraction: float = 0.20
    include_mixtures: bool = True
    temp_range_c: tuple[float, float] = (20.0, 30.0)
    humidity_range_pct: tuple[float, float] = (40.0, 70.0)
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DatasetGenerator:
    """Generates large-scale synthetic datasets for training/validation.

    Example
    -------
    ::

        from src.simulation.gas_response import make_single_peak_sensor, make_analyte
        sensor = make_single_peak_sensor(peak_nm=700.0)
        sensor.analytes = [make_analyte('Ethanol', 1, -0.5)]
        cfg = DatasetConfig(sensor_config=sensor, analyte_names=['Ethanol'])
        gen = DatasetGenerator(cfg)
        df = gen.generate_calibration_dataset()
        X, y = gen.to_numpy(df, target_analyte='Ethanol')
    """

    def __init__(self, config: DatasetConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.random_seed)

    # ── Sensor randomisation ──────────────────────────────────────────────

    def _randomize_sensor(self, base: SensorConfig) -> SensorConfig:
        """Return a new SensorConfig with randomised peaks and analytes."""
        new_peaks = [
            PeakProfile(
                center_nm=p.center_nm + float(
                    self._rng.normal(0.0, self._cfg.peak_position_sigma_nm)
                ),
                fwhm_nm=p.fwhm_nm * (
                    1.0 + float(self._rng.normal(0.0, self._cfg.fwhm_variation_fraction))
                ),
                amplitude=p.amplitude,
                shape=p.shape,
                fano_q=p.fano_q,
            )
            for p in base.peaks
        ]
        new_analytes = []
        for a in base.analytes:
            variation = 1.0 + self._rng.uniform(
                -self._cfg.sensitivity_variation_fraction,
                self._cfg.sensitivity_variation_fraction,
                len(a.sensitivity_per_peak),
            )
            new_analytes.append(AnalyteProfile(
                name=a.name,
                sensitivity_per_peak=[float(s * v) for s, v in zip(a.sensitivity_per_peak, variation)],
                kd_per_peak=list(a.kd_per_peak),
                kon_per_peak=list(a.kon_per_peak),
                koff_per_peak=list(a.koff_per_peak),
            ))
        return SensorConfig(
            peaks=new_peaks,
            wl_start_nm=base.wl_start_nm,
            wl_end_nm=base.wl_end_nm,
            n_pixels=base.n_pixels,
            analytes=new_analytes,
            temp_sensitivity_nm_per_c=base.temp_sensitivity_nm_per_c,
            humidity_sensitivity_nm_per_pct=base.humidity_sensitivity_nm_per_pct,
        )

    def _make_noise_model(self) -> object:
        if self._cfg.domain_randomize:
            return DomainRandomizedNoise()
        return SpectrometerNoise()

    # ── Calibration dataset ───────────────────────────────────────────────

    def generate_calibration_dataset(
        self, analyte_name: str | None = None
    ) -> pd.DataFrame:
        """Generate single-analyte step-concentration calibration dataset.

        Returns a DataFrame with columns:
        - ``session_id``, ``analyte``, ``concentration_ppm``
        - ``peak_shift_{j}`` — mean Δλ per peak over steady-state frames
        - ``peak_shift_{j}_std`` — std over steady-state frames
        - ``wavelengths``, ``mean_spectrum`` (as object columns of arrays)
        - ``temp_c``, ``humidity_pct``
        - ``true_sensitivity_{j}`` — ground-truth S_ij used in this session
        """
        rows: list[dict[str, Any]] = []
        analytes_to_run = (
            [analyte_name] if analyte_name else self._cfg.analyte_names
        )

        for session_idx in range(self._cfg.n_sessions):
            sensor = (
                self._randomize_sensor(self._cfg.sensor_config)
                if self._cfg.domain_randomize
                else self._cfg.sensor_config
            )
            noise_model = self._make_noise_model()
            temp_c = float(self._rng.uniform(*self._cfg.temp_range_c))
            humidity_pct = float(self._rng.uniform(*self._cfg.humidity_range_pct))
            sim = SpectralSimulator(sensor, noise_model=noise_model, rng=self._rng)

            wl = sensor.wavelengths
            ref_spec = sim.reference_spectrum(temp_c, humidity_pct, add_noise=False)

            for analyte_name_i in analytes_to_run:
                analyte_obj = next(
                    (a for a in sensor.analytes if a.name == analyte_name_i), None
                )
                if analyte_obj is None:
                    continue

                for conc in self._cfg.concentration_levels:
                    # Build exposure event: pre → exposure → post
                    events = [ExposureEvent(
                        analyte_name=analyte_name_i,
                        concentration_ppm=conc,
                        start_time_s=self._cfg.pre_exposure_s,
                        duration_s=self._cfg.exposure_duration_s,
                    )]
                    total_s = (
                        self._cfg.pre_exposure_s
                        + self._cfg.exposure_duration_s
                        + self._cfg.post_exposure_s
                    )
                    session: SimulatedSession = sim.simulate_session(
                        events, total_s,
                        frame_rate_hz=self._cfg.frame_rate_hz,
                        temp_c=temp_c,
                        humidity_pct=humidity_pct,
                        add_noise=True,
                    )

                    # Steady-state = last N frames of exposure
                    steady_frames = self._cfg.frames_per_level
                    exp_end_idx = int(
                        (self._cfg.pre_exposure_s + self._cfg.exposure_duration_s)
                        * self._cfg.frame_rate_hz
                    )
                    start_idx = max(0, exp_end_idx - steady_frames)
                    ss_frames = session.frames[start_idx:exp_end_idx]

                    shifts_per_frame = np.array([f.peak_shifts_nm for f in ss_frames])
                    spectra = np.array([f.intensities for f in ss_frames])
                    diff_spectra = spectra - ref_spec

                    row: dict[str, Any] = {
                        "session_id": session_idx,
                        "analyte": analyte_name_i,
                        "concentration_ppm": conc,
                        "temp_c": temp_c,
                        "humidity_pct": humidity_pct,
                        "wavelengths": wl,
                        "mean_spectrum": spectra.mean(axis=0),
                        "mean_diff_spectrum": diff_spectra.mean(axis=0),
                        "reference_spectrum": ref_spec,
                        "n_peaks": len(sensor.peaks),
                    }
                    for j in range(len(sensor.peaks)):
                        shifts_j = shifts_per_frame[:, j] - float(
                            # subtract env baseline: shifts at t=0
                            self._cfg.sensor_config.temp_sensitivity_nm_per_c * (temp_c - 25.0)
                            + self._cfg.sensor_config.humidity_sensitivity_nm_per_pct * (humidity_pct - 50.0)
                        )
                        row[f"peak_shift_{j}"] = float(shifts_j.mean())
                        row[f"peak_shift_{j}_std"] = float(shifts_j.std())
                        row[f"true_sensitivity_{j}"] = analyte_obj.sensitivity_per_peak[j]
                    rows.append(row)

            log.debug("Session %d/%d complete", session_idx + 1, self._cfg.n_sessions)

        return pd.DataFrame(rows)

    # ── Mixture dataset ───────────────────────────────────────────────────

    def generate_mixture_dataset(
        self,
        analyte_pair: tuple[str, str] | None = None,
    ) -> pd.DataFrame:
        """Generate two-analyte mixture dataset.

        All combinations of concentration levels for the two analytes are
        simulated, plus pure-analyte controls (concentration=0 for the other).

        Returns DataFrame with same columns as calibration dataset, plus
        columns for the second analyte concentration.
        """
        if len(self._cfg.analyte_names) < 2:
            raise ValueError("Mixture dataset requires at least 2 analytes in config.")

        pair = analyte_pair or (self._cfg.analyte_names[0], self._cfg.analyte_names[1])
        rows: list[dict[str, Any]] = []

        for session_idx in range(self._cfg.n_sessions):
            sensor = (
                self._randomize_sensor(self._cfg.sensor_config)
                if self._cfg.domain_randomize
                else self._cfg.sensor_config
            )
            noise_model = self._make_noise_model()
            temp_c = float(self._rng.uniform(*self._cfg.temp_range_c))
            humidity_pct = float(self._rng.uniform(*self._cfg.humidity_range_pct))
            sim = SpectralSimulator(sensor, noise_model=noise_model, rng=self._rng)
            ref_spec = sim.reference_spectrum(temp_c, humidity_pct, add_noise=False)
            wl = sensor.wavelengths

            # All (conc_A, conc_B) combinations including zeros
            concs = [0.0] + list(self._cfg.concentration_levels)
            for c_a in concs:
                for c_b in concs:
                    if c_a == 0.0 and c_b == 0.0:
                        continue  # skip blank — use reference

                    events = []
                    if c_a > 0.0:
                        events.append(ExposureEvent(
                            analyte_name=pair[0],
                            concentration_ppm=c_a,
                            start_time_s=self._cfg.pre_exposure_s,
                            duration_s=self._cfg.exposure_duration_s,
                        ))
                    if c_b > 0.0:
                        events.append(ExposureEvent(
                            analyte_name=pair[1],
                            concentration_ppm=c_b,
                            start_time_s=self._cfg.pre_exposure_s,
                            duration_s=self._cfg.exposure_duration_s,
                        ))

                    total_s = (
                        self._cfg.pre_exposure_s
                        + self._cfg.exposure_duration_s
                        + self._cfg.post_exposure_s
                    )
                    session = sim.simulate_session(
                        events, total_s,
                        frame_rate_hz=self._cfg.frame_rate_hz,
                        temp_c=temp_c,
                        humidity_pct=humidity_pct,
                    )

                    exp_end = int(
                        (self._cfg.pre_exposure_s + self._cfg.exposure_duration_s)
                        * self._cfg.frame_rate_hz
                    )
                    start_idx = max(0, exp_end - self._cfg.frames_per_level)
                    ss_frames = session.frames[start_idx:exp_end]
                    shifts_per_frame = np.array([f.peak_shifts_nm for f in ss_frames])
                    spectra = np.array([f.intensities for f in ss_frames])

                    env_shift = (
                        self._cfg.sensor_config.temp_sensitivity_nm_per_c * (temp_c - 25.0)
                        + self._cfg.sensor_config.humidity_sensitivity_nm_per_pct * (humidity_pct - 50.0)
                    )
                    row: dict[str, Any] = {
                        "session_id": session_idx,
                        f"conc_{pair[0]}_ppm": c_a,
                        f"conc_{pair[1]}_ppm": c_b,
                        "temp_c": temp_c,
                        "humidity_pct": humidity_pct,
                        "wavelengths": wl,
                        "mean_spectrum": spectra.mean(axis=0),
                        "mean_diff_spectrum": (spectra - ref_spec).mean(axis=0),
                        "reference_spectrum": ref_spec,
                    }
                    for j in range(len(sensor.peaks)):
                        row[f"peak_shift_{j}"] = float(shifts_per_frame[:, j].mean() - env_shift)
                    rows.append(row)

        return pd.DataFrame(rows)

    # ── Kinetic dataset ───────────────────────────────────────────────────

    def generate_kinetic_dataset(
        self,
        analyte_name: str,
        concentration_ppm: float,
        n_sessions: int | None = None,
    ) -> list[SimulatedSession]:
        """Return full-time-series sessions for kinetic feature extraction.

        Unlike calibration/mixture datasets (which return DataFrames of
        steady-state features), this returns raw :class:`SimulatedSession`
        objects so the caller can extract τ₆₃/τ₉₅/k_on from the full curve.
        """
        n = n_sessions or self._cfg.n_sessions
        sessions = []
        for _ in range(n):
            sensor = (
                self._randomize_sensor(self._cfg.sensor_config)
                if self._cfg.domain_randomize
                else self._cfg.sensor_config
            )
            noise_model = self._make_noise_model()
            temp_c = float(self._rng.uniform(*self._cfg.temp_range_c))
            humidity_pct = float(self._rng.uniform(*self._cfg.humidity_range_pct))
            sim = SpectralSimulator(sensor, noise_model=noise_model, rng=self._rng)

            events = [ExposureEvent(
                analyte_name=analyte_name,
                concentration_ppm=concentration_ppm,
                start_time_s=self._cfg.pre_exposure_s,
                duration_s=self._cfg.exposure_duration_s,
            )]
            total_s = (
                self._cfg.pre_exposure_s
                + self._cfg.exposure_duration_s
                + self._cfg.post_exposure_s
            )
            sessions.append(sim.simulate_session(
                events, total_s,
                frame_rate_hz=self._cfg.frame_rate_hz,
                temp_c=temp_c,
                humidity_pct=humidity_pct,
            ))
        return sessions

    # ── Conversion helpers ────────────────────────────────────────────────

    @staticmethod
    def to_numpy(
        df: pd.DataFrame,
        target_analyte: str,
        peak_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (X, y) numpy arrays from a calibration DataFrame.

        X: (n_samples, n_peaks) peak shift feature matrix
        y: (n_samples,) concentration in ppm
        """
        n_peaks = int(df["n_peaks"].iloc[0])
        X = df[[f"peak_shift_{j}" for j in range(n_peaks)]].values.astype(float)
        y = df["concentration_ppm"].values.astype(float)
        return X, y

    @staticmethod
    def to_spectrum_numpy(
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (spectra, concentrations) as stacked numpy arrays.

        spectra: (n_samples, n_pixels) diff spectra (signal = spectrum − ref)
        concentrations: (n_samples,) concentration in ppm
        """
        spectra = np.stack(df["mean_diff_spectrum"].values)
        concs = df["concentration_ppm"].values.astype(float)
        return spectra, concs


# ---------------------------------------------------------------------------
# Quick-start factory
# ---------------------------------------------------------------------------


def make_ethanol_acetone_dataset(
    n_sessions: int = 20,
    random_seed: int = 42,
) -> tuple[DatasetConfig, DatasetGenerator]:
    """Create a ready-to-use two-analyte calibration/mixture generator.

    Sensor: single peak at 700 nm, 20 nm FWHM.
    Analytes: Ethanol (S = −0.5 nm/ppm), Acetone (S = −0.35 nm/ppm).
    Kinetics: Ethanol τ₆₃ ≈ 30 s, Acetone τ₆₃ ≈ 18 s at 1 ppm.
    """
    sensor = make_single_peak_sensor(peak_nm=700.0, fwhm_nm=20.0)
    sensor.analytes = [
        make_analyte("Ethanol", 1, -0.5, tau_s=30.0, kd_ppm=50.0),
        make_analyte("Acetone", 1, -0.35, tau_s=18.0, kd_ppm=80.0),
    ]
    cfg = DatasetConfig(
        sensor_config=sensor,
        analyte_names=["Ethanol", "Acetone"],
        n_sessions=n_sessions,
        random_seed=random_seed,
    )
    return cfg, DatasetGenerator(cfg)
