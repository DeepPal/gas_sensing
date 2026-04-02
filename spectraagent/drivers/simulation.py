"""
spectraagent.drivers.simulation
================================
Physics-complete simulation hardware driver.

Generates synthetic spectra using the full
:class:`~src.simulation.gas_response.SpectralSimulator` physics engine:

  - Langmuir kinetics: peak shifts as concentration ramps up with τ = 1/(kon×c + koff)
  - Multi-peak, multi-analyte support — any sensor configuration
  - Realistic spectrometer noise (shot + dark + readout)
  - Configurable gas exposure schedule for demos and CI

Gas scenario
------------
By default, the driver cycles through a repeating exposure protocol:

    0–30 s    clean air (baseline)
    30–150 s  Ethanol at 1 ppm  (rising edge → equilibrium via Langmuir kinetics)
    150–210 s clean air (recovery/desorption)
    ... repeat

This means the React frontend will see a real peak-shift response grow,
plateau, and recover — not just random jitter around a fixed wavelength.

Override the sensor and analyte schedule via constructor parameters.
"""
from __future__ import annotations

import time

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

# Defaults (sensor-agnostic)
_N_PIXELS: int = 3648
_WL_MIN: float = 500.0
_WL_MAX: float = 900.0
_PEAK_NM: float = 717.9      # default reference; overridden by sensor config
_FRAME_RATE_HZ: float = 10.0  # driver returns frames at this rate


class SimulationDriver(AbstractHardwareDriver):
    """Physics-complete simulation spectrometer driver.

    Uses the :class:`~src.simulation.gas_response.SpectralSimulator` to
    generate time-resolved spectra with realistic Langmuir kinetics.

    Parameters
    ----------
    integration_time_ms:
        Simulated acquisition time per frame (ms). Controls the sleep
        duration in ``read_spectrum()``.
    peak_nm:
        Sensor reference peak wavelength (nm). Used when no external
        ``sensor_config`` is provided.
    analyte_name:
        Name of the analyte to expose in the demo protocol.
    concentration_ppm:
        Concentration during the active exposure phase (ppm).
    sensitivity_nm_per_ppm:
        Sensor sensitivity (nm/ppm). Negative = blue-shift (the sensor LSPR);
        positive = red-shift.
    exposure_on_s:
        Duration of active gas exposure in each cycle (s).
    clean_air_s:
        Duration of clean-air (recovery) phase in each cycle (s).
    sensor_config:
        Optional full :class:`~src.simulation.gas_response.SensorConfig`.
        When provided, ``peak_nm``, ``analyte_name``, ``sensitivity_nm_per_ppm``
        are ignored — the SensorConfig is used directly.
    noise_level:
        Legacy noise-level parameter kept for API compatibility. Set to 0
        to use the physics noise model only.
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        peak_nm: float = _PEAK_NM,
        analyte_name: str = "Gas",
        concentration_ppm: float = 1.0,
        sensitivity_nm_per_ppm: float = -0.5,
        exposure_on_s: float = 120.0,
        clean_air_s: float = 60.0,
        sensor_config: object | None = None,   # SensorConfig | None
        noise_level: float = 0.0,              # kept for legacy compatibility
    ) -> None:
        self._integration_time_ms = integration_time_ms
        self._connected: bool = False
        self._start_time: float = 0.0

        self._exposure_on_s = exposure_on_s
        self._clean_air_s = clean_air_s
        self._cycle_s = exposure_on_s + clean_air_s

        # Build simulator
        try:
            from src.simulation.gas_response import (
                make_analyte,
                make_single_peak_sensor,
            )
            if sensor_config is not None:
                cfg = sensor_config  # type: ignore[assignment]
            else:
                cfg = make_single_peak_sensor(
                    peak_nm=peak_nm,
                    fwhm_nm=20.0,
                    wl_start=_WL_MIN,
                    wl_end=_WL_MAX,
                    n_pixels=_N_PIXELS,
                )
                cfg.analytes = [
                    make_analyte(
                        analyte_name,
                        n_peaks=1,
                        sensitivity_nm_per_ppm=sensitivity_nm_per_ppm,
                        tau_s=30.0,
                        kd_ppm=50.0,
                    )
                ]

            from src.simulation.gas_response import SpectralSimulator
            from src.simulation.noise_model import SpectrometerNoise
            self._sim = SpectralSimulator(
                cfg,
                noise_model=SpectrometerNoise(integration_time_s=integration_time_ms / 1000.0),
                rng=np.random.default_rng(0),
            )
            self._cfg = cfg
            self._wavelengths: np.ndarray = cfg.wavelengths  # type: ignore[union-attr]
            self._physics_available = True
        except ImportError:
            # Fallback: simple Lorentzian if simulation package unavailable
            self._physics_available = False
            self._wavelengths = np.linspace(_WL_MIN, _WL_MAX, _N_PIXELS)
            self._peak_nm = peak_nm
            self._sensitivity = sensitivity_nm_per_ppm
            self._concentration_ppm = concentration_ppm

        self._analyte_conc = concentration_ppm

    # ── Connection lifecycle ─────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        self._start_time = time.monotonic()

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return self._wavelengths.copy()

    # ── Frame generation ─────────────────────────────────────────────────

    def read_spectrum(self) -> np.ndarray:
        """Sleep for integration time, then return a physics-simulated spectrum.

        The gas exposure follows a repeating cycle:
          - ``clean_air_s`` seconds of baseline (no gas)
          - ``exposure_on_s`` seconds of active analyte exposure
        """
        time.sleep(self._integration_time_ms / 1000.0)
        elapsed = time.monotonic() - self._start_time

        if self._physics_available:
            return self._physics_frame(elapsed)
        return self._fallback_frame(elapsed)

    def _physics_frame(self, elapsed: float) -> np.ndarray:
        """Generate one frame using the physics simulation engine."""
        phase = elapsed % self._cycle_s

        if phase < self._clean_air_s:
            # Clean air — no analyte
            concs: dict[str, float] = {}
            elapsed_exposure = 0.0
        else:
            # Active exposure
            elapsed_exposure = phase - self._clean_air_s
            analyte_name = self._cfg.analytes[0].name if self._cfg.analytes else "Gas"
            concs = {analyte_name: self._analyte_conc}

        spectrum, _ = self._sim.spectrum_at_state(
            analyte_concentrations=concs,
            elapsed_since_exposure_s=elapsed_exposure,
            temp_c=25.0,
            humidity_pct=50.0,
            add_noise=True,
        )
        return np.clip(spectrum, 0.0, None)

    def _fallback_frame(self, elapsed: float) -> np.ndarray:
        """Simple Lorentzian fallback (no src.simulation package)."""
        phase = elapsed % self._cycle_s
        if phase < self._clean_air_s:
            shift = 0.0
        else:
            t_exp = phase - self._clean_air_s
            tau = 30.0
            shift = self._sensitivity * self._concentration_ppm * (1.0 - np.exp(-t_exp / tau))

        peak_wl = self._peak_nm + shift + float(np.random.normal(0.0, 0.3))
        gamma = 9.0
        lorentz = 0.8 / (1.0 + ((self._wavelengths - peak_wl) / gamma) ** 2)
        noise = np.random.normal(0.0, 0.002, _N_PIXELS)
        return np.clip(lorentz + noise, 0.0, None)

    # ── Configuration ────────────────────────────────────────────────────

    def get_integration_time_ms(self) -> float:
        return self._integration_time_ms

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_time_ms = ms

    def set_concentration(self, concentration_ppm: float) -> None:
        """Dynamically change the simulated analyte concentration."""
        self._analyte_conc = concentration_ppm

    @property
    def name(self) -> str:
        return "Simulation"

    @property
    def is_connected(self) -> bool:
        return self._connected
