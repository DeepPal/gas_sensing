"""
src.simulation
==============
Physics-complete spectroscopic sensor simulation.

Generates synthetic spectra for any sensor configuration:
  - Single/multi-peak, any wavelength range
  - Single/multi-analyte with Langmuir kinetics
  - Realistic noise model (shot + dark + readout)
  - Temperature/humidity drift
  - Sensor-to-sensor manufacturing variation

Primary entry point: :class:`SpectralSimulator`
"""

from src.simulation.dataset_generator import DatasetConfig, DatasetGenerator
from src.simulation.gas_response import (
    AnalyteProfile,
    SensorConfig,
    SimulatedSession,
    SpectralSimulator,
)
from src.simulation.noise_model import NoiseModel, SpectrometerNoise

__all__ = [
    "AnalyteProfile",
    "SensorConfig",
    "SpectralSimulator",
    "SimulatedSession",
    "NoiseModel",
    "SpectrometerNoise",
    "DatasetGenerator",
    "DatasetConfig",
]
