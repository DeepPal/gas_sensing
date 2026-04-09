"""
src.spectrometer
================
Hardware-agnostic spectrometer driver layer.

Provides a clean protocol (:class:`AbstractSpectrometer`) that every
spectrometer driver satisfies, so analysis code never depends on specific
hardware.  Swap from a Thorlabs CCS200 to a simulated instrument by
changing a single argument.

Quick start
-----------
::

    from src.spectrometer import SpectrometerRegistry, SpectralFrame

    # Use a simulated spectrometer (no hardware required)
    with SpectrometerRegistry.create("simulated", modality="lspr") as spec:
        spec.set_integration_time(0.05)
        dark  = spec.acquire_dark(accumulations=10)
        ref   = spec.acquire_reference(accumulations=10)
        spec.set_analyte_concentration(2.0)
        frame = spec.acquire(accumulations=3)

    # Plug in real Thorlabs CCS200
    # with SpectrometerRegistry.create("ccs200") as spec:
    #     ...

    # Extend with your own driver
    from src.spectrometer import AbstractSpectrometer, SpectrometerRegistry
    @SpectrometerRegistry.register("my_spectrometer")
    class MyDriver(AbstractSpectrometer):
        ...
"""

from src.spectrometer.base import AbstractSpectrometer, SpectralFrame
from src.spectrometer.registry import SpectrometerRegistry
from src.spectrometer.simulated import SimulatedSpectrometer

__all__ = [
    "AbstractSpectrometer",
    "SpectralFrame",
    "SpectrometerRegistry",
    "SimulatedSpectrometer",
]
