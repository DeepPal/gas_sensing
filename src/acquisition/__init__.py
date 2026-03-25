"""
src.acquisition
===============
Hardware acquisition layer for the Au-MIP LSPR gas sensing platform.

Provides a unified re-export point for both the low-level
:class:`CCS200Spectrometer` (single-shot / blocking) and the high-level
:class:`RealtimeAcquisitionService` (callback-driven continuous acquisition),
regardless of which backend is available (native DLL, Linux USBTMC, PyVISA,
or serial).

Both classes live under ``gas_analysis/acquisition/`` and are bridged here
so that all new code imports from ``src.acquisition`` rather than the legacy
``gas_analysis`` package path.

Usage
-----
**Low-level (single-shot)**::

    from src.acquisition import CCS200Spectrometer

    spec = CCS200Spectrometer(integration_time_s=0.05)
    wavelengths = spec.get_wavelengths()
    intensities = spec.get_spectrum()
    spec.close()

**High-level (continuous callback)**::

    from src.acquisition import RealtimeAcquisitionService

    svc = RealtimeAcquisitionService(integration_time_ms=50, target_wavelength=717.9)
    svc.connect(interface="auto")
    svc.register_callback(my_callback)
    svc.start()
    ...
    svc.stop()
"""

from __future__ import annotations

try:
    from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService
except ImportError:  # gas_analysis not on path (e.g. minimal install)
    RealtimeAcquisitionService = None  # type: ignore[assignment, misc]

try:
    from gas_analysis.acquisition.ccs200_native import CCS200Spectrometer
except ImportError:
    CCS200Spectrometer = None  # type: ignore[assignment, misc]

__all__ = ["RealtimeAcquisitionService", "CCS200Spectrometer"]
