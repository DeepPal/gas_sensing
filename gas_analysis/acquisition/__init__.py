"""
CCS200 Spectrometer Acquisition Package
----------------------------------------
Provides real-time data acquisition from ThorLabs CCS200 spectrometers.

Supports multiple hardware interfaces (auto-detected):
  - native:  TLCCS_64.dll (Windows, best performance)
  - pyvisa:  PyVISA raw communication
  - serial:  Serial binary protocol
  - linux:   USBTMC direct (Linux/Raspberry Pi)
"""

from .ccs200_realtime import RealtimeAcquisitionService
from .device_discovery import discover_ccs200_resources

__all__ = ["RealtimeAcquisitionService", "discover_ccs200_resources"]
