"""
ThorLabs CCS200 Spectrometer Interface
----------------------------------------
Clean wrapper around the PyCCS200 library with auto-detection and error handling.
Compatible with the real-time acquisition service.
"""

import contextlib
from pathlib import Path
import sys

import numpy as np

# Add the ccs200lib to path
sys.path.insert(0, str(Path(__file__).parent / "ccs200lib"))


class CCS200Spectrometer:
    """
    High-level interface for ThorLabs CCS200 spectrometer.

    Usage:
        spec = CCS200Spectrometer()
        spec.set_integration_time(0.05)  # 50ms
        wavelengths, intensities = spec.get_data()
        spec.close()
    """

    # ThorLabs CCS200 USB VID/PID
    VID = 0x1313
    PID = 0x8089  # CCS200

    def __init__(self, device_path=None):
        """
        Initialize and connect to CCS200 spectrometer.

        Args:
            device_path: Optional specific VISA resource string. If None, auto-detects.
        """
        self._spec = None
        self._device_path = device_path
        self._integration_time = 0.01  # Default 10ms
        self._wavelengths = None
        self._connected = False

        self._connect()

    def _connect(self):
        """Connect to the spectrometer."""
        try:
            # Try to import and use the PyCCS200 library
            from CCS200 import Spectrometer as PyCCS200Spectrometer

            # Create spectrometer instance
            self._spec = PyCCS200Spectrometer()

            # Get wavelength calibration (fixed for CCS200)
            wavelengths, _ = self._spec.get_scan_data()
            self._wavelengths = wavelengths
            self._connected = True

        except Exception as e:
            raise RuntimeError(f"Failed to connect to CCS200: {e}")

    def get_wavelengths(self):
        """Return the wavelength calibration array."""
        if self._wavelengths is None:
            raise RuntimeError("Spectrometer not connected")
        return self._wavelengths

    def set_integration_time(self, time_seconds):
        """
        Set integration time.

        Args:
            time_seconds: Integration time in seconds (e.g., 0.05 for 50ms)
        """
        if not self._connected:
            raise RuntimeError("Spectrometer not connected")

        self._spec.set_integration_time(time_seconds)
        self._integration_time = time_seconds

    def get_data(self):
        """
        Acquire a single spectrum.

        Returns:
            numpy.ndarray: Intensity array (3648 points)
        """
        if not self._connected:
            raise RuntimeError("Spectrometer not connected")

        # Start scan and get data
        self._spec.start_scan()
        wavelengths, intensities = self._spec.get_scan_data()

        return intensities

    def get_scan(self):
        """
        Acquire a single spectrum with wavelengths.

        Returns:
            tuple: (wavelengths, intensities) as numpy arrays
        """
        if not self._connected:
            raise RuntimeError("Spectrometer not connected")

        self._spec.start_scan()
        wavelengths, intensities = self._spec.get_scan_data()

        return wavelengths, intensities

    def get_intensity_at_wavelength(self, target_wavelength):
        """
        Get intensity at a specific wavelength.

        Args:
            target_wavelength: Target wavelength in nm

        Returns:
            float: Intensity at the closest pixel to target wavelength
        """
        if self._wavelengths is None:
            raise RuntimeError("Spectrometer not connected")

        intensities = self.get_data()
        idx = np.argmin(np.abs(self._wavelengths - target_wavelength))
        return float(intensities[idx])

    @property
    def connected(self):
        """Return True if spectrometer is connected."""
        return self._connected

    def close(self):
        """Close the spectrometer connection."""
        if self._spec and self._connected:
            with contextlib.suppress(Exception):
                self._spec.close()
            self._connected = False
            self._spec = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def list_ccs_devices():
    """
    List available CCS spectrometers connected via USB.

    Returns:
        list: List of device resource strings
    """
    devices = []
    try:
        import usb

        # Find all ThorLabs CCS devices
        for bus in usb.busses():
            for dev in bus.devices:
                if dev.idVendor == CCS200Spectrometer.VID:
                    if dev.idProduct == CCS200Spectrometer.PID:
                        devices.append(f"USB0::0x{dev.idVendor:04X}::0x{dev.idProduct:04X}::RAW")
    except Exception:
        pass

    return devices


# For compatibility with existing code that expects pylablib-style interface
ThorlabsCCS = CCS200Spectrometer
