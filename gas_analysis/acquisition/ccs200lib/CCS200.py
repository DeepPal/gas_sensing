"""
CCS200.py
---------
A Python module for controlling the CCS200 Spectrometer.

This module provides a class 'Spectrometer' with methods to interface with the CCS200 Spectrometer.
It encapsulates the functionality required to initialize, control, and retrieve data from the spectrometer.

Author: Orkun Açar
Date: 1.1.2024
"""

from ctypes import *
import os

import numpy as np


class Spectrometer:
    """
    A class to represent and control a CCS200 Spectrometer.

    ...

    Attributes
    ----------
    lib : CDLL
        A ctypes CDLL instance to interact with the spectrometer's DLL.
    ccs_handle : c_int
        A ctypes c_int to store the spectrometer's handle.
    ...

    Methods
    -------
    __init__(self):
        Initializes the Spectrometer class.

    _load_library(self):
        Internal method to load the spectrometer's DLL.

    _initialize_device(self):
        Internal method to initialize the spectrometer device.

    set_integration_time(self, time_in_seconds):
        Sets the integration time for the spectrometer.

    start_scan(self):
        Initiates a scan on the spectrometer.

    get_scan_data(self):
        Retrieves the scan data from the spectrometer.

    close(self):
        Closes the connection to the spectrometer.
    """

    def __init__(self, device_resource=None):
        """
        Initializes the Spectrometer class.

        This method initializes the DLL and the spectrometer device.

        Parameters
        ----------
        device_resource : str, optional
            VISA resource string for specific device. If None, auto-detects.
        """
        self.lib = None
        self.ccs_handle = c_int(0)
        self._load_library()
        self._initialize_device(device_resource)

    def _load_library(self):
        """
        Internal method to load the spectrometer's DLL.
        """
        try:
            # Updated path based on actual installation
            dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll"
            if not os.path.exists(dll_path):
                # Fallback to 32-bit if 64-bit not found
                dll_path = r"C:\Program Files (x86)\IVI Foundation\VISA\WinNT\Bin\TLCCS_32.dll"

            os.chdir(os.path.dirname(dll_path))
            self.lib = cdll.LoadLibrary(os.path.basename(dll_path))
        except Exception as e:
            raise RuntimeError("Failed to load spectrometer DLL: " + str(e))

    def _initialize_device(self, device_resource=None):
        """
        Internal method to initialize the spectrometer device.

        Args:
            device_resource: Optional VISA resource string. If None, auto-detects.
        """
        try:
            # Auto-detect if no device specified
            if device_resource is None:
                device_resource = self._find_ccs_device()

            if device_resource is None:
                raise RuntimeError("No CCS200 spectrometer found. Check USB connection.")

            resource_bytes = (
                device_resource.encode() if isinstance(device_resource, str) else device_resource
            )
            print(f"  Initializing with resource: {device_resource}")
            result = self.lib.tlccs_init(resource_bytes, 1, 1, byref(self.ccs_handle))
            if result != 0:
                raise RuntimeError(f"tlccs_init returned error code: {result}")
        except Exception as e:
            raise RuntimeError("Failed to initialize spectrometer: " + str(e))

    def _find_ccs_device(self):
        """Auto-detect CCS200 spectrometer via PyVISA."""
        try:
            import pyvisa

            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            rm.close()

            # Look for USB resources with ThorLabs VID
            for resource in resources:
                if "USB" in resource and "0x1313" in resource:
                    return resource
                if "USB" in resource and "ASRL" not in resource:
                    # Try any USB device
                    return resource

            # If no USB, but we have ASRL5 (the detected port),
            # the DLL might handle it internally
            # Try the default format
            return "USB0::0x1313::0x8089::RAW"
        except Exception as e:
            print(f"Could not auto-detect: {e}")

        # Default fallback
        return "USB0::0x1313::0x8089::RAW"

    def set_integration_time(self, time_in_seconds):
        """
        Sets the integration time for the spectrometer.

        Parameters
        ----------
        time_in_seconds : float
            The integration time in seconds.
        """
        try:
            integration_time = c_double(time_in_seconds)
            self.lib.tlccs_setIntegrationTime(self.ccs_handle, integration_time)
        except Exception as e:
            raise RuntimeError("Failed to set integration time: " + str(e))

    def start_scan(self):
        """
        Initiates a scan on the spectrometer.
        """
        try:
            self.lib.tlccs_startScan(self.ccs_handle)
        except Exception as e:
            raise RuntimeError("Failed to start scan: " + str(e))

    def get_scan_data(self):
        """
        Retrieves the scan data from the spectrometer.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple containing the wavelengths and intensities as numpy arrays.
        """
        try:
            wavelengths = (c_double * 3648)()
            self.lib.tlccs_getWavelengthData(
                self.ccs_handle, 0, byref(wavelengths), c_void_p(None), c_void_p(None)
            )

            data_array = (c_double * 3648)()
            self.lib.tlccs_getScanData(self.ccs_handle, byref(data_array))

            wavelengths_np = np.ctypeslib.as_array(wavelengths)
            data_array_np = np.ctypeslib.as_array(data_array)

            return wavelengths_np, data_array_np
        except Exception as e:
            raise RuntimeError("Failed to retrieve scan data: " + str(e))

    def close(self):
        """
        Closes the connection to the spectrometer.
        """
        try:
            self.lib.tlccs_close(self.ccs_handle)
        except Exception as e:
            raise RuntimeError("Failed to close spectrometer: " + str(e))
