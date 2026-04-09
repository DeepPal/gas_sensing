"""
CCS200 Native DLL Interface - Working Version
-----------------------------------------------
Uses TLCCS_64.dll with correct API calls based on ThorLabs documentation.
"""
import ctypes
from ctypes import c_int, c_double, c_char_p, c_void_p, byref, pointer, create_string_buffer
import numpy as np
import os

class CCS200Spectrometer:
    """Working CCS200 interface using native TLCCS_64.dll"""
    
    # Status flags from TLCCS.h
    STATUS_SCAN_IDLE = 0x0002
    STATUS_SCAN_TRANSFER = 0x0010
    NUM_PIXELS = 3648
    
    def __init__(self, resource_string=None):
        self.handle = c_int(0)
        self._load_dll()
        self._init_device(resource_string)
        self._wavelengths = None
        self._load_wavelengths()
        
    def _load_dll(self):
        dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll"
        dll_dir = os.path.dirname(dll_path)
        prev_cwd = os.getcwd()
        try:
            os.chdir(dll_dir)
            self.dll = ctypes.CDLL(dll_path)
        finally:
            os.chdir(prev_cwd)
        
        # Define function signatures
        self.dll.tlccs_init.argtypes = [c_char_p, c_int, c_int, ctypes.POINTER(c_int)]
        self.dll.tlccs_init.restype = c_int
        
        self.dll.tlccs_close.argtypes = [c_int]
        self.dll.tlccs_close.restype = c_int
        
        self.dll.tlccs_setIntegrationTime.argtypes = [c_int, c_double]
        self.dll.tlccs_setIntegrationTime.restype = c_int
        
        self.dll.tlccs_startScan.argtypes = [c_int]
        self.dll.tlccs_startScan.restype = c_int
        
        self.dll.tlccs_getDeviceStatus.argtypes = [c_int, ctypes.POINTER(c_int)]
        self.dll.tlccs_getDeviceStatus.restype = c_int
        
        self.dll.tlccs_getScanData.argtypes = [c_int, ctypes.POINTER(c_double * self.NUM_PIXELS)]
        self.dll.tlccs_getScanData.restype = c_int
        
        self.dll.tlccs_getWavelengthData.argtypes = [c_int, c_int, ctypes.POINTER(c_double * self.NUM_PIXELS), c_void_p, c_void_p]
        self.dll.tlccs_getWavelengthData.restype = c_int
        
    def _init_device(self, resource_string=None):
        if resource_string is None:
            # Use the detected serial number from Windows
            resource_string = "USB0::0x1313::0x8089::M00505929::RAW"
            print(f"  Using detected device: {resource_string}")
        
        rsrc = resource_string.encode() if isinstance(resource_string, str) else resource_string
        print(f"  Initializing...")
        
        result = self.dll.tlccs_init(rsrc, 1, 1, byref(self.handle))
        if result != 0:
            raise RuntimeError(f"tlccs_init failed with code: {result}")
        print(f"  ✓ Device initialized (handle: {self.handle.value})")
        
    def _load_wavelengths(self):
        wavelengths = (c_double * self.NUM_PIXELS)()
        result = self.dll.tlccs_getWavelengthData(self.handle.value, 0, wavelengths, None, None)
        if result == 0:
            self._wavelengths = np.ctypeslib.as_array(wavelengths)
        else:
            self._wavelengths = np.linspace(200, 1000, self.NUM_PIXELS)
            
    def get_wavelengths(self):
        return self._wavelengths
        
    def set_integration_time(self, time_seconds):
        result = self.dll.tlccs_setIntegrationTime(self.handle.value, c_double(time_seconds))
        if result != 0:
            raise RuntimeError(f"Failed to set integration time: {result}")
            
    def get_scan_data(self):
        # Poll status until scan is complete
        import time
        max_wait = 10.0  # seconds
        wait_start = time.time()
        
        status = c_int(0)
        while time.time() - wait_start < max_wait:
            self.dll.tlccs_getDeviceStatus(self.handle.value, byref(status))
            if status.value & self.STATUS_SCAN_TRANSFER:
                break
            time.sleep(0.01)
        
        # Read data
        data = (c_double * self.NUM_PIXELS)()
        result = self.dll.tlccs_getScanData(self.handle.value, byref(data))
        if result != 0:
            raise RuntimeError(f"Failed to get scan data: {result}")
        
        return np.ctypeslib.as_array(data)
    
    def get_data(self):
        # Start scan and return data
        result = self.dll.tlccs_startScan(self.handle.value)
        if result != 0:
            raise RuntimeError(f"Failed to start scan: {result}")
        return self.get_scan_data()
    
    def get_data_sync(self):
        """Synchronous acquisition with explicit status polling."""
        # Start scan
        result = self.dll.tlccs_startScan(self.handle.value)
        if result != 0:
            raise RuntimeError(f"Failed to start scan: {result}")
        
        # Wait for data ready with timeout - check for IDLE after transfer
        import time
        max_wait = 5.0
        wait_start = time.time()
        status = c_int(0)
        
        while time.time() - wait_start < max_wait:
            self.dll.tlccs_getDeviceStatus(self.handle.value, byref(status))
            if status.value & self.STATUS_SCAN_TRANSFER:
                # Transfer started - wait for completion
                time.sleep(0.01)
                # Now wait for IDLE state
                while time.time() - wait_start < max_wait:
                    self.dll.tlccs_getDeviceStatus(self.handle.value, byref(status))
                    if status.value & self.STATUS_SCAN_IDLE:
                        break
                    time.sleep(0.005)
                break
            time.sleep(0.005)
        else:
            raise RuntimeError("Scan timeout - data not ready")
        
        # Read data
        data = (c_double * self.NUM_PIXELS)()
        result = self.dll.tlccs_getScanData(self.handle.value, byref(data))
        if result != 0:
            raise RuntimeError(f"Failed to get scan data: {result}")
        
        return np.ctypeslib.as_array(data)
        
    def close(self):
        if self.handle.value != 0:
            self.dll.tlccs_close(self.handle.value)
            self.handle.value = 0

# Compatibility aliases
ThorlabsCCS = CCS200Spectrometer

if __name__ == "__main__":
    print("Testing CCS200 Native Interface...")
    try:
        spec = CCS200Spectrometer()
        print(f"Connected! Wavelength range: {spec.get_wavelengths()[0]:.1f} - {spec.get_wavelengths()[-1]:.1f} nm")
        
        spec.set_integration_time(0.01)
        data = spec.get_data()
        print(f"Data acquired: {len(data)} pixels, range {data.min():.2f} - {data.max():.2f}")
        
        spec.close()
        print("Test passed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
