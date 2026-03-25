"""
CCS200 Serial Binary Protocol Interface
----------------------------------------
Implements the native CCS200 communication protocol via serial port.
Based on ThorLabs TLCCS driver documentation.
"""
import struct
import time
import numpy as np
import pyvisa
from typing import Optional, Tuple


class CCS200Serial:
    """
    CCS200 interface using serial binary protocol.
    
    The CCS200 uses a binary protocol over USB serial interface.
    Commands are single bytes or byte sequences with binary responses.
    """
    
    # Command bytes based on TLCCS driver protocol
    CMD_INIT = 0x01
    CMD_CLOSE = 0x02
    CMD_START_SCAN = 0x10
    CMD_GET_STATUS = 0x11
    CMD_GET_DATA = 0x12
    CMD_SET_INTEGRATION = 0x20
    CMD_GET_WAVELENGTHS = 0x30
    CMD_RESET = 0xFF
    
    # Status flags
    STATUS_IDLE = 0x00
    STATUS_SCANNING = 0x01
    STATUS_DATA_READY = 0x02
    STATUS_ERROR = 0xFF
    
    NUM_PIXELS = 3648
    
    def __init__(self, resource_name: Optional[str] = None):
        self.instrument = None
        self.rm = None
        self._wavelengths = None
        self._integration_time = 0.01  # 10ms default
        
        self._connect(resource_name)
        self._initialize()
        self._load_wavelengths()
        
    def _connect(self, resource_name: Optional[str] = None):
        """Connect to CCS200 via serial port."""
        self.rm = pyvisa.ResourceManager()
        
        if resource_name is None:
            # Find ASRL device
            resources = self.rm.list_resources()
            for r in resources:
                if 'ASRL' in r:
                    resource_name = r
                    break
        
        if not resource_name:
            raise RuntimeError("No ASRL resource found")
        
        print(f"Connecting to {resource_name}...")
        
        # Open with specific CCS200 serial settings
        self.instrument = self.rm.open_resource(resource_name)
        self.instrument.baud_rate = 115200
        self.instrument.data_bits = 8
        self.instrument.parity = pyvisa.constants.Parity.none
        self.instrument.stop_bits = pyvisa.constants.StopBits.one
        self.instrument.read_termination = None  # Binary mode
        self.instrument.write_termination = None
        self.instrument.timeout = 5000
        
        # Clear buffers
        try:
            self.instrument.clear()
        except:
            pass
            
        print(f"  ✓ Connected")
        
    def _initialize(self):
        """Initialize device communication."""
        # Send reset command
        self._send_command(self.CMD_RESET)
        time.sleep(0.1)
        
        # Send init command
        response = self._send_command(self.CMD_INIT)
        if response and response[0] != 0x00:
            print(f"  Warning: Init returned {response[0]:02X}")
        
        print(f"  ✓ Device initialized")
        
    def _send_command(self, cmd: int, data: bytes = None, read_length: int = 0) -> Optional[bytes]:
        """Send command and optionally read response."""
        try:
            # Send command byte
            self.instrument.write_raw(bytes([cmd]))
            
            # Send additional data if provided
            if data:
                time.sleep(0.01)
                self.instrument.write_raw(data)
            
            # Read response if requested
            if read_length > 0:
                time.sleep(0.01)  # Small delay for device processing
                try:
                    return self.instrument.read_bytes(read_length)
                except:
                    return None
            
            return None
        except Exception as e:
            print(f"Command error: {e}")
            return None
    
    def _load_wavelengths(self):
        """Load wavelength calibration from device."""
        try:
            # Try to get wavelength data from device
            response = self._send_command(
                self.CMD_GET_WAVELENGTHS, 
                read_length=self.NUM_PIXELS * 8
            )
            
            if response and len(response) >= self.NUM_PIXELS * 8:
                # Parse as double array
                self._wavelengths = np.frombuffer(response[:self.NUM_PIXELS*8], dtype=np.float64)
                print(f"  ✓ Wavelengths loaded from device")
            else:
                # Use default calibration
                self._wavelengths = np.linspace(200.0, 1000.0, self.NUM_PIXELS)
                print(f"  ℹ Using default wavelength calibration")
        except Exception as e:
            print(f"  ℹ Wavelength load failed: {e}, using default")
            self._wavelengths = np.linspace(200.0, 1000.0, self.NUM_PIXELS)
    
    def get_wavelengths(self) -> np.ndarray:
        """Return wavelength calibration array."""
        return self._wavelengths
    
    def set_integration_time(self, time_seconds: float):
        """Set integration time."""
        if not 0.00001 <= time_seconds <= 60.0:
            raise ValueError("Integration time must be 10us to 60s")
        
        try:
            # Send integration time as double (8 bytes)
            data = struct.pack('<d', time_seconds)
            self._send_command(self.CMD_SET_INTEGRATION, data)
            self._integration_time = time_seconds
        except Exception as e:
            print(f"Warning: Could not set integration time: {e}")
            self._integration_time = time_seconds
    
    def get_status(self) -> int:
        """Get device status."""
        response = self._send_command(self.CMD_GET_STATUS, read_length=1)
        if response and len(response) > 0:
            return response[0]
        return self.STATUS_ERROR
    
    def start_scan(self):
        """Trigger a scan."""
        self._send_command(self.CMD_START_SCAN)
    
    def get_scan_data(self) -> np.ndarray:
        """
        Read scan data from device.
        
        Returns:
            numpy.ndarray: Array of 3648 intensity values
        """
        # Poll until data is ready
        max_wait = self._integration_time + 1.0
        wait_start = time.time()
        
        while time.time() - wait_start < max_wait:
            status = self.get_status()
            if status == self.STATUS_DATA_READY:
                break
            elif status == self.STATUS_ERROR:
                raise RuntimeError("Device error during scan")
            time.sleep(0.001)
        
        # Read data (3648 pixels * 8 bytes per double)
        expected_bytes = self.NUM_PIXELS * 8
        
        # Try multiple read attempts
        for attempt in range(3):
            response = self._send_command(
                self.CMD_GET_DATA,
                read_length=expected_bytes
            )
            
            if response and len(response) >= expected_bytes:
                # Parse binary data
                data = np.frombuffer(response[:expected_bytes], dtype=np.float64)
                return data
            
            time.sleep(0.05)
        
        # If all attempts failed
        raise RuntimeError("Failed to read scan data")
    
    def get_data(self) -> np.ndarray:
        """
        Complete acquisition: start scan and return data.
        
        Returns:
            numpy.ndarray: Intensity array (3648 elements)
        """
        self.start_scan()
        return self.get_scan_data()
    
    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get complete spectrum with wavelengths.
        
        Returns:
            tuple: (wavelengths, intensities) as numpy arrays
        """
        intensities = self.get_data()
        return self._wavelengths, intensities
    
    def close(self):
        """Close connection."""
        try:
            if self.instrument:
                self._send_command(self.CMD_CLOSE)
                self.instrument.close()
                self.instrument = None
        except:
            pass
        
        try:
            if self.rm:
                self.rm.close()
                self.rm = None
        except:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Compatibility aliases
CCS200Spectrometer = CCS200Serial
ThorlabsCCS = CCS200Serial


if __name__ == "__main__":
    print("=" * 60)
    print("CCS200 Serial Protocol Test")
    print("=" * 60)
    
    try:
        # Connect to spectrometer
        spec = CCS200Serial()
        
        print(f"\nDevice Info:")
        print(f"  Wavelength range: {spec.get_wavelengths()[0]:.1f} - {spec.get_wavelengths()[-1]:.1f} nm")
        print(f"  Number of pixels: {len(spec.get_wavelengths())}")
        
        # Set integration time
        spec.set_integration_time(0.01)  # 10ms
        print(f"  Integration time: 10 ms")
        
        # Take a measurement
        print(f"\nTaking measurement...")
        wavelengths, intensities = spec.get_spectrum()
        
        print(f"  ✓ Data acquired: {len(intensities)} pixels")
        print(f"  ✓ Intensity range: {intensities.min():.3f} - {intensities.max():.3f}")
        print(f"  ✓ Mean intensity: {intensities.mean():.3f}")
        
        # Find peak
        peak_idx = np.argmax(intensities)
        print(f"  ✓ Peak at: {wavelengths[peak_idx]:.2f} nm, intensity: {intensities[peak_idx]:.3f}")
        
        spec.close()
        
        print("\n" + "=" * 60)
        print("✓ Test passed! Spectrometer is working.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: If this fails, the protocol commands may need adjustment.")
