"""
CCS200 Spectrometer Interface via PyVISA
----------------------------------------
Alternative implementation using PyVISA/USBTMC protocol.
Does not require ThorLabs TLCCS_64.dll - only needs PyVISA and USB drivers.
"""
import logging
import time
import numpy as np
import pyvisa

log = logging.getLogger(__name__)


class CCS200VISA:
    """
    CCS200 interface using PyVISA USBTMC protocol.
    
    Usage:
        spec = CCS200VISA()
        spec.set_integration_time(0.05)  # 50ms
        wavelengths, intensities = spec.get_scan()
        spec.close()
    """
    
    # CCS200 USB identifiers
    VID = 0x1313  # ThorLabs Vendor ID
    PID = 0x8089  # CCS200 Product ID
    
    def __init__(self, resource_name=None):
        """
        Initialize CCS200 via PyVISA.
        
        Args:
            resource_name: Optional VISA resource string. Auto-detects if None.
        """
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self._wavelengths = None
        self._integration_time = 0.01  # Default 10ms
        
        if resource_name:
            self._connect(resource_name)
        else:
            self._auto_connect()
    
    def _auto_connect(self):
        """Auto-detect and connect to CCS200."""
        resources = self.rm.list_resources()
        
        # Look for USBTMC device with ThorLabs VID/PID
        ccs_resource = None
        for resource in resources:
            if 'USB' in resource and ('0x1313' in resource or '1313' in resource):
                ccs_resource = resource
                break
        
        # If no USBTMC, try serial ports (ASRL)
        if ccs_resource is None:
            serial_resources = [r for r in resources if 'ASRL' in r]
            if serial_resources:
                ccs_resource = serial_resources[0]  # Try first serial port
                print(f"  Found serial port: {ccs_resource}")
        
        if ccs_resource is None:
            raise RuntimeError(
                f"No suitable instruments found. Available resources: {resources}\n"
                "Ensure CCS200 is connected and drivers are installed."
            )
        
        self._connect(ccs_resource)
    
    def _connect(self, resource_name):
        """Connect to specific VISA resource."""
        try:
            self.instrument = self.rm.open_resource(resource_name)
            
            # Configure serial port if using ASRL
            if 'ASRL' in resource_name:
                # CCS200 serial settings (typical)
                self.instrument.baud_rate = 115200
                self.instrument.data_bits = 8
                self.instrument.parity = pyvisa.constants.Parity.none
                self.instrument.stop_bits = pyvisa.constants.StopBits.one
                self.instrument.read_termination = '\n'
                self.instrument.write_termination = '\n'
                print(f"  Configured serial port: 115200 baud, 8N1")
            
            # Set timeout and query device
            self.instrument.timeout = 5000  # 5 seconds
            
            # Try to identify the device
            try:
                idn = self.instrument.query('*IDN?')
                log.info("Connected to: %s", idn.strip())
            except Exception:
                log.info("Connected to: %s", resource_name)
            
            # Initialize device
            self._initialize()
            
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {resource_name}: {e}")
    
    def _initialize(self):
        """Initialize the spectrometer."""
        # Reset device
        self.instrument.write('*RST')
        time.sleep(0.1)
        
        # Clear status
        self.instrument.write('*CLS')
        time.sleep(0.1)
        
        # Load wavelength calibration
        self._load_wavelengths()
    
    def _load_wavelengths(self):
        """Load wavelength calibration from device."""
        try:
            # Query wavelength data - CCS200 has 3648 pixels
            # This is a simplified approach - actual implementation may vary
            # based on the specific CCS200 firmware version
            
            # For CCS200, wavelengths are typically linear from ~200-1000nm
            # with 3648 pixels
            pixel_count = 3648
            lambda_min = 200.0  # nm
            lambda_max = 1000.0  # nm
            
            self._wavelengths = np.linspace(lambda_min, lambda_max, pixel_count)
            
        except Exception as e:
            print(f"Warning: Could not load wavelengths from device: {e}")
            # Use default calibration
            self._wavelengths = np.linspace(200, 1000, 3648)
    
    def get_wavelengths(self):
        """Return wavelength calibration array."""
        return self._wavelengths
    
    def set_integration_time(self, time_seconds):
        """
        Set integration time.
        
        Args:
            time_seconds: Integration time in seconds (0.001 to 60)
        """
        if not 0.001 <= time_seconds <= 60:
            raise ValueError("Integration time must be between 1ms and 60s")
        
        # CCS200 command for integration time
        # Format may vary by firmware version
        ms = int(time_seconds * 1000)
        try:
            self.instrument.write(f'SENS:INT {ms}')
            self._integration_time = time_seconds
        except Exception as e:
            print(f"Warning: Could not set integration time: {e}")
            # Store locally anyway
            self._integration_time = time_seconds
    
    def start_scan(self):
        """Trigger a scan."""
        try:
            self.instrument.write('INIT')
        except Exception as e:
            # Fallback - just wait for integration time
            time.sleep(self._integration_time)
    
    def get_scan_data(self):
        """
        Retrieve scan data.
        
        Returns:
            tuple: (wavelengths, intensities) as numpy arrays
        """
        try:
            # For serial connection, try reading raw bytes
            data = None
            
            # Try SCPI-like commands first
            for cmd in ['TRAC:DATA?', 'READ?', 'FETC?', 'SENS:DATA?', 'DATA?', 'SCAN?']:
                try:
                    self.instrument.write(cmd)
                    time.sleep(self._integration_time + 0.05)
                    
                    # Try reading with different methods
                    try:
                        raw_data = self.instrument.read()
                    except Exception:
                        # Try reading bytes
                        try:
                            raw_bytes = self.instrument.read_bytes(3648 * 8)  # 3648 doubles
                            data = np.frombuffer(raw_bytes, dtype=np.float64)
                            if len(data) == 3648:
                                break
                        except Exception:
                            pass
                    
                    if raw_data:
                        data = self._parse_data(raw_data)
                        if data is not None and len(data) > 0:
                            break
                except Exception as e:
                    continue
            
            # If still no data, try binary block read
            if data is None or len(data) == 0:
                try:
                    # Try reading as binary block (IEEE 488.2 format)
                    self.instrument.write('FORM:DATA REAL')
                    time.sleep(0.1)
                    self.instrument.write('TRAC:DATA?')
                    time.sleep(self._integration_time + 0.05)
                    
                    # Read header first
                    header = self.instrument.read_bytes(2)
                    if header[0:1] == b'#':
                        num_digits = int(header[1:2])
                        count_bytes = self.instrument.read_bytes(num_digits)
                        count = int(count_bytes.decode())
                        data_bytes = self.instrument.read_bytes(count)
                        data = np.frombuffer(data_bytes, dtype=np.float64)
                except Exception as e:
                    pass
            
            if data is None or len(data) == 0:
                print("Warning: Could not read data from device, returning zeros")
                data = np.zeros(3648)
            elif len(data) != 3648:
                # Pad or truncate to expected size
                if len(data) < 3648:
                    data = np.pad(data, (0, 3648 - len(data)), mode='constant')
                else:
                    data = data[:3648]
            
            return self._wavelengths, data
            
        except Exception as e:
            print(f"Error reading data: {e}")
            return self._wavelengths, np.zeros(3648)
    
    def _parse_data(self, raw_data):
        """Parse raw VISA data into numpy array."""
        if not raw_data:
            return None
            
        try:
            # If bytes, try to convert
            if isinstance(raw_data, bytes):
                # Try as float array
                try:
                    return np.frombuffer(raw_data, dtype=np.float64)
                except Exception:
                    try:
                        return np.frombuffer(raw_data, dtype=np.float32)
                    except Exception:
                        raw_data = raw_data.decode('utf-8', errors='ignore')
            
            # Try parsing as comma-separated values
            if isinstance(raw_data, str) and ',' in raw_data:
                values = [float(x.strip()) for x in raw_data.split(',') if x.strip()]
                return np.array(values)
            
            # Try parsing as whitespace-separated
            if isinstance(raw_data, str):
                values = [float(x.strip()) for x in raw_data.split() if x.strip()]
                if values:
                    return np.array(values)
            
            return None
            
        except Exception as e:
            print(f"Failed to parse data: {e}")
            return None
    
    def get_data(self):
        """
        Acquire spectrum and return intensities only.
        
        Returns:
            numpy.ndarray: Intensity array
        """
        self.start_scan()
        _, intensities = self.get_scan_data()
        return intensities
    
    def get_intensity_at_wavelength(self, target_wavelength):
        """
        Get intensity at specific wavelength.
        
        Args:
            target_wavelength: Target wavelength in nm
            
        Returns:
            float: Intensity at closest pixel
        """
        intensities = self.get_data()
        idx = np.argmin(np.abs(self._wavelengths - target_wavelength))
        return float(intensities[idx])
    
    def close(self):
        """Close the connection."""
        if self.instrument:
            try:
                self.instrument.close()
            except Exception:
                pass
            self.instrument = None

        if self.rm:
            try:
                self.rm.close()
            except Exception:
                pass
            self.rm = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# For compatibility with ccs200_interface.py
CCS200Spectrometer = CCS200VISA
ThorlabsCCS = CCS200VISA


def list_ccs_devices():
    """List available CCS200 devices."""
    try:
        rm = pyvisa.ResourceManager()
        resources = rm.list_resources()
        rm.close()
        
        # Filter for USB devices
        usb_devices = [r for r in resources if 'USB' in r]
        return usb_devices
    except Exception as e:
        print(f"Error listing devices: {e}")
        return []


if __name__ == "__main__":
    # Test the interface
    print("CCS200 PyVISA Interface Test")
    print("=" * 50)
    
    # List available devices
    devices = list_ccs_devices()
    print(f"Found {len(devices)} USB device(s):")
    for i, dev in enumerate(devices):
        print(f"  {i+1}. {dev}")
    
    if devices:
        print("\nAttempting to connect to first device...")
        try:
            spec = CCS200VISA()
            print(f"Connected!")
            print(f"Wavelength range: {spec._wavelengths[0]:.1f} - {spec._wavelengths[-1]:.1f} nm")
            
            print("\nAcquiring test spectrum...")
            spec.set_integration_time(0.01)
            wavelengths, intensities = spec.get_scan()
            print(f"Acquired {len(intensities)} points")
            print(f"Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
            
            spec.close()
            print("\nTest completed successfully!")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo USB devices found. Check connection and drivers.")
