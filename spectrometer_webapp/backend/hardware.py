import time
import numpy as np
import threading
from typing import Optional, Tuple, List

try:
    from spectrometer_driver import RealtimeAcquisitionService
    HAS_DRIVER = True
except ImportError:
    HAS_DRIVER = False

class MockSpectrometer:
    """Mock interface for an optical spectrometer hardware."""
    def __init__(self):
        self.connected = False
        self.running = False
        self.wavelengths = np.linspace(300, 1000, 500)
        self.gas_type = "Air"
        self.concentration = 0.0

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def start_acquisition(self):
        self.running = True

    def stop_acquisition(self):
        self.running = False

    def set_environment(self, gas_type: str, concentration: float):
        self.gas_type = gas_type
        self.concentration = concentration

    def set_integration_time(self, ms: float) -> None:
        """Simulate latency proportional to integration time (mock only)."""
        self._int_time_ms = max(1.0, float(ms))

    def get_spectrum(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if not self.connected:
            return None, None
        
        # Base background spectrum
        base = np.exp(-((self.wavelengths - 500) ** 2) / (2 * 100 ** 2)) * 10
        noise = np.random.normal(0, 0.1, len(self.wavelengths))
        intensity = base + noise
        
        # Simulated gas peaks
        if self.gas_type.lower() not in ["", "air", "none"]:
            peak_pos = 400 + (hash(self.gas_type) % 500)
            peak_height = 5 + (0.5 * self.concentration)
            peak = np.exp(-((self.wavelengths - peak_pos) ** 2) / (2 * 20 ** 2)) * peak_height
            intensity += peak
            
        return self.wavelengths.tolist(), intensity.tolist()

class Ccs200Hardware:
    """Real Thorlabs CCS200 Spectrometer Hardware Interface."""
    def __init__(self):
        self.service = None
        self.connected = False
        self.wavelengths = None

    def connect(self) -> bool:
        if not HAS_DRIVER:
            print("Spectrometer driver not found.")
            return False
            
        try:
            # We use 50ms integration by default for stable real-time
            self.service = RealtimeAcquisitionService(integration_time_ms=50.0)
            self.service.connect(interface='auto')
            self.wavelengths = self.service.wavelengths
            self.service.start()
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to CCS200: {e}")
            self.connected = False
            return False

    def disconnect(self):
        if self.service:
            self.service.stop()
        self.connected = False

    def set_environment(self, gas_type: str, concentration: float):
        """Simulation only - no effect on real hardware."""
        pass

    def set_integration_time(self, ms: float) -> None:
        """Apply integration time to the real CCS200 if running."""
        if self.service and self.connected:
            try:
                self.service.set_integration_time(ms)
            except Exception:
                pass

    def get_spectrum(self) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        if not self.connected or not self.service:
            return None, None
        
        sample = self.service.get_latest_sample()
        if sample and 'intensities' in sample:
            # Note: ccs200_realtime returns numpy arrays
            return self.wavelengths.tolist(), sample['intensities'].tolist()
        return None, None

def get_spectrometer(use_mock: bool = True):
    if use_mock:
        return MockSpectrometer()
    return Ccs200Hardware()
