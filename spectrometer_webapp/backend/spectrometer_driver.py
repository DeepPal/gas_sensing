"""
CCS200 Real-Time Acquisition - Production Version
--------------------------------------------------
Complete real-time data acquisition system with multiple interface options:
1. Native DLL (TLCCS_64.dll) - Best performance, requires USBTMC driver
2. Serial Binary Protocol - Works with serial port mode
3. PyVISA Raw - Direct VISA communication

Usage:
    python ccs200_realtime.py

Features:
    - Real-time spectral data acquisition
    - Configurable integration time (10us to 60s)
    - Live intensity monitoring at target wavelength
    - CSV data logging
    - Signal processing pipeline hooks
"""
import sys
import os
import time
import threading
import queue
import csv
from datetime import datetime
from typing import Callable, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, defaultdict

class RealtimeAcquisitionService:
    """
    Real-time acquisition service for CCS200 spectrometer.
    
    Provides continuous data acquisition with configurable:
    - Integration time
    - Target wavelength monitoring
    - Data buffering
    - Callback hooks for signal processing
    """
    
    def __init__(self, integration_time_ms: float = 50.0, target_wavelength: float = 808.0,
                 resource_string: Optional[str] = None):
        """
        Initialize acquisition service.
        
        Args:
            integration_time_ms: Integration time in milliseconds (10us - 60s)
            target_wavelength: Wavelength to monitor in nm
        """
        self.integration_time_ms = integration_time_ms
        self.target_wavelength = target_wavelength
        self.resource_string = resource_string
        self.spectrometer = None
        self.wavelengths = None
        self.target_idx = None
        self.dark_spectrum: Optional[np.ndarray] = None
        self.reference_spectrum: Optional[np.ndarray] = None
        self.normalization_enabled = False
        
        self.running = False
        self.sample_count = 0
        self.data_buffer = deque(maxlen=5000)  # Prevent memory leaks
        self.callbacks: List[Callable] = []
        self._thread: Optional[threading.Thread] = None
        self._buffer_lock = threading.Lock()
        self._last_sample_time: Optional[float] = None
        self._interface_used: Optional[str] = None
        self._restart_lock = threading.Lock()
        self._restart_requested = False
        self._restart_reason: Optional[str] = None
        self._health_thread: Optional[threading.Thread] = None
        self._health_running = False
        self.health_check_interval = 2.0  # seconds
        self.max_silence_seconds = max(1.0, 5 * (self.integration_time_ms / 1000.0))
        
        # Statistics
        self.stats = {
            'dropped_samples': 0,
            'avg_acquisition_time': 0.0,
            'start_time': None
        }
        
    def connect(self, interface: str = 'auto'):
        """
        Connect to spectrometer.
        
        Args:
            interface: 'auto', 'native', 'linux', 'serial', or 'pyvisa'
        """
        print(f"Connecting to CCS200 (interface: {interface})...")
        
        if interface == 'auto':
            # Detect platform and try appropriate interfaces
            is_linux = sys.platform.startswith('linux')
            if is_linux:
                # On Linux (including RPi), try linux interface first
                iface_order = ['linux', 'pyvisa', 'serial']
            else:
                # On Windows, try native DLL first
                iface_order = ['native', 'pyvisa', 'serial']
            
            for iface in iface_order:
                try:
                    self._try_interface(iface)
                    if self.spectrometer:
                        print(f"  ✓ Connected via {iface}")
                        break
                except Exception as e:
                    print(f"  ✗ {iface} failed: {str(e)[:60]}")
                    continue
        else:
            self._try_interface(interface)
        
        if not self.spectrometer:
            raise RuntimeError("Could not connect to spectrometer via any interface")
        
        self._interface_used = interface if interface != 'auto' else (self._interface_used or 'native')

        # Get wavelength calibration
        self.wavelengths = self.spectrometer.get_wavelengths()
        self.target_idx = np.argmin(np.abs(self.wavelengths - self.target_wavelength))
        
        print(f"  ✓ Wavelength range: {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm")
        print(f"  ✓ Target: {self.wavelengths[self.target_idx]:.2f} nm (index {self.target_idx})")
        print(f"  ✓ Pixels: {len(self.wavelengths)}")
        
    def _try_interface(self, interface: str):
        """Attempt to connect with specific interface."""
        if interface == 'native':
            from .ccs200_native import CCS200Spectrometer
            attempts = 2
            last_exc: Optional[Exception] = None
            for attempt in range(attempts):
                try:
                    self.spectrometer = CCS200Spectrometer(resource_string=self.resource_string)
                    break
                except RuntimeError as exc:
                    last_exc = exc
                    if attempt == 0:
                        if self._should_retry_native(exc):
                            self._attempt_native_reset()
                            continue
                    raise
            if self.spectrometer is None and last_exc:
                raise last_exc
        elif interface == 'linux':
            # Linux USBTMC interface for Raspberry Pi and other Linux systems
            from .ccs200_linux import CCS200Linux
            serial = None
            if self.resource_string:
                # Extract serial from resource string like "USB0::0x1313::0x8089::M00840499::RAW"
                parts = self.resource_string.split('::')
                for part in parts:
                    if part.startswith('M') or part not in ('USB0', 'USB', 'RAW'):
                        serial = part
                        break
            self.spectrometer = CCS200Linux(serial_number=serial)
        elif interface == 'serial':
            from .ccs200_serial import CCS200Serial
            self.spectrometer = CCS200Serial()
        elif interface == 'pyvisa':
            from .ccs200_visa_interface import CCS200VISA
            self.spectrometer = CCS200VISA()
        else:
            raise ValueError(f"Unknown interface: {interface}")
    
    def register_callback(self, callback: Callable):
        """Register a callback function to receive samples."""
        self.callbacks.append(callback)
        
    def unregister_callback(self, callback: Callable):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def _acquisition_loop(self):
        """Main acquisition thread."""
        self.stats['start_time'] = time.time()
        
        while self.running:
            t_start = time.time()
            
            try:
                intensities = self.spectrometer.get_data()
                if intensities is None or len(intensities) == 0:
                    raise ValueError("Empty or invalid spectral data received")
                
                intensities = self._apply_normalization(intensities)
                t_acquisition = time.time() - t_start
                
                # Validate data quality
                if np.any(np.isnan(intensities)) or np.any(np.isinf(intensities)):
                    raise ValueError("Invalid spectral data (NaN/Inf detected)")
                
                if np.max(intensities) <= 0:
                    raise ValueError("All intensities are zero or negative")
                
                sample = {
                    'timestamp': t_start,
                    'sample_num': self.sample_count,
                    'wavelengths': self.wavelengths,  # Include wavelengths for pipeline
                    'intensities': intensities,
                    'target_intensity': float(intensities[self.target_idx]),
                    'integration_ms': self.integration_time_ms,
                    'acquisition_time_ms': t_acquisition * 1000
                }
                
                self.sample_count += 1
                self.stats['avg_acquisition_time'] = (
                    0.9 * self.stats['avg_acquisition_time'] + 0.1 * t_acquisition
                )
                
                with self._buffer_lock:
                    self.data_buffer.append(sample)
                    # deque handles automatic cleanup
                
                for cb in self.callbacks:
                    try:
                        cb(sample)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                elapsed = time.time() - t_start
                target_period = self.integration_time_ms / 1000.0 + 0.015
                sleep_time = max(0, target_period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.stats['dropped_samples'] += 1
                    
            except Exception as e:
                self.stats['dropped_samples'] += 1
                self._request_restart(f"Acquisition error: {e}")
                time.sleep(0.05)

            # Handle pending restart requests outside the try/except to avoid swallowing
            if self._restart_requested:
                self._perform_restart()
    
    def start(self):
        """Start acquisition thread."""
        if not self.spectrometer:
            raise RuntimeError("Not connected")
            
        self.running = True
        self._thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._thread.start()
        self._start_health_watchdog()
        
        rate = 1000.0 / self.integration_time_ms
        print(f"\n✓ Acquisition started at {rate:.1f} Hz")
        print(f"  Target wavelength: {self.target_wavelength} nm")
        
    def stop(self):
        """Stop acquisition."""
        self.running = False
        self._stop_health_watchdog()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.spectrometer:
            self.spectrometer.close()
        print("\n✓ Acquisition stopped")

    def _apply_normalization(self, intensities):
        if not self.normalization_enabled or self.dark_spectrum is None or self.reference_spectrum is None:
            return intensities

        if not isinstance(intensities, np.ndarray):
            intensities = np.array(intensities, dtype=np.float64)

        dark = self.dark_spectrum
        reference = self.reference_spectrum
        denom = np.maximum(reference - dark, 1e-9)
        return (intensities - dark) / denom

    def get_latest_sample(self) -> Optional[dict]:
        """Get most recent sample."""
        with self._buffer_lock:
            if self.data_buffer:
                return self.data_buffer[-1]
        return None
    
    def get_buffer(self, n: Optional[int] = None) -> list:
        """Get buffered samples."""
        with self._buffer_lock:
            if n is None:
                return list(self.data_buffer)
            return list(self.data_buffer[-n:])
    
    def save_data(self, filename: Optional[str] = None, full_spectrum: bool = False):
        """Save buffered data to CSV."""
        import csv
        import os
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ccs200_spectrum_{timestamp}.csv"
        
        csv_path = os.path.join(os.getcwd(), os.path.basename(filename))
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                if full_spectrum:
                    writer = csv.writer(csvfile)
                    header = ['sample_num', 'timestamp', 'target_wavelength_nm', 'target_intensity']
                    header.extend([f'pixel_{i}' for i in range(len(self.wavelengths))])
                    writer.writerow(header)
                    
                    for d in self.data_buffer:
                        row = [d['sample_num'], d['timestamp'], self.wavelengths[self.target_idx], d['target_intensity']]
                        row.extend(d['intensities'].tolist())
                        writer.writerow(row)
                else:
                    writer = csv.writer(csvfile)
                    header = ['sample_num', 'timestamp', 'target_wavelength_nm', 'target_intensity', 
                              'mean_intensity', 'max_intensity', 'min_intensity']
                    writer.writerow(header)
                    
                    for d in self.data_buffer:
                        intensities = d['intensities']
                        row = [d['sample_num'], d['timestamp'], 
                               self.wavelengths[self.target_idx], d['target_intensity'],
                               float(intensities.mean()), float(intensities.max()), float(intensities.min())]
                        writer.writerow(row)
            
            print(f"✓ Data saved to: {csv_path}")
            print(f"  Samples: {len(self.data_buffer)} | Mode: {'Full spectrum' if full_spectrum else 'Compact'}")
        except PermissionError:
            print(f"  Permission denied for {csv_path}, trying temp location...")
            import tempfile
            temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(filename))
            with open(temp_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = ['sample_num', 'timestamp', 'target_wavelength_nm', 'target_intensity']
                writer.writerow(header)
                for d in self.data_buffer:
                    row = [d['sample_num'], d['timestamp'], self.wavelengths[self.target_idx], d['target_intensity']]
                    writer.writerow(row)
            print(f"✓ Data saved to temp: {temp_path} ({len(self.data_buffer)} samples)")
        except Exception as e:
            print(f"✗ Failed to save data: {e}")

    # ------------------------------------------------------------------
    # Calibration / normalization helpers
    # ------------------------------------------------------------------
    def capture_average_spectrum(self, num_samples: int = 20, settle_ms: float = 0.0) -> np.ndarray:
        if not self.spectrometer:
            raise RuntimeError("Spectrometer must be connected before capturing spectra")

        spectra = []
        for _ in range(max(1, num_samples)):
            data = self.spectrometer.get_data()
            spectra.append(np.array(data, dtype=np.float64))
            if settle_ms > 0:
                time.sleep(settle_ms / 1000.0)

        return np.mean(np.stack(spectra, axis=0), axis=0)

    def configure_normalization(self, dark: np.ndarray, reference: np.ndarray):
        if dark.shape != reference.shape:
            raise ValueError("Dark/reference spectra must have identical shapes")
        if self.wavelengths is not None and dark.shape[0] != len(self.wavelengths):
            raise ValueError("Spectra length does not match instrument pixel count")

        self.dark_spectrum = np.array(dark, dtype=np.float64)
        self.reference_spectrum = np.array(reference, dtype=np.float64)
        self.normalization_enabled = True

    def disable_normalization(self):
        self.dark_spectrum = None
        self.reference_spectrum = None
        self.normalization_enabled = False
    
    def print_stats(self):
        """Print acquisition statistics."""
        duration = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        print(f"\n{'='*60}")
        print("Acquisition Statistics")
        print(f"{'='*60}")
        print(f"  Total samples:    {self.sample_count}")
        print(f"  Duration:         {duration:.1f} s")
        print(f"  Avg rate:           {self.sample_count/max(duration,0.001):.1f} Hz")
        print(f"  Dropped samples:  {self.stats['dropped_samples']}")
        print(f"  Avg acq time:     {self.stats['avg_acquisition_time']*1000:.1f} ms")
        print(f"{'='*60}")

    def auto_select_wavelength(self, approx_wavelength: Optional[float] = None, window_nm: float = 5.0):
        """Auto-detect target wavelength by scanning a fresh spectrum.

        Args:
            approx_wavelength: Optional center wavelength to limit the search window.
            window_nm: Width (±) around the approx wavelength to search. Ignored if approx is None.
        """
        if not self.spectrometer or self.wavelengths is None:
            raise RuntimeError("Spectrometer must be connected before auto-selecting wavelength")

        spectrum = self.spectrometer.get_data()
        search_mask = np.ones_like(self.wavelengths, dtype=bool)

        if approx_wavelength is not None:
            delta = np.abs(self.wavelengths - approx_wavelength)
            search_mask = delta <= max(window_nm, 0.1)
            if not search_mask.any():
                print(f"  ⚠ No pixels within ±{window_nm} nm of {approx_wavelength} nm; scanning full range")
                search_mask = np.ones_like(self.wavelengths, dtype=bool)

        candidate_indices = np.where(search_mask)[0]
        if candidate_indices.size == 0:
            raise RuntimeError("Auto wavelength search produced no candidates")

        local_idx = np.argmax(spectrum[candidate_indices])
        target_idx = int(candidate_indices[local_idx])
        self.target_idx = target_idx
        self.target_wavelength = float(self.wavelengths[target_idx])

        print(
            f"  ✓ Auto-selected wavelength: {self.target_wavelength:.2f} nm (index {self.target_idx})"
        )

    def _should_retry_native(self, exc: Exception) -> bool:
        message = str(exc)
        return "-1073807343" in message or "resource is locked" in message.lower()

    def _attempt_native_reset(self):
        resource = self.resource_string
        if not resource:
            print("  ⚠ Cannot auto-reset device without resource string")
            return

        print(f"  ⚠ Detected locked device ({resource}). Attempting VISA clear...")
        try:
            import pyvisa  # type: ignore

            rm = pyvisa.ResourceManager()
            inst = rm.open_resource(resource)
            try:
                inst.clear()
            except Exception:
                pass
            try:
                inst.close()
            except Exception:
                pass
            try:
                rm.close()
            except Exception:
                pass
            print("  ✓ Issued device clear; retrying connection in 1s")
        except Exception as reset_exc:
            print(f"  ⚠ VISA clear failed: {reset_exc}")

        time.sleep(1.0)

    # ---------------------- Health Watchdog ----------------------
    def _start_health_watchdog(self):
        self._health_running = True
        self._health_thread = threading.Thread(target=self._health_watchdog_loop, daemon=True)
        self._health_thread.start()

    def _stop_health_watchdog(self):
        self._health_running = False
        if self._health_thread:
            self._health_thread.join(timeout=2.0)

    def _health_watchdog_loop(self):
        while self._health_running:
            time.sleep(self.health_check_interval)
            if not self.running:
                continue
            last = self._last_sample_time
            if last is None:
                continue
            silence = time.time() - last
            if silence > self.max_silence_seconds:
                self._request_restart(f"No samples for {silence:.1f}s")

    def _request_restart(self, reason: str):
        with self._restart_lock:
            if not self._restart_requested:
                self._restart_requested = True
                self._restart_reason = reason

    def _perform_restart(self):
        with self._restart_lock:
            if not self._restart_requested:
                return
            reason = self._restart_reason or "Unknown"
            self._restart_requested = False
            self._restart_reason = None
        print(f"\n[Watchdog] Restarting spectrometer due to: {reason}")
        try:
            if self.spectrometer:
                try:
                    self.spectrometer.close()
                except Exception:
                    pass
                time.sleep(0.2)
            interface = self._interface_used or 'native'
            self._try_interface(interface)
            self.wavelengths = self.spectrometer.get_wavelengths()
            self.target_idx = np.argmin(np.abs(self.wavelengths - self.target_wavelength))
            self._last_sample_time = time.time()
            print(f"[Watchdog] Restart successful via {interface}")
        except Exception as exc:
            print(f"[Watchdog] Restart failed: {exc}")
            time.sleep(1.0)


def create_console_visualizer(service, window_size: int = 50):
    """Create a console-based sparkline visualizer."""
    history = []
    
    def visualizer(sample):
        nonlocal history
        history.append(sample['target_intensity'])
        if len(history) > window_size:
            history = history[-window_size:]
        
        # Update every 50 samples to reduce console lag (was 10)
        if sample['sample_num'] % 50 == 0:
            max_val = max(history) if history else 1
            min_val = min(history) if history else 0
            range_val = max_val - min_val if max_val != min_val else 1
            
            line = ""
            for v in history[-50:]:
                idx = int(7 * (v - min_val) / range_val)
                chars = " _▁▂▃▄▅▆▇"
                line += chars[min(idx, 7)]
            
            print(f"\r[{sample['sample_num']:6d}] λ={service.target_wavelength:.1f}nm I={sample['target_intensity']:10.5f} {line}", end="", flush=True)
    
    return visualizer


class RealTimePlotter:
    """Real-time matplotlib plot for live intensity visualization."""
    
    def __init__(self, service, max_points: int = 500):
        self.service = service
        self.max_points = max_points
        self.timestamps = deque(maxlen=max_points)
        self.intensities = deque(maxlen=max_points)
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.line, = self.ax.plot([], [], 'b-', linewidth=1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Intensity')
        self.ax.set_title(f'Real-Time Intensity at {service.target_wavelength:.1f} nm')
        self.ax.grid(True, alpha=0.3)
        self.start_time = None
        
    def update(self, sample):
        """Called for each new sample."""
        if self.start_time is None:
            self.start_time = sample['timestamp']
        
        rel_time = sample['timestamp'] - self.start_time
        self.timestamps.append(rel_time)
        self.intensities.append(sample['target_intensity'])
        
    def animate(self, frame):
        """Animation update function."""
        if len(self.timestamps) > 0:
            self.line.set_data(list(self.timestamps), list(self.intensities))
            self.ax.set_xlim(max(0, self.timestamps[-1] - 10), self.timestamps[-1] + 0.5)
            if len(self.intensities) > 0:
                y_min = min(self.intensities) * 0.9
                y_max = max(self.intensities) * 1.1
                if y_min != y_max:
                    self.ax.set_ylim(y_min, y_max)
        return self.line,
    
    def show(self):
        """Start the plot (blocking)."""
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, interval=50, blit=False, cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the plot."""
        plt.close(self.fig)


def main():
    """Main entry point."""
    print("=" * 70)
    print("CCS200 Real-Time Spectral Acquisition")
    print("=" * 70)
    print()
    
    # Configuration - STABLE acquisition at 532 nm (30ms = ~33 Hz)
    INTEGRATION_MS = 30.0  # 30ms gives stable ~33 Hz (device needs ~20ms + overhead)
    TARGET_WAVELENGTH = 532.0  # nm
    
    # Create service
    service = RealtimeAcquisitionService(
        integration_time_ms=INTEGRATION_MS,
        target_wavelength=TARGET_WAVELENGTH
    )
    
    # Initialize variables
    plotter = None
    csv_file = None
    csv_path = None
    csv_writer = None
    
    try:
        # Connect (auto-detect interface)
        service.connect(interface='auto')
        
        # Setup visualizers
        visualizer = create_console_visualizer(service)
        service.register_callback(visualizer)
        
        # Setup live matplotlib plot
        enable_live_plot = True  # Set to False to disable live plot
        plotter = None
        plot_thread = None
        if enable_live_plot:
            plotter = RealTimePlotter(service, max_points=500)
            service.register_callback(plotter.update)
            # Run plot in separate thread
            plot_thread = threading.Thread(target=plotter.show, daemon=True)
            plot_thread.start()
            print("✓ Live plot started (matplotlib)")
        
        # Optional: Enable real-time CSV logging
        log_to_csv = True  # Set to False to disable real-time logging
        csv_file = None
        csv_writer = None
        
        if log_to_csv:
            import csv
            csv_filename = f"ccs200_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = os.path.join(os.getcwd(), csv_filename)
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            # Write header
            csv_writer.writerow(['sample_num', 'timestamp', 'target_wavelength_nm', 'target_intensity',
                                'mean_intensity', 'max_intensity', 'min_intensity'])
            csv_file.flush()
            print(f"✓ Real-time logging to: {csv_path}")
        
        def file_logger(sample):
            # Log to CSV in real-time
            if log_to_csv and csv_writer:
                intensities = sample['intensities']
                row = [sample['sample_num'], sample['timestamp'],
                       service.wavelengths[service.target_idx], sample['target_intensity'],
                       float(intensities.mean()), float(intensities.max()), float(intensities.min())]
                csv_writer.writerow(row)
                if sample['sample_num'] % 10 == 0:  # Flush every 10 samples
                    csv_file.flush()
            
            # Console log every 500 samples
            if sample['sample_num'] % 500 == 0:
                print(f"\n[Log] Sample {sample['sample_num']}: I={sample['target_intensity']:.5f}")
        
        service.register_callback(file_logger)
        
        # Start acquisition
        service.start()
        
        print(f"\n{'='*70}")
        print("Acquisition running... Press Ctrl+C to stop")
        print(f"{'='*70}\n")
        
        # Run until interrupted
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        service.stop()
        service.print_stats()
        
        # Close plot if running
        if plotter:
            plotter.close()
        
        # Close real-time CSV file if open
        if csv_file:
            csv_file.close()
            print(f"✓ Closed real-time log file")
        
        try:
            service.save_data(full_spectrum=False)
        except KeyboardInterrupt:
            print("\nSave interrupted, skipping...")
        except Exception as e:
            print(f"\nSave error: {e}")
        
        print("\n✓ Done")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
