"""
CCS200 Native DLL Interface
----------------------------
Uses TLCCS_64.dll with the correct calling convention for getScanData.

Key finding: tlccs_getScanData expects  ViReal64*  (double*), NOT
a pointer-to-array.  Passing the ctypes array object directly (not
via byref) lets ctypes auto-convert it to a pointer to the first
element, which is what the DLL expects.

Acquisition pattern:
    tlccs_startScan(handle)
    sleep(integration_time + 0.25)      # wait for CCD + USB transfer
    tlccs_getScanData(handle, buf)      # buf is (c_double * 3648)()
"""

import ctypes
from ctypes import byref, c_char_p, c_double, c_int, c_void_p
import os
import time

import numpy as np


class CCS200Spectrometer:
    """CCS200 interface via native TLCCS_64.dll."""

    NUM_PIXELS = 3648
    DEFAULT_INTEGRATION_S = 0.05  # 50 ms

    def __init__(self, resource_string=None, integration_time_s=None):
        self.handle = c_int(0)
        self._integration_time_s = integration_time_s or self.DEFAULT_INTEGRATION_S
        self._load_dll()
        self._init_device(resource_string)
        self._wavelengths: np.ndarray = np.linspace(200, 1000, self.NUM_PIXELS)
        self._load_wavelengths()
        self.set_integration_time(self._integration_time_s)
        time.sleep(0.15)  # allow device to apply new integration time
        self._warmup_scan()  # drain any stale state from previous sessions

    # ------------------------------------------------------------------
    # DLL loading
    # ------------------------------------------------------------------

    def _load_dll(self):
        dll_path = r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll"
        dll_dir = os.path.dirname(dll_path)
        prev_cwd = os.getcwd()
        try:
            os.chdir(dll_dir)
            self.dll = ctypes.CDLL(dll_path)
        finally:
            os.chdir(prev_cwd)

        # Typed signatures for functions whose args/restype matter
        self.dll.tlccs_init.argtypes = [c_char_p, c_int, c_int, ctypes.POINTER(c_int)]
        self.dll.tlccs_init.restype = c_int

        self.dll.tlccs_close.argtypes = [c_int]
        self.dll.tlccs_close.restype = c_int

        self.dll.tlccs_setIntegrationTime.argtypes = [c_int, c_double]
        self.dll.tlccs_setIntegrationTime.restype = c_int

        self.dll.tlccs_startScan.argtypes = [c_int]
        self.dll.tlccs_startScan.restype = c_int

        self.dll.tlccs_startScanCont.argtypes = [c_int]
        self.dll.tlccs_startScanCont.restype = c_int

        self.dll.tlccs_getDeviceStatus.argtypes = [c_int, ctypes.POINTER(c_int)]
        self.dll.tlccs_getDeviceStatus.restype = c_int

        self.dll.tlccs_getWavelengthData.argtypes = [
            c_int,
            c_int,
            ctypes.POINTER(c_double * self.NUM_PIXELS),
            c_void_p,
            c_void_p,
        ]
        self.dll.tlccs_getWavelengthData.restype = c_int

        # getScanData: ViReal64* (double*) — NO argtypes so ctypes passes
        # the array object as a raw pointer to its first element.
        self.dll.tlccs_getScanData.restype = c_int

    # ------------------------------------------------------------------
    # Device init
    # ------------------------------------------------------------------

    def _init_device(self, resource_string=None):
        if resource_string is None:
            resource_string = "USB0::0x1313::0x8089::M00505929::RAW"
            print(f"  Using detected device: {resource_string}")
        rsrc = resource_string.encode() if isinstance(resource_string, str) else resource_string
        print("  Initializing...")
        result = self.dll.tlccs_init(rsrc, 1, 1, byref(self.handle))
        if result != 0:
            raise RuntimeError(f"tlccs_init failed with code: {result}")
        print(f"  [OK] Device initialized (handle: {self.handle.value})")

    def _load_wavelengths(self):
        buf = (c_double * self.NUM_PIXELS)()
        result = self.dll.tlccs_getWavelengthData(self.handle.value, 0, buf, None, None)
        if result == 0:
            self._wavelengths = np.array(list(buf))
        else:
            self._wavelengths = np.linspace(200, 1000, self.NUM_PIXELS)

    def _warmup_scan(self):
        """Fire one throwaway scan to leave device in a known-good idle state.

        This drains any stale data from a previous VISA session that crashed
        without calling close() and did not consume pending scan data.
        Errors are intentionally swallowed — we only care that the device is
        idle by the time this returns.
        """
        self.dll.tlccs_startScan(self.handle.value)
        time.sleep(max(self._integration_time_s, 0.05) + 0.40)
        buf = (c_double * self.NUM_PIXELS)()
        self.dll.tlccs_getScanData(self.handle.value, buf)  # ignore result
        time.sleep(0.15)  # final settle before first real read

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wavelengths(self):
        return self._wavelengths

    def set_integration_time(self, time_seconds: float):
        """Set CCD integration time in seconds."""
        self._integration_time_s = float(time_seconds)
        result = self.dll.tlccs_setIntegrationTime(
            self.handle.value, c_double(self._integration_time_s)
        )
        if result != 0:
            raise RuntimeError(f"Failed to set integration time: {result}")

    def get_data(self) -> np.ndarray:
        """Acquire one spectrum.

        Pattern:
            tlccs_startScan(handle)
            Poll tlccs_getDeviceStatus until scan is complete (bit 1 set)
            tlccs_getScanData(handle, buf)
            sleep(0.10)     # device idle cool-down

        Retries up to MAX_RETRIES times if getScanData returns a non-zero
        code (e.g. 0xBFFE0000 = scan not ready / scan pending).
        """
        MAX_RETRIES = 3
        # ThorLabs device status bits
        TLCCS_STATUS_SCAN_IDLE = 0x0002  # scan complete, data available
        POLL_INTERVAL = 0.02  # 20 ms polling step
        POLL_TIMEOUT = max(self._integration_time_s * 3, 2.0)

        for retry in range(MAX_RETRIES):
            r = self.dll.tlccs_startScan(self.handle.value)
            if r != 0:
                raise RuntimeError(f"tlccs_startScan failed: {r}")

            # --- base wait: integration time + fixed USB-transfer margin ---
            time.sleep(self._integration_time_s + 0.30)

            # --- poll until device reports scan complete ---
            status = c_int(0)
            elapsed = 0.0
            while elapsed < POLL_TIMEOUT:
                ret = self.dll.tlccs_getDeviceStatus(self.handle.value, ctypes.byref(status))
                if ret != 0 or (status.value & TLCCS_STATUS_SCAN_IDLE):
                    break  # device idle / data ready (or status call failed)
                time.sleep(POLL_INTERVAL)
                elapsed += POLL_INTERVAL

            buf = (c_double * self.NUM_PIXELS)()
            r = self.dll.tlccs_getScanData(self.handle.value, buf)
            if r == 0:
                break  # success
            if retry < MAX_RETRIES - 1:
                # 0xBFFE0000 = scan not ready -- back off and retry
                time.sleep(0.10 * (retry + 1))
            else:
                raise RuntimeError(f"tlccs_getScanData failed: {r}")

        # Let the device's internal state machine return to idle
        time.sleep(0.10)
        return np.array(list(buf), dtype=np.float64)

    # Legacy alias
    def get_data_sync(self):
        return self.get_data()

    def close(self):
        if self.handle.value != 0:
            self.dll.tlccs_close(self.handle.value)
            self.handle.value = 0


# Compatibility aliases
ThorlabsCCS = CCS200Spectrometer


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Testing CCS200 Native Interface...")
    spec = None
    try:
        spec = CCS200Spectrometer()
        wl = spec.get_wavelengths()
        print(f"Wavelength range: {wl[0]:.1f} - {wl[-1]:.1f} nm")

        spec.set_integration_time(0.05)

        N = 10
        print(f"\nAcquiring {N} consecutive spectra (integration=50 ms)...")
        peaks = []
        for i in range(N):
            t0 = time.monotonic()
            data = spec.get_data()
            elapsed = time.monotonic() - t0
            peak_idx = int(np.argmax(data))
            peaks.append(data[peak_idx])
            print(
                f"  [{i + 1:2d}] pixels={len(data)}  "
                f"min={data.min():.4f}  max={data.max():.4f}  "
                f"peak_wl={wl[peak_idx]:.1f} nm  "
                f"frame={elapsed * 1000:.0f} ms"
            )

        peaks_arr = np.array(peaks)
        rsd = peaks_arr.std() / peaks_arr.mean() * 100 if peaks_arr.mean() != 0 else float("nan")
        print("\nRepeatability (peak intensity):")
        print(f"  mean={peaks_arr.mean():.4f}  std={peaks_arr.std():.4f}  RSD={rsd:.2f}%")
        print(f"\nPASS -- all {N} reads succeeded")

    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if spec is not None:
            spec.close()
