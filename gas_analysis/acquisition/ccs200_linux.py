"""
CCS200 Linux Interface - USBTMC Protocol
=========================================
Pure Python interface for Thorlabs CCS200 spectrometer on Linux (including Raspberry Pi).
Uses the Linux usbtmc kernel module or pyusb for direct USB communication.

This avoids dependency on NI-VISA and Thorlabs Windows DLLs.

Requirements (Linux):
    - Linux kernel with usbtmc module (usually built-in or loadable)
    - pyusb (pip install pyusb)
    - udev rules to allow USB access (see deployment guide)

Usage:
    from src.acquisition.ccs200_linux import CCS200Linux
    spec = CCS200Linux()
    spec.set_integration_time(0.1)  # 100ms
    data = spec.get_scan_data()
    wavelengths = spec.get_wavelengths()
"""

import contextlib
import time
from typing import Optional

import numpy as np

# Try to import USB libraries
try:
    import usb.core
    import usb.util

    PYUSB_AVAILABLE = True
except ImportError:
    PYUSB_AVAILABLE = False

# CCS200 USB identifiers
THORLABS_VID = 0x1313
CCS200_PID = 0x8088
CCS200_PID_ALT = 0x8089  # Some variants use this

# USBTMC constants
USBTMC_INTERFACE = 0
USBTMC_ENDPOINT_OUT = 0x01
USBTMC_ENDPOINT_IN = 0x81

# SCPI commands for CCS200
SCPI_COMMANDS = {
    "identify": "*IDN?",
    "reset": "*RST",
    "clear": "*CLS",
    "set_integration_time": ":SENS:INT {time_s:.6f}",
    "get_integration_time": ":SENS:INT?",
    "start_scan": ":SENS:DATA:ACQ",
    "get_scan_data": ":SENS:DATA:COLD",
    "get_wavelength_data": ":SENS:DATA:WAV",
    "get_device_status": ":SENS:STAT?",
    "set_trigger_source": ":TRIG:SOUR IMM",
    "set_averaging": ":SENS:AVER {count:d}",
    "get_averaging": ":SENS:AVER?",
}


class CCS200Linux:
    """
    Linux-compatible CCS200 spectrometer interface using USBTMC.

    This class provides the same API as CCS200Spectrometer but works on
    Linux systems including Raspberry Pi without NI-VISA.
    """

    NUM_PIXELS = 3648
    VID = THORLABS_VID
    PID = CCS200_PID

    def __init__(self, serial_number: Optional[str] = None, device_index: int = 0):
        """
        Initialize CCS200 on Linux.

        Args:
            serial_number: Optional serial number to select specific device
            device_index: Index of device if multiple connected (0-based)
        """
        if not PYUSB_AVAILABLE:
            raise ImportError(
                "pyusb is required for Linux CCS200 support. Install with: pip install pyusb"
            )

        self._device = None
        self._serial_number = serial_number
        self._device_index = device_index
        self._wavelengths: Optional[np.ndarray] = None
        self._timeout_ms = 5000
        self._btag = 0

        self._connect()
        self._load_wavelengths()

    def _connect(self):
        """Find and connect to CCS200 device via USB."""
        # Find all CCS200 devices
        devices = list(usb.core.find(find_all=True, idVendor=self.VID, idProduct=CCS200_PID))

        # Also check alternate PID
        devices_alt = list(
            usb.core.find(find_all=True, idVendor=self.VID, idProduct=CCS200_PID_ALT)
        )
        devices.extend(devices_alt)

        if not devices:
            raise RuntimeError(
                f"No CCS200 spectrometer found. "
                f"Checked VID={self.VID:#x}, PIDs={CCS200_PID:#x}, {CCS200_PID_ALT:#x}. "
                f"Ensure device is connected and udev rules are configured."
            )

        # Select device by serial or index
        if self._serial_number:
            for dev in devices:
                try:
                    serial = usb.util.get_string(dev, dev.iSerialNumber)
                    if serial and self._serial_number in serial:
                        self._device = dev
                        break
                except (usb.core.USBError, ValueError):
                    continue
            if not self._device:
                raise RuntimeError(f"Device with serial '{self._serial_number}' not found")
        else:
            if self._device_index >= len(devices):
                raise RuntimeError(
                    f"Device index {self._device_index} out of range ({len(devices)} devices found)"
                )
            self._device = devices[self._device_index]

        # Store serial number
        try:
            self._serial_number = usb.util.get_string(self._device, self._device.iSerialNumber)
        except (usb.core.USBError, ValueError):
            self._serial_number = "unknown"

        # Detach kernel driver if active (usbtmc module)
        try:
            if self._device.is_kernel_driver_active(0):
                self._device.detach_kernel_driver(0)
        except (usb.core.USBError, NotImplementedError):
            pass  # May not be implemented on all systems

        # Set configuration
        try:
            self._device.set_configuration()
        except usb.core.USBError:
            # Device may already be configured
            pass

        # Claim interface
        try:
            usb.util.claim_interface(self._device, 0)
        except usb.core.USBError:
            pass  # May already be claimed

        print(f"  ✓ Connected to CCS200 (serial: {self._serial_number})")

    def _send_usbtmc_command(self, command: bytes, expect_response: bool = True) -> Optional[bytes]:
        """
        Send USBTMC command and optionally read response.

        USBTMC message format:
        - MsgID (1 byte): 0x01 for DEV_DEP_MSG_OUT, 0x02 for REQUEST_DEV_DEP_MSG_IN
        - bTag (1 byte): 1-255, incremented for each message
        - bTagInverse (1 byte): ~bTag
        - Reserved (1 byte): 0x00
        - For OUT: EOM (1 byte), followed by command
        - For IN: TransferSize (4 bytes), bmTransferAttributes (1 byte), TermChar (1 byte), Reserved (2 bytes)
        """
        self._btag = (self._btag % 255) + 1
        btag_inv = (~self._btag) & 0xFF

        # Build OUT header
        # MsgID = 0x01 (DEV_DEP_MSG_OUT)
        msg_id = 0x01
        eom = 0x01  # End of message

        # Pad command to multiple of 4 bytes for USBTMC
        cmd_padded = command + b"\n"
        padded_len = (len(cmd_padded) + 3) & ~3
        cmd_padded = cmd_padded.ljust(padded_len, b"\x00")

        header = bytes([msg_id, self._btag, btag_inv, 0x00, eom, 0x00, 0x00, 0x00])
        packet = header + cmd_padded

        # Send command
        try:
            self._device.write(USBTMC_ENDPOINT_OUT, packet, timeout=self._timeout_ms)
        except usb.core.USBError as e:
            raise RuntimeError(f"USB write error: {e}")

        if not expect_response:
            return None

        # Read response
        # MsgID = 0x02 (REQUEST_DEV_DEP_MSG_IN)
        msg_id_in = 0x02
        transfer_size = 32768  # Max bytes to read
        header_in = bytes(
            [
                msg_id_in,
                self._btag,
                btag_inv,
                0x00,
                transfer_size & 0xFF,
                (transfer_size >> 8) & 0xFF,
                (transfer_size >> 16) & 0xFF,
                (transfer_size >> 24) & 0xFF,
                0x01,  # bmTransferAttributes: EOM
                0x0A,  # TermChar (newline)
                0x00,
                0x00,
            ]
        )

        try:
            self._device.write(USBTMC_ENDPOINT_OUT, header_in, timeout=self._timeout_ms)
        except usb.core.USBError as e:
            raise RuntimeError(f"USB write error: {e}")

        # Read response data
        try:
            data = self._device.read(USBTMC_ENDPOINT_IN, transfer_size, timeout=self._timeout_ms)
        except usb.core.USBError as e:
            raise RuntimeError(f"USB read error: {e}")

        # Parse USBTMC response header (12 bytes)
        if len(data) < 12:
            return b""

        # Skip USBTMC header
        response = bytes(data[12:])

        # Remove trailing nulls and terminator
        response = response.rstrip(b"\x00\n\r")

        return response

    def _send_scpi(self, command: str, expect_response: bool = True) -> Optional[str]:
        """Send SCPI command and return response as string."""
        response = self._send_usbtmc_command(command.encode(), expect_response)
        if response is None:
            return None
        return response.decode("utf-8", errors="replace").strip()

    def _send_scpi_binary(self, command: str) -> bytes:
        """Send SCPI command and return binary response."""
        return self._send_usbtmc_command(command.encode(), expect_response=True) or b""

    def identify(self) -> str:
        """Get device identification string."""
        return self._send_scpi(SCPI_COMMANDS["identify"]) or "Unknown"

    def reset(self):
        """Reset device to default state."""
        self._send_scpi(SCPI_COMMANDS["reset"], expect_response=False)
        time.sleep(0.5)

    def clear(self):
        """Clear device status."""
        self._send_scpi(SCPI_COMMANDS["clear"], expect_response=False)

    def set_integration_time(self, time_seconds: float):
        """
        Set integration time in seconds.

        Args:
            time_seconds: Integration time (1e-5 to 60 seconds)
        """
        if time_seconds < 1e-5 or time_seconds > 60:
            raise ValueError("Integration time must be between 10µs and 60s")

        cmd = SCPI_COMMANDS["set_integration_time"].format(time_s=time_seconds)
        self._send_scpi(cmd, expect_response=False)

    def get_integration_time(self) -> float:
        """Get current integration time in seconds."""
        response = self._send_scpi(SCPI_COMMANDS["get_integration_time"])
        if response:
            try:
                return float(response)
            except ValueError:
                pass
        return 0.1  # Default 100ms

    def start_scan(self):
        """Start a single scan acquisition."""
        self._send_scpi(SCPI_COMMANDS["start_scan"], expect_response=False)

    def get_scan_data(self) -> np.ndarray:
        """
        Acquire and return spectral data.

        Returns:
            numpy array of intensity values (3648 pixels)
        """
        # Start scan
        self.start_scan()

        # Wait for scan to complete
        integration_time = self.get_integration_time()
        wait_time = integration_time + 0.05  # Integration time + overhead
        time.sleep(wait_time)

        # Read data
        response = self._send_scpi_binary(SCPI_COMMANDS["get_scan_data"])

        # Parse binary data
        # CCS200 returns IEEE 488.2 block format: #NXXXXXXXX<binary data>
        # where N is number of digits in length, XXXXXXXX is length in bytes
        if not response or len(response) < 10:
            raise RuntimeError("Invalid scan data response")

        try:
            # Parse block header
            if response[0:1] != b"#":
                raise RuntimeError(f"Expected IEEE 488.2 block format, got: {response[:20]}")

            n_digits = int(response[1:2].decode())
            length_str = response[2 : 2 + n_digits].decode()
            data_length = int(length_str)

            # Extract binary data
            binary_start = 2 + n_digits
            binary_data = response[binary_start : binary_start + data_length]

            # Convert to float array (assuming 64-bit doubles)
            num_values = data_length // 8
            intensities = np.frombuffer(binary_data[: num_values * 8], dtype=np.float64)

            if len(intensities) != self.NUM_PIXELS:
                # Try 32-bit floats
                num_values = data_length // 4
                intensities = np.frombuffer(binary_data[: num_values * 4], dtype=np.float32)

            return intensities

        except Exception as e:
            raise RuntimeError(f"Failed to parse scan data: {e}")

    def get_data(self) -> np.ndarray:
        """Alias for get_scan_data() for compatibility with Windows interface."""
        return self.get_scan_data()

    def _load_wavelengths(self):
        """Load wavelength calibration data from device."""
        try:
            response = self._send_scpi_binary(SCPI_COMMANDS["get_wavelength_data"])

            if not response or len(response) < 10:
                # Use default wavelengths
                self._wavelengths = np.linspace(200, 1000, self.NUM_PIXELS)
                return

            # Parse IEEE 488.2 block format
            if response[0:1] != b"#":
                self._wavelengths = np.linspace(200, 1000, self.NUM_PIXELS)
                return

            n_digits = int(response[1:2].decode())
            length_str = response[2 : 2 + n_digits].decode()
            data_length = int(length_str)

            binary_start = 2 + n_digits
            binary_data = response[binary_start : binary_start + data_length]

            # Try 64-bit doubles
            num_values = data_length // 8
            self._wavelengths = np.frombuffer(binary_data[: num_values * 8], dtype=np.float64)

            if len(self._wavelengths) != self.NUM_PIXELS:
                # Try 32-bit floats
                num_values = data_length // 4
                self._wavelengths = np.frombuffer(binary_data[: num_values * 4], dtype=np.float32)

            print(
                f"  ✓ Wavelength range: {self._wavelengths[0]:.1f} - {self._wavelengths[-1]:.1f} nm"
            )

        except Exception as e:
            print(f"  ⚠ Failed to load wavelengths, using defaults: {e}")
            self._wavelengths = np.linspace(200, 1000, self.NUM_PIXELS)

    def get_wavelengths(self) -> np.ndarray:
        """Return wavelength array for each pixel."""
        return self._wavelengths

    def close(self):
        """Close device connection."""
        if self._device:
            with contextlib.suppress(usb.core.USBError, ValueError):
                usb.util.release_interface(self._device, 0)
            with contextlib.suppress(usb.core.USBError, ValueError):
                usb.util.dispose_resources(self._device)
            self._device = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def discover_ccs200_devices() -> list[tuple[str, str]]:
    """
    Discover connected CCS200 devices on Linux.

    Returns:
        List of (serial_number, resource_string) tuples
    """
    if not PYUSB_AVAILABLE:
        print("pyusb not available, cannot discover devices")
        return []

    devices = []

    for pid in (CCS200_PID, CCS200_PID_ALT):
        for dev in usb.core.find(find_all=True, idVendor=THORLABS_VID, idProduct=pid):
            try:
                serial = usb.util.get_string(dev, dev.iSerialNumber) or "unknown"
                resource = f"USB::{THORLABS_VID:#06x}::{pid:#06x}::{serial}::RAW"
                devices.append((serial, resource))
            except (usb.core.USBError, ValueError):
                devices.append(("unknown", f"USB::{THORLABS_VID:#06x}::{pid:#06x}::RAW"))

    return devices


if __name__ == "__main__":
    # Test connection
    print("Discovering CCS200 devices on Linux...")
    devices = discover_ccs200_devices()

    if not devices:
        print("No CCS200 devices found.")
        print("\nTroubleshooting:")
        print("1. Ensure device is connected via USB")
        print("2. Check lsusb output for Thorlabs device")
        print("3. Ensure udev rules allow USB access (see deployment guide)")
        print("4. Try running with sudo to test permissions")
        exit(1)

    print(f"Found {len(devices)} device(s):")
    for serial, resource in devices:
        print(f"  - Serial: {serial}, Resource: {resource}")

    print("\nConnecting to first device...")
    try:
        with CCS200Linux() as spec:
            print(f"Device ID: {spec.identify()}")
            print(f"Integration time: {spec.get_integration_time() * 1000:.1f} ms")
            print(
                f"Wavelengths: {spec.get_wavelengths()[0]:.1f} - {spec.get_wavelengths()[-1]:.1f} nm"
            )

            print("\nAcquiring test scan...")
            spec.set_integration_time(0.1)
            data = spec.get_scan_data()
            print(f"Scan acquired: {len(data)} pixels")
            print(f"Intensity range: {data.min():.4f} - {data.max():.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
