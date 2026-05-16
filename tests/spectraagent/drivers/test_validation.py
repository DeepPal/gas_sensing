import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver
from spectraagent.drivers.validation import validate_driver_class, validate_driver_instance


class _GoodDriver(AbstractHardwareDriver):
    def __init__(self) -> None:
        self._connected = False
        self._integration_ms = 25.0
        self._wl = np.linspace(500.0, 900.0, 3648)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return self._wl.copy()

    def read_spectrum(self) -> np.ndarray:
        return np.ones_like(self._wl) * 0.5

    def get_integration_time_ms(self) -> float:
        return self._integration_ms

    def set_integration_time_ms(self, ms: float) -> None:
        self._integration_ms = ms

    @property
    def name(self) -> str:
        return "GoodDriver"

    @property
    def is_connected(self) -> bool:
        return self._connected


class _BadDriverShape(_GoodDriver):
    def get_wavelengths(self) -> np.ndarray:
        return np.array([[1.0, 2.0], [3.0, 4.0]])


class _BadDriverSpectrum(_GoodDriver):
    def read_spectrum(self) -> np.ndarray:
        return np.array([1.0, np.nan, 2.0])


class _NotSubclass:
    pass


def test_validate_driver_class_accepts_valid_class() -> None:
    assert validate_driver_class(_GoodDriver) == []


def test_validate_driver_class_rejects_non_subclass() -> None:
    issues = validate_driver_class(_NotSubclass)
    assert any("subclass AbstractHardwareDriver" in s for s in issues)


def test_validate_driver_instance_accepts_valid_connected_driver() -> None:
    drv = _GoodDriver()
    drv.connect()
    assert validate_driver_instance(drv, require_live_sample=True) == []


def test_validate_driver_instance_rejects_bad_wavelength_shape() -> None:
    drv = _BadDriverShape()
    drv.connect()
    issues = validate_driver_instance(drv, require_live_sample=False)
    assert any("1D numeric numpy array" in s for s in issues)


def test_validate_driver_instance_rejects_invalid_spectrum() -> None:
    drv = _BadDriverSpectrum()
    drv.connect()
    issues = validate_driver_instance(drv, require_live_sample=True)
    assert any("shape" in s or "non-finite" in s for s in issues)
