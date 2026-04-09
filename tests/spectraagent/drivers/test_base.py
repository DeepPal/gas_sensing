import numpy as np
import pytest

from spectraagent.drivers.base import AbstractHardwareDriver


class _ConcreteDriver(AbstractHardwareDriver):
    """Minimal concrete implementation for testing the ABC contract."""

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_wavelengths(self) -> np.ndarray:
        return np.linspace(500.0, 900.0, 3648)

    def read_spectrum(self) -> np.ndarray:
        return np.ones(3648) * 0.5

    def get_integration_time_ms(self) -> float:
        return 50.0

    def set_integration_time_ms(self, ms: float) -> None:
        pass

    @property
    def name(self) -> str:
        return "TestDriver"

    @property
    def is_connected(self) -> bool:
        return getattr(self, "_connected", False)


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        AbstractHardwareDriver()  # type: ignore[abstract]


def test_concrete_driver_satisfies_interface():
    drv = _ConcreteDriver()
    drv.connect()
    assert drv.is_connected
    wl = drv.get_wavelengths()
    assert wl.shape == (3648,)
    sp = drv.read_spectrum()
    assert sp.shape == (3648,)
    assert drv.name == "TestDriver"
    drv.disconnect()
    assert not drv.is_connected


def test_wavelengths_and_spectrum_same_length():
    drv = _ConcreteDriver()
    drv.connect()
    assert drv.get_wavelengths().shape == drv.read_spectrum().shape
