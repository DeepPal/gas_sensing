import time

import numpy as np
import pytest

from spectraagent.drivers.simulation import SimulationDriver


@pytest.fixture
def drv():
    d = SimulationDriver(integration_time_ms=1.0)  # fast for tests
    d.connect()
    yield d
    d.disconnect()


def test_name(drv):
    assert drv.name == "Simulation"


def test_is_connected(drv):
    assert drv.is_connected


def test_disconnect_clears_connected():
    d = SimulationDriver()
    d.connect()
    d.disconnect()
    assert not d.is_connected


def test_wavelengths_shape(drv):
    wl = drv.get_wavelengths()
    assert wl.shape == (3648,)


def test_wavelengths_range(drv):
    wl = drv.get_wavelengths()
    assert wl[0] == pytest.approx(500.0)
    assert wl[-1] == pytest.approx(900.0)


def test_spectrum_shape(drv):
    sp = drv.read_spectrum()
    assert sp.shape == (3648,)


def test_spectrum_non_negative(drv):
    sp = drv.read_spectrum()
    assert np.all(sp >= 0.0)


def test_spectrum_has_peak_near_720nm(drv):
    wl = drv.get_wavelengths()
    sp = drv.read_spectrum()
    peak_idx = int(np.argmax(sp))
    assert 700.0 <= wl[peak_idx] <= 740.0


def test_spectrum_max_amplitude_reasonable(drv):
    sp = drv.read_spectrum()
    assert 0.5 <= sp.max() <= 1.0


def test_integration_time_roundtrip(drv):
    drv.set_integration_time_ms(25.0)
    assert drv.get_integration_time_ms() == pytest.approx(25.0)


def test_read_blocks_for_integration_time():
    drv = SimulationDriver(integration_time_ms=50.0)
    drv.connect()
    t0 = time.monotonic()
    drv.read_spectrum()
    elapsed_ms = (time.monotonic() - t0) * 1000
    drv.disconnect()
    assert elapsed_ms >= 40.0  # at least 80% of integration time
