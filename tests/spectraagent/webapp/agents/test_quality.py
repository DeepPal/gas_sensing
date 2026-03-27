import asyncio

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.quality import QualityAgent, _compute_snr


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _bus() -> tuple[AgentBus, asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop: asyncio.AbstractEventLoop) -> None:
    """Run one event-loop iteration so call_soon_threadsafe callbacks execute."""
    loop.run_until_complete(asyncio.sleep(0))


@pytest.fixture
def wl() -> np.ndarray:
    return np.linspace(500.0, 900.0, 3648)


@pytest.fixture
def normal_spectrum(wl: np.ndarray) -> np.ndarray:
    """Lorentzian at 720 nm, amplitude 0.8 — clear peak, low noise."""
    sp = 0.8 / (1.0 + ((wl - 720.0) / 9.0) ** 2)
    sp += np.random.default_rng(0).normal(0, 0.001, len(wl))
    return np.clip(sp, 0.0, None)


@pytest.fixture
def saturated_spectrum(wl: np.ndarray) -> np.ndarray:
    """Peak at 65000 counts — exceeds saturation threshold."""
    return 65_000.0 / (1.0 + ((wl - 720.0) / 9.0) ** 2)


@pytest.fixture
def flat_noise_spectrum(wl: np.ndarray) -> np.ndarray:
    """Flat noise — no peak, SNR << 3."""
    return np.random.default_rng(1).normal(0.01, 0.01, len(wl)).clip(0.0, None)


# -----------------------------------------------------------------------
# _compute_snr
# -----------------------------------------------------------------------


def test_snr_high_for_lorentzian(wl, normal_spectrum):
    assert _compute_snr(wl, normal_spectrum) > 10.0


def test_snr_low_for_flat_noise(wl, flat_noise_spectrum):
    assert _compute_snr(wl, flat_noise_spectrum) < 3.0


# -----------------------------------------------------------------------
# QualityAgent
# -----------------------------------------------------------------------


def test_normal_frame_returns_true(wl, normal_spectrum):
    bus, loop = _bus()
    result = QualityAgent(bus).process(1, wl, normal_spectrum)
    loop.close()
    assert result is True


def test_normal_frame_emits_ok_event(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, normal_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert event.level == "ok"
    assert event.source == "QualityAgent"
    assert event.type == "quality"
    loop.close()


def test_saturated_frame_returns_false(wl, saturated_spectrum):
    bus, loop = _bus()
    result = QualityAgent(bus).process(1, wl, saturated_spectrum)
    loop.close()
    assert result is False


def test_saturated_frame_emits_error_event(wl, saturated_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, saturated_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert event.level == "error"
    assert event.data["quality"] == "saturated"
    loop.close()


def test_low_snr_frame_returns_true_with_warn(wl, flat_noise_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    result = QualityAgent(bus).process(1, wl, flat_noise_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert result is True        # frame still processed
    assert event.level == "warn"
    assert event.data["quality"] == "low_snr"
    loop.close()


def test_event_contains_frame_number(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(42, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["frame"] == 42
    loop.close()


def test_event_contains_snr(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus).process(1, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["snr"] > 0.0
    loop.close()
