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
    QualityAgent(bus, ok_emit_every=1).process(1, wl, normal_spectrum)
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


def test_low_snr_frame_returns_false_with_warn(wl, flat_noise_spectrum):
    # C6: SNR below threshold is a hard block — frame must be discarded.
    bus, loop = _bus()
    q = bus.subscribe()
    result = QualityAgent(bus).process(1, wl, flat_noise_spectrum)
    _flush(loop)
    event = q.get_nowait()
    assert result is False       # hard block — C6
    assert event.level == "warn"
    assert event.data["quality"] == "low_snr"
    loop.close()


# -----------------------------------------------------------------------
# C1: NaN / Inf and too-short array hard blocks
# -----------------------------------------------------------------------


def test_nan_intensities_returns_false(wl):
    """C1: frame with NaN values must be hard-blocked."""
    bus, loop = _bus()
    q = bus.subscribe()
    bad = np.full(len(wl), np.nan)
    result = QualityAgent(bus).process(1, wl, bad)
    _flush(loop)
    event = q.get_nowait()
    assert result is False
    assert event.level == "error"
    assert event.data["quality"] == "non_finite"
    loop.close()


def test_inf_intensities_returns_false(wl):
    """C1: frame with Inf values must be hard-blocked."""
    bus, loop = _bus()
    q = bus.subscribe()
    bad = np.full(len(wl), np.inf)
    result = QualityAgent(bus).process(1, wl, bad)
    _flush(loop)
    event = q.get_nowait()
    assert result is False
    assert event.level == "error"
    assert event.data["quality"] == "non_finite"
    loop.close()


def test_mixed_nan_inf_reports_count(wl):
    """C1: n_non_finite in event data matches the number of bad pixels."""
    bus, loop = _bus()
    q = bus.subscribe()
    bad = np.zeros(len(wl))
    bad[10] = np.nan
    bad[20] = np.inf
    QualityAgent(bus).process(1, wl, bad)
    _flush(loop)
    event = q.get_nowait()
    assert event.data["n_non_finite"] == 2
    loop.close()


def test_too_short_array_returns_false():
    """C1: array with fewer than 4 pixels must be hard-blocked."""
    bus, loop = _bus()
    q = bus.subscribe()
    tiny_wl = np.array([700.0, 710.0, 720.0])
    tiny_i = np.array([0.1, 0.2, 0.1])
    result = QualityAgent(bus).process(1, tiny_wl, tiny_i)
    _flush(loop)
    event = q.get_nowait()
    assert result is False
    assert event.level == "error"
    assert event.data["quality"] == "too_short"
    loop.close()


def test_nan_check_precedes_saturation(wl):
    """C1: NaN check fires before saturation check — event quality='non_finite', not 'saturated'."""
    bus, loop = _bus()
    q = bus.subscribe()
    bad = np.full(len(wl), 70_000.0)
    bad[0] = np.nan
    QualityAgent(bus).process(1, wl, bad)
    _flush(loop)
    event = q.get_nowait()
    assert event.data["quality"] == "non_finite"
    loop.close()


def test_event_contains_frame_number(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus, ok_emit_every=1).process(42, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["frame"] == 42
    loop.close()


def test_event_contains_snr(wl, normal_spectrum):
    bus, loop = _bus()
    q = bus.subscribe()
    QualityAgent(bus, ok_emit_every=1).process(1, wl, normal_spectrum)
    _flush(loop)
    assert q.get_nowait().data["snr"] > 0.0
    loop.close()


def test_ok_throttle_emits_on_interval(wl, normal_spectrum):
    """ok events are suppressed below the interval, emitted at the interval."""
    bus, loop = _bus()
    q = bus.subscribe()
    agent = QualityAgent(bus, ok_emit_every=3)
    for i in range(1, 4):
        agent.process(i, wl, normal_spectrum)
    _flush(loop)
    # Only 1 event should be in queue (on frame 3)
    assert not q.empty()
    ev = q.get_nowait()
    assert ev.data["frame"] == 3
    assert q.empty()
    loop.close()
