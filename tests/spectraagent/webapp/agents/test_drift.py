import asyncio

import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.drift import DriftAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def test_no_event_before_window_full():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    for i in range(30):           # only half the window
        agent.update(i, 720.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_stable_signal_no_drift_event():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    for i in range(60):
        agent.update(i, 720.0)   # constant wavelength
    _flush(loop)
    assert q.empty()
    loop.close()


def test_fast_drift_emits_warn():
    """1.0 nm/min drift is well above the 0.05 nm/min threshold."""
    bus, loop = _bus()
    q = bus.subscribe()
    # integration_time_ms=50 → 1200 frames/min
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    drift_per_frame = 1.0 / frames_per_min   # 1 nm/min in nm/frame
    for i in range(60):
        agent.update(i, 720.0 + i * drift_per_frame)
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.level == "warn"
    assert event.type == "drift_warn"
    assert abs(event.data["drift_rate_nm_per_min"]) > 0.05
    loop.close()


def test_slow_drift_below_threshold_no_event():
    """0.01 nm/min drift is below the 0.05 nm/min threshold."""
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    drift_per_frame = 0.01 / frames_per_min
    for i in range(60):
        agent.update(i, 720.0 + i * drift_per_frame)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_reset_clears_history():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=10)
    for i in range(10):
        agent.update(i, 720.0 + i * 0.1)
    agent.reset()
    # After reset, window is empty — 5 more frames should not trigger
    for i in range(5):
        agent.update(i, 721.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_drift_event_has_required_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = DriftAgent(bus, integration_time_ms=50.0, window_frames=60)
    frames_per_min = 60_000 / 50.0
    for i in range(60):
        agent.update(i, 720.0 + i * (1.0 / frames_per_min))
    _flush(loop)
    event = q.get_nowait()
    for key in ("frame", "drift_rate_nm_per_min", "window_frames", "peak_wavelength"):
        assert key in event.data, f"missing key: {key}"
    loop.close()
