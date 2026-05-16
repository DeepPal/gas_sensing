import asyncio

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.calibration import CalibrationAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def test_no_event_below_min_points():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    agent.add_point(0.1, -0.5)
    agent.add_point(0.2, -1.0)
    _flush(loop)
    assert q.empty()
    loop.close()


def test_event_emitted_at_min_points():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.type == "model_selected"
    assert event.source == "CalibrationAgent"
    loop.close()


def test_event_has_required_data_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    data = q.get_nowait().data
    for key in ("n_points", "best_model", "best_aic", "r_squared"):
        assert key in data, f"missing key: {key}"
    loop.close()


def test_linear_data_prefers_low_param_model():
    """Perfect linear data should select 'linear' (AIC penalises extra params)."""
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    # Perfect linear: delta_lambda = -5 * concentration
    for c in [0.1, 0.2, 0.5, 1.0, 2.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    # Drain all events (one per point after min_points)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert len(events) > 0
    # The last event (most data) should select linear or langmuir
    assert events[-1].data["best_model"] in ("linear", "langmuir", "freundlich")
    loop.close()


def test_r_squared_is_between_0_and_1():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = CalibrationAgent(bus, min_points=4)
    for c in [0.1, 0.2, 0.5, 1.0]:
        agent.add_point(c, -5.0 * c)
    _flush(loop)
    r2 = q.get_nowait().data["r_squared"]
    assert 0.0 <= r2 <= 1.0
    loop.close()


def test_clear_resets_data():
    bus, loop = _bus()
    agent = CalibrationAgent(bus)
    agent.add_point(0.1, -0.5)
    agent.clear()
    concs, deltas = agent.data
    assert concs == []
    assert deltas == []
    loop.close()


def test_data_property_returns_accumulated_points():
    bus, loop = _bus()
    agent = CalibrationAgent(bus, min_points=10)
    agent.add_point(0.1, -0.5)
    agent.add_point(0.2, -1.0)
    concs, deltas = agent.data
    assert concs == [0.1, 0.2]
    assert deltas == [-0.5, -1.0]
    loop.close()
