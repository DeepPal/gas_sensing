import asyncio
from unittest.mock import MagicMock

import numpy as np
import pytest

from spectraagent.webapp.agent_bus import AgentBus
from spectraagent.webapp.agents.planner import ExperimentPlannerAgent


def _bus():
    loop = asyncio.new_event_loop()
    bus = AgentBus()
    bus.setup_loop(loop)
    return bus, loop


def _flush(loop):
    loop.run_until_complete(asyncio.sleep(0))


def _mock_gpr(peak_at: float = 5.0, n: int = 50, range_: tuple = (0.01, 10.0)):
    """GPR mock returning max uncertainty near peak_at."""
    gpr = MagicMock()
    xs = np.linspace(*range_, n)
    std = np.exp(-0.5 * ((xs - peak_at) / 0.5) ** 2)
    gpr.predict.return_value = (np.zeros(n), std)
    gpr.is_fitted = True
    return gpr


def test_suggest_returns_none_without_gpr():
    bus, loop = _bus()
    result = ExperimentPlannerAgent(bus).suggest()
    assert result is None
    loop.close()


def test_suggest_returns_float_with_gpr():
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    result = agent.suggest()
    assert isinstance(result, float)
    loop.close()


def test_suggest_returns_highest_uncertainty_concentration():
    """Should return concentration near where the mock GPR has highest std."""
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0, n_candidates=50)
    agent.set_gpr(_mock_gpr(peak_at=5.0))
    result = agent.suggest()
    assert 4.0 <= result <= 6.0   # near peak_at=5.0
    loop.close()


def test_suggest_emits_experiment_suggestion_event():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    agent.suggest()
    _flush(loop)
    assert not q.empty()
    event = q.get_nowait()
    assert event.type == "experiment_suggestion"
    assert event.source == "ExperimentPlannerAgent"
    loop.close()


def test_suggestion_event_has_required_fields():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr())
    agent.suggest()
    _flush(loop)
    data = q.get_nowait().data
    for key in ("suggested_concentration", "posterior_std", "search_range"):
        assert key in data, f"missing key: {key}"
    loop.close()


def test_suggest_with_failed_gpr_returns_none():
    bus, loop = _bus()
    gpr = MagicMock()
    gpr.predict.side_effect = RuntimeError("GPR not fitted")
    agent = ExperimentPlannerAgent(bus)
    agent.set_gpr(gpr)
    result = agent.suggest()
    assert result is None
    loop.close()
