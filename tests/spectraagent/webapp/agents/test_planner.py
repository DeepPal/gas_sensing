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


def _mock_gpr_logspace(peak_frac: float = 0.9):
    """GPR mock that returns correct-sized std arrays — peak at given fraction of candidates."""
    gpr = MagicMock()

    def predict_fn(X, return_std=False):
        n = len(X)
        std = np.zeros(n)
        peak_idx = int(peak_frac * (n - 1))
        std[peak_idx] = 10.0
        return np.zeros(n), std

    gpr.predict.side_effect = predict_fn
    return gpr


def test_suggest_returns_value_without_gpr():
    """Without GPR, suggest() uses space-filling fallback — returns a float, not None."""
    bus, loop = _bus()
    result = ExperimentPlannerAgent(bus).suggest()
    assert isinstance(result, float)
    assert result > 0
    loop.close()


def test_suggest_returns_float_with_gpr():
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr_logspace())
    result = agent.suggest()
    assert isinstance(result, float)
    loop.close()


def test_suggest_returns_highest_uncertainty_concentration():
    """With ≥3 measured points, BED uses GPR max-variance (not space-filling)."""
    bus, loop = _bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0, n_candidates=100)
    # Need ≥3 measured points to trigger GPR path (not space-filling)
    agent.record_measured(0.1)
    agent.record_measured(0.3)
    agent.record_measured(1.0)
    agent.set_gpr(_mock_gpr_logspace(peak_frac=0.9))  # peak near high-concentration end
    result = agent.suggest()
    assert result is not None
    assert result >= 4.5  # should be pulled toward high-concentration region
    loop.close()


def test_suggest_emits_experiment_suggestion_event():
    bus, loop = _bus()
    q = bus.subscribe()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.set_gpr(_mock_gpr_logspace())
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
    agent.set_gpr(_mock_gpr_logspace())
    agent.suggest()
    _flush(loop)
    data = q.get_nowait().data
    for key in ("suggested_concentration", "measured_so_far", "search_range", "method"):
        assert key in data, f"missing key: {key}"
    assert data["method"] == "bayesian_logspace"
    loop.close()


def test_suggest_with_failed_gpr_falls_back_to_space_filling():
    """When GPR.predict() raises, suggest() falls back to space-filling (returns float)."""
    bus, loop = _bus()
    gpr = MagicMock()
    gpr.predict.side_effect = RuntimeError("GPR not fitted")
    agent = ExperimentPlannerAgent(bus)
    agent.set_gpr(gpr)
    # 3 measured points so BED attempts GPR path (which then fails + falls back)
    agent.record_measured(0.1)
    agent.record_measured(0.5)
    agent.record_measured(2.0)
    result = agent.suggest()
    assert isinstance(result, float)  # space-filling fallback, not None
    loop.close()
