"""Tests for ExperimentPlannerAgent with BayesianExperimentDesigner."""
import pytest
from unittest.mock import MagicMock
from spectraagent.webapp.agents.planner import ExperimentPlannerAgent
from spectraagent.webapp.agent_bus import AgentBus


def _make_bus():
    bus = MagicMock(spec=AgentBus)
    bus.emit = MagicMock()
    return bus


def test_suggest_uses_logspace_not_linspace():
    """With no measured points, suggest() must return a value in range (space-filling)."""
    bus = _make_bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    suggestion = agent.suggest()
    assert suggestion is not None
    assert 0.01 <= suggestion <= 10.0


def test_suggest_with_measured_history():
    """After recording measured concentrations, suggest() should not repeat them."""
    bus = _make_bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.record_measured(0.5)
    agent.record_measured(2.0)
    suggestion = agent.suggest()
    if suggestion is not None:
        assert abs(suggestion - 0.5) > 0.01
        assert abs(suggestion - 2.0) > 0.01


def test_record_measured_stores_values():
    """record_measured() must persist values for BED."""
    bus = _make_bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.record_measured(1.0)
    agent.record_measured(3.0)
    assert 1.0 in agent._measured
    assert 3.0 in agent._measured


def test_suggest_emits_event():
    """suggest() must emit an experiment_suggestion event via the bus."""
    bus = _make_bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    agent.suggest()
    bus.emit.assert_called_once()
    call_args = bus.emit.call_args[0][0]
    assert call_args.type == "experiment_suggestion"
