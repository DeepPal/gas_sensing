"""
spectraagent.webapp.agents.planner
====================================
ExperimentPlannerAgent — suggests the next calibration concentration.

Upgraded from linspace max-variance to Bayesian Experimental Design using
logspace candidates and space-filling fallback for sparse early-session data.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)


class ExperimentPlannerAgent:
    """Concentration suggestion using Bayesian Experimental Design.

    Uses BayesianExperimentDesigner from src.calibration.active_learning with
    logspace candidates so low and high concentration regions are both explored.
    Falls back to space-filling when no GPR is fitted yet.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    min_conc, max_conc:
        Concentration range to search (ppm).
    n_candidates:
        Grid resolution for BED search (default 100).
    """

    def __init__(
        self,
        bus: AgentBus,
        min_conc: float = 0.01,
        max_conc: float = 10.0,
        n_candidates: int = 100,
    ) -> None:
        if min_conc >= max_conc:
            raise ValueError(f"min_conc ({min_conc}) must be < max_conc ({max_conc})")
        self._bus = bus
        self._min_conc = min_conc
        self._max_conc = max_conc
        self._n_candidates = n_candidates
        self._gpr = None
        self._measured: list[float] = []
        self._designer: Optional[Any] = None  # BayesianExperimentDesigner or None

        try:
            from src.calibration.active_learning import BayesianExperimentDesigner
            self._designer = BayesianExperimentDesigner(
                min_conc=min_conc,
                max_conc=max_conc,
                n_candidates=n_candidates,
            )
        except ImportError:
            log.warning("planner: src.calibration.active_learning unavailable — Bayesian suggest disabled")
            self._designer = None

    def reset(self) -> None:
        """Clear session-scoped state. Call at the start of each new session."""
        self._measured.clear()
        self._gpr = None

    def set_gpr(self, gpr) -> None:
        """Inject a fitted GPRCalibration (or PhysicsInformedGPR) instance."""
        self._gpr = gpr

    def record_measured(self, concentration: float) -> None:
        """Record that a concentration has been measured (updates space-filling avoidance)."""
        self._measured.append(float(concentration))

    def suggest(self) -> Optional[float]:
        """Return the next best concentration using Bayesian Experimental Design.

        Returns None only if an unexpected internal error occurs.
        Emits an ``experiment_suggestion`` event on success.
        """
        if self._designer is None:
            log.warning("planner: BayesianExperimentDesigner unavailable — cannot suggest")
            return None
        try:
            suggestion = self._designer.suggest_next(self._gpr, self._measured)
            self._bus.emit(AgentEvent(
                source="ExperimentPlannerAgent",
                level="info",
                type="experiment_suggestion",
                data={
                    "suggested_concentration": round(suggestion, 4),
                    "measured_so_far": len(self._measured),
                    "search_range": [self._min_conc, self._max_conc],
                    "method": "bayesian_logspace",
                },
                text=f"Suggested next concentration: {suggestion:.4f} ppm (BED logspace)",
            ))
            return float(suggestion)
        except Exception as exc:
            log.warning("ExperimentPlannerAgent.suggest() failed: %s", exc)
            return None
