"""
spectraagent.webapp.agents.planner
====================================
ExperimentPlannerAgent — suggests the next concentration to measure.

Uses GPRCalibration posterior uncertainty: queries predict() on a grid of
candidate concentrations and returns the one with the highest posterior std
(maximum information gain per spec Section 4 — no BoTorch needed).

Called on-demand via POST /api/calibration/suggest or by CalibrationAgent
after model selection.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)

_N_CANDIDATES: int = 50


class ExperimentPlannerAgent:
    """Concentration suggestion using GPR posterior uncertainty.

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    min_conc, max_conc:
        Concentration range to search.
    n_candidates:
        Grid resolution (default 50).
    """

    def __init__(
        self,
        bus: AgentBus,
        min_conc: float = 0.01,
        max_conc: float = 10.0,
        n_candidates: int = _N_CANDIDATES,
    ) -> None:
        if min_conc >= max_conc:
            raise ValueError(f"min_conc ({min_conc}) must be less than max_conc ({max_conc})")
        self._bus = bus
        self._min_conc = min_conc
        self._max_conc = max_conc
        self._n_candidates = n_candidates
        self._gpr = None  # set via set_gpr()

    def set_gpr(self, gpr) -> None:
        """Inject a fitted GPRCalibration (or compatible mock) instance."""
        self._gpr = gpr

    def suggest(self) -> Optional[float]:
        """Return the concentration with the highest GPR posterior std.

        Returns None if no GPR is set or if prediction fails.
        Emits an ``experiment_suggestion`` event on success.
        """
        if self._gpr is None:
            return None

        try:
            candidates = np.linspace(self._min_conc, self._max_conc, self._n_candidates)
            # GPRCalibration.predict() expects shape (n, 1) for 1D features
            _, std_arr = self._gpr.predict(candidates.reshape(-1, 1))
            best_idx = int(np.argmax(std_arr))
            best_conc = float(candidates[best_idx])
            best_std = float(std_arr[best_idx])

            self._bus.emit(AgentEvent(
                source="ExperimentPlannerAgent",
                level="info",
                type="experiment_suggestion",
                data={
                    "suggested_concentration": round(best_conc, 4),
                    "posterior_std": round(best_std, 6),
                    "search_range": [self._min_conc, self._max_conc],
                    "n_candidates": self._n_candidates,
                },
                text=(
                    f"Suggested next concentration: {best_conc:.4f} "
                    f"(posterior σ={best_std:.4g})"
                ),
            ))
            return best_conc

        except Exception as exc:
            log.warning("ExperimentPlannerAgent.suggest() failed: %s", exc)
            return None
