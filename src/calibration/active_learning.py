"""
src.calibration.active_learning
=================================
Bayesian Experimental Design for LSPR calibration.

Uses GPR posterior variance as an acquisition function (maximum uncertainty
sampling). Candidates are drawn on a logspace grid so that sparse
low-concentration and sparse high-concentration regions are both explored.

Public API
----------
- ``BayesianExperimentDesigner``  — suggest_next(gpr, measured) -> float
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class BayesianExperimentDesigner:
    """Suggest the next calibration concentration to maximise information gain.

    Candidates are drawn on a logspace grid so low-concentration (0.01 ppm)
    and high-concentration (10 ppm) regions receive equal log-space density.
    Falls back to space-filling (log-space max-distance) when no GPR is
    available or fewer than 3 points have been measured.

    Parameters
    ----------
    min_conc, max_conc : float
        Concentration search range (ppm). Both must be > 0 (logspace requires
        positive values). ``min_conc`` must be strictly less than ``max_conc``.
    n_candidates : int
        Number of logspace candidates to evaluate (default 100).
    """

    def __init__(
        self,
        min_conc: float = 0.01,
        max_conc: float = 10.0,
        n_candidates: int = 100,
    ) -> None:
        if min_conc >= max_conc:
            raise ValueError("min_conc must be < max_conc")
        if min_conc <= 0:
            raise ValueError("min_conc must be > 0 for logspace grid")
        self._min_conc = min_conc
        self._max_conc = max_conc
        self._n_candidates = n_candidates

    def suggest_next(
        self,
        gpr: Any | None,
        measured: list[float],
    ) -> float:
        """Return the concentration with highest posterior uncertainty.

        Falls back to space-filling when ``gpr`` is ``None`` or
        ``len(measured) < 3``.

        Parameters
        ----------
        gpr      : fitted GPRCalibration / PhysicsInformedGPR (or None)
        measured : concentrations already measured in this session

        Returns
        -------
        Suggested concentration (ppm).
        """
        candidates = np.logspace(
            np.log10(self._min_conc),
            np.log10(self._max_conc),
            self._n_candidates,
        )

        # Exclude candidates too close to already-measured points (log-scale distance,
        # consistent with the log-space candidate grid).
        if measured:
            log_gap = np.log10(self._max_conc / self._min_conc) / (self._n_candidates * 2)
            log_candidates = np.log10(candidates)
            log_measured = np.log10(np.maximum(np.array(measured, dtype=float), 1e-12))
            mask = np.all(
                np.abs(log_candidates[:, None] - log_measured[None, :]) > log_gap,
                axis=1,
            )
            filtered = candidates[mask] if mask.any() else candidates
        else:
            filtered = candidates

        # Space-filling fallback: no GPR or sparse early-session data.
        if gpr is None or len(measured) < 3:
            return float(self._space_filling(filtered, measured))

        # Max-variance acquisition via GPR posterior std.
        try:
            _, std_arr = gpr.predict(filtered.reshape(-1, 1), return_std=True)
            best_idx = int(np.argmax(std_arr))
            return float(filtered[best_idx])
        except Exception as exc:
            log.warning("BayesianExperimentDesigner GPR query failed: %s", exc)
            return float(self._space_filling(filtered, measured))

    def _space_filling(
        self,
        candidates: np.ndarray,
        measured: list[float],
    ) -> float:
        """Return the candidate furthest (in log-space) from any measured point.

        With no measured points returns the geometric midpoint of the range.
        """
        if len(measured) == 0:
            return float(np.sqrt(self._min_conc * self._max_conc))

        log_measured = np.log10(np.array(measured))
        log_candidates = np.log10(candidates)

        # Min log-distance to any measured point, maximised over candidates.
        distances = np.min(
            np.abs(log_candidates[:, None] - log_measured[None, :]),
            axis=1,
        )
        return float(candidates[int(np.argmax(distances))])
