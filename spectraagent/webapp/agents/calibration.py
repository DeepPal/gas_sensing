"""
spectraagent.webapp.agents.calibration
========================================
CalibrationAgent — AIC-based isotherm model selection.

Wraps ``src.calibration.isotherms.select_isotherm`` — do NOT reimplement
AIC fitting here.  ``select_isotherm`` evaluates Langmuir, Freundlich, Hill,
and Linear models, selects the winner by AICc (small-sample corrected AIC),
and returns a dict with ``best_model``, ``best_result``, ``aic_table``.

Emits a ``model_selected`` event after each calibration data point once
``min_points`` is reached.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, cast

import numpy as np

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)

_MIN_POINTS: int = 4   # minimum before fitting is meaningful

try:
    from src.calibration.isotherms import select_isotherm as _select_isotherm
except ImportError:
    _select_isotherm = None  # type: ignore[assignment]


class CalibrationAgent:
    """AIC model selector for calibration curve (wraps select_isotherm).

    Parameters
    ----------
    bus:
        AgentBus for emitting events.
    min_points:
        Minimum calibration points before fitting (default 4).
    """

    def __init__(self, bus: AgentBus, min_points: int = _MIN_POINTS) -> None:
        self._bus = bus
        self._min_points = min_points
        self._concentrations: list[float] = []
        self._delta_lambdas: list[float] = []
        self._lock = threading.Lock()

    def add_point(self, concentration: float, delta_lambda: float) -> None:
        """Add a calibration point and refit if enough data.

        Parameters
        ----------
        concentration:
            Analyte concentration (ppm or consistent units).
        delta_lambda:
            Measured LSPR peak shift in nm (typically negative on adsorption).
        """
        with self._lock:
            self._concentrations.append(float(concentration))
            self._delta_lambdas.append(float(delta_lambda))
            if len(self._concentrations) < self._min_points:
                return
            c = np.array(self._concentrations)
            r = np.array(self._delta_lambdas)

        n = len(c)

        if _select_isotherm is None:
            log.warning("CalibrationAgent: src.calibration.isotherms not available; skipping fit")
            return

        try:
            result = _select_isotherm(c, r)
            best_model: str = str(result["best_model"])
            best_result = result["best_result"]
            if not hasattr(best_result, "aic") or not hasattr(best_result, "r_squared"):
                log.warning("CalibrationAgent: best_result has no .aic/.r_squared: %r", best_result)
                return
            best_aic: float = float(best_result.aic)
            r_squared: float = float(best_result.r_squared)
            self._bus.emit(AgentEvent(
                source="CalibrationAgent",
                level="info",
                type="model_selected",
                data={
                    "n_points": n,
                    "best_model": best_model,
                    "best_aic": round(best_aic, 3),
                    "r_squared": round(r_squared, 4),
                    "aic_table": [
                        {"model": row[0], "aic": round(float(row[1]), 3)}
                        for row in cast(list[Any], result["aic_table"])
                    ],
                },
                text=(
                    f"Calibration: {n} points — best model: {best_model} "
                    f"(AICc={best_aic:.2f}, R²={r_squared:.4f})"
                ),
            ))
        except (RuntimeError, ValueError, TypeError) as exc:
            log.warning("CalibrationAgent: fit failed: %s", exc)

    def clear(self) -> None:
        """Reset calibration data (call at new session start)."""
        with self._lock:
            self._concentrations.clear()
            self._delta_lambdas.clear()

    @property
    def data(self) -> tuple[list[float], list[float]]:
        """Return (concentrations, delta_lambdas) accumulated so far."""
        with self._lock:
            return list(self._concentrations), list(self._delta_lambdas)
