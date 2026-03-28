"""
src.calibration.conformal
==========================
Split conformal prediction wrapper providing distribution-free coverage
guarantees over any regression model (GPRCalibration or PhysicsInformedGPR).

Theory (split conformal prediction)
-------------------------------------
Given calibration set {(x_i, y_i)}_{i=1}^n:
    nonconformity score  alpha_i = |y_i - y_hat_i| / sigma_i
    quantile level       q_hat  = ceil((n+1)(1-alpha)/n)-th smallest alpha_i

For a test point x*, the prediction interval is:
    [y_hat* - q_hat * sigma*, y_hat* + q_hat * sigma*]

This has marginal coverage guarantee: P(y* in interval) >= 1 - alpha
with no distributional assumptions beyond exchangeability.

Public API
----------
- ``ConformalCalibrator``  -- calibrate(model, X_cal, y_cal) / predict_interval(model, X, alpha)
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class _PredictsMeanStd(Protocol):
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class ConformalCalibrator:
    """Distribution-free prediction intervals via split conformal prediction.

    Wrap any model that implements ``predict(X, return_std=True) -> (mean, std)``.

    This implementation uses absolute residual conformal scores to preserve the
    finite-sample coverage guarantee without depending on model uncertainty
    calibration quality.

    Example
    -------
    ::

        cal = ConformalCalibrator()
        cal.calibrate(gpr, X_cal, y_cal)
        lo, hi = cal.predict_interval(gpr, X_test, alpha=0.10)  # 90% coverage
    """

    def __init__(self) -> None:
        self._scores: list[float] = []
        self._n_cal: int = 0

    def calibrate(
        self,
        model: _PredictsMeanStd,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Compute normalised nonconformity scores on the calibration set.

        Parameters
        ----------
        model :
            Fitted model with ``predict(X, return_std=True) -> (mean, std)``.
        X_cal : shape (n, d) -- calibration features
        y_cal : shape (n,)   -- calibration targets
        """
        mean, std = model.predict(X_cal, return_std=True)
        std_safe = np.maximum(std.ravel(), 1e-9)
        scores = np.abs(y_cal.ravel() - mean.ravel()) / std_safe
        self._scores = scores.tolist()
        self._n_cal = len(scores)

    def predict_interval(
        self,
        model: _PredictsMeanStd,
        X: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal prediction interval with (1-alpha) coverage.

        Parameters
        ----------
        model : fitted model (same as passed to ``calibrate``)
        X     : shape (n, d) -- test features
        alpha : miscoverage level, e.g. 0.10 for 90% coverage

        Returns
        -------
        (lower, upper) -- both shape (n,)

        Raises
        ------
        RuntimeError if ``calibrate`` has not been called.
        """
        if self._n_cal == 0:
            raise RuntimeError(
                "ConformalCalibrator.calibrate() must be called before predict_interval()."
            )

        mean, std = model.predict(X, return_std=True)
        mean = mean.ravel()
        std = np.maximum(std.ravel(), 1e-9)

        # Split conformal quantile with conservative "higher" interpolation.
        n = self._n_cal
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        q_hat = float(np.quantile(np.asarray(self._scores, dtype=float), level, method="higher"))

        lower = mean - q_hat * std
        upper = mean + q_hat * std
        return lower, upper
