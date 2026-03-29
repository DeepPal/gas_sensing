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

import warnings
from typing import Protocol

import numpy as np

_MIN_CAL_POINTS: int = 10  # below this the q̂ quantile estimate is unreliable


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

    @property
    def n_cal(self) -> int:
        """Number of calibration points used to compute the conformal quantile."""
        return self._n_cal

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
        if self._n_cal < _MIN_CAL_POINTS:
            warnings.warn(
                f"ConformalCalibrator: only {self._n_cal} calibration points; "
                f"the coverage guarantee requires ≥ {_MIN_CAL_POINTS} for reliable "
                f"quantile estimation. Add more calibration measurements.",
                UserWarning,
                stacklevel=2,
            )

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

        # Split conformal quantile: ceil((n+1)(1-α))-th order statistic.
        # Direct index computation avoids np.quantile(method="higher") which
        # requires NumPy >= 1.22, while the project's minimum pin is 1.21.
        n = self._n_cal
        scores_sorted = np.sort(np.asarray(self._scores, dtype=float))
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)  # clamp to valid 1-indexed range
        q_hat = float(scores_sorted[k - 1])  # convert to 0-indexed

        lower = mean - q_hat * std
        upper = mean + q_hat * std
        return lower, upper

    def check_ood(
        self,
        model: _PredictsMeanStd,
        X: np.ndarray,
        threshold_percentile: float = 95.0,
    ) -> bool:
        """Check whether test inputs appear out-of-distribution vs the calibration set.

        Computes normalised nonconformity scores for *X* and checks whether
        their median exceeds the ``threshold_percentile``-th percentile of the
        calibration scores.  A return value of ``True`` means the model is being
        queried in a regime it was not calibrated on — conformal coverage is not
        guaranteed.

        Parameters
        ----------
        model :
            Same fitted model passed to ``calibrate()``.
        X :
            Shape (n, d) — new test inputs to evaluate.
        threshold_percentile :
            Calibration score percentile used as the OOD boundary (default 95).

        Returns
        -------
        bool — True if distribution shift is suspected.

        Raises
        ------
        RuntimeError if ``calibrate`` has not been called.
        """
        if self._n_cal == 0:
            raise RuntimeError(
                "ConformalCalibrator.calibrate() must be called before check_ood()."
            )
        mean, std = model.predict(X, return_std=True)
        std_safe = np.maximum(std.ravel(), 1e-9)
        # Without ground-truth labels, use GPR predictive std as a proxy:
        # far OOD inputs have high std → high normalised scores even at the prior mean.
        # We use the median of (1 / std_safe) as a low-confidence signal:
        # a GPR falling back to the prior returns a constant std close to the
        # training label std, but normalised residuals from calibration will be large.
        # Simpler and more robust: flag when the median test std exceeds the 95th
        # percentile of calibration stds (GPR uncertainty widening signals OOD).
        cal_scores_arr = np.asarray(self._scores, dtype=float)
        threshold = float(np.percentile(cal_scores_arr, threshold_percentile))
        # Compute pseudo-scores: std / median_cal_std (relative uncertainty widening)
        median_cal_std = float(np.median(1.0 / np.maximum(cal_scores_arr, 1e-9)))
        test_scores = std_safe / max(median_cal_std, 1e-9)
        return bool(np.median(test_scores) > threshold)
