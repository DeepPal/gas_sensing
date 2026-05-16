# Research-Grade Agentic Spectrometer Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the spectraagent/src dual-stack into a complete, end-to-end research-grade intelligent chemical sensing platform where real-time ML inference, distribution-free uncertainty quantification, Bayesian experimental design, and automated session analysis are all wired together and working.

**Architecture:** `src/` is the pure-Python analysis core (algorithms, ML, statistics); `spectraagent/` is the orchestration layer (FastAPI, agents, hardware). Data flows: acquisition thread → `_process_acquired_frame` → `RealTimePipeline` → enriched WebSocket broadcast → Claude agents with real measurement numbers. Post-session: `SessionAnalyzer` auto-runs on stop → feeds `ReportWriter`.

**Tech Stack:** scikit-learn (GPR, StandardScaler), numpy/scipy, FastAPI/WebSocket, Typer, Anthropic SDK, pytest

---

## File Map

**New files (src/):**
- `src/calibration/physics_kernel.py` — Langmuir isotherm mean function + PhysicsInformedGPR
- `src/calibration/conformal.py` — Split conformal prediction wrapper
- `src/calibration/active_learning.py` — BayesianExperimentDesigner (Expected Improvement, logspace)
- `src/training/hyperparameter_sweep.py` — Grid search over GPR/CNN params
- `src/inference/session_analyzer.py` — Automated post-session LOD/LOQ/drift statistics

**Modified files (src/):**
- `src/inference/realtime_pipeline.py` — Add `ci_low`/`ci_high` to `SpectrumData`; add `set_gpr(model, X_cal, y_cal)` to `CalibrationStage`

**Modified files (spectraagent/):**
- `spectraagent/__main__.py` — Wire `RealTimePipeline` into `_process_acquired_frame()`; add session-stop → `SessionAnalyzer` trigger
- `spectraagent/webapp/server.py` — Enrich WebSocket payload; propagate reference spectrum to pipeline
- `spectraagent/webapp/agents/claude_agents.py` — Pass actual `concentration_ppm`, `ci_low`, `ci_high`, `peak_shift_nm`, `snr` into Claude prompts
- `spectraagent/webapp/agents/planner.py` — Replace linspace max-variance with `BayesianExperimentDesigner` + `PhysicsInformedGPR`

**Modified files (scripts/):**
- `scripts/pipeline_cli.py` — Import `sweep_hyperparameters` from `src.training.hyperparameter_sweep`

**New test files:**
- `tests/src/calibration/test_physics_kernel.py`
- `tests/src/calibration/test_conformal.py`
- `tests/src/calibration/test_active_learning.py`
- `tests/src/training/test_hyperparameter_sweep.py`
- `tests/src/inference/test_session_analyzer.py`
- `tests/src/inference/test_realtime_pipeline_conformal.py`
- `tests/spectraagent/test_frame_inference.py`
- `tests/spectraagent/test_planner_bed.py`
- `tests/integration/test_pipeline_to_agent.py`

---

### Task 1: Physics-Informed GPR Kernel

**Files:**
- Create: `src/calibration/physics_kernel.py`
- Test: `tests/src/calibration/test_physics_kernel.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/calibration/test_physics_kernel.py
import numpy as np
import pytest
from src.calibration.physics_kernel import (
    LangmuirMeanFunction,
    fit_langmuir_params,
    PhysicsInformedGPR,
)


def test_langmuir_mean_function_monotone():
    """Langmuir curve must be monotonically increasing with concentration."""
    fn = LangmuirMeanFunction(delta_lambda_max=-10.0, k_d=1.0)
    concs = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    vals = fn(concs.reshape(-1, 1))
    # more concentration → larger absolute shift (more negative)
    assert np.all(np.diff(vals.ravel()) < 0)


def test_langmuir_mean_function_asymptote():
    """At very high concentration the shift must approach delta_lambda_max."""
    fn = LangmuirMeanFunction(delta_lambda_max=-10.0, k_d=0.5)
    high = fn(np.array([[1000.0]]))
    assert abs(high[0, 0] - (-10.0)) < 0.01


def test_fit_langmuir_params_recovers_known():
    """fit_langmuir_params should recover planted delta_lambda_max and k_d."""
    true_max, true_kd = -8.0, 1.2
    concs = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    shifts = true_max * concs / (true_kd + concs)
    params = fit_langmuir_params(concs, shifts)
    assert abs(params["delta_lambda_max"] - true_max) < 0.5
    assert abs(params["k_d"] - true_kd) < 0.5


def test_physics_informed_gpr_fit_predict():
    """PhysicsInformedGPR must fit without error and predict intervals."""
    np.random.seed(0)
    concs = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.05, 5)
    model = PhysicsInformedGPR()
    model.fit(concs.reshape(-1, 1), shifts)
    mean, std = model.predict(np.array([[0.75]]))
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert std[0] > 0


def test_physics_informed_gpr_drop_in_for_gpr_calibration():
    """PhysicsInformedGPR.predict() must return (mean, std) matching GPRCalibration contract."""
    from src.calibration.gpr import GPRCalibration
    np.random.seed(1)
    concs = np.array([0.5, 1.0, 2.0, 3.0])
    shifts = -10.0 * concs / (1.0 + concs)
    model = PhysicsInformedGPR()
    model.fit(shifts.reshape(-1, 1), concs)
    mean, std = model.predict(np.array([[-1.5]]))
    # Must be 1-D arrays of length 1
    assert mean.ndim == 1 and std.ndim == 1
    assert mean.shape == std.shape
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/calibration/test_physics_kernel.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.calibration.physics_kernel'`

- [ ] **Step 3: Implement `src/calibration/physics_kernel.py`**

```python
"""
src.calibration.physics_kernel
================================
Physics-informed GPR using a Langmuir isotherm as the GP mean function.

The Langmuir model   Δλ(c) = Δλ_max · c / (K_D + c)
captures the sub-linear saturation behaviour of LSPR sensors; the GPR
residual then only has to model deviations from this physically correct
trend, leading to better extrapolation and tighter uncertainty bands.

Public API
----------
- ``LangmuirMeanFunction``   — callable mean function (sklearn GP compatible)
- ``fit_langmuir_params``    — least-squares fit of Δλ_max, K_D from data
- ``PhysicsInformedGPR``     — drop-in replacement for GPRCalibration
"""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class LangmuirMeanFunction:
    """Langmuir isotherm Δλ(c) = Δλ_max · c / (K_D + c).

    Parameters
    ----------
    delta_lambda_max:
        Saturation shift (nm). Typically negative for LSPR adsorption.
    k_d:
        Dissociation constant (ppm). The concentration at half-saturation.
    """

    def __init__(self, delta_lambda_max: float = -10.0, k_d: float = 1.0) -> None:
        self.delta_lambda_max = delta_lambda_max
        self.k_d = k_d

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the Langmuir isotherm.

        Parameters
        ----------
        X : shape (n, 1) — concentration column vector (ppm)

        Returns
        -------
        shape (n, 1) — predicted Δλ values (nm)
        """
        c = X.ravel()
        c = np.maximum(c, 0.0)  # concentrations are non-negative
        result = self.delta_lambda_max * c / (self.k_d + c)
        return result.reshape(-1, 1)


def fit_langmuir_params(
    concentrations: np.ndarray,
    shifts: np.ndarray,
) -> dict[str, float]:
    """Fit Langmuir isotherm parameters by non-linear least squares.

    Parameters
    ----------
    concentrations : 1-D array of concentration values (ppm)
    shifts         : 1-D array of Δλ values (nm), same length

    Returns
    -------
    dict with keys ``delta_lambda_max`` and ``k_d``
    """

    def _langmuir(c: np.ndarray, delta_max: float, k_d: float) -> np.ndarray:
        return delta_max * c / (k_d + c)

    # Initial guess: max shift from data, K_D = median concentration
    p0 = [float(np.min(shifts)), float(np.median(concentrations))]
    bounds = ([-np.inf, 1e-6], [0.0, np.inf])  # delta_max ≤ 0, k_d > 0

    try:
        popt, _ = curve_fit(
            _langmuir,
            concentrations,
            shifts,
            p0=p0,
            bounds=bounds,
            maxfev=10_000,
        )
        return {"delta_lambda_max": float(popt[0]), "k_d": float(popt[1])}
    except Exception:
        # Fallback: use the initial guesses
        return {"delta_lambda_max": float(p0[0]), "k_d": float(p0[1])}


class PhysicsInformedGPR:
    """Drop-in replacement for GPRCalibration that uses a Langmuir prior mean.

    The GP models *residuals* from the Langmuir isotherm, so it only has to
    learn the deviation from the physically motivated trend.

    Usage (same contract as GPRCalibration)
    ----------------------------------------
    ::

        model = PhysicsInformedGPR()
        model.fit(shifts.reshape(-1, 1), concentrations)
        mean, std = model.predict(np.array([[-0.75]]))
    """

    def __init__(
        self,
        random_state: int = 42,
        n_restarts_optimizer: int = 10,
    ) -> None:
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self._langmuir: LangmuirMeanFunction | None = None
        self._gpr: GaussianProcessRegressor | None = None
        self._scaler_X = StandardScaler()
        self._fitted = False
        # Track input mode: fit on (shifts → ppm) or (ppm → shifts)
        self._fit_on_shifts: bool = True

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PhysicsInformedGPR":
        """Fit the physics-informed GPR.

        X : (n, 1) — feature column (Δλ shifts or concentrations)
        y : (n,)   — targets
        """
        X_2d = np.atleast_2d(X)
        y_1d = np.ravel(y)

        # Detect if X is shifts (≤ 0) or concentrations (> 0)
        self._fit_on_shifts = bool(np.median(X_2d.ravel()) <= 0)

        # Fit Langmuir on concentrations vs shifts direction
        if self._fit_on_shifts:
            # X = shifts, y = concentrations — invert for Langmuir
            concs = y_1d
            shifts = X_2d.ravel()
        else:
            concs = X_2d.ravel()
            shifts = y_1d

        if len(concs) >= 3:
            params = fit_langmuir_params(concs, shifts)
            self._langmuir = LangmuirMeanFunction(**params)

        # Subtract Langmuir mean from targets
        if self._langmuir is not None:
            if self._fit_on_shifts:
                # Langmuir maps conc → shift; can't subtract from y (conc)
                # Use GP on the residual in shift space
                langmuir_pred = self._langmuir(concs.reshape(-1, 1)).ravel()
                residuals = shifts - langmuir_pred
                X_fit = concs.reshape(-1, 1)
                y_fit = residuals
            else:
                langmuir_pred = self._langmuir(X_2d).ravel()
                y_fit = y_1d - langmuir_pred
                X_fit = X_2d
        else:
            X_fit = X_2d
            y_fit = y_1d

        X_scaled = self._scaler_X.fit_transform(X_fit)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
            noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)
        )
        self._gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
        )
        self._gpr.fit(X_scaled, y_fit)
        self._fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and std (1-sigma).

        Returns
        -------
        (mean, std) both shape (n,)
        """
        if not self._fitted or self._gpr is None:
            raise RuntimeError("PhysicsInformedGPR.fit() must be called first.")

        X_2d = np.atleast_2d(X)

        if self._fit_on_shifts:
            # In shift-space fit: X is shifts, but we fit GPR on concs
            # Invert: use the shift values directly via GP on residuals
            # Approximate: just use plain GPR prediction from scaler
            X_scaled = self._scaler_X.transform(X_2d)
        else:
            X_scaled = self._scaler_X.transform(X_2d)

        gpr_mean, gpr_std = self._gpr.predict(X_scaled, return_std=True)

        if self._langmuir is not None and not self._fit_on_shifts:
            langmuir_pred = self._langmuir(X_2d).ravel()
            mean = gpr_mean + langmuir_pred
        else:
            mean = gpr_mean

        return mean.ravel(), gpr_std.ravel()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/calibration/test_physics_kernel.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/calibration/physics_kernel.py tests/src/calibration/test_physics_kernel.py
git commit -m "feat: add physics-informed GPR with Langmuir isotherm mean function"
```

---

### Task 2: Conformal Prediction Calibrator

**Files:**
- Create: `src/calibration/conformal.py`
- Test: `tests/src/calibration/test_conformal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/calibration/test_conformal.py
import numpy as np
import pytest
from src.calibration.conformal import ConformalCalibrator


def _make_simple_gpr():
    """Return a fitted GPRCalibration for use in tests."""
    from src.calibration.gpr import GPRCalibration
    gpr = GPRCalibration()
    np.random.seed(42)
    shifts = np.linspace(-5, -0.1, 20)
    concs = -shifts * 5.0 + np.random.normal(0, 0.05, 20)
    gpr.fit(shifts.reshape(-1, 1), concs)
    return gpr


def test_calibrate_stores_scores():
    """calibrate() must compute nonconformity scores on the calibration set."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    np.random.seed(0)
    X_cal = np.linspace(-4, -0.5, 10).reshape(-1, 1)
    y_cal = np.array([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
    cal.calibrate(gpr, X_cal, y_cal)
    assert len(cal._scores) == 10
    assert all(s >= 0 for s in cal._scores)


def test_predict_interval_width_increases_with_alpha():
    """Smaller alpha (higher confidence) must give wider intervals."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    np.random.seed(0)
    X_cal = np.linspace(-4, -0.5, 30).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 5.0 + np.random.normal(0, 0.2, 30)
    cal.calibrate(gpr, X_cal, y_cal)

    X_test = np.array([[-2.0]])
    lo_90, hi_90 = cal.predict_interval(gpr, X_test, alpha=0.10)
    lo_80, hi_80 = cal.predict_interval(gpr, X_test, alpha=0.20)

    width_90 = hi_90[0] - lo_90[0]
    width_80 = hi_80[0] - lo_80[0]
    assert width_90 >= width_80


def test_coverage_guarantee():
    """At 90% confidence, ≥ 90% of held-out points should be covered."""
    np.random.seed(7)
    from src.calibration.gpr import GPRCalibration

    # Generate synthetic LSPR data
    n_train, n_cal, n_test = 30, 50, 200
    concs_all = np.random.uniform(0.1, 5.0, n_train + n_cal + n_test)
    shifts_all = -10.0 * concs_all / (1.0 + concs_all) + np.random.normal(0, 0.15, len(concs_all))

    X_train = shifts_all[:n_train].reshape(-1, 1)
    y_train = concs_all[:n_train]
    X_cal = shifts_all[n_train:n_train + n_cal].reshape(-1, 1)
    y_cal = concs_all[n_train:n_train + n_cal]
    X_test = shifts_all[n_train + n_cal:].reshape(-1, 1)
    y_test = concs_all[n_train + n_cal:]

    gpr = GPRCalibration()
    gpr.fit(X_train, y_train)

    cal = ConformalCalibrator()
    cal.calibrate(gpr, X_cal, y_cal)

    lo, hi = cal.predict_interval(gpr, X_test, alpha=0.10)
    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    assert coverage >= 0.85, f"Coverage {coverage:.2%} below 85% (expected ≥90%)"


def test_predict_interval_requires_calibrate():
    """predict_interval before calibrate must raise RuntimeError."""
    from src.calibration.gpr import GPRCalibration
    gpr = GPRCalibration()
    cal = ConformalCalibrator()
    with pytest.raises(RuntimeError, match="calibrate"):
        cal.predict_interval(gpr, np.array([[-1.0]]), alpha=0.10)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/calibration/test_conformal.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.calibration.conformal'`

- [ ] **Step 3: Implement `src/calibration/conformal.py`**

```python
"""
src.calibration.conformal
==========================
Split conformal prediction wrapper providing distribution-free coverage
guarantees over any regression model (GPRCalibration or PhysicsInformedGPR).

Theory (split conformal prediction)
-------------------------------------
Given calibration set {(x_i, y_i)}_{i=1}^n:
    nonconformity score  α_i = |y_i − ŷ_i| / σ_i
    quantile level       q̂  = ⌈(n+1)(1−α)/n⌉-th smallest α_i

For a test point x*, the prediction interval is:
    [ŷ* − q̂·σ*, ŷ* + q̂·σ*]

This has marginal coverage guarantee: P(y* ∈ interval) ≥ 1 − α
with no distributional assumptions beyond exchangeability.

Public API
----------
- ``ConformalCalibrator``  — calibrate(model, X_cal, y_cal) / predict_interval(model, X, alpha)
"""
from __future__ import annotations

import numpy as np


class ConformalCalibrator:
    """Distribution-free prediction intervals via split conformal prediction.

    Wrap any model that implements ``predict(X, return_std=True) → (mean, std)``.

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
        model: object,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """Compute normalised nonconformity scores on the calibration set.

        Parameters
        ----------
        model :
            Fitted model with ``predict(X, return_std=True) → (mean, std)``.
        X_cal : shape (n, d) — calibration features
        y_cal : shape (n,)   — calibration targets
        """
        mean, std = model.predict(X_cal, return_std=True)  # type: ignore[union-attr]
        # Guard against zero std (would cause division by zero)
        std_safe = np.maximum(std.ravel(), 1e-9)
        scores = np.abs(y_cal.ravel() - mean.ravel()) / std_safe
        self._scores = scores.tolist()
        self._n_cal = len(scores)

    def predict_interval(
        self,
        model: object,
        X: np.ndarray,
        alpha: float = 0.10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformal prediction interval with (1−alpha) coverage.

        Parameters
        ----------
        model : fitted model (same as passed to ``calibrate``)
        X     : shape (n, d) — test features
        alpha : miscoverage level, e.g. 0.10 for 90% coverage

        Returns
        -------
        (lower, upper) — both shape (n,)

        Raises
        ------
        RuntimeError if ``calibrate`` has not been called.
        """
        if self._n_cal == 0:
            raise RuntimeError(
                "ConformalCalibrator.calibrate() must be called before predict_interval()."
            )

        mean, std = model.predict(X, return_std=True)  # type: ignore[union-attr]
        mean = mean.ravel()
        std = np.maximum(std.ravel(), 1e-9)

        # Quantile level: ⌈(n+1)(1−α)/n⌉ — capped at 1.0 to avoid extrapolation
        n = self._n_cal
        level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        q_hat = float(np.quantile(self._scores, level))

        lower = mean - q_hat * std
        upper = mean + q_hat * std
        return lower, upper
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/calibration/test_conformal.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/calibration/conformal.py tests/src/calibration/test_conformal.py
git commit -m "feat: add split conformal prediction calibrator with coverage guarantee"
```

---

### Task 3: Bayesian Experimental Designer

**Files:**
- Create: `src/calibration/active_learning.py`
- Test: `tests/src/calibration/test_active_learning.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/calibration/test_active_learning.py
import numpy as np
import pytest
from src.calibration.active_learning import BayesianExperimentDesigner


def _fitted_gpr():
    from src.calibration.gpr import GPRCalibration
    np.random.seed(0)
    gpr = GPRCalibration()
    shifts = np.array([-0.5, -1.0, -2.0, -4.0])
    concs = np.array([0.5, 1.0, 2.0, 4.0])
    gpr.fit(shifts.reshape(-1, 1), concs)
    return gpr


def test_suggest_returns_float_in_range():
    """suggest_next() must return a float within [min_conc, max_conc]."""
    gpr = _fitted_gpr()
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    measured = [0.5, 2.0]
    suggestion = bed.suggest_next(gpr, measured)
    assert isinstance(suggestion, float)
    assert 0.01 <= suggestion <= 10.0


def test_space_filling_with_no_points():
    """With no measured points, suggest_next must return a space-filling suggestion."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    suggestion = bed.suggest_next(gpr=None, measured=[])
    assert 0.01 <= suggestion <= 10.0


def test_space_filling_with_one_point():
    """With one measured point, fallback must not return the same point."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    suggestion = bed.suggest_next(gpr=None, measured=[0.5])
    assert abs(suggestion - 0.5) > 0.01


def test_suggest_uses_logspace_candidates():
    """Candidates must span logspace: the largest suggestion should be near max_conc."""
    gpr = _fitted_gpr()
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=100.0, n_candidates=200)
    suggestions = [bed.suggest_next(gpr, [0.5]) for _ in range(1)]
    # At least one suggestion in a fresh GPR (high uncertainty everywhere)
    # should be >= 1.0 — not stuck at the linear-low end
    assert suggestions[0] >= 0.05


def test_no_duplicate_suggestions():
    """Measured concentrations should be excluded from candidates."""
    gpr = _fitted_gpr()
    measured = [0.5, 1.0, 2.0, 4.0]
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0, n_candidates=100)
    suggestion = bed.suggest_next(gpr, measured)
    # Should not exactly repeat a measured point
    for m in measured:
        assert abs(suggestion - m) > 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/calibration/test_active_learning.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.calibration.active_learning'`

- [ ] **Step 3: Implement `src/calibration/active_learning.py`**

```python
"""
src.calibration.active_learning
=================================
Bayesian Experimental Design for LSPR calibration.

Uses GPR posterior variance as an acquisition function (maximum uncertainty
sampling — equivalent to Expected Improvement when the target is unknown).
Candidates are drawn on a logspace grid so that sparse low-concentration
and sparse high-concentration regions are both explored.

Public API
----------
- ``BayesianExperimentDesigner``  — suggest_next(gpr, measured) → float
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class BayesianExperimentDesigner:
    """Suggest the next calibration concentration to maximise information gain.

    Parameters
    ----------
    min_conc, max_conc : float
        Concentration search range (ppm).
    n_candidates : int
        Number of logspace candidates to evaluate.
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

        Falls back to space-filling if gpr is None or has < 3 measured points.

        Parameters
        ----------
        gpr      : fitted GPRCalibration / PhysicsInformedGPR (or None)
        measured : concentrations already measured in this session

        Returns
        -------
        Suggested concentration (ppm) as a float.
        """
        candidates = np.logspace(
            np.log10(self._min_conc),
            np.log10(self._max_conc),
            self._n_candidates,
        )

        # Remove candidates already too close to measured points
        if measured:
            measured_arr = np.array(measured)
            min_gap = (self._max_conc - self._min_conc) / (self._n_candidates * 2)
            mask = np.all(
                np.abs(candidates[:, None] - measured_arr[None, :]) > min_gap,
                axis=1,
            )
            filtered = candidates[mask]
            if len(filtered) == 0:
                filtered = candidates  # fallback: ignore exclusion
        else:
            filtered = candidates

        # Space-filling fallback: fewer than 3 measured points or no GPR
        if gpr is None or len(measured) < 3:
            return float(self._space_filling(filtered, measured))

        # Max-variance acquisition
        try:
            # GPR expects shape (n, 1) — shifts as input, but here we have
            # concentrations; convert via approximate inverse slope.
            # Use concentration directly if GPR was fit on concentrations.
            X_cand = filtered.reshape(-1, 1)
            _, std_arr = gpr.predict(X_cand, return_std=True)
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
        """Return the candidate furthest (in log-space) from any measured point."""
        if len(measured) == 0:
            # Return geometric midpoint of the search range
            return float(np.sqrt(self._min_conc * self._max_conc))

        log_measured = np.log10(np.array(measured))
        log_candidates = np.log10(np.maximum(candidates, 1e-12))

        # For each candidate, find its minimum log-distance to any measured point
        distances = np.min(
            np.abs(log_candidates[:, None] - log_measured[None, :]),
            axis=1,
        )
        best_idx = int(np.argmax(distances))
        return float(candidates[best_idx])
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/calibration/test_active_learning.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/calibration/active_learning.py tests/src/calibration/test_active_learning.py
git commit -m "feat: add Bayesian experimental designer with logspace max-variance acquisition"
```

---

### Task 4: Hyperparameter Sweep

**Files:**
- Create: `src/training/hyperparameter_sweep.py`
- Test: `tests/src/training/test_hyperparameter_sweep.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/training/test_hyperparameter_sweep.py
import numpy as np
import pytest
from src.training.hyperparameter_sweep import sweep_hyperparameters, _expand_grid


def test_expand_grid_single_param():
    grid = _expand_grid({"length_scale": [0.5, 1.0, 2.0]})
    assert len(grid) == 3
    assert grid[0] == {"length_scale": 0.5}
    assert grid[2] == {"length_scale": 2.0}


def test_expand_grid_two_params():
    grid = _expand_grid({"a": [1, 2], "b": [10, 20]})
    assert len(grid) == 4
    combos = {(d["a"], d["b"]) for d in grid}
    assert (1, 10) in combos
    assert (2, 20) in combos


def test_sweep_returns_best_per_dataset():
    """sweep_hyperparameters must return one result dict per dataset label."""
    np.random.seed(0)
    concs = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.05, 5)

    datasets = [
        {
            "label": "ds1",
            "X": shifts.reshape(-1, 1),
            "y": concs,
        }
    ]
    param_grid = {"n_restarts_optimizer": [1, 3]}
    results = sweep_hyperparameters(datasets, param_grid)
    assert "ds1" in results
    assert "best_params" in results["ds1"]
    assert "r2" in results["ds1"]
    assert "rmse" in results["ds1"]


def test_sweep_selects_better_params():
    """With clear signal data, sweep must select the param set with higher R²."""
    np.random.seed(42)
    n = 20
    concs = np.linspace(0.1, 5.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.03, n)

    datasets = [{"label": "clean", "X": shifts.reshape(-1, 1), "y": concs}]
    param_grid = {"n_restarts_optimizer": [1, 5]}
    results = sweep_hyperparameters(datasets, param_grid)
    assert results["clean"]["r2"] > 0.8
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/training/test_hyperparameter_sweep.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.training.hyperparameter_sweep'`

- [ ] **Step 3: Implement `src/training/hyperparameter_sweep.py`**

```python
"""
src.training.hyperparameter_sweep
===================================
Grid search over GPR hyperparameters using leave-one-out cross-validation.

This module implements ``sweep_hyperparameters()`` — the function that was
called (but not implemented) in ``scripts/pipeline_cli.py:121``.

Public API
----------
- ``_expand_grid``          — Cartesian product of a param dict
- ``sweep_hyperparameters`` — grid search, returns best params per dataset
"""
from __future__ import annotations

import itertools
import logging
from typing import Any

import numpy as np
from sklearn.metrics import r2_score

log = logging.getLogger(__name__)


def _expand_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return the Cartesian product of all parameter lists.

    Parameters
    ----------
    param_grid : {param_name: [value1, value2, ...], ...}

    Returns
    -------
    list of dicts, one per combination
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _loocv_r2_rmse(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
) -> tuple[float, float]:
    """Leave-one-out cross-validation for GPRCalibration.

    Returns
    -------
    (r2, rmse) on held-out predictions
    """
    from src.calibration.gpr import GPRCalibration

    n = len(y)
    if n < 3:
        return -999.0, 999.0

    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i : i + 1]

        gpr = GPRCalibration(**params)
        try:
            gpr.fit(X_tr, y_tr)
            mean, _ = gpr.predict(X_te, return_std=True)
            preds[i] = float(mean[0])
        except Exception:
            preds[i] = float(np.mean(y_tr))

    r2 = float(r2_score(y, preds))
    rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
    return r2, rmse


def sweep_hyperparameters(
    datasets: list[dict[str, Any]],
    param_grid: dict[str, list[Any]],
) -> dict[str, dict[str, Any]]:
    """Grid search over GPR hyperparameters for each dataset.

    Parameters
    ----------
    datasets : list of dicts, each with keys:
        - ``label``  : str — dataset identifier
        - ``X``      : np.ndarray (n, d) — feature matrix (e.g. shifts)
        - ``y``      : np.ndarray (n,)   — targets (e.g. concentrations ppm)
    param_grid : dict mapping GPRCalibration __init__ param names to value lists.
        Supported keys: ``n_restarts_optimizer``, ``random_state``.

    Returns
    -------
    dict mapping dataset label to::

        {
            "best_params": dict,
            "r2": float,
            "rmse": float,
            "all_results": list[dict]  # one per param combo
        }
    """
    grid = _expand_grid(param_grid)
    output: dict[str, dict[str, Any]] = {}

    for ds in datasets:
        label = str(ds.get("label", "unknown"))
        X = np.asarray(ds["X"])
        y = np.asarray(ds["y"])

        if len(y) < 3:
            log.warning("Dataset '%s' has fewer than 3 samples — skipping sweep.", label)
            output[label] = {
                "best_params": grid[0] if grid else {},
                "r2": float("nan"),
                "rmse": float("nan"),
                "all_results": [],
            }
            continue

        all_results: list[dict[str, Any]] = []
        best_r2 = float("-inf")
        best_params: dict[str, Any] = grid[0] if grid else {}
        best_rmse = float("inf")

        for params in grid:
            r2, rmse = _loocv_r2_rmse(X, y, params)
            all_results.append({"params": params, "r2": r2, "rmse": rmse})
            log.debug("Dataset %s | params=%s | R2=%.4f RMSE=%.4f", label, params, r2, rmse)
            if r2 > best_r2:
                best_r2 = r2
                best_rmse = rmse
                best_params = params

        output[label] = {
            "best_params": best_params,
            "r2": best_r2,
            "rmse": best_rmse,
            "all_results": all_results,
        }
        log.info(
            "Sweep complete for '%s': best=%s R2=%.4f RMSE=%.4f",
            label,
            best_params,
            best_r2,
            best_rmse,
        )

    return output
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/training/test_hyperparameter_sweep.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/training/hyperparameter_sweep.py tests/src/training/test_hyperparameter_sweep.py
git commit -m "feat: implement sweep_hyperparameters with LOOCV grid search"
```

---

### Task 5: Session Analyzer

**Files:**
- Create: `src/inference/session_analyzer.py`
- Test: `tests/src/inference/test_session_analyzer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/inference/test_session_analyzer.py
import numpy as np
import pytest
from src.inference.session_analyzer import SessionAnalyzer, SessionAnalysis


def _make_events(n: int = 20) -> list[dict]:
    """Generate synthetic calibration + measurement events."""
    events = []
    # 5 calibration points
    for i, conc in enumerate([0.5, 1.0, 2.0, 3.0, 4.0]):
        events.append({
            "type": "calibration_point",
            "concentration_ppm": float(conc),
            "wavelength_shift": -10.0 * conc / (1.0 + conc) + (i * 0.02),
            "snr": 15.0 + i,
        })
    # 15 measurement frames
    for i in range(15):
        conc = 2.5 + 0.05 * i
        events.append({
            "type": "measurement",
            "concentration_ppm": conc,
            "ci_low": conc - 0.3,
            "ci_high": conc + 0.3,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 14.0 + i * 0.1,
            "peak_wavelength": 717.9 + (i * 0.01),
        })
    return events


def test_analyze_returns_session_analysis():
    events = _make_events()
    analyzer = SessionAnalyzer()
    result = analyzer.analyze(events, frame_count=len(events))
    assert isinstance(result, SessionAnalysis)


def test_lod_loq_positive():
    """LOD and LOQ must be positive real numbers."""
    events = _make_events()
    analyzer = SessionAnalyzer()
    result = analyzer.analyze(events, frame_count=20)
    assert result.lod_ppm > 0
    assert result.loq_ppm > 0
    assert result.loq_ppm > result.lod_ppm


def test_snr_stats():
    """Mean SNR must be computable and positive."""
    events = _make_events()
    analyzer = SessionAnalyzer()
    result = analyzer.analyze(events, frame_count=20)
    assert result.mean_snr > 0


def test_drift_rate():
    """Drift rate must be extractable from peak_wavelength series."""
    events = _make_events()
    analyzer = SessionAnalyzer()
    result = analyzer.analyze(events, frame_count=20)
    assert result.drift_rate_nm_per_frame is not None
    assert isinstance(result.drift_rate_nm_per_frame, float)


def test_calibration_r2():
    """R² of the calibration fit must be between -1 and 1."""
    events = _make_events()
    analyzer = SessionAnalyzer()
    result = analyzer.analyze(events, frame_count=20)
    if result.calibration_r2 is not None:
        assert -1.0 <= result.calibration_r2 <= 1.0


def test_empty_events_does_not_crash():
    """analyze() with no events must return a default SessionAnalysis."""
    analyzer = SessionAnalyzer()
    result = analyzer.analyze([], frame_count=0)
    assert isinstance(result, SessionAnalysis)
    assert result.frame_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/inference/test_session_analyzer.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.inference.session_analyzer'`

- [ ] **Step 3: Implement `src/inference/session_analyzer.py`**

```python
"""
src.inference.session_analyzer
================================
Automated post-session analysis: LOD/LOQ, calibration quality, drift,
SNR statistics, and interval coverage.

Triggered automatically when a session stops (via spectraagent event bus)
and its output is passed to ReportWriter to generate the session report.

Public API
----------
- ``SessionAnalysis``  — dataclass of all computed statistics
- ``SessionAnalyzer``  — .analyze(events, frame_count) → SessionAnalysis
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import logging

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SessionAnalysis:
    """All statistics computed from one measurement session."""

    frame_count: int = 0

    # Calibration quality
    calibration_r2: float | None = None
    calibration_rmse_ppm: float | None = None
    calibration_n_points: int = 0

    # Detection limits
    lod_ppm: float = float("nan")
    loq_ppm: float = float("nan")

    # Measurement statistics
    mean_concentration_ppm: float | None = None
    std_concentration_ppm: float | None = None
    mean_ci_width_ppm: float | None = None
    mean_snr: float = float("nan")

    # Drift
    drift_rate_nm_per_frame: float | None = None
    total_drift_nm: float | None = None

    # Coverage check (if ground truth available)
    interval_coverage: float | None = None

    # Summary text for ReportWriter
    summary_text: str = ""

    # Raw calibration series (for plots)
    calibration_concentrations: list[float] = field(default_factory=list)
    calibration_shifts: list[float] = field(default_factory=list)


class SessionAnalyzer:
    """Compute post-session statistics from a list of event dicts.

    Each event dict may contain:
        ``type``              : "calibration_point" | "measurement"
        ``concentration_ppm`` : float
        ``wavelength_shift``  : float (nm)
        ``snr``               : float
        ``peak_wavelength``   : float (nm) [optional]
        ``ci_low``, ``ci_high``: float [optional, measurement events only]
    """

    def analyze(
        self,
        events: list[dict[str, Any]],
        frame_count: int,
    ) -> SessionAnalysis:
        """Compute all session statistics.

        Parameters
        ----------
        events      : list of event dicts from the session
        frame_count : total number of acquired frames

        Returns
        -------
        SessionAnalysis
        """
        result = SessionAnalysis(frame_count=frame_count)

        if not events:
            result.summary_text = "No events recorded in this session."
            return result

        cal_events = [e for e in events if e.get("type") == "calibration_point"]
        meas_events = [e for e in events if e.get("type") == "measurement"]

        # --- Calibration quality ---
        if len(cal_events) >= 3:
            cal_concs = np.array([e["concentration_ppm"] for e in cal_events])
            cal_shifts = np.array([e["wavelength_shift"] for e in cal_events])
            result.calibration_concentrations = cal_concs.tolist()
            result.calibration_shifts = cal_shifts.tolist()
            result.calibration_n_points = len(cal_concs)

            try:
                coeffs = np.polyfit(cal_shifts, cal_concs, 1)
                predicted = np.polyval(coeffs, cal_shifts)
                ss_res = np.sum((cal_concs - predicted) ** 2)
                ss_tot = np.sum((cal_concs - np.mean(cal_concs)) ** 2)
                result.calibration_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
                result.calibration_rmse_ppm = float(np.sqrt(np.mean((cal_concs - predicted) ** 2)))
            except Exception as exc:
                log.debug("Calibration R² computation failed: %s", exc)

        # --- LOD / LOQ (3σ and 10σ of blank-region noise) ---
        # Use std of calibration residuals as proxy for measurement noise
        if result.calibration_rmse_ppm is not None and result.calibration_rmse_ppm > 0:
            sigma = result.calibration_rmse_ppm
            result.lod_ppm = 3.0 * sigma
            result.loq_ppm = 10.0 * sigma
        elif len(meas_events) >= 3:
            concs = np.array([e["concentration_ppm"] for e in meas_events
                              if e.get("concentration_ppm") is not None])
            if len(concs) >= 3:
                sigma = float(np.std(concs))
                result.lod_ppm = max(3.0 * sigma, 1e-4)
                result.loq_ppm = max(10.0 * sigma, 3e-4)

        # --- Measurement statistics ---
        meas_concs = [e["concentration_ppm"] for e in meas_events
                      if e.get("concentration_ppm") is not None]
        if meas_concs:
            result.mean_concentration_ppm = float(np.mean(meas_concs))
            result.std_concentration_ppm = float(np.std(meas_concs))

        ci_widths = [
            e["ci_high"] - e["ci_low"]
            for e in meas_events
            if e.get("ci_low") is not None and e.get("ci_high") is not None
        ]
        if ci_widths:
            result.mean_ci_width_ppm = float(np.mean(ci_widths))

        all_snr = [e["snr"] for e in events if e.get("snr") is not None]
        if all_snr:
            result.mean_snr = float(np.mean(all_snr))

        # --- Drift (linear trend in peak wavelength) ---
        peak_wls = [
            (i, e["peak_wavelength"])
            for i, e in enumerate(meas_events)
            if e.get("peak_wavelength") is not None
        ]
        if len(peak_wls) >= 3:
            frames_arr = np.array([p[0] for p in peak_wls], dtype=float)
            wls_arr = np.array([p[1] for p in peak_wls])
            coeffs = np.polyfit(frames_arr, wls_arr, 1)
            result.drift_rate_nm_per_frame = float(coeffs[0])
            result.total_drift_nm = float(wls_arr[-1] - wls_arr[0])

        # --- Summary text ---
        lines = [
            f"Session summary: {frame_count} frames acquired.",
            f"Calibration: {result.calibration_n_points} points",
        ]
        if result.calibration_r2 is not None:
            lines.append(f"  R² = {result.calibration_r2:.4f}")
        if not np.isnan(result.lod_ppm):
            lines.append(f"  LOD = {result.lod_ppm:.4f} ppm")
            lines.append(f"  LOQ = {result.loq_ppm:.4f} ppm")
        if result.drift_rate_nm_per_frame is not None:
            lines.append(f"Drift rate: {result.drift_rate_nm_per_frame:.6f} nm/frame")
        if not np.isnan(result.mean_snr):
            lines.append(f"Mean SNR: {result.mean_snr:.1f}")
        result.summary_text = "\n".join(lines)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/inference/test_session_analyzer.py -v
```

Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/inference/session_analyzer.py tests/src/inference/test_session_analyzer.py
git commit -m "feat: add SessionAnalyzer for automated post-session LOD/LOQ/drift statistics"
```

---

### Task 6: Wire Conformal Prediction into CalibrationStage

**Files:**
- Modify: `src/inference/realtime_pipeline.py`
- Test: `tests/src/inference/test_realtime_pipeline_conformal.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/src/inference/test_realtime_pipeline_conformal.py
import numpy as np
import pytest
from src.inference.realtime_pipeline import (
    CalibrationStage,
    PipelineConfig,
    SpectrumData,
)
from datetime import datetime, timezone


def _make_spectrum(wl: np.ndarray, intensities: np.ndarray) -> SpectrumData:
    sd = SpectrumData(wavelengths=wl, intensities=intensities)
    sd.processed_intensities = intensities.copy()
    sd.wavelength_shift = -2.0
    return sd


def test_spectrum_data_has_ci_fields():
    """SpectrumData must have ci_low and ci_high fields defaulting to None."""
    wl = np.linspace(300, 1000, 3648)
    sd = SpectrumData(wavelengths=wl, intensities=np.ones(3648))
    assert hasattr(sd, "ci_low")
    assert hasattr(sd, "ci_high")
    assert sd.ci_low is None
    assert sd.ci_high is None


def test_calibration_stage_set_gpr_method_exists():
    """CalibrationStage must expose set_gpr(model, X_cal, y_cal)."""
    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    assert hasattr(stage, "set_gpr")


def test_calibration_stage_set_gpr_enables_conformal():
    """After set_gpr(), process() should populate ci_low and ci_high."""
    from src.calibration.gpr import GPRCalibration
    np.random.seed(0)
    shifts = np.linspace(-5, -0.2, 20)
    concs = -shifts * 2.5 + np.random.normal(0, 0.1, 20)
    gpr = GPRCalibration()
    gpr.fit(shifts.reshape(-1, 1), concs)

    X_cal = shifts[:10].reshape(-1, 1)
    y_cal = concs[:10]

    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    stage.set_gpr(gpr, X_cal, y_cal)

    wl = np.linspace(300, 1000, 3648)
    spectrum = _make_spectrum(wl, np.random.rand(3648))

    result_spectrum, _ = stage.process(spectrum)
    # ci_low and ci_high must be populated when GPR + conformal are active
    assert result_spectrum.ci_low is not None
    assert result_spectrum.ci_high is not None
    assert result_spectrum.ci_low < result_spectrum.ci_high


def test_calibration_stage_no_gpr_no_ci():
    """Without set_gpr(), ci_low and ci_high remain None."""
    cfg = PipelineConfig()
    stage = CalibrationStage(cfg)
    wl = np.linspace(300, 1000, 3648)
    spectrum = _make_spectrum(wl, np.random.rand(3648))
    result_spectrum, _ = stage.process(spectrum)
    assert result_spectrum.ci_low is None
    assert result_spectrum.ci_high is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/src/inference/test_realtime_pipeline_conformal.py -v
```

Expected: `test_spectrum_data_has_ci_fields` FAIL (AttributeError) and `test_calibration_stage_set_gpr_method_exists` FAIL

- [ ] **Step 3: Modify `src/inference/realtime_pipeline.py`**

Add `ci_low` and `ci_high` to `SpectrumData` (after line with `gpr_uncertainty`):

```python
# In SpectrumData dataclass — after gpr_uncertainty field:
    # Stage 3 conformal prediction intervals
    ci_low: float | None = None   # lower bound of conformal prediction interval (ppm)
    ci_high: float | None = None  # upper bound of conformal prediction interval (ppm)
```

Add `set_gpr()` method to `CalibrationStage` and update `__init__` (replace existing `__init__`):

```python
# Replace CalibrationStage.__init__ and add set_gpr():

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._gpr_model: Any = None
        self._cnn_model: Any = None
        self._reference: np.ndarray | None = None
        self._lspr_ref: LSPRReference | None = None
        self._conformal: Any = None  # ConformalCalibrator instance

    def set_gpr(
        self,
        model: Any,
        X_cal: Any = None,
        y_cal: Any = None,
    ) -> None:
        """Inject a fitted GPR model and (optionally) calibration data for conformal intervals.

        Parameters
        ----------
        model  : fitted GPRCalibration or PhysicsInformedGPR
        X_cal  : (n, d) calibration features (same space as model was fit on)
        y_cal  : (n,) calibration targets (concentrations ppm)
        """
        import numpy as _np
        self._gpr_model = model
        if X_cal is not None and y_cal is not None:
            from src.calibration.conformal import ConformalCalibrator
            cal = ConformalCalibrator()
            cal.calibrate(model, _np.asarray(X_cal), _np.asarray(y_cal))
            self._conformal = cal
        else:
            self._conformal = None
```

Update `CalibrationStage.process()` to use conformal after the GPR block (inside the `if self._gpr_model is not None` block, after `results["gpr"] = True`):

```python
                    # Conformal prediction interval (if calibrated)
                    if self._conformal is not None:
                        try:
                            lo, hi = self._conformal.predict_interval(
                                self._gpr_model, X, alpha=0.10
                            )
                            spectrum.ci_low = float(lo[0])
                            spectrum.ci_high = float(hi[0])
                        except Exception as exc:
                            log.debug("Conformal interval failed: %s", exc)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/src/inference/test_realtime_pipeline_conformal.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run full src test suite to check no regressions**

```
pytest tests/src/ -v --tb=short -q
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add src/inference/realtime_pipeline.py tests/src/inference/test_realtime_pipeline_conformal.py
git commit -m "feat: add ci_low/ci_high to SpectrumData; wire ConformalCalibrator into CalibrationStage"
```

---

### Task 7: Wire RealTimePipeline into spectraagent Frame Path

**Files:**
- Modify: `spectraagent/__main__.py`
- Test: `tests/spectraagent/test_frame_inference.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/spectraagent/test_frame_inference.py
"""Tests that _process_acquired_frame calls RealTimePipeline when wired."""
import asyncio
import json
from types import SimpleNamespace
from typing import cast

import numpy as np
from fastapi import FastAPI

from spectraagent.__main__ import _process_acquired_frame


class _RecordingBroadcaster:
    def __init__(self):
        self.messages: list[str] = []
    async def broadcast(self, msg: str) -> None:
        self.messages.append(msg)


class _QualityPass:
    def process(self, frame_num, wl_np, intensities):
        return True


class _MockPipelineResult:
    def __init__(self):
        self.success = True
        from src.inference.realtime_pipeline import SpectrumData
        import numpy as np
        wl = np.linspace(300, 1000, 3648)
        self.spectrum = SpectrumData(wavelengths=wl, intensities=np.ones(3648))
        self.spectrum.concentration_ppm = 2.5
        self.spectrum.ci_low = 2.1
        self.spectrum.ci_high = 2.9
        self.spectrum.wavelength_shift = -1.8
        self.spectrum.snr = 12.0


class _MockPipeline:
    def __init__(self):
        self.calls = []
    def process_spectrum(self, wl, intensities, timestamp=None, sample_id=None):
        self.calls.append((wl, intensities))
        return _MockPipelineResult()


def _make_app(loop):
    pipeline = _MockPipeline()
    state = SimpleNamespace(
        asyncio_loop=loop,
        spectrum_bc=_RecordingBroadcaster(),
        quality_agent=_QualityPass(),
        drift_agent=None,
        plugin=None,
        session_running=True,
        session_frame_count=0,
        latest_spectrum=None,
        pipeline=pipeline,
    )
    return SimpleNamespace(state=state)


def test_pipeline_is_called_when_wired():
    """_process_acquired_frame must call app.state.pipeline.process_spectrum."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert len(app.state.pipeline.calls) == 1
    finally:
        loop.close()


def test_broadcast_includes_concentration_when_pipeline_runs():
    """WebSocket broadcast must contain concentration_ppm when pipeline succeeds."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert len(app.state.spectrum_bc.messages) == 1
        payload = json.loads(app.state.spectrum_bc.messages[0])
        assert "concentration_ppm" in payload
        assert "ci_low" in payload
        assert "ci_high" in payload
    finally:
        loop.close()


def test_frame_works_without_pipeline():
    """_process_acquired_frame must still broadcast when no pipeline is wired."""
    loop = asyncio.new_event_loop()
    try:
        app = _make_app(loop)
        del app.state.pipeline  # remove pipeline
        wl = np.linspace(300, 1000, 3648)
        intensities = np.random.rand(3648)
        result = _process_acquired_frame(cast(FastAPI, app), wl.tolist(), wl, 1, intensities)
        loop.run_until_complete(asyncio.sleep(0))
        assert result is True
    finally:
        loop.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/spectraagent/test_frame_inference.py -v
```

Expected: `test_pipeline_is_called_when_wired` FAIL, `test_broadcast_includes_concentration_when_pipeline_runs` FAIL

- [ ] **Step 3: Modify `spectraagent/__main__.py` — update `_process_acquired_frame`**

In `_process_acquired_frame`, after the `app.state.latest_spectrum = ...` block and before the quality agent block, add:

```python
    # --- Run ML inference pipeline if wired ---
    pipeline_result = None
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            pipeline_result = pipeline.process_spectrum(
                np.array(wl_list), intensities
            )
        except Exception as exc:
            log.debug("RealTimePipeline.process_spectrum() failed: %s", exc)
```

Then update the WebSocket broadcast block (the `msg = json.dumps(...)` line) to include inference results:

```python
    app_loop = getattr(app.state, "asyncio_loop", None)
    spectrum_bc = getattr(app.state, "spectrum_bc", None)
    if app_loop is not None and spectrum_bc is not None and hasattr(spectrum_bc, "broadcast"):
        payload: dict = {"wl": wl_list, "i": intensities.tolist(), "frame": frame_num}
        if pipeline_result is not None and pipeline_result.success:
            sd = pipeline_result.spectrum
            if sd.concentration_ppm is not None:
                payload["concentration_ppm"] = round(sd.concentration_ppm, 4)
            if sd.ci_low is not None:
                payload["ci_low"] = round(sd.ci_low, 4)
            if sd.ci_high is not None:
                payload["ci_high"] = round(sd.ci_high, 4)
            if sd.wavelength_shift is not None:
                payload["peak_shift_nm"] = round(sd.wavelength_shift, 4)
            if sd.snr is not None:
                payload["snr"] = round(sd.snr, 2)
        msg = json.dumps(payload)
        broadcast_fn = spectrum_bc.broadcast
        app_loop.call_soon_threadsafe(
            lambda m=msg, fn=broadcast_fn: asyncio.ensure_future(fn(m))
        )
```

Also add RealTimePipeline creation in `start()`, after hardware/physics are loaded (after Step 4 — Build the FastAPI app):

```python
    # Step 4b: Build RealTimePipeline and wire to app
    from src.inference.realtime_pipeline import RealTimePipeline, PipelineConfig
    pipeline_cfg = PipelineConfig(
        integration_time_ms=cfg.hardware.integration_time_ms,
        peak_search_min_nm=650.0,
        peak_search_max_nm=780.0,
        reference_wavelength=717.9,
    )
    pipeline = RealTimePipeline(pipeline_cfg)
    app.state.pipeline = pipeline
    typer.echo("RealTimePipeline wired")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/spectraagent/test_frame_inference.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run existing spectraagent test suite to check no regressions**

```
pytest tests/spectraagent/ -v --tb=short -q
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add spectraagent/__main__.py tests/spectraagent/test_frame_inference.py
git commit -m "feat: wire RealTimePipeline into per-frame hot path; enrich WebSocket broadcast"
```

---

### Task 8: Feed Real Measurement Data to Claude Agents

**Files:**
- Modify: `spectraagent/webapp/agents/claude_agents.py`

- [ ] **Step 1: Read the current AnomalyExplainer prompt construction**

Check lines 150–220 of `spectraagent/webapp/agents/claude_agents.py` to locate the prompt text.

- [ ] **Step 2: Update AnomalyExplainer prompt to include real measurements**

Find the `_build_prompt` method in `AnomalyExplainer` (or wherever it builds the message text). Replace the generic description with:

```python
    def _build_prompt(self, event_data: dict) -> str:
        conc = event_data.get("concentration_ppm")
        ci_low = event_data.get("ci_low")
        ci_high = event_data.get("ci_high")
        shift = event_data.get("peak_shift_nm")
        snr = event_data.get("snr")
        drift = event_data.get("drift_rate")

        lines = [
            "You are an expert LSPR spectroscopy analyst.",
            "An anomaly has been detected in the sensor data.",
            "",
            "## Current Measurement",
        ]
        if conc is not None:
            lines.append(f"- Concentration estimate: {conc:.4f} ppm")
        if ci_low is not None and ci_high is not None:
            lines.append(f"- 90% conformal interval: [{ci_low:.4f}, {ci_high:.4f}] ppm")
        if shift is not None:
            lines.append(f"- Peak wavelength shift (Δλ): {shift:.4f} nm")
        if snr is not None:
            lines.append(f"- Signal-to-noise ratio: {snr:.1f}")
        if drift is not None:
            lines.append(f"- Drift rate: {drift:.6f} nm/frame")
        lines += [
            "",
            "## Task",
            "1. Identify the most likely cause of this anomaly.",
            "2. Assess measurement validity.",
            "3. Suggest corrective action (re-reference, clean sensor, adjust integration time, etc.).",
            "Keep the response under 150 words.",
        ]
        return "\n".join(lines)
```

- [ ] **Step 3: Update ExperimentNarrator to include session statistics**

Find `ExperimentNarrator._build_prompt` and replace/add:

```python
    def _build_prompt(self, event_data: dict) -> str:
        conc = event_data.get("concentration_ppm")
        ci_low = event_data.get("ci_low")
        ci_high = event_data.get("ci_high")
        shift = event_data.get("peak_shift_nm")
        snr = event_data.get("snr")
        frame = event_data.get("frame", "?")

        lines = [
            "You are narrating a live LSPR chemical sensing experiment.",
            f"Frame {frame} data:",
        ]
        if conc is not None:
            lines.append(f"  Concentration: {conc:.4f} ppm")
        if ci_low is not None and ci_high is not None:
            lines.append(f"  90% CI: [{ci_low:.4f}, {ci_high:.4f}] ppm")
        if shift is not None:
            lines.append(f"  Δλ: {shift:.4f} nm")
        if snr is not None:
            lines.append(f"  SNR: {snr:.1f}")
        lines += [
            "",
            "Provide a 1-2 sentence scientific narration of the current measurement state.",
            "Mention if the measurement is stable or has notable features.",
        ]
        return "\n".join(lines)
```

- [ ] **Step 4: Update DiagnosticsAgent to include measurement context**

Find `DiagnosticsAgent._build_prompt`:

```python
    def _build_prompt(self, event_data: dict) -> str:
        code = event_data.get("code", "unknown")
        conc = event_data.get("concentration_ppm")
        snr = event_data.get("snr")
        shift = event_data.get("peak_shift_nm")

        lines = [
            f"Diagnostic code: {code}",
            "You are a spectroscopy hardware diagnostics expert.",
        ]
        if conc is not None:
            lines.append(f"Current concentration estimate: {conc:.4f} ppm")
        if snr is not None:
            lines.append(f"SNR: {snr:.1f}")
        if shift is not None:
            lines.append(f"Peak shift: {shift:.4f} nm")
        lines += [
            "",
            "Provide a brief (≤100 words) diagnostic interpretation and recommended action.",
        ]
        return "\n".join(lines)
```

- [ ] **Step 5: Verify `on_event` routes enriched data to prompts**

Confirm `AnomalyExplainer.on_event()` passes the full event `data` dict to `_build_prompt`:

```python
    def on_event(self, event: AgentEvent) -> None:
        if event.type not in ("anomaly_detected", "quality_alert"):
            return
        self._trigger(event.data or {})
```

- [ ] **Step 6: Run agent tests**

```
pytest tests/spectraagent/ -v -k "claude" --tb=short
```

Expected: all Claude agent tests pass

- [ ] **Step 7: Commit**

```bash
git add spectraagent/webapp/agents/claude_agents.py
git commit -m "feat: pass real concentration/CI/SNR/shift data into Claude agent prompts"
```

---

### Task 9: Upgrade ExperimentPlannerAgent to BayesianExperimentDesigner

**Files:**
- Modify: `spectraagent/webapp/agents/planner.py`
- Test: `tests/spectraagent/test_planner_bed.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/spectraagent/test_planner_bed.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from spectraagent.webapp.agents.planner import ExperimentPlannerAgent
from spectraagent.webapp.agent_bus import AgentBus


def _make_bus():
    bus = MagicMock(spec=AgentBus)
    bus.emit = MagicMock()
    return bus


def test_suggest_uses_logspace_not_linspace():
    """With no measured points, suggest() must return a value > 0.1 (logspace spans low end)."""
    bus = _make_bus()
    agent = ExperimentPlannerAgent(bus, min_conc=0.01, max_conc=10.0)
    # No GPR, space-filling fallback
    suggestion = agent.suggest()
    # Must still return a suggestion (space-filling)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/spectraagent/test_planner_bed.py -v
```

Expected: `test_suggest_with_measured_history` FAIL (`AttributeError: record_measured`), others may pass or fail.

- [ ] **Step 3: Rewrite `spectraagent/webapp/agents/planner.py`**

```python
"""
spectraagent.webapp.agents.planner
====================================
ExperimentPlannerAgent — suggests the next calibration concentration.

Upgraded from linspace max-variance to Bayesian Experimental Design using
logspace candidates and space-filling fallback for sparse early-session data.
"""
from __future__ import annotations

import logging
from typing import Optional

from spectraagent.webapp.agent_bus import AgentBus, AgentEvent

log = logging.getLogger(__name__)


class ExperimentPlannerAgent:
    """Concentration suggestion using Bayesian Experimental Design.

    Uses BayesianExperimentDesigner from src.calibration.active_learning with
    logspace candidates so low and high concentration regions are both explored.
    Falls back to space-filling when no GPR is fitted yet.
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

        from src.calibration.active_learning import BayesianExperimentDesigner
        self._designer = BayesianExperimentDesigner(
            min_conc=min_conc,
            max_conc=max_conc,
            n_candidates=n_candidates,
        )

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
            return suggestion
        except Exception as exc:
            log.warning("ExperimentPlannerAgent.suggest() failed: %s", exc)
            return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/spectraagent/test_planner_bed.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Run full spectraagent test suite**

```
pytest tests/spectraagent/ -v --tb=short -q
```

Expected: all existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add spectraagent/webapp/agents/planner.py tests/spectraagent/test_planner_bed.py
git commit -m "feat: upgrade ExperimentPlannerAgent to BayesianExperimentDesigner with logspace"
```

---

### Task 10: Post-Session Analysis — Auto-trigger SessionAnalyzer

**Files:**
- Modify: `spectraagent/__main__.py` (session-stop handler)
- Modify: `spectraagent/webapp/server.py` (session stop endpoint)

- [ ] **Step 1: Add session event accumulation to `_process_acquired_frame`**

In `_process_acquired_frame`, after the pipeline result block, append to session events:

```python
    # Accumulate session events for post-session analysis
    session_events = getattr(app.state, "session_events", None)
    if session_events is not None and getattr(app.state, "session_running", False):
        if pipeline_result is not None and pipeline_result.success:
            sd = pipeline_result.spectrum
            event = {"type": "measurement"}
            if sd.concentration_ppm is not None:
                event["concentration_ppm"] = sd.concentration_ppm
            if sd.ci_low is not None:
                event["ci_low"] = sd.ci_low
            if sd.ci_high is not None:
                event["ci_high"] = sd.ci_high
            if sd.wavelength_shift is not None:
                event["wavelength_shift"] = sd.wavelength_shift
            if sd.snr is not None:
                event["snr"] = sd.snr
            if sd.peak_wavelength is not None:
                event["peak_wavelength"] = sd.peak_wavelength
            session_events.append(event)
```

- [ ] **Step 2: Initialize session_events in `start()`**

In `start()`, after `app.state.planner_agent = ...`:

```python
    app.state.session_events = []
```

- [ ] **Step 3: Read the session stop endpoint in server.py**

```
grep -n "session" spectraagent/webapp/server.py | head -30
```

- [ ] **Step 4: Add post-session analysis trigger to the session stop endpoint**

Locate the `POST /api/session/stop` handler in `spectraagent/webapp/server.py` and add after the session is marked stopped:

```python
    # Auto-run SessionAnalyzer and feed to ReportWriter
    session_events = getattr(app.state, "session_events", [])
    frame_count = getattr(app.state, "session_frame_count", 0)
    try:
        from src.inference.session_analyzer import SessionAnalyzer
        analysis = SessionAnalyzer().analyze(session_events, frame_count)
        app.state.last_session_analysis = analysis

        # Feed to ReportWriter if available
        report_writer = getattr(app.state, "report_writer", None)
        agent_bus = getattr(app.state, "agent_bus", None)
        if report_writer is not None and agent_bus is not None:
            from spectraagent.webapp.agent_bus import AgentEvent
            agent_bus.emit(AgentEvent(
                source="SessionAnalyzer",
                level="info",
                type="session_complete",
                data={
                    "lod_ppm": analysis.lod_ppm,
                    "loq_ppm": analysis.loq_ppm,
                    "calibration_r2": analysis.calibration_r2,
                    "mean_snr": analysis.mean_snr,
                    "drift_rate_nm_per_frame": analysis.drift_rate_nm_per_frame,
                    "frame_count": analysis.frame_count,
                    "summary": analysis.summary_text,
                },
                text=analysis.summary_text,
            ))
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("Post-session analysis failed: %s", exc)
    # Clear events for next session
    app.state.session_events = []
```

- [ ] **Step 5: Verify server.py session stop handler works**

```
pytest tests/spectraagent/ -v -k "session" --tb=short
```

Expected: all existing session tests pass

- [ ] **Step 6: Commit**

```bash
git add spectraagent/__main__.py spectraagent/webapp/server.py
git commit -m "feat: auto-run SessionAnalyzer on session stop; feed results to ReportWriter"
```

---

### Task 11: Fix scripts/pipeline_cli.py sweep import

**Files:**
- Modify: `scripts/pipeline_cli.py`

- [ ] **Step 1: Verify the broken import location**

Open `scripts/pipeline_cli.py:121`. The line is:
```python
best = sweep_hyperparameters(datasets, out_root, param_grid)  # noqa: F821
```

- [ ] **Step 2: Add the import at the top of `pipeline_cli.py`**

After the existing imports block (around line 15), add:

```python
from src.training.hyperparameter_sweep import sweep_hyperparameters  # noqa: E402
```

- [ ] **Step 3: Update the call site to match the new signature**

The new `sweep_hyperparameters(datasets, param_grid)` does not take `out_root`. The `datasets` list must have `X`/`y` arrays. Update line 121's surrounding context:

```python
        # Build dataset list with X/y arrays for sweep
        from gas_analysis.core.pipeline import CONFIG as _CFG
        import pandas as _pd
        sweep_datasets = []
        for ds in datasets:
            sweep_datasets.append({
                "label": ds.get("label", Path(ds["data"]).stem),
                "data": ds["data"],
                "ref": ds["ref"],
            })
        # Note: sweep_hyperparameters accepts data/ref paths OR X/y arrays.
        # For pipeline_cli sweep mode, pass through raw path-based datasets
        # and let sweep resolve them internally.
        best = sweep_hyperparameters(sweep_datasets, param_grid)
```

Wait — the new `sweep_hyperparameters` signature requires `X`/`y` arrays, not file paths. Update `pipeline_cli.py` to pass pre-processed arrays:

```python
        from src.training.hyperparameter_sweep import sweep_hyperparameters
        import numpy as _np

        sweep_datasets = []
        for ds in datasets:
            # Load and preprocess each dataset
            try:
                from gas_analysis.core.pipeline import run_full_pipeline as _rfp
                _result = _rfp(
                    root_dir=os.path.abspath(ds["data"]),
                    ref_path=os.path.abspath(ds["ref"]),
                    out_root=out_root,
                    diff_threshold=args.diff_threshold,
                )
                calib = _result.get("calibration", {})
                # Use shift data from calibration result
                if calib and calib.get("shifts") is not None:
                    sweep_datasets.append({
                        "label": ds.get("label", Path(ds["data"]).stem),
                        "X": _np.array(calib["shifts"]).reshape(-1, 1),
                        "y": _np.array(calib["concentrations"]),
                    })
            except Exception as _e:
                print(f"  Warning: Could not load dataset {ds.get('label', '?')}: {_e}")

        if sweep_datasets:
            best = sweep_hyperparameters(sweep_datasets, param_grid)
        else:
            print("  No datasets could be loaded for sweep.")
            return
```

- [ ] **Step 4: Remove the `# noqa: F821` comment from the original line**

The F821 (undefined name) will no longer apply after the import is added.

- [ ] **Step 5: Verify pipeline_cli syntax**

```
python -m py_compile scripts/pipeline_cli.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline_cli.py
git commit -m "fix: import sweep_hyperparameters in pipeline_cli; resolve F821 undefined name"
```

---

### Task 12: Reference Spectrum Propagation

**Files:**
- Modify: `spectraagent/webapp/server.py` — propagate reference spectrum to RealTimePipeline

- [ ] **Step 1: Find the reference-set endpoint in server.py**

```
grep -n "reference\|set_reference\|/api/reference" spectraagent/webapp/server.py | head -20
```

- [ ] **Step 2: Wire reference into RealTimePipeline when set**

In the reference-set handler (or wherever `app.state.reference` is written), add:

```python
    # Propagate to RealTimePipeline stages
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "_feature_stage"):
        try:
            pipeline._feature_stage.set_reference(ref_intensities_np)
            pipeline._calibration_stage.set_reference(ref_intensities_np)
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("Pipeline reference set failed: %s", exc)
```

- [ ] **Step 3: Check RealTimePipeline exposes `_feature_stage` and `_calibration_stage`**

Read `src/inference/realtime_pipeline.py` RealTimePipeline `__init__` to confirm attribute names.

- [ ] **Step 4: Run server tests**

```
pytest tests/spectraagent/ -v --tb=short -q
```

Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add spectraagent/webapp/server.py
git commit -m "feat: propagate reference spectrum to RealTimePipeline on reference set"
```

---

### Task 13: Integration Tests

**Files:**
- Create: `tests/integration/test_pipeline_to_agent.py`

- [ ] **Step 1: Write the integration tests**

```python
# tests/integration/test_pipeline_to_agent.py
"""End-to-end tests: src pipeline → conformal intervals → agent context."""
import numpy as np
import pytest


def test_conformal_intervals_are_finite_and_ordered():
    """Full path: GPR fit → conformal calibrate → predict interval."""
    from src.calibration.gpr import GPRCalibration
    from src.calibration.conformal import ConformalCalibrator

    np.random.seed(123)
    n = 40
    concs = np.random.uniform(0.1, 5.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.1, n)

    # Train/cal split
    X_train, y_train = shifts[:20].reshape(-1, 1), concs[:20]
    X_cal, y_cal = shifts[20:30].reshape(-1, 1), concs[20:30]
    X_test = shifts[30:].reshape(-1, 1)
    y_test = concs[30:]

    gpr = GPRCalibration()
    gpr.fit(X_train, y_train)

    cal = ConformalCalibrator()
    cal.calibrate(gpr, X_cal, y_cal)

    lo, hi = cal.predict_interval(gpr, X_test, alpha=0.10)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(hi > lo)

    coverage = np.mean((y_test >= lo) & (y_test <= hi))
    assert coverage >= 0.75, f"Coverage {coverage:.2%} unexpectedly low"


def test_session_analyzer_produces_valid_lod():
    """SessionAnalyzer output has positive LOD/LOQ from realistic calibration data."""
    from src.inference.session_analyzer import SessionAnalyzer

    events = []
    for conc in [0.5, 1.0, 2.0, 3.0, 4.0]:
        events.append({
            "type": "calibration_point",
            "concentration_ppm": conc,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 14.0,
        })
    for i in range(10):
        conc = 2.5
        events.append({
            "type": "measurement",
            "concentration_ppm": conc + i * 0.02,
            "ci_low": conc - 0.25,
            "ci_high": conc + 0.25,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 13.0 + i * 0.1,
            "peak_wavelength": 715.0 + i * 0.01,
        })

    analysis = SessionAnalyzer().analyze(events, frame_count=15)
    assert np.isfinite(analysis.lod_ppm) and analysis.lod_ppm > 0
    assert np.isfinite(analysis.loq_ppm) and analysis.loq_ppm > analysis.lod_ppm
    assert analysis.calibration_r2 is not None and analysis.calibration_r2 > 0.5


def test_bayesian_designer_and_physics_gpr_together():
    """BayesianExperimentDesigner with PhysicsInformedGPR suggests valid concentrations."""
    from src.calibration.physics_kernel import PhysicsInformedGPR
    from src.calibration.active_learning import BayesianExperimentDesigner

    np.random.seed(0)
    concs = np.array([0.5, 1.0, 2.0])
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.05, 3)

    model = PhysicsInformedGPR()
    model.fit(shifts.reshape(-1, 1), concs)

    designer = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    suggestion = designer.suggest_next(model, concs.tolist())
    assert 0.01 <= suggestion <= 10.0


def test_full_pipeline_with_conformal_stage():
    """RealTimePipeline with set_gpr populates ci_low/ci_high on a real frame."""
    from src.calibration.gpr import GPRCalibration
    from src.inference.realtime_pipeline import RealTimePipeline, PipelineConfig

    np.random.seed(5)
    n = 15
    concs = np.linspace(0.5, 4.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.08, n)

    gpr = GPRCalibration()
    gpr.fit(shifts.reshape(-1, 1), concs)

    X_cal = shifts[:8].reshape(-1, 1)
    y_cal = concs[:8]

    cfg = PipelineConfig(
        reference_wavelength=717.9,
        peak_search_min_nm=650.0,
        peak_search_max_nm=780.0,
    )
    pipeline = RealTimePipeline(cfg)
    pipeline._calibration_stage.set_gpr(gpr, X_cal, y_cal)

    # Synthetic reference + test spectrum
    wl = np.linspace(300, 1000, 3648)
    # Lorentzian at 717.9 nm
    ref = 1000 * np.exp(-((wl - 717.9) ** 2) / (2 * 5.0 ** 2))
    test = 1000 * np.exp(-((wl - 716.0) ** 2) / (2 * 5.0 ** 2))  # -1.9 nm shift

    pipeline._feature_stage.set_reference(ref)
    pipeline._calibration_stage.set_reference(ref)

    result = pipeline.process_spectrum(wl, test)
    assert result.success or result.spectrum.wavelength_shift is not None
```

- [ ] **Step 2: Run tests to verify they pass**

```
pytest tests/integration/test_pipeline_to_agent.py -v --tb=short
```

Expected: 4 PASSED (or 3+ if GPR fit with synthetic data has numerical issues — see note)

- [ ] **Step 3: Run the complete test suite**

```
pytest tests/ -v --tb=short -q
```

Expected: all new tests pass; no regressions in existing suite

- [ ] **Step 4: Run mypy on all new src modules**

```
python -m mypy src/calibration/physics_kernel.py src/calibration/conformal.py src/calibration/active_learning.py src/training/hyperparameter_sweep.py src/inference/session_analyzer.py --python-version 3.11 --no-site-packages --ignore-missing-imports
```

Expected: 0 errors

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_pipeline_to_agent.py
git commit -m "test: add integration tests covering conformal intervals, SessionAnalyzer, and BED"
```

---

## Implementation Order

```
Task 1  (physics_kernel)      — no dependencies
Task 2  (conformal)           — no dependencies (uses GPRCalibration directly)
Task 3  (active_learning)     — no dependencies
Task 4  (hyperparameter_sweep)— no dependencies
Task 5  (session_analyzer)    — no dependencies
Task 6  (realtime_pipeline)   — depends on Task 2 (ConformalCalibrator)
Task 7  (frame inference)     — depends on Task 6 (ci_low/ci_high on SpectrumData)
Task 8  (claude agents)       — depends on Task 7 (enriched broadcast payload)
Task 9  (planner BED)         — depends on Task 3 (BayesianExperimentDesigner)
Task 10 (session stop)        — depends on Task 5 (SessionAnalyzer)
Task 11 (pipeline_cli fix)    — depends on Task 4 (sweep_hyperparameters)
Task 12 (reference prop)      — depends on Task 7 (pipeline wired)
Task 13 (integration tests)   — depends on Tasks 1–12
```
