"""
src.training.hyperparameter_sweep
===================================
Grid search over GPR hyperparameters using leave-one-out cross-validation.

With only 5–20 calibration points typical in LSPR experiments, LOOCV
maximally uses available data — n-1 points train, 1 tests — giving the most
stable estimate of generalisation error on small datasets.

This module resolves the ``sweep_hyperparameters`` reference that was called
(but undefined) in ``scripts/pipeline_cli.py:121``.

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
    list of dicts, one per parameter combination
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _loocv_r2_rmse(
    X: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
) -> tuple[float, float]:
    """Leave-one-out cross-validation R² and RMSE for GPRCalibration.

    Parameters
    ----------
    X : (n, d) feature matrix
    y : (n,) targets
    params : keyword arguments forwarded to GPRCalibration.__init__

    Returns
    -------
    (r2, rmse) computed on the n held-out predictions
    """
    from src.calibration.gpr import GPRCalibration

    n = len(y)
    if n < 3:
        return float("nan"), float("nan")

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
    datasets : list of dicts, each with:
        - ``label`` : str — dataset identifier
        - ``X``     : np.ndarray (n, d) — feature matrix (e.g. wavelength shifts)
        - ``y``     : np.ndarray (n,)   — targets (e.g. concentrations in ppm)
    param_grid : GPRCalibration __init__ param names mapped to value lists.
        Supported keys: ``n_restarts_optimizer``, ``random_state``.

    Returns
    -------
    dict mapping dataset label to::

        {
            "best_params": dict,
            "r2": float,
            "rmse": float,
            "all_results": list[dict]   # one entry per param combination
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
            log.debug(
                "Dataset %s | params=%s | R2=%.4f RMSE=%.4f",
                label, params, r2, rmse,
            )
            if not np.isnan(r2) and r2 > best_r2:
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
            label, best_params, best_r2, best_rmse,
        )

    return output
