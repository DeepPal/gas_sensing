import numpy as np
import pytest
from src.training.hyperparameter_sweep import _expand_grid, sweep_hyperparameters


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

    datasets = [{"label": "ds1", "X": shifts.reshape(-1, 1), "y": concs}]
    param_grid = {"n_restarts_optimizer": [1, 3]}
    results = sweep_hyperparameters(datasets, param_grid)
    assert "ds1" in results
    assert "best_params" in results["ds1"]
    assert "r2" in results["ds1"]
    assert "rmse" in results["ds1"]


def test_sweep_selects_better_params():
    """With clean signal data, sweep must select the param set with higher R²."""
    np.random.seed(42)
    n = 20
    concs = np.linspace(0.1, 5.0, n)
    shifts = -10.0 * concs / (1.0 + concs) + np.random.normal(0, 0.03, n)

    datasets = [{"label": "clean", "X": shifts.reshape(-1, 1), "y": concs}]
    param_grid = {"n_restarts_optimizer": [1, 5]}
    results = sweep_hyperparameters(datasets, param_grid)
    assert results["clean"]["r2"] > 0.8


def test_sweep_handles_too_few_samples():
    """Datasets with < 3 samples must be skipped without raising."""
    datasets = [{"label": "tiny", "X": np.array([[1.0], [2.0]]), "y": np.array([1.0, 2.0])}]
    param_grid = {"n_restarts_optimizer": [1]}
    results = sweep_hyperparameters(datasets, param_grid)
    assert "tiny" in results
    assert np.isnan(results["tiny"]["r2"])
