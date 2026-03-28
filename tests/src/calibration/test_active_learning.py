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
    """Candidates must span logspace: suggestion should be >= 0.05 given a low-end measured point."""
    gpr = _fitted_gpr()
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=100.0, n_candidates=200)
    suggestions = [bed.suggest_next(gpr, [0.5]) for _ in range(1)]
    assert suggestions[0] >= 0.05


def test_no_duplicate_suggestions():
    """Measured concentrations should be excluded from candidates."""
    gpr = _fitted_gpr()
    measured = [0.5, 1.0, 2.0, 4.0]
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0, n_candidates=100)
    suggestion = bed.suggest_next(gpr, measured)
    for m in measured:
        assert abs(suggestion - m) > 1e-4
