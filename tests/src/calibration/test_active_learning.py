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


# ── has_converged + expected_information_gain ─────────────────────────────

def test_has_converged_false_for_too_few_suggestions():
    """has_converged must return False when fewer than n_window suggestions exist."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    assert bed.has_converged([1.0, 2.0], n_window=3) is False


def test_has_converged_true_for_stable_suggestions():
    """has_converged must return True when last n_window suggestions span < tol_log decades."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    # Three suggestions all within 10f each other in log-space
    stable = [1.000, 1.005, 1.010]
    assert bed.has_converged(stable, tol_log=0.05, n_window=3) is True


def test_has_converged_false_for_spread_suggestions():
    """has_converged must return False when suggestions span >> tol_log decades."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    spread = [0.1, 1.0, 10.0]  # 2 decades spread
    assert bed.has_converged(spread, tol_log=0.05, n_window=3) is False


def test_expected_information_gain_returns_float_or_nan():
    """expected_information_gain must return a finite float when GPR is available."""
    gpr = _fitted_gpr()
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    measured = [0.5, 1.0, 2.0]
    gain = bed.expected_information_gain(gpr, measured)
    assert isinstance(gain, float)


def test_expected_information_gain_nan_without_gpr():
    """expected_information_gain must return nan when gpr=None."""
    bed = BayesianExperimentDesigner(min_conc=0.01, max_conc=10.0)
    import math
    assert math.isnan(bed.expected_information_gain(None, []))
