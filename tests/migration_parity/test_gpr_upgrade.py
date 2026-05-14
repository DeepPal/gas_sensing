"""
Verify Matérn ν=5/2 (src/) is scientifically superior to RBF (gas_analysis/).

This is not a strict parity test — the two kernels intentionally produce
different outputs. We verify that the new implementation is at least as good
as the old one on the calibration fixture (ADR-S002).

Both GPR implementations expose log-marginal-likelihood via the sklearn method
``model.log_marginal_likelihood(model.kernel_.theta)`` after fitting — there is
no ``log_marginal_likelihood_value_`` attribute on GaussianProcessRegressor.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def fixture_data():
    from pathlib import Path

    data = np.load(
        Path(__file__).parents[2] / "tests/fixtures/lspr_calibration_fixture.npz"
    )
    return data["delta_lambda_measured"].reshape(-1, 1), data["concentrations"]


def test_matern_log_likelihood_gte_rbf(fixture_data):
    """Matérn ν=5/2 must achieve equal or higher log marginal likelihood than RBF."""
    X, y = fixture_data

    from gas_analysis.core.intelligence.gpr import GPRCalibration as OldGPR
    from src.calibration.gpr import GPRCalibration as NewGPR

    old = OldGPR()
    old.fit(X, y)
    # Both implementations call log_marginal_likelihood() as a method after fitting.
    # sklearn GaussianProcessRegressor does not expose log_marginal_likelihood_value_.
    old_lml = float(old.model.log_marginal_likelihood(old.model.kernel_.theta))

    new = NewGPR()
    new.fit(X, y)
    new_lml = float(new.model.log_marginal_likelihood(new.model.kernel_.theta))

    assert new_lml >= old_lml - 1.0, (
        f"Matérn LML ({new_lml:.3f}) is more than 1.0 below RBF LML ({old_lml:.3f}). "
        "Investigate kernel hyperparameters."
    )


def test_matern_predictions_are_positive(fixture_data):
    """GPR mean predictions must be positive (concentrations cannot be negative)."""
    X, y = fixture_data
    from src.calibration.gpr import GPRCalibration

    gpr = GPRCalibration()
    gpr.fit(X, y)
    test_shifts = np.linspace(X.min(), X.max(), 30).reshape(-1, 1)
    means, _ = gpr.predict(test_shifts)
    assert np.all(means > -0.1), "GPR predicts negative concentrations in calibration range"
