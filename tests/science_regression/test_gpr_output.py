"""
Science regression: GPR calibration quality must not degrade.

The Matérn ν=5/2 kernel is the canonical implementation (ADR-002).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.calibration.gpr import GPRCalibration


def test_gpr_posterior_std_bounded(cal_fixture, baselines):
    """GPR posterior std at -1.0 nm shift must be within 5% of baseline."""
    dl = cal_fixture["delta_lambda_measured"].reshape(-1, 1)
    conc = cal_fixture["concentrations"]

    gpr = GPRCalibration()
    gpr.fit(dl, conc)
    _, std = gpr.predict(np.array([[-1.0]]))

    std_val = float(std[0])
    baseline = baselines["gpr_std_at_neg1nm"]
    rel_err = abs(std_val - baseline) / baseline

    assert rel_err < 0.05, (
        f"GPR std drifted {rel_err:.1%} from baseline {baseline:.6f}. "
        f"Got {std_val:.6f}. If kernel changed, update ADR-002 and reset baselines."
    )


def test_gpr_mean_monotone(cal_fixture):
    """GPR predictions must be monotone over the calibration domain."""
    dl = cal_fixture["delta_lambda_measured"].reshape(-1, 1)
    conc = cal_fixture["concentrations"]

    gpr = GPRCalibration()
    gpr.fit(dl, conc)

    test_shifts = np.linspace(dl.min(), dl.max(), 20).reshape(-1, 1)
    means, _ = gpr.predict(test_shifts)

    diffs = np.diff(means)
    assert np.all(diffs <= 0.05), (
        "GPR predictions are not monotone over calibration domain. "
        "Check kernel hyperparameters."
    )


def test_gpr_uses_matern_kernel():
    """GPR must use Matérn ν=5/2 kernel — not RBF (ADR-002)."""
    gpr = GPRCalibration()
    kernel_str = str(gpr.kernel)
    assert "Matern" in kernel_str, (
        f"GPR kernel is not Matérn. Got: {kernel_str}. "
        "See ADR-002: RBF was intentionally replaced with Matérn ν=5/2."
    )


def test_gpr_n_restarts():
    """GPR must use 10 optimizer restarts for robust hyperparameter search (ADR-002)."""
    gpr = GPRCalibration()
    assert gpr.n_restarts_optimizer == 10, (
        f"Expected 10 restarts, got {gpr.n_restarts_optimizer}. See ADR-002."
    )
