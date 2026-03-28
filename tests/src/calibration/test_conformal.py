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
    """Normalised conformal prediction interval must achieve near-nominal coverage.

    The theoretical guarantee is marginal coverage ≥ 90% (1 - alpha). We test
    empirically with n_cal=80, n_test=300, seed=42 and accept ≥ 80% empirical
    coverage — the finite-sample guarantee holds marginally over the random draw
    of calibration/test splits, so a 10% buffer avoids test flakiness while
    still verifying the conformal property holds on realistic LSPR data.
    """
    np.random.seed(42)
    from src.calibration.gpr import GPRCalibration

    # Generate synthetic LSPR data: Langmuir shift + measurement noise
    n_train, n_cal, n_test = 40, 80, 300
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
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    assert coverage >= 0.80, f"Coverage {coverage:.2%} below 80% — conformal property not holding"


def test_scores_are_normalised_by_sigma():
    """Nonconformity scores must be |y - ŷ| / σ, not raw residuals."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    X_cal = np.linspace(-4, -0.5, 5).reshape(-1, 1)
    y_cal = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    cal.calibrate(gpr, X_cal, y_cal)

    # Recompute scores manually to verify normalisation
    mean, std = gpr.predict(X_cal, return_std=True)
    expected_scores = np.abs(y_cal - mean.ravel()) / np.maximum(std.ravel(), 1e-9)
    np.testing.assert_allclose(cal._scores, expected_scores.tolist(), rtol=1e-6)


def test_predict_interval_requires_calibrate():
    """predict_interval before calibrate must raise RuntimeError."""
    from src.calibration.gpr import GPRCalibration
    gpr = GPRCalibration()
    cal = ConformalCalibrator()
    with pytest.raises(RuntimeError, match="calibrate"):
        cal.predict_interval(gpr, np.array([[-1.0]]), alpha=0.10)
