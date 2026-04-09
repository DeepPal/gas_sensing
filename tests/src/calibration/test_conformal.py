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


def test_small_calibration_set_emits_warning():
    """calibrate() with n < MIN_CAL_POINTS must emit a UserWarning."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    X_cal = np.linspace(-3, -0.5, 5).reshape(-1, 1)   # only 5 points
    y_cal = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.warns(UserWarning, match="calibration points"):
        cal.calibrate(gpr, X_cal, y_cal)


def test_n_cal_property_exposed():
    """ConformalCalibrator must expose n_cal as a readable property."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    X_cal = np.linspace(-4, -0.5, 15).reshape(-1, 1)
    y_cal = np.linspace(2.0, 8.0, 15)
    cal.calibrate(gpr, X_cal, y_cal)
    assert cal.n_cal == 15


def test_check_ood_returns_false_for_in_distribution():
    """check_ood() must return False when test scores are within calibration range."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    np.random.seed(7)
    X_cal = np.linspace(-4, -0.5, 30).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 5.0 + np.random.normal(0, 0.1, 30)
    cal.calibrate(gpr, X_cal, y_cal)
    # Test on similar in-distribution data
    X_test = np.linspace(-3.5, -1.0, 10).reshape(-1, 1)
    assert cal.check_ood(gpr, X_test) is False


def test_check_ood_returns_true_for_out_of_distribution():
    """check_ood() must return True when test scores are far outside calibration range."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    np.random.seed(7)
    X_cal = np.linspace(-4, -0.5, 30).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 5.0 + np.random.normal(0, 0.1, 30)
    cal.calibrate(gpr, X_cal, y_cal)
    # Manufacture OOD: query at a point where GPR has low std but residual will be large
    # (far outside training distribution — GPR returns near-prior, causing large normalised error)
    X_ood = np.array([[-50.0], [-60.0], [-70.0]])  # far outside calibration range
    assert cal.check_ood(gpr, X_ood) is True


def test_check_ood_raises_before_calibrate():
    """check_ood() before calibrate() must raise RuntimeError."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    with pytest.raises(RuntimeError, match="calibrate"):
        cal.check_ood(gpr, np.array([[-1.0]]))


# ── cross_validate_coverage ───────────────────────────────────────────────────

def test_cross_validate_coverage_achieves_target():
    """LOO coverage must be >= (1 - alpha) for a well-calibrated model."""
    gpr = _make_simple_gpr()
    np.random.seed(99)
    n = 20
    X_cal = np.linspace(-4.5, -0.3, n).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 5.0 + np.random.normal(0, 0.1, n)
    cal = ConformalCalibrator()
    alpha = 0.10
    coverage = cal.cross_validate_coverage(gpr, X_cal, y_cal, alpha=alpha)
    assert 0.0 <= coverage <= 1.0
    # With a well-fitted GPR, LOO coverage should be close to or above the target
    assert coverage >= (1.0 - alpha) - 0.15  # allow 15% tolerance for small n


def test_cross_validate_coverage_returns_float():
    """cross_validate_coverage must return a float in [0, 1]."""
    gpr = _make_simple_gpr()
    np.random.seed(11)
    X_cal = np.linspace(-3, -1, 15).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 4.5 + np.random.normal(0, 0.05, 15)
    cal = ConformalCalibrator()
    result = cal.cross_validate_coverage(gpr, X_cal, y_cal, alpha=0.05)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_cross_validate_coverage_raises_on_too_few_points():
    """cross_validate_coverage must raise ValueError when n < 3."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    with pytest.raises(ValueError, match="n ≥ 3"):
        cal.cross_validate_coverage(
            gpr,
            np.array([[-1.0], [-2.0]]),
            np.array([5.0, 10.0]),
        )


# ── stratified_coverage ───────────────────────────────────────────────────────

def test_stratified_coverage_returns_dict_with_overall():
    """stratified_coverage must return a dict with an 'overall' key."""
    gpr = _make_simple_gpr()
    np.random.seed(77)
    n = 18
    X_cal = np.linspace(-4.5, -0.3, n).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 5.0 + np.random.normal(0, 0.1, n)
    cal = ConformalCalibrator()
    result = cal.stratified_coverage(gpr, X_cal, y_cal, alpha=0.10, n_bands=3)
    assert isinstance(result, dict)
    assert "overall" in result
    assert "band_0_low" in result
    assert "band_2_high" in result


def test_stratified_coverage_values_in_unit_interval():
    """All coverage fractions must be in [0, 1]."""
    gpr = _make_simple_gpr()
    np.random.seed(55)
    n = 15
    X_cal = np.linspace(-4.0, -0.5, n).reshape(-1, 1)
    y_cal = -X_cal.ravel() * 4.0 + np.random.normal(0, 0.05, n)
    cal = ConformalCalibrator()
    result = cal.stratified_coverage(gpr, X_cal, y_cal, n_bands=3)
    for k, v in result.items():
        assert 0.0 <= v <= 1.0, f"Coverage {k}={v} outside [0, 1]"


def test_stratified_coverage_raises_on_too_few_points():
    """stratified_coverage must raise ValueError when n < n_bands * 2."""
    gpr = _make_simple_gpr()
    cal = ConformalCalibrator()
    with pytest.raises(ValueError, match="n ≥"):
        cal.stratified_coverage(
            gpr,
            np.array([[-1.0], [-2.0], [-3.0]]),
            np.array([5.0, 10.0, 15.0]),
            n_bands=3,
        )
