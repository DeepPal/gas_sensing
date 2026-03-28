# tests/src/calibration/test_physics_kernel.py
import numpy as np
import pytest

from src.calibration.physics_kernel import (
    LangmuirMeanFunction,
    PhysicsInformedGPR,
    fit_langmuir_params,
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


def test_physics_informed_gpr_explicit_mode():
    """mode='shift_to_conc' must work identically to auto-detection for negative shifts."""
    np.random.seed(2)
    concs = np.array([0.5, 1.0, 2.0, 3.0])
    shifts = -10.0 * concs / (1.0 + concs)
    auto_model = PhysicsInformedGPR(mode="auto")
    auto_model.fit(shifts.reshape(-1, 1), concs)
    explicit_model = PhysicsInformedGPR(mode="shift_to_conc")
    explicit_model.fit(shifts.reshape(-1, 1), concs)
    X_test = np.array([[-1.5]])
    mean_auto, _ = auto_model.predict(X_test)
    mean_explicit, _ = explicit_model.predict(X_test)
    assert abs(mean_auto[0] - mean_explicit[0]) < 0.1


def test_physics_informed_gpr_invalid_mode():
    """mode with unexpected value must raise ValueError immediately."""
    with pytest.raises(ValueError, match="mode must be"):
        PhysicsInformedGPR(mode="bad_mode")


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
    # A shift of -1.5 nm corresponds to a positive concentration
    assert mean[0] > 0.0, f"Expected positive concentration for Δλ=-1.5 nm, got {mean[0]}"
