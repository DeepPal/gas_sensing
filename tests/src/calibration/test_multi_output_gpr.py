"""Tests for src.calibration.multi_output_gpr."""
import numpy as np
import pytest

from src.calibration.multi_output_gpr import (
    IndependentMultiOutputGPR,
    JointMultiOutputGPR,
    build_feature_vector,
)


@pytest.fixture
def calibration_data():
    """Simple linear calibration: Ethanol and Acetone vs 1 peak each."""
    concs = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    # Ethanol: 2 ppm/nm sensitivity; Acetone: 1.5 ppm/nm
    X = np.column_stack([
        -0.25 * concs,    # peak 0 shift
        -0.12 * concs,    # peak 1 shift
    ])
    Y = np.column_stack([concs, concs * 0.75])  # Ethanol, Acetone
    return X, Y


class TestIndependentMultiOutputGPR:
    def test_fit_returns_dict(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone"])
        result = gpr.fit(X, Y)
        assert "Ethanol" in result
        assert "Acetone" in result

    def test_is_fitted_after_fit(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol"])
        gpr.fit(X[:, :1], Y[:, :1])
        assert gpr.is_fitted

    def test_predict_returns_correct_analytes(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        means, stds = gpr.predict(X)
        assert "Ethanol" in means and "Acetone" in means
        assert "Ethanol" in stds and "Acetone" in stds

    def test_predict_shape_correct(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        means, stds = gpr.predict(X)
        assert means["Ethanol"].shape == (5,)
        assert stds["Ethanol"].shape == (5,)

    def test_predict_single_returns_scalars(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        m, s = gpr.predict_single(X[2])
        assert isinstance(m["Ethanol"], float)
        assert isinstance(s["Acetone"], float)

    def test_unfitted_raises(self):
        gpr = IndependentMultiOutputGPR(["X"])
        with pytest.raises(RuntimeError):
            gpr.predict(np.array([[1.0]]))

    def test_wrong_output_count_raises(self, calibration_data):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone", "Toluene"])
        with pytest.raises(ValueError):
            gpr.fit(X, Y)  # Y has 2 columns, but 3 analytes registered

    def test_save_load_roundtrip(self, calibration_data, tmp_path):
        X, Y = calibration_data
        gpr = IndependentMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        m_orig, _ = gpr.predict_single(X[2])

        path = str(tmp_path / "gpr.joblib")
        gpr.save(path)
        loaded = IndependentMultiOutputGPR.load(path)
        m_load, _ = loaded.predict_single(X[2])
        assert abs(m_orig["Ethanol"] - m_load["Ethanol"]) < 1e-6


class TestJointMultiOutputGPR:
    def test_fit_runs(self, calibration_data):
        X, Y = calibration_data
        gpr = JointMultiOutputGPR(["Ethanol", "Acetone"])
        result = gpr.fit(X, Y)
        assert gpr.is_fitted
        assert "n_analytes" in result

    def test_predict_returns_both_analytes(self, calibration_data):
        X, Y = calibration_data
        gpr = JointMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        means, stds = gpr.predict(X)
        assert set(means.keys()) == {"Ethanol", "Acetone"}

    def test_predict_single_scalar(self, calibration_data):
        X, Y = calibration_data
        gpr = JointMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        m, s = gpr.predict_single(X[0])
        assert isinstance(m["Ethanol"], float)

    def test_save_load(self, calibration_data, tmp_path):
        X, Y = calibration_data
        gpr = JointMultiOutputGPR(["Ethanol", "Acetone"])
        gpr.fit(X, Y)
        path = str(tmp_path / "joint.joblib")
        gpr.save(path)
        loaded = JointMultiOutputGPR.load(path)
        assert loaded.is_fitted


class TestBuildFeatureVector:
    def test_spectral_only(self):
        fv, names = build_feature_vector([-0.5, -0.2])
        assert len(fv) == 2
        assert names == ["delta_lambda_0", "delta_lambda_1"]

    def test_with_tau_63(self):
        fv, names = build_feature_vector([-0.5], tau_63=[30.0])
        assert "tau_63_0" in names
        assert len(fv) == 2

    def test_full_kinetic_extension(self):
        fv, names = build_feature_vector(
            [-0.5, -0.2],
            tau_63=[30.0, 25.0],
            tau_95=[90.0, 75.0],
            k_on=[0.033, 0.05],
        )
        # 2 shifts + 2 tau63 + 2 tau95 + 2 kon = 8 features
        assert len(fv) == 8
        assert "tau_63_0" in names
        assert "k_on_1" in names
