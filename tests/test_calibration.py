"""
tests.test_calibration
======================
Unit tests for src.calibration.gpr (GPRCalibration).
"""

import numpy as np
import pytest

from src.calibration.gpr import GPRCalibration

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fitted_gpr():
    """A GPRCalibration fitted on a simple monotone LSPR calibration curve."""
    shifts = np.array([-0.1, -0.5, -1.0, -2.0, -3.0, -4.0])  # Δλ nm
    concs = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0])  # ppm
    gpr = GPRCalibration(n_restarts_optimizer=1)
    gpr.fit(shifts.reshape(-1, 1), concs)
    return gpr


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestGPRCalibrationInit:
    def test_not_fitted_initially(self):
        gpr = GPRCalibration()
        assert not gpr.is_fitted

    def test_predict_single_returns_none_when_unfitted(self):
        gpr = GPRCalibration()
        result = gpr.predict_single(-0.5)
        assert result == (None, None)

    def test_predict_raises_when_unfitted(self):
        gpr = GPRCalibration()
        with pytest.raises(RuntimeError, match="fitted"):
            gpr.predict(np.array([[-0.5]]))


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


class TestGPRCalibrationFit:
    def test_fit_sets_is_fitted(self):
        gpr = GPRCalibration(n_restarts_optimizer=1)
        X = np.array([[-1.0], [-2.0], [-3.0]])
        y = np.array([1.0, 2.0, 3.0])
        gpr.fit(X, y)
        assert gpr.is_fitted

    def test_fit_returns_dict_with_expected_keys(self):
        gpr = GPRCalibration(n_restarts_optimizer=1)
        X = np.linspace(-0.5, -4.0, 8).reshape(-1, 1)
        y = np.linspace(0.5, 4.0, 8)
        info = gpr.fit(X, y)
        assert "log_marginal_likelihood" in info
        assert "n_samples" in info
        assert info["n_samples"] == 8

    def test_fit_1d_input_auto_reshape(self):
        """1-D array for X should be reshaped automatically."""
        gpr = GPRCalibration(n_restarts_optimizer=1)
        X = np.array([-0.5, -1.0, -2.0])
        y = np.array([0.5, 1.0, 2.0])
        gpr.fit(X, y)
        assert gpr.is_fitted

    def test_optimize_hyperparameters_alias(self):
        gpr = GPRCalibration(n_restarts_optimizer=1)
        X = np.array([[-1.0], [-2.0], [-3.0]])
        y = np.array([1.0, 2.0, 3.0])
        # Should not raise
        gpr.optimize_hyperparameters(X, y)
        assert gpr.is_fitted


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestGPRCalibrationPredict:
    def test_predict_shape(self, fitted_gpr):
        X = np.array([[-0.5], [-1.5], [-2.5]])
        mean, std = fitted_gpr.predict(X)
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_predict_monotone_trend(self, fitted_gpr):
        """Larger shift magnitude should map to higher concentration."""
        shifts = np.array([[-0.5], [-1.0], [-2.0], [-3.0]])
        mean, _ = fitted_gpr.predict(shifts)
        # Concentrations should be increasing (more shift → more gas)
        assert mean[0] < mean[1] < mean[2] < mean[3]

    def test_predict_non_negative_std(self, fitted_gpr):
        X = np.linspace(-0.5, -3.5, 5).reshape(-1, 1)
        _, std = fitted_gpr.predict(X)
        assert np.all(std >= 0)

    def test_predict_std_false_returns_zeros(self, fitted_gpr):
        X = np.array([[-1.0]])
        mean, std = fitted_gpr.predict(X, return_std=False)
        assert std[0] == 0.0

    def test_predict_single_returns_floats(self, fitted_gpr):
        mean, std = fitted_gpr.predict_single(-1.0)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_predict_single_reasonable_value(self, fitted_gpr):
        """A shift of −1 nm should predict roughly 1 ppm."""
        mean, std = fitted_gpr.predict_single(-1.0)
        assert 0.0 < mean < 5.0  # loose sanity check

    def test_predict_1d_input_auto_reshape(self, fitted_gpr):
        """1-D X in predict() should be handled."""
        mean, std = fitted_gpr.predict(np.array([-1.0, -2.0]))
        assert mean.shape == (2,)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------


class TestGPRCalibrationPersistence:
    def test_save_load_roundtrip(self, fitted_gpr, tmp_path):
        path = str(tmp_path / "gpr.joblib")
        fitted_gpr.save(path)

        loaded = GPRCalibration.load(path)
        assert loaded.is_fitted

        orig_mean, _ = fitted_gpr.predict(np.array([[-1.5]]))
        load_mean, _ = loaded.predict(np.array([[-1.5]]))
        np.testing.assert_allclose(orig_mean, load_mean, rtol=1e-6)

    def test_load_preserves_n_train(self, fitted_gpr, tmp_path):
        path = str(tmp_path / "gpr.joblib")
        fitted_gpr.save(path)
        loaded = GPRCalibration.load(path)
        assert loaded._n_train == fitted_gpr._n_train


# ===========================================================================
# src.calibration.transforms
# ===========================================================================


import math  # noqa: E402

from src.calibration.multi_roi import fit_multi_roi_fusion, select_multi_roi_candidates
from src.calibration.transforms import transform_concentrations


class TestTransformConcentrations:
    def test_linear_passthrough(self):
        concs = np.array([0.5, 1.0, 2.0])
        t, meta = transform_concentrations(concs, "linear")
        np.testing.assert_array_equal(t, concs)
        assert meta == {}

    def test_unknown_mode_passthrough(self):
        concs = np.array([1.0, 2.0, 3.0])
        t, meta = transform_concentrations(concs, "power_law")
        np.testing.assert_array_equal(t, concs)
        assert meta == {}

    def test_log10_transform_monotone(self):
        concs = np.array([0.1, 1.0, 10.0])
        t, meta = transform_concentrations(concs, "log10")
        assert "offset" in meta
        assert t[0] < t[1] < t[2]

    def test_log_transform_monotone(self):
        concs = np.array([1.0, 2.0, 4.0])
        t, meta = transform_concentrations(concs, "log")
        assert "offset" in meta
        assert t[0] < t[1] < t[2]

    def test_sqrt_transform(self):
        concs = np.array([0.0, 1.0, 4.0, 9.0])
        t, meta = transform_concentrations(concs, "sqrt")
        np.testing.assert_allclose(t, [0.0, 1.0, 2.0, 3.0])
        assert meta == {}

    def test_log10_case_insensitive(self):
        concs = np.array([1.0, 2.0])
        t1, _ = transform_concentrations(concs, "LOG10")
        t2, _ = transform_concentrations(concs, "log10")
        np.testing.assert_array_equal(t1, t2)

    def test_log_negative_clamped_no_nan(self):
        """Negative concentrations are clamped to the positive offset — no NaN."""
        concs = np.array([-1.0, 0.5, 1.0])
        t, _ = transform_concentrations(concs, "log10")
        assert np.all(np.isfinite(t))

    def test_log10_offset_positive(self):
        concs = np.array([0.5, 1.0, 2.0])
        _, meta = transform_concentrations(concs, "log10")
        assert meta["offset"] > 0.0


# ===========================================================================
# src.calibration.multi_roi — select_multi_roi_candidates
# ===========================================================================


def _make_candidate(
    center_nm: float,
    slope: float = -2.0,
    r2: float = 0.95,
    quality_ok: bool = True,
    n: int = 4,
) -> dict:
    concs = list(np.linspace(0.5, 2.0, n))
    deltas = [slope * c for c in concs]
    return {
        "quality_ok": quality_ok,
        "center_nm": center_nm,
        "slope_nm_per_ppm": slope,
        "r2": r2,
        "snr": 5.0,
        "deltas_valid_nm": deltas,
        "concentrations_ppm": concs,
    }


class TestSelectMultiRoiCandidates:
    def test_empty_dict_returns_empty(self):
        assert select_multi_roi_candidates({}) == []

    def test_non_dict_returns_empty(self):
        assert select_multi_roi_candidates("invalid") == []  # type: ignore[arg-type]

    def test_quality_false_filtered(self):
        roi = {"candidates": [_make_candidate(700.0, quality_ok=False)]}
        assert select_multi_roi_candidates(roi) == []

    def test_max_features_respected(self):
        candidates = [_make_candidate(700.0 + i * 5) for i in range(10)]
        roi = {"candidates": candidates}
        assert len(select_multi_roi_candidates(roi, max_features=3)) == 3

    def test_deduplication_by_center(self):
        candidates = [_make_candidate(700.0), _make_candidate(700.0, slope=-1.5)]
        roi = {"candidates": candidates}
        assert len(select_multi_roi_candidates(roi)) == 1

    def test_sorted_by_slope_magnitude(self):
        """Candidate with higher |slope| should rank first."""
        low_slope = _make_candidate(700.0, slope=-1.0, r2=0.99)
        high_slope = _make_candidate(710.0, slope=-3.0, r2=0.80)
        roi = {"candidates": [low_slope, high_slope]}
        result = select_multi_roi_candidates(roi)
        assert float(result[0]["center_nm"]) == pytest.approx(710.0)

    def test_too_few_deltas_filtered(self):
        cand = {
            "quality_ok": True,
            "center_nm": 700.0,
            "slope_nm_per_ppm": -2.0,
            "r2": 0.95,
            "deltas_valid_nm": [-1.0],
            "concentrations_ppm": [0.5],
        }
        assert select_multi_roi_candidates({"candidates": [cand]}) == []


# ===========================================================================
# src.calibration.multi_roi — fit_multi_roi_fusion
# ===========================================================================


def _make_fusion_roi(
    n_features: int = 2,
    n_points: int = 5,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)
    concs = list(np.linspace(0.5, 2.5, n_points))
    candidates = []
    for k in range(n_features):
        slope = -(k + 1) * 1.5
        deltas = [slope * c + float(rng.normal(0, 0.05)) for c in concs]
        candidates.append(
            {
                "quality_ok": True,
                "center_nm": 700.0 + k * 10.0,
                "slope_nm_per_ppm": slope,
                "r2": 0.97,
                "snr": 8.0,
                "deltas_valid_nm": deltas,
                "concentrations_ppm": concs,
            }
        )
    return {"candidates": candidates}, concs


class TestFitMultiRoiFusion:
    def test_returns_none_too_few_concentrations(self):
        roi, concs = _make_fusion_roi(n_points=2)
        assert fit_multi_roi_fusion(roi, concs) is None

    def test_returns_none_too_few_features(self):
        roi, concs = _make_fusion_roi(n_features=1)
        assert fit_multi_roi_fusion(roi, concs, max_features=4) is None

    def test_returns_metrics_dict_with_all_keys(self):
        roi, concs = _make_fusion_roi(n_features=2, n_points=5)
        metrics = fit_multi_roi_fusion(roi, concs)
        assert metrics is not None
        for key in [
            "n_points", "n_features", "feature_centers_nm", "coefficients",
            "intercept_ppm", "r2", "rmse_ppm", "lod_ppm", "r2_cv",
            "rmse_cv_ppm", "actual_concentrations_ppm",
            "predicted_concentrations_ppm", "cv_predictions_ppm",
            "residuals_ppm", "features",
        ]:
            assert key in metrics

    def test_r2_reasonable_on_clean_data(self):
        roi, concs = _make_fusion_roi(n_features=2, n_points=6)
        metrics = fit_multi_roi_fusion(roi, concs)
        assert metrics is not None
        assert float(metrics["r2"]) > 0.8

    def test_loocv_finite_for_n_ge_4(self):
        roi, concs = _make_fusion_roi(n_features=2, n_points=6)
        metrics = fit_multi_roi_fusion(roi, concs)
        assert metrics is not None
        assert metrics["cv_predictions_ppm"] is not None
        assert math.isfinite(float(metrics["r2_cv"]))
        assert math.isfinite(float(metrics["rmse_cv_ppm"]))

    def test_loocv_absent_for_n_lt_4(self):
        roi, concs = _make_fusion_roi(n_features=2, n_points=3)
        metrics = fit_multi_roi_fusion(roi, concs)
        assert metrics is not None
        assert metrics["cv_predictions_ppm"] is None
        assert not math.isfinite(float(metrics["r2_cv"]))

    def test_n_features_capped_by_max_features(self):
        roi, concs = _make_fusion_roi(n_features=6, n_points=7)
        metrics = fit_multi_roi_fusion(roi, concs, max_features=2)
        assert metrics is not None
        assert int(metrics["n_features"]) <= 2

    def test_infinite_concentrations_returns_none(self):
        roi, _ = _make_fusion_roi(n_features=2, n_points=4)
        assert fit_multi_roi_fusion(roi, [0.5, float("inf"), 1.0, 2.0]) is None
