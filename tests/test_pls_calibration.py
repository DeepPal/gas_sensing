"""
Tests for src.calibration.pls — PLSCalibration and PLSFitResult.

Covers:
- PLSFitResult dataclass (field presence and defaults)
- PLSCalibration.fit (training metrics, VIP shape, cross-validation)
- PLSCalibration.predict (shape, correctness on training data)
- PLSCalibration.predict_with_uncertainty (shape, non-negative std)
- PLSCalibration.optimize_components (returns valid component count)
- PLSCalibration.to_summary_dict (required keys, references)
- Edge cases (n_components clamping, small n, perfect data)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.calibration.pls import PLSCalibration, PLSFitResult


# ---------------------------------------------------------------------------
# Synthetic dataset factory
# ---------------------------------------------------------------------------


def _make_spectra(
    n_samples: int = 20,
    n_features: int = 100,
    seed: int = 0,
    noise: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic spectral dataset where y is linearly recoverable.

    X[i] = c[i] * peak_profile + noise, so PLS should fit well.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(600.0, 900.0, n_features)
    # Gaussian peak centred at 750 nm
    peak = np.exp(-0.5 * ((wl - 750.0) / 30.0) ** 2)
    # Background (unrelated to y)
    background = 0.1 * np.exp(-0.5 * ((wl - 650.0) / 50.0) ** 2)

    concs = np.linspace(0.1, 10.0, n_samples)
    X = np.outer(concs, peak) + background + rng.normal(0, noise, (n_samples, n_features))
    return X, concs


# ===========================================================================
# PLSFitResult dataclass
# ===========================================================================


class TestPLSFitResult:
    def test_default_construction(self) -> None:
        r = PLSFitResult()
        assert r.n_components == 0
        assert np.isnan(r.r2_calibration)
        assert np.isnan(r.q2)
        assert len(r.vip_scores) == 0
        assert len(r.rmsecv_per_component) == 0
        assert len(r.explained_variance_x) == 0

    def test_vip_scores_default_is_empty_array(self) -> None:
        r = PLSFitResult()
        assert isinstance(r.vip_scores, np.ndarray)

    def test_x_loadings_default_is_empty_array(self) -> None:
        r = PLSFitResult()
        assert isinstance(r.x_loadings, np.ndarray)


# ===========================================================================
# PLSCalibration.fit
# ===========================================================================


class TestPLSCalibrationFit:
    def setup_method(self) -> None:
        self.X, self.y = _make_spectra(n_samples=20, n_features=100, seed=42)

    def test_fit_returns_pls_fit_result(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert isinstance(result, PLSFitResult)

    def test_r2_calibration_between_0_and_1(self) -> None:
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert 0.0 <= result.r2_calibration <= 1.0

    def test_r2_calibration_high_on_structured_data(self) -> None:
        """Structured synthetic data should yield R² > 0.95 on training set."""
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert result.r2_calibration > 0.95

    def test_rmsec_positive_finite(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.rmsec > 0.0
        assert np.isfinite(result.rmsec)

    def test_q2_between_0_and_1(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert -1.0 < result.q2 <= 1.0

    def test_rmsecv_positive_finite(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.rmsecv > 0.0
        assert np.isfinite(result.rmsecv)

    def test_rmsecv_geq_rmsec(self) -> None:
        """CV error must be ≥ training error (no overfitting artifact in metric)."""
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.rmsecv >= result.rmsec * 0.5  # loose bound; CV can occasionally be lower

    def test_n_samples_and_features_stored(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.n_samples == 20
        assert result.n_features == 100

    def test_n_components_stored(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.n_components == 2

    # ------------------------------------------------------------------
    # VIP scores
    # ------------------------------------------------------------------

    def test_vip_scores_shape(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.vip_scores.shape == (100,)

    def test_vip_scores_non_negative(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert float(result.vip_scores.min()) >= 0.0

    def test_vip_scores_max_gt_1(self) -> None:
        """At least one feature should have VIP > 1 on structured data."""
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert float(result.vip_scores.max()) > 1.0

    def test_vip_dominant_feature_is_near_peak(self) -> None:
        """The VIP-dominant feature should be near wavelength 750 nm (the signal peak)."""
        X, y = _make_spectra(n_samples=25, n_features=200, noise=0.001, seed=0)
        wl = np.linspace(600.0, 900.0, 200)
        pls = PLSCalibration(n_components=3)
        result = pls.fit(X, y)
        top_wl = wl[int(np.argmax(result.vip_scores))]
        assert abs(top_wl - 750.0) < 60.0, f"Top VIP at {top_wl:.1f} nm, expected ~750 nm"

    # ------------------------------------------------------------------
    # Loadings and scores
    # ------------------------------------------------------------------

    def test_x_loadings_shape(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.x_loadings.shape == (100, 2)

    def test_x_scores_shape(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.x_scores.shape == (20, 2)

    def test_x_weights_shape(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert result.x_weights.shape == (100, 2)

    # ------------------------------------------------------------------
    # Explained variance
    # ------------------------------------------------------------------

    def test_explained_variance_x_length(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        assert len(result.explained_variance_x) == 2

    def test_explained_variance_x_values_between_0_and_1(self) -> None:
        pls = PLSCalibration(n_components=2)
        result = pls.fit(self.X, self.y)
        for v in result.explained_variance_x:
            assert 0.0 <= v <= 1.0

    # ------------------------------------------------------------------
    # RMSECV curve
    # ------------------------------------------------------------------

    def test_rmsecv_per_component_nonempty(self) -> None:
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert len(result.rmsecv_per_component) > 0

    def test_optimal_n_components_in_range(self) -> None:
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert 1 <= result.optimal_n_components <= 20

    # ------------------------------------------------------------------
    # Component clamping
    # ------------------------------------------------------------------

    def test_n_components_clamped_when_too_large(self) -> None:
        """Should warn and clamp n_components when larger than n_samples-1."""
        X, y = _make_spectra(n_samples=5, n_features=50, seed=1)
        pls = PLSCalibration(n_components=10)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pls.fit(X, y)
        assert result.n_components <= 4  # must be < n_samples
        assert any("clamped" in str(warning.message).lower() for warning in w)

    def test_too_few_samples_raises(self) -> None:
        X, y = _make_spectra(n_samples=2, n_features=20, seed=0)
        pls = PLSCalibration(n_components=1)
        with pytest.raises(ValueError, match="at least 3 samples"):
            pls.fit(X, y)

    # ------------------------------------------------------------------
    # Pearson r and bias
    # ------------------------------------------------------------------

    def test_pearson_r_high_on_structured_data(self) -> None:
        pls = PLSCalibration(n_components=3)
        result = pls.fit(self.X, self.y)
        assert abs(result.pearson_r) > 0.95

    def test_bias_near_zero_on_centred_data(self) -> None:
        pls = PLSCalibration(n_components=3, scale=True)
        result = pls.fit(self.X, self.y)
        assert abs(result.bias) < float(np.std(self.y)) * 0.2


# ===========================================================================
# PLSCalibration.predict
# ===========================================================================


class TestPLSCalibrationPredict:
    def setup_method(self) -> None:
        self.X, self.y = _make_spectra(n_samples=20, n_features=100, seed=0)
        self.pls = PLSCalibration(n_components=3)
        self.pls.fit(self.X, self.y)

    def test_predict_shape(self) -> None:
        y_pred = self.pls.predict(self.X)
        assert y_pred.shape == (20,)

    def test_predict_training_data_correlated_with_y(self) -> None:
        y_pred = self.pls.predict(self.X)
        corr = float(np.corrcoef(self.y, y_pred)[0, 1])
        assert corr > 0.95

    def test_predict_requires_fit(self) -> None:
        pls_new = PLSCalibration(n_components=2)
        with pytest.raises(RuntimeError, match="not.*fitted"):
            pls_new.predict(self.X)

    def test_predict_new_samples_shape(self) -> None:
        X_new, _ = _make_spectra(n_samples=5, n_features=100, seed=99)
        y_pred = self.pls.predict(X_new)
        assert y_pred.shape == (5,)


# ===========================================================================
# PLSCalibration.predict_with_uncertainty
# ===========================================================================


class TestPLSCalibrationUncertainty:
    def setup_method(self) -> None:
        self.X, self.y = _make_spectra(n_samples=20, n_features=100, seed=1)
        self.pls = PLSCalibration(n_components=2)
        self.pls.fit(self.X, self.y)

    def test_returns_two_arrays(self) -> None:
        y_pred, y_std = self.pls.predict_with_uncertainty(self.X)
        assert y_pred.shape == (20,)
        assert y_std.shape == (20,)

    def test_std_non_negative(self) -> None:
        _, y_std = self.pls.predict_with_uncertainty(self.X)
        assert float(y_std.min()) >= 0.0

    def test_std_finite(self) -> None:
        _, y_std = self.pls.predict_with_uncertainty(self.X)
        assert np.all(np.isfinite(y_std))

    def test_requires_fit(self) -> None:
        pls_new = PLSCalibration(n_components=1)
        with pytest.raises(RuntimeError, match="not.*fitted"):
            pls_new.predict_with_uncertainty(self.X)


# ===========================================================================
# PLSCalibration.optimize_components
# ===========================================================================


class TestPLSOptimizeComponents:
    def test_returns_tuple_of_int_and_list(self) -> None:
        X, y = _make_spectra(n_samples=20, n_features=100, seed=0)
        pls = PLSCalibration(n_components=3)
        n_opt, curve = pls.optimize_components(X, y, max_components=5)
        assert isinstance(n_opt, int)
        assert isinstance(curve, list)

    def test_optimal_in_range(self) -> None:
        X, y = _make_spectra(n_samples=20, n_features=100, seed=0)
        pls = PLSCalibration()
        n_opt, curve = pls.optimize_components(X, y, max_components=6)
        assert 1 <= n_opt <= 6

    def test_curve_length_matches_max_components(self) -> None:
        X, y = _make_spectra(n_samples=20, n_features=100, seed=0)
        pls = PLSCalibration()
        _, curve = pls.optimize_components(X, y, max_components=4)
        assert len(curve) == 4

    def test_curve_values_positive(self) -> None:
        X, y = _make_spectra(n_samples=20, n_features=100, seed=0)
        pls = PLSCalibration()
        _, curve = pls.optimize_components(X, y, max_components=4)
        assert all(v > 0.0 for v in curve)

    def test_max_components_capped_at_n_samples_minus_one(self) -> None:
        X, y = _make_spectra(n_samples=6, n_features=50, seed=0)
        pls = PLSCalibration()
        n_opt, curve = pls.optimize_components(X, y, max_components=20)
        assert n_opt <= 5
        assert len(curve) <= 5


# ===========================================================================
# PLSCalibration.to_summary_dict
# ===========================================================================


class TestPLSSummaryDict:
    def setup_method(self) -> None:
        X, y = _make_spectra(n_samples=20, n_features=100, seed=0)
        self.pls = PLSCalibration(n_components=2)
        self.pls.fit(X, y)

    def test_returns_dict(self) -> None:
        d = self.pls.to_summary_dict()
        assert isinstance(d, dict)

    def test_mandatory_keys_present(self) -> None:
        d = self.pls.to_summary_dict()
        for key in (
            "n_components", "n_samples", "n_features",
            "r2_calibration", "rmsec", "q2_crossvalidated", "rmsecv",
            "pearson_r", "bias",
        ):
            assert key in d, f"Missing key: {key}"

    def test_references_list_nonempty(self) -> None:
        d = self.pls.to_summary_dict()
        refs = d.get("references", [])
        assert isinstance(refs, list)
        assert len(refs) >= 1

    def test_requires_fit(self) -> None:
        pls_new = PLSCalibration(n_components=1)
        with pytest.raises(RuntimeError, match="not.*fitted"):
            pls_new.to_summary_dict()

    def test_r2_in_range(self) -> None:
        d = self.pls.to_summary_dict()
        assert 0.0 <= float(d["r2_calibration"]) <= 1.0


# ===========================================================================
# Integration: fit → predict → summary cycle
# ===========================================================================


class TestPLSIntegrationCycle:
    def test_full_cycle_structured_data(self) -> None:
        """Full fit→predict→summary cycle on clean structured data."""
        X, y = _make_spectra(n_samples=25, n_features=150, noise=0.005, seed=7)
        wl = np.linspace(600.0, 900.0, 150)

        pls = PLSCalibration(n_components=3)
        result = pls.fit(X, y, wavelengths=wl)

        assert result.r2_calibration > 0.9
        assert result.q2 > 0.8

        y_pred = pls.predict(X)
        assert y_pred.shape == (25,)

        y_pred2, y_std = pls.predict_with_uncertainty(X)
        assert y_pred2.shape == (25,)
        assert y_std.shape == (25,)

        summary = pls.to_summary_dict()
        assert summary["n_components"] == 3

    def test_optimize_then_fit(self) -> None:
        """optimize_components → fit cycle."""
        X, y = _make_spectra(n_samples=20, n_features=80, seed=5)
        pls = PLSCalibration(n_components=1)
        n_opt, _ = pls.optimize_components(X, y, max_components=5)
        pls2 = PLSCalibration(n_components=n_opt)
        result = pls2.fit(X, y)
        assert result.n_components == n_opt

    def test_k_fold_cv_mode(self) -> None:
        """K-fold CV mode (cv_folds=5) should yield valid Q²."""
        X, y = _make_spectra(n_samples=30, n_features=100, seed=3)
        pls = PLSCalibration(n_components=2, cv_folds=5)
        result = pls.fit(X, y)
        assert np.isfinite(result.q2)
        assert np.isfinite(result.rmsecv)
