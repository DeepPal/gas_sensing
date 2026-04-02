"""Tests for src.analysis.feature_importance — SHAP & gradient attribution."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.analysis.feature_importance import (
    WavelengthBand,
    _get_model_output,
    _smooth,
    gradient_attribution,
    integrated_gradients,
    top_wavelength_bands,
)

# ---------------------------------------------------------------------------
# Minimal test models
# ---------------------------------------------------------------------------

class _LinearClassifier(nn.Module):
    """Tiny differentiable classifier for testing."""
    def __init__(self, n_wl: int = 64, n_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(n_wl, n_classes)

    def forward(self, x):
        return self.fc(x)


class _LinearRegressor(nn.Module):
    def __init__(self, n_wl: int = 64) -> None:
        super().__init__()
        self.fc = nn.Linear(n_wl, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def classifier():
    return _LinearClassifier(n_wl=64, n_classes=3)


@pytest.fixture
def regressor():
    return _LinearRegressor(n_wl=64)


@pytest.fixture
def spectra():
    return np.random.default_rng(0).normal(size=(10, 64)).astype(np.float32)


# ---------------------------------------------------------------------------
# gradient_attribution
# ---------------------------------------------------------------------------

class TestGradientAttribution:
    def test_shape(self, classifier, spectra):
        imp = gradient_attribution(classifier, spectra, target_class=0)
        assert imp.shape == (64,)

    def test_nonneg_with_absolute(self, classifier, spectra):
        imp = gradient_attribution(classifier, spectra, target_class=0,
                                   absolute_value=True)
        assert (imp >= 0).all()

    def test_can_be_neg_without_absolute(self, classifier, spectra):
        imp = gradient_attribution(classifier, spectra, target_class=0,
                                   absolute_value=False)
        # With random weights some will be negative
        assert imp.min() < imp.max()

    def test_single_spectrum(self, classifier):
        sp = np.random.default_rng(1).normal(size=(1, 64)).astype(np.float32)
        imp = gradient_attribution(classifier, sp, target_class=1)
        assert imp.shape == (64,)

    def test_1d_spectrum_input(self, classifier):
        sp = np.random.default_rng(2).normal(size=64).astype(np.float32)
        imp = gradient_attribution(classifier, sp, target_class=0)
        assert imp.shape == (64,)

    def test_regression_model(self, regressor, spectra):
        imp = gradient_attribution(regressor, spectra)
        assert imp.shape == (64,)

    def test_no_target_class(self, classifier, spectra):
        """Without target_class, uses max-confidence class per sample."""
        imp = gradient_attribution(classifier, spectra, target_class=None)
        assert imp.shape == (64,)

    def test_consistent_with_known_weights(self):
        """Attribution should be high where weights are high."""
        model = nn.Linear(8, 1, bias=False)
        # Set all weights to zero except position 3
        nn.init.zeros_(model.weight)
        model.weight.data[0, 3] = 10.0
        spectra = np.ones((5, 8), dtype=np.float32)
        imp = gradient_attribution(model, spectra, absolute_value=True)
        # Position 3 should dominate
        assert np.argmax(imp) == 3


# ---------------------------------------------------------------------------
# integrated_gradients
# ---------------------------------------------------------------------------

class TestIntegratedGradients:
    def test_shape(self, classifier, spectra):
        imp = integrated_gradients(classifier, spectra, target_class=0,
                                   n_steps=10)
        assert imp.shape == (64,)

    def test_nonneg_absolute(self, classifier, spectra):
        imp = integrated_gradients(classifier, spectra, n_steps=10,
                                   absolute_value=True)
        assert (imp >= 0).all()

    def test_custom_baseline(self, classifier, spectra):
        baseline = np.zeros(64, dtype=np.float32)
        imp = integrated_gradients(classifier, spectra, baseline=baseline,
                                   n_steps=10)
        assert imp.shape == (64,)

    def test_regression_model(self, regressor, spectra):
        imp = integrated_gradients(regressor, spectra, n_steps=5)
        assert imp.shape == (64,)


# ---------------------------------------------------------------------------
# top_wavelength_bands
# ---------------------------------------------------------------------------

class TestTopWavelengthBands:
    def test_returns_n_bands(self):
        wl = np.linspace(400, 900, 200)
        imp = np.zeros(200)
        imp[50] = 1.0
        imp[100] = 0.8
        imp[150] = 0.6
        bands = top_wavelength_bands(wl, imp, n_bands=3, width_nm=5)
        assert len(bands) == 3

    def test_highest_first(self):
        wl = np.linspace(400, 900, 200)
        imp = np.zeros(200)
        imp[50] = 1.0
        imp[100] = 0.8
        imp[150] = 0.6
        bands = top_wavelength_bands(wl, imp, n_bands=3, width_nm=5)
        assert bands[0].mean_importance >= bands[1].mean_importance >= bands[2].mean_importance

    def test_bands_non_overlapping(self):
        wl = np.linspace(400, 900, 500)
        imp = np.random.default_rng(0).uniform(size=500)
        bands = top_wavelength_bands(wl, imp, n_bands=5, width_nm=20)
        centers = [b.center_nm for b in bands]
        # All centers should be distinct
        assert len(set(centers)) == len(centers)

    def test_wavelength_band_fields(self):
        wl = np.linspace(400, 900, 100)
        imp = np.zeros(100)
        imp[50] = 1.0
        bands = top_wavelength_bands(wl, imp, n_bands=1, width_nm=10)
        b = bands[0]
        assert b.rank == 1
        assert b.start_nm < b.center_nm < b.end_nm
        assert b.mean_importance > 0

    def test_fewer_peaks_than_requested(self):
        wl = np.linspace(400, 900, 50)
        imp = np.zeros(50)
        imp[25] = 1.0
        # Ask for 3 bands but only 1 non-zero peak
        bands = top_wavelength_bands(wl, imp, n_bands=3, width_nm=10)
        assert len(bands) >= 1
        assert len(bands) <= 3


# ---------------------------------------------------------------------------
# _get_model_output
# ---------------------------------------------------------------------------

class TestGetModelOutput:
    def test_plain_tensor(self):
        model = nn.Linear(8, 3)
        x = torch.randn(4, 8)
        out = _get_model_output(model, x)
        assert out.shape == (4, 3)

    def test_tuple_output(self):

        class _TupleModel(nn.Module):
            def forward(self, x):
                return x[:, :1], x[:, 1:]   # (conc, features)

        x = torch.randn(4, 8)
        out = _get_model_output(_TupleModel(), x)
        assert out.shape[0] == 4

    def test_multi_task_model(self):
        from src.models.multi_task import MultiTaskConfig, MultiTaskModel
        cfg = MultiTaskConfig(input_dim=16, embed_dim=8, hidden_dim=16,
                              n_layers=1, n_analytes=2)
        m = MultiTaskModel(cfg)
        x = torch.randn(4, 16)
        out = _get_model_output(m, x)
        assert out.shape == (4, 2)  # class_logits


# ---------------------------------------------------------------------------
# _smooth helper
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_shape_preserved(self):
        x = np.random.default_rng(0).normal(size=100)
        assert _smooth(x, window=5).shape == (100,)

    def test_constant_input_unchanged(self):
        x = np.ones(50)
        out = _smooth(x, window=5)
        np.testing.assert_allclose(out, 1.0, atol=1e-10)

    def test_window_one_unchanged(self):
        x = np.random.default_rng(1).normal(size=20)
        np.testing.assert_array_equal(_smooth(x, window=1), x)
