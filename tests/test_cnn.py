"""
tests.test_cnn
==============
Unit tests for src.models.cnn:
  - GasCNN  (nn.Module — architecture, forward pass)
  - CNNGasClassifier  (fit, predict, predict_single, predict_with_uncertainty,
                       save, load)

All tests are skipped gracefully when PyTorch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False

_SKIP = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_clf():
    """Minimal fitted CNNGasClassifier: 40-point spectra, 3 classes, 3 epochs."""
    if not _TORCH:
        pytest.skip("PyTorch not installed")
    from src.models.cnn import CNNGasClassifier

    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 40)).astype(np.float32)
    y_cls = np.array([i % 3 for i in range(n)], dtype=int)
    y_conc = rng.uniform(0.5, 5.0, n).astype(np.float32)

    clf = CNNGasClassifier(input_length=40, num_classes=3, device="cpu")
    clf.fit(X, y_cls, y_conc, class_names=["GasA", "GasB", "GasC"], epochs=3, batch_size=10)
    return clf


# ---------------------------------------------------------------------------
# GasCNN
# ---------------------------------------------------------------------------


class TestGasCNN:
    @_SKIP
    def test_forward_output_shapes(self):
        from src.models.cnn import GasCNN

        model = GasCNN(input_length=100, num_classes=4)
        x = torch.zeros(2, 1, 100)
        logits, conc = model(x)
        assert logits.shape == (2, 4)
        assert conc.shape == (2, 1)

    @_SKIP
    def test_different_input_lengths(self):
        from src.models.cnn import GasCNN

        for length in [50, 200, 1000]:
            model = GasCNN(input_length=length, num_classes=3)
            x = torch.zeros(1, 1, length)
            logits, conc = model(x)
            assert logits.shape == (1, 3)

    @_SKIP
    def test_num_classes_reflected_in_output(self):
        from src.models.cnn import GasCNN

        for n_cls in [2, 5, 10]:
            model = GasCNN(input_length=100, num_classes=n_cls)
            x = torch.zeros(1, 1, 100)
            logits, _ = model(x)
            assert logits.shape[1] == n_cls

    @_SKIP
    def test_output_finite(self):
        from src.models.cnn import GasCNN

        model = GasCNN(input_length=80, num_classes=2)
        x = torch.randn(4, 1, 80)
        logits, conc = model(x)
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isfinite(conc))


# ---------------------------------------------------------------------------
# CNNGasClassifier — init and state
# ---------------------------------------------------------------------------


class TestCNNGasClassifierInit:
    @_SKIP
    def test_not_fitted_initially(self):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=100, num_classes=3, device="cpu")
        assert not clf.is_fitted
        assert clf.model is None

    @_SKIP
    def test_device_auto_selects_cpu(self):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(device="auto")
        # On CI (no GPU), auto should resolve to cpu
        assert str(clf.device) in ("cpu", "cuda:0")


# ---------------------------------------------------------------------------
# CNNGasClassifier — fit
# ---------------------------------------------------------------------------


class TestCNNGasClassifierFit:
    @_SKIP
    def test_fit_sets_is_fitted(self, tiny_clf):
        assert tiny_clf.is_fitted

    @_SKIP
    def test_fit_returns_history_dict(self):
        from src.models.cnn import CNNGasClassifier

        rng = np.random.default_rng(0)
        X = rng.standard_normal((12, 40)).astype(np.float32)
        y_cls = np.array([i % 2 for i in range(12)], dtype=int)
        y_conc = rng.uniform(0, 1, 12).astype(np.float32)
        clf = CNNGasClassifier(input_length=40, num_classes=2, device="cpu")
        hist = clf.fit(X, y_cls, y_conc, class_names=["A", "B"], epochs=2, batch_size=6)
        assert "loss" in hist and "cls_acc" in hist
        assert len(hist["loss"]) == 2

    @_SKIP
    def test_class_map_populated_after_fit(self, tiny_clf):
        assert len(tiny_clf.class_map) == 3
        assert set(tiny_clf.class_map.values()) == {"GasA", "GasB", "GasC"}


# ---------------------------------------------------------------------------
# CNNGasClassifier — predict
# ---------------------------------------------------------------------------


class TestCNNGasClassifierPredict:
    @_SKIP
    def test_predict_returns_names_and_concentrations(self, tiny_clf):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((5, 40)).astype(np.float32)
        gas_names, concs = tiny_clf.predict(X)
        assert len(gas_names) == 5
        assert concs.shape == (5,)

    @_SKIP
    def test_predict_gas_names_in_class_map(self, tiny_clf):
        X = np.ones((3, 40), dtype=np.float32)
        gas_names, _ = tiny_clf.predict(X)
        for name in gas_names:
            assert name in {"GasA", "GasB", "GasC"}

    @_SKIP
    def test_predict_concentrations_finite(self, tiny_clf):
        X = np.random.default_rng(0).standard_normal((4, 40)).astype(np.float32)
        _, concs = tiny_clf.predict(X)
        assert np.all(np.isfinite(concs))

    @_SKIP
    def test_predict_unfitted_raises(self):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=40, num_classes=2, device="cpu")
        with pytest.raises(RuntimeError, match="fitted"):
            clf.predict(np.ones((2, 40)))

    @_SKIP
    def test_predict_resamples_different_length(self, tiny_clf):
        """Input length != model input_length should be resampled silently."""
        X = np.ones((2, 100), dtype=np.float32)  # 100 != 40
        gas_names, concs = tiny_clf.predict(X)
        assert len(gas_names) == 2


# ---------------------------------------------------------------------------
# CNNGasClassifier — predict_single
# ---------------------------------------------------------------------------


class TestCNNGasClassifierPredictSingle:
    @_SKIP
    def test_returns_three_values(self, tiny_clf):
        spec = np.ones(40, dtype=np.float32)
        result = tiny_clf.predict_single(spec)
        assert len(result) == 3

    @_SKIP
    def test_confidence_in_range(self, tiny_clf):
        spec = np.random.default_rng(0).standard_normal(40).astype(np.float32)
        _, _, conf = tiny_clf.predict_single(spec)
        assert 0.0 <= conf <= 1.0

    @_SKIP
    def test_unfitted_returns_unknown(self):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=40, num_classes=2, device="cpu")
        gas, conc, conf = clf.predict_single(np.ones(40))
        assert gas == "unknown"
        assert conf == 0.0


# ---------------------------------------------------------------------------
# CNNGasClassifier — predict_with_uncertainty (MC Dropout)
# ---------------------------------------------------------------------------


class TestCNNGasClassifierMCDropout:
    @_SKIP
    def test_returns_four_values(self, tiny_clf):
        spec = np.ones(40, dtype=np.float32)
        result = tiny_clf.predict_with_uncertainty(spec, n_samples=5)
        assert len(result) == 4

    @_SKIP
    def test_gas_name_valid(self, tiny_clf):
        spec = np.random.default_rng(7).standard_normal(40).astype(np.float32)
        gas, _, _, _ = tiny_clf.predict_with_uncertainty(spec, n_samples=5)
        assert gas in {"GasA", "GasB", "GasC"}

    @_SKIP
    def test_conc_std_nonnegative(self, tiny_clf):
        spec = np.random.default_rng(8).standard_normal(40).astype(np.float32)
        _, _, conc_std, _ = tiny_clf.predict_with_uncertainty(spec, n_samples=10)
        assert conc_std >= 0.0

    @_SKIP
    def test_cls_entropy_nonnegative(self, tiny_clf):
        spec = np.ones(40, dtype=np.float32)
        _, _, _, entropy = tiny_clf.predict_with_uncertainty(spec, n_samples=5)
        assert entropy >= 0.0

    @_SKIP
    def test_model_restored_to_eval_after_mc(self, tiny_clf):
        """model.training must be False after MC Dropout call."""
        spec = np.ones(40, dtype=np.float32)
        tiny_clf.predict_with_uncertainty(spec, n_samples=5)
        assert not tiny_clf.model.training

    @_SKIP
    def test_unfitted_returns_zeros(self):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=40, num_classes=2, device="cpu")
        gas, mean, std, ent = clf.predict_with_uncertainty(np.ones(40))
        assert gas == "unknown"
        assert mean == 0.0 and std == 0.0 and ent == 0.0


# ---------------------------------------------------------------------------
# CNNGasClassifier — save / load roundtrip
# ---------------------------------------------------------------------------


class TestCNNGasClassifierPersistence:
    @_SKIP
    def test_save_creates_file(self, tiny_clf, tmp_path):
        path = tmp_path / "clf.pt"
        tiny_clf.save(str(path))
        assert path.exists()
        assert path.stat().st_size > 0

    @_SKIP
    def test_load_roundtrip(self, tiny_clf, tmp_path):
        from src.models.cnn import CNNGasClassifier

        path = tmp_path / "clf.pt"
        tiny_clf.save(str(path))
        loaded = CNNGasClassifier.load(str(path), device="cpu")
        assert loaded.is_fitted
        assert loaded.input_length == tiny_clf.input_length
        assert loaded.class_map == tiny_clf.class_map

    @_SKIP
    def test_loaded_model_predicts_consistently(self, tiny_clf, tmp_path):
        from src.models.cnn import CNNGasClassifier

        path = tmp_path / "clf.pt"
        tiny_clf.save(str(path))
        loaded = CNNGasClassifier.load(str(path), device="cpu")
        spec = np.ones(40, dtype=np.float32)
        gas_orig, conc_orig, conf_orig = tiny_clf.predict_single(spec)
        gas_load, conc_load, conf_load = loaded.predict_single(spec)
        assert gas_orig == gas_load
        assert abs(conc_orig - conc_load) < 1e-4

    @_SKIP
    def test_save_unfitted_raises(self, tmp_path):
        from src.models.cnn import CNNGasClassifier

        clf = CNNGasClassifier(input_length=40, num_classes=2, device="cpu")
        with pytest.raises(RuntimeError, match="No model"):
            clf.save(str(tmp_path / "bad.pt"))


# ---------------------------------------------------------------------------
# augment_spectra (pure-numpy, no torch required)
# ---------------------------------------------------------------------------


class TestAugmentSpectra:
    def _make_data(self, n=10, L=100):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, L)).astype(np.float32)
        y_cls = np.array([i % 3 for i in range(n)], dtype=int)
        y_conc = rng.uniform(0.5, 5.0, n).astype(np.float32)
        return X, y_cls, y_conc

    def test_output_size_with_n_augment_4(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=10)
        Xa, ya, ca = augment_spectra(X, y, c, n_augment=4)
        assert Xa.shape[0] == 10 * 5  # originals + 4 copies

    def test_output_width_unchanged(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=8, L=50)
        Xa, ya, ca = augment_spectra(X, y, c)
        assert Xa.shape[1] == 50

    def test_labels_repeated_correctly(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=6)
        Xa, ya, ca = augment_spectra(X, y, c, n_augment=2)
        # First 6 rows are originals; labels must match
        np.testing.assert_array_equal(ya[:6], y)

    def test_augmented_data_differs_from_originals(self):
        """Augmented spectra should not be identical to originals."""
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=5, L=80)
        Xa, ya, ca = augment_spectra(X, y, c, n_augment=1)
        # Rows 5:10 should differ from rows 0:5
        orig = Xa[:5]
        aug = Xa[5:10]
        assert not np.allclose(orig, aug)

    def test_concentrations_preserved(self):
        """Concentration targets must not change under augmentation."""
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=4)
        Xa, ya, ca = augment_spectra(X, y, c, n_augment=3)
        # Each block of 4 rows should have the same concentrations
        np.testing.assert_array_almost_equal(ca[:4], c)
        np.testing.assert_array_almost_equal(ca[4:8], c)

    def test_zero_augment_returns_originals_only(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data(n=5)
        Xa, ya, ca = augment_spectra(X, y, c, n_augment=0)
        assert Xa.shape[0] == 5
        np.testing.assert_array_equal(ya, y)

    def test_output_dtype_float32(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data()
        Xa, _, _ = augment_spectra(X, y, c)
        assert Xa.dtype == np.float32

    def test_reproducible_with_same_seed(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data()
        Xa1, _, _ = augment_spectra(X, y, c, random_state=7)
        Xa2, _, _ = augment_spectra(X, y, c, random_state=7)
        np.testing.assert_array_equal(Xa1, Xa2)

    def test_different_seeds_give_different_results(self):
        from src.models.cnn import augment_spectra

        X, y, c = self._make_data()
        Xa1, _, _ = augment_spectra(X, y, c, random_state=1)
        Xa2, _, _ = augment_spectra(X, y, c, random_state=2)
        assert not np.allclose(Xa1[10:], Xa2[10:])  # augmented parts differ
