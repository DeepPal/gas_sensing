"""Tests for src.models.multi_task — multi-task spectral model."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.multi_task import (
    MultiTaskConfig,
    MultiTaskModel,
    MultiTaskOutput,
    MultiTaskTargets,
    train_multi_task,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    return MultiTaskConfig(
        input_dim=32,
        embed_dim=16,
        hidden_dim=32,
        n_layers=1,
        backbone="gru",
        n_analytes=3,
        predict_concentration=True,
        predict_qc=True,
    )


@pytest.fixture
def model(small_config):
    return MultiTaskModel(small_config)


@pytest.fixture
def batch():
    B = 8
    return torch.randn(B, 32)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_class_logits_shape(self, model, batch):
        out = model(batch)
        assert out.class_logits is not None
        assert out.class_logits.shape == (8, 3)

    def test_concentration_shape(self, model, batch):
        out = model(batch)
        assert out.concentration is not None
        assert out.concentration.shape == (8, 1)

    def test_qc_score_shape(self, model, batch):
        out = model(batch)
        assert out.qc_score is not None
        assert out.qc_score.shape == (8, 1)

    def test_features_shape(self, model, batch):
        out = model(batch)
        assert out.features.shape == (8, 16)

    def test_qc_in_zero_one(self, model, batch):
        out = model(batch)
        assert out.qc_score.min().item() >= 0.0 - 1e-6
        assert out.qc_score.max().item() <= 1.0 + 1e-6

    def test_concentration_nonneg(self, model, batch):
        out = model(batch)
        assert out.concentration.min().item() >= 0.0 - 1e-6


class TestDisabledHeads:
    def test_no_class_head(self, batch):
        cfg = MultiTaskConfig(input_dim=32, embed_dim=16, hidden_dim=32,
                              n_analytes=0, n_layers=1)
        m = MultiTaskModel(cfg)
        out = m(batch)
        assert out.class_logits is None

    def test_no_conc_head(self, batch):
        cfg = MultiTaskConfig(input_dim=32, embed_dim=16, hidden_dim=32,
                              predict_concentration=False, n_layers=1)
        m = MultiTaskModel(cfg)
        out = m(batch)
        assert out.concentration is None

    def test_no_qc_head(self, batch):
        cfg = MultiTaskConfig(input_dim=32, embed_dim=16, hidden_dim=32,
                              predict_qc=False, n_layers=1)
        m = MultiTaskModel(cfg)
        out = m(batch)
        assert out.qc_score is None


# ---------------------------------------------------------------------------
# Backbone variants
# ---------------------------------------------------------------------------

class TestBackbones:
    @pytest.mark.parametrize("backbone", ["cnn", "gru", "transformer"])
    def test_backbone_runs(self, backbone):
        cfg = MultiTaskConfig(input_dim=64, embed_dim=16, hidden_dim=32,
                              n_layers=1, backbone=backbone, n_analytes=2)
        m = MultiTaskModel(cfg)
        x = torch.randn(4, 64)
        out = m(x)
        assert out.class_logits.shape == (4, 2)


# ---------------------------------------------------------------------------
# High-dim input projection
# ---------------------------------------------------------------------------

class TestHighDimInput:
    def test_input_dim_3648(self):
        cfg = MultiTaskConfig(input_dim=3648, embed_dim=32, hidden_dim=64,
                              n_layers=1, n_analytes=4)
        m = MultiTaskModel(cfg)
        x = torch.randn(2, 3648)
        out = m(x)
        assert out.class_logits.shape == (2, 4)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def test_loss_with_all_targets(self, model, batch):
        targets = MultiTaskTargets(
            analyte_labels=torch.randint(0, 3, (8,)),
            concentrations=torch.rand(8, 1),
            qc_labels=torch.ones(8, 1),
        )
        loss = model.loss(batch, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_loss_with_no_targets(self, model, batch):
        targets = MultiTaskTargets()
        loss = model.loss(batch, targets)
        assert loss.item() == 0.0
        assert loss.requires_grad

    def test_loss_classification_only(self, model, batch):
        targets = MultiTaskTargets(
            analyte_labels=torch.randint(0, 3, (8,)))
        loss = model.loss(batch, targets)
        assert loss.requires_grad
        assert not torch.isnan(loss)

    def test_backward_works(self, model, batch):
        targets = MultiTaskTargets(
            analyte_labels=torch.randint(0, 3, (8,)),
            concentrations=torch.rand(8, 1),
        )
        loss = model.loss(batch, targets)
        loss.backward()  # Should not raise
        # Check gradients propagated
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Numpy interface
# ---------------------------------------------------------------------------

class TestNumpyInterface:
    def test_embed_numpy(self, model):
        X = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
        feats = model.embed_numpy(X)
        assert feats.shape == (10, 16)
        assert isinstance(feats, np.ndarray)

    def test_predict_analyte(self, model):
        X = np.random.default_rng(1).normal(size=(5, 32)).astype(np.float32)
        preds = model.predict_analyte(X)
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_concentration(self, model):
        X = np.random.default_rng(2).normal(size=(5, 32)).astype(np.float32)
        concs = model.predict_concentration(X)
        assert concs.shape == (5, 1)
        assert (concs >= 0).all()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TestTrainMultiTask:
    def test_history_structure(self):
        cfg = MultiTaskConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                              n_layers=1, n_analytes=2)
        m = MultiTaskModel(cfg)
        X = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
        labels = (np.arange(20) % 2).astype(np.int64)
        hist = train_multi_task(m, X, analyte_labels=labels,
                                n_epochs=3, verbose=False)
        assert len(hist["train_loss"]) == 3
        assert "val_loss" in hist

    def test_loss_decreases_with_supervision(self):
        cfg = MultiTaskConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                              n_layers=1, n_analytes=2, predict_concentration=True)
        m = MultiTaskModel(cfg)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(40, 8)).astype(np.float32)
        labels = (np.arange(40) % 2).astype(np.int64)
        concs = np.abs(rng.normal(size=40)).astype(np.float32)
        hist = train_multi_task(m, X, analyte_labels=labels,
                                concentrations=concs,
                                n_epochs=30, verbose=False)
        assert hist["train_loss"][-1] < hist["train_loss"][0]
