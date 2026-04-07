"""Tests for src.models.transfer — cross-config domain adaptation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.transfer import (
    DomainAdaptConfig,
    DomainAdaptModel,
    evaluate_transfer,
    fine_tune,
    grad_reverse,
    train_domain_adapt,
)

# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class TestGradReversal:
    def test_forward_is_identity(self):
        x = torch.randn(4, 8, requires_grad=True)
        y = grad_reverse(x, lam=1.0)
        assert torch.allclose(x, y)

    def test_backward_reverses_gradient(self):
        x = torch.ones(4, 8, requires_grad=True)
        y = grad_reverse(x, lam=1.0)
        loss = y.sum()
        loss.backward()
        # Gradient should be -1 (reversed from +1)
        assert (x.grad < 0).all()

    def test_lambda_scaling(self):
        x = torch.ones(4, 8, requires_grad=True)
        y = grad_reverse(x, lam=2.5)
        y.sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x.grad, -2.5))


# ---------------------------------------------------------------------------
# DomainAdaptModel shapes
# ---------------------------------------------------------------------------

@pytest.fixture
def small_adapt_cfg():
    return DomainAdaptConfig(
        input_dim=32, embed_dim=16, hidden_dim=32,
        n_analytes=3, predict_concentration=True, n_layers=1,
    )


@pytest.fixture
def adapt_model(small_adapt_cfg):
    return DomainAdaptModel(small_adapt_cfg)


class TestDomainAdaptShapes:
    def test_encode_shape(self, adapt_model):
        x = torch.randn(6, 32)
        feats = adapt_model.encode(x)
        assert feats.shape == (6, 16)

    def test_forward_output_keys(self, adapt_model):
        x = torch.randn(6, 32)
        out = adapt_model.forward(x, lam=1.0)
        assert "features" in out
        assert "class_logits" in out
        assert "concentration" in out
        assert "domain_logits" in out

    def test_class_logits_shape(self, adapt_model):
        x = torch.randn(6, 32)
        out = adapt_model.forward(x)
        assert out["class_logits"].shape == (6, 3)

    def test_domain_logits_shape(self, adapt_model):
        x = torch.randn(6, 32)
        out = adapt_model.forward(x)
        assert out["domain_logits"].shape == (6, 1)

    def test_concentration_nonneg(self, adapt_model):
        x = torch.randn(6, 32)
        out = adapt_model.forward(x)
        assert out["concentration"].min().item() >= 0.0 - 1e-6


# ---------------------------------------------------------------------------
# High-dim input
# ---------------------------------------------------------------------------

class TestHighDimInput:
    def test_raw_spectra_3648(self):
        cfg = DomainAdaptConfig(input_dim=3648, embed_dim=16, hidden_dim=32,
                                n_analytes=2, n_layers=1)
        m = DomainAdaptModel(cfg)
        x = torch.randn(2, 3648)
        feats = m.encode(x)
        assert feats.shape == (2, 16)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class TestLoss:
    def test_loss_returns_scalar(self, adapt_model):
        src_x = torch.randn(8, 32)
        loss, breakdown = adapt_model.loss(src_x)
        assert loss.ndim == 0
        assert "task" in breakdown
        assert "domain" in breakdown

    def test_loss_with_labels(self, adapt_model):
        src_x = torch.randn(8, 32)
        labels = torch.randint(0, 3, (8,))
        loss, _ = adapt_model.loss(src_x, src_labels=labels)
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_loss_with_target(self, adapt_model):
        src_x = torch.randn(8, 32)
        tgt_x = torch.randn(6, 32)
        labels = torch.randint(0, 3, (8,))
        loss, breakdown = adapt_model.loss(src_x, src_labels=labels, tgt_x=tgt_x)
        assert not torch.isnan(loss)
        assert breakdown["domain"] > 0

    def test_backward_works(self, adapt_model):
        src_x = torch.randn(8, 32)
        labels = torch.randint(0, 3, (8,))
        loss, _ = adapt_model.loss(src_x, src_labels=labels)
        loss.backward()
        has_grad = any(
            p.grad is not None for p in adapt_model.parameters())
        assert has_grad


# ---------------------------------------------------------------------------
# Numpy interface
# ---------------------------------------------------------------------------

class TestNumpyInterface:
    def test_embed_numpy(self, adapt_model):
        X = np.random.default_rng(0).normal(size=(8, 32)).astype(np.float32)
        emb = adapt_model.embed_numpy(X)
        assert emb.shape == (8, 16)
        assert isinstance(emb, np.ndarray)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

class TestTrainDomainAdapt:
    def test_history_keys(self):
        cfg = DomainAdaptConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                                n_analytes=2, n_layers=1, ramp_lambda=False)
        m = DomainAdaptModel(cfg)
        rng = np.random.default_rng(0)
        src = rng.normal(size=(20, 8)).astype(np.float32)
        tgt = rng.normal(size=(15, 8)).astype(np.float32)
        labels = (np.arange(20) % 2).astype(np.int64)
        hist = train_domain_adapt(m, src, source_labels=labels,
                                  target_spectra=tgt,
                                  n_epochs=3, verbose=False)
        assert "train_loss" in hist
        assert "task_loss" in hist
        assert "domain_loss" in hist
        assert len(hist["train_loss"]) == 3

    def test_no_target_runs(self):
        cfg = DomainAdaptConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                                n_analytes=2, n_layers=1)
        m = DomainAdaptModel(cfg)
        src = np.random.default_rng(1).normal(size=(16, 8)).astype(np.float32)
        labels = (np.arange(16) % 2).astype(np.int64)
        hist = train_domain_adapt(m, src, source_labels=labels,
                                  n_epochs=2, verbose=False)
        assert len(hist["train_loss"]) == 2


class TestFineTune:
    def test_fine_tune_runs(self):
        from src.models.multi_task import MultiTaskConfig, MultiTaskModel
        cfg = MultiTaskConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                              n_layers=1, n_analytes=2)
        m = MultiTaskModel(cfg)
        X = np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32)
        labels = (np.arange(20) % 2).astype(np.int64)
        hist = fine_tune(m, X, analyte_labels=labels, n_epochs=5, verbose=False)
        assert len(hist["train_loss"]) == 5

    def test_fine_tune_freezes_encoder(self):
        from src.models.multi_task import MultiTaskConfig, MultiTaskModel
        cfg = MultiTaskConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                              n_layers=1, n_analytes=2)
        m = MultiTaskModel(cfg)
        # Record encoder weights before fine-tuning
        enc_before = {n: p.clone() for n, p in m.named_parameters()
                      if "class_head" not in n and "conc_head" not in n
                      and "qc_head" not in n}
        X = np.random.default_rng(2).normal(size=(16, 8)).astype(np.float32)
        labels = (np.arange(16) % 2).astype(np.int64)
        fine_tune(m, X, analyte_labels=labels, n_epochs=10,
                  freeze_encoder=True, verbose=False)
        for name, before in enc_before.items():
            after = dict(m.named_parameters())[name]
            assert torch.allclose(before, after), f"Encoder param {name} changed!"


# ---------------------------------------------------------------------------
# Evaluate transfer
# ---------------------------------------------------------------------------

class TestEvaluateTransfer:
    def test_accuracy_in_range(self):
        cfg = DomainAdaptConfig(input_dim=8, embed_dim=8, hidden_dim=16,
                                n_analytes=2, n_layers=1)
        m = DomainAdaptModel(cfg)
        X = np.random.default_rng(3).normal(size=(20, 8)).astype(np.float32)
        y = (np.arange(20) % 2).astype(np.int64)
        metrics = evaluate_transfer(m, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] == 20
