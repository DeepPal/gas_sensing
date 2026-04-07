"""Tests for src.models.contrastive — analyte fingerprinting."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.contrastive import (
    ContrastiveConfig,
    ContrastiveEncoder,
    _ntxent_loss,
    _supervised_contrastive_loss,
    _triplet_loss,
    build_gallery,
    identify_analyte,
    train_contrastive,
)


@pytest.fixture
def model():
    cfg = ContrastiveConfig(input_dim=32, embed_dim=16, hidden_dim=32,
                            n_layers=2, loss_type="supcon")
    return ContrastiveEncoder(cfg)


@pytest.fixture
def two_class_data():
    rng = np.random.default_rng(42)
    # Class 0: centred at 0; Class 1: centred at 1
    X0 = rng.normal(0.0, 0.1, (10, 32)).astype(np.float32)
    X1 = rng.normal(1.0, 0.1, (10, 32)).astype(np.float32)
    X = np.vstack([X0, X1])
    y = np.array([0] * 10 + [1] * 10)
    return X, y


# ---------------------------------------------------------------------------
# ContrastiveConfig
# ---------------------------------------------------------------------------

class TestContrastiveConfig:
    def test_defaults(self):
        cfg = ContrastiveConfig()
        assert cfg.embed_dim == 64
        assert cfg.normalise_embeddings is True
        assert cfg.loss_type == "supcon"


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------

class TestContrastiveEncoderShapes:
    def test_embed_shape(self, model):
        x = torch.randn(8, 32)
        z = model.embed(x)
        assert z.shape == (8, 16)

    def test_embeddings_unit_norm(self, model):
        x = torch.randn(8, 32)
        z = model.embed(x)
        norms = z.norm(dim=-1)
        np.testing.assert_allclose(norms.detach().numpy(), 1.0, atol=1e-5)

    def test_embed_numpy_shape(self, model):
        spectra = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
        z = model.embed_numpy(spectra)
        assert z.shape == (10, 16)
        assert isinstance(z, np.ndarray)

    def test_high_dim_input_projection(self):
        cfg = ContrastiveConfig(input_dim=3648, embed_dim=32, hidden_dim=64, n_layers=2)
        model = ContrastiveEncoder(cfg)
        x = torch.randn(4, 3648)
        z = model.embed(x)
        assert z.shape == (4, 32)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("loss_type", ["supcon", "triplet", "ntxent"])
class TestLossFunctions:
    def test_loss_positive(self, loss_type):
        cfg = ContrastiveConfig(input_dim=16, embed_dim=8, hidden_dim=16,
                                n_layers=2, loss_type=loss_type)
        model = ContrastiveEncoder(cfg)
        x = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
        loss = model.loss(x, labels)
        assert loss.item() >= 0

    def test_loss_gradients_flow(self, loss_type):
        cfg = ContrastiveConfig(input_dim=16, embed_dim=8, hidden_dim=16,
                                n_layers=2, loss_type=loss_type)
        model = ContrastiveEncoder(cfg)
        x = torch.randn(8, 16)
        labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
        loss = model.loss(x, labels)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).all()


class TestSupConLoss:
    def test_all_same_class_no_negatives(self):
        z = torch.randn(4, 8)
        z = torch.nn.functional.normalize(z, dim=-1)
        labels = torch.zeros(4, dtype=torch.long)
        # All same class — loss should still be defined
        loss = _supervised_contrastive_loss(z, labels, temperature=0.07)
        assert loss.item() >= 0

    def test_all_different_no_positives(self):
        z = torch.randn(4, 8)
        z = torch.nn.functional.normalize(z, dim=-1)
        labels = torch.arange(4)
        # No positives — should return 0
        loss = _supervised_contrastive_loss(z, labels, temperature=0.07)
        assert loss.item() == 0.0


class TestTripletLoss:
    def test_zero_when_well_separated(self):
        # Perfectly separated: class 0 at [1,0,...], class 1 at [-1,0,...]
        z = torch.zeros(4, 8)
        z[0, 0] = z[1, 0] = 1.0
        z[2, 0] = z[3, 0] = -1.0
        labels = torch.tensor([0, 0, 1, 1])
        loss = _triplet_loss(z, labels, margin=0.1)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Gallery and identification
# ---------------------------------------------------------------------------

class TestGalleryAndIdentification:
    def test_build_gallery_shape(self, model):
        spectra = np.random.default_rng(0).normal(size=(6, 32)).astype(np.float32)
        labels = ["Ethanol"] * 3 + ["IPA"] * 3
        gallery_emb, gallery_labels = build_gallery(model, spectra, labels)
        assert gallery_emb.shape == (6, 16)
        assert gallery_labels == labels

    def test_identify_returns_correct_length(self, model):
        ref = np.random.default_rng(0).normal(size=(6, 32)).astype(np.float32)
        ref_labels = ["A"] * 3 + ["B"] * 3
        gallery_emb, gallery_labels = build_gallery(model, ref, ref_labels)

        query = np.random.default_rng(1).normal(size=(4, 32)).astype(np.float32)
        preds = identify_analyte(model, query, gallery_emb, gallery_labels)
        assert len(preds) == 4
        assert all(p in {"A", "B"} for p in preds)

    def test_identify_top_k(self, model):
        ref = np.random.default_rng(0).normal(size=(6, 32)).astype(np.float32)
        ref_labels = ["A"] * 3 + ["B"] * 3
        gallery_emb, gallery_labels = build_gallery(model, ref, ref_labels)
        query = np.random.default_rng(1).normal(size=(2, 32)).astype(np.float32)
        preds = identify_analyte(model, query, gallery_emb, gallery_labels, top_k=2)
        assert len(preds) == 2
        assert len(preds[0]) == 2

    def test_trained_model_separates_analytes(self, two_class_data):
        """After training, same-class spectra should have higher cosine sim."""
        X, y = two_class_data
        cfg = ContrastiveConfig(input_dim=32, embed_dim=16, hidden_dim=32,
                                n_layers=2, loss_type="supcon")
        model = ContrastiveEncoder(cfg)
        train_contrastive(model, X, y, n_epochs=30, batch_size=8, verbose=False)

        z = model.embed_numpy(X)
        # Average intra-class cosine similarity vs inter-class
        z_norm = z / np.linalg.norm(z, axis=1, keepdims=True)
        intra = (z_norm[:10] @ z_norm[:10].T).mean()   # class 0
        inter = (z_norm[:10] @ z_norm[10:].T).mean()   # cross-class
        assert intra > inter, "Trained model should separate analyte classes"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrainContrastive:
    def test_loss_decreases(self, two_class_data):
        X, y = two_class_data
        cfg = ContrastiveConfig(input_dim=32, embed_dim=16, hidden_dim=32, n_layers=2)
        model = ContrastiveEncoder(cfg)
        hist = train_contrastive(model, X, y, n_epochs=20, batch_size=8, verbose=False)
        assert hist["train_loss"][-1] < hist["train_loss"][0]

    def test_history_keys(self, two_class_data):
        X, y = two_class_data
        cfg = ContrastiveConfig(input_dim=32, embed_dim=16, hidden_dim=32, n_layers=2)
        model = ContrastiveEncoder(cfg)
        hist = train_contrastive(model, X, y, n_epochs=5, verbose=False)
        assert "train_loss" in hist and "val_loss" in hist
