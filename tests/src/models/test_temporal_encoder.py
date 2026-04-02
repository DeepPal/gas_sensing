"""Tests for src.models.temporal_encoder — physics-agnostic kinetic feature extraction."""
import numpy as np
import pytest
import torch
from src.models.temporal_encoder import (
    TemporalConfig,
    TemporalEncoder,
    train_temporal_encoder,
)


# ---------------------------------------------------------------------------
# TemporalConfig
# ---------------------------------------------------------------------------

class TestTemporalConfig:
    def test_defaults(self):
        cfg = TemporalConfig()
        assert cfg.backbone == "gru"
        assert cfg.predict_concentration is True

    def test_invalid_backbone_raises(self):
        cfg = TemporalConfig(backbone="lstm")  # type: ignore
        with pytest.raises(ValueError, match="backbone"):
            TemporalEncoder(cfg)


# ---------------------------------------------------------------------------
# Shape contract — all three backbones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", ["cnn", "gru", "transformer"])
class TestTemporalEncoderShapes:
    @pytest.fixture
    def model(self, backbone):
        cfg = TemporalConfig(input_dim=16, hidden_dim=32, output_dim=16,
                             n_layers=1, n_heads=4, backbone=backbone,
                             predict_concentration=True, predict_analyte=False)
        return TemporalEncoder(cfg)

    def test_forward_concentration_shape(self, model):
        x = torch.randn(4, 20, 16)  # (batch=4, time=20, dim=16)
        conc, feats = model(x)
        assert conc is not None
        assert conc.shape == (4, 1)
        assert feats.shape == (4, 16)

    def test_concentration_non_negative(self, model):
        x = torch.randn(4, 20, 16)
        conc, _ = model(x)
        assert (conc >= 0).all(), "Softplus ensures non-negative concentrations"

    def test_features_finite(self, model):
        x = torch.randn(4, 20, 16)
        _, feats = model(x)
        assert torch.isfinite(feats).all()

    def test_variable_sequence_length(self, model, backbone):
        if backbone == "cnn":
            pytest.skip("CNN backbone uses AdaptiveAvgPool — fine with any T")
        for T in [5, 20, 100]:
            x = torch.randn(2, T, 16)
            conc, feats = model(x)
            assert feats.shape == (2, 16)


# ---------------------------------------------------------------------------
# Analyte classification head
# ---------------------------------------------------------------------------

class TestAnalyteClassification:
    @pytest.fixture
    def model(self):
        cfg = TemporalConfig(input_dim=16, hidden_dim=32, output_dim=16,
                             backbone="gru", predict_concentration=False,
                             predict_analyte=True, n_analytes=3)
        return TemporalEncoder(cfg)

    def test_analyte_logits_shape(self, model):
        x = torch.randn(5, 10, 16)
        logits = model.predict_analyte_logits(x)
        assert logits.shape == (5, 3)

    def test_no_concentration_head(self, model):
        x = torch.randn(5, 10, 16)
        conc, feats = model(x)
        assert conc is None
        assert feats.shape == (5, 16)

    def test_predict_analyte_without_head_raises(self):
        cfg = TemporalConfig(input_dim=8, hidden_dim=16, output_dim=8,
                             predict_analyte=False)
        model = TemporalEncoder(cfg)
        x = torch.randn(2, 5, 8)
        with pytest.raises(RuntimeError, match="predict_analyte=False"):
            model.predict_analyte_logits(x)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class TestLoss:
    @pytest.fixture
    def model(self):
        cfg = TemporalConfig(input_dim=16, hidden_dim=32, output_dim=16,
                             backbone="gru", predict_concentration=True,
                             predict_analyte=True, n_analytes=2)
        return TemporalEncoder(cfg)

    def test_loss_with_concentration(self, model):
        x = torch.randn(4, 10, 16)
        concs = torch.rand(4, 1) * 10
        loss = model.loss(x, concentrations=concs)
        assert loss.item() >= 0

    def test_loss_with_both_targets(self, model):
        x = torch.randn(4, 10, 16)
        concs = torch.rand(4, 1) * 10
        labels = torch.randint(0, 2, (4,))
        loss = model.loss(x, concentrations=concs, analyte_labels=labels)
        assert loss.item() >= 0

    def test_loss_no_targets_zero(self, model):
        x = torch.randn(4, 10, 16)
        loss = model.loss(x)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# High-dimensional input (raw spectra)
# ---------------------------------------------------------------------------

class TestHighDimInput:
    def test_raw_spectra_input(self):
        """Input projection reduces 3648-dim raw spectra to 256 before backbone."""
        cfg = TemporalConfig(input_dim=3648, hidden_dim=64, output_dim=32,
                             backbone="gru", n_layers=1)
        model = TemporalEncoder(cfg)
        x = torch.randn(2, 10, 3648)
        conc, feats = model(x)
        assert conc.shape == (2, 1)
        assert feats.shape == (2, 32)


# ---------------------------------------------------------------------------
# Numpy interface
# ---------------------------------------------------------------------------

class TestNumpyInterface:
    def test_encode_numpy_shape(self):
        cfg = TemporalConfig(input_dim=16, hidden_dim=32, output_dim=8,
                             backbone="gru")
        model = TemporalEncoder(cfg)
        seqs = np.random.default_rng(0).normal(size=(10, 20, 16)).astype(np.float32)
        feats = model.encode_numpy(seqs)
        assert feats.shape == (10, 8)
        assert isinstance(feats, np.ndarray)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TestTrainTemporalEncoder:
    def test_loss_decreases(self):
        cfg = TemporalConfig(input_dim=8, hidden_dim=16, output_dim=8,
                             backbone="gru", predict_concentration=True)
        model = TemporalEncoder(cfg)
        seqs = np.random.default_rng(0).normal(size=(20, 10, 8)).astype(np.float32)
        concs = (np.arange(20, dtype=np.float32) * 0.5)
        hist = train_temporal_encoder(model, seqs, concentrations=concs,
                                      n_epochs=20, batch_size=8, verbose=False)
        assert hist["train_loss"][-1] < hist["train_loss"][0]

    def test_history_structure(self):
        cfg = TemporalConfig(input_dim=4, hidden_dim=8, output_dim=4, backbone="cnn")
        model = TemporalEncoder(cfg)
        seqs = np.random.default_rng(1).normal(size=(10, 5, 4)).astype(np.float32)
        hist = train_temporal_encoder(model, seqs, n_epochs=3, verbose=False)
        assert len(hist["train_loss"]) == 3
        assert "val_loss" in hist
