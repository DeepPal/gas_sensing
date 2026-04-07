"""Tests for src.models.spectral_autoencoder — physics-agnostic feature learning."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.spectral_autoencoder import (
    AutoencoderConfig,
    SpectralAutoencoder,
    train_autoencoder,
)

# ---------------------------------------------------------------------------
# AutoencoderConfig
# ---------------------------------------------------------------------------

class TestAutoencoderConfig:
    def test_default_config(self):
        cfg = AutoencoderConfig()
        assert cfg.input_length == 3648
        assert cfg.latent_dim == 64
        assert len(cfg.channels) == len(cfg.kernel_sizes) == len(cfg.strides)

    def test_compression_ratio(self):
        cfg = AutoencoderConfig(strides=[2, 2, 2, 2])
        assert cfg.compression_ratio == 16

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            AutoencoderConfig(channels=[32, 64], kernel_sizes=[15], strides=[2, 2])

    def test_even_kernel_raises(self):
        with pytest.raises(ValueError, match="odd"):
            AutoencoderConfig(kernel_sizes=[14, 11, 7, 5])

    def test_encoded_length_positive(self):
        cfg = AutoencoderConfig(input_length=3648)
        assert cfg.encoded_length > 0


# ---------------------------------------------------------------------------
# SpectralAutoencoder — shape contract
# ---------------------------------------------------------------------------

class TestSpectralAutoencoderShapes:
    @pytest.fixture
    def small_model(self):
        cfg = AutoencoderConfig(
            input_length=256,
            latent_dim=16,
            channels=[8, 16],
            kernel_sizes=[7, 5],
            strides=[2, 2],
        )
        return SpectralAutoencoder(cfg)

    def test_encode_2d_input(self, small_model):
        x = torch.randn(4, 256)
        z = small_model.encode(x)
        assert z.shape == (4, 16)

    def test_encode_3d_input(self, small_model):
        x = torch.randn(4, 1, 256)
        z = small_model.encode(x)
        assert z.shape == (4, 16)

    def test_decode_output_shape(self, small_model):
        z = torch.randn(4, 16)
        x_hat = small_model.decode(z)
        assert x_hat.shape == (4, 1, 256)

    def test_forward_shapes(self, small_model):
        x = torch.randn(4, 256)
        x_hat, z = small_model(x)
        assert x_hat.shape == (4, 1, 256)
        assert z.shape == (4, 16)

    def test_reconstruction_error_per_sample(self, small_model):
        x = torch.randn(5, 256)
        errs = small_model.reconstruction_error(x)
        assert errs.shape == (5,)
        assert (errs >= 0).all()

    def test_loss_scalar(self, small_model):
        x = torch.randn(4, 256)
        loss = small_model.loss(x)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# VAE variant
# ---------------------------------------------------------------------------

class TestVAE:
    @pytest.fixture
    def vae_model(self):
        cfg = AutoencoderConfig(input_length=128, latent_dim=8,
                                channels=[8, 16], kernel_sizes=[7, 5],
                                strides=[2, 2], vae=True)
        return SpectralAutoencoder(cfg)

    def test_encode_returns_mean(self, vae_model):
        x = torch.randn(3, 128)
        z = vae_model.encode(x)
        assert z.shape == (3, 8)

    def test_encode_vae_returns_mu_logvar(self, vae_model):
        x = torch.randn(3, 128)
        mu, logvar = vae_model.encode_vae(x)
        assert mu.shape == (3, 8)
        assert logvar.shape == (3, 8)

    def test_vae_loss_includes_kl(self, vae_model):
        x = torch.randn(3, 128)
        loss = vae_model.loss(x)
        assert loss.item() > 0

    def test_reparameterise_deterministic_at_eval(self, vae_model):
        vae_model.eval()
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)
        z = vae_model.reparameterise(mu, logvar)
        np.testing.assert_allclose(z.detach().numpy(), mu.numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Numpy interface
# ---------------------------------------------------------------------------

class TestNumpyInterface:
    @pytest.fixture
    def model_and_spectra(self):
        cfg = AutoencoderConfig(input_length=200, latent_dim=12,
                                channels=[8, 16], kernel_sizes=[7, 5],
                                strides=[2, 2])
        model = SpectralAutoencoder(cfg)
        spectra = np.random.default_rng(0).normal(size=(10, 200)).astype(np.float32)
        return model, spectra

    def test_encode_numpy_shape(self, model_and_spectra):
        model, spectra = model_and_spectra
        z = model.encode_numpy(spectra)
        assert z.shape == (10, 12)
        assert isinstance(z, np.ndarray)

    def test_reconstruct_numpy_shape(self, model_and_spectra):
        model, spectra = model_and_spectra
        recon = model.reconstruct_numpy(spectra)
        assert recon.shape == (10, 200)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class TestTrainAutoencoder:
    def test_loss_decreases(self):
        cfg = AutoencoderConfig(input_length=128, latent_dim=8,
                                channels=[8, 16], kernel_sizes=[7, 5],
                                strides=[2, 2])
        model = SpectralAutoencoder(cfg)
        spectra = np.random.default_rng(42).normal(0.5, 0.1, (30, 128)).astype(np.float32)
        hist = train_autoencoder(model, spectra, n_epochs=20, batch_size=8,
                                 lr=1e-3, verbose=False)
        assert hist["train_loss"][-1] < hist["train_loss"][0]

    def test_history_keys(self):
        cfg = AutoencoderConfig(input_length=64, latent_dim=4,
                                channels=[8], kernel_sizes=[7], strides=[2])
        model = SpectralAutoencoder(cfg)
        spectra = np.random.default_rng(0).normal(size=(15, 64)).astype(np.float32)
        hist = train_autoencoder(model, spectra, n_epochs=5, verbose=False)
        assert "train_loss" in hist
        assert "val_loss" in hist
        assert len(hist["train_loss"]) == 5


# ---------------------------------------------------------------------------
# Different input lengths
# ---------------------------------------------------------------------------

class TestVariableInputLength:
    @pytest.mark.parametrize("n_wl", [128, 512, 1024, 3648])
    def test_input_length(self, n_wl):
        cfg = AutoencoderConfig(input_length=n_wl, latent_dim=16,
                                channels=[8, 16], kernel_sizes=[7, 5],
                                strides=[2, 2])
        model = SpectralAutoencoder(cfg)
        x = torch.randn(2, n_wl)
        x_hat, z = model(x)
        assert x_hat.shape[-1] == n_wl
        assert z.shape == (2, 16)
