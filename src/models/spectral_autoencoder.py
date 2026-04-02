"""Physics-agnostic spectral autoencoder.

Learns a compact latent representation from ANY 1-D spectral dataset using
a purely data-driven 1-D convolutional encoder-decoder, with NO assumptions
about peak shape, sensor material, or adsorption physics.

Architecture
------------
::

    Input  (B, 1, N_wl)
      ↓  Encoder: Conv1d blocks with stride-2 downsampling
    Latent (B, latent_dim)          ← discriminative features
      ↓  Decoder: ConvTranspose1d blocks with stride-2 upsampling
    Output (B, 1, N_wl)             ← reconstructed spectrum

The encoder bottleneck is the learned feature vector used for:
- Downstream concentration regression / analyte classification
- Visualisation via UMAP / t-SNE
- Contrastive analyte fingerprinting
- Cross-configuration transfer learning

Optional VAE mode adds KL-divergence regularisation (``vae=True``), which
forces the latent space to be smooth and better-structured for interpolation
and few-shot transfer.

Usage
-----
::

    import torch
    from src.models.spectral_autoencoder import SpectralAutoencoder, AutoencoderConfig

    cfg = AutoencoderConfig(input_length=3648, latent_dim=64)
    model = SpectralAutoencoder(cfg)

    # Encode a batch of spectra
    x = torch.randn(32, 1, 3648)    # (batch, channels, wavelengths)
    z = model.encode(x)             # (32, 64)  ← latent vectors
    x_hat = model.decode(z)         # (32, 1, 3648)  ← reconstructions
    loss = model.loss(x)            # scalar

    # Extract features for downstream ML
    features = model.encode(x).detach().numpy()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "AutoencoderConfig",
    "SpectralAutoencoder",
    "train_autoencoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AutoencoderConfig:
    """Hyperparameters for ``SpectralAutoencoder``.

    Parameters
    ----------
    input_length :
        Number of wavelength points (pixels).  CCS200 = 3648.  Any value ≥ 64.
    latent_dim :
        Size of the bottleneck vector.  Larger → more expressive, slower.
        Recommended: 32–128 for typical spectral datasets.
    channels :
        Number of feature maps in each encoder stage.
    kernel_sizes :
        Kernel size per encoder stage.  Larger kernels capture broader spectral
        features (peaks, shoulders).  Must be odd.
    strides :
        Downsampling stride per stage.  Product of strides determines total
        compression ratio.
    dropout :
        Dropout probability applied in the bottleneck (regularisation).
    vae :
        If True, use Variational Autoencoder with KL-divergence loss.
        Produces smoother, more structured latent space.
    kl_weight :
        Weight for KL divergence term when ``vae=True``.
    activation :
        Non-linearity in encoder/decoder.  ``'relu'`` or ``'leaky_relu'``.
    """
    input_length: int = 3648
    latent_dim: int = 64
    channels: list[int] = field(default_factory=lambda: [32, 64, 128, 64])
    kernel_sizes: list[int] = field(default_factory=lambda: [15, 11, 7, 5])
    strides: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.1
    vae: bool = False
    kl_weight: float = 1e-3
    activation: Literal["relu", "leaky_relu"] = "leaky_relu"

    def __post_init__(self) -> None:
        n = len(self.channels)
        if len(self.kernel_sizes) != n or len(self.strides) != n:
            raise ValueError(
                "channels, kernel_sizes, and strides must have the same length."
            )
        for k in self.kernel_sizes:
            if k % 2 == 0:
                raise ValueError(f"kernel_sizes must be odd, got {k}.")

    @property
    def compression_ratio(self) -> float:
        return math.prod(self.strides)

    @property
    def encoded_length(self) -> int:
        """Length of the spatial dimension after all encoder strides."""
        length = self.input_length
        for s in self.strides:
            length = math.ceil(length / s)
        return length


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _ConvBlock(nn.Module):
    """Conv1d → BatchNorm → Activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        activation: str,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1) if activation == "leaky_relu" \
            else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class _DeconvBlock(nn.Module):
    """ConvTranspose1d → BatchNorm → Activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        activation: str,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride,
            padding=padding, output_padding=output_padding,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.LeakyReLU(0.1) if activation == "leaky_relu" \
            else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SpectralAutoencoder(nn.Module):
    """Physics-agnostic 1-D convolutional autoencoder for spectral data.

    Parameters
    ----------
    config :
        ``AutoencoderConfig`` instance.  Defaults give a good starting point
        for 3648-point spectra.
    """

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig()
        cfg = self.config

        act = cfg.activation

        # ── Encoder ──────────────────────────────────────────────────────
        enc_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, s in zip(cfg.channels, cfg.kernel_sizes, cfg.strides):
            enc_layers.append(_ConvBlock(in_ch, out_ch, k, s, act))
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        flat_size = cfg.channels[-1] * cfg.encoded_length

        if cfg.vae:
            self.fc_mu = nn.Linear(flat_size, cfg.latent_dim)
            self.fc_logvar = nn.Linear(flat_size, cfg.latent_dim)
        else:
            self.fc_enc = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(flat_size, cfg.latent_dim),
            )

        # ── Decoder ──────────────────────────────────────────────────────
        self.fc_dec = nn.Linear(cfg.latent_dim, flat_size)

        dec_channels = [cfg.channels[-1]] + list(reversed(cfg.channels[:-1])) + [1]
        dec_kernels = list(reversed(cfg.kernel_sizes))
        dec_strides = list(reversed(cfg.strides))

        dec_layers: list[nn.Module] = []
        for i, (out_ch, k, s) in enumerate(
            zip(dec_channels[1:], dec_kernels, dec_strides)
        ):
            is_last = i == len(dec_strides) - 1
            dec_layers.append(
                _DeconvBlock(dec_channels[i], out_ch, k, s, act,
                             output_padding=s - 1) if not is_last
                else nn.ConvTranspose1d(dec_channels[i], out_ch, k,
                                       stride=s, padding=k // 2,
                                       output_padding=s - 1)
            )
        self.decoder_conv = nn.Sequential(*dec_layers)

        self._flat_size = flat_size
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectra to latent vectors.

        Parameters
        ----------
        x : Tensor, shape (B, 1, N_wl) or (B, N_wl)
            Input spectra.  If 2-D, a channel dimension is added automatically.

        Returns
        -------
        z : Tensor, shape (B, latent_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x)
        h = h.flatten(1)
        if self.config.vae:
            return self.fc_mu(h)  # return mean for inference
        return self.fc_enc(h)

    def encode_vae(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode returning (mu, logvar) for VAE training."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.encoder_conv(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to spectra.

        Parameters
        ----------
        z : Tensor, shape (B, latent_dim)

        Returns
        -------
        x_hat : Tensor, shape (B, 1, N_wl)
        """
        h = self.fc_dec(z)
        h = h.view(-1, self.config.channels[-1], self.config.encoded_length)
        x_hat = self.decoder_conv(h)
        # Trim or pad to exact input_length
        target = self.config.input_length
        if x_hat.shape[-1] > target:
            x_hat = x_hat[..., :target]
        elif x_hat.shape[-1] < target:
            x_hat = F.pad(x_hat, (0, target - x_hat.shape[-1]))
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full autoencoder pass.

        Returns
        -------
        x_hat : reconstructed spectrum, shape (B, 1, N_wl)
        z     : latent vector, shape (B, latent_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.config.vae:
            mu, logvar = self.encode_vae(x)
            z = self.reparameterise(mu, logvar)
            x_hat = self.decode(z)
            return x_hat, mu
        else:
            z = self.encode(x)
            x_hat = self.decode(z)
            return x_hat, z

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        x: torch.Tensor,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> torch.Tensor:
        """Compute reconstruction loss (+ KL divergence if VAE).

        Parameters
        ----------
        x : Tensor, shape (B, 1, N_wl) or (B, N_wl)
        reduction : 'mean' or 'sum'

        Returns
        -------
        total_loss : scalar Tensor
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        if self.config.vae:
            mu, logvar = self.encode_vae(x)
            z = self.reparameterise(mu, logvar)
            x_hat = self.decode(z)
            recon = F.mse_loss(x_hat, x, reduction=reduction)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + self.config.kl_weight * kl

        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction=reduction)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (anomaly score).

        Returns
        -------
        errors : Tensor, shape (B,) — MSE per sample
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        with torch.no_grad():
            x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))

    # ------------------------------------------------------------------
    # Convenience: numpy interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_numpy(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Encode a numpy array of spectra → numpy latent matrix.

        Parameters
        ----------
        spectra : ndarray, shape (N, N_wl)

        Returns
        -------
        z : ndarray, shape (N, latent_dim)
        """
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        return self.encode(x).cpu().numpy()

    @torch.no_grad()
    def reconstruct_numpy(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Reconstruct spectra from numpy input."""
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        x_hat, _ = self.forward(x)
        return x_hat.squeeze(1).cpu().numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_autoencoder(
    model: SpectralAutoencoder,
    spectra: "np.ndarray",  # type: ignore[name-defined]
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
    epoch_callback: "Callable[[int, int, float, float], None] | None" = None,  # type: ignore[name-defined]
) -> dict[str, list[float]]:
    """Train the autoencoder on a numpy spectra matrix.

    Parameters
    ----------
    model :
        ``SpectralAutoencoder`` instance.
    spectra :
        ndarray, shape (N, N_wl).  Will be normalised internally if not done.
    n_epochs :
        Number of full passes over the training set.
    batch_size :
        Mini-batch size.
    lr :
        Adam learning rate.
    val_fraction :
        Fraction held out for validation loss tracking.
    device :
        ``'cuda'``, ``'cpu'``, or ``None`` (auto-detect).
    verbose :
        Print epoch losses.

    Returns
    -------
    history : dict with keys ``'train_loss'`` and ``'val_loss'``
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Split train/val
    n = len(spectra)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    def _make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        X = torch.from_numpy(spectra[indices].astype(np.float32))
        return DataLoader(TensorDataset(X), batch_size=batch_size,
                          shuffle=shuffle)

    train_loader = _make_loader(train_idx, shuffle=True)
    val_loader = _make_loader(val_idx, shuffle=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, min_lr=1e-5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(train_idx)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                val_loss += model.loss(batch).item() * len(batch)
        val_loss /= n_val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs} — "
                  f"train={train_loss:.6f}  val={val_loss:.6f}")

        if epoch_callback is not None:
            epoch_callback(epoch, n_epochs, train_loss, val_loss)

    return history
