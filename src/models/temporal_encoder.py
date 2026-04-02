"""Temporal sequence encoder for kinetic pattern extraction.

Learns response dynamics (fast/slow binding, transient shape, recovery
rate) from a TIME SERIES of spectral frames — without fitting any explicit
physical model (no Langmuir equation, no tau formula).

The encoder treats the sequence of spectra as a multivariate time series and
learns what temporal patterns correlate with analyte identity and
concentration.  This captures physics-agnostic kinetic fingerprints:
analytes with identical equilibrium spectra but different response speeds are
perfectly discriminable because the temporal trajectory is different.

Architecture options
--------------------
Three backbone options are provided, selectable via ``TemporalConfig.backbone``:

``'cnn'`` (default, fastest)
    1-D convolutions along the time axis over spectral feature vectors.
    Best for short sequences (≤ 200 frames).

``'gru'``
    Gated Recurrent Unit.  Handles longer sequences and variable-length
    inputs naturally.  Better for noisy/irregular time series.

``'transformer'``
    Self-attention encoder.  Best for long sequences with complex temporal
    dependencies.  Requires more data.

Usage
-----
::

    import torch
    from src.models.temporal_encoder import TemporalEncoder, TemporalConfig
    from src.models.spectral_autoencoder import SpectralAutoencoder

    # Option 1: encode raw spectra sequences directly
    cfg = TemporalConfig(input_dim=3648, hidden_dim=128, output_dim=64)
    model = TemporalEncoder(cfg)

    seq = torch.randn(8, 60, 3648)   # (batch, time_steps, wavelengths)
    conc_pred, features = model(seq)  # (8, 1), (8, 64)

    # Option 2: encode pre-extracted latent vectors (preferred — faster)
    cfg2 = TemporalConfig(input_dim=64, hidden_dim=128, output_dim=64)
    model2 = TemporalEncoder(cfg2)

    latents = torch.randn(8, 60, 64)  # (batch, time, latent_dim)
    conc_pred, features = model2(latents)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "TemporalConfig",
    "TemporalEncoder",
    "train_temporal_encoder",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Hyperparameters for ``TemporalEncoder``.

    Parameters
    ----------
    input_dim :
        Dimensionality of each frame's input vector.
        If raw spectra: N_wavelengths (e.g. 3648).
        If pre-encoded latents: latent_dim (e.g. 64).
    hidden_dim :
        Internal hidden size for the backbone.
    output_dim :
        Dimensionality of the output temporal feature vector.
    n_layers :
        Number of GRU layers / Transformer layers.
    n_heads :
        Attention heads (Transformer only).
    dropout :
        Dropout probability.
    backbone :
        Which sequence model to use: ``'cnn'``, ``'gru'``, ``'transformer'``.
    predict_concentration :
        If True, add a linear regression head for concentration prediction.
    predict_analyte :
        If True, add a classification head for analyte identification.
    n_analytes :
        Number of analyte classes for the classification head.
    """
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    backbone: Literal["cnn", "gru", "transformer"] = "gru"
    predict_concentration: bool = True
    predict_analyte: bool = False
    n_analytes: int = 2


# ---------------------------------------------------------------------------
# Backbone modules
# ---------------------------------------------------------------------------

class _CNNBackbone(nn.Module):
    """1-D temporal CNN: Conv over time axis."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for i in range(n_layers):
            out_ch = hidden_dim
            kernel = min(7, 3 + 2 * i)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → (B, D, T) for Conv1d
        h = self.net(x.permute(0, 2, 1))   # (B, hidden, T)
        return self.pool(h).squeeze(-1)     # (B, hidden)


class _GRUBackbone(nn.Module):
    """Bidirectional GRU: captures both past and future temporal context."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        _, h = self.gru(x)           # h: (2*n_layers, B, hidden//2)
        # Concatenate last forward + backward hidden states
        return torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden)


class _TransformerBackbone(nn.Module):
    """Transformer encoder with positional encoding."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 n_heads: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.proj(x)  # (B, T, hidden)
        # Prepend CLS token
        cls = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, hidden)
        h = torch.cat([cls, h], dim=1)                   # (B, T+1, hidden)
        h = self.encoder(h)                               # (B, T+1, hidden)
        return h[:, 0]                                    # (B, hidden) CLS output


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """Physics-agnostic temporal encoder for spectral time series.

    Parameters
    ----------
    config :
        ``TemporalConfig`` instance.
    """

    def __init__(self, config: TemporalConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalConfig()
        cfg = self.config

        # Optional input projection (reduces dim if input is raw spectra)
        if cfg.input_dim > 256:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(cfg.input_dim, 256),
                nn.GELU(),
            )
            backbone_in = 256
        else:
            self.input_proj = nn.Identity()
            backbone_in = cfg.input_dim

        # Backbone
        if cfg.backbone == "cnn":
            self.backbone: nn.Module = _CNNBackbone(
                backbone_in, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        elif cfg.backbone == "gru":
            self.backbone = _GRUBackbone(
                backbone_in, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        elif cfg.backbone == "transformer":
            self.backbone = _TransformerBackbone(
                backbone_in, cfg.hidden_dim, cfg.n_layers, cfg.n_heads,
                cfg.dropout)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone!r}. "
                             "Choose 'cnn', 'gru', or 'transformer'.")

        # Temporal feature projection
        self.feature_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
            nn.LayerNorm(cfg.output_dim),
        )

        # Prediction heads
        if cfg.predict_concentration:
            self.conc_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.output_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softplus(),  # concentration is always ≥ 0
            )
        else:
            self.conc_head = None

        if cfg.predict_analyte:
            self.analyte_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.output_dim, 32),
                nn.GELU(),
                nn.Linear(32, cfg.n_analytes),
            )
        else:
            self.analyte_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Encode a spectral time series.

        Parameters
        ----------
        x : Tensor, shape (B, T, D)
            Batch of spectral sequences.  D = input_dim.

        Returns
        -------
        concentration : Tensor or None, shape (B, 1)
            Predicted concentration (only if ``predict_concentration=True``).
        features : Tensor, shape (B, output_dim)
            Temporal feature vector (temporal fingerprint of the response).
        """
        h = self.input_proj(x)           # (B, T, backbone_in)
        h = self.backbone(h)             # (B, hidden_dim)
        features = self.feature_head(h)  # (B, output_dim)

        conc = self.conc_head(features) if self.conc_head is not None else None
        return conc, features

    def predict_analyte_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw analyte classification logits."""
        _, features = self.forward(x)
        if self.analyte_head is None:
            raise RuntimeError("predict_analyte=False in config.")
        return self.analyte_head(features)

    def loss(
        self,
        x: torch.Tensor,
        concentrations: torch.Tensor | None = None,
        analyte_labels: torch.Tensor | None = None,
        conc_weight: float = 1.0,
        analyte_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute combined prediction loss.

        Parameters
        ----------
        x : (B, T, D) input sequence
        concentrations : (B, 1) ground truth concentrations
        analyte_labels : (B,) integer class labels
        conc_weight : weight for regression loss
        analyte_weight : weight for classification loss
        """
        conc_pred, features = self.forward(x)
        # Initialise as graph-connected zero so backward() works even with no supervision
        total: torch.Tensor = features.mean() * 0.0

        if conc_pred is not None and concentrations is not None:
            total = total + conc_weight * F.mse_loss(conc_pred, concentrations)

        if self.analyte_head is not None and analyte_labels is not None:
            logits = self.analyte_head(features)
            total = total + analyte_weight * F.cross_entropy(logits, analyte_labels)

        return total

    @torch.no_grad()
    def encode_numpy(
        self, sequences: "np.ndarray"  # type: ignore[name-defined]
    ) -> "np.ndarray":  # type: ignore[name-defined]
        """Encode numpy sequences → numpy temporal feature matrix.

        Parameters
        ----------
        sequences : ndarray, shape (N, T, D)

        Returns
        -------
        features : ndarray, shape (N, output_dim)
        """
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(sequences.astype(np.float32)).to(device)
        _, features = self.forward(x)
        return features.cpu().numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_temporal_encoder(
    model: TemporalEncoder,
    sequences: "np.ndarray",        # (N, T, D)  # type: ignore[name-defined]
    concentrations: "np.ndarray | None" = None,   # (N,) # type: ignore[name-defined]
    analyte_labels: "np.ndarray | None" = None,   # (N,) # type: ignore[name-defined]
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the temporal encoder.

    Parameters
    ----------
    model :
        ``TemporalEncoder`` instance.
    sequences :
        ndarray, shape (N, T, D).  T = number of frames per session.
    concentrations :
        Optional target concentrations, shape (N,).
    analyte_labels :
        Optional integer class labels, shape (N,).
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n = len(sequences)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    tensors: list[torch.Tensor] = [
        torch.from_numpy(sequences.astype(np.float32))
    ]
    if concentrations is not None:
        tensors.append(
            torch.from_numpy(concentrations.astype(np.float32)).unsqueeze(1)
        )
    if analyte_labels is not None:
        tensors.append(torch.from_numpy(analyte_labels.astype(np.int64)))

    def _loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        subset = [t[indices] for t in tensors]
        return DataLoader(TensorDataset(*subset), batch_size=batch_size,
                          shuffle=shuffle)

    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, min_lr=1e-5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _step(batch: tuple) -> torch.Tensor:
        x = batch[0].to(device)
        conc = batch[1].to(device) if concentrations is not None else None
        labels = batch[-1].to(device) if analyte_labels is not None else None
        if concentrations is not None and analyte_labels is not None:
            labels = batch[2].to(device)
        return model.loss(x, conc, labels)

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for batch in train_loader:
            optimiser.zero_grad()
            loss = _step(batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tl += loss.item() * len(batch[0])
        tl /= len(train_idx)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                vl += _step(batch).item() * len(batch[0])
        vl /= n_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step(vl)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs} — "
                  f"train={tl:.6f}  val={vl:.6f}")

    return history
