"""Multi-task learning model for simultaneous analyte classification,
concentration regression, and spectral quality control.

A single shared encoder learns representations that are simultaneously
useful for all three objectives — producing richer, more generalizable
features than training each task independently.

Architecture
------------
::

    Input spectrum (N_wl,)
        │
    Shared Encoder  (CNN → GRU or Transformer)
        │
    Feature vector (embed_dim,)
        ├─→ Classification head  → analyte class probabilities
        ├─→ Regression head      → concentration (ppm)  [Softplus output]
        └─→ QC head              → quality score in [0, 1]  [Sigmoid output]

Usage
-----
::

    from src.models.multi_task import MultiTaskConfig, MultiTaskModel, train_multi_task

    cfg = MultiTaskConfig(
        input_dim=3648,
        n_analytes=4,
        predict_concentration=True,
        predict_qc=True,
    )
    model = MultiTaskModel(cfg)

    # Forward pass
    out = model(spectra)
    print(out.class_logits.shape)    # (B, n_analytes)
    print(out.concentration.shape)   # (B, 1)
    print(out.qc_score.shape)        # (B, 1)

    # Training
    from src.models.multi_task import MultiTaskTargets
    targets = MultiTaskTargets(
        analyte_labels=torch.randint(0, 4, (B,)),
        concentrations=torch.rand(B, 1),
        qc_labels=torch.ones(B, 1),
    )
    loss = model.loss(spectra, targets)
    loss.backward()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "MultiTaskConfig",
    "MultiTaskTargets",
    "MultiTaskOutput",
    "MultiTaskModel",
    "train_multi_task",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskConfig:
    """Hyperparameters for ``MultiTaskModel``.

    Parameters
    ----------
    input_dim :
        Input spectrum length (e.g. 3648 for CCS200) or latent_dim from
        ``SpectralAutoencoder`` (e.g. 64).
    embed_dim :
        Dimensionality of the shared representation.
    hidden_dim :
        Hidden size inside encoder layers.
    n_layers :
        Number of encoder layers.
    backbone :
        Encoder backbone: ``'cnn'``, ``'gru'``, or ``'transformer'``.
    n_analytes :
        Number of analyte classes for the classification head.
        Set to 0 to disable classification.
    predict_concentration :
        Whether to include the regression head.
    predict_qc :
        Whether to include the spectral quality control head.
        QC head outputs a scalar in [0, 1] (1 = high quality).
    class_weight :
        Loss weight for the classification head.
    conc_weight :
        Loss weight for the concentration regression head.
    qc_weight :
        Loss weight for the QC head.
    dropout :
        Dropout applied within the encoder.
    """
    input_dim: int = 3648
    embed_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 2
    backbone: Literal["cnn", "gru", "transformer"] = "gru"
    n_analytes: int = 4
    predict_concentration: bool = True
    predict_qc: bool = True
    class_weight: float = 1.0
    conc_weight: float = 1.0
    qc_weight: float = 0.5
    dropout: float = 0.1
    n_heads: int = 4       # only used for transformer backbone


# ---------------------------------------------------------------------------
# Targets / Outputs
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskTargets:
    """Ground-truth targets for one training batch.

    Any field may be ``None`` to skip that head's loss contribution.
    """
    analyte_labels: torch.Tensor | None = None    # (B,) int
    concentrations: torch.Tensor | None = None    # (B, 1) float
    qc_labels: torch.Tensor | None = None         # (B, 1) float in [0,1]


@dataclass
class MultiTaskOutput:
    """Prediction outputs from a forward pass."""
    class_logits: torch.Tensor | None    # (B, n_analytes)
    concentration: torch.Tensor | None   # (B, 1)
    qc_score: torch.Tensor | None        # (B, 1) in [0, 1]
    features: torch.Tensor               # (B, embed_dim) shared repr


# ---------------------------------------------------------------------------
# Backbone modules  (reuse patterns from temporal_encoder)
# ---------------------------------------------------------------------------

class _CNNEncoder(nn.Module):
    """1-D CNN over the spectrum dimension → global average pooled feature."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        out_ch = hidden_dim
        for i in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) → treat as 1-channel signal
        h = x.unsqueeze(1)              # (B, 1, input_dim)
        h = self.conv(h)                # (B, hidden_dim, input_dim)
        h = self.pool(h).squeeze(-1)    # (B, hidden_dim)
        return h


class _GRUEncoder(nn.Module):
    """Bidirectional GRU treating the spectrum as a sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 dropout: float) -> None:
        super().__init__()
        # Treat every spectrum point as a token with dim=1
        self.proj = nn.Linear(1, hidden_dim // 2)
        self.gru = nn.GRU(
            hidden_dim // 2, hidden_dim // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        h = x.unsqueeze(-1)            # (B, input_dim, 1)
        h = self.proj(h)               # (B, input_dim, hidden//2)
        _, hn = self.gru(h)            # hn: (2*n_layers, B, hidden//2)
        # Concat last fwd + bwd hidden state
        return torch.cat([hn[-2], hn[-1]], dim=-1)  # (B, hidden_dim)


class _TransformerEncoder(nn.Module):
    """CLS-token Transformer over down-sampled spectrum chunks."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 n_heads: int, dropout: float) -> None:
        super().__init__()
        # Chunk the spectrum into segments of ~32 points → tokens
        self.chunk_size = 32
        n_tokens = max(1, input_dim // self.chunk_size)
        self.chunk_proj = nn.Linear(self.chunk_size, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        # Pad to multiple of chunk_size
        pad = (self.chunk_size - L % self.chunk_size) % self.chunk_size
        if pad:
            x = F.pad(x, (0, pad))
        chunks = x.reshape(B, -1, self.chunk_size)    # (B, n_tokens, chunk_size)
        tokens = self.chunk_proj(chunks)               # (B, n_tokens, hidden_dim)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, 1+n_tokens, hidden_dim)
        out = self.transformer(tokens)                 # (B, 1+n_tokens, hidden_dim)
        return out[:, 0, :]                            # CLS token (B, hidden_dim)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MultiTaskModel(nn.Module):
    """Physics-agnostic multi-task model.

    Simultaneously predicts analyte class, concentration, and spectral quality
    from a raw or latent spectrum, sharing a single encoder backbone.
    """

    def __init__(self, config: MultiTaskConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiTaskConfig()
        cfg = self.config

        # Optional input projection for very high-dimensional spectra
        if cfg.input_dim > 512:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(cfg.input_dim, 512),
                nn.GELU(),
            )
            enc_in = 512
        else:
            self.input_proj = nn.Identity()
            enc_in = cfg.input_dim

        # Shared backbone encoder
        if cfg.backbone == "cnn":
            self.encoder: nn.Module = _CNNEncoder(
                enc_in, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        elif cfg.backbone == "gru":
            self.encoder = _GRUEncoder(
                enc_in, cfg.hidden_dim, cfg.n_layers, cfg.dropout)
        elif cfg.backbone == "transformer":
            self.encoder = _TransformerEncoder(
                enc_in, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.dropout)
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone!r}")

        # Feature projection to embed_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
        )

        # Task heads
        if cfg.n_analytes > 0:
            self.class_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.embed_dim // 2, cfg.n_analytes),
            )
        else:
            self.class_head = None

        if cfg.predict_concentration:
            self.conc_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.embed_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softplus(),   # concentration is always ≥ 0
            )
        else:
            self.conc_head = None

        if cfg.predict_qc:
            self.qc_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.embed_dim, 16),
                nn.GELU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),    # quality in [0, 1]
            )
        else:
            self.qc_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> MultiTaskOutput:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)

        Returns
        -------
        MultiTaskOutput with class_logits, concentration, qc_score, features
        """
        h = self.input_proj(x)          # (B, enc_in)
        h = self.encoder(h)             # (B, hidden_dim)
        features = self.feature_proj(h) # (B, embed_dim)

        class_logits = self.class_head(features) if self.class_head else None
        concentration = self.conc_head(features) if self.conc_head else None
        qc_score = self.qc_head(features) if self.qc_head else None

        return MultiTaskOutput(
            class_logits=class_logits,
            concentration=concentration,
            qc_score=qc_score,
            features=features,
        )

    def loss(
        self,
        x: torch.Tensor,
        targets: MultiTaskTargets,
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
        targets : MultiTaskTargets with optional ground-truth for each head

        Returns
        -------
        total_loss : scalar Tensor (always graph-connected for backward)
        """
        out = self.forward(x)
        # Initialise as graph-connected zero
        total: torch.Tensor = out.features.mean() * 0.0

        if out.class_logits is not None and targets.analyte_labels is not None:
            total = total + self.config.class_weight * F.cross_entropy(
                out.class_logits, targets.analyte_labels)

        if out.concentration is not None and targets.concentrations is not None:
            total = total + self.config.conc_weight * F.mse_loss(
                out.concentration, targets.concentrations)

        if out.qc_score is not None and targets.qc_labels is not None:
            total = total + self.config.qc_weight * F.binary_cross_entropy(
                out.qc_score, targets.qc_labels.float())

        return total

    @torch.no_grad()
    def embed_numpy(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Encode numpy spectra → numpy shared feature matrix.

        Parameters
        ----------
        spectra : ndarray, shape (N, input_dim)

        Returns
        -------
        features : ndarray, shape (N, embed_dim)
        """
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        return self.forward(x).features.cpu().numpy()

    @torch.no_grad()
    def predict_analyte(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Predict analyte class indices from numpy spectra."""
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        logits = self.forward(x).class_logits
        if logits is None:
            raise RuntimeError("n_analytes=0 — classification head disabled.")
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_concentration(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Predict concentration from numpy spectra."""
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        conc = self.forward(x).concentration
        if conc is None:
            raise RuntimeError("predict_concentration=False in config.")
        return conc.cpu().numpy()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_multi_task(
    model: MultiTaskModel,
    spectra: "np.ndarray",                    # (N, input_dim)  # type: ignore[name-defined]
    analyte_labels: "np.ndarray | None" = None,   # (N,) int  # type: ignore[name-defined]
    concentrations: "np.ndarray | None" = None,    # (N,) float  # type: ignore[name-defined]
    qc_labels: "np.ndarray | None" = None,         # (N,) float in [0,1]  # type: ignore[name-defined]
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the multi-task model.

    Parameters
    ----------
    model : MultiTaskModel
    spectra : ndarray, shape (N, input_dim)
    analyte_labels : optional integer analyte class labels, shape (N,)
    concentrations : optional concentration targets, shape (N,)
    qc_labels : optional QC quality scores in [0, 1], shape (N,)
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n = len(spectra)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    tensors: list[torch.Tensor] = [
        torch.from_numpy(spectra.astype(np.float32))
    ]
    has_class = analyte_labels is not None
    has_conc = concentrations is not None
    has_qc = qc_labels is not None

    if has_class:
        assert analyte_labels is not None
        tensors.append(torch.from_numpy(analyte_labels.astype(np.int64)))
    if has_conc:
        assert concentrations is not None
        tensors.append(
            torch.from_numpy(concentrations.astype(np.float32)).unsqueeze(1)
        )
    if has_qc:
        assert qc_labels is not None
        tensors.append(
            torch.from_numpy(qc_labels.astype(np.float32)).unsqueeze(1)
        )

    def _loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        subset = [t[indices] for t in tensors]
        return DataLoader(TensorDataset(*subset), batch_size=batch_size,
                          shuffle=shuffle)

    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=1e-5)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _make_targets(batch: tuple) -> MultiTaskTargets:
        i = 1
        cls_lbl = batch[i].to(device) if has_class else None
        if has_class:
            i += 1
        concs = batch[i].to(device) if has_conc else None
        if has_conc:
            i += 1
        qc = batch[i].to(device) if has_qc else None
        return MultiTaskTargets(
            analyte_labels=cls_lbl,
            concentrations=concs,
            qc_labels=qc,
        )

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            targets = _make_targets(batch)
            optimiser.zero_grad()
            loss = model.loss(x, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tl += loss.item() * len(x)
        tl /= len(train_idx)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                targets = _make_targets(batch)
                vl += model.loss(x, targets).item() * len(x)
        vl /= n_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs} — "
                  f"train={tl:.6f}  val={vl:.6f}")

    return history
