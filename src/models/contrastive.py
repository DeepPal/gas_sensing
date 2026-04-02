"""Contrastive analyte fingerprinting.

Trains an encoder to produce embeddings where spectra from the SAME analyte
cluster tightly together, while spectra from DIFFERENT analytes are pushed
apart — WITHOUT needing to know any physics.

This enables few-shot analyte identification: given as few as 5–10 reference
spectra from a new analyte, the encoder immediately places new measurements
in the correct cluster.

Loss functions
--------------
``'supcon'`` (recommended)
    Supervised Contrastive Loss (Khosla et al., 2020).  Uses ALL positive
    pairs in the batch simultaneously.  More stable and sample-efficient
    than vanilla triplet loss.

``'triplet'``
    Triplet margin loss with hard negative mining within the batch.

``'ntxent'``
    NT-Xent (Normalized Temperature-scaled Cross Entropy).  The SimCLR
    loss — treats augmented versions of the same spectrum as positives.

Usage
-----
::

    import torch
    from src.models.contrastive import ContrastiveEncoder, ContrastiveConfig

    cfg = ContrastiveConfig(input_dim=3648, embed_dim=64)
    model = ContrastiveEncoder(cfg)

    # Training: spectra + integer analyte labels
    spectra = torch.randn(32, 3648)
    labels = torch.randint(0, 4, (32,))
    loss = model.loss(spectra, labels)

    # Inference: get embeddings for new spectra
    embeddings = model.embed(spectra)   # (32, 64)  L2-normalised

    # Few-shot identification: build reference gallery
    gallery_embeddings = model.embed(reference_spectra)  # (K, 64)
    gallery_labels = ["Ethanol"] * 5 + ["IPA"] * 5

    predictions = model.identify(new_spectra, gallery_embeddings, gallery_labels)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "ContrastiveConfig",
    "ContrastiveEncoder",
    "train_contrastive",
    "build_gallery",
    "identify_analyte",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveConfig:
    """Hyperparameters for ``ContrastiveEncoder``.

    Parameters
    ----------
    input_dim :
        Input spectrum length or latent vector dimensionality.
        Use latent_dim from SpectralAutoencoder for best results (e.g. 64).
        Can also accept raw spectra (e.g. 3648) — a projection layer is added.
    embed_dim :
        Dimensionality of the contrastive embedding space.
        Typical: 32–128.  Larger = more expressive but needs more data.
    hidden_dim :
        Size of the MLP projection head hidden layer.
    n_layers :
        Number of layers in the projection MLP.
    loss_type :
        Which contrastive loss to use: ``'supcon'``, ``'triplet'``, ``'ntxent'``.
    temperature :
        Softmax temperature for SupCon / NT-Xent.  Lower = sharper clusters.
    triplet_margin :
        Margin for triplet loss.
    dropout :
        Dropout in the MLP.
    normalise_embeddings :
        If True (default), L2-normalise all embeddings to unit sphere.
        Required for cosine similarity comparisons.
    """
    input_dim: int = 64
    embed_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 3
    loss_type: Literal["supcon", "triplet", "ntxent"] = "supcon"
    temperature: float = 0.07
    triplet_margin: float = 0.5
    dropout: float = 0.1
    normalise_embeddings: bool = True


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------

class _ProjectionMLP(nn.Module):
    """Multi-layer perceptron that maps input → embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(n_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ContrastiveEncoder(nn.Module):
    """Physics-agnostic contrastive encoder for analyte fingerprinting.

    Parameters
    ----------
    config :
        ``ContrastiveConfig`` instance.
    """

    def __init__(self, config: ContrastiveConfig | None = None) -> None:
        super().__init__()
        self.config = config or ContrastiveConfig()
        cfg = self.config

        # Input projection for high-dim raw spectra
        if cfg.input_dim > 512:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(cfg.input_dim, 256),
                nn.GELU(),
            )
            proj_in = 256
        else:
            self.input_proj = nn.Identity()
            proj_in = cfg.input_dim

        self.projector = _ProjectionMLP(
            proj_in, cfg.hidden_dim, cfg.embed_dim, cfg.n_layers, cfg.dropout
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalised embeddings.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)

        Returns
        -------
        z : Tensor, shape (B, embed_dim)  — unit norm if normalise_embeddings
        """
        h = self.input_proj(x)
        z = self.projector(h)
        if self.config.normalise_embeddings:
            z = F.normalize(z, dim=-1)
        return z

    def loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
        labels : Tensor, shape (B,) — integer analyte class labels

        Returns
        -------
        loss : scalar Tensor
        """
        z = self.embed(x)

        if self.config.loss_type == "supcon":
            return _supervised_contrastive_loss(z, labels, self.config.temperature)
        elif self.config.loss_type == "triplet":
            return _triplet_loss(z, labels, self.config.triplet_margin)
        elif self.config.loss_type == "ntxent":
            return _ntxent_loss(z, labels, self.config.temperature)
        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type!r}")

    def identify(
        self,
        x: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        gallery_labels: list[Any],
        top_k: int = 1,
    ) -> list[Any]:
        """Identify analytes by nearest-neighbour in embedding space.

        Parameters
        ----------
        x : Tensor, shape (N, input_dim) — query spectra
        gallery_embeddings : Tensor, shape (K, embed_dim) — reference embeddings
        gallery_labels : list of length K — analyte labels for gallery
        top_k : return top-k predictions per query

        Returns
        -------
        predictions : list of length N (or list of lists if top_k > 1)
        """
        with torch.no_grad():
            query = self.embed(x)  # (N, embed_dim)

        # Cosine similarity (embeddings are L2-normalised)
        sims = query @ gallery_embeddings.T  # (N, K)

        if top_k == 1:
            best = sims.argmax(dim=-1).tolist()
            return [gallery_labels[i] for i in best]
        else:
            topk_idx = sims.topk(top_k, dim=-1).indices.tolist()
            return [[gallery_labels[i] for i in row] for row in topk_idx]

    @torch.no_grad()
    def embed_numpy(self, spectra: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
        """Embed numpy spectra → numpy embedding matrix.

        Parameters
        ----------
        spectra : ndarray, shape (N, input_dim)

        Returns
        -------
        embeddings : ndarray, shape (N, embed_dim)
        """
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        return self.embed(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Loss implementations
# ---------------------------------------------------------------------------

def _supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor, maximises similarity to ALL other samples from the same
    analyte class relative to all negatives.
    """
    B = z.shape[0]
    device = z.device

    # Pairwise cosine similarity matrix (z is already L2-normalised)
    sim_matrix = z @ z.T / temperature  # (B, B)

    # Mask out self-similarity (-1e9 avoids 0 * -inf = nan in pos_mask * log_prob)
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)

    # Positive mask: same label, not self
    labels_col = labels.unsqueeze(1)
    pos_mask = (labels_col == labels_col.T) & ~self_mask  # (B, B)

    # Number of positives per anchor
    n_pos = pos_mask.sum(dim=1).float()

    # Skip anchors with no positives
    has_pos = n_pos > 0

    if not has_pos.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-softmax over all non-self pairs
    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

    # Mean log-prob over positive pairs per anchor
    per_anchor = (pos_mask * log_prob).sum(dim=1) / n_pos.clamp(min=1)
    return -per_anchor[has_pos].mean()


def _triplet_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Batch hard triplet loss with hardest positive and hardest negative mining."""
    B = z.shape[0]
    device = z.device

    # Pairwise squared L2 distances
    dist = torch.cdist(z, z, p=2)  # (B, B)

    labels_col = labels.unsqueeze(1)
    same = (labels_col == labels_col.T)
    diff = ~same

    # Hardest positive (furthest same-class sample)
    pos_dist = dist.clone()
    pos_dist[diff] = 0.0
    hardest_pos, _ = pos_dist.max(dim=1)

    # Hardest negative (closest different-class sample)
    neg_dist = dist.clone()
    neg_dist[same] = float("inf")
    hardest_neg, _ = neg_dist.min(dim=1)

    loss = F.relu(hardest_pos - hardest_neg + margin)
    return loss.mean()


def _ntxent_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """NT-Xent (SimCLR) loss treating first positive as the target."""
    # Treat same-label samples as positives for one anchor each
    B = z.shape[0]
    device = z.device

    sim = z @ z.T / temperature  # (B, B)
    self_mask = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(self_mask, float("-inf"))

    labels_col = labels.unsqueeze(1)
    pos_mask = (labels_col == labels_col.T) & ~self_mask

    # Use first positive as target for cross-entropy
    losses = []
    for i in range(B):
        pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(-1)
        if len(pos_idx) == 0:
            continue
        target = pos_idx[0]
        losses.append(F.cross_entropy(sim[i].unsqueeze(0),
                                      target.unsqueeze(0)))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Gallery helpers
# ---------------------------------------------------------------------------

def build_gallery(
    model: ContrastiveEncoder,
    spectra: "np.ndarray",  # type: ignore[name-defined]
    labels: list[Any],
) -> tuple[torch.Tensor, list[Any]]:
    """Build a reference gallery by encoding labelled reference spectra.

    Parameters
    ----------
    model :
        Trained ``ContrastiveEncoder``.
    spectra :
        ndarray, shape (K, input_dim) — reference spectra.
    labels :
        List of K analyte labels (strings or ints).

    Returns
    -------
    gallery_embeddings : Tensor, shape (K, embed_dim)
    gallery_labels : list of K labels
    """
    embeddings = torch.from_numpy(
        model.embed_numpy(spectra).astype("float32")
    )
    return embeddings, list(labels)


def identify_analyte(
    model: ContrastiveEncoder,
    query_spectra: "np.ndarray",  # type: ignore[name-defined]
    gallery_embeddings: torch.Tensor,
    gallery_labels: list[Any],
    top_k: int = 1,
) -> list[Any]:
    """Identify analytes in query spectra using the reference gallery.

    Parameters
    ----------
    model :
        Trained ``ContrastiveEncoder``.
    query_spectra :
        ndarray, shape (N, input_dim).
    gallery_embeddings :
        Tensor, shape (K, embed_dim).
    gallery_labels :
        List of K analyte labels.
    top_k :
        Return top-k candidates per query.

    Returns
    -------
    predictions : list of N predicted labels (or list of lists if top_k > 1)
    """
    import numpy as np
    x = torch.from_numpy(query_spectra.astype(np.float32))
    return model.identify(x, gallery_embeddings, gallery_labels, top_k=top_k)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_contrastive(
    model: ContrastiveEncoder,
    spectra: "np.ndarray",  # type: ignore[name-defined]
    labels: "np.ndarray",   # type: ignore[name-defined]
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the contrastive encoder.

    Parameters
    ----------
    model :
        ``ContrastiveEncoder`` instance.
    spectra :
        ndarray, shape (N, input_dim).
    labels :
        ndarray, shape (N,) — integer analyte class labels.
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

    X = torch.from_numpy(spectra.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.int64))

    def _loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(X[indices], y[indices]),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = _loader(train_idx, shuffle=True)
    val_loader = _loader(val_idx, shuffle=False)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=1e-5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = model.loss(xb, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tl += loss.item() * len(xb)
        tl /= len(train_idx)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vl += model.loss(xb, yb).item() * len(xb)
        vl /= n_val

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs} — "
                  f"train={tl:.6f}  val={vl:.6f}")

    return history
