"""Cross-configuration domain adaptation for spectrometer transfer learning.

Problem
-------
A model trained on sensor configuration A (e.g. one instrument, one wavelength
range) must work on configuration B without re-training from scratch.  Spectral
differences between configurations arise from: different wavelength grids,
different integration times, different optical alignments, and instrument-
specific noise profiles.

This module provides two complementary strategies:

1. **Fine-tuning** (``fine_tune``): freeze the shared encoder, retrain only
   the task heads on a small labelled set from the new configuration.
   Requires as few as 10–50 labelled spectra from config B.

2. **Adversarial domain adaptation** (``DomainAdaptModel`` / ``train_domain_adapt``):
   train the encoder to produce config-invariant embeddings using a gradient
   reversal layer (GRL).  The adversarial discriminator tries to predict which
   configuration a spectrum came from; the GRL makes the encoder fool it.
   No labels required from config B — fully unsupervised alignment.

Usage
-----
::

    from src.models.transfer import (
        fine_tune,
        DomainAdaptModel, DomainAdaptConfig, train_domain_adapt,
    )

    # 1. Fine-tuning — freeze encoder, adapt heads with 50 new spectra
    from src.models.multi_task import MultiTaskModel, MultiTaskConfig
    model = MultiTaskModel(MultiTaskConfig(n_analytes=3))
    # ... load pretrained weights ...
    fine_tune(model, new_spectra, analyte_labels=new_labels, n_epochs=20)

    # 2. Adversarial domain adaptation
    adapt_cfg = DomainAdaptConfig(embed_dim=128, n_analytes=3)
    adapt_model = DomainAdaptModel(adapt_cfg)
    train_domain_adapt(
        adapt_model,
        source_spectra=src_X, source_labels=src_y,
        target_spectra=tgt_X,   # no labels needed
        n_epochs=50,
    )
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np

    from src.models.multi_task import MultiTaskModel

__all__ = [
    "DomainAdaptConfig",
    "DomainAdaptModel",
    "fine_tune",
    "train_domain_adapt",
    "evaluate_transfer",
]


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReversal(torch.autograd.Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward: identity.
    Backward: multiply gradient by -lambda (reverses the gradient direction).
    This forces the encoder to produce domain-invariant features.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx,
                x: torch.Tensor, lam: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lam))
        return x.clone()

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx,
                 grad_output: torch.Tensor):
        (lam,) = ctx.saved_tensors
        return -lam * grad_output, None


def grad_reverse(x: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
    return _GradReversal.apply(x, lam)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DomainAdaptConfig:
    """Hyperparameters for ``DomainAdaptModel``.

    Parameters
    ----------
    input_dim :
        Spectrum length or pre-computed latent dimension.
    embed_dim :
        Shared encoder output dimension.
    hidden_dim :
        Hidden size for encoder layers.
    n_analytes :
        Number of analyte classes (classification head).
    predict_concentration :
        Include concentration regression head.
    adapt_lambda :
        Gradient reversal strength.  Ramp from 0 → adapt_lambda during
        training (set ``ramp_lambda=True``) for stable early training.
    ramp_lambda :
        If True, linearly ramp lambda from 0 to adapt_lambda over training.
    backbone :
        Encoder backbone architecture.
    dropout :
        Dropout rate in encoder.
    """
    input_dim: int = 3648
    embed_dim: int = 128
    hidden_dim: int = 256
    n_analytes: int = 4
    predict_concentration: bool = True
    adapt_lambda: float = 1.0
    ramp_lambda: bool = True
    backbone: Literal["cnn", "gru", "transformer"] = "gru"
    dropout: float = 0.1
    n_layers: int = 2


# ---------------------------------------------------------------------------
# Domain adaptation model
# ---------------------------------------------------------------------------

class DomainAdaptModel(nn.Module):
    """Adversarial domain-adaptive spectral encoder.

    Contains:
    - Shared encoder (produces config-invariant features via GRL)
    - Task head(s): concentration regression and/or analyte classification
    - Domain discriminator: binary classifier (source vs target config)
    """

    def __init__(self, config: DomainAdaptConfig | None = None) -> None:
        super().__init__()
        self.config = config or DomainAdaptConfig()
        cfg = self.config

        # Shared encoder (reuse GRU backbone from multi_task)
        if cfg.input_dim > 512:
            self.input_proj: nn.Module = nn.Sequential(
                nn.Linear(cfg.input_dim, 512), nn.GELU())
        else:
            self.input_proj = nn.Identity()

        # Simple GRU encoder
        self.proj_in = nn.Linear(1, cfg.hidden_dim // 2)
        self.gru = nn.GRU(
            cfg.hidden_dim // 2, cfg.hidden_dim // 2,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
            nn.LayerNorm(cfg.embed_dim),
            nn.GELU(),
        )

        # Task head: analyte classification
        if cfg.n_analytes > 0:
            self.class_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.embed_dim // 2),
                nn.GELU(),
                nn.Linear(cfg.embed_dim // 2, cfg.n_analytes),
            )
        else:
            self.class_head = None

        # Task head: concentration regression
        if cfg.predict_concentration:
            self.conc_head: nn.Module | None = nn.Sequential(
                nn.Linear(cfg.embed_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softplus(),
            )
        else:
            self.conc_head = None

        # Domain discriminator (source=0, target=1)
        self.domain_disc = nn.Sequential(
            nn.Linear(cfg.embed_dim, 64),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectra to shared feature space.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)

        Returns
        -------
        features : Tensor, shape (B, embed_dim)
        """
        h = self.input_proj(x)         # (B, enc_in)
        h = h.unsqueeze(-1)            # (B, enc_in, 1)
        h = self.proj_in(h)            # (B, enc_in, hidden//2)
        _, hn = self.gru(h)            # hn: (2*n_layers, B, hidden//2)
        h = torch.cat([hn[-2], hn[-1]], dim=-1)  # (B, hidden_dim)
        return self.feature_proj(h)    # (B, embed_dim)

    def forward(
        self, x: torch.Tensor, lam: float = 1.0
    ) -> dict[str, torch.Tensor | None]:
        """Forward pass with gradient reversal for domain adaptation.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
        lam : gradient reversal strength

        Returns
        -------
        dict with keys: 'features', 'class_logits', 'concentration',
                        'domain_logits'
        """
        features = self.encode(x)

        class_logits = self.class_head(features) if self.class_head else None
        concentration = self.conc_head(features) if self.conc_head else None

        # Domain discriminator receives reversed gradients
        domain_logits = self.domain_disc(grad_reverse(features, lam))

        return {
            "features": features,
            "class_logits": class_logits,
            "concentration": concentration,
            "domain_logits": domain_logits,
        }

    def loss(
        self,
        src_x: torch.Tensor,
        src_labels: torch.Tensor | None = None,       # (B_s,) int analyte
        src_conc: torch.Tensor | None = None,          # (B_s, 1) float
        tgt_x: torch.Tensor | None = None,             # (B_t, input_dim)
        lam: float = 1.0,
        task_weight: float = 1.0,
        domain_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined task + adversarial domain loss.

        Returns
        -------
        total_loss, breakdown dict with keys 'task', 'domain'
        """
        src_out = self.forward(src_x, lam)
        features_src = src_out["features"]
        assert features_src is not None

        # Task losses on source data
        task_loss: torch.Tensor = features_src.mean() * 0.0

        if src_out["class_logits"] is not None and src_labels is not None:
            task_loss = task_loss + F.cross_entropy(
                src_out["class_logits"], src_labels)

        if src_out["concentration"] is not None and src_conc is not None:
            task_loss = task_loss + F.mse_loss(
                src_out["concentration"], src_conc)

        # Domain adversarial loss
        domain_loss: torch.Tensor = features_src.mean() * 0.0
        B_s = src_x.shape[0]
        src_domain = torch.zeros(B_s, 1, device=src_x.device)  # source = 0
        domain_loss = domain_loss + F.binary_cross_entropy_with_logits(
            src_out["domain_logits"], src_domain)

        if tgt_x is not None:
            tgt_out = self.forward(tgt_x, lam)
            B_t = tgt_x.shape[0]
            tgt_domain = torch.ones(B_t, 1, device=tgt_x.device)  # target = 1
            domain_loss = domain_loss + F.binary_cross_entropy_with_logits(
                tgt_out["domain_logits"], tgt_domain)
            domain_loss = domain_loss / 2.0

        total = task_weight * task_loss + domain_weight * domain_loss
        breakdown = {
            "task": task_loss.item(),
            "domain": domain_loss.item(),
        }
        return total, breakdown

    @torch.no_grad()
    def embed_numpy(self, spectra: np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
        """Encode numpy spectra to features."""
        import numpy as np
        device = next(self.parameters()).device
        x = torch.from_numpy(spectra.astype(np.float32)).to(device)
        return self.encode(x).cpu().numpy()


# ---------------------------------------------------------------------------
# Fine-tuning (freeze encoder, adapt heads)
# ---------------------------------------------------------------------------

def fine_tune(
    model: MultiTaskModel,   # type: ignore[name-defined]
    spectra: np.ndarray,     # type: ignore[name-defined]
    analyte_labels: np.ndarray | None = None,
    concentrations: np.ndarray | None = None,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
    freeze_encoder: bool = True,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Fine-tune a pretrained ``MultiTaskModel`` on a small new-config dataset.

    Parameters
    ----------
    model :
        Pretrained ``MultiTaskModel`` (from source configuration).
    spectra :
        ndarray, shape (N, input_dim) — spectra from the new configuration.
    analyte_labels :
        Integer class labels for the new spectra (optional).
    concentrations :
        Concentration values for the new spectra (optional).
    freeze_encoder :
        If True (default), freeze the shared encoder and only update the
        task heads.  Set False to fine-tune the entire model.
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.multi_task import MultiTaskTargets

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if freeze_encoder:
        # Freeze all encoder parameters
        for name, param in model.named_parameters():
            if "class_head" in name or "conc_head" in name or "qc_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    n = len(spectra)
    X = torch.from_numpy(spectra.astype(np.float32))
    tensors: list[torch.Tensor] = [X]
    has_class = analyte_labels is not None
    has_conc = concentrations is not None

    if has_class:
        assert analyte_labels is not None
        tensors.append(torch.from_numpy(analyte_labels.astype(np.int64)))
    if has_conc:
        assert concentrations is not None
        tensors.append(
            torch.from_numpy(concentrations.astype(np.float32)).unsqueeze(1))

    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                        shuffle=True)

    # Only update trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.Adam(trainable, lr=lr)

    history: dict[str, list[float]] = {"train_loss": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        tl = 0.0
        for batch in loader:
            x = batch[0].to(device)
            i = 1
            cls_lbl = batch[i].to(device) if has_class else None
            if has_class:
                i += 1
            concs = batch[i].to(device) if has_conc else None

            targets = MultiTaskTargets(
                analyte_labels=cls_lbl, concentrations=concs)

            optimiser.zero_grad()
            loss = model.loss(x, targets)
            loss.backward()
            optimiser.step()
            tl += loss.item() * len(x)
        tl /= n
        history["train_loss"].append(tl)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"FineTune epoch {epoch:4d}/{n_epochs} — loss={tl:.6f}")

    # Restore all parameters to trainable
    for param in model.parameters():
        param.requires_grad = True

    return history


# ---------------------------------------------------------------------------
# Adversarial domain adaptation training loop
# ---------------------------------------------------------------------------

def train_domain_adapt(
    model: DomainAdaptModel,
    source_spectra: np.ndarray,          # (N_s, input_dim)  # type: ignore[name-defined]
    source_labels: np.ndarray | None = None,   # (N_s,) int  # type: ignore[name-defined]
    source_conc: np.ndarray | None = None,      # (N_s,) float  # type: ignore[name-defined]
    target_spectra: np.ndarray | None = None,   # (N_t, input_dim) — unlabelled  # type: ignore[name-defined]
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    task_weight: float = 1.0,
    domain_weight: float = 1.0,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train the domain-adaptive model.

    Aligns source and target configuration representations via GRL adversarial
    training.  Source labels are used for task supervision; target spectra
    provide domain alignment signal (no target labels required).

    Parameters
    ----------
    source_spectra : labeled spectra from source instrument configuration
    source_labels : analyte class labels for source spectra
    source_conc : concentration labels for source spectra
    target_spectra : unlabelled spectra from target instrument configuration
    """
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    n_src = len(source_spectra)

    src_tensors: list[torch.Tensor] = [
        torch.from_numpy(source_spectra.astype(np.float32))]
    has_labels = source_labels is not None
    has_conc = source_conc is not None
    if has_labels:
        assert source_labels is not None
        src_tensors.append(
            torch.from_numpy(source_labels.astype(np.int64)))
    if has_conc:
        assert source_conc is not None
        src_tensors.append(
            torch.from_numpy(source_conc.astype(np.float32)).unsqueeze(1))

    src_loader = DataLoader(
        TensorDataset(*src_tensors), batch_size=batch_size, shuffle=True)

    tgt_loader = None
    if target_spectra is not None:
        tgt_loader = DataLoader(
            TensorDataset(torch.from_numpy(target_spectra.astype(np.float32))),
            batch_size=batch_size, shuffle=True)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=n_epochs, eta_min=1e-5)

    history: dict[str, list[float]] = {
        "train_loss": [], "task_loss": [], "domain_loss": []}

    for epoch in range(1, n_epochs + 1):
        # Linearly ramp lambda from 0 → adapt_lambda
        if model.config.ramp_lambda:
            lam = model.config.adapt_lambda * (epoch / n_epochs)
        else:
            lam = model.config.adapt_lambda

        model.train()
        epoch_total, epoch_task, epoch_domain = 0.0, 0.0, 0.0

        tgt_iter = iter(tgt_loader) if tgt_loader else None

        for src_batch in src_loader:
            src_x = src_batch[0].to(device)
            i = 1
            src_lbl = src_batch[i].to(device) if has_labels else None
            if has_labels:
                i += 1
            src_c = src_batch[i].to(device) if has_conc else None

            # Fetch target batch (cycle if exhausted)
            tgt_x = None
            if tgt_iter is not None:
                try:
                    tgt_x = next(tgt_iter)[0].to(device)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)  # type: ignore[arg-type]
                    tgt_x = next(tgt_iter)[0].to(device)

            optimiser.zero_grad()
            loss, breakdown = model.loss(
                src_x, src_lbl, src_c, tgt_x,
                lam=lam,
                task_weight=task_weight,
                domain_weight=domain_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            bs = len(src_x)
            epoch_total += loss.item() * bs
            epoch_task += breakdown["task"] * bs
            epoch_domain += breakdown["domain"] * bs

        epoch_total /= n_src
        epoch_task /= n_src
        epoch_domain /= n_src

        history["train_loss"].append(epoch_total)
        history["task_loss"].append(epoch_task)
        history["domain_loss"].append(epoch_domain)
        scheduler.step()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:4d}/{n_epochs} — "
                  f"total={epoch_total:.4f}  task={epoch_task:.4f}  "
                  f"domain={epoch_domain:.4f}  lam={lam:.3f}")

    return history


# ---------------------------------------------------------------------------
# Transfer evaluation utility
# ---------------------------------------------------------------------------

def evaluate_transfer(
    model: DomainAdaptModel,
    test_spectra: np.ndarray,       # (N, input_dim)  # type: ignore[name-defined]
    test_labels: np.ndarray,        # (N,) int  # type: ignore[name-defined]
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate analyte classification accuracy on a test set.

    Returns
    -------
    dict with 'accuracy' and 'n_samples'
    """
    import numpy as np
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    X = torch.from_numpy(test_spectra.astype(np.float32)).to(device)
    with torch.no_grad():
        out = model.forward(X, lam=0.0)
    logits = out["class_logits"]
    if logits is None:
        raise RuntimeError("n_analytes=0 — no classification head.")

    preds = logits.argmax(dim=-1).cpu().numpy()
    accuracy = float((preds == test_labels).mean())
    return {"accuracy": accuracy, "n_samples": len(test_labels)}
