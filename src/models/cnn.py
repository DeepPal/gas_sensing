"""
src.models.cnn
==============
1-D CNN gas classifier for Au-MIP LSPR spectra.

Architecture
------------
Dual-head network sharing a convolutional feature extractor:

  Input (batch, 1, L)
    │
    ├── Conv1d(1→16, k=7) → BN → ReLU → MaxPool(2)
    ├── Conv1d(16→32, k=5) → BN → ReLU → MaxPool(2)
    └── Conv1d(32→64, k=3) → BN → ReLU → MaxPool(2)
    │
    ├── Flatten → Linear → Dropout(0.3)
    │
    ├─ [cls_head]  Linear → softmax logits   (gas type)
    └─ [reg_head]  Linear → scalar            (concentration ppm)

The shared backbone makes joint training efficient; the network learns
features useful for both tasks simultaneously.

Public API
----------
- ``GasCNN``            — raw nn.Module (define-once, use everywhere)
- ``CNNGasClassifier``  — high-level fit / predict / save / load wrapper
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def augment_spectra(
    X: np.ndarray,
    y_label: np.ndarray,
    y_conc: np.ndarray,
    n_augment: int = 4,
    noise_frac: float = 0.015,
    scale_range: tuple[float, float] = (0.90, 1.10),
    shift_pixels: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectral data augmentation for small LSPR datasets.

    Generates synthetic spectra by applying three physically motivated
    transformations to each original spectrum:

    1. **Gaussian noise** — simulates detector shot noise and readout noise
       (σ = ``noise_frac`` × per-spectrum standard deviation).
    2. **Intensity scaling** — models lamp/LED intensity fluctuations ±10%.
    3. **Spectral shift** — accounts for thermal pixel registration drift
       (±``shift_pixels`` pixels, circular wrapping preserves array length).

    All label and concentration values are copied from the original sample,
    because these augmentations do not change the physical gas identity or
    concentration.

    Parameters
    ----------
    X:
        Original spectra, shape ``(n_samples, n_points)``.
    y_label, y_conc:
        Class labels and concentrations, shape ``(n_samples,)``.
    n_augment:
        Number of augmented copies per original spectrum (default 4 → 5×
        dataset size including originals).
    noise_frac:
        Gaussian noise level as a fraction of per-spectrum std.  0.015 ≈
        1.5 % noise, representative of CCS200 dark-noise contribution.
    scale_range:
        ``(min_scale, max_scale)`` for multiplicative intensity jitter.
    shift_pixels:
        Maximum pixel shift in either direction.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    X_aug, y_label_aug, y_conc_aug:
        Augmented arrays (originals + synthetic), shapes
        ``(n_samples * (1 + n_augment), n_points)`` etc.
    """
    rng = np.random.default_rng(random_state)
    n, L = X.shape

    aug_X: list[np.ndarray] = [X]
    aug_y: list[np.ndarray] = [y_label]
    aug_c: list[np.ndarray] = [y_conc]

    for _ in range(n_augment):
        Xb = X.copy()

        # 1. Multiplicative intensity jitter (lamp / source variability)
        scales = rng.uniform(scale_range[0], scale_range[1], size=(n, 1))
        Xb = Xb * scales

        # 2. Additive Gaussian noise (detector shot + readout noise)
        per_std = Xb.std(axis=1, keepdims=True) + 1e-10
        noise = rng.normal(0.0, noise_frac * per_std, size=Xb.shape)
        Xb = Xb + noise

        # 3. Circular spectral shift (thermal pixel registration drift)
        shifts = np.asarray(rng.integers(-shift_pixels, shift_pixels + 1, size=n), dtype=int)
        Xb = np.stack([np.roll(Xb[i], int(shift)) for i, shift in enumerate(shifts)])

        aug_X.append(Xb)
        aug_y.append(y_label.copy())
        aug_c.append(y_conc.copy())

    X_out = np.concatenate(aug_X, axis=0).astype(np.float32)
    y_out = np.concatenate(aug_y, axis=0)
    c_out = np.concatenate(aug_c, axis=0).astype(np.float32)
    return X_out, y_out, c_out


# Torch is optional — wrap all references so the module imports cleanly
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------


def _requires_torch(name: str) -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(f"{name} requires PyTorch. Install with: pip install torch")


if _TORCH_AVAILABLE:

    class GasCNN(nn.Module):
        """Dual-head 1-D CNN: classification + concentration regression.

        Parameters
        ----------
        input_length:
            Number of spectral points in each input spectrum.
        num_classes:
            Number of gas type classes (e.g., 4 for Ethanol/IPA/Methanol/MixVOC).
        """

        def __init__(self, input_length: int = 1000, num_classes: int = 4) -> None:
            super().__init__()
            self.input_length = input_length
            self.num_classes = num_classes

            self.features = nn.Sequential(
                # Block 1
                nn.Conv1d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.MaxPool1d(2),
                # Block 2
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                # Block 3
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )

            # Compute flattened size with a dummy forward pass
            with torch.no_grad():
                dummy = torch.zeros(1, 1, input_length)
                flat_size = self.features(dummy).view(1, -1).size(1)

            self.neck = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_size, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
            )

            self.cls_head = nn.Linear(128, num_classes)
            self.reg_head = nn.Linear(128, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.

            Parameters
            ----------
            x:
                Input tensor, shape ``(batch, 1, L)``.

            Returns
            -------
            cls_logits : Tensor, shape (batch, num_classes)
            conc_pred  : Tensor, shape (batch, 1)
            """
            feat = self.features(x)
            neck = self.neck(feat)
            return self.cls_head(neck), self.reg_head(neck)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


class CNNGasClassifier:
    """High-level wrapper for :class:`GasCNN` — fit, predict, save, load.

    Handles spectrum resizing, device selection, training loop, and
    checkpoint I/O.

    Parameters
    ----------
    input_length:
        Expected number of points per spectrum. Inputs of different length
        are resampled to this size before inference.
    num_classes:
        Number of gas type labels.
    device:
        ``"cpu"``, ``"cuda"``, or ``"auto"`` (selects GPU if available).
    """

    def __init__(
        self,
        input_length: int = 1000,
        num_classes: int = 4,
        device: str = "auto",
    ) -> None:
        _requires_torch("CNNGasClassifier")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.input_length = input_length
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model: GasCNN | None = None
        self.class_map: dict[int, str] = {}  # int → gas name
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_X(self, X: np.ndarray) -> torch.Tensor:  # noqa: N802
        """Resample *X* to ``input_length`` and convert to tensor."""
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]

        if arr.shape[1] != self.input_length:
            x_old = np.linspace(0, 1, arr.shape[1])
            x_new = np.linspace(0, 1, self.input_length)
            resampled = np.stack([np.interp(x_new, x_old, row) for row in arr], axis=0)
            arr = resampled.astype(np.float32)

        # Normalize each spectrum to zero-mean, unit-std
        mu = arr.mean(axis=1, keepdims=True)
        sd = arr.std(axis=1, keepdims=True) + 1e-8
        arr = (arr - mu) / sd

        # Shape: (batch, 1, L)
        return torch.from_numpy(arr[:, np.newaxis, :]).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y_label: np.ndarray,
        y_conc: np.ndarray,
        class_names: list[str],
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        augment: bool = True,
        n_augment: int = 4,
    ) -> dict[str, list[float]]:
        """Train the network on labelled spectral data.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_points)``.
        y_label:
            Integer class labels, shape ``(n_samples,)``.
        y_conc:
            Concentration targets in ppm, shape ``(n_samples,)``.
        class_names:
            List of gas names, e.g. ``["Ethanol", "IPA", "Methanol", "MixVOC"]``.
        epochs, batch_size, lr:
            Standard training hyperparameters.
        augment:
            If ``True`` (default), applies spectral augmentation before
            training to expand small datasets via noise, scaling, and
            pixel-shift jitter (see :func:`augment_spectra`).
        n_augment:
            Copies per original spectrum when augmentation is enabled (4 →
            5× dataset size including originals).

        Returns
        -------
        dict
            ``{"loss": [...], "cls_acc": [...], "lr": [...]}`` — per-epoch history.
        """
        _requires_torch("CNNGasClassifier.fit")

        self.class_map = {i: name for i, name in enumerate(class_names)}
        self.model = GasCNN(input_length=self.input_length, num_classes=len(class_names)).to(
            self.device
        )
        self.num_classes = len(class_names)

        # ── Spectral data augmentation ────────────────────────────────────
        # Expand small training sets via noise injection, intensity scaling,
        # and circular pixel shift.  Critical for n < 30 spectra/gas class.
        if augment and len(X) > 0:
            X_train, y_label_train, y_conc_train = augment_spectra(
                np.asarray(X, dtype=np.float32),
                np.asarray(y_label),
                np.asarray(y_conc, dtype=np.float32),
                n_augment=n_augment,
            )
            log.info(
                "Augmentation: %d → %d spectra (%dx expansion)",
                len(X),
                len(X_train),
                len(X_train) // max(len(X), 1),
            )
        else:
            X_train = np.asarray(X, dtype=np.float32)
            y_label_train = np.asarray(y_label)
            y_conc_train = np.asarray(y_conc, dtype=np.float32)

        X_t = self._prepare_X(X_train)
        y_cls_t = torch.tensor(np.asarray(y_label_train, dtype=np.int64), device=self.device)
        y_reg_t = torch.tensor(
            np.asarray(y_conc_train, dtype=np.float32), device=self.device
        ).unsqueeze(1)

        dataset = TensorDataset(X_t, y_cls_t, y_reg_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # ── Class-weighted loss ───────────────────────────────────────────
        # Compensates for class imbalance (unequal spectra per gas type).
        # Weight_i = n_total / (n_classes × n_i); inverse-frequency weighting.
        y_label_arr = np.asarray(y_label_train, dtype=np.int64)
        n_classes = len(class_names)
        class_weights = np.ones(n_classes, dtype=np.float32)
        for cls_i in range(n_classes):
            n_cls: int = int(np.sum(y_label_arr == cls_i))
            if n_cls > 0:
                class_weights[cls_i] = len(y_label_arr) / (n_classes * n_cls)
        weight_tensor = torch.tensor(class_weights, device=self.device)

        # ── Optimizer: L2 regularisation via weight_decay ─────────────────
        # weight_decay=1e-4 prevents overfitting on small training sets
        # (typical: 4–5 concentrations × 3–5 trials = 12–25 spectra/gas).
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # ── Learning rate schedule: ReduceLROnPlateau ─────────────────────
        # Halves lr when training loss stagnates for 10 epochs.
        # Prevents convergence stall without manual lr tuning.
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
        )

        cls_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        reg_loss_fn = nn.MSELoss()

        history: dict[str, list[float]] = {"loss": [], "cls_acc": [], "lr": []}

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for xb, yb_cls, yb_reg in loader:
                optimizer.zero_grad()
                logits, conc_pred = self.model(xb)
                loss = cls_loss_fn(logits, yb_cls) + reg_loss_fn(conc_pred, yb_reg)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb_cls).sum().item()
                total += len(xb)

            avg_loss = epoch_loss / max(total, 1)
            acc = correct / max(total, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            history["loss"].append(avg_loss)
            history["cls_acc"].append(acc)
            history["lr"].append(current_lr)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                log.info(
                    "Epoch %d/%d — loss=%.4f acc=%.3f lr=%.2e",
                    epoch + 1,
                    epochs,
                    avg_loss,
                    acc,
                    current_lr,
                )

        self.is_fitted = True
        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Classify spectra and estimate concentrations.

        Parameters
        ----------
        X:
            Spectra, shape ``(n_samples, n_points)``.

        Returns
        -------
        gas_names : List[str]
            Predicted gas type per spectrum.
        concentrations : ndarray, shape (n_samples,)
            Predicted concentration in ppm.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("CNNGasClassifier must be fitted before calling predict().")

        self.model.eval()
        with torch.no_grad():
            X_t = self._prepare_X(X)
            logits, conc_pred = self.model(X_t)
            class_indices = logits.argmax(dim=1).cpu().numpy()
            concentrations = conc_pred.squeeze(1).cpu().numpy()

        gas_names = [self.class_map.get(int(i), "unknown") for i in class_indices]
        return gas_names, concentrations.astype(float)

    def predict_with_uncertainty(
        self,
        spectrum: np.ndarray,
        n_samples: int = 30,
    ) -> tuple[str, float, float, float]:
        """Monte Carlo Dropout uncertainty estimate for a single spectrum.

        Runs *n_samples* stochastic forward passes with Dropout active (i.e.
        model stays in ``train()`` mode so dropout layers sample different
        masks each pass).  The spread of predictions measures **epistemic
        uncertainty** — how confident the model is given its training data.

        Parameters
        ----------
        spectrum:
            1-D intensity array (n_points,).
        n_samples:
            Number of Monte Carlo forward passes (30 is a good default;
            increase to 100 for publication-quality uncertainty estimates).

        Returns
        -------
        gas_name : str
            Majority-vote predicted gas class.
        conc_mean : float
            Mean concentration prediction across MC samples (ppm).
        conc_std : float
            Standard deviation of concentration predictions — epistemic
            uncertainty in ppm.  A large value means the model is unsure.
        cls_entropy : float
            Entropy of the averaged class-probability distribution [0, log K].
            Zero = completely certain; log(K) = maximally uncertain.
        """
        if not self.is_fitted or self.model is None:
            return "unknown", 0.0, 0.0, 0.0

        X_t = self._prepare_X(spectrum[np.newaxis, :] if spectrum.ndim == 1 else spectrum)

        cls_probs_list = []
        conc_list = []

        # Keep model in train() mode so Dropout layers remain active
        self.model.train()
        try:
            with torch.no_grad():
                for _ in range(n_samples):
                    logits, conc_pred = self.model(X_t)
                    probs = torch.softmax(logits, dim=1)
                    cls_probs_list.append(probs.cpu().numpy())
                    conc_list.append(float(conc_pred[0, 0].item()))
        finally:
            self.model.eval()  # always restore eval mode

        # Aggregate MC samples
        cls_probs_arr = np.stack(cls_probs_list, axis=0)  # (n_samples, 1, n_classes)
        mean_probs = cls_probs_arr.mean(axis=0)[0]  # (n_classes,)
        top_idx = int(np.argmax(mean_probs))
        gas_name = self.class_map.get(top_idx, "unknown")

        conc_arr = np.array(conc_list, dtype=float)
        conc_mean = float(conc_arr.mean())
        conc_std = float(conc_arr.std())

        # Shannon entropy of averaged class distribution
        eps = 1e-9
        cls_entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))

        return gas_name, conc_mean, conc_std, cls_entropy

    def predict_single(self, spectrum: np.ndarray) -> tuple[str, float, float]:
        """Predict gas type, concentration, and confidence for one spectrum.

        Returns
        -------
        gas_name : str
        concentration_ppm : float
        confidence : float
            Softmax probability of the top-1 prediction [0, 1].
        """
        if not self.is_fitted or self.model is None:
            return "unknown", 0.0, 0.0

        self.model.eval()
        with torch.no_grad():
            X_t = self._prepare_X(spectrum[np.newaxis, :] if spectrum.ndim == 1 else spectrum)
            logits, conc_pred = self.model(X_t)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = probs.max(dim=1)
            idx = int(top_idx[0].item())
            conf = float(top_prob[0].item())
            conc = float(conc_pred[0, 0].item())

        return self.class_map.get(idx, "unknown"), conc, conf

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and metadata to *path* (``.pt`` checkpoint).

        Parameters
        ----------
        path:
            Destination path, e.g. ``"models/registry/cnn_classifier.pt"``.
        """
        _requires_torch("CNNGasClassifier.save")
        if self.model is None:
            raise RuntimeError("No model to save — call fit() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_length": self.input_length,
                "num_classes": self.num_classes,
                "class_map": self.class_map,
                "is_fitted": self.is_fitted,
            },
            path,
        )
        log.info("CNN checkpoint saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> CNNGasClassifier:
        """Load a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path:
            Path to a ``.pt`` file.
        device:
            Torch device (``"cpu"``, ``"cuda"``, or ``"auto"``).

        Returns
        -------
        CNNGasClassifier
            Ready-to-predict instance.
        """
        _requires_torch("CNNGasClassifier.load")

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls(
            input_length=ckpt["input_length"],
            num_classes=ckpt["num_classes"],
            device=device,
        )
        obj.model = GasCNN(
            input_length=ckpt["input_length"],
            num_classes=ckpt["num_classes"],
        ).to(obj.device)
        obj.model.load_state_dict(ckpt["model_state_dict"])
        obj.model.eval()
        obj.class_map = ckpt.get("class_map", {})
        obj.is_fitted = ckpt.get("is_fitted", True)
        log.info("CNN loaded from %s (device=%s)", path, obj.device)
        return obj
