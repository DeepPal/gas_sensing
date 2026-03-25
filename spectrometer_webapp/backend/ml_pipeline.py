"""
ML Pipeline for Optical Spectrometer Gas Analysis
==================================================
Implements configurable preprocessing → feature extraction → classification.

The IDENTICAL preprocessing config is stored inside every saved model artifact
so that the inference path is always consistent with the training path.

Preprocessing chain (per spectrum):
  1. Denoising         — Savitzky-Golay | moving-average | none
  2. Differential sig  — ΔI = I_sample − I_reference  (LSPR sensors)
  3. Baseline removal  — ALS | min-subtract | none
  4. Normalization      — MinMax | Z-score | none

Feature extraction:
  • LSPR mode (reference set + use_lspr=True): [Δλ, ΔI_peak, ΔI_area, ΔI_std]
  • Standard mode: full spectrum vector → compressed by PCA downstream
"""

import time
import joblib
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── ALS Baseline Correction ────────────────────────────────────────────────────
def _als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                  max_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens, 2005).

    D has shape (n-2, n) so that D.T @ D is (n, n) — same size as W = diag(w).
    np.diff(np.eye(n), 2) produces (n, n-2), so we transpose it.
    """
    n = len(y)
    w = np.ones(n)
    D = np.diff(np.eye(n), 2).T   # (n-2, n)  ← correct second-difference matrix
    H = lam * D.T @ D             # (n, n)     ← matches diag(w)
    for _ in range(max_iter):
        W = np.diag(w)
        z = np.linalg.solve(W + H, w * y)
        w = np.where(y > z, p, 1.0 - p)
    return z


# ── Preprocessing Configuration ────────────────────────────────────────────────
@dataclass
class PreprocessingConfig:
    denoising: str    = "savgol"    # "savgol" | "moving_avg" | "none"
    savgol_window: int = 11
    savgol_poly: int   = 3
    baseline: str      = "als"      # "als" | "min_subtract" | "none"
    normalization: str = "minmax"   # "minmax" | "zscore" | "none"
    use_pca: bool      = True
    pca_components: int = 10
    use_lspr_features: bool = False  # requires reference spectrum


# ── Main Pipeline Class ─────────────────────────────────────────────────────────
class MLPipeline:
    def __init__(self, models_dir: str = "../data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.current_model: Optional[Pipeline] = None
        self.current_model_name: Optional[str] = None
        self.current_metadata: dict = {}
        self.config = PreprocessingConfig()

        # Reference spectrum (for LSPR differential signal)
        self._ref_wl:       Optional[np.ndarray] = None
        self._ref_int:      Optional[np.ndarray] = None
        self._ref_peak_wl:  Optional[float]      = None

    # ─────────────────────────────────────────────────────────────────────────
    # Reference spectrum
    # ─────────────────────────────────────────────────────────────────────────
    def set_reference(self, wavelengths: List[float], intensities: List[float]) -> None:
        """Set air/blank reference for LSPR Δλ computation."""
        self._ref_wl      = np.asarray(wavelengths,  dtype=float)
        self._ref_int     = np.asarray(intensities,  dtype=float)
        self._ref_peak_wl = float(self._ref_wl[np.argmax(self._ref_int)])

    def clear_reference(self) -> None:
        self._ref_wl = self._ref_int = self._ref_peak_wl = None

    # ─────────────────────────────────────────────────────────────────────────
    # Single-spectrum preprocessing
    # ─────────────────────────────────────────────────────────────────────────
    def preprocess_one(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        cfg: Optional[PreprocessingConfig] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess one spectrum.

        Returns
        -------
        wl         : wavelength axis (unchanged)
        sig        : preprocessed intensity
        diff_signal: ΔI = raw − reference (None if no reference loaded)
        """
        if cfg is None:
            cfg = self.config

        wl  = np.asarray(wavelengths, dtype=float)
        sig = np.asarray(intensities, dtype=float).copy()

        # 1. Denoising
        if cfg.denoising == "savgol":
            wlen = min(cfg.savgol_window, len(sig) - 1)
            if wlen % 2 == 0:
                wlen -= 1
            wlen = max(3, wlen)
            sig = savgol_filter(sig, window_length=wlen, polyorder=min(cfg.savgol_poly, wlen - 1))
        elif cfg.denoising == "moving_avg":
            sig = np.convolve(sig, np.ones(5) / 5, mode="same")

        # 2. Differential signal (before baseline so ΔI preserves the shift)
        diff_signal = None
        if self._ref_int is not None:
            ref_interp  = np.interp(wl, self._ref_wl, self._ref_int)
            diff_signal = sig - ref_interp

        # 3. Baseline removal
        if cfg.baseline == "als":
            sig = sig - _als_baseline(sig)
        elif cfg.baseline == "min_subtract":
            sig = sig - np.min(sig)

        # 4. Normalization
        if cfg.normalization == "minmax":
            rng = sig.max() - sig.min()
            sig = (sig - sig.min()) / rng if rng > 1e-10 else sig * 0.0
        elif cfg.normalization == "zscore":
            std = sig.std()
            sig = (sig - sig.mean()) / (std if std > 1e-10 else 1.0)

        return wl, sig, diff_signal

    # ─────────────────────────────────────────────────────────────────────────
    # Feature extraction
    # ─────────────────────────────────────────────────────────────────────────
    def extract_features(
        self,
        wavelengths: np.ndarray,
        sig_processed: np.ndarray,
        diff_signal: Optional[np.ndarray] = None,
        cfg: Optional[PreprocessingConfig] = None,
    ) -> np.ndarray:
        """
        Extract feature vector.

        LSPR mode (diff available + use_lspr_features=True):
            [Δλ(nm), ΔI_peak, ΔI_area, ΔI_std]  — 4 physics-grounded features

        Standard mode:
            full preprocessed spectrum → PCA compression happens in sklearn Pipeline
        """
        if cfg is None:
            cfg = self.config

        wl  = np.asarray(wavelengths,   dtype=float)
        sig = np.asarray(sig_processed, dtype=float)

        if cfg.use_lspr_features and diff_signal is not None and self._ref_peak_wl is not None:
            d = np.asarray(diff_signal, dtype=float)
            peak_idx  = int(np.argmax(np.abs(d)))
            delta_lam = float(wl[peak_idx]) - self._ref_peak_wl
            return np.array([delta_lam, float(d[peak_idx]),
                             float(np.trapz(d, wl)), float(np.std(d))])
        else:
            return sig  # full spectrum; PCA compresses in the sklearn pipeline

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    def train_model(
        self,
        X_raw:       np.ndarray,
        y_labels:    np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        model_type:  str  = "RandomForest",
        config:      Optional[dict] = None,
        test_size:   float = 0.2,
    ) -> dict:
        """
        Train a gas-type classifier.

        Parameters
        ----------
        X_raw       : (N, L) raw intensity arrays (one row per spectrum)
        y_labels    : (N,)   gas-type strings
        wavelengths : (L,)   common wavelength axis
        model_type  : "RandomForest" | "SVM" | "LogisticRegression"
        config      : dict of PreprocessingConfig fields (overrides current config)
        test_size   : fraction held out for validation

        Returns
        -------
        dict with accuracy, confusion_matrix, classes, classification_report, model_name
        """
        # Apply config overrides
        if config:
            for k, v in config.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, type(getattr(self.config, k))(v))

        classes = sorted(set(y_labels))
        if len(classes) < 2:
            return {"error": "Need at least 2 different gas classes to train."}

        wl = (wavelengths if wavelengths is not None
              else np.linspace(300, 1000, X_raw.shape[1]))

        # ── Feature matrix ────────────────────────────────────────────────────
        X_feats = []
        for row in X_raw:
            _, sig_proc, diff = self.preprocess_one(wl, row)
            feat = self.extract_features(wl, sig_proc, diff)
            X_feats.append(feat)
        X = np.array(X_feats)

        # ── sklearn Pipeline ──────────────────────────────────────────────────
        steps = [("scaler", StandardScaler())]
        lspr_active = self.config.use_lspr_features and self._ref_int is not None
        if self.config.use_pca and not lspr_active:
            n_comp = max(1, min(self.config.pca_components, len(X) - 1, X.shape[1]))
            steps.append(("pca", PCA(n_components=n_comp)))

        if model_type == "RandomForest":
            steps.append(("clf", RandomForestClassifier(n_estimators=100, random_state=42)))
        elif model_type == "SVM":
            steps.append(("clf", SVC(kernel="rbf", probability=True, C=10, random_state=42)))
        elif model_type == "LogisticRegression":
            steps.append(("clf", LogisticRegression(max_iter=1000, random_state=42)))
        else:
            return {"error": f"Unknown model type: {model_type}"}

        pipeline = Pipeline(steps)

        # ── Train / validation split ──────────────────────────────────────────
        y_int = np.array([classes.index(g) for g in y_labels])
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_int, test_size=test_size, random_state=42, stratify=y_int)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y_int, test_size=test_size, random_state=42)

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        acc    = float(accuracy_score(y_te, y_pred))
        cm     = confusion_matrix(y_te, y_pred, labels=list(range(len(classes)))).tolist()
        report = classification_report(y_te, y_pred, target_names=classes, output_dict=True)

        # ── Save versioned artifact ───────────────────────────────────────────
        ts         = int(time.time())
        model_name = f"{model_type}_acc{int(acc * 100)}_{ts}.joblib"
        metadata = {
            "model_type":   model_type,
            "accuracy":     acc,
            "classes":      classes,
            "n_train":      int(len(X_tr)),
            "n_test":       int(len(X_te)),
            "trained_at":   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "preprocessing": asdict(self.config),
            "has_reference": self._ref_int is not None,
            "ref_peak_wl":  self._ref_peak_wl,
        }
        artifact = {
            "pipeline":    pipeline,
            "metadata":    metadata,
            "config":      asdict(self.config),
            "ref_wl":      self._ref_wl.tolist()  if self._ref_wl  is not None else None,
            "ref_int":     self._ref_int.tolist() if self._ref_int is not None else None,
            "ref_peak_wl": self._ref_peak_wl,
        }
        joblib.dump(artifact, self.models_dir / model_name)

        self.current_model      = pipeline
        self.current_model_name = model_name
        self.current_metadata   = metadata

        return {
            "status":           "success",
            "model_name":       model_name,
            "accuracy":         acc,
            "report":           report,
            "classes":          classes,
            "confusion_matrix": cm,
            "metadata":         metadata,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────
    def load_model(self, model_name: str) -> bool:
        model_path = self.models_dir / model_name
        if not model_path.exists():
            return False

        artifact = joblib.load(model_path)

        if isinstance(artifact, dict):
            self.current_model    = artifact["pipeline"]
            self.current_metadata = artifact.get("metadata", {})
            # Restore preprocessing config
            for k, v in artifact.get("config", {}).items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            # Restore reference spectrum
            if artifact.get("ref_wl") and artifact.get("ref_int"):
                self._ref_wl      = np.asarray(artifact["ref_wl"])
                self._ref_int     = np.asarray(artifact["ref_int"])
                self._ref_peak_wl = artifact.get("ref_peak_wl")
        else:
            # Legacy bare pipeline
            self.current_model    = artifact
            self.current_metadata = {}

        self.current_model_name = model_name
        return True

    def get_available_models(self) -> List[str]:
        return sorted([f.name for f in self.models_dir.glob("*.joblib")], reverse=True)

    def get_model_info(self, model_name: str) -> dict:
        model_path = self.models_dir / model_name
        if not model_path.exists():
            return {}
        artifact = joblib.load(model_path)
        return artifact.get("metadata", {}) if isinstance(artifact, dict) else {}

    # ─────────────────────────────────────────────────────────────────────────
    # Real-time inference  (SAME pipeline as training)
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, wavelengths: List[float], intensities: List[float]) -> dict:
        if not self.current_model:
            return {"error": "No model loaded"}

        wl  = np.asarray(wavelengths,  dtype=float)
        raw = np.asarray(intensities,  dtype=float)
        classes = self.current_metadata.get("classes", [])

        _, sig_proc, diff = self.preprocess_one(wl, raw)
        feat  = self.extract_features(wl, sig_proc, diff)
        X_in  = feat.reshape(1, -1)

        pred_idx   = int(self.current_model.predict(X_in)[0])
        pred_label = (classes[pred_idx]
                      if classes and pred_idx < len(classes) else str(pred_idx))

        try:
            probs       = self.current_model.predict_proba(X_in)[0]
            confidence  = float(max(probs))
            probs_dict  = {classes[i]: round(float(p), 4)
                           for i, p in enumerate(probs) if i < len(classes)}
        except Exception:
            confidence = 1.0
            probs_dict = {}

        return {
            "prediction":    pred_label,
            "confidence":    confidence,
            "probabilities": probs_dict,
            "model":         self.current_model_name,
        }
