"""
Model Registry
--------------
Central loader for pre-trained models used in real-time inference.
Handles CNN gas classifier, GPR calibration, and calibration JSON files.

All loads are optional — the system degrades gracefully if models are absent:
  - No CNN:  gas_type reported as "unknown"
  - No GPR:  concentration uses slope/intercept only (no uncertainty)
  - No calibration JSON: uses PipelineConfig defaults (0.116 nm/ppm)

Calibration search priority:
  1. output/models/calibration_params.json  (manually curated)
  2. Latest output/calibration_memory_*.json  (written by deployable/sensor mode)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Loads and holds pre-trained models for real-time inference."""

    def __init__(self) -> None:
        self.cnn_classifier = None  # CNNGasClassifier instance or None
        self.gpr_model = None  # GPRCalibration instance or None
        self.calibration_params: dict[str, Any] | None = None
        self._models_dir: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self, model_dir: str) -> dict[str, bool]:
        """
        Load all available models from *model_dir*.

        Expected files (all optional):
            cnn_classifier.pt         - PyTorch CNN checkpoint
            gpr_calibration.joblib    - scikit-learn GPR model
            calibration_params.json   - slope/intercept/reference_wavelength

        Args:
            model_dir: Directory containing model files.

        Returns:
            Dict with keys ``'cnn'``, ``'gpr'``, ``'calibration'`` and bool values.
        """
        self._models_dir = Path(model_dir)
        status = {"cnn": False, "gpr": False, "calibration": False}

        # ---- CNN classifier ----
        cnn_path = self._models_dir / "cnn_classifier.pt"
        if cnn_path.exists():
            try:
                from gas_analysis.core.intelligence.classifier import CNNGasClassifier

                self.cnn_classifier = CNNGasClassifier.load(str(cnn_path))
                status["cnn"] = True
                logger.info("CNN classifier loaded from %s", cnn_path)
            except Exception as exc:
                logger.warning("CNN load failed: %s", exc)

        # ---- GPR model ----
        gpr_path = self._models_dir / "gpr_calibration.joblib"
        if gpr_path.exists():
            try:
                import joblib

                self.gpr_model = joblib.load(str(gpr_path))
                status["gpr"] = True
                logger.info("GPR model loaded from %s", gpr_path)
            except Exception as exc:
                logger.warning("GPR load failed: %s", exc)

        # ---- Calibration JSON ----
        calib_path = self._models_dir / "calibration_params.json"
        if not calib_path.exists():
            calib_path = self._find_latest_calibration_json()
        if calib_path is not None and calib_path.exists():
            try:
                with open(calib_path) as fh:
                    raw = json.load(fh)
                # Normalise keys from both the deployable JSON format and batch format
                self.calibration_params = self._normalise_calibration(raw)
                status["calibration"] = True
                logger.info("Calibration params loaded from %s", calib_path)
            except Exception as exc:
                logger.warning("Calibration JSON load failed: %s", exc)

        return status

    def has_cnn(self) -> bool:
        return self.cnn_classifier is not None

    def has_gpr(self) -> bool:
        return self.gpr_model is not None

    def predict_gas_type(self, intensities_normalized: np.ndarray) -> tuple[str, float]:
        """
        Predict gas type from a full spectrum.

        Args:
            intensities_normalized: Intensity array normalised to [0, 1] (shape: [n_pixels]).

        Returns:
            Tuple of (gas_name: str, confidence: float in [0, 1]).
        """
        if not self.has_cnn():
            return ("unknown", 0.0)

        try:
            import torch

            X = intensities_normalized.reshape(1, -1)
            self.cnn_classifier.model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = self.cnn_classifier.model(X_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]

            predicted_idx = int(np.argmax(probs))
            class_names = getattr(self.cnn_classifier, "class_names_", None) or []
            gas_name = class_names[predicted_idx] if class_names else f"class_{predicted_idx}"
            return (gas_name, float(probs[predicted_idx]))

        except Exception as exc:
            logger.debug("CNN predict error: %s", exc)
            return ("unknown", 0.0)

    def predict_concentration_gpr(
        self, wavelength_shift: float
    ) -> tuple[float | None, float | None]:
        """
        Predict concentration with uncertainty using GPR.

        Args:
            wavelength_shift: ROI wavelength shift in nm.

        Returns:
            Tuple of (concentration_ppm: float, uncertainty_ppm: float),
            or (None, None) if GPR is not loaded.
        """
        if not self.has_gpr():
            return (None, None)

        try:
            X = np.array([[wavelength_shift]])
            mean, std = self.gpr_model.predict(X, return_std=True)
            return (float(mean[0]), float(std[0]))
        except Exception as exc:
            logger.debug("GPR predict error: %s", exc)
            return (None, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_latest_calibration_json(self) -> Path | None:
        """Scan output/ for the most recent calibration_memory JSON."""
        output_dir = Path("output")
        if not output_dir.exists():
            return None
        calib_files = sorted(
            output_dir.glob("calibration_memory_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return calib_files[0] if calib_files else None

    @staticmethod
    def _normalise_calibration(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Normalise calibration dict from multiple possible formats to a
        common schema:  {slope, intercept, reference_wavelength, r_squared}.
        """
        # Format 1: deployable / calibration_memory format
        if "current_calibration" in raw:
            cc = raw["current_calibration"]
            if not isinstance(cc, dict):
                # current_calibration is null or non-dict — skip this format
                pass
            else:
                return {
                    "slope": cc.get("slope", 0.116),
                    "intercept": cc.get("intercept", 0.0),
                    "reference_wavelength": cc.get("reference_wavelength", 531.5),
                    "r_squared": cc.get("r_squared", 0.0),
                }

        # Format 2: batch pipeline aggregated format
        if "wavelength_shift_slope" in raw:
            return {
                "slope": raw.get("wavelength_shift_slope", 0.116),
                "intercept": raw.get("wavelength_shift_intercept", 0.0),
                "reference_wavelength": raw.get("baseline_wavelength", 531.5),
                "r_squared": raw.get("r2_loocv", 0.0),
            }

        # Format 3: already normalised (manual / calibration_params.json)
        return raw
