"""
src.models.registry
===================
Central model registry — loads CNN, GPR, and calibration artefacts from the
``models/registry/`` directory and exposes a unified prediction interface.

All model components are **optional**.  The registry degrades gracefully:

- No CNN  → ``predict_gas_type`` returns ``("unknown", 0.0)``
- No GPR  → ``predict_concentration_gpr`` returns ``(None, None)``
- No calibration JSON → linear fallback from pipeline config

This design lets the inference API serve requests even on a clean install
with no trained models, returning sensible defaults instead of crashing.

Public API
----------
- ``ModelRegistry``  — load_all / predict_gas_type / predict_concentration_gpr
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Optional heavy dependencies
try:
    from src.models.cnn import CNNGasClassifier

    _CNN_IMPORTABLE = True
except ImportError:
    _CNN_IMPORTABLE = False
    CNNGasClassifier = None  # type: ignore[assignment, misc]

try:
    from src.calibration.gpr import GPRCalibration

    _GPR_IMPORTABLE = True
except ImportError:
    _GPR_IMPORTABLE = False
    GPRCalibration = None  # type: ignore[assignment, misc]


class ModelRegistry:
    """Loads and manages all inference models for the sensing pipeline.

    Usage
    -----
    ::

        registry = ModelRegistry()
        status = registry.load_all("models/registry")
        # {"cnn": True, "gpr": False, "calibration": True}

        gas, conf = registry.predict_gas_type(intensities)
        conc, unc  = registry.predict_concentration_gpr(delta_lambda)
    """

    def __init__(self) -> None:
        self.cnn_classifier: CNNGasClassifier | None = None
        self.gpr_model: GPRCalibration | None = None
        self.calibration_params: dict[str, Any] | None = None
        self._models_dir: Path | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_all(self, model_dir: str) -> dict[str, bool]:
        """Load all available models from *model_dir*.

        Scans for:

        - ``cnn_classifier.pt``  → :class:`~src.models.cnn.CNNGasClassifier`
        - ``gpr_calibration.joblib`` → :class:`~src.calibration.gpr.GPRCalibration`
        - ``calibration_params.json`` → linear calibration parameters

        Parameters
        ----------
        model_dir:
            Path to the model registry directory.

        Returns
        -------
        dict
            ``{"cnn": bool, "gpr": bool, "calibration": bool}`` — which
            components were loaded successfully.
        """
        model_path = Path(model_dir)
        self._models_dir = model_path
        status: dict[str, bool] = {"cnn": False, "gpr": False, "calibration": False}

        # --- CNN ---
        cnn_path = model_path / "cnn_classifier.pt"
        if cnn_path.exists() and _CNN_IMPORTABLE:
            try:
                self.cnn_classifier = CNNGasClassifier.load(str(cnn_path))
                status["cnn"] = True
                log.info("CNN loaded from %s", cnn_path)
            except Exception as exc:
                log.warning("Failed to load CNN: %s", exc)

        # --- GPR ---
        gpr_path = model_path / "gpr_calibration.joblib"
        if gpr_path.exists() and _GPR_IMPORTABLE:
            try:
                self.gpr_model = GPRCalibration.load(str(gpr_path))
                status["gpr"] = True
                log.info("GPR loaded from %s", gpr_path)
            except Exception as exc:
                log.warning("Failed to load GPR: %s", exc)

        # --- Calibration JSON ---
        cal_path = model_path / "calibration_params.json"
        if cal_path.exists():
            try:
                with open(cal_path) as f:
                    raw = json.load(f)
                self.calibration_params = self._normalise_calibration(raw)
                status["calibration"] = True
                log.info("Calibration JSON loaded from %s", cal_path)
            except Exception as exc:
                log.warning("Failed to load calibration JSON: %s", exc)

        # Fallback: scan output/ for legacy calibration_memory_*.json
        if not status["calibration"]:
            legacy = self._find_latest_calibration_json()
            if legacy is not None:
                try:
                    with open(legacy) as f:
                        raw = json.load(f)
                    self.calibration_params = self._normalise_calibration(raw)
                    status["calibration"] = True
                    log.info("Calibration loaded from legacy path %s", legacy)
                except Exception as exc:
                    log.warning("Failed to load legacy calibration: %s", exc)

        return status

    def _find_latest_calibration_json(self) -> Path | None:
        """Scan ``output/`` for the most recent ``calibration_memory_*.json``."""
        candidates = list(Path("output").glob("calibration_memory_*.json"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _normalise_calibration(raw: dict[str, Any]) -> dict[str, Any]:
        """Normalise calibration JSON from multiple source formats.

        Handles:
        - **Deployable** format: ``{"current_calibration": {...}, ...}``
        - **Batch pipeline** format: ``{"wavelength_shift_slope": ..., ...}``
        """
        # Format 1: has "current_calibration" wrapper
        cc = raw.get("current_calibration")
        if isinstance(cc, dict):
            return {
                "slope": float(cc.get("slope", 0.116)),
                "intercept": float(cc.get("intercept", 0.0)),
                "reference_wavelength": float(cc.get("reference_wavelength", 531.5)),
                "r_squared": float(cc.get("r_squared", 0.0)),
            }

        # Format 2: flat batch pipeline keys
        slope_key = next(
            (k for k in raw if "slope" in k.lower() and "wavelength" in k.lower()), None
        )
        if slope_key:
            return {
                "slope": float(raw.get(slope_key, 0.116)),
                "intercept": float(raw.get("intercept", 0.0)),
                "reference_wavelength": float(raw.get("reference_wavelength", 531.5)),
                "r_squared": float(raw.get("r_squared", 0.0)),
            }

        # Format 3: already normalised
        if "slope" in raw:
            return {
                "slope": float(raw.get("slope", 0.116)),
                "intercept": float(raw.get("intercept", 0.0)),
                "reference_wavelength": float(raw.get("reference_wavelength", 531.5)),
                "r_squared": float(raw.get("r_squared", 0.0)),
            }

        return raw

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    def has_cnn(self) -> bool:
        """Return True if a CNN classifier is loaded and ready."""
        return self.cnn_classifier is not None and self.cnn_classifier.is_fitted

    def has_gpr(self) -> bool:
        """Return True if a GPR calibration model is loaded and ready."""
        return self.gpr_model is not None and self.gpr_model.is_fitted

    def has_calibration(self) -> bool:
        """Return True if linear calibration parameters are available."""
        return self.calibration_params is not None

    def models_loaded(self) -> dict[str, bool]:
        """Return a status dict for all three components."""
        return {
            "cnn": self.has_cnn(),
            "gpr": self.has_gpr(),
            "calibration": self.has_calibration(),
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_gas_type(self, intensities_normalized: np.ndarray) -> tuple[str, float]:
        """Classify the gas type from a spectrum.

        Parameters
        ----------
        intensities_normalized:
            Spectrum array, shape ``(n_points,)``.  Does not need to be
            pre-normalised — the CNN wrapper handles it internally.

        Returns
        -------
        gas_name : str
            Predicted gas type, or ``"unknown"`` if no CNN loaded.
        confidence : float
            Softmax confidence in ``[0, 1]``, or ``0.0`` if no CNN.
        """
        if not self.has_cnn():
            return "unknown", 0.0
        try:
            gas_name, _conc, confidence = self.cnn_classifier.predict_single(  # type: ignore[union-attr]
                intensities_normalized
            )
            return gas_name, float(confidence)
        except Exception as exc:
            log.debug("CNN prediction failed: %s", exc)
            return "unknown", 0.0

    def predict_concentration_gpr(
        self, wavelength_shift: float
    ) -> tuple[float | None, float | None]:
        """Estimate concentration from a wavelength shift using GPR.

        Parameters
        ----------
        wavelength_shift:
            Δλ = λ_gas − λ_reference in nm (typically negative).

        Returns
        -------
        concentration_ppm : float or None
        uncertainty_ppm : float or None
            One-sigma posterior std from the GPR, or ``None`` if no GPR.
        """
        if not self.has_gpr():
            return None, None
        try:
            return self.gpr_model.predict_single(wavelength_shift)  # type: ignore[union-attr]
        except Exception as exc:
            log.debug("GPR prediction failed: %s", exc)
            return None, None

    def get_calibration_slope(self) -> float:
        """Return the linear calibration slope (nm/ppm), default 0.116."""
        if self.calibration_params:
            return float(self.calibration_params.get("slope", 0.116))
        return 0.116

    def get_reference_wavelength(self) -> float:
        """Return the reference (zero-gas) peak wavelength in nm."""
        if self.calibration_params:
            return float(self.calibration_params.get("reference_wavelength", 531.5))
        return 531.5
