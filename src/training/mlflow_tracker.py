"""
src.training.mlflow_tracker
============================
MLflow experiment tracking wrapper for Au-MIP LSPR training runs.

All training scripts should use ``ExperimentTracker`` — never call
``mlflow.*`` directly in training code.  This keeps tracking logic in
one place and makes it easy to swap backends later.

Quick start
-----------
::

    from src.training.mlflow_tracker import ExperimentTracker

    with ExperimentTracker("ethanol_calibration") as tracker:
        tracker.log_params({"gas_type": "Ethanol", "model": "GPR"})
        # ... train model ...
        tracker.log_metrics({"r2": 0.997, "rmse_ppm": 0.043, "lod_ppm": 0.08})
        tracker.log_model(gpr_model, "gpr_Ethanol")

After a run, view results with::

    mlflow ui --backend-store-uri experiments/mlruns
    # Open http://localhost:5000
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Default tracking directory (relative to project root)
_DEFAULT_TRACKING_URI = "experiments/mlruns"
_DEFAULT_EXPERIMENT = "AuMIP_LSPR_Gas_Sensing"


def _hash_file(path: str | Path) -> str:
    """Return SHA-256 hex digest of a file (for dataset fingerprinting)."""
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()[:16]


class ExperimentTracker:
    """Context-manager wrapper around MLflow runs.

    Parameters
    ----------
    run_name:
        Human-readable name for this run (e.g. ``"GPR_Ethanol_v3"``).
    experiment_name:
        MLflow experiment to group this run under.
    tracking_uri:
        Path to the MLflow backend store.  Defaults to
        ``experiments/mlruns`` (local directory).
    tags:
        Extra key-value tags attached to the run for easy filtering.
    """

    def __init__(
        self,
        run_name: str = "unnamed_run",
        experiment_name: str = _DEFAULT_EXPERIMENT,
        tracking_uri: str = _DEFAULT_TRACKING_URI,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self._run: Any = None
        self._mlflow: Any = None
        self._active = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> ExperimentTracker:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            if self._mlflow and self._active:
                self._mlflow.end_run(status="FAILED")
                self._active = False
        else:
            self.end()
        # Return None (falsy) → do not suppress exceptions

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start (or resume) an MLflow run."""
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError:
            log.warning(
                "mlflow not installed — tracking disabled. Install with: pip install mlflow"
            )
            return

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        default_tags = {"project": "Au-MIP-LSPR", "lab": "Chula"}
        default_tags.update(self.tags)

        self._run = mlflow.start_run(run_name=self.run_name, tags=default_tags)
        self._active = True
        log.info(
            "MLflow run started: %s (run_id=%s)",
            self.run_name,
            self._run.info.run_id,
        )

    def end(self) -> None:
        """End the active MLflow run with status FINISHED."""
        if self._mlflow and self._active:
            self._mlflow.end_run(status="FINISHED")
            self._active = False
            log.info("MLflow run ended: %s", self.run_name)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        if not self._active:
            return
        # MLflow param values must be strings
        str_params = {k: str(v) for k, v in params.items()}
        self._mlflow.log_params(str_params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dictionary of scalar metrics.

        Parameters
        ----------
        metrics:
            e.g. ``{"r2": 0.997, "rmse_ppm": 0.043, "lod_ppm": 0.08}``
        step:
            Optional training step (epoch number for per-epoch logging).
        """
        if not self._active:
            return
        clean = {k: float(v) for k, v in metrics.items() if np.isfinite(float(v))}
        if step is not None:
            for k, v in clean.items():
                self._mlflow.log_metric(k, v, step=step)
        else:
            self._mlflow.log_metrics(clean)

    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log a scikit-learn or PyTorch model as an MLflow artifact.

        Automatically detects model type (sklearn vs torch).
        """
        if not self._active:
            return
        try:
            # PyTorch
            import torch

            if isinstance(model, torch.nn.Module):
                self._mlflow.pytorch.log_model(model, artifact_path)
                log.info("Logged PyTorch model as '%s'.", artifact_path)
                return
        except ImportError:
            pass

        try:
            # scikit-learn / joblib-compatible
            self._mlflow.sklearn.log_model(model, artifact_path)
            log.info("Logged sklearn model as '%s'.", artifact_path)
        except Exception:
            # Generic: save with joblib and log as artifact
            import tempfile

            import joblib

            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                joblib.dump(model, tmp.name)
                self._mlflow.log_artifact(tmp.name, artifact_path)
                os.unlink(tmp.name)
            log.info("Logged model artifact as '%s'.", artifact_path)

    def log_figure(self, fig: Any, filename: str) -> None:
        """Log a matplotlib figure as an artifact.

        Parameters
        ----------
        fig:
            matplotlib ``Figure`` object.
        filename:
            Filename inside the MLflow artifact store (e.g. ``"calibration_curve.png"``).
        """
        if not self._active:
            return
        try:
            import tempfile

            import matplotlib.pyplot as plt

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
                self._mlflow.log_artifact(tmp.name, filename)
                os.unlink(tmp.name)
            plt.close(fig)
        except Exception as exc:
            log.warning("Failed to log figure '%s': %s", filename, exc)

    def log_dataset_info(
        self,
        gas_type: str,
        n_samples: int,
        concentrations: list[float],
        data_paths: list[str | Path] | None = None,
    ) -> None:
        """Log dataset metadata (gas type, sample count, concentrations).

        Parameters
        ----------
        gas_type:
            Gas analyte label (e.g. ``"Ethanol"``).
        n_samples:
            Total number of spectra used for training.
        concentrations:
            List of unique concentration levels (ppm).
        data_paths:
            Optional list of source file paths for dataset fingerprinting.
        """
        if not self._active:
            return
        self.log_params(
            {
                "gas_type": gas_type,
                "n_samples": n_samples,
                "n_concentrations": len(concentrations),
                "concentration_min_ppm": min(concentrations) if concentrations else 0,
                "concentration_max_ppm": max(concentrations) if concentrations else 0,
                "concentrations": json.dumps(sorted(concentrations)),
            }
        )
        if data_paths:
            hashes = [_hash_file(p) for p in data_paths if Path(p).exists()]
            if hashes:
                self.log_params({"dataset_fingerprint": ",".join(hashes[:5])})

    def log_calibration_results(
        self,
        r2: float,
        rmse_ppm: float,
        lod_ppm: float | None,
        slope: float | None = None,
        slope_se: float | None = None,
        intercept: float | None = None,
        loq_ppm: float | None = None,
        lod_ppm_ci_lower: float | None = None,
        lod_ppm_ci_upper: float | None = None,
        loq_ppm_ci_lower: float | None = None,
        loq_ppm_ci_upper: float | None = None,
        noise_std: float | None = None,
        per_concentration_r2: dict[float, float] | None = None,
    ) -> None:
        """Log ICH Q2(R1)-compliant calibration quality metrics.

        Parameters
        ----------
        r2, rmse_ppm:
            Standard regression metrics.
        lod_ppm, loq_ppm:
            Limit of detection/quantification (point estimates, ppm).
        slope, slope_se:
            Calibration curve slope and its standard error (OLS covariance).
        lod_ppm_ci_lower, lod_ppm_ci_upper:
            Bootstrap 95 % CI bounds for LOD (ppm).
        loq_ppm_ci_lower, loq_ppm_ci_upper:
            Bootstrap 95 % CI bounds for LOQ (ppm).
        noise_std:
            Baseline noise standard deviation used for LOD calculation.
        per_concentration_r2:
            Optional dict mapping concentration (ppm) → per-level R².
            Logs as ``r2_at_{conc}ppm`` metrics.
        """
        import math

        metrics: dict[str, float] = {"r2": r2, "rmse_ppm": rmse_ppm}

        # Core calibration metrics
        if lod_ppm is not None and math.isfinite(lod_ppm):
            metrics["lod_ppm"] = lod_ppm
        if loq_ppm is not None and math.isfinite(loq_ppm):
            metrics["loq_ppm"] = loq_ppm
        if slope is not None and math.isfinite(slope):
            metrics["calibration_slope"] = slope
        if slope_se is not None and math.isfinite(slope_se):
            metrics["calibration_slope_se"] = slope_se
        if intercept is not None and math.isfinite(intercept):
            metrics["calibration_intercept"] = intercept
        if noise_std is not None and math.isfinite(noise_std):
            metrics["noise_std"] = noise_std

        # Bootstrap CI bounds
        if lod_ppm_ci_lower is not None and math.isfinite(lod_ppm_ci_lower):
            metrics["lod_ppm_ci_lower"] = lod_ppm_ci_lower
        if lod_ppm_ci_upper is not None and math.isfinite(lod_ppm_ci_upper):
            metrics["lod_ppm_ci_upper"] = lod_ppm_ci_upper
        if loq_ppm_ci_lower is not None and math.isfinite(loq_ppm_ci_lower):
            metrics["loq_ppm_ci_lower"] = loq_ppm_ci_lower
        if loq_ppm_ci_upper is not None and math.isfinite(loq_ppm_ci_upper):
            metrics["loq_ppm_ci_upper"] = loq_ppm_ci_upper

        self.log_metrics(metrics)

        # Per-concentration R² (logged separately so they appear as individual metrics)
        if per_concentration_r2:
            conc_metrics: dict[str, float] = {}
            for conc, r2_c in per_concentration_r2.items():
                if math.isfinite(r2_c):
                    key = f"r2_at_{conc:.1f}ppm".replace(".", "p")
                    conc_metrics[key] = r2_c
            if conc_metrics:
                self.log_metrics(conc_metrics)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log a local file (JSON, CSV, image, etc.) as an MLflow artifact.

        Parameters
        ----------
        local_path:
            Path to the file on disk.
        artifact_path:
            Optional sub-directory inside the MLflow artifact store.
        """
        if not self._active:
            return
        try:
            self._mlflow.log_artifact(str(local_path), artifact_path)
            log.info("Logged artifact: %s", local_path)
        except Exception as exc:
            log.warning("Failed to log artifact '%s': %s", local_path, exc)

    @property
    def run_id(self) -> str | None:
        """Return the active MLflow run ID, or ``None`` if not started."""
        if self._run is None:
            return None
        return str(self._run.info.run_id)
