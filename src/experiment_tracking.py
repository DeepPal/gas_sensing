"""
src.experiment_tracking
=======================
MLflow-based experiment tracking for the SpectraAgent — Spectrometer-Based Sensing Platform.

Every model training run is logged as an MLflow experiment run, capturing:

- **Parameters**: model type, n_components, CV strategy, gas name, n_samples
- **Metrics**: R², RMSEC, Q², RMSECV, LOB, LOD (+ CI), LOQ, LOL, sensitivity
- **Artifacts**: fitted model pickle, VIP scores CSV, sensor metrics JSON,
  calibration figure (PNG)

The tracking server stores run history in a local SQLite DB (``mlruns.db``)
at the project root.
Launch the UI with::

    mlflow ui --port 5000
    # Open http://localhost:5000

Usage
-----
::

    from src.experiment_tracking import ExperimentTracker

    tracker = ExperimentTracker(experiment_name="LSPR_Ethanol")

    with tracker.start_run(run_name="GPR_run_1", tags={"gas": "Ethanol"}):
        # ... train model ...
        tracker.log_gpr_run(
            model=gpr_model,
            sensor_metrics=characterization_dict,
            y_concs=concentrations,
            X_features=feature_matrix,
        )

    # Later — in the dashboard
    runs = tracker.list_runs()   # DataFrame of all past runs
    best = tracker.best_run(metric="metrics.r_squared")

Architecture
------------
``ExperimentTracker`` is a thin wrapper around the MLflow Python API.  It
adds domain-specific helpers for the LSPR calibration workflow without
locking the rest of the codebase to MLflow-specific calls.
"""

from __future__ import annotations

from collections.abc import Generator
import contextlib
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy MLflow import — graceful fallback if not installed
# ---------------------------------------------------------------------------

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

# Path to the SQLite tracking database (mlflow 3.x recommended backend)
_DEFAULT_TRACKING_DB = Path(__file__).resolve().parents[1] / "mlruns.db"


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """MLflow experiment tracker for LSPR calibration runs.

    Parameters
    ----------
    experiment_name :
        MLflow experiment name.  All runs are grouped under this name.
        Defaults to ``"LSPR_Gas_Sensing"``.
    tracking_uri :
        MLflow tracking URI.  Defaults to a local SQLite DB
        (``sqlite:///.../mlruns.db``) at the project root. Pass
        ``"http://localhost:5000"`` to use a remote tracking server.
    """

    def __init__(
        self,
        experiment_name: str = "LSPR_Gas_Sensing",
        tracking_uri: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.available = _MLFLOW_AVAILABLE

        if not self.available:
            log.warning(
                "MLflow not installed.  Experiment tracking disabled.  "
                "Install with:  pip install mlflow"
            )
            return

        # Set tracking URI — SQLite backend (avoids mlruns/ file-store deprecation)
        uri = tracking_uri or f"sqlite:///{_DEFAULT_TRACKING_DB.as_posix()}"
        mlflow.set_tracking_uri(uri)

        # Create or get experiment
        existing = mlflow.get_experiment_by_name(experiment_name)
        if existing is None:
            self._experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self._experiment_id = existing.experiment_id

        log.debug(
            "ExperimentTracker: experiment=%r id=%s uri=%s",
            experiment_name, self._experiment_id, uri,
        )

    # ------------------------------------------------------------------
    # Context manager — wraps an MLflow run
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Generator[ExperimentTracker, None, None]:
        """Context manager that wraps code in an MLflow run.

        Example
        -------
        ::

            with tracker.start_run(run_name="GPR_v3"):
                tracker.log_gpr_run(model, metrics, ...)
        """
        if not self.available:
            yield self
            return

        with mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name,
            tags=tags or {},
        ):
            yield self

    # ------------------------------------------------------------------
    # GPR run logging
    # ------------------------------------------------------------------

    def log_gpr_run(
        self,
        model: Any,
        sensor_metrics: dict[str, Any],
        y_concs: np.ndarray,
        X_features: np.ndarray,
        gas_name: str = "unknown",
        n_components: int | None = None,
        cal_figure_path: str | Path | None = None,
    ) -> str | None:
        """Log a GPR calibration run to MLflow.

        Parameters
        ----------
        model :
            Fitted GPR model (sklearn-compatible).
        sensor_metrics :
            Dict from :func:`~src.reporting.metrics.compute_comprehensive_sensor_characterization`.
        y_concs :
            Training concentrations.
        X_features :
            Feature matrix used for training.
        gas_name :
            Analyte name.
        n_components :
            Not used for GPR; kept for API symmetry.
        cal_figure_path :
            Optional path to a calibration figure PNG to attach as artifact.

        Returns
        -------
        str or None
            MLflow run ID, or ``None`` if tracking is unavailable.
        """
        if not self.available:
            return None

        run = mlflow.active_run()
        if run is None:
            log.warning("ExperimentTracker.log_gpr_run called outside a start_run() context")
            return None

        # ── Parameters ──────────────────────────────────────────────────
        mlflow.log_params({
            "model_type": "GPR",
            "gas_name": gas_name,
            "n_training_samples": int(len(y_concs)),
            "n_features": int(X_features.shape[1]) if X_features.ndim == 2 else 1,
        })

        # ── Metrics ─────────────────────────────────────────────────────
        self._log_sensor_metrics(sensor_metrics)

        # ── Model artifact ──────────────────────────────────────────────
        try:
            mlflow.sklearn.log_model(model, name="model")
        except Exception as exc:
            log.warning("Could not log GPR model artifact: %s", exc)

        # ── Sensor metrics JSON ─────────────────────────────────────────
        self._log_metrics_json(sensor_metrics, "sensor_metrics.json")

        # ── Calibration figure ──────────────────────────────────────────
        if cal_figure_path and Path(cal_figure_path).exists():
            mlflow.log_artifact(str(cal_figure_path), artifact_path="figures")

        run_id: str = run.info.run_id
        log.info("ExperimentTracker: GPR run logged — run_id=%s", run_id)
        return run_id

    # ------------------------------------------------------------------
    # PLS run logging
    # ------------------------------------------------------------------

    def log_pls_run(
        self,
        pls_model: Any,
        pls_result: Any,
        sensor_metrics: dict[str, Any],
        y_concs: np.ndarray,
        wavelengths: np.ndarray | None = None,
        gas_name: str = "unknown",
        cal_figure_path: str | Path | None = None,
    ) -> str | None:
        """Log a PLS calibration run to MLflow.

        Parameters
        ----------
        pls_model :
            Fitted :class:`~src.calibration.pls.PLSCalibration` instance.
        pls_result :
            :class:`~src.calibration.pls.PLSFitResult` from ``pls_model.fit()``.
        sensor_metrics :
            Dict from ``compute_comprehensive_sensor_characterization``.
        y_concs :
            Training concentrations.
        wavelengths :
            Wavelength axis (used to save VIP CSV with wavelength labels).
        gas_name :
            Analyte name.
        cal_figure_path :
            Optional path to a calibration/PLS-diagnostics figure.

        Returns
        -------
        str or None
            MLflow run ID.
        """
        if not self.available:
            return None

        run = mlflow.active_run()
        if run is None:
            log.warning("ExperimentTracker.log_pls_run called outside a start_run() context")
            return None

        # ── Parameters ──────────────────────────────────────────────────
        _raw_comp = getattr(pls_result, "n_components", None) or getattr(pls_model, "n_components", 0)
        n_comp: int = int(_raw_comp) if _raw_comp is not None else 0
        _raw_opt = getattr(pls_result, "optimal_n_components", n_comp)
        opt_n: int = int(_raw_opt) if _raw_opt is not None else n_comp
        cv_strat = (
            "LOO" if getattr(pls_model, "cv_folds", -1) == -1
            else f"{pls_model.cv_folds}-fold"
        )
        mlflow.log_params({
            "model_type": "PLS",
            "gas_name": gas_name,
            "n_components": int(n_comp),
            "optimal_n_components": int(opt_n),
            "cv_strategy": cv_strat,
            "scale": bool(getattr(pls_model, "scale", True)),
            "n_training_samples": int(len(y_concs)),
            "n_features": int(getattr(pls_result, "n_features", 0)),
        })

        # ── PLS-specific metrics ─────────────────────────────────────────
        pls_metrics: dict[str, float] = {}
        for attr, key in [
            ("r2_calibration", "r2_calibration"),
            ("rmsec", "rmsec"),
            ("q2", "q2_crossvalidated"),
            ("rmsecv", "rmsecv"),
            ("pearson_r", "pearson_r"),
            ("bias", "bias"),
        ]:
            val = getattr(pls_result, attr, None)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                pls_metrics[key] = float(val)

        n_vip_above_1 = 0
        vip = getattr(pls_result, "vip_scores", None)
        if vip is not None and len(vip):
            n_vip_above_1 = int((vip > 1.0).sum())
        pls_metrics["n_vip_above_threshold"] = float(n_vip_above_1)

        if pls_metrics:
            mlflow.log_metrics(pls_metrics)

        # ── IUPAC sensor metrics ─────────────────────────────────────────
        self._log_sensor_metrics(sensor_metrics)

        # ── VIP scores CSV artifact ──────────────────────────────────────
        if vip is not None and len(vip):
            self._log_vip_csv(vip, wavelengths)

        # ── RMSECV curve ─────────────────────────────────────────────────
        rmsecv_curve = getattr(pls_result, "rmsecv_per_component", [])
        if rmsecv_curve:
            self._log_rmsecv_curve(rmsecv_curve)

        # ── Model artifact ───────────────────────────────────────────────
        try:
            mlflow.sklearn.log_model(pls_model._model, name="pls_sklearn_model")
        except Exception as exc:
            log.warning("Could not log PLS sklearn model: %s", exc)

        # ── Sensor metrics JSON ──────────────────────────────────────────
        self._log_metrics_json(sensor_metrics, "sensor_metrics.json")

        # ── Calibration / PLS diagnostics figure ─────────────────────────
        if cal_figure_path and Path(cal_figure_path).exists():
            mlflow.log_artifact(str(cal_figure_path), artifact_path="figures")

        run_id: str = run.info.run_id
        log.info("ExperimentTracker: PLS run logged — run_id=%s", run_id)
        return run_id

    # ------------------------------------------------------------------
    # Neural network run logging
    # ------------------------------------------------------------------

    def log_nn_run(
        self,
        model_type: str,
        params: dict[str, Any],
        history: dict[str, list[float]],
        analyte: str | None = None,
        n_samples: int | None = None,
        n_features: int | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Log a neural network training run (multi-task, contrastive, autoencoder).

        Parameters
        ----------
        model_type :
            Human-readable model type string, e.g. ``"Multi-Task CNN"``.
        params :
            Hyperparameter dict (e.g. ``{"embed_dim": 64, "lr": 0.001, ...}``).
            All values must be MLflow-serialisable (str, int, float, bool).
        history :
            Training history dict with keys ``"train_loss"`` and optionally
            ``"val_loss"``.  The final value of each key is logged as a metric.
        analyte :
            Analyte / gas name, if known.
        n_samples :
            Number of training samples.
        n_features :
            Input feature / wavelength dimension.
        session_id :
            Source session ID (e.g. ``"20260401_120000"``).

        Returns
        -------
        str or None
            MLflow run ID, or ``None`` if MLflow is unavailable.
        """
        if not self.available:
            return None

        run = mlflow.active_run()
        if run is None:
            log.warning("log_nn_run called outside a start_run() context")
            return None

        # ── Parameters ──────────────────────────────────────────────────
        logged_params: dict[str, Any] = {
            "model_type": model_type,
            "analyte": analyte or "unknown",
            "n_samples": n_samples,
            "n_features": n_features,
            "source_session": session_id or "unknown",
        }
        logged_params.update(params)
        # MLflow requires all param values to be str/int/float/bool
        safe_params = {
            k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
            for k, v in logged_params.items()
            if v is not None
        }
        mlflow.log_params(safe_params)

        # ── Metrics — final epoch values ────────────────────────────────
        metrics: dict[str, float] = {}
        for key in ("train_loss", "val_loss"):
            series = history.get(key, [])
            if series:
                metrics[f"final_{key}"] = float(series[-1])
                metrics[f"min_{key}"] = float(min(series))
        if metrics:
            mlflow.log_metrics(metrics)

        run_id: str = run.info.run_id
        log.info("ExperimentTracker: NN run logged — run_id=%s model=%s", run_id, model_type)
        return run_id

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def list_runs(self, max_results: int = 100) -> Any:
        """Return a DataFrame of recent runs for this experiment.

        Returns
        -------
        pandas.DataFrame or None
            Columns include ``run_id``, ``status``, ``start_time``,
            ``params.*``, ``metrics.*``.  Returns ``None`` if MLflow is
            unavailable.
        """
        if not self.available:
            return None
        try:
            return mlflow.search_runs(
                experiment_ids=[self._experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"],
            )
        except Exception as exc:
            log.warning("ExperimentTracker.list_runs failed: %s", exc)
            return None

    def best_run(
        self,
        metric: str = "metrics.q2_crossvalidated",
        higher_is_better: bool = True,
    ) -> dict[str, Any] | None:
        """Return the metadata dict for the best run by *metric*.

        Parameters
        ----------
        metric :
            Column name in the MLflow runs DataFrame, e.g.
            ``"metrics.r2_calibration"`` or ``"metrics.rmsecv"``.
        higher_is_better :
            If ``True``, return the run with the highest metric value.

        Returns
        -------
        dict or None
            Row from the runs DataFrame as a plain dict.
        """
        df = self.list_runs()
        if df is None or df.empty or metric not in df.columns:
            return None
        df_valid = df.dropna(subset=[metric])
        if df_valid.empty:
            return None
        idx = df_valid[metric].idxmax() if higher_is_better else df_valid[metric].idxmin()
        from typing import cast
        return cast(dict[str, Any], df_valid.loc[idx].to_dict())

    def get_run_url(self) -> str:
        """Return the MLflow UI URL for the active run (if any)."""
        if not self.available:
            return ""
        run = mlflow.active_run()
        if run is None:
            return ""
        tracking_uri = mlflow.get_tracking_uri()
        return f"{tracking_uri}/#/experiments/{self._experiment_id}/runs/{run.info.run_id}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_sensor_metrics(self, sensor_metrics: dict[str, Any]) -> None:
        """Log IUPAC sensor metrics as MLflow metrics."""
        scalar_keys = [
            ("sensitivity",        "sensitivity"),
            ("sensitivity_se",     "sensitivity_se"),
            ("r_squared",          "r_squared"),
            ("rmse",               "rmse"),
            ("noise_std",          "noise_std"),
            ("lob_ppm",            "lob_ppm"),
            ("lod_ppm",            "lod_ppm"),
            ("loq_ppm",            "loq_ppm"),
            ("lol_ppm",            "lol_ppm"),
            ("lod_ppm_ci_lower",   "lod_ppm_ci_lower"),
            ("lod_ppm_ci_upper",   "lod_ppm_ci_upper"),
            ("loq_ppm_ci_lower",   "loq_ppm_ci_lower"),
            ("loq_ppm_ci_upper",   "loq_ppm_ci_upper"),
        ]
        logged: dict[str, float] = {}
        for src_key, mlf_key in scalar_keys:
            v = sensor_metrics.get(src_key)
            if v is not None:
                try:
                    fv = float(v)
                    if not np.isnan(fv):
                        logged[mlf_key] = fv
                except (TypeError, ValueError):
                    pass
        if logged:
            mlflow.log_metrics(logged)

    def _log_metrics_json(self, metrics: dict[str, Any], filename: str) -> None:
        """Save metrics dict as a JSON artifact."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(metrics, f, indent=2, default=_json_default)
            tmp_path = f.name
        try:
            mlflow.log_artifact(tmp_path, artifact_path="metrics")
        finally:
            os.unlink(tmp_path)

    def _log_vip_csv(
        self, vip: np.ndarray, wavelengths: np.ndarray | None
    ) -> None:
        """Save VIP scores as a CSV artifact."""
        try:
            import pandas as pd  # already a project dependency

            if wavelengths is not None and len(wavelengths) == len(vip):
                df = pd.DataFrame({"wavelength_nm": wavelengths, "vip_score": vip})
            else:
                df = pd.DataFrame({"feature_index": np.arange(len(vip)), "vip_score": vip})

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False, encoding="utf-8"
            ) as f:
                df.to_csv(f, index=False)
                tmp_path = f.name
            try:
                mlflow.log_artifact(tmp_path, artifact_path="vip_scores")
            finally:
                os.unlink(tmp_path)
        except Exception as exc:
            log.warning("Could not log VIP CSV: %s", exc)

    def _log_rmsecv_curve(self, rmsecv_curve: list[float]) -> None:
        """Log the RMSECV-per-component curve as step metrics."""
        try:
            for i, val in enumerate(rmsecv_curve):
                mlflow.log_metric("rmsecv_curve", float(val), step=i + 1)
        except Exception as exc:
            log.warning("Could not log RMSECV curve: %s", exc)


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_default_tracker: ExperimentTracker | None = None


def get_tracker(
    experiment_name: str = "LSPR_Gas_Sensing",
    tracking_uri: str | None = None,
) -> ExperimentTracker:
    """Return (or create) the module-level default :class:`ExperimentTracker`.

    The tracker is initialised once and reused across calls, so the
    MLflow experiment is only looked up once per process.

    Parameters
    ----------
    experiment_name :
        Experiment name passed to MLflow.
    tracking_uri :
        Override the tracking URI.

    Returns
    -------
    ExperimentTracker
    """
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
    return _default_tracker


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
