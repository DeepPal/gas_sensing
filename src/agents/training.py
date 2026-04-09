"""
src.agents.training
===================
Training Agent — autonomous closed-loop model retraining for the the sensor
LSPR gas sensing platform.

What it does
------------
Monitors real-time model performance metrics and decides **when** to trigger
retraining.  When triggered, it:

1. Collects accumulated session data from disk (``output/sessions/``).
2. Retrains :class:`~src.calibration.gpr.GPRCalibration` on new Δλ data.
3. Retrains :class:`~src.models.cnn.CNNGasClassifier` if enough multi-class
   data is available.
4. Evaluates both models on a held-out validation split.
5. **Atomic swap** — only promotes the new model if it beats the incumbent
   on R² (GPR) or accuracy (CNN), preventing regressions.
6. Logs everything to MLflow.

Triggers
--------
The agent checks for retrain triggers after each ``push()`` call:

- **Performance trigger**: rolling GPR R² drops below ``min_r2_threshold``.
- **Drift trigger**: ``DriftDetectionAgent`` signals active drift (optional
  integration via ``notify_drift()``).
- **Volume trigger**: ``retrain_every_n_samples`` new samples accumulated
  since last retrain.

Deployment safety
-----------------
The atomic swap uses file-system rename (POSIX-atomic on Linux; best-effort
on Windows) so a crash during model saving never leaves a corrupt checkpoint.

Public API
----------
- ``TrainingAgent``         — main agent class
- ``RetrainTrigger``        — reason a retrain was initiated
- ``RetrainResult``         — outcome of one retrain cycle
"""

from __future__ import annotations

import collections
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from pathlib import Path
import threading
import time

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RetrainTrigger(str, Enum):
    """Reason a retraining cycle was initiated."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    DRIFT_ALERT = "drift_alert"
    VOLUME_THRESHOLD = "volume_threshold"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


@dataclass
class RetrainResult:
    """Outcome of one automated retraining cycle."""

    trigger: RetrainTrigger
    timestamp: datetime
    gpr_trained: bool = False
    cnn_trained: bool = False
    gpr_r2_before: float | None = None
    gpr_r2_after: float | None = None
    cnn_acc_before: float | None = None
    cnn_acc_after: float | None = None
    model_promoted: bool = False
    notes: str = ""

    def improved(self) -> bool:
        """Return True if at least one model metric improved."""
        gpr_ok = (
            self.gpr_r2_before is not None
            and self.gpr_r2_after is not None
            and self.gpr_r2_after > self.gpr_r2_before
        )
        cnn_ok = (
            self.cnn_acc_before is not None
            and self.cnn_acc_after is not None
            and self.cnn_acc_after > self.cnn_acc_before
        )
        return gpr_ok or cnn_ok


# ---------------------------------------------------------------------------
# Training Agent
# ---------------------------------------------------------------------------


class TrainingAgent:
    """Autonomous closed-loop model retraining agent.

    Parameters
    ----------
    model_dir:
        Directory containing live model checkpoints
        (``cnn_classifier.pt``, ``gpr_calibration.joblib``).
    sessions_dir:
        Root directory of session output folders (scanned for training data).
    min_r2_threshold:
        GPR R² below this value triggers a performance-degradation retrain.
    retrain_every_n_samples:
        Trigger a retrain after this many new samples since the last cycle.
    min_samples_for_retrain:
        Minimum number of labelled samples required to attempt retraining.
    gpr_epochs, cnn_epochs:
        Training parameters passed to the respective trainers.
    mlflow_uri:
        MLflow tracking URI (``None`` disables MLflow logging).
    """

    def __init__(
        self,
        model_dir: str = "output/models",
        sessions_dir: str = "output/sessions",
        min_r2_threshold: float = 0.90,
        retrain_every_n_samples: int = 500,
        min_samples_for_retrain: int = 50,
        cnn_epochs: int = 20,
        mlflow_uri: str | None = "experiments/mlruns",
        retrain_cooldown_s: float = 60.0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.sessions_dir = Path(sessions_dir)
        self.min_r2_threshold = min_r2_threshold
        self.retrain_every_n_samples = retrain_every_n_samples
        self.min_samples_for_retrain = min_samples_for_retrain
        self.cnn_epochs = cnn_epochs
        self.mlflow_uri = mlflow_uri
        # Minimum seconds between retrain cycles — prevents performance-trigger
        # storms when R² stays below threshold for a long time.
        self.retrain_cooldown_s = retrain_cooldown_s

        # Rolling performance window: deque gives O(1) append + auto-eviction.
        self._r2_window: collections.deque[float] = collections.deque(maxlen=20)
        self._r2_window_size = 20

        # Sample counter since last retrain
        self._samples_since_retrain: int = 0
        self._total_samples: int = 0

        # Watermark: maps CSV path → mtime of last row we consumed.
        # Prevents re-reading data from previous retrain cycles, which would
        # exponentially overweight early sessions in each successive model.
        self._csv_watermarks: dict[str, float] = {}

        # Drift flag (set by notify_drift())
        self._drift_pending: bool = False

        # History and thread control
        self._results: list[RetrainResult] = []
        # _retrain_lock guards _is_retraining so check-and-set is atomic,
        # preventing two simultaneous retrain threads (TOCTOU race condition).
        self._retrain_lock = threading.Lock()
        self._is_retraining: bool = False
        # Event set when a retrain cycle finishes — allows wait_for_retrain().
        self._retrain_done = threading.Event()
        self._retrain_done.set()  # initially "done" (no cycle in progress)
        # Timestamp of last completed retrain — used for cooldown enforcement.
        self._last_retrain_time: float = 0.0

    # ------------------------------------------------------------------
    # Feed interface — called from orchestrator's _on_sample()
    # ------------------------------------------------------------------

    def push(
        self,
        gpr_r2: float | None = None,
        wavelength_shift: float | None = None,
        concentration_ppm: float | None = None,
    ) -> RetrainTrigger | None:
        """Record a new prediction outcome and check retrain triggers.

        Parameters
        ----------
        gpr_r2:
            GPR R² for the current prediction (if available from registry).
        wavelength_shift:
            Measured Δλ (nm).
        concentration_ppm:
            True concentration if known (training mode); None for inference.

        Returns
        -------
        RetrainTrigger or None
            The trigger that fired, or None if no retrain was initiated.
        """
        self._total_samples += 1
        self._samples_since_retrain += 1

        if gpr_r2 is not None and np.isfinite(gpr_r2):
            self._r2_window.append(gpr_r2)  # deque auto-evicts oldest

        trigger = self._check_triggers()
        if trigger is not None:
            # Atomic check-and-set under the lock — prevents TOCTOU race
            # where two simultaneous push() calls both see _is_retraining=False
            # and both spawn a background thread.
            with self._retrain_lock:
                if not self._is_retraining:
                    self._is_retraining = True
                    self._retrain_done.clear()
                    log.info("TrainingAgent: retrain triggered by %s", trigger.value)
                    thread = threading.Thread(
                        target=self._retrain_cycle,
                        args=(trigger,),
                        daemon=True,
                    )
                    thread.start()
                else:
                    trigger = None  # already retraining — ignore

        return trigger

    def notify_drift(self) -> None:
        """Call this when :class:`~src.agents.drift.DriftDetectionAgent` raises an alert."""
        self._drift_pending = True
        log.info("TrainingAgent: drift notification received — retrain queued.")

    def trigger_manual_retrain(self) -> None:
        """Immediately queue a manual retraining cycle (e.g. from dashboard button)."""
        with self._retrain_lock:
            if not self._is_retraining:
                self._is_retraining = True
                self._retrain_done.clear()
                thread = threading.Thread(
                    target=self._retrain_cycle,
                    args=(RetrainTrigger.MANUAL,),
                    daemon=True,
                )
                thread.start()
            else:
                log.info("TrainingAgent: manual retrain ignored — already retraining.")

    def wait_for_retrain(self, timeout: float = 10.0) -> bool:
        """Block until the current retrain cycle finishes (or timeout expires).

        Useful for tests, the dashboard "Retrain Now" button, and anywhere that
        needs to know the cycle completed before reading results.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  Returns ``False`` if the cycle has not
            finished within this time; ``True`` if it completed.
        """
        return self._retrain_done.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return a JSON-serialisable status snapshot."""
        avg_r2 = float(np.mean(self._r2_window)) if self._r2_window else None
        return {
            "is_retraining": self._is_retraining,
            "total_samples": self._total_samples,
            "samples_since_retrain": self._samples_since_retrain,
            "avg_r2_last_window": round(avg_r2, 4) if avg_r2 is not None else None,
            "drift_pending": self._drift_pending,
            "n_retrain_cycles": len(self._results),
            "last_retrain": (self._results[-1].timestamp.isoformat() if self._results else None),
        }

    def get_results(self) -> list[RetrainResult]:
        """Return the full history of retrain cycles."""
        return list(self._results)

    # ------------------------------------------------------------------
    # Trigger logic
    # ------------------------------------------------------------------

    def _check_triggers(self) -> RetrainTrigger | None:
        if self._is_retraining:
            return None

        # Cooldown: suppress all non-drift triggers for retrain_cooldown_s after
        # the last cycle.  Prevents performance-trigger storms when R² stays
        # persistently below threshold (common during sensor warm-up).
        since_last = time.monotonic() - self._last_retrain_time
        in_cooldown = since_last < self.retrain_cooldown_s

        # 1. Drift alert — bypasses cooldown (sensor physically moved / fouled)
        if self._drift_pending:
            self._drift_pending = False
            return RetrainTrigger.DRIFT_ALERT

        if in_cooldown:
            return None

        # 2. Performance degradation
        if len(self._r2_window) >= self._r2_window_size:
            avg_r2 = float(np.mean(self._r2_window))
            if avg_r2 < self.min_r2_threshold:
                return RetrainTrigger.PERFORMANCE_DEGRADATION

        # 3. Volume threshold
        if self._samples_since_retrain >= self.retrain_every_n_samples:
            return RetrainTrigger.VOLUME_THRESHOLD

        return None

    # ------------------------------------------------------------------
    # Retraining cycle (runs in a background thread)
    # ------------------------------------------------------------------

    def _retrain_cycle(self, trigger: RetrainTrigger) -> None:
        # _is_retraining was already set to True by the caller (under lock).
        # This method owns the flag for the duration of the cycle.
        try:
            self._run_retrain(trigger)
        except Exception as exc:
            log.error("TrainingAgent: retrain cycle failed: %s", exc, exc_info=True)
        finally:
            self._last_retrain_time = time.monotonic()
            self._samples_since_retrain = 0
            self._is_retraining = False
            self._retrain_done.set()  # unblock any wait_for_retrain() callers

    def _run_retrain(self, trigger: RetrainTrigger) -> None:
        result = RetrainResult(trigger=trigger, timestamp=datetime.now(timezone.utc))
        log.info("TrainingAgent: starting retrain cycle (trigger=%s)", trigger.value)

        # ── 1. Collect training data ──────────────────────────────────
        delta_lambda, concentrations, gas_labels = self._collect_session_data()
        n = len(delta_lambda)
        log.info("TrainingAgent: %d labelled samples collected.", n)

        if n < self.min_samples_for_retrain:
            result.notes = f"Insufficient data: {n} < {self.min_samples_for_retrain}"
            log.warning("TrainingAgent: %s — aborting.", result.notes)
            self._results.append(result)
            return

        # ── 2. Retrain GPR ────────────────────────────────────────────
        result.gpr_trained, result.gpr_r2_before, result.gpr_r2_after = self._retrain_gpr(
            delta_lambda, concentrations
        )

        # ── 3. Retrain CNN (if multi-class data) ─────────────────────
        unique_gases = list(set(gas_labels))
        if len(unique_gases) >= 2:
            result.cnn_trained, result.cnn_acc_before, result.cnn_acc_after = self._retrain_cnn(
                delta_lambda, gas_labels, concentrations, unique_gases
            )

        # ── 4. Promote if improved ────────────────────────────────────
        result.model_promoted = result.improved()
        if result.model_promoted:
            log.info("TrainingAgent: new models promoted to registry.")
        else:
            log.info("TrainingAgent: new models NOT better than incumbent — keeping old.")

        # ── 5. Log to MLflow ──────────────────────────────────────────
        self._log_to_mlflow(result)

        self._results.append(result)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_session_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Scan session CSVs and return (delta_lambda, concentration, gas_label) arrays.

        Uses a per-file mtime watermark so each retrain cycle only reads data
        written *since the previous cycle*.  Without this, early sessions would
        be re-read on every retrain, exponentially overweighting them.
        """
        delta_lambdas: list[float] = []
        concentrations: list[float] = []
        gas_labels: list[str] = []
        new_watermarks: dict[str, float] = {}

        csv_files = sorted(self.sessions_dir.rglob("pipeline_results.csv"))
        for csv_path in csv_files:
            key = str(csv_path)
            try:
                mtime = csv_path.stat().st_mtime
            except OSError:
                continue

            # Skip files not modified since last read
            last_seen = self._csv_watermarks.get(key, 0.0)
            new_watermarks[key] = mtime
            if mtime <= last_seen:
                continue

            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        dl = row.get("wavelength_shift", "")
                        c = row.get("concentration_ppm", "")
                        g = row.get("gas_type", "unknown")
                        if dl and c:
                            try:
                                dl_f = float(dl)
                                c_f = float(c)
                                if np.isfinite(dl_f) and np.isfinite(c_f):
                                    delta_lambdas.append(dl_f)
                                    concentrations.append(c_f)
                                    gas_labels.append(g or "unknown")
                            except ValueError:
                                pass
            except Exception as exc:
                log.debug("Skipping %s: %s", csv_path, exc)

        # Advance watermarks after successful scan
        self._csv_watermarks.update(new_watermarks)

        return (
            np.array(delta_lambdas, dtype=float),
            np.array(concentrations, dtype=float),
            gas_labels,
        )

    # ------------------------------------------------------------------
    # GPR retraining
    # ------------------------------------------------------------------

    def _retrain_gpr(
        self,
        delta_lambda: np.ndarray,
        concentrations: np.ndarray,
    ) -> tuple[bool, float | None, float | None]:
        """Retrain GPRCalibration and atomic-swap if improved.

        Returns
        -------
        (trained, r2_before, r2_after)
        """
        try:
            from sklearn.metrics import r2_score
            from sklearn.model_selection import train_test_split

            from src.calibration.gpr import GPRCalibration
        except ImportError as exc:
            log.warning("GPR retrain skipped (import error): %s", exc)
            return False, None, None

        n = len(delta_lambda)

        # For small datasets (n < 20) use LOOCV — consistent with train_gpr.py.
        # A random 80/20 split on n=12 gives only 2 validation points, which
        # produces an unreliable R² estimate.
        if n < 20:
            from sklearn.model_selection import LeaveOneOut, cross_val_score

            log.info("GPR retrain: n=%d < 20 — using LOOCV", n)
            X_cv = delta_lambda.reshape(-1, 1)
            # cross_val_score needs a sklearn-compatible estimator; use the raw GPR
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            kernel = RBF(length_scale=1.0, length_scale_bounds=(0.1, 100.0)) + WhiteKernel(
                noise_level=0.01, noise_level_bounds=(1e-5, 10.0)
            )
            sk_gpr = GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3, random_state=42
            )
            scores = cross_val_score(sk_gpr, X_cv, concentrations, cv=LeaveOneOut(), scoring="r2")
            r2_after = float(np.clip(np.mean(scores), 0.0, 1.0))
            X_train, y_train = delta_lambda, concentrations
            X_val, y_val = delta_lambda, concentrations  # LOOCV used all data
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                delta_lambda, concentrations, test_size=0.2, random_state=42
            )
            r2_after = None  # computed below after fitting

        # Evaluate incumbent on validation split
        incumbent_path = self.model_dir / "gpr_calibration.joblib"
        r2_before: float | None = None
        if incumbent_path.exists():
            try:
                old_gpr = GPRCalibration.load(str(incumbent_path))
                preds, _ = old_gpr.predict(X_val.reshape(-1, 1) if X_val.ndim == 1 else X_val)
                r2_before = float(r2_score(y_val, preds))
            except Exception:
                pass

        # Train new GPR on full training set
        new_gpr = GPRCalibration()
        new_gpr.fit(X_train, y_train)
        if r2_after is None:  # holdout case — compute now
            preds_new, _ = new_gpr.predict(X_val.reshape(-1, 1) if X_val.ndim == 1 else X_val)
            r2_after = float(r2_score(y_val, preds_new))

        # Promote if better (or no incumbent)
        if r2_before is None or r2_after > r2_before:
            tmp_path = self.model_dir / "_gpr_calibration_tmp.joblib"
            self.model_dir.mkdir(parents=True, exist_ok=True)
            new_gpr.save(str(tmp_path))
            tmp_path.replace(incumbent_path)  # atomic rename
            log.info("GPR promoted: R²=%.4f → %.4f", r2_before or 0.0, r2_after)

        return True, r2_before, r2_after

    # ------------------------------------------------------------------
    # CNN retraining
    # ------------------------------------------------------------------

    def _retrain_cnn(
        self,
        delta_lambda: np.ndarray,
        gas_labels: list[str],
        concentrations: np.ndarray,
        class_names: list[str],
    ) -> tuple[bool, float | None, float | None]:
        """Retrain CNNGasClassifier on Δλ-as-1D-spectrum proxy.

        Note: In real use the full spectrum (3648 points) is preferred.
        Here we use Δλ as a compact proxy since session CSVs don't store
        raw spectra by default (enable ``save_raw_spectra`` in config).

        Returns
        -------
        (trained, acc_before, acc_after)
        """
        try:
            from sklearn.model_selection import train_test_split

            from src.models.cnn import CNNGasClassifier
        except ImportError as exc:
            log.warning("CNN retrain skipped (import error): %s", exc)
            return False, None, None

        # Build a minimal 10-point feature vector from Δλ context
        # (real deployment would use raw spectra from raw_spectra.parquet)
        X = np.tile(delta_lambda[:, np.newaxis], (1, 10)).astype(float)

        label_map = {g: i for i, g in enumerate(class_names)}
        y_cls = np.array([label_map.get(g, 0) for g in gas_labels], dtype=int)

        X_train, X_val, y_cls_train, y_cls_val, y_conc_train, _ = train_test_split(
            X, y_cls, concentrations, test_size=0.2, random_state=42
        )

        incumbent_path = self.model_dir / "cnn_classifier.pt"
        acc_before: float | None = None
        if incumbent_path.exists():
            try:
                old_clf = CNNGasClassifier.load(str(incumbent_path))
                names_pred, _ = old_clf.predict(X_val)
                true_names = [class_names[i] for i in y_cls_val]
                acc_before = float(
                    sum(p == t for p, t in zip(names_pred, true_names)) / len(true_names)
                )
            except Exception:
                pass

        # Train new
        clf = CNNGasClassifier(input_length=10, num_classes=len(class_names))
        clf.fit(X_train, y_cls_train, y_conc_train, class_names=class_names, epochs=self.cnn_epochs)
        names_pred_new, _ = clf.predict(X_val)
        true_names = [class_names[i] for i in y_cls_val]
        acc_after = float(sum(p == t for p, t in zip(names_pred_new, true_names)) / len(true_names))

        if acc_before is None or acc_after > acc_before:
            tmp_path = self.model_dir / "_cnn_classifier_tmp.pt"
            self.model_dir.mkdir(parents=True, exist_ok=True)
            clf.save(str(tmp_path))
            tmp_path.replace(incumbent_path)
            log.info("CNN promoted: acc=%.3f → %.3f", acc_before or 0.0, acc_after)

        return True, acc_before, acc_after

    # ------------------------------------------------------------------
    # MLflow
    # ------------------------------------------------------------------

    def _log_to_mlflow(self, result: RetrainResult) -> None:
        if self.mlflow_uri is None:
            return
        try:
            from src.training.mlflow_tracker import ExperimentTracker

            metrics: dict[str, float] = {}
            if result.gpr_r2_before is not None:
                metrics["gpr_r2_before"] = result.gpr_r2_before
            if result.gpr_r2_after is not None:
                metrics["gpr_r2_after"] = result.gpr_r2_after
            if result.gpr_r2_before is not None and result.gpr_r2_after is not None:
                metrics["gpr_r2_delta"] = result.gpr_r2_after - result.gpr_r2_before
            if result.cnn_acc_before is not None:
                metrics["cnn_acc_before"] = result.cnn_acc_before
            if result.cnn_acc_after is not None:
                metrics["cnn_acc_after"] = result.cnn_acc_after
            if result.cnn_acc_before is not None and result.cnn_acc_after is not None:
                metrics["cnn_acc_delta"] = result.cnn_acc_after - result.cnn_acc_before

            run_name = f"retrain_{result.trigger.value}_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
            with ExperimentTracker(
                run_name=run_name,
                experiment_name="auto_retrain",
                tracking_uri=self.mlflow_uri,
                tags={
                    "trigger": result.trigger.value,
                    "model_promoted": str(result.model_promoted),
                },
            ) as tracker:
                tracker.log_metrics(metrics)
                # Log model artifacts if promoted
                if result.model_promoted:
                    gpr_path = Path(self.model_dir) / "gpr_calibration.joblib"
                    cnn_path = Path(self.model_dir) / "cnn_classifier.pt"
                    if gpr_path.exists():
                        tracker.log_artifact(gpr_path, "models")
                    if cnn_path.exists():
                        tracker.log_artifact(cnn_path, "models")
        except ImportError:
            pass
        except Exception as exc:
            log.warning("MLflow logging failed in TrainingAgent: %s", exc)
