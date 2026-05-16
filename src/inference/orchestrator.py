"""
src.inference.orchestrator
==========================
SensorOrchestrator — glues hardware acquisition → pipeline → live store + disk.

Architecture
------------
::

    CCS200 (hardware)
        │  ~20 Hz callback
        ▼
    _on_sample()          ← acquisition thread
        │
        ├── RealTimePipeline.process_spectrum()   5–20 ms
        ├── ModelRegistry.predict_gas_type()      2–5 ms (optional)
        ├── ModelRegistry.predict_concentration_gpr() 1–2 ms (optional)
        ├── LiveDataStore.push(result)             <0.1 ms
        └── _SessionWriter.enqueue(result)         <0.1 ms (disk I/O in thread)

The acquisition callback must complete within ``integration_time_ms`` to avoid
back-pressure.  All disk I/O is deferred to a daemon thread via ``_SessionWriter``.

Public API
----------
- ``SensorOrchestrator``  — start_session / stop_session / from_config
"""

from __future__ import annotations

import contextlib
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import threading
import time
from typing import Any

import numpy as np

from src.agents.drift import DriftDetectionAgent
from src.agents.training import TrainingAgent
from src.inference.live_state import LiveDataStore, _LiveDataStore
from src.inference.realtime_pipeline import PipelineConfig, RealTimePipeline
from src.models.registry import ModelRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Background session writer
# ---------------------------------------------------------------------------


class _SessionWriter:
    """Non-blocking daemon writer: enqueue results in-thread, flush to disk asynchronously.

    Parameters
    ----------
    session_dir:
        Directory to write ``pipeline_results.csv`` and optionally
        ``raw_spectra.parquet``.
    flush_interval_s:
        How often the daemon thread wakes to flush pending rows.
    save_raw:
        If True, also save raw spectra to Parquet (requires pyarrow).
    """

    _CSV_COLUMNS: list[str] = [
        "timestamp",
        "sample_id",
        "peak_wavelength",
        "wavelength_shift",
        "concentration_ppm",
        "snr",
        "confidence_score",
        "gas_type",
        "gpr_uncertainty",
        "quality_score",
        "success",
        "processing_time_ms",
    ]

    def __init__(
        self,
        session_dir: Path,
        flush_interval_s: float = 0.5,
        save_raw: bool = True,
    ) -> None:
        self._session_dir = session_dir
        self._flush_interval_s = flush_interval_s
        self._save_raw = save_raw
        self._csv_path = session_dir / "pipeline_results.csv"
        self._raw_path = session_dir / "raw_spectra.parquet"
        # Per-flush partition files go here; merged into raw_spectra.parquet at stop().
        # This avoids the O(n²) read-entire-file + rewrite pattern of the naive approach.
        self._raw_parts_dir = session_dir / "_raw_parts"

        self._pending: list[dict[str, Any]] = []
        self._raw_buf: list[tuple] = []
        self._lock = threading.Lock()
        self._header_written = False
        self._raw_part_idx = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon write thread."""
        self._running = True
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def enqueue(
        self,
        result_dict: dict[str, Any],
        raw_intensities: np.ndarray | None = None,
    ) -> None:
        """Non-blocking enqueue from the acquisition thread.

        Raw intensities are stored as a numpy array (not converted to a Python
        list here) so the acquisition thread is not blocked by 3648 PyObject
        allocations per frame.  The list conversion happens in the daemon flush
        thread at 2 Hz instead.
        """
        with self._lock:
            # result_dict is a fresh local from _on_sample — no copy needed.
            self._pending.append(result_dict)
            if self._save_raw and raw_intensities is not None:
                ts = result_dict.get("timestamp", "")
                self._raw_buf.append((ts, raw_intensities.copy()))

    def stop(self) -> None:
        """Flush remaining data, stop the thread, and merge raw Parquet partitions."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self._flush()
        if self._save_raw:
            self._merge_raw_parts()

    def _write_loop(self) -> None:
        while self._running:
            time.sleep(self._flush_interval_s)
            self._flush()

    def _flush(self) -> None:
        with self._lock:
            rows = self._pending[:]
            raw_rows = self._raw_buf[:]
            self._pending.clear()
            self._raw_buf.clear()

        if rows:
            try:
                mode = "a" if self._header_written else "w"
                with open(self._csv_path, mode, newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS, extrasaction="ignore")
                    if not self._header_written:
                        writer.writeheader()
                        self._header_written = True
                    writer.writerows(rows)
            except Exception as exc:
                log.error("SessionWriter flush failed: %s", exc)

        if raw_rows and self._save_raw:
            try:
                import pandas as pd

                self._raw_parts_dir.mkdir(parents=True, exist_ok=True)
                part_path = self._raw_parts_dir / f"part_{self._raw_part_idx:06d}.parquet"
                self._raw_part_idx += 1
                # Convert numpy arrays to lists here (daemon thread, 2 Hz) rather
                # than in enqueue() (acquisition thread, 20 Hz).
                df = pd.DataFrame(
                    [(ts, arr.tolist()) for ts, arr in raw_rows],
                    columns=["timestamp", "intensities"],
                )
                df.to_parquet(part_path, index=False)
            except Exception as exc:
                log.debug("Raw spectrum flush failed: %s", exc)

    def _merge_raw_parts(self) -> None:
        """Merge per-flush Parquet partitions into a single raw_spectra.parquet.

        Called once at session end.  O(n) in total frames — partitions are
        concatenated in order and the temporary directory is removed.
        """
        if not self._raw_parts_dir.exists():
            return
        parts = sorted(self._raw_parts_dir.glob("part_*.parquet"))
        if not parts:
            return
        try:
            import pandas as pd

            df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
            df.to_parquet(self._raw_path, index=False)
            shutil.rmtree(self._raw_parts_dir, ignore_errors=True)
            log.info(
                "Raw spectra merged: %d frames → %s",
                len(df),
                self._raw_path.name,
            )
        except Exception as exc:
            log.warning("Raw spectra merge failed: %s", exc)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class SensorOrchestrator:
    """Orchestrates the full real-time sensing loop.

    Connects the spectrometer hardware (or simulation), feeds frames through
    the :class:`~src.inference.realtime_pipeline.RealTimePipeline`, optionally
    enriches with CNN/GPR predictions, and broadcasts results to both the
    :data:`~src.inference.live_state.LiveDataStore` (for Streamlit) and a
    disk-backed CSV session file.

    Parameters
    ----------
    integration_time_ms:
        Spectrometer integration time in milliseconds.
    target_wavelength:
        Nominal LSPR peak wavelength in nm (used as pipeline default reference).
    calibration_slope, calibration_intercept:
        Linear Δλ → ppm calibration coefficients (overridden by registry if GPR loaded).
    reference_wavelength:
        Zero-gas reference peak in nm for wavelength shift calculation.
    model_dir:
        Directory containing model files (``cnn_classifier.pt``, etc.).
    sessions_dir:
        Root directory for session output folders.
    save_raw_spectra:
        Whether to save raw spectra to Parquet.
    resource_string:
        Optional VISA resource string override for the CCS200.
    live_store:
        Live data store instance (defaults to the module-level singleton).
    """

    def __init__(
        self,
        integration_time_ms: float = 50.0,
        target_wavelength: float = 532.0,
        calibration_slope: float = 0.116,
        calibration_intercept: float = 0.0,
        reference_wavelength: float = 531.5,
        model_dir: str = "models/registry",
        sessions_dir: str = "output/sessions",
        save_raw_spectra: bool = False,
        resource_string: str | None = None,
        live_store: _LiveDataStore | None = None,
    ) -> None:
        self.integration_time_ms = integration_time_ms
        self.sessions_dir = Path(sessions_dir)
        self.save_raw_spectra = save_raw_spectra
        self.resource_string = resource_string
        self._live_store = live_store or LiveDataStore

        # Pipeline
        cfg = PipelineConfig(
            target_wavelength=target_wavelength,
            calibration_slope=calibration_slope,
            calibration_intercept=calibration_intercept,
            reference_wavelength=reference_wavelength,
        )
        self.pipeline = RealTimePipeline(cfg)

        # Model registry (loads lazily; errors are non-fatal)
        self.registry = ModelRegistry()
        try:
            status = self.registry.load_all(model_dir)
            log.info("ModelRegistry status: %s", status)
            self._apply_calibration()
        except Exception as exc:
            log.warning("ModelRegistry.load_all failed (running without models): %s", exc)

        # Drift monitor — runs continuously, logs alerts to standard logger
        self.drift_agent = DriftDetectionAgent(
            window_size=120,
            drift_threshold_nm_per_min=0.5,
            offset_threshold_nm=1.0,
        )

        # Training agent — autonomous closed-loop retraining
        self.training_agent = TrainingAgent(
            model_dir=model_dir,
            sessions_dir=sessions_dir,
        )

        # Session state (populated in start_session)
        self._session_id: str | None = None
        self._session_dir: Path | None = None
        self._writer: _SessionWriter | None = None
        self._gas_label: str = "unknown"

        # Watchdog: consecutive pipeline errors → auto-stop to prevent silent stall
        self._consecutive_errors: int = 0
        self._MAX_CONSECUTIVE_ERRORS: int = 10

        # Hardware service (populated in start_session)
        try:
            from src.acquisition import RealtimeAcquisitionService

            self._acquisition_cls = RealtimeAcquisitionService
        except (ImportError, TypeError):
            try:
                from gas_analysis.acquisition.ccs200_realtime import RealtimeAcquisitionService

                self._acquisition_cls = RealtimeAcquisitionService
            except ImportError:
                self._acquisition_cls = None  # type: ignore[assignment]
        self.service: Any = None  # RealtimeAcquisitionService or None; typed Any to avoid stubs

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SensorOrchestrator:
        """Construct from a config dict (e.g., the ``sensor`` section of config.yaml)."""
        sensor = cfg.get("sensor", {})
        api = cfg.get("api", {})
        return cls(
            integration_time_ms=float(sensor.get("integration_time_ms", 50.0)),
            target_wavelength=float(sensor.get("target_wavelength", 532.0)),
            calibration_slope=float(sensor.get("calibration_slope", 0.116)),
            calibration_intercept=float(sensor.get("calibration_intercept", 0.0)),
            reference_wavelength=float(sensor.get("reference_wavelength", 531.5)),
            model_dir=str(api.get("model_dir", "models/registry")),
            sessions_dir=str(sensor.get("sessions_dir", "output/sessions")),
            save_raw_spectra=bool(sensor.get("save_raw_spectra", False)),
            resource_string=sensor.get("resource_string"),
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def start_session(
        self,
        gas_label: str = "unknown",
        notes: str = "",
        duration_s: float | None = None,
        interface: str = "visa",
    ) -> str:
        """Start a new acquisition session.

        Parameters
        ----------
        gas_label:
            Gas analyte label for this session.
        notes:
            Free-text notes stored in session metadata.
        duration_s:
            Auto-stop after this many seconds (None = run until stop_session()).
        interface:
            Hardware interface: ``"visa"`` or ``"simulation"``.

        Returns
        -------
        str
            Session ID (timestamp-based).
        """
        self._gas_label = gas_label
        ts = datetime.now(timezone.utc)
        self._session_id = ts.strftime("%Y%m%d_%H%M%S")
        self._session_dir = self.sessions_dir / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Reset pipeline and live store for fresh session
        self.pipeline = RealTimePipeline(self.pipeline.config)
        self._apply_calibration()
        self._live_store.clear()

        # Write session metadata
        meta = {
            "session_id": self._session_id,
            "gas_label": gas_label,
            "notes": notes,
            "start_time": ts.isoformat(),
            "integration_time_ms": self.integration_time_ms,
            "interface": interface,
        }
        self._live_store.set_session_meta(meta)
        self._write_session_json(meta, "session_meta.json")

        # Start disk writer
        self._writer = _SessionWriter(
            self._session_dir,
            save_raw=self.save_raw_spectra,
        )
        self._writer.start()

        # Connect acquisition service
        if interface == "simulation" or self._acquisition_cls is None:
            self._start_simulation(duration_s)
        else:
            self._start_hardware(duration_s, interface=interface)

        self._live_store.set_running(True)
        log.info(
            "Session %s started (gas=%s, interface=%s)", self._session_id, gas_label, interface
        )
        return self._session_id

    def stop_session(self) -> Path | None:
        """Stop acquisition, flush writers, finalise session.

        Returns
        -------
        Path or None
            Path to the session directory, or None if no session was active.
        """
        if self._session_id is None:
            return None

        self._live_store.set_running(False)

        # Stop hardware/simulation
        if self.service is not None:
            try:
                self.service.stop()
            except Exception as exc:
                log.warning("Service stop error: %s", exc)
            self.service = None

        # Flush writer
        writer = self._writer
        if writer is not None:
            writer.stop()
            self._writer = None

        # Update session metadata with end time
        session_dir = self._session_dir
        if session_dir:
            meta = self._live_store.get_session_meta()
            meta["end_time"] = datetime.now(timezone.utc).isoformat()
            meta["total_samples"] = self._live_store.get_sample_count()
            self._write_session_json(meta, "session_meta.json")

        log.info(
            "Session %s stopped (%d samples)",
            self._session_id,
            self._live_store.get_sample_count(),
        )
        self._session_id = None
        self._session_dir = None
        return session_dir

    # ------------------------------------------------------------------
    # Frame callback
    # ------------------------------------------------------------------

    def _on_sample(self, sample: dict[str, Any]) -> None:
        """Process one frame from the acquisition thread (~20 Hz).

        Must complete within ``integration_time_ms`` ms to avoid back-pressure.
        """
        intensities = np.asarray(sample.get("intensities", []), dtype=float)
        sample_num = int(sample.get("sample_num", 0))

        # Normalise timestamp: hardware callback delivers a unix float;
        # convert to datetime so the pipeline and CSV writer always see datetime.
        ts_raw = sample.get("timestamp")
        if isinstance(ts_raw, (int, float)):
            timestamp: datetime = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, datetime):
            timestamp = ts_raw
        else:
            timestamp = datetime.now(timezone.utc)

        if intensities.size == 0:
            return

        wl = self._live_store.get_wavelengths()
        if wl is None:
            # First frame — store wavelengths from the service
            try:
                wl_arr = np.asarray(sample.get("wavelengths", []), dtype=float)
                if wl_arr.size > 0:
                    self._live_store.set_wavelengths(wl_arr)
                    wl = wl_arr
            except Exception:
                pass
        if wl is None:
            return

        # ── Pipeline ──────────────────────────────────────────────────
        try:
            result = self.pipeline.process_spectrum(
                wl, intensities, timestamp=timestamp, sample_id=f"S{sample_num:06d}"
            )
            self._consecutive_errors = 0  # reset on success
        except Exception as exc:
            self._consecutive_errors += 1
            log.warning(
                "Pipeline error on sample %d (%d consecutive): %s",
                sample_num, self._consecutive_errors, exc,
            )
            if self._consecutive_errors >= self._MAX_CONSECUTIVE_ERRORS:
                log.error(
                    "Stopping acquisition after %d consecutive pipeline errors",
                    self._consecutive_errors,
                )
                self._live_store.set_running(False)
            return

        spec = result.spectrum

        # ── CNN gas classification ─────────────────────────────────────
        gas_type_pred = spec.gas_type or "unknown"
        confidence = 0.0
        if self.registry.has_cnn():
            with contextlib.suppress(Exception):
                gas_type_pred, confidence = self.registry.predict_gas_type(intensities)

        # ── GPR concentration refinement ──────────────────────────────
        gpr_conc: float | None = None
        gpr_unc: float | None = None
        if self.registry.has_gpr() and spec.wavelength_shift is not None:
            gpr_conc, gpr_unc = self.registry.predict_concentration_gpr(spec.wavelength_shift)

        # ── Build result dict ──────────────────────────────────────────
        result_dict: dict[str, Any] = {
            "timestamp": timestamp,
            "sample_id": f"S{sample_num:06d}",
            "peak_wavelength": spec.peak_wavelength,
            "wavelength_shift": spec.wavelength_shift,
            "concentration_ppm": gpr_conc if gpr_conc is not None else spec.concentration_ppm,
            "snr": spec.snr,
            "confidence_score": confidence,
            "gas_type": gas_type_pred,
            "gpr_uncertainty": gpr_unc,
            "quality_score": result.spectrum.quality_score,
            "success": result.success,
            "processing_time_ms": result.processing_time_ms,
        }

        # ── Drift monitoring ──────────────────────────────────────────
        if spec.peak_wavelength is not None:
            drift_alert = self.drift_agent.push(spec.peak_wavelength, timestamp)
            if drift_alert is not None:
                result_dict["drift_alert"] = drift_alert.alert_type.value
                result_dict["drift_rate_nm_per_min"] = drift_alert.drift_rate_nm_per_min
                log.warning("DriftAlert [%s]: %s", drift_alert.severity, drift_alert.message)
                # Notify training agent so it can queue a retrain cycle
                self.training_agent.notify_drift()

        # ── Training agent feed ───────────────────────────────────────
        self.training_agent.push(
            gpr_r2=None,  # R² not available per-frame; agent uses volume+drift triggers
            wavelength_shift=spec.wavelength_shift,
            concentration_ppm=gpr_conc if gpr_conc is not None else spec.concentration_ppm,
        )

        # ── Publish ────────────────────────────────────────────────────
        self._live_store.push(result_dict, raw_intensities=intensities)
        writer = self._writer
        if writer is not None:
            writer.enqueue(result_dict, raw_intensities=intensities)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_calibration(self) -> None:
        """Load calibration from registry into the pipeline (if available)."""
        if self.registry.has_calibration():
            slope = self.registry.get_calibration_slope()
            ref_wl = self.registry.get_reference_wavelength()
            self.pipeline.set_calibration(
                slope=slope,
                intercept=0.0,
                reference_wl=ref_wl,
            )

    def _start_hardware(self, duration_s: float | None, interface: str = "auto") -> None:
        """Connect to CCS200 and start the acquisition service."""
        if self._acquisition_cls is None:
            log.warning("RealtimeAcquisitionService not available; switching to simulation.")
            self._start_simulation(duration_s)
            return
        try:
            self.service = self._acquisition_cls(
                integration_time_ms=self.integration_time_ms,
                resource_string=self.resource_string,
            )
            # connect() must be called before start() — it initialises the
            # spectrometer handle and populates self.service.wavelengths
            hw_iface = "auto" if interface in ("visa", "auto") else interface
            self.service.connect(interface=hw_iface)
            # Store wavelength axis (populated by connect())
            wl = np.asarray(self.service.wavelengths, dtype=float)
            self._live_store.set_wavelengths(wl)
            self.service.register_callback(self._on_sample)
            self.service.start()
            if duration_s:
                threading.Timer(duration_s, self.stop_session).start()
        except Exception as exc:
            log.error("Hardware connection failed: %s — switching to simulation.", exc)
            self._start_simulation(duration_s)

    def _start_simulation(self, duration_s: float | None) -> None:
        """Run a simulated acquisition thread for testing/demo."""
        wl = np.linspace(480.0, 600.0, 1000)
        self._live_store.set_wavelengths(wl)

        def _sim_loop():
            frame = 0
            start = time.monotonic()
            rng = np.random.default_rng(42)
            while self._live_store.is_running():
                elapsed = time.monotonic() - start
                if duration_s and elapsed >= duration_s:
                    break
                peak = 531.5 - 0.1 * min(elapsed / 60.0, 5.0)  # simulated shift
                intensity = (
                    10_000.0
                    - 300.0 * np.exp(-((wl - peak) ** 2) / (2 * 2.0**2))
                    + rng.normal(0, 20, len(wl))
                )
                self._on_sample(
                    {
                        "intensities": intensity,
                        "wavelengths": wl,
                        "timestamp": datetime.now(timezone.utc),  # pass datetime, not str
                        "sample_num": frame,
                    }
                )
                frame += 1
                time.sleep(self.integration_time_ms / 1000.0)
            self._live_store.set_running(False)

        thread = threading.Thread(target=_sim_loop, daemon=True)
        thread.start()
        if duration_s:
            threading.Timer(duration_s + 0.5, self.stop_session).start()

    def _write_session_json(self, meta: dict[str, Any], filename: str) -> None:
        """Write *meta* to ``session_dir/filename`` as JSON."""
        import json

        if self._session_dir is None:
            return
        try:
            path = self._session_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception as exc:
            log.warning("Failed to write session JSON (%s): %s", filename, exc)
