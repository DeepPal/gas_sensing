#!/usr/bin/env python3
"""
Au-MIP LSPR Gas Sensing Platform — Unified CLI Entry Point
===========================================================

Thin wrapper that delegates to existing, well-tested modules.  No business
logic lives here; this file only wires arguments to the correct subsystem.

Modes
-----
  sensor      Real-time acquisition via CCS200 + SensorOrchestrator
  realtime    Lightweight real-time loop using RealTimePipeline directly
  simulate    Synthetic spectra (no hardware required)
  deployable  Real-time with auto-calibration, performance monitoring, ML export
  batch       Offline analysis of a directory of CSV spectra

Examples
--------
  python run.py --mode sensor --gas Ethanol --duration 3600
  python run.py --mode simulate --duration 30
  python run.py --mode batch --data data/JOY_Data/Ethanol
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# Windows console UTF-8 fix (must happen before any output)
# Unicode symbols (✗ ⚠ ✓) in hardware error messages would crash cp1252.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Suppress only hardware driver / VISA warnings — not all warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyvisa")
warnings.filterwarnings("ignore", message=".*VISA.*", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Project root on sys.path (enables absolute imports from any working dir)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging — must be configured before any gas_analysis import so that
# module-level loggers inherit the correct handler.
# ---------------------------------------------------------------------------
from gas_analysis.logging_setup import configure_logging  # noqa: E402

_LOG_FILE = PROJECT_ROOT / "logs" / "run.log"
configure_logging(level=logging.INFO, log_file=_LOG_FILE)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

from gas_analysis.core.realtime_pipeline import (  # noqa: E402
    PipelineConfig,
    PipelineResult,
    RealTimePipeline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_GAS_LABELS = {
    "Ethanol",
    "EtOH",
    "IPA",
    "Isopropanol",
    "MeOH",
    "Methanol",
    "MixVOC",
    "Mix",
    "unknown",
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """User-facing configuration that maps to :class:`PipelineConfig`.

    All physical constants have documentation references so researchers can
    trace their origin.
    """

    integration_time_ms: float = 30.0
    #: LSPR reference peak for Au nanoparticles (green region, ~531–532 nm)
    target_wavelength: float = 532.0
    #: Literature sensitivity for ethanol on Au-MIP: 0.116 nm/ppm
    calibration_slope: float = 0.116
    calibration_intercept: float = 0.0
    reference_wavelength: float = 532.0
    session_id: str = field(default="")
    data_dir: str = "output"

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to the internal :class:`PipelineConfig` representation."""
        return PipelineConfig(
            integration_time_ms=self.integration_time_ms,
            target_wavelength=self.target_wavelength,
            calibration_slope=self.calibration_slope,
            calibration_intercept=self.calibration_intercept,
            reference_wavelength=self.reference_wavelength,
        )


# ---------------------------------------------------------------------------
# Spectrometer interface
# ---------------------------------------------------------------------------


class SpectrometerInterface:
    """Thin wrapper around the hardware acquisition module.

    Automatically falls back to simulation mode when the CCS200 spectrometer
    is unavailable (no VISA resources, DLL error, etc.).
    """

    # Reference LSPR peak used in simulated spectra (nm)
    _SIM_CENTER_NM = 531.5
    _SIM_WIDTH_NM = 1.5
    _SIM_AMPLITUDE = 50.0
    _SIM_BASELINE = 10_000.0
    _SIM_NOISE_STD = 30.0
    _SIM_POINTS = 1_000

    def __init__(self, config: Config) -> None:
        self.config = config
        self._service = None
        self.wavelengths: np.ndarray | None = None
        self.is_simulation = False

    def connect(self, resource: str | None = None) -> bool:
        """Connect to the spectrometer.

        Returns ``True`` on success (hardware or simulation); ``False`` only
        if simulation initialisation itself fails (should never happen).
        """
        try:
            from gas_analysis.acquisition.ccs200_realtime import (
                RealtimeAcquisitionService,
            )

            self._service = RealtimeAcquisitionService(
                integration_time_ms=self.config.integration_time_ms,
                target_wavelength=self.config.target_wavelength,
                resource_string=resource,
            )
            self._service.connect()
            self.wavelengths = self._service.wavelengths
            log.info("Connected to CCS200 spectrometer")
            return True

        except Exception as exc:
            log.warning("Spectrometer unavailable (%s) — using simulation mode", exc)
            return self._init_simulation()

    def _init_simulation(self) -> bool:
        self.is_simulation = True
        self.wavelengths = np.linspace(400, 700, self._SIM_POINTS)
        log.info("Simulation mode initialised (%d wavelength points)", self._SIM_POINTS)
        return True

    def acquire(self) -> dict | None:
        """Return the next spectrum sample dict."""
        if self.is_simulation:
            return self._simulate_spectrum()
        if self._service:
            sample = self._service.get_latest_sample()
            if sample:
                sample["wavelengths"] = self.wavelengths
            return sample
        return None

    def _simulate_spectrum(self) -> dict:
        """Generate a realistic Gaussian-absorption LSPR spectrum."""
        wl = self.wavelengths
        noise = np.random.normal(0, self._SIM_NOISE_STD, len(wl))
        absorption = self._SIM_AMPLITUDE * np.exp(
            -((wl - self._SIM_CENTER_NM) ** 2) / (2 * self._SIM_WIDTH_NM**2)
        )
        intensities = self._SIM_BASELINE + noise - absorption
        return {
            "timestamp": time.time(),
            "sample_num": int(time.time() * 1000),
            "wavelengths": wl,
            "intensities": intensities,
            "target_intensity": float(intensities[len(wl) // 2]),
            "integration_ms": self.config.integration_time_ms,
        }

    def close(self) -> None:
        """Release the spectrometer connection."""
        if self._service:
            try:
                self._service.stop()
            except Exception as exc:
                log.warning("Error closing spectrometer: %s", exc)


# ---------------------------------------------------------------------------
# Main system
# ---------------------------------------------------------------------------


class GasSensingSystem:
    """Orchestrates spectrometer acquisition through :class:`RealTimePipeline`.

    This class is intentionally lightweight.  All signal processing and
    calibration logic lives in :mod:`gas_analysis.core.realtime_pipeline`.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.pipeline = RealTimePipeline(self.config.to_pipeline_config())
        self.spectrometer = SpectrometerInterface(self.config)
        self._running = False
        self._stats: dict[str, int] = {"total": 0, "valid": 0, "filtered": 0}
        self._results: deque[PipelineResult] = deque(maxlen=10_000)

        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        log.info("GasSensingSystem ready (session=%s)", self.config.session_id)

    # ------------------------------------------------------------------
    # Public run modes
    # ------------------------------------------------------------------

    def run_realtime(self, duration: float = 60.0, resource: str | None = None) -> None:
        """Acquire and process spectra for *duration* seconds."""
        log.info(
            "=== REAL-TIME MODE | duration=%.0fs | target=%.1fnm ===",
            duration,
            self.config.target_wavelength,
        )

        if not self.spectrometer.connect(resource):
            log.error("Failed to initialise acquisition — aborting")
            return

        self.pipeline.set_calibration(
            slope=self.config.calibration_slope,
            intercept=self.config.calibration_intercept,
            reference_wl=self.config.reference_wavelength,
        )

        self._running = True
        start = time.time()

        try:
            while self._running:
                if time.time() - start >= duration:
                    break
                self._process_one_frame()
                time.sleep(0.01)

        except KeyboardInterrupt:
            log.info("Acquisition interrupted by user")
        finally:
            self.spectrometer.close()
            self._running = False
            self._save_results()

    def run_simulation(self, duration: float = 10.0) -> None:
        """Run the pipeline on synthetic spectra (no hardware needed)."""
        log.info("=== SIMULATION MODE | duration=%.0fs ===", duration)
        self.spectrometer._init_simulation()
        self.pipeline.set_calibration(
            slope=self.config.calibration_slope,
            intercept=self.config.calibration_intercept,
            reference_wl=self.config.reference_wavelength,
        )
        self.run_realtime(duration)

    def run_deployable(self, duration: float = 60.0, resource: str | None = None) -> None:
        """Real-time mode with auto-calibration, performance monitoring, and ML export."""
        log.info("=== DEPLOYABLE MODE | duration=%.0fs ===", duration)

        if not self.spectrometer.connect(resource):
            log.error("Failed to initialise acquisition — aborting")
            return

        self.pipeline.set_calibration(
            slope=self.config.calibration_slope,
            intercept=self.config.calibration_intercept,
            reference_wl=self.config.reference_wavelength,
        )

        ml_dir = Path(self.config.data_dir) / "ml_dataset"
        ml_dir.mkdir(parents=True, exist_ok=True)

        self._running = True
        start = time.time()

        try:
            while self._running:
                if time.time() - start >= duration:
                    break
                self._process_one_frame()

                # Periodic ML feature export
                if self._stats["total"] % 100 == 0 and self._stats["total"] > 0:
                    ml_file = (
                        ml_dir
                        / f"ml_features_{self.config.session_id}_{self._stats['total']:06d}.csv"
                    )
                    self.pipeline.export_ml_data(str(ml_file))
                    log.debug("ML features exported → %s", ml_file)

                time.sleep(0.01)

        except KeyboardInterrupt:
            log.info("Deployable session interrupted by user")
        finally:
            self.spectrometer.close()
            self._running = False
            self._save_deployable_results()

    def run_batch(self, data_path: str) -> None:
        """Run offline batch analysis on a directory of CSV files."""
        log.info("=== BATCH MODE | data=%s ===", data_path)
        try:
            from pathlib import Path

            from gas_analysis.core.pipeline import run_full_pipeline

            data_dir = Path(data_path)
            parent_dir = data_dir.parent

            # Auto-detect reference spectrum: look for ref*.csv in parent or data dir
            ref_candidates = sorted(parent_dir.glob("ref*.csv")) + sorted(data_dir.glob("ref*.csv"))
            if ref_candidates:
                ref_path = str(ref_candidates[0])
                log.info("Auto-detected reference spectrum: %s", ref_path)
            else:
                # Fall back to first CSV found in data dir
                first_csv = next(data_dir.rglob("*.csv"), None)
                if first_csv is None:
                    log.error("No CSV files found in %s", data_path)
                    return
                ref_path = str(first_csv)
                log.warning("No ref*.csv found — using first spectrum as reference: %s", ref_path)

            out_root = str(Path("output") / "batch" / data_dir.name)
            run_full_pipeline(data_path, ref_path=ref_path, out_root=out_root)
            log.info("Batch processing complete → %s", out_root)
        except ImportError as exc:
            log.error("Batch pipeline not available: %s", exc)
        except Exception as exc:
            log.error("Batch processing failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_one_frame(self) -> None:
        sample = self.spectrometer.acquire()
        if not sample:
            return

        wl = np.asarray(sample.get("wavelengths", []))
        intensities = np.asarray(sample.get("intensities", []))

        if wl.size == 0 or intensities.size == 0:
            return

        result = self.pipeline.process_spectrum(wl, intensities)
        self._stats["total"] += 1

        if result.success:
            self._stats["valid"] += 1
            self._results.append(result)
        else:
            self._stats["filtered"] += 1

        if self._stats["total"] % 10 == 0:
            self._log_progress(result)

    def _log_progress(self, result: PipelineResult) -> None:
        status = "OK  " if result.success else "FAIL"
        peak = result.spectrum.peak_wavelength or 0.0
        conc = result.spectrum.concentration_ppm or 0.0
        snr = result.spectrum.snr or 0.0
        log.info(
            "Sample %5d | Peak: %6.2f nm | Conc: %6.2f ppm | SNR: %5.1f | %s",
            self._stats["total"],
            peak,
            conc,
            snr,
            status,
        )

    def _save_results(self) -> None:
        concentrations = [
            r.spectrum.concentration_ppm
            for r in self._results
            if r.spectrum.concentration_ppm is not None
        ]
        summary = {
            "session_id": self.config.session_id,
            "total_samples": self._stats["total"],
            "valid_samples": self._stats["valid"],
            "validity_rate_pct": self._stats["valid"] / max(self._stats["total"], 1) * 100,
            "mean_concentration_ppm": float(np.mean(concentrations)) if concentrations else None,
            "std_concentration_ppm": float(np.std(concentrations)) if concentrations else None,
        }
        out = Path(self.config.data_dir) / f"summary_{self.config.session_id}.json"
        out.write_text(json.dumps(summary, indent=2))
        log.info("Session summary → %s", out)
        log.info(
            "Total: %d | Valid: %d | Rate: %.1f%%",
            summary["total_samples"],
            summary["valid_samples"],
            summary["validity_rate_pct"],
        )

    def _save_deployable_results(self) -> None:
        pipeline_stats = self.pipeline.get_statistics()
        perf_summary = self.pipeline.get_performance_summary(window_minutes=30)
        calib_memory = self.pipeline.get_calibration_memory()

        calib_file = (
            Path(self.config.data_dir) / f"calibration_memory_{self.config.session_id}.json"
        )
        calib_memory.save_calibration(calib_file)

        perf_file = (
            Path(self.config.data_dir) / f"performance_summary_{self.config.session_id}.json"
        )
        perf_file.write_text(json.dumps(perf_summary, indent=2, default=str))

        log.info("=== DEPLOYABLE SESSION SUMMARY ===")
        log.info(
            "Total: %d | Valid: %d | Rate: %.1f%%",
            pipeline_stats["total_processed"],
            pipeline_stats["valid_samples"],
            pipeline_stats["validity_rate"],
        )
        log.info("Calibration saved → %s", calib_file)
        log.info("Performance saved → %s", perf_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Au-MIP LSPR Gas Sensing Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["realtime", "simulate", "deployable", "batch", "sensor"],
        default="simulate",
        help="Operating mode (default: simulate)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Acquisition duration (default: 60)",
    )
    parser.add_argument(
        "--data",
        type=str,
        metavar="PATH",
        help="Data directory for batch mode",
    )
    parser.add_argument(
        "--resource",
        type=str,
        metavar="VISA_STRING",
        help="Spectrometer VISA resource string (e.g. USB0::0x1313::...)",
    )
    parser.add_argument(
        "--target-wavelength",
        type=float,
        default=532.0,
        metavar="NM",
        help="Expected LSPR peak wavelength in nm (default: 532.0)",
    )
    parser.add_argument(
        "--calibration-slope",
        type=float,
        default=0.116,
        metavar="NM_PPM",
        help="Calibration sensitivity in nm/ppm (default: 0.116)",
    )
    parser.add_argument(
        "--gas",
        type=str,
        default="unknown",
        metavar="LABEL",
        help=(
            f"Analyte label saved in session metadata. "
            f"Known labels: {', '.join(sorted(ALLOWED_GAS_LABELS))}. "
            "Any string is accepted; unknown labels generate a warning."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log verbosity (default: INFO)",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed arguments and emit early, actionable error messages."""
    if args.mode == "batch" and not args.data:
        log.error("--data PATH is required for batch mode")
        sys.exit(1)

    if args.data and not Path(args.data).exists():
        log.error("Data path does not exist: %s", args.data)
        sys.exit(1)

    if args.duration <= 0:
        log.error("--duration must be positive, got %.1f", args.duration)
        sys.exit(1)

    if args.gas not in ALLOWED_GAS_LABELS:
        log.warning(
            "Gas label '%s' is not in the known-labels list %s. "
            "Session metadata will record it as-is; calibration lookup may not match.",
            args.gas,
            sorted(ALLOWED_GAS_LABELS),
        )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Reconfigure console level if user asked for something different from INFO
    level = getattr(logging, args.log_level)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            handler.setLevel(level)

    _validate_args(args)

    log.info("Mode: %s | Gas: %s | Duration: %.0fs", args.mode, args.gas, args.duration)

    # ----------------------------------------------------------------
    # Sensor mode — full SensorOrchestrator path
    # ----------------------------------------------------------------
    if args.mode == "sensor":
        from src.inference.orchestrator import SensorOrchestrator

        try:
            from config.config_loader import load_config

            cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
        except Exception as exc:
            log.warning("Could not load config.yaml (%s) — using defaults", exc)
            cfg = {}

        orch = SensorOrchestrator.from_config(cfg)
        if args.resource:
            orch.resource_string = args.resource

        log.info("Starting sensor session (gas=%s, duration=%.0fs)", args.gas, args.duration)
        log.info("Launch dashboard: streamlit run dashboard/app.py")
        log.info("Press Ctrl+C to stop early")

        try:
            orch.start_session(gas_label=args.gas, duration_s=args.duration)
        except RuntimeError as exc:
            log.error("%s", exc)
            log.info(
                "Tip: use the dashboard Connect button instead — "
                "streamlit run dashboard/app.py → Live Sensor tab"
            )
            sys.exit(1)

        try:
            time.sleep(args.duration)
        except KeyboardInterrupt:
            log.info("Sensor session interrupted by user")
        finally:
            session_dir = orch.stop_session()
            if session_dir:
                log.info("Session data saved → %s", session_dir)
        return

    # ----------------------------------------------------------------
    # All other modes — lightweight GasSensingSystem path
    # ----------------------------------------------------------------
    config = Config(
        target_wavelength=args.target_wavelength,
        calibration_slope=args.calibration_slope,
    )
    system = GasSensingSystem(config)

    if args.mode == "realtime":
        system.run_realtime(args.duration, args.resource)
    elif args.mode == "simulate":
        system.run_simulation(args.duration)
    elif args.mode == "deployable":
        system.run_deployable(args.duration, args.resource)
    elif args.mode == "batch":
        system.run_batch(args.data)


if __name__ == "__main__":
    main()
