"""
src.inference.realtime_pipeline
================================
Real-time 4-stage spectral processing pipeline.

Stages
------
1. Preprocessing  — denoise, baseline-correct, validate
2. Feature Extraction — peak detection, Δλ via cross-correlation, SNR
3. Calibration — heuristic (Δλ/slope) + optional GPR + optional CNN
4. Quality Control — SNR gate, saturation check, quality score

This module is the **inference entry point** for both the FastAPI server
and the live-sensor Streamlit dashboard.  It is **stateless per-frame**:
``RealTimePipeline.process_spectrum()`` takes a raw spectrum and returns a
``PipelineResult`` — no hidden mutable state between calls (statistics are
accumulated separately in ``_stats``).

Physics reminder: the primary signal is Δλ = λ_gas − λ_reference (nm).
A negative Δλ indicates analyte adsorption (LSPR redshift).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import time
from typing import Any, Callable
import uuid

import numpy as np

from src.features.lspr_features import (
    LSPRReference,
    compute_lspr_reference,
    concentration_from_shift,
    detect_lspr_peak,
    estimate_shift_xcorr,
    extract_lspr_features,
    refine_peak_centroid,
)
from src.preprocessing import (
    als_baseline,
    compute_snr,
    is_valid_spectrum,
    smooth_spectrum,
    spike_rejection,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Per-run configuration for the real-time pipeline.

    All fields have sensible defaults so ``PipelineConfig()`` works
    without any arguments.
    """

    # Sensor / hardware
    integration_time_ms: float = 50.0
    target_wavelength: float = -1.0  # sentinel: auto → midpoint of search window
    acquisition_rate_hz: float = 2.0

    # Stage 1: Preprocessing
    denoising_method: str = "savgol"  # savgol | gaussian | wavelet
    denoising_window: int = 11
    denoising_poly: int = 2
    apply_baseline: bool = False  # For LSPR peak-shift: usually False
    baseline_method: str = "als"
    spike_rejection: bool = True   # Hampel identifier before smoothing
    spike_window: int = 7          # local neighbourhood half-width (pixels)
    spike_threshold: float = 3.0   # rejection threshold (× local σ_MAD)

    # Stage 2: Feature extraction
    # ── SENSOR-SPECIFIC — must be set for your sensor ──────────────────────
    # reference_wavelength: expected peak location for the reference (blank)
    #   spectrum.  Used as the xcorr ROI centre and as the argmax fallback.
    #   Default (-1.0 sentinel) → auto-set to midpoint of search window in
    #   __post_init__.  Set explicitly to your sensor's known reference peak.
    # peak_search_min/max_nm: wavelength window searched for the sensor peak.
    #   Set to the narrowest range that reliably contains your sensor's
    #   response peak across all operating conditions.  A too-wide window
    #   will pick up analyte absorptions or spectrometer noise edges.
    # Default: 350–950 nm covers most CCD spectrometer sensors (visible range).
    # ──────────────────────────────────────────────────────────────────────
    reference_wavelength: float = -1.0  # sentinel: auto → midpoint of search window
    peak_search_min_nm: float = 350.0
    peak_search_max_nm: float = 950.0
    shift_window_nm: float = 20.0
    shift_upsample: int = 10

    # Stage 3: Calibration (heuristic fallback)
    # calibration_slope sign convention:
    #   Negative (e.g. -0.116 nm/ppm) → sensor peak blue-shifts on gas exposure.
    #   Positive                       → sensor peak red-shifts on gas exposure.
    calibration_slope: float = 0.0  # nm/ppm — must be set from calibration; sign depends on sensor
    calibration_intercept: float = 0.0

    # Stage 4: Quality control
    min_snr: float = 4.0
    saturation_threshold: float = 60_000.0

    # Buffer / output
    buffer_size: int = 10_000

    def __post_init__(self) -> None:
        # Auto-set sentinel fields to the midpoint of the search window when
        # the caller did not provide an explicit value (-1.0 sentinel).
        midpoint = (self.peak_search_min_nm + self.peak_search_max_nm) / 2
        if self.reference_wavelength < 0:
            self.reference_wavelength = midpoint
        if self.target_wavelength < 0:
            self.target_wavelength = midpoint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SpectrumData:
    """Mutable container populated incrementally by each pipeline stage."""

    wavelengths: np.ndarray
    intensities: np.ndarray
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sample_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Stage 1 outputs
    processed_intensities: np.ndarray | None = None

    # Stage 2 outputs
    peak_wavelength: float | None = None
    wavelength_shift: float | None = None  # Δλ in nm
    snr: float | None = None
    roi_center: float | None = None

    # Stage 3 outputs
    concentration_ppm: float | None = None
    concentration_std_ppm: float | None = None  # GPR uncertainty
    confidence_score: float = 0.0

    # Stage 3 intelligence outputs (set by SensorOrchestrator if models loaded)
    gas_type: str | None = None
    gpr_uncertainty: float | None = None

    # Stage 3 conformal prediction interval (90% coverage, set when GPR + calibration data present)
    ci_low: float | None = None   # lower bound (ppm)
    ci_high: float | None = None  # upper bound (ppm)

    # Stage 4 outputs
    saturation_flag: bool = False
    quality_score: float = 0.0
    noise_level: float = 0.0


@dataclass
class PipelineResult:
    """Complete output from one pipeline run."""

    success: bool
    spectrum: SpectrumData
    processing_time_ms: float
    stage_results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------


class PreprocessingStage:
    """Apply denoising and optional baseline correction."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict[str, bool]]:
        results: dict[str, bool] = {"denoising": False, "baseline": False}

        wl = spectrum.wavelengths
        raw = spectrum.intensities

        if len(wl) == 0 or len(raw) == 0:
            return spectrum, results

        try:
            # Spike rejection must run BEFORE smoothing.
            # Savitzky-Golay cannot flatten single-pixel spikes (Lorentzian fit
            # will return wrong FWHM if even one spike pixel is in the fit window).
            cleaned = (
                spike_rejection(
                    raw,
                    window=self.config.spike_window,
                    threshold=self.config.spike_threshold,
                )
                if self.config.spike_rejection
                else raw
            )
            results["spike_rejection"] = self.config.spike_rejection

            smoothed = smooth_spectrum(
                cleaned,
                window=self.config.denoising_window,
                poly_order=self.config.denoising_poly,
                method=self.config.denoising_method,
            )
            results["denoising"] = True

            if self.config.apply_baseline:
                # als_baseline() returns the baseline-corrected signal
                # (intensities − z), not the raw baseline estimate.
                corrected = als_baseline(smoothed)
                results["baseline"] = True
            else:
                corrected = smoothed

            spectrum.processed_intensities = corrected

        except Exception as exc:
            log.error("Preprocessing stage error: %s", exc)
            spectrum.metadata["preprocessing_error"] = str(exc)
            spectrum.processed_intensities = raw.copy()

        return spectrum, results


# ---------------------------------------------------------------------------
# Stage 2: Feature Extraction
# ---------------------------------------------------------------------------


class FeatureExtractionStage:
    """Detect LSPR peak and compute wavelength shift (Δλ)."""

    def __init__(
        self,
        config: PipelineConfig,
        reference_intensities: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self._reference: np.ndarray | None = (
            None if reference_intensities is None else reference_intensities.copy()
        )

    def set_reference(self, reference_intensities: np.ndarray) -> None:
        """Update the reference spectrum used for shift calculation."""
        self._reference = reference_intensities.copy()

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict[str, bool]]:
        results: dict[str, bool] = {"peak_detection": False, "shift_calculation": False}

        wl = spectrum.wavelengths
        proc = spectrum.processed_intensities
        if proc is None:
            proc = spectrum.intensities
        if len(wl) == 0:
            return spectrum, results

        try:
            # Peak detection
            peak_wl = detect_lspr_peak(
                wl,
                proc,
                search_min=self.config.peak_search_min_nm,
                search_max=self.config.peak_search_max_nm,
            )
            if peak_wl is not None:
                peak_wl = refine_peak_centroid(wl, proc, peak_wl, half_width_nm=2.0)
                spectrum.peak_wavelength = peak_wl
                results["peak_detection"] = True

            # SNR
            spectrum.snr = compute_snr(proc)

            # Wavelength shift (Δλ) — requires reference spectrum
            if self._reference is not None and len(self._reference) == len(wl):
                delta = estimate_shift_xcorr(
                    wl,
                    proc,
                    self._reference,
                    window_nm=self.config.shift_window_nm,
                    center_nm=self.config.reference_wavelength,
                    upsample=self.config.shift_upsample,
                )
                if delta is not None:
                    spectrum.wavelength_shift = delta
                    results["shift_calculation"] = True
            elif peak_wl is not None:
                # Fallback: shift = current peak − reference wavelength
                spectrum.wavelength_shift = peak_wl - self.config.reference_wavelength
                results["shift_calculation"] = True

        except Exception as exc:
            log.error("Feature extraction stage error: %s", exc)
            spectrum.metadata["feature_extraction_error"] = str(exc)

        return spectrum, results


# ---------------------------------------------------------------------------
# Stage 3: Calibration / Inference
# ---------------------------------------------------------------------------


class CalibrationStage:
    """Estimate concentration from Δλ using heuristic or GPR model."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._gpr_model: Any = None  # set by SensorOrchestrator or set_gpr()
        self._cnn_model: Any = None
        self._reference: np.ndarray | None = None  # raw intensities, set via set_reference()
        self._lspr_ref: LSPRReference | None = None  # lazy-init on first frame with wavelengths
        self._conformal: Any = None  # ConformalCalibrator, set by set_gpr()

    def set_gpr(
        self,
        model: Any,
        X_cal: Any = None,
        y_cal: Any = None,
    ) -> None:
        """Inject a fitted GPR model and calibration data for conformal intervals.

        Parameters
        ----------
        model  : fitted GPRCalibration or PhysicsInformedGPR
        X_cal  : (n, d) calibration features — same feature space the model was fit on
        y_cal  : (n,) calibration targets (concentrations ppm)
        """
        self._gpr_model = model
        if X_cal is not None and y_cal is not None:
            from src.calibration.conformal import ConformalCalibrator
            cal = ConformalCalibrator()
            cal.calibrate(model, np.asarray(X_cal), np.asarray(y_cal))
            self._conformal = cal
        else:
            self._conformal = None

    def set_reference(self, reference_intensities: np.ndarray) -> None:
        """Provide the reference spectrum used by GPR feature extraction."""
        self._reference = reference_intensities.copy()
        self._lspr_ref = None  # invalidate cache — rebuilt on next frame

    def set_calibration(
        self,
        slope: float,
        intercept: float = 0.0,
        reference_wl: float | None = None,
    ) -> None:
        """Update heuristic calibration parameters."""
        self.config.calibration_slope = slope
        self.config.calibration_intercept = intercept
        if reference_wl is not None:
            self.config.reference_wavelength = reference_wl
        log.info(
            "Calibration updated: slope=%.4f nm/ppm, intercept=%.4f",
            slope,
            intercept,
        )

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict[str, bool]]:
        results: dict[str, bool] = {"heuristic": False, "gpr": False, "cnn": False}

        delta = spectrum.wavelength_shift
        if delta is not None:
            conc = concentration_from_shift(
                delta,
                slope=self.config.calibration_slope,
                intercept=self.config.calibration_intercept,
            )
            if conc is not None:
                spectrum.concentration_ppm = conc
                results["heuristic"] = True

        # GPR (if model loaded externally by orchestrator)
        if self._gpr_model is not None and spectrum.processed_intensities is not None:
            try:
                if self._reference is not None:
                    # Lazy-init the reference Lorentzian fit on the first frame.
                    # Wavelengths are only available here (not in set_reference),
                    # so we defer until we have both arrays.
                    if self._lspr_ref is None:
                        self._lspr_ref = compute_lspr_reference(
                            spectrum.wavelengths, self._reference
                        )
                    feat = extract_lspr_features(
                        spectrum.wavelengths,
                        spectrum.processed_intensities,
                        self._reference,
                        lspr_ref=self._lspr_ref,
                    )
                    X = np.array([feat.feature_vector])
                    mean, std = self._gpr_model.predict(X, return_std=True)
                    spectrum.concentration_ppm = float(max(0.0, mean[0]))
                    spectrum.concentration_std_ppm = float(std[0])
                    spectrum.gpr_uncertainty = float(std[0])
                    results["gpr"] = True

                    # Conformal prediction interval (90% coverage, if calibrated)
                    if self._conformal is not None:
                        try:
                            lo, hi = self._conformal.predict_interval(
                                self._gpr_model, X, alpha=0.10
                            )
                            spectrum.ci_low = float(lo[0])
                            spectrum.ci_high = float(hi[0])
                        except Exception as ci_exc:
                            log.debug("Conformal interval failed: %s", ci_exc)
                elif spectrum.wavelength_shift is not None:
                    # Fallback: no reference spectrum available — use Δλ directly
                    # as a 1-D input to the GPR (works when GPR was fit on shifts).
                    X_shift = np.array([[spectrum.wavelength_shift]])
                    mean, std = self._gpr_model.predict(X_shift, return_std=True)
                    spectrum.concentration_ppm = float(max(0.0, mean[0]))
                    spectrum.concentration_std_ppm = float(std[0])
                    spectrum.gpr_uncertainty = float(std[0])
                    results["gpr"] = True
                    if self._conformal is not None:
                        try:
                            lo, hi = self._conformal.predict_interval(
                                self._gpr_model, X_shift, alpha=0.10
                            )
                            spectrum.ci_low = float(lo[0])
                            spectrum.ci_high = float(hi[0])
                        except Exception as ci_exc:
                            log.debug("Conformal interval (shift fallback) failed: %s", ci_exc)
            except Exception as exc:
                log.debug("GPR inference failed: %s", exc)

        # CNN (if model loaded externally by orchestrator)
        if self._cnn_model is not None and spectrum.processed_intensities is not None:
            try:
                gas, conf = self._cnn_model.predict(
                    spectrum.wavelengths, spectrum.processed_intensities
                )
                spectrum.gas_type = gas
                spectrum.confidence_score = float(conf)
                results["cnn"] = True
            except Exception as exc:
                log.debug("CNN inference failed: %s", exc)

        return spectrum, results


# ---------------------------------------------------------------------------
# Stage 4: Quality Control
# ---------------------------------------------------------------------------


class QualityControlStage:
    """Compute quality score; hard-block saturated spectra."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict[str, bool], bool]:
        """
        Returns
        -------
        tuple[SpectrumData, dict, bool]
            ``(spectrum, stage_results, passed)`` — ``passed`` is ``False``
            only on hard-block conditions.
        """
        proc = (
            spectrum.processed_intensities
            if spectrum.processed_intensities is not None
            else spectrum.intensities
        )
        results: dict[str, bool] = {"saturation": True, "snr": True}

        # Hard block: saturation
        if np.any(proc >= self.config.saturation_threshold):
            spectrum.saturation_flag = True
            results["saturation"] = False
            return spectrum, results, False

        # Compute quality score [0, 1]
        snr = spectrum.snr or 0.0
        snr_score = min(1.0, snr / 20.0)  # saturates at SNR=20

        has_peak = 1.0 if spectrum.peak_wavelength is not None else 0.0
        has_shift = 1.0 if spectrum.wavelength_shift is not None else 0.0

        spectrum.quality_score = float(0.5 * snr_score + 0.25 * has_peak + 0.25 * has_shift)

        if snr < self.config.min_snr:
            results["snr"] = False  # soft warning — does not block

        return spectrum, results, True


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------


class RealTimePipeline:
    """Orchestrates the 4-stage real-time processing pipeline.

    Usage
    -----
    ::

        pipeline = RealTimePipeline()
        pipeline.set_calibration(slope=0.116, intercept=0.0)

        result = pipeline.process_spectrum(wavelengths, intensities)
        if result.success:
            print(result.spectrum.concentration_ppm)
    """

    _VERSION = "3.0.0"

    def __init__(
        self,
        config: PipelineConfig | None = None,
        reference_intensities: np.ndarray | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._preprocessing = PreprocessingStage(self.config)
        self._features = FeatureExtractionStage(self.config, reference_intensities)
        self._calibration = CalibrationStage(self.config)
        self._quality = QualityControlStage(self.config)

        self._stats: dict[str, int | float] = {
            "total_processed": 0,
            "total_valid": 0,
            "total_failed": 0,
        }
        self._result_buffer: deque[PipelineResult] = deque(maxlen=self.config.buffer_size)
        self._callbacks: list[Callable[[PipelineResult], None]] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_reference(self, reference_intensities: np.ndarray) -> None:
        """Set or update the reference spectrum for shift calculation."""
        self._features.set_reference(reference_intensities)
        self._calibration.set_reference(reference_intensities)  # invalidates LSPRReference cache
        log.info("Reference spectrum updated (%d points).", len(reference_intensities))

    def set_calibration(
        self,
        slope: float,
        intercept: float = 0.0,
        reference_wl: float | None = None,
    ) -> None:
        """Update heuristic calibration parameters."""
        self._calibration.set_calibration(slope, intercept, reference_wl)

    def register_callback(self, fn: Callable[[PipelineResult], None]) -> None:
        """Register a callback invoked with each ``PipelineResult``."""
        self._callbacks.append(fn)

    def process_spectrum(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        timestamp: datetime | None = None,
        sample_id: str | None = None,
    ) -> PipelineResult:
        """Process one raw spectrum through all 4 pipeline stages.

        Parameters
        ----------
        wavelengths, intensities:
            Raw spectral data arrays.
        timestamp:
            Acquisition time; defaults to ``datetime.now(UTC)``.
        sample_id:
            Optional frame identifier.

        Returns
        -------
        PipelineResult
            Always returns a result (never raises); ``success=False`` on
            hard failures.
        """
        t0 = time.perf_counter()
        self._stats["total_processed"] = int(self._stats["total_processed"]) + 1

        # --- Input validation ---
        wl = np.asarray(wavelengths, dtype=np.float64)
        raw = np.asarray(intensities, dtype=np.float64)

        valid, issues = is_valid_spectrum(
            wl, raw, saturation_threshold=self.config.saturation_threshold
        )
        if not valid or len(wl) < 4:
            msg = "; ".join(issues) if issues else "Invalid spectrum"
            self._stats["total_failed"] = int(self._stats["total_failed"]) + 1
            return PipelineResult(
                success=False,
                spectrum=SpectrumData(wavelengths=wl, intensities=raw),
                processing_time_ms=(time.perf_counter() - t0) * 1000,
                errors=[msg],
            )

        spectrum = SpectrumData(
            wavelengths=wl,
            intensities=raw,
            timestamp=timestamp or datetime.now(timezone.utc),
            sample_id=sample_id or str(uuid.uuid4()),
        )
        stage_results: dict[str, bool] = {}
        all_errors: list[str] = []
        all_warnings: list[str] = list(issues)  # carry soft warnings forward

        # Stage 1
        spectrum, s1 = self._preprocessing.process(spectrum)
        stage_results.update(s1)

        # Stage 2
        spectrum, s2 = self._features.process(spectrum)
        stage_results.update(s2)

        # Stage 3
        spectrum, s3 = self._calibration.process(spectrum)
        stage_results.update(s3)

        # Stage 4
        spectrum, s4, passed = self._quality.process(spectrum)
        stage_results.update(s4)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        success = passed and not all_errors

        if success:
            self._stats["total_valid"] = int(self._stats["total_valid"]) + 1
        else:
            self._stats["total_failed"] = int(self._stats["total_failed"]) + 1

        result = PipelineResult(
            success=success,
            spectrum=spectrum,
            processing_time_ms=elapsed_ms,
            stage_results=stage_results,
            errors=all_errors,
            warnings=all_warnings,
        )

        self._result_buffer.append(result)
        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as exc:
                log.debug("Pipeline callback error: %s", exc)

        return result

    def get_statistics(self) -> dict[str, Any]:
        """Return accumulated pipeline statistics."""
        total = int(self._stats["total_processed"])
        valid = int(self._stats["total_valid"])
        return {
            "total_processed": total,
            "total_valid": valid,
            "total_failed": int(self._stats["total_failed"]),
            "valid_rate": valid / total if total > 0 else 0.0,
            "pipeline_version": self._VERSION,
        }

    def get_recent_results(self, n: int = 100) -> list[PipelineResult]:
        """Return the last *n* results from the buffer."""
        buf = list(self._result_buffer)
        return buf[-n:] if len(buf) > n else buf

    def process_frame(self, frame: Any) -> PipelineResult:
        """Process a :class:`~src.spectrometer.SpectralFrame` through the pipeline.

        Convenience bridge that extracts ``wavelengths`` and ``intensities``
        from *frame* and calls :meth:`process_spectrum`.  Accepts any object
        with ``wavelengths``, ``intensities``, and ``timestamp`` attributes,
        including :class:`~src.spectrometer.SpectralFrame` and plain dicts.

        Parameters
        ----------
        frame :
            A ``SpectralFrame`` (or duck-typed equivalent).

        Returns
        -------
        PipelineResult
            Same result as :meth:`process_spectrum`.

        Example
        -------
        ::

            from src.spectrometer import SpectrometerRegistry
            from src.inference.realtime_pipeline import RealTimePipeline, PipelineConfig

            pipeline = RealTimePipeline(PipelineConfig())
            pipeline.set_calibration(slope=-0.116, intercept=0.0)

            with SpectrometerRegistry.create("simulated") as spec:
                frame = spec.acquire()
                result = pipeline.process_frame(frame)
                if result.success:
                    print(f"{result.spectrum.concentration_ppm:.3f} ppm")
        """
        if isinstance(frame, dict):
            wl = np.asarray(frame["wavelengths"], dtype=np.float64)
            intensities = np.asarray(frame["intensities"], dtype=np.float64)
            ts = frame.get("timestamp")
            sample_id = frame.get("sample_id") or frame.get("serial_number")
        else:
            wl = np.asarray(frame.wavelengths, dtype=np.float64)
            intensities = np.asarray(frame.intensities, dtype=np.float64)
            ts = getattr(frame, "timestamp", None)
            sample_id = getattr(frame, "serial_number", None)

        return self.process_spectrum(
            wavelengths=wl,
            intensities=intensities,
            timestamp=ts,
            sample_id=sample_id,
        )
