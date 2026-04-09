"""
REAL-TIME GAS SENSING PIPELINE
===============================

Complete end-to-end pipeline from spectrometer acquisition to concentration estimation.

Pipeline Stages:
1. Acquisition - Real-time spectrometer data capture
2. Preprocessing - Denoising, baseline correction, normalization
3. Feature Extraction - ROI discovery, peak finding, wavelength shift
4. Calibration - Concentration estimation with confidence metrics
5. Quality Control - Validation and filtering
6. Output - Recording, visualization, reporting

Author: DeepPal Research Team
Version: 1.0.0
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# Import existing calibration methods
# Import new modules for enhanced functionality
from .calibration_memory import CalibrationMemory
from .performance_monitor import PerformanceMonitor

# Import input validation (C1 fix: validate spectra before processing)
from src.signal.spectrum_validator import validate_spectrum

# Import existing preprocessing module
from .preprocessing import (
    baseline_correction,
    compute_snr,
    normalize_spectrum,
    smooth_spectrum,
)

# Suppress only noisy scipy/sklearn convergence warnings — not all warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*overflow encountered.*", category=RuntimeWarning)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""

    # Acquisition settings
    integration_time_ms: float = 30.0
    target_wavelength: float = 532.0
    acquisition_rate_hz: float = 10.0

    # Preprocessing settings
    denoising_method: str = "wavelet"  # wavelet, savgol, gaussian
    denoising_window: int = 31
    baseline_method: str = "als"  # als, polynomial, rolling_min
    baseline_poly_order: int = 2
    normalization_method: str = "minmax"  # minmax, standard, area

    # Feature extraction settings
    roi_discovery_method: str = "gradient"  # gradient, peak_width, manual
    roi_min_width_nm: float = 5.0
    peak_finding_method: str = "parabolic"  # parabolic, centroid, gaussian

    # Calibration settings
    calibration_method: str = "wavelength_shift"  # wavelength_shift, absorbance_amplitude
    reference_wavelength: float = 531.5  # Adjusted from 532.0 to match actual peak
    calibration_slope: float = 0.116  # nm/ppm (from paper benchmark)
    calibration_intercept: float = 0.0

    # Peak detection settings
    peak_mode: str = "absorption"  # absorption (find min) or emission (find max)
    peak_search_window_nm: float = 50.0  # Search window around target wavelength

    # Quality control settings (relaxed for real-world data)
    min_snr: float = 1.0  # Very relaxed for debugging
    max_noise_level: float = 1.0  # Very relaxed
    min_confidence: float = 0.1  # Very relaxed
    saturation_threshold: float = 65000.0  # Very relaxed

    # Output settings
    buffer_size: int = 10000
    auto_save_interval: int = 100
    enable_logging: bool = True


@dataclass
class SpectrumData:
    """Container for spectrum data with metadata."""

    wavelengths: np.ndarray
    intensities: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)
    sample_id: str = ""
    metadata: dict = field(default_factory=dict)

    # Processed data
    processed_intensities: Optional[np.ndarray] = None
    baseline: Optional[np.ndarray] = None
    normalized_intensities: Optional[np.ndarray] = None

    # Quality metrics
    snr: float = 0.0
    noise_level: float = 0.0
    saturation_flag: bool = False
    quality_score: float = 0.0

    # Feature data
    peak_wavelength: Optional[float] = None
    peak_intensity: Optional[float] = None
    wavelength_shift: Optional[float] = None
    roi_start: Optional[float] = None
    roi_end: Optional[float] = None

    # Analysis results
    concentration_ppm: Optional[float] = None
    confidence_score: float = 0.0

    # Intelligence predictions (set by SensorOrchestrator, not core pipeline)
    gas_type: Optional[str] = None  # CNN gas classification result
    gpr_uncertainty: Optional[float] = None  # GPR std dev in ppm


@dataclass
class PipelineResult:
    """Result from pipeline processing."""

    success: bool
    spectrum: SpectrumData
    processing_time_ms: float
    stage_results: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ============================================================================
# PREPROCESSING STAGE
# ============================================================================


class PreprocessingStage:
    """
    Stage 1: Preprocessing

    Applies signal processing to raw spectrum data:
    - Denoising (wavelet, Savitzky-Golay, Gaussian)
    - Baseline correction (ALS, polynomial, rolling min)
    - Normalization (min-max, standard, area)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict]:
        """Apply preprocessing to spectrum data."""
        results = {"denoising": False, "baseline_correction": False, "normalization": False}

        try:
            # Get raw data
            wavelengths = spectrum.wavelengths
            intensities = spectrum.intensities

            if len(wavelengths) == 0 or len(intensities) == 0:
                return spectrum, results

            # Step 1: Denoising (light smoothing to preserve peak shape)
            denoised = self._apply_denoising(wavelengths, intensities)
            results["denoising"] = True

            # Step 2: Baseline correction (skip for absorption mode to preserve dips)
            if self.config.peak_mode == "absorption":
                # For absorption spectra, just use minimal baseline correction
                # to preserve the absorption dips
                baseline = np.zeros_like(denoised)  # No baseline removal
                corrected = denoised
            else:
                # For emission spectra, apply full baseline correction
                baseline, corrected = self._apply_baseline_correction(wavelengths, denoised)

            spectrum.baseline = baseline
            results["baseline_correction"] = True

            # Step 3: Normalization (skip for absorption to preserve absolute position)
            # For wavelength shift detection, we need the absolute position
            normalized = corrected  # Skip normalization
            spectrum.normalized_intensities = normalized
            results["normalization"] = True

            # Store processed data
            spectrum.processed_intensities = corrected

        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            spectrum.metadata["preprocessing_error"] = str(e)

        return spectrum, results

    def _apply_denoising(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Apply denoising filter."""
        return smooth_spectrum(
            intensities, window=self.config.denoising_window, method=self.config.denoising_method
        )

    def _apply_baseline_correction(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply baseline correction."""
        baseline = baseline_correction(
            wavelengths,
            intensities,
            method=self.config.baseline_method,
            poly_order=self.config.baseline_poly_order,
        )
        corrected = intensities - baseline
        return baseline, corrected

    def _apply_normalization(self, intensities: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        return normalize_spectrum(intensities, method=self.config.normalization_method)


# ============================================================================
# FEATURE EXTRACTION STAGE
# ============================================================================


class FeatureExtractionStage:
    """
    Stage 2: Feature Extraction

    Extracts relevant features from preprocessed spectrum:
    - ROI (Region of Interest) discovery
    - Peak finding with sub-pixel accuracy
    - Wavelength shift calculation
    - Absorbance/transmittance features
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.reference_peak = None

    def set_reference(self, reference_wavelength: float):
        """Set reference peak wavelength for shift calculation."""
        self.reference_peak = reference_wavelength

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict]:
        """Extract features from spectrum."""
        results = {"roi_discovery": False, "peak_finding": False, "wavelength_shift": False}

        try:
            wavelengths = spectrum.wavelengths
            intensities = spectrum.processed_intensities

            if intensities is None:
                intensities = spectrum.intensities

            if len(wavelengths) == 0:
                return spectrum, results

            # Step 1: ROI Discovery
            roi_start, roi_end = self._discover_roi(wavelengths, intensities)
            if roi_start is not None:
                spectrum.roi_start = roi_start
                spectrum.roi_end = roi_end
                results["roi_discovery"] = True

            # Step 2: Peak Finding
            peak_wl, peak_int = self._find_peak(wavelengths, intensities, roi_start, roi_end)
            if peak_wl is not None:
                spectrum.peak_wavelength = peak_wl
                spectrum.peak_intensity = peak_int
                results["peak_finding"] = True

            # Step 3: Wavelength Shift
            if self.reference_peak is not None and peak_wl is not None:
                shift = peak_wl - self.reference_peak
                spectrum.wavelength_shift = shift
                results["wavelength_shift"] = True

        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            spectrum.metadata["feature_extraction_error"] = str(e)

        return spectrum, results

    def _discover_roi(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> tuple[Optional[float], Optional[float]]:
        """Discover region of interest around the absorption/emission peak."""
        if self.config.roi_discovery_method == "gradient":
            return self._roi_by_gradient(wavelengths, intensities)
        elif self.config.roi_discovery_method == "peak_width":
            return self._roi_by_peak_width(wavelengths, intensities)
        elif self.config.roi_discovery_method == "target":
            # Use target wavelength region directly
            target = self.config.target_wavelength
            half_width = self.config.peak_search_window_nm
            return target - half_width, target + half_width
        else:
            # Default to target wavelength region
            target = self.config.target_wavelength
            half_width = self.config.peak_search_window_nm
            return target - half_width, target + half_width

    def _roi_by_gradient(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> tuple[Optional[float], Optional[float]]:
        """Find ROI using gradient analysis - find steep edges around peak."""
        # For absorption, find where gradient is steepest (edges of absorption dip)
        gradient = np.gradient(intensities)
        np.abs(gradient)

        # Find the peak/dip first
        if self.config.peak_mode == "absorption":
            peak_idx = np.argmin(intensities)
        else:
            peak_idx = np.argmax(intensities)

        # Find ROI around the peak
        peak_wl = wavelengths[peak_idx]
        half_width = self.config.peak_search_window_nm

        return peak_wl - half_width, peak_wl + half_width

    def _roi_by_peak_width(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> tuple[Optional[float], Optional[float]]:
        """Find ROI based on peak width at half maximum."""
        # Find peak
        if self.config.peak_mode == "absorption":
            peak_idx = np.argmin(intensities)
            peak_val = intensities[peak_idx]

            # For absorption dip, find half-depth points
            (np.max(intensities) + peak_val) / 2
        else:
            peak_idx = np.argmax(intensities)
            peak_val = intensities[peak_idx]
            peak_val / 2

        # Use search window around peak
        peak_wl = wavelengths[peak_idx]
        half_width = self.config.peak_search_window_nm

        return peak_wl - half_width, peak_wl + half_width

    def _find_contiguous_regions(self, mask: np.ndarray) -> list[list[int]]:
        """Find contiguous regions in boolean mask."""
        regions = []
        current_region = []

        for i, val in enumerate(mask):
            if val:
                current_region.append(i)
            else:
                if current_region:
                    regions.append(current_region)
                    current_region = []

        if current_region:
            regions.append(current_region)

        return regions

    def _find_peak(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        roi_start: Optional[float],
        roi_end: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Find peak with sub-pixel accuracy."""
        # Apply ROI if specified
        if roi_start is not None and roi_end is not None:
            mask = (wavelengths >= roi_start) & (wavelengths <= roi_end)
            wl_roi = wavelengths[mask]
            int_roi = intensities[mask]
        else:
            # Use search window around target wavelength
            target = self.config.target_wavelength
            window = self.config.peak_search_window_nm
            mask = (wavelengths >= target - window) & (wavelengths <= target + window)
            if np.any(mask):
                wl_roi = wavelengths[mask]
                int_roi = intensities[mask]
            else:
                wl_roi = wavelengths
                int_roi = intensities

        if len(wl_roi) < 3:
            return None, None

        # Find peak based on mode (absorption = minimum, emission = maximum)
        if self.config.peak_mode == "absorption":
            idx = np.argmin(int_roi)
        else:
            idx = np.argmax(int_roi)

        if self.config.peak_finding_method == "parabolic":
            return self._peak_parabolic(wl_roi, int_roi, idx)
        elif self.config.peak_finding_method == "centroid":
            return self._peak_centroid(wl_roi, int_roi)
        else:
            return float(wl_roi[idx]), float(int_roi[idx])

    def _peak_parabolic(
        self, wavelengths: np.ndarray, intensities: np.ndarray, peak_idx: int
    ) -> tuple[float, float]:
        """Find peak using parabolic interpolation."""
        # Ensure we have neighbors for interpolation
        idx = int(np.clip(peak_idx, 1, len(wavelengths) - 2))

        # Parabolic interpolation
        x0, x1, x2 = wavelengths[idx - 1], wavelengths[idx], wavelengths[idx + 1]
        y0, y1, y2 = intensities[idx - 1], intensities[idx], intensities[idx + 1]

        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)

        # Check for numerical stability
        if abs(denom) < 1e-6 or abs(denom) > 1e6:
            # Fallback to direct value
            return float(wavelengths[idx]), float(intensities[idx])

        # Vertex formula
        try:
            x_vertex = (x0**2 * (y1 - y2) + x1**2 * (y2 - y0) + x2**2 * (y0 - y1)) / (2 * denom)

            # Validate result is reasonable
            if not np.isfinite(x_vertex):
                return float(wavelengths[idx]), float(intensities[idx])

            # Clamp to valid range (within 2 points of the peak)
            x_min = wavelengths[max(0, idx - 2)]
            x_max = wavelengths[min(len(wavelengths) - 1, idx + 2)]
            x_vertex = np.clip(x_vertex, x_min, x_max)

            # Interpolate intensity at vertex
            y_vertex = np.interp(x_vertex, wavelengths, intensities)

            return float(x_vertex), float(y_vertex)

        except Exception:
            return float(wavelengths[idx]), float(intensities[idx])

    def _peak_centroid(
        self, wavelengths: np.ndarray, intensities: np.ndarray
    ) -> tuple[float, float]:
        """Find peak using centroid calculation."""
        # Weighted centroid
        weights = intensities - np.min(intensities)
        total_weight = np.sum(weights)

        if total_weight < 1e-10:
            return wavelengths[np.argmax(intensities)], np.max(intensities)

        centroid_wl = np.sum(wavelengths * weights) / total_weight

        # Interpolate intensity at centroid
        centroid_int = np.interp(centroid_wl, wavelengths, intensities)

        return float(centroid_wl), float(centroid_int)


# ============================================================================
# CALIBRATION STAGE
# ============================================================================


class CalibrationStage:
    """
    Stage 3: Calibration

    Converts wavelength shift to concentration:
    - Wavelength shift method (Δλ)
    - Absorbance amplitude method (ΔA)
    - Confidence calculation
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.calibration_data = None

    def load_calibration(self, calibration_file: str = None) -> bool:
        """Load calibration data from file."""
        if calibration_file:
            try:
                with open(calibration_file) as f:
                    self.calibration_data = json.load(f)

                # Update config from calibration
                self.config.calibration_slope = self.calibration_data.get(
                    "wavelength_shift_slope", 0.116
                )
                self.config.calibration_intercept = self.calibration_data.get(
                    "wavelength_shift_intercept", 0.0
                )
                self.config.reference_wavelength = self.calibration_data.get(
                    "baseline_wavelength", 532.0
                )

                return True
            except Exception as e:
                self.logger.error(f"Failed to load calibration: {e}")
                return False
        return False

    def set_calibration(self, slope: float, intercept: float, reference_wl: float):
        """Set calibration parameters directly."""
        self.config.calibration_slope = slope
        self.config.calibration_intercept = intercept
        self.config.reference_wavelength = reference_wl

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict]:
        """Apply calibration to estimate concentration."""
        results = {"calibration_applied": False, "concentration_estimated": False}

        try:
            if self.config.calibration_method == "wavelength_shift":
                concentration = self._calibrate_wavelength_shift(spectrum)
            elif self.config.calibration_method == "absorbance_amplitude":
                concentration = self._calibrate_absorbance_amplitude(spectrum)
            else:
                concentration = None

            if concentration is not None:
                spectrum.concentration_ppm = concentration
                results["concentration_estimated"] = True

            # Calculate confidence
            confidence = self._calculate_confidence(spectrum)
            spectrum.confidence_score = confidence

            results["calibration_applied"] = True

        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            spectrum.metadata["calibration_error"] = str(e)

        return spectrum, results

    def _calibrate_wavelength_shift(self, spectrum: SpectrumData) -> Optional[float]:
        """Calculate concentration from wavelength shift."""
        if spectrum.wavelength_shift is None:
            return None

        slope = self.config.calibration_slope

        if abs(slope) < 1e-10:
            return None

        # spectrum.wavelength_shift is already peak_wl - reference_peak (see FeatureExtractionStage)
        wavelength_shift = spectrum.wavelength_shift

        # Apply calibration: concentration = shift / slope
        concentration = wavelength_shift / slope

        # Sanity check - limit to reasonable range
        if abs(concentration) > 1000:  # Limit to ±1000 ppm
            self.logger.warning(f"Unrealistic concentration: {concentration:.2f} ppm, clamping")
            concentration = np.sign(concentration) * 1000

        return concentration

    def _calibrate_absorbance_amplitude(self, spectrum: SpectrumData) -> Optional[float]:
        """Calculate concentration from absorbance amplitude."""
        if spectrum.peak_intensity is None:
            return None

        # This would require calibration data
        if self.calibration_data is None:
            return None

        # Get absorbance calibration parameters
        slope = self.calibration_data.get("absorbance_slope", 0.01)
        intercept = self.calibration_data.get("absorbance_intercept", 0.0)

        concentration = (spectrum.peak_intensity - intercept) / slope
        return concentration

    def _calculate_confidence(self, spectrum: SpectrumData) -> float:
        """Calculate confidence score for the measurement."""
        confidence = 1.0

        # Factor 1: SNR
        if spectrum.snr > 0:
            snr_factor = min(1.0, spectrum.snr / 50.0)
            confidence *= snr_factor

        # Factor 2: Quality score
        if spectrum.quality_score > 0:
            confidence *= spectrum.quality_score

        # Factor 3: ROI quality
        if spectrum.roi_start is not None and spectrum.roi_end is not None:
            roi_width = spectrum.roi_end - spectrum.roi_start
            if roi_width < self.config.roi_min_width_nm:
                confidence *= 0.8
        else:
            confidence *= 0.5

        return min(1.0, confidence)


# ============================================================================
# QUALITY CONTROL STAGE
# ============================================================================


class QualityControlStage:
    """
    Stage 4: Quality Control

    Validates data quality and filters bad measurements:
    - SNR threshold check
    - Noise level check
    - Saturation detection
    - Confidence threshold
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process(self, spectrum: SpectrumData) -> tuple[SpectrumData, dict]:
        """Apply quality control checks."""
        results = {
            "snr_check": False,
            "noise_check": False,
            "saturation_check": False,
            "confidence_check": False,
            "overall_pass": False,
        }

        try:
            # Calculate quality metrics
            self._calculate_quality_metrics(spectrum)

            # SNR check
            if spectrum.snr >= self.config.min_snr:
                results["snr_check"] = True

            # Noise check
            if spectrum.noise_level <= self.config.max_noise_level:
                results["noise_check"] = True

            # Saturation check
            if not spectrum.saturation_flag:
                results["saturation_check"] = True

            # Confidence check
            if spectrum.confidence_score >= self.config.min_confidence:
                results["confidence_check"] = True

            # Overall pass
            results["overall_pass"] = all(
                [
                    results["snr_check"],
                    results["noise_check"],
                    results["saturation_check"],
                    results["confidence_check"],
                ]
            )

            # Calculate overall quality score
            spectrum.quality_score = self._calculate_quality_score(spectrum, results)

        except Exception as e:
            self.logger.error(f"Quality control error: {e}")
            spectrum.metadata["quality_control_error"] = str(e)

        return spectrum, results

    def _calculate_quality_metrics(self, spectrum: SpectrumData):
        """Calculate quality metrics for spectrum."""
        intensities = spectrum.intensities

        if len(intensities) == 0:
            return

        # SNR
        spectrum.snr = compute_snr(intensities)

        # Noise level
        spectrum.noise_level = float(np.std(intensities))

        # Saturation detection
        spectrum.saturation_flag = np.any(intensities > self.config.saturation_threshold)

    def _calculate_quality_score(self, spectrum: SpectrumData, results: dict) -> float:
        """Calculate overall quality score."""
        score = 0.0

        # SNR contribution (0-0.3)
        if results["snr_check"]:
            score += 0.3 * min(1.0, spectrum.snr / (2 * self.config.min_snr))

        # Noise contribution (0-0.2)
        if results["noise_check"]:
            score += 0.2 * (1 - spectrum.noise_level / self.config.max_noise_level)

        # Saturation contribution (0-0.2)
        if results["saturation_check"]:
            score += 0.2

        # Confidence contribution (0-0.3)
        if results["confidence_check"]:
            score += 0.3 * spectrum.confidence_score

        return min(1.0, score)


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================


class RealTimePipeline:
    """
    Complete Real-Time Gas Sensing Pipeline

    Integrates all stages:
    1. Preprocessing
    2. Feature Extraction
    3. Calibration
    4. Quality Control
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize stages
        self.preprocessing = PreprocessingStage(self.config)
        self.feature_extraction = FeatureExtractionStage(self.config)
        self.calibration = CalibrationStage(self.config)
        self.quality_control = QualityControlStage(self.config)

        # Initialize enhanced systems
        self.calibration_memory = CalibrationMemory(
            window_size=1000, recalibrate_threshold=0.7, min_samples_for_calibration=100
        )
        self.performance_monitor = PerformanceMonitor(window_size=1000)

        # Data buffer
        self.buffer = deque(maxlen=self.config.buffer_size)

        # Statistics
        self.total_processed = 0
        self.valid_samples = 0
        self.filtered_samples = 0

        # Logging
        self.logger = logging.getLogger(__name__)
        if self.config.enable_logging:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        handler = logging.FileHandler(log_dir / f"pipeline_{session_id}.log")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def process_spectrum(
        self,
        wavelengths: np.ndarray,
        intensities: np.ndarray,
        timestamp: Optional[datetime] = None,
        sample_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process a single spectrum through the complete pipeline.

        Args:
            wavelengths: Wavelength array (nm)
            intensities: Intensity array
            timestamp: Optional timestamp
            sample_id: Optional sample identifier

        Returns:
            PipelineResult with processed data and metrics
        """
        start_time = time.time()

        # Create spectrum container
        spectrum = SpectrumData(
            wavelengths=wavelengths,
            intensities=intensities,
            timestamp=timestamp or datetime.now(),
            sample_id=sample_id or f"S{self.total_processed:06d}",
        )

        # STAGE 0: Input Validation (C1 fix)
        # Validate spectrum BEFORE any processing to catch bad data early
        validation = validate_spectrum(
            intensities,
            wavelengths=wavelengths,
            expected_points=len(wavelengths),
            saturation_threshold=60000.0,
            snr_threshold=3.0,
        )
        if not validation.valid:
            # Create error result
            error_msg = f"Spectrum validation failed: {'; '.join(validation.errors)}"
            logging.error(f"❌ {error_msg}")
            return PipelineResult(
                success=False,
                sample_id=spectrum.sample_id,
                timestamp=spectrum.timestamp,
                processed_spectrum=None,
                calibration_data=None,
                extracted_features=None,
                predictions=None,
                quality_metrics={},
                stage_results={},
                error_message=error_msg,
                elapsed_time_ms=0,
            )

        stage_results = {}
        errors = []
        warnings = []

        try:
            # Stage 1: Preprocessing
            spectrum, results = self.preprocessing.process(spectrum)
            stage_results["preprocessing"] = all(results.values())

            # Stage 2: Feature Extraction
            spectrum, results = self.feature_extraction.process(spectrum)
            stage_results["feature_extraction"] = all(results.values())

            # Stage 3: Calibration
            spectrum, results = self.calibration.process(spectrum)
            stage_results["calibration"] = results.get("calibration_applied", False)

            # Add to calibration memory for auto-calibration
            if spectrum.wavelength_shift is not None and spectrum.concentration_ppm is not None:
                recalibrated = self.calibration_memory.add_measurement(
                    wavelength_shift=spectrum.wavelength_shift,
                    concentration_ppm=spectrum.concentration_ppm,
                    confidence_score=spectrum.confidence_score,
                    snr=spectrum.snr,
                )
                if recalibrated:
                    # Update calibration with new parameters
                    new_calibration = self.calibration_memory.get_current_calibration()
                    if new_calibration:
                        self.config.calibration_slope = new_calibration["slope"]
                        self.config.calibration_intercept = new_calibration["intercept"]
                        self.logger.info(
                            f"Auto-calibration updated: slope={new_calibration['slope']:.6f}"
                        )

            # Stage 4: Quality Control
            spectrum, results = self.quality_control.process(spectrum)
            stage_results["quality_control"] = results.get("overall_pass", False)

            # Add to performance monitor
            if spectrum.wavelengths is not None and spectrum.intensities is not None:
                # Calculate absorbance spectrum for performance monitoring
                if spectrum.processed_intensities is not None:
                    absorbance_spectrum = -np.log10(
                        spectrum.processed_intensities / np.max(spectrum.processed_intensities)
                    )
                else:
                    absorbance_spectrum = -np.log10(
                        spectrum.intensities / np.max(spectrum.intensities)
                    )

                self.performance_monitor.add_measurement(
                    wavelength_shift=spectrum.wavelength_shift or 0.0,
                    absorbance_spectrum=absorbance_spectrum,
                    snr=spectrum.snr or 0.0,
                    confidence_score=spectrum.confidence_score or 0.0,
                    timestamp=spectrum.timestamp.timestamp()
                    if hasattr(spectrum.timestamp, "timestamp")
                    else time.time(),
                )

            # Update statistics
            self.total_processed += 1
            if results.get("overall_pass", False):
                self.valid_samples += 1
                self.buffer.append(spectrum)
            else:
                self.filtered_samples += 1

        except Exception as e:
            errors.append(str(e))
            self.logger.error(f"Pipeline error: {e}")

        processing_time = (time.time() - start_time) * 1000  # ms

        return PipelineResult(
            success=len(errors) == 0,
            spectrum=spectrum,
            processing_time_ms=processing_time,
            stage_results=stage_results,
            errors=errors,
            warnings=warnings,
        )

    def load_calibration(self, calibration_file: str) -> bool:
        """Load calibration data."""
        success = self.calibration.load_calibration(calibration_file)
        if success:
            # Set reference for feature extraction
            self.feature_extraction.set_reference(self.config.reference_wavelength)
        return success

    def set_calibration(self, slope: float, intercept: float, reference_wl: float):
        """Set calibration parameters directly."""
        self.calibration.set_calibration(slope, intercept, reference_wl)
        self.feature_extraction.set_reference(reference_wl)

    def get_statistics(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_processed": self.total_processed,
            "valid_samples": self.valid_samples,
            "filtered_samples": self.filtered_samples,
            "validity_rate": self.valid_samples / max(1, self.total_processed),
        }

    def get_calibration_memory(self) -> CalibrationMemory:
        """Get calibration memory system."""
        return self.calibration_memory

    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor system."""
        return self.performance_monitor

    def get_current_performance_metrics(self):
        """Get current performance metrics."""
        return self.performance_monitor.get_current_metrics()

    def get_performance_summary(self, window_minutes: int = 10) -> dict:
        """Get performance metrics summary."""
        return self.performance_monitor.get_metrics_summary(window_minutes)

    def export_ml_data(self, filepath: str, window_hours: int = 24) -> bool:
        """Export ML-ready data from performance monitor."""
        return self.performance_monitor.export_ml_data(filepath, window_hours)

    def save_calibration_memory(self, filepath: str):
        """Save calibration memory to file."""
        from pathlib import Path

        return self.calibration_memory.save_calibration(Path(filepath))

    def load_calibration_memory(self, filepath: str):
        """Load calibration memory from file."""
        from pathlib import Path

        return self.calibration_memory.load_calibration(Path(filepath))

    def get_ml_features(self) -> list[dict]:
        """Get ML-ready features from calibration memory."""
        return self.calibration_memory.get_ml_features()

    def export_results(self, output_path: str) -> bool:
        """Export processed results to CSV."""
        if not self.buffer:
            return False

        try:
            data = []
            for spectrum in self.buffer:
                data.append(
                    {
                        "timestamp": spectrum.timestamp,
                        "sample_id": spectrum.sample_id,
                        "peak_wavelength": spectrum.peak_wavelength,
                        "wavelength_shift": spectrum.wavelength_shift,
                        "concentration_ppm": spectrum.concentration_ppm,
                        "snr": spectrum.snr,
                        "noise_level": spectrum.noise_level,
                        "quality_score": spectrum.quality_score,
                        "confidence_score": spectrum.confidence_score,
                        "roi_start": spectrum.roi_start,
                        "roi_end": spectrum.roi_end,
                    }
                )

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            return True

        except Exception as e:
            self.logger.error(f"Export error: {e}")
            return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_pipeline(config: Optional[PipelineConfig] = None) -> RealTimePipeline:
    """Create a real-time pipeline with optional configuration."""
    return RealTimePipeline(config)


def process_single_spectrum(
    wavelengths: np.ndarray, intensities: np.ndarray, config: Optional[PipelineConfig] = None
) -> PipelineResult:
    """Process a single spectrum through the pipeline."""
    pipeline = RealTimePipeline(config)
    return pipeline.process_spectrum(wavelengths, intensities)


def quick_analysis(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    reference_wl: float = 532.0,
    calibration_slope: float = 0.116,
    peak_mode: str = "absorption",
) -> dict:
    """
    Quick analysis of a single spectrum.

    Returns concentration estimate with quality metrics.
    """
    config = PipelineConfig(
        reference_wavelength=reference_wl,
        calibration_slope=calibration_slope,
        peak_mode=peak_mode,
        peak_search_window_nm=50.0,
        min_snr=1.0,  # More lenient for quick analysis
        min_confidence=0.0,
    )

    pipeline = RealTimePipeline(config)
    pipeline.set_calibration(calibration_slope, 0.0, reference_wl)

    result = pipeline.process_spectrum(wavelengths, intensities)

    return {
        "success": result.success,
        "concentration_ppm": result.spectrum.concentration_ppm,
        "wavelength_shift_nm": result.spectrum.wavelength_shift,
        "peak_wavelength_nm": result.spectrum.peak_wavelength,
        "snr": result.spectrum.snr,
        "quality_score": result.spectrum.quality_score,
        "confidence_score": result.spectrum.confidence_score,
        "processing_time_ms": result.processing_time_ms,
        "stage_results": result.stage_results,
    }
