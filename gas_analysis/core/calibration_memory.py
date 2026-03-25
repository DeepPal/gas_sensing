"""
Calibration Memory System for Auto-Calibration
==========================================

Maintains rolling memory of measurements and manages
automatic calibration updates based on data quality and drift detection.
"""

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CalibrationPoint:
    """Single calibration data point."""

    timestamp: float
    wavelength_shift: float
    concentration_ppm: float
    confidence_score: float
    snr: float
    calibration_version: int


@dataclass
class CalibrationStats:
    """Calibration statistics for quality assessment."""

    mean_shift: float
    std_shift: float
    drift_rate: float  # nm/minute
    confidence_score: float
    sample_count: int
    last_update: float


class CalibrationMemory:
    """Memory-based auto-calibration system."""

    def __init__(
        self,
        window_size: int = 1000,
        recalibrate_threshold: float = 0.7,
        min_samples_for_calibration: int = 100,
    ):
        """
        Initialize calibration memory.

        Args:
            window_size: Number of measurements to keep in memory
            recalibrate_threshold: Confidence threshold for auto-recalibration
            min_samples_for_calibration: Minimum samples needed for calibration
        """
        self.window_size = window_size
        self.recalibrate_threshold = recalibrate_threshold
        self.min_samples_for_calibration = min_samples_for_calibration

        # Data storage
        self.measurements = deque(maxlen=window_size)
        self.calibration_history = []
        self.current_calibration = None
        self.calibration_version = 0

        # Performance tracking
        self.stats = CalibrationStats(
            mean_shift=0.0,
            std_shift=0.0,
            drift_rate=0.0,
            confidence_score=1.0,
            sample_count=0,
            last_update=0.0,
        )

        # Setup logging
        self.logger = logging.getLogger("CalibrationMemory")

    def add_measurement(
        self, wavelength_shift: float, concentration_ppm: float, confidence_score: float, snr: float
    ) -> bool:
        """
        Add a new measurement to memory.

        Returns:
            True if calibration was updated, False otherwise
        """
        timestamp = datetime.now().timestamp()

        # Create calibration point
        point = CalibrationPoint(
            timestamp=timestamp,
            wavelength_shift=wavelength_shift,
            concentration_ppm=concentration_ppm,
            confidence_score=confidence_score,
            snr=snr,
            calibration_version=self.calibration_version,
        )

        # Add to memory
        self.measurements.append(point)
        self.stats.sample_count = len(self.measurements)
        self.stats.last_update = timestamp

        # Update statistics
        self._update_statistics()

        # Check if recalibration is needed
        if self._should_recalibrate():
            self._perform_recalibration()
            return True

        return False

    def _update_statistics(self):
        """Update calibration statistics from current measurements."""
        if len(self.measurements) < 10:
            return

        # Extract recent measurements
        recent_measurements = list(self.measurements)[-min(100, len(self.measurements)) :]

        # Calculate statistics
        shifts = [m.wavelength_shift for m in recent_measurements if m.wavelength_shift is not None]
        confidences = [m.confidence_score for m in recent_measurements]

        if shifts:
            self.stats.mean_shift = np.mean(shifts)
            self.stats.std_shift = np.std(shifts)

            # Calculate drift rate (nm/minute)
            if len(recent_measurements) >= 2:
                time_span = (
                    recent_measurements[-1].timestamp - recent_measurements[0].timestamp
                ) / 60.0
                if time_span > 0:
                    shift_change = (
                        recent_measurements[-1].wavelength_shift
                        - recent_measurements[0].wavelength_shift
                    )
                    self.stats.drift_rate = shift_change / time_span

            self.stats.confidence_score = np.mean(confidences)

    def _should_recalibrate(self) -> bool:
        """Determine if recalibration is needed."""
        if len(self.measurements) < self.min_samples_for_calibration:
            return False

        # Check confidence threshold (more conservative)
        if self.stats.confidence_score < 0.4:  # Increased from 0.7
            self.logger.info(f"Confidence {self.stats.confidence_score:.3f} below threshold 0.4")
            return True

        # Check drift rate (more lenient)
        if abs(self.stats.drift_rate) > 0.5:  # Increased from 0.1 nm/minute
            self.logger.info(f"Drift rate {self.stats.drift_rate:.4f} nm/min exceeds threshold 0.5")
            return True

        # Check stability (more lenient)
        if self.stats.std_shift > 5.0:  # Increased from 2nm standard deviation
            self.logger.info(f"Shift variance {self.stats.std_shift:.3f} nm indicates instability")
            return True

        return False

    def _perform_recalibration(self):
        """Perform automatic recalibration."""
        self.logger.info("Performing automatic recalibration...")

        # Use recent stable measurements for new calibration
        stable_measurements = [
            m
            for m in self.measurements
            if m.confidence_score > 0.5 and m.wavelength_shift is not None
        ]

        if len(stable_measurements) >= 50:
            # Calculate new calibration parameters
            shifts = [m.wavelength_shift for m in stable_measurements]
            concentrations = [m.concentration_ppm for m in stable_measurements]

            # Linear regression for new calibration
            if len(shifts) >= 2 and len(concentrations) >= 2:
                from scipy.stats import linregress

                slope, intercept, r_value, p_value, std_err = linregress(concentrations, shifts)

                # Update calibration
                self.current_calibration = {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "std_error": std_err,
                    "timestamp": datetime.now().timestamp(),
                    "version": self.calibration_version + 1,
                    "sample_count": len(stable_measurements),
                }

                self.calibration_version += 1
                self.calibration_history.append(self.current_calibration.copy())

                self.logger.info(f"Calibration updated: slope={slope:.6f}, R²={r_value**2:.3f}")

    def get_current_calibration(self) -> Optional[dict]:
        """Get current calibration parameters."""
        return self.current_calibration

    def get_statistics(self) -> CalibrationStats:
        """Get current calibration statistics."""
        return self.stats

    def get_ml_features(self) -> list[dict]:
        """
        Extract ML-ready features from current memory.

        Returns:
            List of feature dictionaries for ML training
        """
        features = []

        for i, measurement in enumerate(self.measurements):
            if measurement.wavelength_shift is None:
                continue

            # Time-based features
            time_features = {
                "hour": datetime.fromtimestamp(measurement.timestamp).hour,
                "day_of_week": datetime.fromtimestamp(measurement.timestamp).weekday(),
                "time_since_start": measurement.timestamp - self.measurements[0].timestamp
                if i > 0
                else 0,
            }

            # Measurement features
            measurement_features = {
                "wavelength_shift": measurement.wavelength_shift,
                "concentration_ppm": measurement.concentration_ppm,
                "confidence_score": measurement.confidence_score,
                "snr": measurement.snr,
                "calibration_version": measurement.calibration_version,
            }

            # Statistical features (rolling window)
            if i >= 10:
                recent_shifts = [
                    self.measurements[j].wavelength_shift
                    for j in range(max(0, i - 10), i)
                    if self.measurements[j].wavelength_shift is not None
                ]
                if recent_shifts:
                    measurement_features.update(
                        {
                            "rolling_mean_shift": np.mean(recent_shifts),
                            "rolling_std_shift": np.std(recent_shifts),
                            "shift_trend": recent_shifts[-1] - recent_shifts[0]
                            if len(recent_shifts) > 1
                            else 0,
                        }
                    )

            # Combine features
            feature_dict = {**time_features, **measurement_features}
            features.append(feature_dict)

        return features

    def save_calibration(self, filepath: Path):
        """Save calibration history to file."""
        calibration_data = {
            "current_calibration": self.current_calibration,
            "calibration_history": self.calibration_history,
            "statistics": asdict(self.stats),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(calibration_data, f, indent=2, default=str)

        self.logger.info(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: Path):
        """Load calibration history from file."""
        if not filepath.exists():
            return False

        try:
            with open(filepath) as f:
                calibration_data = json.load(f)

            self.current_calibration = calibration_data.get("current_calibration")
            self.calibration_history = calibration_data.get("calibration_history", [])

            # Load statistics
            stats_dict = calibration_data.get("statistics", {})
            self.stats = CalibrationStats(**stats_dict)

            self.logger.info(f"Calibration loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
