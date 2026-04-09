"""
Real-Time Performance Monitoring for Gas Sensing
============================================

Monitors wavelength shift, absorbance patterns, and calculates
performance metrics including LOD, response time, and drift.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""

    timestamp: float
    wavelength_shift: float
    wavelength_shift_std: float
    absorbance_change: float
    absorbance_pattern_score: float
    lod_estimate: float
    response_time_s: float
    drift_rate_nm_per_min: float
    signal_quality: float
    confidence_interval: tuple[float, float]


@dataclass
class ResponseTimeTracker:
    """Tracks response time measurements."""

    start_time: Optional[float] = None
    baseline_value: Optional[float] = None
    response_times: deque = None

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = deque(maxlen=100)


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(self, window_size: int = 1000):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of recent measurements to analyze
        """
        self.window_size = window_size

        # Data storage
        self.wavelength_history = deque(maxlen=window_size)
        self.absorbance_history = deque(maxlen=window_size)
        self.snr_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)

        # Performance metrics
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = None

        # Response time tracking
        self.response_tracker = ResponseTimeTracker()

        # Baseline for change detection
        self.baseline_wavelength = None
        self.baseline_absorbance = None
        self.baseline_samples = 0

        # Setup logging
        self.logger = logging.getLogger("PerformanceMonitor")

    def add_measurement(
        self,
        wavelength_shift: float,
        absorbance_spectrum: np.ndarray,
        snr: float,
        confidence_score: float,
        timestamp: Optional[float] = None,
    ) -> PerformanceMetrics:
        """
        Add a new measurement and calculate performance metrics.

        Args:
            wavelength_shift: Current wavelength shift in nm
            absorbance_spectrum: Full absorbance spectrum
            snr: Signal-to-noise ratio
            confidence_score: Measurement confidence
            timestamp: Measurement timestamp

        Returns:
            Current performance metrics
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Store measurements
        self.wavelength_history.append(wavelength_shift)
        self.snr_history.append(snr)
        self.confidence_history.append(confidence_score)

        # Calculate absorbance metrics
        absorbance_change = self._calculate_absorbance_change(absorbance_spectrum)
        absorbance_pattern_score = self._analyze_absorbance_pattern(absorbance_spectrum)
        self.absorbance_history.append(absorbance_change)

        # Update baseline
        self._update_baseline(wavelength_shift, absorbance_change)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(
            wavelength_shift,
            absorbance_change,
            absorbance_pattern_score,
            snr,
            confidence_score,
            timestamp,
        )

        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        # Track response time
        self._track_response_time(wavelength_shift)

        return metrics

    def _calculate_absorbance_change(self, absorbance_spectrum: np.ndarray) -> float:
        """Calculate overall absorbance change from baseline."""
        if len(absorbance_spectrum) == 0:
            return 0.0

        # Use multiple metrics for robustness
        mean_change = np.mean(absorbance_spectrum)
        max_change = np.max(np.abs(absorbance_spectrum))
        std_change = np.std(absorbance_spectrum)

        # Composite score
        return float(mean_change + 0.3 * max_change + 0.2 * std_change)

    def _analyze_absorbance_pattern(self, absorbance_spectrum: np.ndarray) -> float:
        """Analyze absorbance pattern for gas identification."""
        if len(absorbance_spectrum) < 10:
            return 0.0

        # Pattern analysis metrics
        len(absorbance_spectrum)

        # Peak characteristics
        peak_idx = np.argmin(absorbance_spectrum)  # Absorption peak
        peak_width = self._calculate_peak_width(absorbance_spectrum, peak_idx)
        peak_asymmetry = self._calculate_peak_asymmetry(absorbance_spectrum, peak_idx)

        # Spectral shape features
        spectral_skewness = self._calculate_spectral_skewness(absorbance_spectrum)
        spectral_kurtosis = self._calculate_spectral_kurtosis(absorbance_spectrum)

        # Pattern score (higher for more distinct features)
        pattern_score = (
            1.0 / (1.0 + peak_width)  # Narrower peaks = higher score
            + 1.0 / (1.0 + abs(peak_asymmetry))  # More symmetric = higher score
            + abs(spectral_skewness) * 0.5  # Moderate skewness
            + min(abs(spectral_kurtosis - 3.0) * 0.3, 1.0)  # Near-normal kurtosis (clamped)
        )

        return float(pattern_score)

    def _calculate_peak_width(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Calculate peak width at half maximum."""
        if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
            return 10.0

        peak_value = spectrum[peak_idx]
        half_max = peak_value + (np.max(spectrum) - peak_value) / 2.0

        # Find left and right half-maximum points
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] < half_max:
            left_idx -= 1

        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] < half_max:
            right_idx += 1

        return float(right_idx - left_idx)

    def _calculate_peak_asymmetry(self, spectrum: np.ndarray, peak_idx: int) -> float:
        """Calculate peak asymmetry."""
        if peak_idx <= 5 or peak_idx >= len(spectrum) - 5:
            return 0.0

        # Simple asymmetry measure
        left_area = np.trapezoid(spectrum[peak_idx - 5 : peak_idx + 1])
        right_area = np.trapezoid(spectrum[peak_idx : peak_idx + 6])

        if left_area + right_area > 0:
            return (right_area - left_area) / (left_area + right_area)
        return 0.0

    def _calculate_spectral_skewness(self, spectrum: np.ndarray) -> float:
        """Calculate spectral skewness."""
        if len(spectrum) < 3:
            return 0.0

        mean_val = np.mean(spectrum)
        std_val = np.std(spectrum)

        if std_val == 0:
            return 0.0

        return float(np.mean(((spectrum - mean_val) / std_val) ** 3))

    def _calculate_spectral_kurtosis(self, spectrum: np.ndarray) -> float:
        """Calculate spectral kurtosis."""
        if len(spectrum) < 4:
            return 3.0  # Normal distribution kurtosis

        mean_val = np.mean(spectrum)
        std_val = np.std(spectrum)

        if std_val == 0:
            return 3.0

        return float(np.mean(((spectrum - mean_val) / std_val) ** 4))

    def _update_baseline(self, wavelength_shift: float, absorbance_change: float):
        """Update baseline measurements."""
        if self.baseline_samples < 50:
            # Collect baseline samples
            if self.baseline_wavelength is None:
                self.baseline_wavelength = wavelength_shift
                self.baseline_absorbance = absorbance_change
            else:
                # Exponential moving average for baseline
                alpha = 2.0 / (self.baseline_samples + 1.0)
                self.baseline_wavelength = (
                    alpha * wavelength_shift + (1 - alpha) * self.baseline_wavelength
                )
                self.baseline_absorbance = (
                    alpha * absorbance_change + (1 - alpha) * self.baseline_absorbance
                )

            self.baseline_samples += 1
        else:
            # Baseline established, track deviations
            pass

    def _calculate_performance_metrics(
        self,
        wavelength_shift: float,
        absorbance_change: float,
        absorbance_pattern_score: float,
        snr: float,
        confidence_score: float,
        timestamp: float,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""

        # Wavelength shift statistics
        if len(self.wavelength_history) >= 10:
            recent_shifts = list(self.wavelength_history)[-min(100, len(self.wavelength_history)) :]
            shift_mean = np.mean(recent_shifts)
            shift_std = np.std(recent_shifts)
        else:
            shift_mean = wavelength_shift
            shift_std = 0.0

        # Drift rate calculation — Theil-Sen estimator (robust to outliers).
        # Unlike linear regression, Theil-Sen uses the median of all pairwise
        # slopes and is unaffected by a single glitched frame.
        # Reference: Sen (1968), JASA 63(324):1379-1389.
        if len(self.wavelength_history) >= 4 and len(self.metrics_history) >= 4:
            wl_history = np.array(list(self.wavelength_history), dtype=float)
            ts_history = np.array([m.timestamp for m in self.metrics_history], dtype=float)
            n_hist = min(len(wl_history), len(ts_history))
            if n_hist >= 4:
                t_min = (ts_history[-n_hist:] - ts_history[-n_hist]) / 60.0  # minutes
                try:
                    from scipy.stats import theilslopes

                    result = theilslopes(wl_history[-n_hist:], t_min)
                    drift_rate = float(result.slope)  # nm/min
                except Exception:
                    # Fallback: simple two-point slope
                    time_span = (timestamp - list(self.metrics_history)[0].timestamp) / 60.0
                    drift_rate = (
                        (wavelength_shift - float(list(self.wavelength_history)[0])) / time_span
                        if time_span > 0
                        else 0.0
                    )
            else:
                drift_rate = 0.0
        else:
            drift_rate = 0.0

        # LOD estimation (3-sigma method)
        if shift_std > 0:
            lod_estimate = 3.0 * shift_std
        else:
            lod_estimate = 0.0

        # Response time
        response_time = self._get_current_response_time()

        # Signal quality score
        signal_quality = (
            confidence_score * 0.4  # Confidence
            + min(1.0, snr / 50.0) * 0.3  # SNR (normalized)
            + min(1.0, absorbance_pattern_score / 5.0) * 0.3  # Pattern quality
        )

        # Confidence interval
        if shift_std > 0 and len(recent_shifts) > 10:
            confidence_interval = (
                shift_mean - 1.96 * shift_std / np.sqrt(len(recent_shifts)),
                shift_mean + 1.96 * shift_std / np.sqrt(len(recent_shifts)),
            )
        else:
            confidence_interval = (shift_mean, shift_mean)

        return PerformanceMetrics(
            timestamp=timestamp,
            wavelength_shift=shift_mean,
            wavelength_shift_std=shift_std,
            absorbance_change=absorbance_change,
            absorbance_pattern_score=absorbance_pattern_score,
            lod_estimate=lod_estimate,
            response_time_s=response_time,
            drift_rate_nm_per_min=drift_rate,
            signal_quality=signal_quality,
            confidence_interval=confidence_interval,
        )

    def _track_response_time(self, current_value: float):
        """Track response time for step changes."""
        if self.baseline_samples < 50:
            return  # Still establishing baseline

        # Detect significant change (> 3 sigma from baseline)
        if self.baseline_wavelength is not None:
            threshold = 3.0 * np.std(
                list(self.wavelength_history)[-min(50, len(self.wavelength_history)) :]
            )
            change = abs(current_value - self.baseline_wavelength)

            if change > threshold:
                if self.response_tracker.start_time is None:
                    # Start of response
                    self.response_tracker.start_time = datetime.now().timestamp()
                    self.response_tracker.baseline_value = self.baseline_wavelength
                else:
                    # Check if this is approaching new steady state
                    if abs(current_value - self.response_tracker.baseline_value) < threshold * 0.1:
                        # Response complete
                        response_time = (
                            datetime.now().timestamp() - self.response_tracker.start_time
                        )
                        self.response_tracker.response_times.append(response_time)

                        # Reset for next response
                        self.response_tracker.start_time = None
                        self.response_tracker.baseline_value = None

    def _get_current_response_time(self) -> float:
        """Get current response time estimate."""
        if len(self.response_tracker.response_times) > 0:
            return float(np.mean(list(self.response_tracker.response_times)[-10:]))
        return 0.0

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        return self.current_metrics

    def get_metrics_summary(self, window_minutes: int = 10) -> dict:
        """Get performance metrics summary for recent time window."""
        if len(self.metrics_history) == 0:
            return {}

        # Filter recent metrics
        current_time = datetime.now().timestamp()
        recent_metrics = [
            m for m in self.metrics_history if current_time - m.timestamp <= window_minutes * 60
        ]

        if not recent_metrics:
            return {}

        # Calculate summary statistics
        summary = {
            "time_window_minutes": window_minutes,
            "sample_count": len(recent_metrics),
            "avg_wavelength_shift": np.mean([m.wavelength_shift for m in recent_metrics]),
            "std_wavelength_shift": np.std([m.wavelength_shift for m in recent_metrics]),
            "avg_lod": np.mean([m.lod_estimate for m in recent_metrics]),
            "avg_response_time": np.mean(
                [m.response_time_s for m in recent_metrics if m.response_time_s > 0]
            ),
            "avg_drift_rate": np.mean([m.drift_rate_nm_per_min for m in recent_metrics]),
            "avg_signal_quality": np.mean([m.signal_quality for m in recent_metrics]),
            "max_confidence_interval_width": np.mean(
                [
                    m.confidence_interval[1] - m.confidence_interval[0]
                    for m in recent_metrics
                    if m.confidence_interval
                ]
            ),
        }

        return summary

    def export_ml_data(self, filepath: str, window_hours: int = 24):
        """Export performance data for ML training."""
        if len(self.metrics_history) == 0:
            return False

        # Filter data for time window
        current_time = datetime.now().timestamp()
        ml_data = []

        for metrics in self.metrics_history:
            if current_time - metrics.timestamp <= window_hours * 3600:
                # Create ML-ready feature vector
                feature_dict = {
                    # Time features
                    "timestamp": metrics.timestamp,
                    "hour": datetime.fromtimestamp(metrics.timestamp).hour,
                    "day_of_week": datetime.fromtimestamp(metrics.timestamp).weekday(),
                    # Performance features
                    "wavelength_shift": metrics.wavelength_shift,
                    "wavelength_shift_std": metrics.wavelength_shift_std,
                    "absorbance_change": metrics.absorbance_change,
                    "absorbance_pattern_score": metrics.absorbance_pattern_score,
                    "lod_estimate": metrics.lod_estimate,
                    "response_time_s": metrics.response_time_s,
                    "drift_rate_nm_per_min": metrics.drift_rate_nm_per_min,
                    "signal_quality": metrics.signal_quality,
                    # Statistical features
                    "confidence_interval_width": metrics.confidence_interval[1]
                    - metrics.confidence_interval[0],
                    "is_stable": metrics.signal_quality > 0.7,
                    "has_high_drift": abs(metrics.drift_rate_nm_per_min) > 0.05,
                }

                ml_data.append(feature_dict)

        # Export to CSV
        df = pd.DataFrame(ml_data)
        df.to_csv(filepath, index=False)

        self.logger.info(f"Exported {len(ml_data)} samples for ML training to {filepath}")
        return True
