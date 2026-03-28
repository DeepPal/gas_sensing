import numpy as np
import pytest
from src.inference.session_analyzer import SessionAnalyzer, SessionAnalysis


def _make_events(n: int = 20) -> list[dict]:
    """Generate synthetic calibration + measurement events."""
    events = []
    for i, conc in enumerate([0.5, 1.0, 2.0, 3.0, 4.0]):
        events.append({
            "type": "calibration_point",
            "concentration_ppm": float(conc),
            "wavelength_shift": -10.0 * conc / (1.0 + conc) + (i * 0.02),
            "snr": 15.0 + i,
        })
    for i in range(15):
        conc = 2.5 + 0.05 * i
        events.append({
            "type": "measurement",
            "concentration_ppm": conc,
            "ci_low": conc - 0.3,
            "ci_high": conc + 0.3,
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 14.0 + i * 0.1,
            "peak_wavelength": 717.9 + (i * 0.01),
        })
    return events


def test_analyze_returns_session_analysis():
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert isinstance(result, SessionAnalysis)


def test_lod_loq_positive():
    """LOD and LOQ must be positive real numbers with LOQ > LOD."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=20)
    assert result.lod_ppm > 0
    assert result.loq_ppm > 0
    assert result.loq_ppm > result.lod_ppm


def test_snr_stats():
    """Mean SNR must be computable and positive."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=20)
    assert result.mean_snr > 0


def test_drift_rate():
    """Drift rate must be extractable from peak_wavelength series."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=20)
    assert result.drift_rate_nm_per_frame is not None
    assert isinstance(result.drift_rate_nm_per_frame, float)


def test_calibration_r2():
    """R² of the calibration fit must be between -1 and 1."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=20)
    if result.calibration_r2 is not None:
        assert -1.0 <= result.calibration_r2 <= 1.0


def test_empty_events_does_not_crash():
    """analyze() with no events must return a default SessionAnalysis."""
    result = SessionAnalyzer().analyze([], frame_count=0)
    assert isinstance(result, SessionAnalysis)
    assert result.frame_count == 0
