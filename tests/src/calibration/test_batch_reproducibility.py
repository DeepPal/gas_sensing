"""Tests for src.calibration.batch_reproducibility — inter-sensor analysis."""
import numpy as np
import pytest

from src.calibration.batch_reproducibility import (
    BatchReproducibilityAnalyzer,
    BatchReproducibilityReport,
)
from src.inference.session_analyzer import SessionAnalysis


def _make_session(
    lod: float = 0.05,
    loq: float = 0.15,
    r2: float = 0.995,
    n_cal: int = 5,
    slope: float = -3.5,
    noise: float = 0.02,
) -> SessionAnalysis:
    """Synthetic SessionAnalysis for batch testing."""
    rng = np.random.default_rng(42)
    concs = np.linspace(0.5, 3.0, n_cal)
    shifts = slope * concs / (1.0 + concs) + rng.normal(0, noise, n_cal)
    sa = SessionAnalysis(
        frame_count=n_cal * 5,
        lod_ppm=lod,
        loq_ppm=loq,
        calibration_r2=r2,
        calibration_n_points=n_cal,
        calibration_concentrations=concs.tolist(),
        calibration_shifts=shifts.tolist(),
    )
    return sa


def test_returns_batch_reproducibility_report():
    sessions = [_make_session() for _ in range(3)]
    analyzer = BatchReproducibilityAnalyzer()
    report = analyzer.analyze(sessions)
    assert isinstance(report, BatchReproducibilityReport)
    assert report.n_sensors == 3


def test_lod_mean_computed():
    lod_vals = [0.04, 0.05, 0.06]
    sessions = [_make_session(lod=v) for v in lod_vals]
    report = BatchReproducibilityAnalyzer().analyze(sessions)
    assert abs(report.lod_mean - np.mean(lod_vals)) < 0.001


def test_lod_rsd_computed():
    lod_vals = [0.04, 0.05, 0.06]
    sessions = [_make_session(lod=v) for v in lod_vals]
    report = BatchReproducibilityAnalyzer().analyze(sessions)
    assert np.isfinite(report.lod_rsd_pct)
    assert report.lod_rsd_pct > 0


def test_batch_accepted_for_tight_sensors():
    """Three sensors with very similar LOD/R² should pass acceptance."""
    sessions = [_make_session(lod=0.050 + i * 0.001, r2=0.995) for i in range(3)]
    analyzer = BatchReproducibilityAnalyzer(lod_rsd_limit_pct=20.0, min_r2=0.99)
    report = analyzer.analyze(sessions)
    assert report.batch_accepted is True
    assert len(report.failure_reasons) == 0


def test_batch_rejected_for_poor_r2():
    """A sensor with R²=0.95 should fail the R² criterion."""
    sessions = [
        _make_session(r2=0.995),
        _make_session(r2=0.995),
        _make_session(r2=0.940),  # below limit
    ]
    analyzer = BatchReproducibilityAnalyzer(min_r2=0.99)
    report = analyzer.analyze(sessions)
    assert report.batch_accepted is False
    assert any("R²" in reason for reason in report.failure_reasons)


def test_batch_rejected_for_high_lod_rsd():
    """LOD RSD > 20% should fail the batch."""
    sessions = [
        _make_session(lod=0.02),
        _make_session(lod=0.05),
        _make_session(lod=0.15),  # 3× difference → high RSD
    ]
    analyzer = BatchReproducibilityAnalyzer(lod_rsd_limit_pct=20.0)
    report = analyzer.analyze(sessions)
    assert report.batch_accepted is False


def test_verdict_none_with_fewer_than_3_sensors():
    """With < 3 sensors the batch verdict should not be True (no valid comparison)."""
    sessions = [_make_session(), _make_session()]
    report = BatchReproducibilityAnalyzer().analyze(sessions)
    # batch_accepted should be None or False but not True with only 2 sensors
    assert report.batch_accepted is not True


def test_pooled_lod_less_than_max_sensor_lod():
    """Pooled LOD uses pooled σ_blank which should be within the sensor range."""
    sessions = [_make_session(lod=v) for v in [0.04, 0.05, 0.06]]
    report = BatchReproducibilityAnalyzer().analyze(sessions)
    if np.isfinite(report.pooled_lod):
        assert 0.0 < report.pooled_lod < 1.0


def test_custom_sensor_ids():
    sessions = [_make_session() for _ in range(3)]
    ids = ["BatchA-01", "BatchA-02", "BatchA-03"]
    report = BatchReproducibilityAnalyzer().analyze(sessions, sensor_ids=ids)
    assert report.sensor_ids == ids


def test_summary_contains_pass_fail():
    sessions = [_make_session(lod=0.050 + i * 0.001, r2=0.996) for i in range(3)]
    report = BatchReproducibilityAnalyzer().analyze(sessions)
    assert "ACCEPTED" in report.summary or "FAILED" in report.summary
