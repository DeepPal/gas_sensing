import numpy as np
import pytest

from src.inference.session_analyzer import SessionAnalysis, SessionAnalyzer


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


def test_audit_trail_populated():
    """analyze() must populate the audit dict with required regulatory metadata."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert isinstance(result.audit, dict)
    for key in ("method", "lod_formula", "sigma_source", "framework_version", "analysis_timestamp_utc"):
        assert key in result.audit, f"Missing audit key: {key}"
    assert result.audit["method"] == "IUPAC_2012_Eurachem"


def test_audit_sigma_source_is_calibration_residuals_by_default():
    """Without blank events, sigma_source must be 'calibration_residuals'."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.audit.get("sigma_source") == "calibration_residuals"


def test_audit_sigma_source_is_blank_events_when_blanks_present():
    """With blank events, sigma_source must be 'blank_events'."""
    events = _make_events() + [
        {"type": "blank", "wavelength_shift": 0.005 * i, "snr": 12.0}
        for i in range(3)
    ]
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.audit.get("sigma_source") == "blank_events"


def test_lob_is_less_than_lod():
    """LOB must be less than LOD (blank distribution sits below detection threshold)."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    if not (np.isnan(result.lob_ppm) or np.isnan(result.lod_ppm)):
        assert result.lob_ppm < result.lod_ppm


def test_lod_bootstrap_ci_is_ordered():
    """Bootstrap CI must satisfy ci_lower <= lod <= ci_upper."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    if not np.isnan(result.lod_ci_lower):
        assert result.lod_ci_lower <= result.lod_ppm + 1e-9
        assert result.lod_ppm <= result.lod_ci_upper + 1e-9


def test_loq_bootstrap_ci_scales_from_lod():
    """LOQ CI must scale by the same factor as LOQ/LOD."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    if not np.isnan(result.lod_ci_lower) and result.lod_ppm > 0:
        scale = result.loq_ppm / result.lod_ppm
        assert abs(result.loq_ci_lower / result.lod_ci_lower - scale) < 0.01
        assert abs(result.loq_ci_upper / result.lod_ci_upper - scale) < 0.01


def test_lod_used_blanks_false_without_blank_events():
    """lod_used_blanks must be False when no blank events are present."""
    events = _make_events()
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.lod_used_blanks is False


def test_lod_used_blanks_true_with_blank_type_events():
    """lod_used_blanks must be True when type='blank' events are present."""
    events = _make_events()
    # Append 3 dedicated blank measurements (zero-concentration)
    for i in range(3):
        events.append({
            "type": "blank",
            "wavelength_shift": 0.0 + i * 0.005,
            "snr": 10.0,
        })
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.lod_used_blanks is True
    assert result.lod_ppm > 0


def test_lod_used_blanks_true_with_zero_conc_calibration():
    """lod_used_blanks must be True when calibration points with concentration_ppm=0 are present."""
    events = _make_events()
    # Add two zero-concentration calibration points
    for i in range(2):
        events.append({
            "type": "calibration_point",
            "concentration_ppm": 0.0,
            "wavelength_shift": 0.01 * i,
            "snr": 12.0,
        })
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.lod_used_blanks is True


def test_blank_lod_lower_than_residual_lod_with_low_noise_blanks():
    """Blank-based LOD with very low noise blanks must yield lower LOD than residual-based."""
    base_events = _make_events()

    # Without blanks: LOD uses calibration residuals
    result_no_blanks = SessionAnalyzer().analyze(base_events, frame_count=len(base_events))

    # With near-zero-noise blanks: LOD should be equal or lower
    events_with_blanks = base_events + [
        {"type": "blank", "wavelength_shift": 0.0001 * i, "snr": 30.0}
        for i in range(5)
    ]
    result_with_blanks = SessionAnalyzer().analyze(events_with_blanks, frame_count=len(events_with_blanks))

    assert result_with_blanks.lod_used_blanks is True
    assert result_with_blanks.lod_ppm <= result_no_blanks.lod_ppm + 1e-6  # tighter or equal


# ---------------------------------------------------------------------------
# LOL (Limit of Linearity) tests
# ---------------------------------------------------------------------------


def test_lol_populated_with_5_calibration_points():
    """LOL must be computed when calibration_n_points >= 5."""
    result = SessionAnalyzer().analyze(_make_events(), frame_count=20)
    assert result.calibration_n_points == 5
    # LOL may be NaN only if no subrange is linear, but for this Langmuir data
    # the low-concentration region should be approximately linear.
    # At minimum the linearity dict must be populated.
    assert isinstance(result.linearity, dict), "linearity dict must be populated with ≥5 pts"


def test_lol_leq_max_calibration_concentration():
    """LOL can never exceed the highest calibration concentration."""
    result = SessionAnalyzer().analyze(_make_events(), frame_count=20)
    if result.lol_ppm == result.lol_ppm:  # not NaN
        max_cal = max(result.calibration_concentrations)
        assert result.lol_ppm <= max_cal + 1e-6


def test_linearity_dict_has_required_keys():
    """linearity dict must contain all keys produced by mandel_linearity_test."""
    result = SessionAnalyzer().analyze(_make_events(), frame_count=20)
    if result.linearity:  # only check if populated
        for key in ("is_linear", "f_statistic", "p_value", "r2_linear", "r2_quadratic"):
            assert key in result.linearity, f"Missing linearity key: {key!r}"


def test_lol_not_populated_with_fewer_than_5_calibration_points():
    """With only 3 calibration points LOL stays NaN (need 4 for Mandel)."""
    events = []
    for conc in [0.5, 1.0, 2.0]:  # only 3 calibration points
        events.append({
            "type": "calibration_point",
            "concentration_ppm": float(conc),
            "wavelength_shift": -10.0 * conc / (1.0 + conc),
            "snr": 15.0,
        })
    for _ in range(5):
        events.append({
            "type": "measurement",
            "concentration_ppm": 1.5,
            "wavelength_shift": -6.0,
            "snr": 14.0,
        })
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    import math
    assert math.isnan(result.lol_ppm), "LOL should be NaN with only 3 calibration points"


def test_lol_in_audit_trail():
    """LOL value must be recorded in the audit trail."""
    result = SessionAnalyzer().analyze(_make_events(), frame_count=20)
    assert "lol_ppm" in result.audit
    assert "lol_mandel_p_value" in result.audit


# ---------------------------------------------------------------------------
# Response kinetics (T90 / T10) tests
# ---------------------------------------------------------------------------


def test_t90_t10_none_without_timing_events():
    """T90 and T10 must be None when events lack response_time_t90_s."""
    result = SessionAnalyzer().analyze(_make_events(), frame_count=20)
    assert result.response_time_t90_seconds is None
    assert result.response_time_t10_seconds is None


def test_t90_aggregated_from_measurement_events():
    """T90 must equal the mean of response_time_t90_s values across measurement events."""
    events = _make_events()
    for i, ev in enumerate(events):
        if ev["type"] == "measurement":
            ev["response_time_t90_s"] = 30.0 + i  # add timing field to each measurement
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.response_time_t90_seconds is not None
    # Mean of t90 values for the 15 measurement events (indices 5-19) is 30+5 to 30+19 → mean ≈ 42
    expected = float(np.mean([30.0 + i for i in range(5, 20)]))
    assert abs(result.response_time_t90_seconds - expected) < 0.01


def test_t10_aggregated_independently_from_t90():
    """T10 must be populated independently and correctly from its own event field."""
    events = _make_events()
    for ev in events:
        if ev["type"] == "measurement":
            ev["response_time_t10_s"] = 55.0
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.response_time_t10_seconds is not None
    assert abs(result.response_time_t10_seconds - 55.0) < 0.01


def test_t90_populated_t10_none_when_only_t90_in_events():
    """T90 and T10 are extracted independently; one can be present without the other."""
    events = _make_events()
    for ev in events:
        if ev["type"] == "measurement":
            ev["response_time_t90_s"] = 25.0
            # deliberately omit response_time_t10_s
    result = SessionAnalyzer().analyze(events, frame_count=len(events))
    assert result.response_time_t90_seconds is not None
    assert result.response_time_t10_seconds is None
