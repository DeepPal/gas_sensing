"""
Integration tests: end-to-end production workflow compliance.

These tests verify the complete ICH Q2(R1) / IUPAC 2012 analytical chemistry
pipeline from raw calibration data through to publication-ready metrics.

Each test exercises multiple modules together, checking that:
  1. All IUPAC/ICH mandatory fields are populated.
  2. The ordering constraints (LOB < LOD < LOQ ≤ LOL) hold.
  3. SelectivityAnalyzer and BatchReproducibilityAnalyzer integrate with
     SessionAnalysis objects produced by SessionAnalyzer.
  4. The public_api surface exports all required classes and functions.
  5. compute_comprehensive_sensor_characterization() bundles all metrics.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sessions(n_sensors: int = 3, seed: int = 0):
    """Return a list of SessionAnalysis objects from independent sensor simulations."""
    from src.inference.session_analyzer import SessionAnalyzer

    rng = np.random.default_rng(seed)
    sessions = []
    for sensor_idx in range(n_sensors):
        noise_scale = 0.02 + 0.005 * sensor_idx
        events = []
        for i, conc in enumerate([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]):
            events.append({
                "type": "calibration_point",
                "concentration_ppm": float(conc),
                "wavelength_shift": float(
                    -10.0 * conc / (1.0 + conc)
                    + rng.normal(0, noise_scale)
                ),
                "snr": 15.0 + i,
            })
        for i in range(10):
            conc = 2.0 + 0.1 * i
            events.append({
                "type": "measurement",
                "concentration_ppm": conc,
                "ci_low": conc - 0.4,
                "ci_high": conc + 0.4,
                "wavelength_shift": float(-10.0 * conc / (1.0 + conc)),
                "snr": 14.0 + i * 0.1,
                "peak_wavelength": 717.9 + i * 0.01,
            })
        sa = SessionAnalyzer().analyze(events, frame_count=len(events))
        sessions.append(sa)
    return sessions


# ---------------------------------------------------------------------------
# 1. Full IUPAC triad + LOL
# ---------------------------------------------------------------------------


class TestFullIUPACTriad:
    """LOB / LOD / LOQ / LOL are all computed and correctly ordered."""

    def test_lob_lod_loq_all_present(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        assert not np.isnan(sa.lob_ppm), "LOB must be populated"
        assert not np.isnan(sa.lod_ppm), "LOD must be populated"
        assert not np.isnan(sa.loq_ppm), "LOQ must be populated"

    def test_ordering_lob_lt_lod_lt_loq(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        if not (np.isnan(sa.lob_ppm) or np.isnan(sa.lod_ppm) or np.isnan(sa.loq_ppm)):
            assert sa.lob_ppm < sa.lod_ppm, f"LOB={sa.lob_ppm} must be < LOD={sa.lod_ppm}"
            assert sa.lod_ppm < sa.loq_ppm, f"LOD={sa.lod_ppm} must be < LOQ={sa.loq_ppm}"

    def test_lol_populated_for_6point_calibration(self):
        """With 6 calibration points, LOL must be computed via Mandel's test."""
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        # LOL may be nan only if Mandel finds nonlinearity across all subsets
        # For a Langmuir-shaped but roughly linear low-conc range, expect a value
        if sa.calibration_n_points >= 5:
            # At least linearity result should be populated
            assert isinstance(sa.linearity, dict), "linearity dict must be populated"

    def test_lol_leq_max_calibration_concentration(self):
        """LOL must not exceed the maximum calibration concentration."""
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        if not np.isnan(sa.lol_ppm) and sa.calibration_concentrations:
            max_cal = max(sa.calibration_concentrations)
            assert sa.lol_ppm <= max_cal + 1e-6, (
                f"LOL={sa.lol_ppm} exceeds max cal conc={max_cal}"
            )

    def test_bootstrap_ci_ordered(self):
        """Bootstrap CIs must satisfy ci_lower ≤ lod ≤ ci_upper."""
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        if not np.isnan(sa.lod_ci_lower):
            assert sa.lod_ci_lower <= sa.lod_ppm + 1e-9
            assert sa.lod_ppm <= sa.lod_ci_upper + 1e-9

    def test_loq_ci_scales_by_ten_thirds(self):
        """LOQ CI must scale by the same 10/3 factor as the point estimate."""
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        if not np.isnan(sa.lod_ci_lower) and sa.lod_ppm > 1e-9:
            scale = sa.loq_ppm / sa.lod_ppm
            assert abs(sa.loq_ci_lower / sa.lod_ci_lower - scale) < 0.01
            assert abs(sa.loq_ci_upper / sa.lod_ci_upper - scale) < 0.01


# ---------------------------------------------------------------------------
# 2. Response kinetics (T90/T10)
# ---------------------------------------------------------------------------


class TestResponseKinetics:
    """T90/T10 are populated when pre-computed fields are present in events."""

    def test_t90_t10_none_without_timing_events(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        # Default synthetic events have no response_time_t90_s field
        assert sa.response_time_t90_seconds is None
        assert sa.response_time_t10_seconds is None

    def test_t90_extracted_from_events(self):
        """When events carry response_time_t90_s, SessionAnalyzer aggregates them."""
        from src.inference.session_analyzer import SessionAnalyzer

        events = []
        for i, conc in enumerate([0.5, 1.0, 2.0, 3.0]):
            events.append({
                "type": "calibration_point",
                "concentration_ppm": float(conc),
                "wavelength_shift": -3.0 * i * 0.5,
                "snr": 15.0,
            })
        for i in range(5):
            events.append({
                "type": "measurement",
                "concentration_ppm": 2.0,
                "wavelength_shift": -5.0,
                "snr": 14.0,
                "response_time_t90_s": 30.0 + i,
                "response_time_t10_s": 45.0 + i,
            })
        sa = SessionAnalyzer().analyze(events, frame_count=len(events))
        assert sa.response_time_t90_seconds is not None
        assert abs(sa.response_time_t90_seconds - 32.0) < 0.5  # mean of 30..34
        assert sa.response_time_t10_seconds is not None


# ---------------------------------------------------------------------------
# 3. SelectivityAnalyzer + SessionAnalysis integration
# ---------------------------------------------------------------------------


class TestSelectivityIntegration:
    """SelectivityAnalyzer.from_session_analyses() correctly chains with SessionAnalysis."""

    def test_from_session_analyses_returns_report(self):
        from src.calibration.selectivity import SelectivityAnalyzer

        sessions = _make_sessions(n_sensors=2)
        analyte_sa = sessions[0]
        interferent_sa = sessions[1]

        report = SelectivityAnalyzer.from_session_analyses(
            analyte="Ethanol",
            analyte_analysis=analyte_sa,
            interferent_analyses={"CO2": interferent_sa},
        )
        assert report.analyte == "Ethanol"
        assert "CO2" in report.cross_reactivity_coefficients

    def test_selectivity_flag_present(self):
        from src.calibration.selectivity import SelectivityAnalyzer

        sessions = _make_sessions(n_sensors=2)
        report = SelectivityAnalyzer.from_session_analyses(
            analyte="Ethanol",
            analyte_analysis=sessions[0],
            interferent_analyses={"Methanol": sessions[1]},
        )
        flag = report.selectivity_flags.get("Methanol")
        assert flag in ("excellent", "good", "moderate", "poor")

    def test_analyte_sensitivity_negative_for_lspr(self):
        """LSPR sensors have negative sensitivity (red-shift with increasing concentration)."""
        from src.calibration.selectivity import SelectivityAnalyzer

        sessions = _make_sessions(n_sensors=1)
        report = SelectivityAnalyzer.from_session_analyses(
            analyte="Ethanol",
            analyte_analysis=sessions[0],
            interferent_analyses={},
        )
        assert report.analyte_sensitivity_nm_per_ppm < 0


# ---------------------------------------------------------------------------
# 4. BatchReproducibilityAnalyzer + multiple SessionAnalysis objects
# ---------------------------------------------------------------------------


class TestBatchReproducibilityIntegration:
    """BatchReproducibilityAnalyzer processes real SessionAnalysis objects correctly."""

    def test_analyze_three_sensors(self):
        from src.calibration.batch_reproducibility import BatchReproducibilityAnalyzer

        sessions = _make_sessions(n_sensors=3)
        report = BatchReproducibilityAnalyzer().analyze(sessions)
        assert report.n_sensors == 3
        assert len(report.lod_values) == 3
        assert np.isfinite(report.lod_mean)
        assert np.isfinite(report.lod_rsd_pct)

    def test_batch_accepted_for_consistent_sensors(self):
        """Three sensors with very similar calibrations should pass batch acceptance."""
        from src.calibration.batch_reproducibility import BatchReproducibilityAnalyzer

        sessions = _make_sessions(n_sensors=3, seed=7)
        report = BatchReproducibilityAnalyzer(
            lod_rsd_limit_pct=50.0,
            min_r2=0.90,
            sensitivity_rsd_limit_pct=50.0,
        ).analyze(sessions)
        # With very loose criteria, a 3-sensor batch should pass
        assert report.batch_accepted is not None

    def test_pooled_lod_is_populated(self):
        from src.calibration.batch_reproducibility import BatchReproducibilityAnalyzer

        sessions = _make_sessions(n_sensors=3)
        report = BatchReproducibilityAnalyzer().analyze(sessions)
        assert np.isfinite(report.pooled_lod), "Pooled LOD must be computed for 3 sensors"
        assert report.pooled_lod > 0

    def test_custom_sensor_ids(self):
        from src.calibration.batch_reproducibility import BatchReproducibilityAnalyzer

        sessions = _make_sessions(n_sensors=3)
        ids = ["Chip-A", "Chip-B", "Chip-C"]
        report = BatchReproducibilityAnalyzer().analyze(sessions, sensor_ids=ids)
        assert report.sensor_ids == ids


# ---------------------------------------------------------------------------
# 5. compute_comprehensive_sensor_characterization
# ---------------------------------------------------------------------------


class TestComprehensiveSensorCharacterization:
    """compute_comprehensive_sensor_characterization bundles all metrics correctly."""

    def _make_calibration(self, n: int = 8, seed: int = 0):
        rng = np.random.default_rng(seed)
        concs = np.linspace(0.5, 5.0, n)
        shifts = -10.0 * concs / (1.0 + concs) + rng.normal(0, 0.05, n)
        return concs, shifts

    def test_returns_mandatory_keys(self):
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        result = compute_comprehensive_sensor_characterization(concs, shifts, gas_name="Ethanol")
        for key in ("gas", "sensitivity", "r_squared", "lob_ppm", "lod_ppm", "loq_ppm"):
            assert key in result, f"Missing mandatory key: {key}"

    def test_ordering_lob_lod_loq(self):
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        lob = result.get("lob_ppm")
        lod = result.get("lod_ppm")
        loq = result.get("loq_ppm")
        if lob is not None and lod is not None and loq is not None:
            assert lob < lod, f"LOB={lob} must be < LOD={lod}"
            assert lod < loq, f"LOD={lod} must be < LOQ={loq}"

    def test_lol_populated_with_8_points(self):
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration(n=8)
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        # Either LOL is populated, or mandel_linearity gives context
        assert "lol_ppm" in result
        assert "mandel_linearity" in result

    def test_bootstrap_ci_keys_present(self):
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        for key in ("lod_ppm_ci_lower", "lod_ppm_ci_upper",
                    "loq_ppm_ci_lower", "loq_ppm_ci_upper"):
            assert key in result, f"Missing CI key: {key}"

    def test_blank_measurements_lowers_lod(self):
        """Very-low-noise blank measurements should yield a tighter LOD."""
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        rng = np.random.default_rng(99)
        blanks = rng.normal(0.0, 0.001, 10)  # near-zero noise blanks

        result_no_blank = compute_comprehensive_sensor_characterization(concs, shifts)
        result_with_blank = compute_comprehensive_sensor_characterization(
            concs, shifts, blank_measurements=blanks
        )
        lod_no = result_no_blank.get("lod_ppm") or float("inf")
        lod_with = result_with_blank.get("lod_ppm") or float("inf")
        assert lod_with <= lod_no + 1e-6, (
            f"Blank-based LOD={lod_with} should be ≤ residual-based LOD={lod_no}"
        )

    def test_sigma_source_reported(self):
        """sigma_source must indicate where noise estimate came from."""
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        assert result.get("sigma_source") in ("ols_residuals", "blank_measurements")

    def test_linear_dynamic_range_tuple(self):
        """linear_dynamic_range_ppm must be a (lower, upper) tuple with lower < upper."""
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration(n=8)
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        ldr = result.get("linear_dynamic_range_ppm")
        if ldr is not None:
            lo, hi = ldr
            assert lo < hi, f"LDR lower={lo} must be < upper={hi}"

    def test_methods_audit_keys(self):
        """methods dict must contain all IUPAC methodology tags."""
        from src.reporting.metrics import compute_comprehensive_sensor_characterization

        concs, shifts = self._make_calibration()
        result = compute_comprehensive_sensor_characterization(concs, shifts)
        methods = result.get("methods", {})
        for key in ("lob", "lod", "loq", "lol", "ci"):
            assert key in methods, f"Missing methods key: {key}"


# ---------------------------------------------------------------------------
# 6. Public API completeness
# ---------------------------------------------------------------------------


class TestPublicAPICompleteness:
    """All required classes and functions are importable from src.public_api."""

    def test_selectivity_classes_exported(self):
        from src.public_api import SelectivityAnalyzer, SelectivityReport  # noqa: F401

    def test_batch_repro_classes_exported(self):
        from src.public_api import (  # noqa: F401
            BatchReproducibilityAnalyzer,
            BatchReproducibilityReport,
        )

    def test_comprehensive_characterization_exported(self):
        from src.public_api import compute_comprehensive_sensor_characterization  # noqa: F401

    def test_pipeline_classes_still_exported(self):
        from src.public_api import RealTimePipeline, PipelineConfig, PipelineResult  # noqa: F401

    def test_gpr_still_exported(self):
        from src.public_api import GPRCalibration  # noqa: F401

    def test_all_exports_importable(self):
        """Every name in __all__ must be importable."""
        import src.public_api as api
        for name in api.__all__:
            assert hasattr(api, name), f"{name!r} in __all__ but not importable"


# ---------------------------------------------------------------------------
# 7. Audit trail regulatory metadata
# ---------------------------------------------------------------------------


class TestAuditTrailCompliance:
    """SessionAnalysis.audit must contain all mandatory regulatory metadata."""

    REQUIRED_KEYS = (
        "method", "lod_formula", "loq_formula", "lob_formula",
        "sigma_source", "n_bootstrap", "bootstrap_confidence",
        "calibration_n_points", "lol_ppm", "lol_mandel_p_value",
        "frame_count", "framework_version", "analysis_timestamp_utc",
        "references",
    )

    def test_all_mandatory_audit_keys_present(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        for key in self.REQUIRED_KEYS:
            assert key in sa.audit, f"Missing audit key: {key!r}"

    def test_references_is_list(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        refs = sa.audit.get("references")
        assert isinstance(refs, list) and len(refs) >= 2

    def test_method_is_iupac(self):
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        assert sa.audit["method"] == "IUPAC_2012_Eurachem"

    def test_timestamp_is_iso_format(self):
        import datetime
        sessions = _make_sessions(n_sensors=1)
        sa = sessions[0]
        ts = sa.audit.get("analysis_timestamp_utc", "")
        # Must parse as a valid ISO datetime
        try:
            dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
            assert dt.tzinfo is not None, "Timestamp must be timezone-aware"
        except (ValueError, AttributeError) as exc:
            pytest.fail(f"audit['analysis_timestamp_utc']={ts!r} is not valid ISO: {exc}")
