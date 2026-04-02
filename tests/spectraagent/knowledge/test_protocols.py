"""
Unit tests for spectraagent.knowledge.protocols —
ValidationTracker ICH Q2(R1) state machine.
"""
from __future__ import annotations

import pytest

from spectraagent.knowledge.protocols import (
    ICH_Q2_PROTOCOL,
    ValidationRequirement,
    ValidationStatus,
    ValidationTracker,
)


# ---------------------------------------------------------------------------
# ICH_Q2_PROTOCOL registry
# ---------------------------------------------------------------------------

class TestICHProtocol:
    def test_registry_not_empty(self):
        assert len(ICH_Q2_PROTOCOL) >= 5

    def test_mandatory_tests_exist(self):
        mandatory = {k for k, v in ICH_Q2_PROTOCOL.items() if v.is_mandatory}
        # Core ICH Q2(R1) tests must all be present
        for expected in ("linearity", "lod_loq"):
            assert expected in mandatory, f"'{expected}' missing from mandatory tests"

    def test_each_requirement_has_ich_section(self):
        for key, req in ICH_Q2_PROTOCOL.items():
            assert req.ich_section.startswith("§"), f"{key} section does not start with §"

    def test_each_requirement_has_acceptance_criteria(self):
        for key, req in ICH_Q2_PROTOCOL.items():
            assert len(req.acceptance_criteria) > 0

    def test_depends_on_only_valid_keys(self):
        all_keys = set(ICH_Q2_PROTOCOL.keys())
        for key, req in ICH_Q2_PROTOCOL.items():
            for dep in req.depends_on:
                assert dep in all_keys, f"{key} depends on unknown '{dep}'"


# ---------------------------------------------------------------------------
# ValidationTracker — initial state
# ---------------------------------------------------------------------------

class TestValidationTrackerInit:
    def test_all_tests_not_started_initially(self):
        tracker = ValidationTracker("Ethanol")
        for status in tracker._status.values():
            assert status == ValidationStatus.NOT_STARTED

    def test_completion_pct_zero_initially(self):
        tracker = ValidationTracker("Ethanol")
        assert tracker.completion_pct() == pytest.approx(0.0)

    def test_get_gaps_returns_all_required(self):
        tracker = ValidationTracker("Ethanol")
        gaps = tracker.get_gaps()
        assert len(gaps) == len(tracker._required)

    def test_custom_required_tests(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        assert set(tracker._required) == {"linearity", "lod_loq"}

    def test_get_next_test_returns_requirement_object(self):
        tracker = ValidationTracker("Ethanol")
        nxt = tracker.get_next_test()
        assert nxt is None or isinstance(nxt, ValidationRequirement)


# ---------------------------------------------------------------------------
# ValidationTracker.update
# ---------------------------------------------------------------------------

class TestValidationTrackerUpdate:
    def test_update_changes_status(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        tracker.update("linearity", ValidationStatus.COMPLETE)
        assert tracker._status["linearity"] == ValidationStatus.COMPLETE

    def test_update_unknown_test_is_noop(self):
        tracker = ValidationTracker("Ethanol")
        tracker.update("nonexistent_test", ValidationStatus.COMPLETE)
        assert "nonexistent_test" not in tracker._status

    def test_update_stores_result(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity"])
        tracker.update("linearity", ValidationStatus.COMPLETE, {"r_squared": 0.9995})
        assert tracker._results["linearity"]["r_squared"] == pytest.approx(0.9995)

    def test_completion_pct_after_one_update(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        tracker.update("linearity", ValidationStatus.COMPLETE)
        assert tracker.completion_pct() == pytest.approx(50.0)

    def test_completion_pct_100_when_all_complete(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        tracker.update("linearity", ValidationStatus.COMPLETE)
        tracker.update("lod_loq", ValidationStatus.COMPLETE)
        assert tracker.completion_pct() == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# ValidationTracker.get_gaps / get_next_test
# ---------------------------------------------------------------------------

class TestValidationTrackerGaps:
    def test_get_gaps_excludes_complete_tests(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        tracker.update("linearity", ValidationStatus.COMPLETE)
        gaps = tracker.get_gaps()
        gap_ids = [g.id for g in gaps]
        assert "linearity" not in gap_ids
        assert "lod_loq" in gap_ids

    def test_get_next_test_returns_none_when_all_complete(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        tracker.update("linearity", ValidationStatus.COMPLETE)
        tracker.update("lod_loq", ValidationStatus.COMPLETE)
        assert tracker.get_next_test() is None

    def test_get_next_test_skips_tests_with_unmet_deps(self):
        """A test that depends on an incomplete test should not be returned first."""
        # Find a test with dependencies in the full protocol
        dep_test = next(
            (k for k, v in ICH_Q2_PROTOCOL.items() if v.depends_on and v.is_mandatory),
            None,
        )
        if dep_test is None:
            pytest.skip("No test with dependencies found in ICH_Q2_PROTOCOL")

        req = ICH_Q2_PROTOCOL[dep_test]
        # Use only the dependent test and its first dependency
        dep = req.depends_on[0]
        tracker = ValidationTracker("Ethanol", required_tests=[dep, dep_test])
        # Neither complete yet — next test must be the dependency, not the dependent
        nxt = tracker.get_next_test()
        assert nxt is not None
        assert nxt.id == dep  # dependency must come first


# ---------------------------------------------------------------------------
# ValidationTracker.infer_from_calibration_data
# ---------------------------------------------------------------------------

class TestValidationTrackerInfer:
    def test_infer_linearity_complete_high_r2(self):
        tracker = ValidationTracker("Ethanol")
        tracker.infer_from_calibration_data({
            "n_points": 7,
            "r_squared": 0.9993,
            "lod_ppm": None,
            "loq_ppm": None,
        })
        assert tracker._status.get("linearity") == ValidationStatus.COMPLETE

    def test_infer_linearity_in_progress_medium_r2(self):
        tracker = ValidationTracker("Ethanol")
        tracker.infer_from_calibration_data({
            "n_points": 5,
            "r_squared": 0.995,
        })
        assert tracker._status.get("linearity") == ValidationStatus.IN_PROGRESS

    def test_infer_lod_loq_complete(self):
        tracker = ValidationTracker("Ethanol")
        tracker.infer_from_calibration_data({
            "lod_ppm": 0.015,
            "loq_ppm": 0.050,
        })
        assert tracker._status.get("lod_loq") == ValidationStatus.COMPLETE

    def test_infer_with_no_r2_leaves_linearity_untouched(self):
        tracker = ValidationTracker("Ethanol")
        tracker.infer_from_calibration_data({})
        assert tracker._status.get("linearity") == ValidationStatus.NOT_STARTED

    def test_infer_too_few_points_does_not_complete_linearity(self):
        tracker = ValidationTracker("Ethanol")
        tracker.infer_from_calibration_data({
            "n_points": 3,
            "r_squared": 0.9999,
        })
        # 3 points < 5 minimum → should NOT be COMPLETE
        assert tracker._status.get("linearity") != ValidationStatus.COMPLETE


# ---------------------------------------------------------------------------
# ValidationTracker.format_status_for_prompt
# ---------------------------------------------------------------------------

class TestValidationTrackerFormat:
    def test_format_contains_analyte_name(self):
        tracker = ValidationTracker("Ethanol")
        text = tracker.format_status_for_prompt()
        assert "Ethanol" in text

    def test_format_is_string(self):
        tracker = ValidationTracker("Ethanol")
        assert isinstance(tracker.format_status_for_prompt(), str)

    def test_format_contains_percentage(self):
        tracker = ValidationTracker("Ethanol")
        text = tracker.format_status_for_prompt()
        assert "%" in text

    def test_format_updates_after_completion(self):
        tracker = ValidationTracker("Ethanol", required_tests=["linearity", "lod_loq"])
        before = tracker.format_status_for_prompt()
        tracker.update("linearity", ValidationStatus.COMPLETE)
        tracker.update("lod_loq", ValidationStatus.COMPLETE)
        after = tracker.format_status_for_prompt()
        assert "100" in after
        assert before != after
