"""
tests/src/test_batch_coverage.py
=================================
Coverage boosters for src.batch.preprocessing and src.batch.response.

These tests exercise pure-logic paths that have no tests elsewhere.
No disk I/O or external services required.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# src.batch.preprocessing — sort_frame_paths
# ---------------------------------------------------------------------------
from src.batch.preprocessing import sort_frame_paths


class TestSortFramePaths:
    def test_sorts_by_trailing_number(self, tmp_path):
        names = ["frame_003.csv", "frame_001.csv", "frame_002.csv"]
        paths = [str(tmp_path / n) for n in names]
        for p in paths:
            open(p, "w").close()
        result = sort_frame_paths(paths)
        assert [p.split("frame_")[1] for p in result] == ["001.csv", "002.csv", "003.csv"]

    def test_empty_list(self):
        assert sort_frame_paths([]) == []

    def test_sorts_by_timestamp_pattern(self, tmp_path):
        # t1_20241029_11h25m41s826ms.csv should come before t1_20241029_14h00m00s000ms.csv
        early = "t1_20241029_11h25m41s826ms.csv"
        late = "t1_20241029_14h00m00s000ms.csv"
        paths = [str(tmp_path / n) for n in [late, early]]
        for p in paths:
            open(p, "w").close()
        result = sort_frame_paths(paths)
        assert result[0].endswith(early)
        assert result[1].endswith(late)

    def test_fallback_for_no_digits(self, tmp_path):
        # Names with no numeric component should not crash
        names = ["alpha.csv", "beta.csv"]
        paths = [str(tmp_path / n) for n in names]
        for p in paths:
            open(p, "w").close()
        result = sort_frame_paths(paths)
        assert len(result) == 2

    def test_handles_mixed_patterns(self, tmp_path):
        names = ["scan_001.csv", "t1_20241029_10h00m00s000ms.csv"]
        paths = [str(tmp_path / n) for n in names]
        for p in paths:
            open(p, "w").close()
        result = sort_frame_paths(paths)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# src.batch.response — safe_float and scale_reference helpers
# ---------------------------------------------------------------------------
from src.batch.response import _safe_float, scale_reference_to_baseline


class TestSafeFloat:
    def test_converts_int(self):
        assert _safe_float(42) == 42.0

    def test_converts_string_number(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_nan_for_non_numeric(self):
        assert math.isnan(_safe_float("not_a_number"))

    def test_nan_for_inf(self):
        assert math.isnan(_safe_float(float("inf")))

    def test_nan_for_none(self):
        assert math.isnan(_safe_float(None))


class TestScaleReferenceToBaseline:
    def _make_spectrum(self, wl_start=500.0, n=50):
        wl = np.linspace(wl_start, wl_start + n, n)
        intensity = np.ones(n) * 1000.0
        return pd.DataFrame({"wavelength": wl, "intensity": intensity})

    def test_none_ref_returns_none(self):
        ref, scale = scale_reference_to_baseline(None, [])
        assert ref is None
        assert scale == 1.0

    def test_empty_baselines_returns_unchanged(self):
        ref = self._make_spectrum()
        result_ref, scale = scale_reference_to_baseline(ref, [])
        assert scale == 1.0

    def test_missing_wavelength_column(self):
        bad = pd.DataFrame({"x": [1, 2], "intensity": [100, 200]})
        result, scale = scale_reference_to_baseline(bad, [self._make_spectrum()])
        assert scale == 1.0

    def test_scale_applied_when_baselines_match(self):
        ref = self._make_spectrum()
        baseline = self._make_spectrum()
        baseline["intensity"] = baseline["intensity"] * 2.0
        scaled, factor = scale_reference_to_baseline(ref, [baseline])
        assert factor == pytest.approx(2.0, rel=0.05)
