"""Shared fixtures for science regression tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "lspr_calibration_fixture.npz"
BASELINES_PATH = ROOT / "tests" / "science_regression" / "baselines.json"


@pytest.fixture(scope="session")
def cal_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            "Science regression fixture missing. "
            "Run: python tests/fixtures/generate_fixture.py"
        )
    data = np.load(FIXTURE_PATH)
    return {k: data[k] for k in data.files}


@pytest.fixture(scope="session")
def baselines() -> dict:
    if not BASELINES_PATH.exists():
        pytest.fail(
            "Science regression fixture missing. "
            "Run: python tests/fixtures/generate_fixture.py"
        )
    return json.loads(BASELINES_PATH.read_text())
