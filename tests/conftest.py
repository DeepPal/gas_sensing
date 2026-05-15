"""
tests/conftest.py
=================
Shared pytest fixtures for the the sensor LSPR gas sensing platform.

All fixtures in this file are available automatically to every test
module in the ``tests/`` directory — no explicit import needed.

Fixture quick-reference
-----------------------
- ``minimal_config_yaml``  — write a minimal valid config.yaml to tmp_path
- ``full_config_path``     — real config/config.yaml resolved from project root
- ``synthetic_spectrum``   — realistic the sensor LSPR intensity spectrum (np arrays)
- ``canonical_spectra``    — dict mapping concentration → DataFrame (for batch tests)
- ``gaussian``             — helper function for building Gaussian peaks

Design notes
------------
- All file-system fixtures use ``tmp_path`` (pytest built-in) so each test
  gets its own isolated temporary directory.
- Fixtures that are expensive to compute are scoped at ``session`` level where
  the result is deterministic (e.g. ``full_config_path``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure tests import local workspace packages before any similarly named installed packages.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian() -> Callable[[np.ndarray, float, float], np.ndarray]:
    """Return a Gaussian peak function ``f(wl, center, width) → ndarray``."""

    def _gaussian(wl: np.ndarray, center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((wl - center) / width) ** 2)

    return _gaussian


# ---------------------------------------------------------------------------
# Wavelength axis used throughout tests
# ---------------------------------------------------------------------------


@pytest.fixture
def wavelengths_400_700() -> np.ndarray:
    """400-point wavelength axis from 400 to 700 nm."""
    return np.linspace(400.0, 700.0, 400)


@pytest.fixture
def wavelengths_lspr() -> np.ndarray:
    """200-point axis centred on the LSPR region (480–600 nm)."""
    return np.linspace(480.0, 600.0, 200)


# ---------------------------------------------------------------------------
# Synthetic LSPR spectra
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_spectrum(wavelengths_400_700: np.ndarray) -> dict:
    """Realistic synthetic the sensor LSPR spectrum.

    Returns a dict with keys:
        - ``wavelengths``: 1-D ndarray (nm)
        - ``intensities``: 1-D ndarray (a.u.) with a Gaussian absorption peak
        - ``peak_nm``: float — true peak position (nm)
        - ``concentration_ppm``: float — true analyte concentration
    """
    rng = np.random.default_rng(42)
    wl = wavelengths_400_700
    baseline = np.ones_like(wl) * 10_000.0
    noise = rng.normal(0, 30, wl.size)

    peak_nm = 531.5
    absorption = 200.0 * np.exp(-((wl - peak_nm) ** 2) / (2 * 1.5**2))
    intensities = baseline + noise - absorption

    return {
        "wavelengths": wl,
        "intensities": intensities,
        "peak_nm": peak_nm,
        "concentration_ppm": 1.0,
    }


@pytest.fixture
def canonical_spectra(wavelengths_400_700: np.ndarray) -> dict[float, pd.DataFrame]:
    """Dict mapping concentration (ppm) → spectrum DataFrame.

    Simulates 5 concentration levels with a linear shift in the LSPR peak
    wavelength (−0.116 nm/ppm) matching the the sensor literature sensitivity.
    """
    rng = np.random.default_rng(0)
    wl = wavelengths_400_700
    sensitivity = 0.116  # nm/ppm
    ref_peak = 531.5  # nm
    canonical: dict[float, pd.DataFrame] = {}

    for conc in [0.5, 1.0, 2.0, 3.0, 5.0]:
        peak = ref_peak - sensitivity * conc  # blueshift on adsorption
        absorption = 500.0 * np.exp(-((wl - peak) ** 2) / (2 * 1.8**2))
        intensity = 10_000.0 + rng.normal(0, 25, wl.size) - absorption
        canonical[conc] = pd.DataFrame({"wavelength": wl, "intensity": intensity})

    return canonical


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_config_path() -> Path:
    """Absolute path to the real ``config/config.yaml``.

    Skips the test if the file is not present (e.g. in a stripped CI checkout).
    """
    root = Path(__file__).resolve().parents[1]
    cfg = root / "config" / "config.yaml"
    if not cfg.exists():
        pytest.skip(f"config.yaml not found at {cfg}")
    return cfg


@pytest.fixture
def minimal_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid config.yaml to *tmp_path* and return its path."""
    config: dict = {
        "preprocessing": {"enabled": True},
        "roi": {
            "shift": {
                "step_nm": 0.1,
                "window_nm": [5.0, 10.0],
            },
            "discovery": {
                "enabled": False,
                "step_nm": 0.5,
                "window_nm": 10.0,
            },
        },
        "response_series": {
            "enabled": True,
            "min_activation_frames": 3,
        },
        "quality": {
            "min_snr": 4.0,
            "max_rsd_pct": 7.5,
        },
        "sensor": {
            "integration_time_ms": 50,
            "target_wavelength_nm": 532.0,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


@pytest.fixture
def invalid_config_yaml_duplicate_key(tmp_path: Path) -> Path:
    """YAML file with a duplicate key (should raise on load)."""
    content = "roi:\n  step_nm: 0.1\n  step_nm: 0.2\n"
    path = tmp_path / "bad_config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def invalid_config_yaml_negative_step(tmp_path: Path) -> Path:
    """YAML file with a non-positive ``roi.shift.step_nm`` (should raise)."""
    config = {
        "roi": {"shift": {"step_nm": -1.0, "window_nm": 5.0}},
    }
    path = tmp_path / "bad_step.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path
