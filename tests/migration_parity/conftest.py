"""
Shared test vectors for migration parity tests.

Every parity test imports BOTH old (gas_analysis.core) and new (src.)
implementations and asserts numpy.testing.assert_allclose(rtol=1e-6).

If a parity test fails, it means the two implementations genuinely
differ — resolve the divergence explicitly before deleting old code.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=2026)


@pytest.fixture(scope="session")
def sample_spectra(rng) -> list[np.ndarray]:
    """Ten 500-point spectra with realistic SNR for parity testing."""
    base = np.exp(-((np.linspace(0, 10, 500) - 5) ** 2) / 2)
    return [base + rng.normal(0, 0.02, 500) for _ in range(10)]


@pytest.fixture(scope="session")
def single_spectrum(rng) -> np.ndarray:
    """Single 3648-point spectrum matching CCS200 pixel count."""
    wl = np.linspace(500, 900, 3648)
    peak = 1.0 / (1.0 + ((wl - 717.9) / 6.25) ** 2)
    return peak + rng.normal(0, 0.003, 3648)
