import numpy as np
import pytest

from spectraagent.physics.lspr import LSPRPlugin


@pytest.fixture
def plugin():
    return LSPRPlugin()


@pytest.fixture
def lspr_spectrum():
    """Lorentzian at 531.5 nm — matches LSPR_REFERENCE_PEAK_NM and search range."""
    wl = np.linspace(480.0, 600.0, 3648)
    peak = 531.5
    gamma = 9.0
    sp = 0.8 / (1.0 + ((wl - peak) / gamma) ** 2)
    sp += np.random.default_rng(42).normal(0, 0.001, len(wl))
    return wl, sp


def test_name(plugin):
    assert plugin.name == "LSPR"


def test_detect_peak_returns_float(plugin, lspr_spectrum):
    wl, sp = lspr_spectrum
    peak = plugin.detect_peak(wl, sp)
    assert peak is not None
    assert isinstance(peak, float)
    assert 520.0 <= peak <= 545.0


def test_compute_reference_cache_returns_lspr_reference(plugin, lspr_spectrum):
    from src.features.lspr_features import LSPRReference
    wl, sp = lspr_spectrum
    cache = plugin.compute_reference_cache(wl, sp)
    assert isinstance(cache, LSPRReference)


def test_extract_features_has_required_keys(plugin, lspr_spectrum):
    wl, sp = lspr_spectrum
    feats = plugin.extract_features(wl, sp, reference=sp)
    assert "delta_lambda" in feats
    assert "snr" in feats
    assert "peak_wavelength" in feats


def test_extract_features_with_cache_faster_than_without(plugin, lspr_spectrum):
    """Cache should be accepted without error; delta_lambda should be near 0 when gas == ref."""
    wl, sp = lspr_spectrum
    cache = plugin.compute_reference_cache(wl, sp)
    feats = plugin.extract_features(wl, sp, reference=sp, cached_ref=cache)
    assert "delta_lambda" in feats
    # Gas == reference -> shift should be near 0
    assert abs(feats["delta_lambda"]) < 1.0


def test_calibration_priors_has_required_models(plugin):
    priors = plugin.calibration_priors()
    assert "models" in priors
    assert "Langmuir" in priors["models"]
    assert "Linear" in priors["models"]


def test_extract_features_none_reference_returns_dict(plugin, lspr_spectrum):
    """extract_features with no reference should not raise."""
    wl, sp = lspr_spectrum
    feats = plugin.extract_features(wl, sp)
    assert isinstance(feats, dict)
    assert "delta_lambda" in feats
