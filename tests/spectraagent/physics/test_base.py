import numpy as np
import pytest

from spectraagent.physics.base import AbstractSensorPhysicsPlugin


class _ConcretePhysics(AbstractSensorPhysicsPlugin):
    def detect_peak(self, wl, intensities):
        return float(wl[int(np.argmax(intensities))])

    def extract_features(self, wl, intensities, reference=None, cached_ref=None):
        return {"delta_lambda": -0.5, "snr": 10.0, "peak_wavelength": 720.0}

    def compute_reference_cache(self, wl, reference):
        return {"peak": float(wl[int(np.argmax(reference))])}

    def calibration_priors(self):
        return {"models": ["Linear"], "bounds": {}}

    @property
    def name(self):
        return "TestPhysics"


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        AbstractSensorPhysicsPlugin()  # type: ignore[abstract]


def test_concrete_satisfies_interface():
    ph = _ConcretePhysics()
    wl = np.linspace(500, 900, 3648)
    sp = np.zeros(3648)
    sp[1000] = 1.0
    peak = ph.detect_peak(wl, sp)
    assert isinstance(peak, float)
    feats = ph.extract_features(wl, sp)
    assert "delta_lambda" in feats
    assert "snr" in feats
    cache = ph.compute_reference_cache(wl, sp)
    assert cache is not None
    assert ph.name == "TestPhysics"


def test_extract_features_has_required_keys():
    ph = _ConcretePhysics()
    wl = np.linspace(500, 900, 3648)
    sp = np.ones(3648) * 0.5
    feats = ph.extract_features(wl, sp)
    assert "delta_lambda" in feats
    assert "snr" in feats
