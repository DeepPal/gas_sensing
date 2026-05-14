"""
Parity: src.preprocessing.quality.detect_outliers must return identical
results to gas_analysis.core.signal_proc.basic.detect_outliers.
"""
import numpy as np
import pytest


def test_detect_outliers_parity(sample_spectra):
    from gas_analysis.core.signal_proc.basic import detect_outliers as old
    from src.preprocessing.quality import detect_outliers as new

    result_old = old(sample_spectra, threshold=3.0)
    result_new = new(sample_spectra, threshold=3.0)
    assert result_old == result_new, (
        f"detect_outliers parity failed.\nold: {result_old}\nnew: {result_new}"
    )


def test_detect_outliers_no_outliers(single_spectrum):
    from src.preprocessing.quality import detect_outliers
    result = detect_outliers([single_spectrum] * 5)
    assert result == [False, False, False, False, False]


def test_detect_outliers_catches_spike(single_spectrum, rng):
    from src.preprocessing.quality import detect_outliers
    spike = single_spectrum.copy()
    spike[100:110] = 100.0  # obvious spike
    spectra = [single_spectrum] * 9 + [spike]
    # threshold=2.9 so that z=3.0 (produced by the spike in a 10-sample set)
    # strictly exceeds the threshold; the strict > operator means threshold=3.0
    # would NOT flag z=3.0 exactly.
    result = detect_outliers(spectra, threshold=2.9)
    assert result[-1] is True, "Spiked spectrum should be flagged as outlier"
