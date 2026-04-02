"""Tests for src.features.cross_peak_features."""
import numpy as np
import pytest
from src.features.cross_peak_features import (
    CrossPeakPCA,
    cosine_similarity_to_reference,
    extract_cross_peak_features,
    pattern_match_scores,
    shift_ratios,
    shift_vector_direction,
    shift_vector_norm,
    spectral_angle,
    spectral_entropy,
    spectral_similarity_scores,
)


# ---------------------------------------------------------------------------
# Shift vector features
# ---------------------------------------------------------------------------

class TestShiftVectorNorm:
    def test_single_peak(self):
        assert abs(shift_vector_norm(np.array([-0.5])) - 0.5) < 1e-9

    def test_two_peaks(self):
        v = np.array([-0.3, -0.4])
        assert abs(shift_vector_norm(v) - 0.5) < 1e-9

    def test_zero_vector(self):
        assert shift_vector_norm(np.array([0.0, 0.0])) == 0.0


class TestShiftVectorDirection:
    def test_unit_length(self):
        d = shift_vector_direction(np.array([-0.5, -0.2]))
        assert abs(np.linalg.norm(d) - 1.0) < 1e-9

    def test_zero_vector_returns_zeros(self):
        d = shift_vector_direction(np.array([0.0, 0.0]))
        assert np.all(d == 0.0)

    def test_sign_preserved(self):
        d = shift_vector_direction(np.array([-0.5, -0.2]))
        assert d[0] < 0 and d[1] < 0


class TestShiftRatios:
    def test_two_peaks_produces_two_ratios(self):
        ratios, names = shift_ratios(np.array([-0.5, -0.2]))
        assert len(ratios) == 2  # ratio_0_1 and ratio_1_0
        assert "ratio_0_1" in names

    def test_ratio_correct(self):
        ratios, names = shift_ratios(np.array([-0.5, -0.25]))
        idx = names.index("ratio_0_1")
        assert abs(ratios[idx] - 2.0) < 1e-9

    def test_near_zero_denominator_gives_nan_then_zero(self):
        # min_abs_shift = 0.01: shift < 0.01 → ratio set to NaN, then 0.0 in full extractor
        ratios, _ = shift_ratios(np.array([-0.5, 0.005]), min_abs_shift=0.01)
        # ratio_0_1 uses peak 1 as denominator → should be NaN
        assert np.isnan(ratios[0])


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = np.array([-0.5, -0.2])
        assert abs(cosine_similarity_to_reference(v, v) - 1.0) < 1e-9

    def test_opposite_vectors_give_minus_one(self):
        v = np.array([0.5, 0.2])
        assert abs(cosine_similarity_to_reference(v, -v) - (-1.0)) < 1e-9

    def test_orthogonal_vectors_give_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_similarity_to_reference(v1, v2)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity_to_reference(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0


class TestPatternMatchScores:
    def test_returns_dict_per_analyte(self):
        patterns = {"A": np.array([-0.5, -0.1]), "B": np.array([-0.1, -0.4])}
        scores = pattern_match_scores(np.array([-0.5, -0.1]), patterns)
        assert set(scores.keys()) == {"A", "B"}

    def test_matching_analyte_has_highest_score(self):
        patterns = {"A": np.array([-0.5, -0.1]), "B": np.array([-0.1, -0.4])}
        scores = pattern_match_scores(np.array([-0.5, -0.1]), patterns)
        assert scores["A"] > scores["B"]


# ---------------------------------------------------------------------------
# Spectral angle
# ---------------------------------------------------------------------------

class TestSpectralAngle:
    def test_identical_spectra_give_zero_angle(self):
        spec = np.random.default_rng(0).uniform(0, 1, 100)
        assert abs(spectral_angle(spec, spec)) < 1e-9

    def test_orthogonal_spectra_give_pi_over_2(self):
        s1 = np.array([1.0, 0.0])
        s2 = np.array([0.0, 1.0])
        assert abs(spectral_angle(s1, s2) - np.pi / 2.0) < 1e-9

    def test_returns_radians_in_range(self):
        rng = np.random.default_rng(42)
        s1 = rng.uniform(0, 1, 100)
        s2 = rng.uniform(0, 1, 100)
        angle = spectral_angle(s1, s2)
        assert 0.0 <= angle <= np.pi


class TestSpectralEntropy:
    def test_uniform_signal_has_high_entropy(self):
        spec = np.ones(100)
        e = spectral_entropy(spec)
        assert e > 0

    def test_sparse_signal_has_low_entropy(self):
        spec = np.zeros(100)
        spec[50] = 1.0  # delta function
        e = spectral_entropy(spec)
        assert e < 0.01

    def test_zero_signal_returns_zero(self):
        assert spectral_entropy(np.zeros(100)) == 0.0


# ---------------------------------------------------------------------------
# CrossPeakPCA
# ---------------------------------------------------------------------------

class TestCrossPeakPCA:
    @pytest.fixture
    def fitted_pca(self):
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 4))
        pca = CrossPeakPCA(n_components=2)
        pca.fit(X)
        return pca, X

    def test_fit_sets_is_fitted(self):
        pca = CrossPeakPCA()
        pca.fit(np.random.default_rng(0).normal(0, 1, (20, 3)))
        assert pca.is_fitted

    def test_transform_output_shape(self, fitted_pca):
        pca, X = fitted_pca
        out = pca.transform(X)
        assert out.shape == (50, 2)

    def test_transform_single_sample(self, fitted_pca):
        pca, X = fitted_pca
        out = pca.transform(X[0])
        assert out.shape == (1, 2)

    def test_unfitted_transform_raises(self):
        pca = CrossPeakPCA()
        with pytest.raises(RuntimeError):
            pca.transform(np.array([[1.0, 2.0]]))

    def test_explained_variance_sums_to_leq_one(self, fitted_pca):
        pca, _ = fitted_pca
        assert pca.explained_variance_ratio.sum() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Full extractor
# ---------------------------------------------------------------------------

class TestExtractCrossPeakFeatures:
    def test_single_peak_no_ratios(self):
        feats, names = extract_cross_peak_features(np.array([-0.5]))
        assert "ratio" not in " ".join(names)  # no ratios for 1 peak

    def test_two_peaks_has_ratios(self):
        feats, names = extract_cross_peak_features(np.array([-0.5, -0.2]))
        assert any("ratio" in n for n in names)

    def test_with_analyte_patterns(self):
        patterns = {"A": np.array([-0.5, -0.1]), "B": np.array([-0.1, -0.4])}
        feats, names = extract_cross_peak_features(np.array([-0.5, -0.1]), analyte_patterns=patterns)
        assert "pattern_match_A" in feats
        assert "pattern_match_B" in feats

    def test_with_diff_spectrum(self):
        diff = np.random.default_rng(0).uniform(-0.1, 0.1, 3648)
        feats, names = extract_cross_peak_features(np.array([-0.5]), diff_spectrum=diff)
        assert "spectral_entropy" in feats

    def test_with_spectral_analyte_references(self):
        diff = np.random.default_rng(0).uniform(0, 0.1, 100)
        analyte_spectra = {"A": diff * 0.9, "B": diff[::-1]}
        feats, names = extract_cross_peak_features(
            np.array([-0.5]),
            diff_spectrum=diff,
            analyte_spectra=analyte_spectra,
        )
        assert "sam_A" in feats
        assert "sam_B" in feats

    def test_with_pca(self):
        pca = CrossPeakPCA(n_components=2)
        X = np.random.default_rng(0).normal(0, 1, (20, 2))
        pca.fit(X)
        feats, names = extract_cross_peak_features(np.array([-0.5, -0.2]), pca=pca)
        assert "pc_0" in feats
        assert "pc_1" in feats
