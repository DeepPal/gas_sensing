"""
src.features.cross_peak_features
===================================
Cross-peak feature extraction for multi-peak optical sensors.

Motivation
----------
When a sensor has M spectral peaks, the analyte-specific information is not
just in the individual shifts Δλ_j, but in the PATTERN of shifts across all
peaks.  Cross-peak features encode this pattern explicitly:

1. **Shift ratios**: Δλᵢ / Δλⱼ — concentration-independent in the linear
   regime; analyte-specific fingerprint of binding modes.
2. **Pearson correlation** between the observed shift vector and stored
   reference patterns (calibrated signatures per analyte).
3. **Principal component projections**: project the shift vector onto the
   dominant PCs of the calibration set — captures the most variance-rich
   directions.
4. **Spectral angle mapper (SAM)**: cosine distance between the diff-spectrum
   and a reference spectrum. Widely used in remote sensing; maps directly
   to analyte identity.
5. **Differential entropy**: approximate information content of the shift
   distribution across peaks — increases with more simultaneous analytes.

These features are most powerful for:
  - Configuration 2: single sensor, multiple peaks, single analyte
    (ratios give analyte fingerprint; robust to concentration variation)
  - Configuration 3: multi-analyte, multiple peaks
    (SAM + correlation separate analytes with overlapping Δλ ranges)
  - Configuration 4: multiple analytes, cross-interference
    (PC projections give low-dimensional embedding separating mixture states)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shift-vector features (peak-shift domain)
# ---------------------------------------------------------------------------


def shift_ratios(
    peak_shifts_nm: np.ndarray,
    min_abs_shift: float = 0.01,
) -> tuple[np.ndarray, list[str]]:
    """Compute all pairwise shift ratios Δλᵢ / Δλⱼ.

    Parameters
    ----------
    peak_shifts_nm:
        Observed peak shifts (nm), shape (M,).
    min_abs_shift:
        Denominator threshold below which ratio is set to NaN (avoids
        division by near-zero noise).

    Returns
    -------
    (ratios, names)
    ratios:
        Array of all i<j ratios, shape (M*(M-1)//2,).
    names:
        Feature names like 'ratio_0_1', 'ratio_0_2', ...
    """
    shifts = np.asarray(peak_shifts_nm, dtype=float)
    M = len(shifts)
    ratios: list[float] = []
    names: list[str] = []
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if abs(shifts[j]) < min_abs_shift:
                ratios.append(float("nan"))
            else:
                ratios.append(float(shifts[i] / shifts[j]))
            names.append(f"ratio_{i}_{j}")
    return np.array(ratios, dtype=float), names


def shift_vector_norm(peak_shifts_nm: np.ndarray) -> float:
    """L2 norm of the shift vector — total spectral response magnitude."""
    return float(np.linalg.norm(peak_shifts_nm))


def shift_vector_direction(peak_shifts_nm: np.ndarray) -> np.ndarray:
    """Unit vector of peak shifts — analyte fingerprint independent of concentration."""
    shifts = np.asarray(peak_shifts_nm, dtype=float)
    n = np.linalg.norm(shifts)
    if n < 1e-9:
        return np.zeros_like(shifts)
    return shifts / n


def cosine_similarity_to_reference(
    peak_shifts_nm: np.ndarray,
    reference_pattern: np.ndarray,
) -> float:
    """Cosine similarity between observed shift pattern and a reference.

    A value of 1.0 means the analyte matches the reference exactly.
    A value of 0.0 means orthogonal (completely different analyte binding).
    A negative value means opposite pattern (unusual, indicates interference).

    Parameters
    ----------
    peak_shifts_nm:
        Observed peak shifts, shape (M,).
    reference_pattern:
        Reference shift pattern (from calibration), shape (M,).
    """
    obs = np.asarray(peak_shifts_nm, dtype=float)
    ref = np.asarray(reference_pattern, dtype=float)
    n_obs = np.linalg.norm(obs)
    n_ref = np.linalg.norm(ref)
    if n_obs < 1e-12 or n_ref < 1e-12:
        return 0.0
    return float(np.dot(obs, ref) / (n_obs * n_ref))


def pattern_match_scores(
    peak_shifts_nm: np.ndarray,
    analyte_patterns: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute cosine similarity to each analyte's reference pattern.

    Returns a dict mapping analyte_name → similarity score in [-1, 1].
    The analyte with the highest score is the best match.
    """
    return {
        name: cosine_similarity_to_reference(peak_shifts_nm, pat)
        for name, pat in analyte_patterns.items()
    }


# ---------------------------------------------------------------------------
# Spectral domain features (raw spectrum)
# ---------------------------------------------------------------------------


def spectral_angle(
    diff_spectrum: np.ndarray,
    reference_diff: np.ndarray,
) -> float:
    """Spectral Angle Mapper (SAM) distance (radians).

    Measures the angle between two spectral vectors in n-dimensional space.
    Insensitive to illumination scaling (same analyte at different concentrations
    gives the same angle); angle=0 means identical analyte.

    Parameters
    ----------
    diff_spectrum:
        Observed difference spectrum (raw − reference), shape (N,).
    reference_diff:
        Reference difference spectrum from calibration, shape (N,).
    """
    obs = np.asarray(diff_spectrum, dtype=float)
    ref = np.asarray(reference_diff, dtype=float)
    n_obs = np.linalg.norm(obs)
    n_ref = np.linalg.norm(ref)
    if n_obs < 1e-12 or n_ref < 1e-12:
        return float(np.pi / 2.0)  # orthogonal = max distance
    cos_a = np.clip(np.dot(obs, ref) / (n_obs * n_ref), -1.0, 1.0)
    return float(np.arccos(cos_a))


def spectral_similarity_scores(
    diff_spectrum: np.ndarray,
    analyte_spectra: dict[str, np.ndarray],
) -> dict[str, float]:
    """Spectral angle (radians) from each analyte's calibration spectrum.

    Smaller angle = better spectral match.
    """
    return {
        name: spectral_angle(diff_spectrum, ref)
        for name, ref in analyte_spectra.items()
    }


def spectral_entropy(diff_spectrum: np.ndarray) -> float:
    """Approximate spectral information entropy.

    Measures how 'complex' the differential spectrum is:
    - Single analyte: entropy is low (single shifted peak)
    - Multiple analytes: entropy is higher (multiple shifted features)
    - Noise only: high entropy (random)

    Uses the absolute value of the normalised differential spectrum as
    a probability distribution.
    """
    s = np.abs(np.asarray(diff_spectrum, dtype=float))
    total = s.sum()
    if total < 1e-12:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Principal component projection (requires fit on calibration set)
# ---------------------------------------------------------------------------


class CrossPeakPCA:
    """Project peak-shift vectors onto principal components of the calibration space.

    Fitted on calibration data; used to project new observations into the
    low-dimensional analyte space.

    This is most useful for:
    - Visualisation (2D/3D analyte separation plot)
    - Feature dimensionality reduction before GPR input
    - Anomaly detection (distance from the calibration manifold)
    """

    def __init__(self, n_components: int = 3) -> None:
        self.n_components = n_components
        self._mean: np.ndarray | None = None
        self._components: np.ndarray | None = None   # (n_components, n_features)
        self._explained_variance_ratio: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray) -> CrossPeakPCA:
        """Fit PCA on calibration peak-shift matrix.

        X: (n_samples, n_peaks) — each row is one peak-shift observation.
        """
        X = np.asarray(X, dtype=float)
        self._mean = X.mean(axis=0)
        X_c = X - self._mean
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self._components = Vt[:k]
        total_var = float((S ** 2).sum())
        self._explained_variance_ratio = (S[:k] ** 2) / total_var if total_var > 0 else np.zeros(k)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the fitted PCs. Returns shape (n_samples, n_components)."""
        if not self.is_fitted:
            raise RuntimeError("Fit CrossPeakPCA before calling transform().")
        assert self._mean is not None and self._components is not None
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X - self._mean) @ self._components.T

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        return self._explained_variance_ratio


# ---------------------------------------------------------------------------
# Full cross-peak feature extractor
# ---------------------------------------------------------------------------


def extract_cross_peak_features(
    peak_shifts_nm: np.ndarray,
    analyte_patterns: dict[str, np.ndarray] | None = None,
    diff_spectrum: np.ndarray | None = None,
    analyte_spectra: dict[str, np.ndarray] | None = None,
    pca: CrossPeakPCA | None = None,
    min_abs_shift: float = 0.01,
) -> tuple[dict[str, float], list[str]]:
    """Extract all cross-peak features from one frame.

    Parameters
    ----------
    peak_shifts_nm:
        Observed peak shifts, shape (M,).
    analyte_patterns:
        Dict of reference shift patterns per analyte (from calibration).
        If provided, adds cosine similarity scores.
    diff_spectrum:
        Difference spectrum (raw − reference). If provided with
        ``analyte_spectra``, adds SAM scores and entropy.
    analyte_spectra:
        Dict of reference diff spectra per analyte.
    pca:
        Fitted CrossPeakPCA for PC projection.
    min_abs_shift:
        Threshold for ratio features (see :func:`shift_ratios`).

    Returns
    -------
    (features_dict, feature_names)
    """
    features: dict[str, float] = {}
    names: list[str] = []

    # 1. Shift vector magnitude and direction
    features["shift_norm"] = shift_vector_norm(peak_shifts_nm)
    names.append("shift_norm")

    direction = shift_vector_direction(peak_shifts_nm)
    for j, d in enumerate(direction):
        features[f"shift_dir_{j}"] = float(d)
        names.append(f"shift_dir_{j}")

    # 2. Pairwise shift ratios
    if len(peak_shifts_nm) > 1:
        ratios, ratio_names = shift_ratios(peak_shifts_nm, min_abs_shift)
        for name, val in zip(ratio_names, ratios):
            features[name] = float(val) if np.isfinite(val) else 0.0
            names.append(name)

    # 3. Cosine similarity to analyte reference patterns
    if analyte_patterns:
        for analyte, score in pattern_match_scores(peak_shifts_nm, analyte_patterns).items():
            key = f"pattern_match_{analyte}"
            features[key] = score
            names.append(key)

    # 4. Spectral angle and entropy
    if diff_spectrum is not None:
        features["spectral_entropy"] = spectral_entropy(diff_spectrum)
        names.append("spectral_entropy")

        if analyte_spectra:
            for analyte, angle in spectral_similarity_scores(diff_spectrum, analyte_spectra).items():
                key = f"sam_{analyte}"
                features[key] = angle
                names.append(key)

    # 5. PC projection
    if pca is not None and pca.is_fitted:
        proj = pca.transform(np.array(peak_shifts_nm))[0]
        for k, v in enumerate(proj):
            key = f"pc_{k}"
            features[key] = float(v)
            names.append(key)

    return features, names
