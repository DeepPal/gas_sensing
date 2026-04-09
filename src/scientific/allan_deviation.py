"""
src.scientific.allan_deviation
================================
Allan deviation analysis for optical gas sensor noise characterisation.

Allan deviation is **the** standard metric for characterising noise in
time-series sensor measurements. It is mandatory in:

  - Sensors & Actuators B (Elsevier)
  - IEEE Sensors Journal
  - Analytical Chemistry (ACS)
  - Review of Scientific Instruments (AIP)

It answers questions OLS and RSD cannot:

  - What is the **optimal averaging time τ** to minimise measurement noise?
  - What is the **fundamental noise floor** (σ_min)?
  - Which **noise process** dominates: white, flicker (1/f), or random walk?
  - At what τ does **instrumental drift** overwhelm averaging benefits?

Theory
------
The overlapping Allan deviation σ(τ) at averaging time τ is:

.. math::

    \\sigma(\\tau) = \\sqrt{
        \\frac{1}{2(N-2m)} \\sum_{n=0}^{N-2m-1} (\\bar{y}_{n+m} - \\bar{y}_n)^2
    }

where :math:`m = \\tau / dt` (samples per averaging block) and
:math:`\\bar{y}_n = \\frac{1}{m} \\sum_{i=0}^{m-1} y[n+i]`.

Log-log slope reveals noise type:

+------------+-------+--------------------------------------------+
| Noise type | Slope | Physical meaning                           |
+============+=======+============================================+
| Quantisation | -1  | Bit-depth limited; averaging always helps  |
| White noise  | -0.5 | Shot/thermal; σ ∝ 1/√τ                    |
| Flicker      | ~0   | 1/f noise; irreducible floor               |
| Random walk  | +0.5 | Correlated drift; σ ∝ √τ                  |
| Linear drift | +1   | Systematic trend; dominant at long τ       |
+------------+-------+--------------------------------------------+

Usage
-----
::

    from src.scientific.allan_deviation import allan_deviation

    # shifts_nm: 1-D peak shift time series, dt=integration time in seconds
    result = allan_deviation(shifts_nm, dt=0.05)

    print(f"Optimal averaging time : {result.tau_opt:.2f} s")
    print(f"Noise floor (σ_min)    : {result.sigma_min:.4f} nm")
    print(f"Dominant noise         : {result.noise_type}")
    print(result.summary())

Public API
----------
- ``AllanDeviationResult``  — result dataclass
- ``allan_deviation``       — compute ADEV from a time series
- ``adev_noise_fractions``  — fractional contribution per noise type
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

NoiseType = Literal["white", "flicker", "random_walk", "drift", "mixed", "insufficient_data"]

# Minimum number of samples to compute a meaningful ADEV
_MIN_SAMPLES: int = 10
# Minimum number of tau octaves for noise-type classification
_MIN_TAU_POINTS: int = 4


@dataclass
class AllanDeviationResult:
    """Allan deviation analysis result for one sensor time series.

    Attributes
    ----------
    taus : np.ndarray
        Averaging times in seconds (logarithmically spaced).
    adevs : np.ndarray
        Allan deviation value at each tau (same units as input signal).
    tau_opt : float
        Averaging time that minimises ADEV (optimal integration time, in s).
    sigma_min : float
        Minimum Allan deviation — the fundamental noise floor.
    noise_type : NoiseType
        Dominant noise type at the ADEV minimum: ``'white'``, ``'flicker'``,
        ``'random_walk'``, ``'drift'``, or ``'mixed'``.
    slopes : np.ndarray
        Local log-log slope d(log σ)/d(log τ) at each tau.
        Allows identifying the τ range where each noise type dominates.
    white_noise_coeff : float
        White noise coefficient A so that σ_white ≈ A / √τ.
        Estimated from the slope ≈ -0.5 region (shortest tau values).
        Set to NaN if the short-tau segment is ambiguous.
    flicker_floor : float or None
        Approximate flicker noise floor (τ-independent plateau level).
        Set to None if no clear flicker region is found.
    random_walk_coeff : float or None
        Random-walk coefficient B so that σ_rw ≈ B × √τ.
        Set to None if no clear random-walk region is found.
    tau_drift_onset : float or None
        The tau at which drift begins to dominate (slope crosses +0.5
        decisively). None if no drift onset is detected.
    n_samples : int
        Length of the input time series.
    dt_s : float
        Sample interval in seconds.
    """

    taus: np.ndarray = field(repr=False)
    adevs: np.ndarray = field(repr=False)
    tau_opt: float
    sigma_min: float
    noise_type: NoiseType
    slopes: np.ndarray = field(repr=False)
    white_noise_coeff: float
    flicker_floor: float | None
    random_walk_coeff: float | None
    tau_drift_onset: float | None
    n_samples: int
    dt_s: float

    def summary(self) -> str:
        """Return a one-paragraph human-readable ADEV summary for reports."""
        lines = [
            f"Allan deviation analysis ({self.n_samples} samples, "
            f"dt = {self.dt_s*1000:.1f} ms):",
            f"  Noise floor (σ_min) : {self.sigma_min:.4g}  "
            f"at τ_opt = {self.tau_opt:.3g} s",
            f"  Dominant noise type : {self.noise_type}",
        ]
        if not np.isnan(self.white_noise_coeff):
            lines.append(
                f"  White noise coeff A : {self.white_noise_coeff:.4g}  [σ ≈ A/√τ]"
            )
        if self.flicker_floor is not None:
            lines.append(
                f"  Flicker floor       : {self.flicker_floor:.4g}"
            )
        if self.random_walk_coeff is not None:
            lines.append(
                f"  Random walk coeff B : {self.random_walk_coeff:.4g}  [σ ≈ B·√τ]"
            )
        if self.tau_drift_onset is not None:
            lines.append(
                f"  Drift onset τ       : {self.tau_drift_onset:.3g} s  "
                f"(averaging beyond this degrades precision)"
            )
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        """Serialisable dict for JSON export / dossier inclusion."""
        return {
            "tau_opt_s": round(float(self.tau_opt), 6),
            "sigma_min": round(float(self.sigma_min), 8),
            "noise_type": self.noise_type,
            "white_noise_coeff": (
                round(float(self.white_noise_coeff), 8)
                if not np.isnan(self.white_noise_coeff)
                else None
            ),
            "flicker_floor": (
                round(float(self.flicker_floor), 8)
                if self.flicker_floor is not None
                else None
            ),
            "random_walk_coeff": (
                round(float(self.random_walk_coeff), 8)
                if self.random_walk_coeff is not None
                else None
            ),
            "tau_drift_onset_s": (
                round(float(self.tau_drift_onset), 6)
                if self.tau_drift_onset is not None
                else None
            ),
            "n_samples": self.n_samples,
            "dt_s": self.dt_s,
            "taus": [round(float(v), 6) for v in self.taus],
            "adevs": [round(float(v), 8) for v in self.adevs],
        }


def _overlapping_adev(data: np.ndarray, m: int) -> float:
    """Compute the overlapping Allan deviation for a single averaging factor m.

    Uses the overlapping (maximum-likelihood) estimator which is more
    efficient than the non-overlapping estimator (same tau, more pairs used).

    Parameters
    ----------
    data : 1-D array, length N
    m    : averaging factor (samples per block), m >= 1

    Returns
    -------
    float — Allan deviation value (NaN if insufficient data)
    """
    n = len(data)
    if m < 1 or n < 2 * m + 1:
        return float("nan")

    # Build cluster means using cumulative sum for O(N) instead of O(N·m)
    cs = np.concatenate(([0.0], np.cumsum(data)))
    block_sums = cs[m:] - cs[:-m]          # length N - m + 1
    block_means = block_sums / m            # ȳ_n for every starting index n

    # Overlapping differences: ȳ_{n+m} - ȳ_n
    diffs = block_means[m:] - block_means[:-m]   # length N - 2m + 1
    n_pairs = len(diffs)
    if n_pairs < 1:
        return float("nan")

    adev2 = float(np.mean(diffs ** 2)) / 2.0
    return float(np.sqrt(max(adev2, 0.0)))


def allan_deviation(
    signal: np.ndarray,
    dt: float = 1.0,
    taus: np.ndarray | None = None,
    n_tau_per_decade: int = 5,
) -> AllanDeviationResult:
    """Compute the overlapping Allan deviation of a sensor time series.

    Parameters
    ----------
    signal : array_like, shape (N,)
        Sensor time series (e.g. peak wavelength shifts in nm, or raw
        intensity values). Must be uniformly sampled.
    dt : float
        Sampling interval in seconds. Default 1.0.
        Example: for a 20 Hz acquisition rate, dt = 0.05.
    taus : array_like or None
        Explicit averaging times (s) to evaluate. If None, a logarithmic
        grid spanning dt → N*dt/3 is used (3 full-width windows minimum).
    n_tau_per_decade : int
        Points per decade when using the automatic log grid. Default 5.

    Returns
    -------
    AllanDeviationResult
        All noise characterisation metrics.

    Raises
    ------
    ValueError
        If ``signal`` has fewer than ``_MIN_SAMPLES`` points or ``dt <= 0``.
    """
    data = np.asarray(signal, dtype=float).ravel()
    n = len(data)

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if n < _MIN_SAMPLES:
        raise ValueError(
            f"Allan deviation requires at least {_MIN_SAMPLES} samples; "
            f"got {n}. Run a longer baseline before requesting ADEV."
        )

    # Remove NaN / Inf
    valid_mask = np.isfinite(data)
    if not valid_mask.all():
        n_bad = int((~valid_mask).sum())
        log.warning("allan_deviation: removed %d non-finite values", n_bad)
        data = data[valid_mask]
        n = len(data)
        if n < _MIN_SAMPLES:
            raise ValueError(
                f"After removing non-finite values, only {n} samples remain "
                f"(need {_MIN_SAMPLES})."
            )

    # Averaging factor grid
    if taus is not None:
        m_array = np.unique(np.round(np.asarray(taus, dtype=float) / dt).astype(int))
        m_array = m_array[(m_array >= 1) & (m_array <= n // 3)]
    else:
        m_max = max(1, n // 3)
        if m_max <= 1:
            m_array = np.array([1], dtype=int)
        else:
            log_min, log_max = 0.0, np.log10(float(m_max))
            n_pts = max(4, int((log_max - log_min) * n_tau_per_decade) + 1)
            m_float = np.logspace(log_min, log_max, n_pts)
            m_array = np.unique(np.round(m_float).astype(int))
            m_array = m_array[(m_array >= 1) & (m_array <= m_max)]

    if len(m_array) == 0:
        m_array = np.array([1], dtype=int)

    taus_out = m_array * dt
    adevs_out = np.array([_overlapping_adev(data, int(m)) for m in m_array])

    # Remove NaN entries (insufficient data)
    valid = np.isfinite(adevs_out)
    taus_out = taus_out[valid]
    adevs_out = adevs_out[valid]
    m_valid = m_array[valid]

    if len(adevs_out) == 0:
        return AllanDeviationResult(
            taus=np.array([dt]),
            adevs=np.array([float("nan")]),
            tau_opt=dt,
            sigma_min=float("nan"),
            noise_type="insufficient_data",
            slopes=np.array([float("nan")]),
            white_noise_coeff=float("nan"),
            flicker_floor=None,
            random_walk_coeff=None,
            tau_drift_onset=None,
            n_samples=n,
            dt_s=dt,
        )

    # Optimal averaging time (minimum ADEV)
    min_idx = int(np.argmin(adevs_out))
    tau_opt = float(taus_out[min_idx])
    sigma_min = float(adevs_out[min_idx])

    # Local log-log slopes  d(log σ) / d(log τ)
    slopes = _compute_slopes(taus_out, adevs_out)

    # Noise type classification
    noise_type = _classify_noise(slopes, taus_out, adevs_out, min_idx)

    # White noise coefficient A (fit σ ≈ A/√τ on short-tau segment)
    white_noise_coeff = _fit_white_noise_coeff(taus_out, adevs_out, slopes)

    # Flicker floor (plateau region where |slope| < 0.2)
    flicker_floor = _fit_flicker_floor(taus_out, adevs_out, slopes)

    # Random walk coefficient B (fit σ ≈ B·√τ)
    random_walk_coeff = _fit_rw_coeff(taus_out, adevs_out, slopes)

    # Drift onset (first tau where slope > 0.5 decisively)
    tau_drift_onset = _find_drift_onset(taus_out, slopes)

    return AllanDeviationResult(
        taus=taus_out,
        adevs=adevs_out,
        tau_opt=tau_opt,
        sigma_min=sigma_min,
        noise_type=noise_type,
        slopes=slopes,
        white_noise_coeff=white_noise_coeff,
        flicker_floor=flicker_floor,
        random_walk_coeff=random_walk_coeff,
        tau_drift_onset=tau_drift_onset,
        n_samples=n,
        dt_s=dt,
    )


# ---------------------------------------------------------------------------
# Private analysis helpers
# ---------------------------------------------------------------------------

def _compute_slopes(taus: np.ndarray, adevs: np.ndarray) -> np.ndarray:
    """Compute local log-log slopes via central finite differences."""
    log_t = np.log10(taus)
    log_a = np.log10(np.maximum(adevs, 1e-20))
    slopes = np.gradient(log_a, log_t)
    return slopes


def _classify_noise(
    slopes: np.ndarray,
    taus: np.ndarray,
    adevs: np.ndarray,
    min_idx: int,
) -> NoiseType:
    """Classify dominant noise type from slope near the ADEV minimum."""
    if len(slopes) < _MIN_TAU_POINTS:
        return "insufficient_data"

    # Use the first third of the tau range (short-τ white noise region)
    n_short = max(1, len(slopes) // 3)
    short_slope = float(np.median(slopes[:n_short]))

    # Use the last third (long-τ drift region)
    n_long = max(1, len(slopes) // 3)
    long_slope = float(np.median(slopes[-n_long:]))

    # Slope at the minimum
    min_slope = float(slopes[min_idx]) if min_idx < len(slopes) else 0.0

    # Classify: near the minimum, slope crosses zero
    # Short-tau behaviour dominates noise type nomenclature
    if abs(short_slope - (-0.5)) < 0.25:
        if long_slope > 0.3:
            return "random_walk"      # classic white→rw pattern
        if abs(long_slope) < 0.2:
            return "flicker"          # white short, flicker floor long
        return "white"
    if abs(short_slope) < 0.25:
        return "flicker"
    if short_slope > 0.3:
        return "drift"
    if long_slope > 0.7:
        return "random_walk"
    return "mixed"


def _fit_white_noise_coeff(
    taus: np.ndarray,
    adevs: np.ndarray,
    slopes: np.ndarray,
) -> float:
    """Fit A such that σ ≈ A / √τ in the white-noise region (slope ≈ -0.5)."""
    mask = slopes < -0.3
    if mask.sum() < 2:
        # Fallback: use shortest-tau point
        return float(adevs[0] * np.sqrt(taus[0]))
    t_wn = taus[mask]
    a_wn = adevs[mask]
    # A = σ × √τ — take median for robustness
    coeffs = a_wn * np.sqrt(t_wn)
    return float(np.median(coeffs))


def _fit_flicker_floor(
    taus: np.ndarray,
    adevs: np.ndarray,
    slopes: np.ndarray,
) -> float | None:
    """Estimate flicker floor as median ADEV in flat-slope (|slope| < 0.2) region."""
    mask = np.abs(slopes) < 0.2
    if mask.sum() < 2:
        return None
    return float(np.median(adevs[mask]))


def _fit_rw_coeff(
    taus: np.ndarray,
    adevs: np.ndarray,
    slopes: np.ndarray,
) -> float | None:
    """Fit B such that σ ≈ B × √τ in the random-walk region (slope ≈ +0.5)."""
    mask = slopes > 0.3
    if mask.sum() < 2:
        return None
    t_rw = taus[mask]
    a_rw = adevs[mask]
    # B = σ / √τ — take median
    coeffs = a_rw / np.sqrt(t_rw)
    return float(np.median(coeffs))


def _find_drift_onset(taus: np.ndarray, slopes: np.ndarray) -> float | None:
    """Find the first tau where the slope exceeds +0.5 persistently."""
    if len(taus) < 3:
        return None
    for i in range(1, len(slopes)):
        if slopes[i] > 0.5 and (i + 1 >= len(slopes) or slopes[i + 1] > 0.3):
            return float(taus[i])
    return None


def adev_noise_fractions(result: AllanDeviationResult) -> dict[str, float]:
    """Estimate the fractional power of each noise type.

    Uses the ADEV decomposition principle: at τ_opt, white noise and random
    walk contribute roughly equally, and the flicker floor contributes the
    remainder.  This is a heuristic decomposition for reporting purposes only.

    Parameters
    ----------
    result : AllanDeviationResult

    Returns
    -------
    dict with keys ``'white'``, ``'flicker'``, ``'random_walk'`` summing to 1.0.
    """
    if not np.isfinite(result.sigma_min) or result.sigma_min == 0:
        return {"white": float("nan"), "flicker": float("nan"), "random_walk": float("nan")}

    # Estimated contributions at tau_opt
    white = (
        (result.white_noise_coeff / np.sqrt(result.tau_opt)) ** 2
        if not np.isnan(result.white_noise_coeff)
        else 0.0
    )
    rw = (
        (result.random_walk_coeff * np.sqrt(result.tau_opt)) ** 2
        if result.random_walk_coeff is not None
        else 0.0
    )
    flicker = result.flicker_floor**2 if result.flicker_floor is not None else 0.0

    total = white + rw + flicker
    if total == 0:
        return {"white": 1.0, "flicker": 0.0, "random_walk": 0.0}

    return {
        "white": round(float(white / total), 4),
        "flicker": round(float(flicker / total), 4),
        "random_walk": round(float(rw / total), 4),
    }
