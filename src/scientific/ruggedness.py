"""
src.scientific.ruggedness
==========================
**Youden ruggedness test** and **spike recovery** protocol for LSPR/SPR
optical gas sensor analytical method validation.

Both tests are required for ICH Q2(R1), AOAC 2002.06, and EPA analytical
method validation when submitting to:

  - Sensors & Actuators B (Elsevier) — mandatory when claiming LOD
  - Analytical Chemistry (ACS) — standard review criterion
  - AOAC Methods Committee publications

Youden Ruggedness Test
-----------------------
Based on: W.J. Youden & E.H. Steiner, "Statistical Manual of the
AOAC", AOAC International, 1975.

Seven factors (A–G) are varied ±1 level in a Plackett-Burman design
of 8 experiments. Each factor's effect is estimated from the signed
difference of responses at the two levels:

.. math::

    e_k = \\frac{ \\sum_{i: x_{ik}=+1} y_i - \\sum_{i: x_{ik}=-1} y_i }{4}

The residual standard deviation from factor effects estimates reproducibility.

Rule of thumb (Youden 1967): |e_k| < 2·s_resid → factor is *not critical*;
otherwise the method is rugged only when factor k is tightly controlled.

Spike Recovery
--------------
The standard protocol adds known concentrations (low / mid / high, typically
~0.5×, 1×, 2× of the expected analyte concentration) to the matrix and
measures the actual sensor response to compute recovery fraction:

.. math::

    R_k = \\frac{C_{\\text{found},k} - C_{\\text{background}}}{C_{\\text{added},k}}

Acceptable ICH Q2(R1) recovery: 98–102 % (routine QC: 90–110 %).

Public API
----------
- ``YoudensDesign``                — 8-run Plackett-Burman design matrix
- ``RuggednessResult``             — result dataclass
- ``youden_ruggedness``            — estimate factor effects from measurements
- ``SpikeRecoveryResult``          — spike recovery dataclass
- ``spike_recovery``               — compute recovery % with CI
- ``recovery_acceptance``          — classify pass/fail per ICH Q2(R1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

# ICH Q2(R1) default acceptance limits for spike recovery
_ICH_LOW = 0.98   # 98 %
_ICH_HIGH = 1.02  # 102 %
_ROUTINE_LOW = 0.90
_ROUTINE_HIGH = 1.10

# ─── Youden 8-run Plackett-Burman for 7 factors ───────────────────────────────
#
# Standard PB(8) design generated from cyclic generator row [+,+,+,-,+,-,-]
# (Plackett & Burman, Biometrika 1946; Youden & Steiner 1975, Table 3).
#
# Properties:
#   - Each column has exactly 4 runs at +1 and 4 at −1 (balanced)
#   - All pairs of columns are orthogonal (column dot-products = 0)
#   - Run 8 is the all-minus fold-over run
#
_YPB_DESIGN = np.array([
    [+1, +1, +1, -1, +1, -1, -1],  # run 1
    [-1, +1, +1, +1, -1, +1, -1],  # run 2
    [-1, -1, +1, +1, +1, -1, +1],  # run 3
    [+1, -1, -1, +1, +1, +1, -1],  # run 4
    [-1, +1, -1, -1, +1, +1, +1],  # run 5
    [+1, -1, +1, -1, -1, +1, +1],  # run 6
    [+1, +1, -1, +1, -1, -1, +1],  # run 7
    [-1, -1, -1, -1, -1, -1, -1],  # run 8  (all-minus)
], dtype=float)


class YoudensDesign:
    """8-run Plackett-Burman design for 7 two-level factors (Youden & Steiner 1975).

    Attributes
    ----------
    factors : list[str]
        Factor labels (length 7).
    levels : dict[str, tuple[value, value]]
        Low / high levels for each factor.
    design : np.ndarray of shape (8, 7)
        ±1 design matrix.  Rows = runs, columns = factors A–G.

    Usage
    -----
    ::

        d = YoudensDesign(
            factors=["integration_ms", "temperature_C", "flow_rate_sccm",
                     "baseline_wait_s", "purge_wait_s", "fiber_bend_mm", "lamp_power_pct"],
            levels={
                "integration_ms":   (48, 52),
                "temperature_C":    (22, 24),
                "flow_rate_sccm":   (95, 105),
                "baseline_wait_s":  (58, 62),
                "purge_wait_s":     (28, 32),
                "fiber_bend_mm":    (9, 11),
                "lamp_power_pct":   (98, 102),
            },
        )
        for run_idx in range(8):
            settings = d.run_settings(run_idx)
            # → {"integration_ms": 48, "temperature_C": 24, ...}
    """

    N_RUNS = 8
    N_FACTORS = 7

    def __init__(
        self,
        factors: list[str],
        levels: dict[str, tuple[float, float]],
    ) -> None:
        if len(factors) != self.N_FACTORS:
            raise ValueError(
                f"Youden design requires exactly {self.N_FACTORS} factors; "
                f"got {len(factors)}"
            )
        missing = [f for f in factors if f not in levels]
        if missing:
            raise ValueError(f"Missing level definitions for factors: {missing}")

        self.factors = list(factors)
        self.levels = {f: tuple(levels[f]) for f in factors}  # type: ignore[assignment]
        self.design: np.ndarray = _YPB_DESIGN.copy()

    def run_settings(self, run_index: int) -> dict[str, float]:
        """Return the factor settings for a specific run (0-indexed).

        Parameters
        ----------
        run_index : int
            0-based index, 0–7.

        Returns
        -------
        dict[str, float]
            Mapping factor_name → actual level value.
        """
        if not 0 <= run_index < self.N_RUNS:
            raise IndexError(f"run_index must be 0–{self.N_RUNS - 1}")
        row = self.design[run_index]  # shape (7,)
        return {
            f: (self.levels[f][1] if row[i] > 0 else self.levels[f][0])
            for i, f in enumerate(self.factors)
        }

    def all_run_settings(self) -> list[dict[str, float]]:
        """Return settings for all 8 runs."""
        return [self.run_settings(i) for i in range(self.N_RUNS)]


# ─── Ruggedness result ────────────────────────────────────────────────────────

@dataclass
class RuggednessResult:
    """Output of :func:`youden_ruggedness`.

    Attributes
    ----------
    factors : list[str]
        Factor labels (length 7).
    effects : np.ndarray
        Estimated main effect of each factor on the response.
        Effect = (mean response at +1) − (mean response at −1).
    residual_std : float
        Residual standard deviation from factor effects — estimates
        method reproducibility under the tested range of variation.
    critical_factors : list[str]
        Factors where |effect| ≥ 2 · residual_std  (Youden threshold).
        These must be tightly controlled in the published method.
    response_mean : float
        Grand mean of the 8 responses.
    response_std : float
        Standard deviation of the 8 responses.
    """

    factors: list[str]
    effects: np.ndarray
    residual_std: float
    critical_factors: list[str]
    response_mean: float
    response_std: float

    def summary(self) -> str:
        """Human-readable summary suitable for a lab notebook."""
        lines = [
            "── Youden Ruggedness Test ──────────────────",
            f"Grand mean: {self.response_mean:.4g}  σ_resid: {self.residual_std:.4g}",
            f"{'Factor':<22}  {'Effect':>10}  {'|e|/σ':>8}  Status",
            "-" * 56,
        ]
        for f, e in zip(self.factors, self.effects):
            ratio = abs(e) / max(self.residual_std, 1e-12)
            status = "CRITICAL" if f in self.critical_factors else "ok"
            lines.append(f"{f:<22}  {e:>10.4g}  {ratio:>8.2f}  {status}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "factors": self.factors,
            "effects": self.effects.tolist(),
            "residual_std": self.residual_std,
            "critical_factors": self.critical_factors,
            "response_mean": self.response_mean,
            "response_std": self.response_std,
        }


def youden_ruggedness(
    design: YoudensDesign,
    responses: np.ndarray | list[float],
    *,
    threshold_sigma: float = 2.0,
) -> RuggednessResult:
    """Compute Youden ruggedness estimates from 8 experimental responses.

    Parameters
    ----------
    design : YoudensDesign
        The 8-run Plackett-Burman design used to generate the experiments.
    responses : array_like of shape (8,)
        Measured response for each of the 8 runs (in the same order as
        ``design.design``).  Use the primary metric of interest, e.g.
        peak-shift Δλ (nm) or inferred concentration (ppm).
    threshold_sigma : float, optional
        Multiplier for the critical-factor threshold.
        Default 2.0 follows Youden & Steiner 1975.

    Returns
    -------
    RuggednessResult

    Raises
    ------
    ValueError
        If ``responses`` does not have exactly 8 elements, or if any
        response is non-finite.
    """
    y = np.asarray(responses, dtype=float)
    if y.shape != (8,):
        raise ValueError(
            f"Youden ruggedness requires exactly 8 responses; got shape {y.shape}"
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("All responses must be finite (no NaN/inf)")

    X = design.design  # shape (8, 7), ±1

    # Each effect = (sum of responses at +1 level) / 4 − (sum at −1 level) / 4
    # Equivalently: e_k = (X[:, k] @ y) / 4   (since ±1 and 8 runs → 4 per level)
    effects: np.ndarray = (X.T @ y) / 4.0  # shape (7,)

    # Residual standard deviation: variance not explained by the 7 main effects
    # With 8 runs and 7 effect estimates, only 1 degree of freedom remains.
    # s² = (total_SS − effects_SS) / df
    total_ss = float(np.sum((y - y.mean()) ** 2))
    effects_ss = float(np.sum(4.0 * effects ** 2))  # each effect contributes 4·e²
    resid_ss = max(total_ss - effects_ss, 0.0)
    df = max(8 - 7 - 1, 1)  # 1 dof after grand mean + 7 effects
    residual_std = float(np.sqrt(resid_ss / df))

    critical: list[str] = [
        f for f, e in zip(design.factors, effects)
        if (
            abs(e) > 0 and abs(e) >= threshold_sigma * residual_std
            if residual_std > 0
            else abs(e) > 0
        )
    ]

    return RuggednessResult(
        factors=list(design.factors),
        effects=effects,
        residual_std=residual_std,
        critical_factors=critical,
        response_mean=float(y.mean()),
        response_std=float(y.std(ddof=1)),
    )


# ─── Spike recovery ──────────────────────────────────────────────────────────

AcceptanceLevel = Literal["ich_q2r1", "routine", "custom"]


@dataclass
class SpikeRecoveryPoint:
    """Recovery at a single spike concentration level.

    Attributes
    ----------
    added_conc : float
        Nominal added concentration (ppm or same unit as calibration).
    found_conc : float
        Concentration measured by the sensor after spiking.
    background_conc : float
        Pre-spike background concentration (blank measurement).
    recovery : float
        Recovery fraction R = (found − background) / added.
        1.0 = 100 % recovery.
    recovery_pct : float
        Same as ``recovery * 100``.
    pass_ich : bool
        True if 0.98 ≤ R ≤ 1.02 (ICH Q2(R1) criterion).
    pass_routine : bool
        True if 0.90 ≤ R ≤ 1.10 (routine QC criterion).
    """

    added_conc: float
    found_conc: float
    background_conc: float
    recovery: float
    recovery_pct: float
    pass_ich: bool
    pass_routine: bool


@dataclass
class SpikeRecoveryResult:
    """Output of :func:`spike_recovery`.

    Attributes
    ----------
    points : list[SpikeRecoveryPoint]
        Recovery data per spike level.
    mean_recovery : float
        Mean recovery fraction across all spike levels.
    std_recovery : float
        Standard deviation of recovery fractions.
    overall_pass_ich : bool
        All levels pass ICH Q2(R1) bounds.
    overall_pass_routine : bool
        All levels pass routine QC bounds.
    n_levels : int
        Number of spike concentration levels tested.
    """

    points: list[SpikeRecoveryPoint]
    mean_recovery: float
    std_recovery: float
    overall_pass_ich: bool
    overall_pass_routine: bool
    n_levels: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_levels = len(self.points)

    def summary(self) -> str:
        """Human-readable summary suitable for a lab notebook."""
        lines = [
            "── Spike Recovery ─────────────────────────────────",
            f"{'Added (ppm)':<14}  {'Found (ppm)':<14}  {'Recovery':>10}  ICH  Routine",
            "-" * 60,
        ]
        for p in self.points:
            ich = "PASS" if p.pass_ich else "fail"
            rou = "PASS" if p.pass_routine else "fail"
            lines.append(
                f"{p.added_conc:<14.4g}  {p.found_conc:<14.4g}"
                f"  {p.recovery_pct:>9.2f}%  {ich:<4} {rou}"
            )
        lines += [
            "-" * 60,
            f"Mean recovery: {self.mean_recovery * 100:.2f} ± {self.std_recovery * 100:.2f}% (1σ)",
            f"Overall ICH Q2(R1): {'PASS' if self.overall_pass_ich else 'FAIL'}  "
            f"Routine QC: {'PASS' if self.overall_pass_routine else 'FAIL'}",
        ]
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_levels": self.n_levels,
            "mean_recovery_pct": self.mean_recovery * 100,
            "std_recovery_pct": self.std_recovery * 100,
            "overall_pass_ich": self.overall_pass_ich,
            "overall_pass_routine": self.overall_pass_routine,
            "levels": [
                {
                    "added_ppm": p.added_conc,
                    "found_ppm": p.found_conc,
                    "background_ppm": p.background_conc,
                    "recovery_pct": p.recovery_pct,
                    "pass_ich": p.pass_ich,
                    "pass_routine": p.pass_routine,
                }
                for p in self.points
            ],
        }


def spike_recovery(
    added_concentrations: np.ndarray | list[float],
    found_concentrations: np.ndarray | list[float],
    background_concentration: float = 0.0,
    *,
    ich_low: float = _ICH_LOW,
    ich_high: float = _ICH_HIGH,
    routine_low: float = _ROUTINE_LOW,
    routine_high: float = _ROUTINE_HIGH,
) -> SpikeRecoveryResult:
    """Compute spike recovery at each spiked concentration level.

    Parameters
    ----------
    added_concentrations : array_like of shape (n,)
        Nominal concentration added for each spike level (e.g. [50, 100, 200] ppm).
    found_concentrations : array_like of shape (n,)
        Concentration reported by the sensor after spiking.  Must be the same
        length as ``added_concentrations``.
    background_concentration : float, optional
        Blank / pre-spike measurement.  Default 0.0 (clean carrier gas assumed).
    ich_low, ich_high : float, optional
        ICH Q2(R1) acceptance bounds for recovery fraction (default 0.98–1.02).
    routine_low, routine_high : float, optional
        Routine QC acceptance bounds (default 0.90–1.10).

    Returns
    -------
    SpikeRecoveryResult

    Raises
    ------
    ValueError
        If arrays have different lengths, contain non-finite values, or any
        ``added_concentrations`` element is ≤ 0.
    """
    added = np.asarray(added_concentrations, dtype=float)
    found = np.asarray(found_concentrations, dtype=float)
    bg = float(background_concentration)

    if added.ndim != 1 or added.shape != found.shape:
        raise ValueError(
            "added_concentrations and found_concentrations must be 1-D arrays "
            "of equal length"
        )
    if added.size == 0:
        raise ValueError("At least one spike level is required")
    if not np.all(np.isfinite(added)) or not np.all(np.isfinite(found)):
        raise ValueError("Spike concentrations must be finite (no NaN/inf)")
    if np.any(added <= 0):
        raise ValueError("added_concentrations must all be positive (> 0)")

    recoveries: list[float] = []
    points: list[SpikeRecoveryPoint] = []

    for a, f in zip(added.tolist(), found.tolist()):
        r = (f - bg) / a
        recoveries.append(r)
        points.append(
            SpikeRecoveryPoint(
                added_conc=a,
                found_conc=f,
                background_conc=bg,
                recovery=r,
                recovery_pct=r * 100.0,
                pass_ich=bool(ich_low <= r <= ich_high),
                pass_routine=bool(routine_low <= r <= routine_high),
            )
        )

    rec_arr = np.array(recoveries)
    return SpikeRecoveryResult(
        points=points,
        mean_recovery=float(rec_arr.mean()),
        std_recovery=float(rec_arr.std(ddof=1)) if len(rec_arr) > 1 else 0.0,
        overall_pass_ich=all(p.pass_ich for p in points),
        overall_pass_routine=all(p.pass_routine for p in points),
    )


def recovery_acceptance(
    result: SpikeRecoveryResult,
    level: AcceptanceLevel = "ich_q2r1",
) -> bool:
    """Return True if the spike recovery result meets the acceptance criterion.

    Parameters
    ----------
    result : SpikeRecoveryResult
    level : {"ich_q2r1", "routine", "custom"}
        ``"ich_q2r1"``  — 98–102 % (ICH Q2(R1), publication-grade)
        ``"routine"``   — 90–110 % (routine QC)
        ``"custom"``    — checks ``overall_pass_ich`` (same as ``"ich_q2r1"``)

    Returns
    -------
    bool
    """
    if level in ("ich_q2r1", "custom"):
        return result.overall_pass_ich
    if level == "routine":
        return result.overall_pass_routine
    raise ValueError(f"Unknown acceptance level: {level!r}")
