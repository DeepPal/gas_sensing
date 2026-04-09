"""
src.scientific.selectivity
==========================
Cross-sensitivity and selectivity analysis for LSPR gas sensors.

Background
----------
A sensor tuned for analyte A that also responds to interferent B has a
cross-sensitivity coefficient:

    K_{B/A} = S_B / S_A

where S_A and S_B are the calibration sensitivities (slope of response vs
concentration) for A and B respectively.  K close to 0 → high selectivity;
K = 1 → equal sensitivity; K > 1 → more sensitive to interferent than target.

The selectivity factor (selectivity coefficient) is defined such that:

    c_interferent = K_{B/A} × c_apparent_A

i.e. a concentration ``c`` of B appears as ``K × c`` of A in the reading.

Reference
---------
IUPAC (2000). *Nomenclature in evaluation of analytical methods including
detection and quantification capabilities*, Pure Appl. Chem. 72, 835–845.

Nomenclature after: Umezawa, Y. et al. (2000). *Potentiometric selectivity
coefficients of ion-selective electrodes*, Pure Appl. Chem. 72, 1851–2082
(extended to optical sensors here).

Public API
----------
- ``compute_cross_sensitivity``   — pairwise K for one target vs interferents
- ``selectivity_matrix``          — full N×N selectivity matrix for N gases
- ``SelectivityResult``           — dataclass with matrix, rankings, recommendations
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SelectivityResult:
    """Full selectivity characterisation for a set of gases.

    Attributes
    ----------
    gases:
        Ordered list of gas names.
    sensitivities:
        Dict mapping gas → calibration slope (sensitivity, Δλ/ppm or units/ppm).
    matrix:
        N×N array where ``matrix[i, j] = K_{j/i}`` = cross-sensitivity of
        interferent *j* against target *i*.  Diagonal is 1.0 by definition.
        Values > 0.1 are significant; values > 0.5 are problematic.
    rankings:
        For each target gas, interferents ranked by cross-sensitivity (worst first).
    interpretation:
        Human-readable selectivity assessment per target gas.
    worst_interferents:
        Dict mapping target → (worst_interferent_name, K_value).
    """

    gases: list[str]
    sensitivities: dict[str, float]
    matrix: np.ndarray = field(repr=False)
    rankings: dict[str, list[tuple[str, float]]]
    interpretation: dict[str, str]
    worst_interferents: dict[str, tuple[str, float]]
    metadata_warnings: list[str] = field(default_factory=list)
    """Warnings about inter-session validity of K values (chip mismatch, date drift, etc.)"""

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for JSON export / MLflow logging."""
        return {
            "gases": self.gases,
            "sensitivities": self.sensitivities,
            "matrix": self.matrix.tolist(),
            "rankings": {
                k: [(name, float(kval)) for name, kval in v] for k, v in self.rankings.items()
            },
            "interpretation": self.interpretation,
            "worst_interferents": {
                k: (name, float(kval)) for k, (name, kval) in self.worst_interferents.items()
            },
            "metadata_warnings": self.metadata_warnings,
        }

    def summary_table(self) -> str:
        """Return a formatted ASCII table of the selectivity matrix."""
        n = len(self.gases)
        col_w = 14
        header = "Target \\ Interf".ljust(col_w) + "".join(
            g[:col_w].rjust(col_w) for g in self.gases
        )
        rows = [header, "-" * (col_w * (n + 1))]
        for i, tgt in enumerate(self.gases):
            row = tgt[:col_w].ljust(col_w)
            for j in range(n):
                val = self.matrix[i, j]
                cell = "1.000" if i == j else f"{val:.4f}"
                row += cell.rjust(col_w)
            rows.append(row)
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_cross_sensitivity(
    target_sensitivity: float,
    interferent_sensitivities: dict[str, float],
) -> dict[str, float]:
    """Compute cross-sensitivity coefficients K_{j/target} for one target gas.

    Parameters
    ----------
    target_sensitivity:
        Calibration slope of the target analyte (Δλ/ppm or signal/ppm).
        Should be non-zero; sign is preserved.
    interferent_sensitivities:
        Dict mapping interferent gas name → its calibration slope.

    Returns
    -------
    dict
        Mapping interferent name → K value.  |K| > 0.1 is significant.
    """
    if abs(target_sensitivity) < 1e-12:
        raise ValueError(
            f"Target sensitivity is too close to zero ({target_sensitivity:.2e}); "
            "cannot compute selectivity coefficient."
        )
    return {name: float(s / target_sensitivity) for name, s in interferent_sensitivities.items()}


def selectivity_matrix(
    gas_sensitivities: dict[str, float],
    threshold_significant: float = 0.1,
    threshold_problematic: float = 0.5,
    session_metadata: dict[str, dict] | None = None,
) -> SelectivityResult:
    """Build the full cross-sensitivity matrix for all gas pairs.

    For each target gas *i* and interferent gas *j*, computes::

        K[i, j] = sensitivity_j / sensitivity_i

    The diagonal K[i, i] = 1.0 by definition.

    .. warning::

        **Validity condition**: All gas sensitivities MUST be measured under
        identical experimental conditions (same chip, same day / chip age,
        same integration time, same temperature).  If gases were measured in
        different sessions, the K values will be contaminated by drift between
        sessions, not by true cross-sensitivity.  Provide ``session_metadata``
        to enable automated validation.

    Parameters
    ----------
    gas_sensitivities:
        Dict mapping gas name → calibration sensitivity (slope, signed).
        All gases must have been measured under the same conditions.
    threshold_significant:
        |K| above this triggers a "significant cross-sensitivity" warning.
        Default 0.1 (10 % — common analytical chemistry criterion).
    threshold_problematic:
        |K| above this triggers a "problematic" flag.  Default 0.5 (50 %).
    session_metadata:
        Optional dict mapping gas name → metadata dict with keys
        ``"chip_serial"``, ``"measurement_date"`` (ISO date string),
        ``"integration_time_ms"`` (float), ``"temperature_c"`` (float).
        When provided, gases with mismatched chip_serial or measurement_date
        (>1 day apart) trigger a warning that the selectivity matrix may be
        contaminated by inter-session drift.

    Returns
    -------
    :class:`SelectivityResult`

    Examples
    --------
    >>> from src.scientific.selectivity import selectivity_matrix
    >>> sensitivities = {
    ...     "Ethanol": -2.1,   # Δλ/ppm
    ...     "IPA":     -1.8,
    ...     "Methanol": -0.9,
    ... }
    >>> result = selectivity_matrix(sensitivities)
    >>> print(result.summary_table())
    """
    gases = list(gas_sensitivities.keys())
    n = len(gases)

    if n < 2:
        raise ValueError(f"At least 2 gases are required for selectivity analysis, got {n}.")

    # ── Session metadata validation ──────────────────────────────────────────
    # K_B/A = S_B / S_A is only meaningful when both sensitivities were measured
    # under *identical* conditions. Mismatched chip serials or measurement dates
    # indicate inter-session drift contamination of the K values.
    _meta_warnings: list[str] = []
    if session_metadata is not None and len(session_metadata) >= 2:
        from datetime import datetime, timedelta

        chip_serials = {
            g: session_metadata[g].get("chip_serial")
            for g in gases if g in session_metadata
        }
        meas_dates = {
            g: session_metadata[g].get("measurement_date")
            for g in gases if g in session_metadata
        }

        # Check chip serial consistency
        unique_chips = set(v for v in chip_serials.values() if v)
        if len(unique_chips) > 1:
            _meta_warnings.append(
                f"VALIDITY WARNING: Gases were measured on different chip batches "
                f"({unique_chips}). Cross-sensitivity K values may reflect chip-to-chip "
                f"sensitivity variation, not true selectivity. "
                f"Re-measure all gases on the same chip for valid K values."
            )

        # Check date consistency (>1 day apart = potential drift contamination)
        parsed_dates: dict[str, datetime] = {}
        for g, d in meas_dates.items():
            if d:
                try:
                    parsed_dates[g] = datetime.fromisoformat(str(d)[:10])
                except (ValueError, TypeError):
                    pass
        if len(parsed_dates) >= 2:
            date_range = max(parsed_dates.values()) - min(parsed_dates.values())
            if date_range > timedelta(days=1):
                _meta_warnings.append(
                    f"VALIDITY WARNING: Gases were measured on different days "
                    f"(range: {date_range.days} days). Baseline drift between sessions "
                    f"may contaminate K values. Confirm chip sensitivity was stable "
                    f"(check sensor_memory.json LOD/sensitivity trends)."
                )

        # Check integration time consistency
        int_times = {
            g: session_metadata[g].get("integration_time_ms")
            for g in gases if g in session_metadata
        }
        unique_its = set(v for v in int_times.values() if v is not None)
        if len(unique_its) > 1 and max(unique_its) / min(unique_its) > 1.1:
            _meta_warnings.append(
                f"NOTE: Integration times differ across gases ({unique_its} ms). "
                "Sensitivity slopes are integration-time-dependent for some noise regimes. "
                "Verify that sensitivities were acquired at the same integration setting."
            )

        for w in _meta_warnings:
            log.warning(w)

    sens = np.array([gas_sensitivities[g] for g in gases], dtype=float)

    # Build N×N matrix: matrix[i, j] = K_{j/i}
    matrix: np.ndarray = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            elif abs(sens[i]) < 1e-12:
                matrix[i, j] = float("inf")
            else:
                matrix[i, j] = float(sens[j] / sens[i])

    # Rankings: for each target, sort interferents by |K| descending
    rankings: dict[str, list[tuple[str, float]]] = {}
    for i, tgt in enumerate(gases):
        pairs = [(gases[j], float(matrix[i, j])) for j in range(n) if j != i]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        rankings[tgt] = pairs

    # Worst interferent per target
    worst_interferents: dict[str, tuple[str, float]] = {}
    for tgt, rank in rankings.items():
        if rank:
            worst_interferents[tgt] = rank[0]

    # Interpretation strings
    interpretation: dict[str, str] = {}
    for i, tgt in enumerate(gases):
        interf_strs = []
        for j, name in enumerate(gases):
            if j == i:
                continue
            k_val = matrix[i, j]
            abs_k = abs(k_val)
            if abs_k < threshold_significant:
                level = "negligible"
            elif abs_k < threshold_problematic:
                level = "SIGNIFICANT"
            else:
                level = "PROBLEMATIC"
            interf_strs.append(f"{name}: K={k_val:+.4f} ({level})")
        max_k = max(abs(matrix[i, j]) for j in range(n) if j != i)
        if max_k < threshold_significant:
            overall = "HIGH selectivity — no significant cross-sensitivity detected."
        elif max_k < threshold_problematic:
            overall = "MODERATE selectivity — at least one significant interferent."
        else:
            overall = "LOW selectivity — at least one interferent has K > 0.5. Report carefully."
        interpretation[tgt] = f"{overall} | {'; '.join(interf_strs)}"

    log.info(
        "Selectivity matrix (%d gases): %s",
        n,
        ", ".join(
            f"{tgt}→worst={intf}(K={k:.3f})" for tgt, (intf, k) in worst_interferents.items()
        ),
    )

    return SelectivityResult(
        gases=gases,
        sensitivities={g: float(gas_sensitivities[g]) for g in gases},
        matrix=matrix,
        rankings=rankings,
        interpretation=interpretation,
        worst_interferents=worst_interferents,
        metadata_warnings=_meta_warnings,
    )


def selectivity_from_calibration_data(
    gas_data: dict[str, tuple[np.ndarray, np.ndarray]],
    regression: str = "ols",
    session_metadata: dict[str, dict] | None = None,
) -> SelectivityResult:
    """Build selectivity matrix directly from raw calibration arrays.

    This is the convenience entry point: pass (concentrations, responses)
    per gas, get back the full selectivity characterisation.

    Parameters
    ----------
    gas_data:
        Dict mapping gas name → ``(concentrations, responses)`` where both
        are 1-D arrays of the same length.  Concentration units must be
        consistent across gases (all in ppm).
    regression:
        Sensitivity estimation method: ``'ols'`` (ordinary least squares) or
        ``'huber'`` (robust, outlier-resistant).
    session_metadata:
        Optional dict mapping gas name → metadata dict (see
        :func:`selectivity_matrix` for the expected keys: ``chip_serial``,
        ``measurement_date``, ``integration_time_ms``, ``temperature_c``).
        Passed through to :func:`selectivity_matrix` to validate that all
        gas sensitivities were measured under identical conditions.

    Returns
    -------
    :class:`SelectivityResult`

    Examples
    --------
    >>> data = {
    ...     "Ethanol": (np.array([0.5,1,2,5]), np.array([-1.1,-2.0,-4.1,-10.0])),
    ...     "IPA":     (np.array([0.5,1,2,5]), np.array([-0.9,-1.7,-3.5, -8.5])),
    ... }
    >>> result = selectivity_from_calibration_data(data)
    """
    from src.scientific.lod import calculate_sensitivity, robust_sensitivity

    sensitivities: dict[str, float] = {}
    for gas, (c_arr, r_arr) in gas_data.items():
        c = np.asarray(c_arr, dtype=float).ravel()
        r = np.asarray(r_arr, dtype=float).ravel()
        if len(c) < 2:
            raise ValueError(f"Gas '{gas}' needs at least 2 calibration points, got {len(c)}.")
        if regression == "ols":
            slope, _, _, _ = calculate_sensitivity(c, r)
        elif regression == "huber":
            result = robust_sensitivity(c, r, method="huber")
            slope = float(result["slope"])  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown regression method: {regression!r}. Choose 'ols' or 'huber'.")
        sensitivities[gas] = float(slope)

    return selectivity_matrix(sensitivities, session_metadata=session_metadata)
