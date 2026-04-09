"""
src.calibration.batch_reproducibility
=======================================
Inter-sensor batch reproducibility analysis.

Background
----------
A single calibration on a single sensor is insufficient for publication or
regulatory submission.  ICH Q2(R1) §5.4 (Reproducibility) requires
demonstrating consistent performance across ≥ 3 independently manufactured
sensors, operators, laboratories, or time points.

This module computes:

* **Inter-sensor RSD** (Relative Standard Deviation) of LOD, LOQ, sensitivity
  and R² across a batch of independently calibrated sensors.
* **Pooled LOD/LOQ** — combines σ_blank estimates across sensors via pooled
  standard deviation (weighted by degrees of freedom), then applies a single
  pooled sensitivity estimate.
* **Batch acceptance criteria** — flags the batch as acceptable when
  inter-sensor LOD RSD < 20% and all R² ≥ 0.99 (configurable).

Public API
----------
- ``BatchReproducibilityReport``  — dataclass of all batch statistics
- ``BatchReproducibilityAnalyzer``— .analyze(sessions) → BatchReproducibilityReport
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np


@dataclass
class BatchReproducibilityReport:
    """Inter-sensor reproducibility statistics for a batch of sensors.

    All RSD values are expressed as a percentage (0–100).
    """

    n_sensors: int = 0
    """Number of sensors / sessions included in the analysis."""

    sensor_ids: list[str] = field(default_factory=list)

    # Per-sensor metrics
    lod_values: list[float] = field(default_factory=list)
    loq_values: list[float] = field(default_factory=list)
    r2_values: list[float] = field(default_factory=list)
    sensitivity_values: list[float] = field(default_factory=list)

    # Batch statistics
    lod_mean: float = float("nan")
    lod_std: float = float("nan")
    lod_rsd_pct: float = float("nan")      # inter-sensor RSD (%)

    loq_mean: float = float("nan")
    loq_std: float = float("nan")
    loq_rsd_pct: float = float("nan")

    pooled_lod: float = float("nan")       # pooled σ_blank approach
    pooled_loq: float = float("nan")

    r2_mean: float = float("nan")
    r2_std: float = float("nan")
    r2_min: float = float("nan")

    sensitivity_mean: float = float("nan")  # nm/ppm
    sensitivity_rsd_pct: float = float("nan")

    # Acceptance verdict
    batch_accepted: bool | None = None
    """True = batch meets acceptance criteria; False = batch failed; None = not evaluated."""

    acceptance_criteria: dict[str, Any] = field(default_factory=dict)
    """Criteria used for acceptance decision."""

    failure_reasons: list[str] = field(default_factory=list)
    """Human-readable list of criteria that failed."""

    summary: str = ""


class BatchReproducibilityAnalyzer:
    """Compute inter-sensor reproducibility from a list of SessionAnalysis objects.

    Usage
    -----
    ::

        analyzer = BatchReproducibilityAnalyzer(
            lod_rsd_limit_pct=20.0,
            min_r2=0.99,
        )
        report = analyzer.analyze(
            sessions=[sa1, sa2, sa3],
            sensor_ids=["S001", "S002", "S003"],
        )
        print(report.summary)
    """

    def __init__(
        self,
        lod_rsd_limit_pct: float = 20.0,
        loq_rsd_limit_pct: float = 20.0,
        min_r2: float = 0.99,
        sensitivity_rsd_limit_pct: float = 15.0,
    ) -> None:
        """
        Parameters
        ----------
        lod_rsd_limit_pct :
            Maximum acceptable inter-sensor LOD RSD (%).  Default 20% per ICH Q2(R1).
        loq_rsd_limit_pct :
            Maximum acceptable inter-sensor LOQ RSD (%).
        min_r2 :
            Minimum acceptable calibration R² for each sensor.
        sensitivity_rsd_limit_pct :
            Maximum acceptable sensitivity (nm/ppm) RSD across sensors.
        """
        self._lod_rsd_limit = lod_rsd_limit_pct
        self._loq_rsd_limit = loq_rsd_limit_pct
        self._min_r2 = min_r2
        self._sens_rsd_limit = sensitivity_rsd_limit_pct

    def analyze(
        self,
        sessions: list[Any],
        sensor_ids: list[str] | None = None,
    ) -> BatchReproducibilityReport:
        """Compute batch reproducibility statistics.

        Parameters
        ----------
        sessions :
            List of ``SessionAnalysis`` objects from independently calibrated sensors.
            Each must have valid ``lod_ppm``, ``loq_ppm``, ``calibration_r2``, and
            ``calibration_concentrations`` / ``calibration_shifts`` fields.
        sensor_ids :
            Optional list of sensor identifiers.  If None, uses S001, S002, ...

        Returns
        -------
        BatchReproducibilityReport
        """
        n = len(sessions)
        report = BatchReproducibilityReport(n_sensors=n)

        if sensor_ids is None:
            sensor_ids = [f"S{i+1:03d}" for i in range(n)]
        if len(sensor_ids) != n:
            raise ValueError(
                f"sensor_ids length ({len(sensor_ids)}) must match sessions length ({n})."
            )
        report.sensor_ids = list(sensor_ids)

        if n < 2:
            warnings.warn(
                "BatchReproducibilityAnalyzer: at least 2 sensors required for reproducibility; "
                f"got {n}.  Statistics will be incomplete.",
                UserWarning,
                stacklevel=2,
            )

        report.acceptance_criteria = {
            "lod_rsd_limit_pct": self._lod_rsd_limit,
            "loq_rsd_limit_pct": self._loq_rsd_limit,
            "min_r2": self._min_r2,
            "sensitivity_rsd_limit_pct": self._sens_rsd_limit,
        }

        # Extract per-sensor metrics
        lod_vals, loq_vals, r2_vals, sens_vals = [], [], [], []
        pooled_variances: list[float] = []  # (ddof * variance) for pooled std
        pooled_dofs: list[int] = []

        for sa in sessions:
            lod = float(getattr(sa, "lod_ppm", float("nan")))
            loq = float(getattr(sa, "loq_ppm", float("nan")))
            r2 = getattr(sa, "calibration_r2", None)
            concs = np.asarray(getattr(sa, "calibration_concentrations", []), dtype=float)
            shifts = np.asarray(getattr(sa, "calibration_shifts", []), dtype=float)

            if not np.isnan(lod):
                lod_vals.append(lod)
            if not np.isnan(loq):
                loq_vals.append(loq)
            if r2 is not None and np.isfinite(r2):
                r2_vals.append(float(r2))

            # Sensitivity from calibration data
            if len(concs) >= 2 and len(concs) == len(shifts):
                from scipy.stats import linregress as _linreg
                slope, *_ = _linreg(concs, shifts)
                sens_vals.append(float(slope))

                # Pooled variance for σ_blank pooling
                resid = shifts - (np.polyval(np.polyfit(concs, shifts, 1), concs))
                n_cal = len(concs)
                if n_cal >= 2:
                    pooled_variances.append(float(np.sum(resid**2)))
                    pooled_dofs.append(n_cal - 2)

        report.lod_values = lod_vals
        report.loq_values = loq_vals
        report.r2_values = r2_vals
        report.sensitivity_values = sens_vals

        def _stats(vals: list[float], name: str) -> tuple[float, float, float]:
            """Return mean, std, rsd%."""
            if len(vals) < 1:
                return float("nan"), float("nan"), float("nan")
            arr = np.asarray(vals)
            m = float(np.mean(arr))
            s = float(np.std(arr, ddof=min(1, len(arr)-1)))
            rsd = 100.0 * s / abs(m) if abs(m) > 1e-12 else float("nan")
            return m, s, rsd

        report.lod_mean, report.lod_std, report.lod_rsd_pct = _stats(lod_vals, "LOD")
        report.loq_mean, report.loq_std, report.loq_rsd_pct = _stats(loq_vals, "LOQ")
        report.sensitivity_mean, _, report.sensitivity_rsd_pct = _stats(sens_vals, "sensitivity")

        if r2_vals:
            report.r2_mean = float(np.mean(r2_vals))
            report.r2_std = float(np.std(r2_vals, ddof=min(1, len(r2_vals)-1)))
            report.r2_min = float(np.min(r2_vals))

        # Pooled LOD (pooled σ_blank / mean sensitivity)
        if pooled_variances and pooled_dofs and sum(pooled_dofs) > 0:
            pooled_sigma_blank = float(
                np.sqrt(sum(pooled_variances) / sum(pooled_dofs))
            )
            m_pooled = abs(report.sensitivity_mean) if np.isfinite(report.sensitivity_mean) else float("nan")
            if np.isfinite(m_pooled) and m_pooled > 1e-9:
                report.pooled_lod = 3.0 * pooled_sigma_blank / m_pooled
                report.pooled_loq = 10.0 * pooled_sigma_blank / m_pooled

        # Acceptance verdict
        report.failure_reasons = []
        can_evaluate = n >= 3  # need ≥3 sensors for a meaningful verdict

        if can_evaluate:
            if np.isfinite(report.lod_rsd_pct) and report.lod_rsd_pct > self._lod_rsd_limit:
                report.failure_reasons.append(
                    f"LOD RSD {report.lod_rsd_pct:.1f}% > {self._lod_rsd_limit:.1f}% limit"
                )
            if np.isfinite(report.loq_rsd_pct) and report.loq_rsd_pct > self._loq_rsd_limit:
                report.failure_reasons.append(
                    f"LOQ RSD {report.loq_rsd_pct:.1f}% > {self._loq_rsd_limit:.1f}% limit"
                )
            if np.isfinite(report.r2_min) and report.r2_min < self._min_r2:
                report.failure_reasons.append(
                    f"Minimum R² {report.r2_min:.4f} < {self._min_r2:.4f} limit"
                )
            if np.isfinite(report.sensitivity_rsd_pct) and report.sensitivity_rsd_pct > self._sens_rsd_limit:
                report.failure_reasons.append(
                    f"Sensitivity RSD {report.sensitivity_rsd_pct:.1f}% > {self._sens_rsd_limit:.1f}% limit"
                )
            report.batch_accepted = len(report.failure_reasons) == 0
        else:
            report.failure_reasons.append(
                f"Batch has only {n} sensor(s); ≥ 3 required for acceptance verdict."
            )

        # Build summary
        lines = [
            f"Batch Reproducibility Report ({n} sensors)",
            f"  Sensor IDs: {', '.join(sensor_ids)}",
            "",
        ]
        if np.isfinite(report.lod_mean):
            lines.append(
                f"  LOD: {report.lod_mean:.4f} ± {report.lod_std:.4f} ppm "
                f"(RSD {report.lod_rsd_pct:.1f}%)"
            )
        if np.isfinite(report.pooled_lod):
            lines.append(f"  Pooled LOD: {report.pooled_lod:.4f} ppm")
        if np.isfinite(report.loq_mean):
            lines.append(
                f"  LOQ: {report.loq_mean:.4f} ± {report.loq_std:.4f} ppm "
                f"(RSD {report.loq_rsd_pct:.1f}%)"
            )
        if np.isfinite(report.r2_mean):
            lines.append(
                f"  R²: mean={report.r2_mean:.4f}, min={report.r2_min:.4f}"
            )
        if np.isfinite(report.sensitivity_mean):
            lines.append(
                f"  Sensitivity: {report.sensitivity_mean:.4f} nm/ppm "
                f"(RSD {report.sensitivity_rsd_pct:.1f}%)"
            )
        if report.batch_accepted is True:
            lines.append("\n  BATCH ACCEPTED — all acceptance criteria met.")
        elif report.batch_accepted is False:
            lines.append("\n  BATCH FAILED:")
            for reason in report.failure_reasons:
                lines.append(f"    - {reason}")
        else:
            lines.append("\n  Acceptance verdict: insufficient sensors (need ≥ 3).")

        report.summary = "\n".join(lines)
        return report
