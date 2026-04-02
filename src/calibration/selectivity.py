"""
src.calibration.selectivity
============================
Cross-reactivity and selectivity analysis for LSPR gas sensors.

Background
----------
An LSPR sensor responds to any gas that adsorbs on the sensor surface.
For a sensor targeting Ethanol, the response to CO₂ or humidity constitutes
a cross-reactivity that degrades specificity.  ICH Q2(R1) §5.3 and IUPAC
2012 require selectivity characterisation for any claimed analytical method.

The selectivity coefficient K_AB (ISO 8655, modified for sensors) quantifies
what concentration of interferent B produces the same response as 1 ppm of
analyte A:

    K_AB = (response per ppm of B) / (response per ppm of A)
         = m_B / m_A

where m_A, m_B are the calibration slopes (nm/ppm) at low concentration.

A sensor with K_AB < 0.01 for all interferents has < 1% cross-reactivity.

Public API
----------
- ``SelectivityReport``         — dataclass of all selectivity statistics
- ``SelectivityAnalyzer``       — .analyze(analyte, interferents) → SelectivityReport
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import linregress


@dataclass
class SelectivityReport:
    """Cross-reactivity statistics for one analyte vs. a set of interferents.

    All coefficients are dimensionless ratios.  A coefficient K_AB < 0.01
    means < 1% cross-reactivity (the interferent at 1 ppm produces a response
    equivalent to 0.01 ppm of analyte).
    """

    analyte: str
    """Name of the target analyte."""

    analyte_sensitivity_nm_per_ppm: float = float("nan")
    """Calibration slope m_A (nm/ppm) for the target analyte at low concentration."""

    interferents: list[str] = field(default_factory=list)
    """Names of interferent gases / species tested."""

    cross_reactivity_coefficients: dict[str, float] = field(default_factory=dict)
    """K_AB = m_B / m_A for each interferent B.  Values close to zero = high selectivity."""

    cross_reactivity_ppm_equivalent: dict[str, float] = field(default_factory=dict)
    """Interferent concentration (ppm) that produces the same response as 1 ppm analyte."""

    selectivity_flags: dict[str, str] = field(default_factory=dict)
    """'excellent' (<1%), 'good' (1–5%), 'moderate' (5–20%), 'poor' (>20%) per interferent."""

    r2_analyte: float | None = None
    """Calibration R² for analyte (quality of sensitivity estimate)."""

    r2_interferents: dict[str, float] = field(default_factory=dict)
    """Calibration R² for each interferent."""

    n_analyte_points: int = 0
    """Number of analyte calibration points used."""

    n_interferent_points: dict[str, int] = field(default_factory=dict)
    """Number of calibration points per interferent."""

    summary: str = ""
    """Human-readable selectivity summary."""


class SelectivityAnalyzer:
    """Compute cross-reactivity coefficients from multi-gas calibration data.

    Usage
    -----
    ::

        analyzer = SelectivityAnalyzer(analyte="Ethanol")
        report = analyzer.analyze(
            analyte_concs   = np.array([0.5, 1.0, 2.0, 3.0]),
            analyte_shifts  = np.array([-3.2, -5.8, -9.1, -11.4]),
            interferent_data = {
                "CO2":      (co2_concs, co2_shifts),
                "Humidity": (hum_concs, hum_shifts),
            },
        )
        print(report.summary)
    """

    def __init__(self, analyte: str = "target") -> None:
        self._analyte = analyte

    def analyze(
        self,
        analyte_concs: np.ndarray,
        analyte_shifts: np.ndarray,
        interferent_data: dict[str, tuple[np.ndarray, np.ndarray]],
        n_low_fraction: float = 0.5,
    ) -> SelectivityReport:
        """Compute cross-reactivity from calibration data.

        Parameters
        ----------
        analyte_concs :
            Analyte concentrations (ppm), shape (n,).
        analyte_shifts :
            Measured wavelength shifts for analyte (nm), same length.
        interferent_data :
            Dict mapping interferent name → (concentrations, shifts).
        n_low_fraction :
            Fraction of lowest concentrations to use for slope estimation
            (default 0.5 = lower half of the calibration range).
            This corresponds to the Henry's-law linear regime.

        Returns
        -------
        SelectivityReport
        """
        report = SelectivityReport(analyte=self._analyte)

        # Analyte sensitivity: slope from low-concentration linear region
        a_concs = np.asarray(analyte_concs, dtype=float).ravel()
        a_shifts = np.asarray(analyte_shifts, dtype=float).ravel()
        n_a = len(a_concs)
        report.n_analyte_points = n_a

        if n_a < 2:
            warnings.warn(
                f"SelectivityAnalyzer: analyte has only {n_a} calibration points; "
                "at least 2 required for slope estimation.",
                UserWarning,
                stacklevel=2,
            )
            report.summary = "Insufficient analyte data for selectivity analysis."
            return report

        m_a, _, r_a, _, _ = linregress(a_concs, a_shifts)
        report.analyte_sensitivity_nm_per_ppm = float(m_a)
        report.r2_analyte = float(r_a ** 2)

        # Low-concentration subset for Henry's-law sensitivity
        n_low = max(2, int(np.ceil(n_a * n_low_fraction)))
        sorted_idx = np.argsort(a_concs)
        low_concs = a_concs[sorted_idx[:n_low]]
        low_shifts = a_shifts[sorted_idx[:n_low]]
        if np.ptp(low_concs) > 1e-9:
            m_a_low, *_ = linregress(low_concs, low_shifts)
            m_a = float(m_a_low)  # use low-conc slope as the definitive sensitivity

        if abs(m_a) < 1e-9:
            warnings.warn(
                "SelectivityAnalyzer: analyte sensitivity ~0; cross-reactivity coefficients undefined.",
                UserWarning,
                stacklevel=2,
            )
            report.summary = "Analyte sensitivity near zero; cannot compute selectivity."
            return report

        # Per-interferent cross-reactivity
        for name, (b_concs_raw, b_shifts_raw) in interferent_data.items():
            b_concs = np.asarray(b_concs_raw, dtype=float).ravel()
            b_shifts = np.asarray(b_shifts_raw, dtype=float).ravel()
            n_b = len(b_concs)
            report.interferents.append(name)
            report.n_interferent_points[name] = n_b

            if n_b < 2:
                report.cross_reactivity_coefficients[name] = float("nan")
                report.cross_reactivity_ppm_equivalent[name] = float("nan")
                report.selectivity_flags[name] = "insufficient_data"
                report.r2_interferents[name] = float("nan")
                continue

            m_b, _, r_b, _, _ = linregress(b_concs, b_shifts)
            report.r2_interferents[name] = float(r_b ** 2)

            k_ab = float(m_b / m_a)
            report.cross_reactivity_coefficients[name] = k_ab
            # ppm of interferent that produces the same response as 1 ppm analyte
            ppm_equiv = 1.0 / abs(k_ab) if abs(k_ab) > 1e-9 else float("inf")
            report.cross_reactivity_ppm_equivalent[name] = ppm_equiv

            abs_k = abs(k_ab)
            if abs_k < 0.01:
                flag = "excellent"
            elif abs_k < 0.05:
                flag = "good"
            elif abs_k < 0.20:
                flag = "moderate"
            else:
                flag = "poor"
            report.selectivity_flags[name] = flag

        # Build summary
        lines = [
            f"Selectivity report for {self._analyte}",
            f"  Analyte sensitivity: {report.analyte_sensitivity_nm_per_ppm:.4f} nm/ppm "
            f"(R²={report.r2_analyte:.4f}, n={n_a})",
            "",
            "  Cross-reactivity (K_AB = m_B / m_A):",
        ]
        for name in report.interferents:
            k = report.cross_reactivity_coefficients.get(name, float("nan"))
            flag = report.selectivity_flags.get(name, "?")
            ppm_eq = report.cross_reactivity_ppm_equivalent.get(name, float("nan"))
            lines.append(
                f"    {name:20s}  K_AB={k:+.4f}  [{flag}]"
                f"  (1 ppm analyte ≡ {ppm_eq:.1f} ppm interferent)"
            )

        report.summary = "\n".join(lines)
        return report

    @staticmethod
    def from_session_analyses(
        analyte: str,
        analyte_analysis: Any,
        interferent_analyses: dict[str, Any],
    ) -> SelectivityReport:
        """Convenience factory: build a SelectivityReport from SessionAnalysis objects.

        Parameters
        ----------
        analyte_analysis, interferent_analyses :
            ``SessionAnalysis`` instances (must have
            ``calibration_concentrations`` and ``calibration_shifts`` populated).
        """
        def _extract(sa: Any) -> tuple[np.ndarray, np.ndarray]:
            concs = np.asarray(sa.calibration_concentrations, dtype=float)
            shifts = np.asarray(sa.calibration_shifts, dtype=float)
            return concs, shifts

        a_concs, a_shifts = _extract(analyte_analysis)
        interferent_data = {
            name: _extract(sa)
            for name, sa in interferent_analyses.items()
        }
        return SelectivityAnalyzer(analyte).analyze(a_concs, a_shifts, interferent_data)
