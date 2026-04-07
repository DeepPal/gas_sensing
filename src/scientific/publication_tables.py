"""
src.scientific.publication_tables
===================================
Auto-generate publication-ready sensor performance tables (Table 1 and
Supplementary Table S1) from analysis results.

These tables follow the standard format required by:

  - ACS Sensors / Analytical Chemistry
  - Sensors & Actuators B (Elsevier)
  - Analytica Chimica Acta / Talanta
  - IEEE Sensors Journal

Table 1: Sensor Performance Summary
--------------------------------------
One row per analyte.  Columns:

  Analyte | Sensitivity (nm/ppm) | R² | LOD (ppm) [95% CI] |
  LOQ (ppm) | Linear Range (ppm) | Reproducibility (RSD%)

Supplementary Table S1: Batch Reproducibility
-----------------------------------------------
One row per analyte per session/batch.  Columns:

  Analyte | Session | n | Mean Signal | SD | RSD% | Passes ICH (<20%)

Supplementary Table S2: Residual Diagnostics Checklist
--------------------------------------------------------
One row per analyte.  Columns:

  Analyte | DW | SW p | BP p | Normality | Homoscedasticity | Autocorr.

Public API
----------
- ``SensorPerformanceRow``    — dataclass for one analyte row
- ``BatchReproducibilityRow`` — dataclass for one batch row
- ``build_table1``            — build Table 1 from list of summary dicts
- ``build_supplementary_s1``  — build Supp. S1 from batch data
- ``build_supplementary_s2``  — build Supp. S2 from diagnostics data
- ``format_table1_text``      — ASCII text table for terminal / lab notebook
- ``format_table1_latex``     — LaTeX tabular for journal submission
- ``format_table1_csv``       — CSV export for Excel
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SensorPerformanceRow:
    """One row in Table 1: Sensor Performance Summary.

    Attributes
    ----------
    analyte : str
        Gas analyte name.
    sensitivity : float
        OLS slope (nm/ppm or counts/ppm).
    sensitivity_se : float
        Standard error of sensitivity (1σ).
    r_squared : float
        Coefficient of determination.
    lod_ppm : float | None
        Limit of Detection (IUPAC 3σ/S) in ppm.
    lod_ci_lower, lod_ci_upper : float | None
        Bootstrap 95% CI on LOD.
    loq_ppm : float | None
        Limit of Quantification (10σ/S) in ppm.
    linear_range_low, linear_range_high : float | None
        Linear range bounds (LOQ to LOL) in ppm.
    reproducibility_rsd_pct : float | None
        Intra-batch signal RSD in %.
    n_cal_points : int
        Number of calibration points used.
    lod_method : str
        LOD estimation method tag.
    allan_tau_opt : float | None
        Allan deviation optimal averaging time (s), if available.
    allan_sigma_min : float | None
        Allan deviation noise floor (σ_min), if available.
    residual_pass : bool | None
        True = all residual diagnostics passed.
    """

    analyte: str
    sensitivity: float
    sensitivity_se: float
    r_squared: float
    lod_ppm: float | None
    lod_ci_lower: float | None
    lod_ci_upper: float | None
    loq_ppm: float | None
    linear_range_low: float | None
    linear_range_high: float | None
    reproducibility_rsd_pct: float | None
    n_cal_points: int
    lod_method: str = "ICH Q2(R1) 3σ/S"
    allan_tau_opt: float | None = None
    allan_sigma_min: float | None = None
    residual_pass: bool | None = None
    lod_ci_from_blank: bool = False
    """True when LOD bootstrap CI used measured σ_blank (not OLS residuals)."""
    missing_reasons: dict[str, str] = field(default_factory=dict)
    """Maps field name → reason string for any None field. Used in table footnotes.
    Example: {"lod_ppm": "blank_measurements_not_provided",
              "linear_range_high": "mandel_low_power_n4"}
    Reviewers ask *why* a field is missing — this provides the audit trail.
    """

    def _fmt_lod(self) -> str:
        if self.lod_ppm is None:
            return "N/A"
        base = f"{self.lod_ppm:.3g}"
        if self.lod_ci_lower is not None and self.lod_ci_upper is not None:
            return f"{base} [{self.lod_ci_lower:.3g}–{self.lod_ci_upper:.3g}]"
        return base

    def _fmt_range(self) -> str:
        lo = self.linear_range_low
        hi = self.linear_range_high
        if lo is None and hi is None:
            return "N/A"
        if lo is None:
            return f"–{hi:.3g}"
        if hi is None:
            return f"{lo:.3g}–"
        return f"{lo:.3g}–{hi:.3g}"

    def _fmt_rsd(self) -> str:
        if self.reproducibility_rsd_pct is None:
            return "N/A"
        rsd = self.reproducibility_rsd_pct
        flag = "" if rsd <= 20.0 else " *"
        return f"{rsd:.1f}%{flag}"

    def _fmt_sensitivity(self) -> str:
        s = self.sensitivity
        se = self.sensitivity_se
        if se > 0:
            return f"{s:.4g} ± {se:.2g}"
        return f"{s:.4g}"


@dataclass
class BatchReproducibilityRow:
    """One row in Supplementary Table S1: Batch Reproducibility.

    Attributes
    ----------
    analyte : str
    session_id : str
        Session label (e.g. "20260401_Ethanol_1").
    n_frames : int
        Number of measurement frames in this batch.
    mean_signal : float
        Mean sensor signal (Δλ or peak intensity).
    std_signal : float
        Standard deviation of signal.
    rsd_pct : float
        RSD = (std / |mean|) × 100.
    passes_ich : bool
        True if RSD < 20% (ICH Q2(R1) intra-batch criterion).
    concentration_ppm : float | None
        Known analyte concentration if available.
    """

    analyte: str
    session_id: str
    n_frames: int
    mean_signal: float
    std_signal: float
    rsd_pct: float
    passes_ich: bool
    concentration_ppm: float | None = None


@dataclass
class DiagnosticsRow:
    """One row in Supplementary Table S2: Residual Diagnostics."""

    analyte: str
    durbin_watson: float
    dw_pass: bool
    shapiro_wilk_p: float
    sw_pass: bool
    breusch_pagan_p: float
    bp_pass: bool
    overall_pass: bool
    n: int


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_table1(
    performance_summaries: list[dict[str, Any]],
    batch_rsds: dict[str, float] | None = None,
) -> list[SensorPerformanceRow]:
    """Build Table 1 rows from a list of ``sensor_performance_summary`` dicts.

    Parameters
    ----------
    performance_summaries:
        List of dicts returned by
        :func:`src.scientific.lod.sensor_performance_summary`.
        One dict per analyte.
    batch_rsds:
        Optional mapping {analyte_name: rsd_pct} for the reproducibility
        column.  If not provided, the column is left as None.

    Returns
    -------
    list[SensorPerformanceRow]
        One row per analyte, sorted alphabetically.
    """
    rows: list[SensorPerformanceRow] = []
    for s in performance_summaries:
        gas = str(s.get("gas", "unknown"))
        lod = s.get("lod_ppm")
        loq = s.get("loq_ppm")
        lol = s.get("lol_ppm")
        linear_lo = loq
        linear_hi = lol

        adev = s.get("allan_deviation") or {}
        rdiag = s.get("residual_diagnostics") or {}

        rsd = None
        if batch_rsds and gas in batch_rsds:
            rsd = float(batch_rsds[gas])

        # Build missing_reasons audit trail — reviewers ask WHY a field is None
        missing: dict[str, str] = {}
        if lod is None:
            missing["lod_ppm"] = str(s.get("lod_method", "not_computed"))
        if loq is None:
            missing["loq_ppm"] = "not_computed"
        if linear_hi is None:
            if s.get("mandel_low_power_warning"):
                missing["linear_range_high"] = (
                    f"mandel_low_power (n={s.get('n_calibration_points', '?')} ≤ 5)"
                )
            else:
                missing["linear_range_high"] = "mandel_test_not_run_or_failed"
        if rsd is None:
            missing["reproducibility_rsd_pct"] = "batch_rsd_not_provided"
        if not s.get("lob_from_blank_measurements", False):
            missing["lob_ppm"] = "estimated_from_ols_residuals_not_blank_measurements"

        rows.append(SensorPerformanceRow(
            analyte=gas,
            sensitivity=float(s.get("sensitivity", 0)),
            sensitivity_se=float(s.get("sensitivity_se", 0)),
            r_squared=float(s.get("r_squared", 0)),
            lod_ppm=float(lod) if lod is not None else None,
            lod_ci_lower=float(s["lod_ppm_ci_lower"]) if s.get("lod_ppm_ci_lower") else None,
            lod_ci_upper=float(s["lod_ppm_ci_upper"]) if s.get("lod_ppm_ci_upper") else None,
            loq_ppm=float(loq) if loq is not None else None,
            linear_range_low=float(linear_lo) if linear_lo is not None else None,
            linear_range_high=float(linear_hi) if linear_hi is not None else None,
            reproducibility_rsd_pct=rsd,
            n_cal_points=int(s.get("n_calibration_points", 0)),
            lod_method=str(s.get("lod_method", "ICH Q2(R1) 3σ/S")),
            allan_tau_opt=float(adev["tau_opt_s"]) if adev.get("tau_opt_s") else None,
            allan_sigma_min=float(adev["sigma_min"]) if adev.get("sigma_min") else None,
            residual_pass=bool(rdiag["overall_pass"]) if "overall_pass" in rdiag else None,
            lod_ci_from_blank=bool(s.get("lod_ci_from_blank", False)),
            missing_reasons=missing,
        ))

    rows.sort(key=lambda r: r.analyte.lower())
    return rows


def build_supplementary_s1(
    batch_data: list[dict[str, Any]],
) -> list[BatchReproducibilityRow]:
    """Build Supplementary S1 rows from batch measurement data.

    Parameters
    ----------
    batch_data:
        List of dicts, each with keys:
        ``analyte``, ``session_id``, ``signals`` (array-like),
        and optionally ``concentration_ppm``.

    Returns
    -------
    list[BatchReproducibilityRow]
    """
    rows: list[BatchReproducibilityRow] = []
    for entry in batch_data:
        gas = str(entry.get("analyte", "unknown"))
        session = str(entry.get("session_id", ""))
        signals = np.asarray(entry["signals"], dtype=float).ravel()
        signals = signals[np.isfinite(signals)]
        if len(signals) == 0:
            continue
        mean_s = float(np.mean(signals))
        std_s = float(np.std(signals, ddof=1)) if len(signals) > 1 else 0.0
        rsd = (std_s / abs(mean_s) * 100.0) if abs(mean_s) > 1e-12 else 0.0
        rows.append(BatchReproducibilityRow(
            analyte=gas,
            session_id=session,
            n_frames=len(signals),
            mean_signal=round(mean_s, 6),
            std_signal=round(std_s, 6),
            rsd_pct=round(rsd, 2),
            passes_ich=bool(rsd <= 20.0),
            concentration_ppm=float(entry["concentration_ppm"])
            if entry.get("concentration_ppm") is not None else None,
        ))
    return rows


def build_supplementary_s2(
    performance_summaries: list[dict[str, Any]],
) -> list[DiagnosticsRow]:
    """Build Supplementary S2 (residual diagnostics checklist) rows.

    Parameters
    ----------
    performance_summaries:
        List of dicts from :func:`src.scientific.lod.sensor_performance_summary`.

    Returns
    -------
    list[DiagnosticsRow]
    """
    rows: list[DiagnosticsRow] = []
    for s in performance_summaries:
        gas = str(s.get("gas", "unknown"))
        rdiag = s.get("residual_diagnostics") or {}
        if not rdiag:
            rows.append(DiagnosticsRow(
                analyte=gas, durbin_watson=float("nan"), dw_pass=False,
                shapiro_wilk_p=float("nan"), sw_pass=False,
                breusch_pagan_p=float("nan"), bp_pass=False,
                overall_pass=False, n=int(s.get("n_calibration_points", 0)),
            ))
            continue
        rows.append(DiagnosticsRow(
            analyte=gas,
            durbin_watson=float(rdiag.get("durbin_watson", float("nan"))),
            dw_pass=bool(rdiag.get("dw_pass", False)),
            shapiro_wilk_p=float(rdiag.get("shapiro_wilk_p", float("nan"))),
            sw_pass=bool(rdiag.get("sw_pass", False)),
            breusch_pagan_p=float(rdiag.get("breusch_pagan_p", float("nan"))),
            bp_pass=bool(rdiag.get("bp_pass", False)),
            overall_pass=bool(rdiag.get("overall_pass", False)),
            n=int(rdiag.get("n", s.get("n_calibration_points", 0))),
        ))
    return rows


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "Analyte": 14,
    "Sensitivity": 18,
    "R²": 7,
    "LOD (ppm)": 22,
    "LOQ (ppm)": 10,
    "Lin. Range": 16,
    "RSD%": 8,
    "n": 4,
}


def format_table1_text(rows: list[SensorPerformanceRow], title: str = "") -> str:
    """Format Table 1 as an ASCII table for terminal / lab notebook.

    Parameters
    ----------
    rows : list[SensorPerformanceRow]
    title : str, optional

    Returns
    -------
    str
    """
    cols = ["Analyte", "Sensitivity (nm/ppm)", "R²", "LOD [95%CI] (ppm)",
            "LOQ (ppm)", "Lin. Range (ppm)", "RSD%", "n"]
    widths = [16, 22, 7, 28, 11, 18, 8, 4]

    def _cell(s: str, w: int) -> str:
        return s[:w].ljust(w)

    header = "  ".join(_cell(c, w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(header))
    lines += [header, sep]

    for r in rows:
        lod_cell = r._fmt_lod()
        if r.lod_ci_from_blank:
            lod_cell += " ‡"
        data = [
            r.analyte,
            r._fmt_sensitivity(),
            f"{r.r_squared:.4f}",
            lod_cell,
            f"{r.loq_ppm:.3g}" if r.loq_ppm is not None else "N/A",
            r._fmt_range(),
            r._fmt_rsd(),
            str(r.n_cal_points),
        ]
        lines.append("  ".join(_cell(str(d), w) for d, w in zip(data, widths)))

    lines.append("")
    lines.append("  LOD = 3σ/S (IUPAC); LOQ = 10σ/S; CI = bootstrap 95% (n=1000)")
    lines.append("  * RSD > 20% — fails ICH Q2(R1) intra-batch reproducibility criterion")
    if any(r.residual_pass is False for r in rows):
        lines.append("  † Residual diagnostics failed — see Supplementary Table S2")
    if any(r.lod_ci_from_blank for r in rows):
        lines.append("  ‡ LOD CI: σ_blank from dedicated blank measurements (IUPAC 2012)")
    else:
        lines.append("  ! LOD CI: σ_blank estimated from OLS residuals — provide blank")
        lines.append("    measurements for regulatory submissions (IUPAC 2012 §3.3)")
    # Missing-field audit trail: reviewers ask *why* a field is None
    missing_notes: list[str] = []
    for r in rows:
        for field_name, reason in (r.missing_reasons or {}).items():
            missing_notes.append(
                f"  [{r.analyte}] {field_name}: {reason.replace('_', ' ')}"
            )
    if missing_notes:
        lines.append("")
        lines.append("  Missing fields (audit trail):")
        lines.extend(missing_notes)
    return "\n".join(lines)


def format_table1_latex(
    rows: list[SensorPerformanceRow],
    caption: str = "Sensor performance summary.",
    label: str = "tab:sensor_performance",
    units: str = "nm/ppm",
) -> str:
    """Format Table 1 as a LaTeX ``tabular`` environment.

    Parameters
    ----------
    rows, caption, label, units

    Returns
    -------
    str — ready to paste into a LaTeX document or SI.
    """
    def _esc(s: str) -> str:
        """Escape common LaTeX special chars."""
        return (s.replace("%", r"\%")
                 .replace("&", r"\&")
                 .replace("_", r"\_")
                 .replace("±", r"$\pm$")
                 .replace("–", "--")
                 .replace("[", "{[}")
                 .replace("]", "{]}"))

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{" + _esc(caption) + "}",
        r"  \label{" + label + "}",
        r"  \begin{tabular}{l r r l r l r}",
        r"    \toprule",
        (r"    Analyte & Sensitivity & $R^2$ & LOD [95\,\%\,CI] & LOQ"
         r" & Lin. range & RSD\% \\"),
        r"    & (" + _esc(units) + r") & & (ppm) & (ppm) & (ppm) & \\",
        r"    \midrule",
    ]

    for r in rows:
        sens_str = f"${r.sensitivity:.4g} \\pm {r.sensitivity_se:.2g}$"
        r2_str = f"{r.r_squared:.4f}"
        lod_str = _esc(r._fmt_lod())
        loq_str = f"{r.loq_ppm:.3g}" if r.loq_ppm is not None else "N/A"
        range_str = _esc(r._fmt_range())
        rsd_str = _esc(r._fmt_rsd())
        analyte = _esc(r.analyte)
        lines.append(
            f"    {analyte} & {sens_str} & {r2_str} & {lod_str}"
            f" & {loq_str} & {range_str} & {rsd_str} \\\\"
        )

    # Collect all missing-reason footnotes across all rows
    footnote_items: list[str] = []
    for r in rows:
        for field_name, reason in (r.missing_reasons or {}).items():
            footnote_items.append(
                f"\\item {_esc(r.analyte)}: {_esc(field_name)} not reported — "
                f"{_esc(reason.replace('_', ' '))}."
            )
    lob_from_blank_note = all(r.lod_ci_from_blank for r in rows)

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}",
        r"    \small",
        r"    \item LOD = $3\sigma/S$ (IUPAC); LOQ = $10\sigma/S$;"
        r" 95\,\% CI from bootstrap ($n = 1000$).",
        "    \\item " + (
            r"LOD CI: $\sigma_{\text{blank}}$ from dedicated blank measurements (IUPAC 2012)."
            if lob_from_blank_note
            else r"LOD CI: $\sigma_{\text{blank}}$ estimated from OLS residuals"
                 r" — provide blank measurements for regulatory submissions."
        ),
        r"    \item Bonferroni-corrected residual diagnostics: $\alpha = 0.05/3 \approx 0.0167$"
        r" per test (DW, SW, BP).",
        r"    \item * RSD $> 20\%$ fails ICH Q2(R1) intra-batch reproducibility.",
    ] + footnote_items + [
        r"  \end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def format_table1_csv(rows: list[SensorPerformanceRow]) -> str:
    """Format Table 1 as CSV (for Excel / supplementary data file).

    Returns
    -------
    str — UTF-8 CSV text.
    """
    header = (
        "Analyte,Sensitivity (nm/ppm),Sensitivity SE,"
        "R2,LOD (ppm),LOD CI lower (ppm),LOD CI upper (ppm),"
        "LOQ (ppm),Linear range low (ppm),Linear range high (ppm),"
        "Reproducibility RSD (%),n cal points,LOD method,"
        "Allan tau_opt (s),Allan sigma_min,Residual diagnostics pass,"
        "LOD CI from blank measurements,Missing fields (audit trail)"
    )
    csv_lines = [header]
    for r in rows:
        def _v(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, float) and not np.isfinite(x):
                return ""
            return str(x)

        missing_summary = "; ".join(
            f"{k}={v}" for k, v in (r.missing_reasons or {}).items()
        )
        csv_lines.append(",".join([
            r.analyte,
            _v(r.sensitivity),
            _v(r.sensitivity_se),
            _v(r.r_squared),
            _v(r.lod_ppm),
            _v(r.lod_ci_lower),
            _v(r.lod_ci_upper),
            _v(r.loq_ppm),
            _v(r.linear_range_low),
            _v(r.linear_range_high),
            _v(r.reproducibility_rsd_pct),
            _v(r.n_cal_points),
            r.lod_method,
            _v(r.allan_tau_opt),
            _v(r.allan_sigma_min),
            _v(r.residual_pass),
            "yes" if r.lod_ci_from_blank else "no (OLS residuals)",
            f'"{missing_summary}"' if missing_summary else "",
        ]))
    return "\n".join(csv_lines)


def format_supplementary_s1_text(rows: list[BatchReproducibilityRow]) -> str:
    """Format Supplementary S1 as ASCII table."""
    cols = ["Analyte", "Session", "n", "Mean signal", "SD", "RSD%", "ICH"]
    widths = [14, 26, 5, 12, 12, 8, 6]

    def _cell(s: str, w: int) -> str:
        return s[:w].ljust(w)

    header = "  ".join(_cell(c, w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    lines = [
        "Supplementary Table S1: Batch Reproducibility",
        "=" * len(header),
        header, sep,
    ]
    for r in rows:
        data = [
            r.analyte,
            r.session_id,
            str(r.n_frames),
            f"{r.mean_signal:.5g}",
            f"{r.std_signal:.5g}",
            f"{r.rsd_pct:.1f}%",
            "PASS" if r.passes_ich else "FAIL",
        ]
        lines.append("  ".join(_cell(str(d), w) for d, w in zip(data, widths)))

    lines.append("")
    lines.append("  ICH Q2(R1) batch criterion: RSD < 20%")
    return "\n".join(lines)


def format_supplementary_s2_text(rows: list[DiagnosticsRow]) -> str:
    """Format Supplementary S2 as ASCII table."""
    cols = ["Analyte", "n", "DW", "SW p", "BP p", "Normality", "Homoscedast.", "AutoCorr.", "Overall"]
    widths = [14, 4, 7, 8, 8, 11, 13, 11, 8]

    def _cell(s: str, w: int) -> str:
        return s[:w].ljust(w)

    def _ok(b: bool) -> str:
        return "PASS" if b else "FAIL"

    header = "  ".join(_cell(c, w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    lines = [
        "Supplementary Table S2: Residual Diagnostics Checklist",
        "=" * len(header),
        header, sep,
    ]
    for r in rows:
        dw_str = f"{r.durbin_watson:.3f}" if np.isfinite(r.durbin_watson) else "N/A"
        sw_str = f"{r.shapiro_wilk_p:.4f}" if np.isfinite(r.shapiro_wilk_p) else "N/A"
        bp_str = f"{r.breusch_pagan_p:.4f}" if np.isfinite(r.breusch_pagan_p) else "N/A"
        data = [
            r.analyte,
            str(r.n),
            dw_str,
            sw_str,
            bp_str,
            _ok(r.sw_pass),
            _ok(r.bp_pass),
            _ok(r.dw_pass),
            _ok(r.overall_pass),
        ]
        lines.append("  ".join(_cell(str(d), w) for d, w in zip(data, widths)))

    lines.append("")
    lines.append("  DW = Durbin-Watson; SW = Shapiro-Wilk; BP = Breusch-Pagan")
    lines.append("  PASS threshold: DW in [1.5, 2.5]; SW/BP p >= 0.05")
    return "\n".join(lines)
