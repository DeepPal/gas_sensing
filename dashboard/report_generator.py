"""
dashboard.report_generator
===========================
One-click HTML analysis report for the SpectraAgent dashboard.

Generates a self-contained HTML file (inline CSS + base64 figures) from a
:class:`~dashboard.project_store.ProjectStore` instance.  The report covers:

- Session metadata and provenance
- Calibration data table and curve figure
- Sensor performance metrics (R², LOD, LOQ, sensitivity, residual diagnostics)
- Auto-generated Methods section (journal boilerplate filled from session data)
- Prediction history (if any predictions have been recorded)

The report is completely standalone — no external assets required.

Public API
----------
- ``generate_html_report(proj)``          — returns HTML string
- ``render_report_download_button(proj)`` — Streamlit download widget
"""
from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0;
       background: #f7f8fa; color: #222; }
.page { max-width: 900px; margin: 0 auto; padding: 24px 32px 48px 32px; background: #fff; }
h1 { color: #1a3a5c; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }
h2 { color: #1a3a5c; margin-top: 32px; }
h3 { color: #374151; margin-top: 20px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th { background: #1a3a5c; color: #fff; padding: 8px 12px; text-align: left; }
td { padding: 6px 12px; border-bottom: 1px solid #e5e7eb; }
tr:nth-child(even) td { background: #f1f5f9; }
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 12px 0; }
.metric { background: #f1f5f9; border-radius: 8px; padding: 12px 20px;
          min-width: 130px; text-align: center; }
.metric .label { font-size: 0.78em; color: #6b7280; text-transform: uppercase;
                 letter-spacing: 0.05em; }
.metric .value { font-size: 1.5em; font-weight: 700; color: #1a3a5c; }
.pass { color: #16a34a; font-weight: 600; }
.fail { color: #dc2626; font-weight: 600; }
.warn { color: #ca8a04; font-weight: 600; }
.methods { background: #f0f9ff; border-left: 4px solid #3b82f6;
           padding: 12px 20px; margin: 16px 0; border-radius: 0 6px 6px 0; }
.footer { margin-top: 48px; padding-top: 12px; border-top: 1px solid #e5e7eb;
          font-size: 0.8em; color: #9ca3af; text-align: center; }
img { max-width: 100%; border-radius: 6px; margin: 8px 0; }
pre { background: #f1f5f9; padding: 12px; border-radius: 6px;
      font-size: 0.85em; overflow-x: auto; white-space: pre-wrap; }
"""


# ---------------------------------------------------------------------------
# Methods section template
# ---------------------------------------------------------------------------

def _build_methods_section(proj_data: dict) -> str:
    """Auto-generate a journal-style Methods paragraph from session metadata."""
    gas = proj_data.get("gas_name", "the target analyte")
    model_type = proj_data.get("model_type", "Gaussian Process Regression (GPR)")
    n_cal = proj_data.get("n_cal_points", "N")
    r2 = proj_data.get("r_squared")
    lod = proj_data.get("lod_ppm")
    loq = proj_data.get("loq_ppm")
    sensitivity = proj_data.get("sensitivity")
    session_id = proj_data.get("session_id", "")
    date_str = session_id[:8] if len(session_id) >= 8 else datetime.now().strftime("%Y%m%d")
    try:
        date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%d %B %Y")
    except ValueError:
        date_fmt = date_str

    prep_cfg = proj_data.get("preprocessing_config", {})
    baseline = prep_cfg.get("baseline_method", "ALS (Asymmetric Least Squares)")
    norm = prep_cfg.get("norm_method", "peak normalisation")

    r2_str = f"R² = {r2:.4f}" if r2 is not None else "R² not computed"
    lod_str = f"LOD = {lod:.3g} ppm" if lod else "LOD not computed"
    loq_str = f"LOQ = {loq:.3g} ppm" if loq else ""
    sens_str = f"sensitivity = {sensitivity:.4g} nm ppm⁻¹" if sensitivity else ""

    lines = [
        f"Calibration experiments for {gas} detection were performed on {date_fmt}. ",
        f"A series of {n_cal} calibration standards spanning the working concentration range "
        f"were prepared in clean carrier gas. ",
        f"Spectral data were preprocessed using {baseline} baseline correction "
        f"followed by {norm}. ",
        f"The peak wavelength shift (Δλ, nm) relative to the reference spectrum was extracted "
        f"as the analytical signal. ",
        f"A {model_type} model was trained on the (concentration, Δλ) data pairs. ",
        f"Model performance was evaluated using the coefficient of determination ({r2_str}), "
        f"the limit of detection ({lod_str})",
    ]
    if loq_str:
        lines.append(f", the limit of quantification ({loq_str})")
    if sens_str:
        lines.append(f", and the analytical sensitivity ({sens_str})")
    lines.append(
        ". Residual diagnostics (Durbin-Watson autocorrelation test, Shapiro-Wilk normality test, "
        "and Breusch-Pagan homoscedasticity test) were performed according to ICH Q2(R1) guidelines. "
    )
    # Environmental conditions
    temp = proj_data.get("temperature_c")
    hum = proj_data.get("humidity_pct")
    if temp is not None or hum is not None:
        env_parts = []
        if temp is not None:
            env_parts.append(f"temperature {temp:.1f} °C")
        if hum is not None:
            env_parts.append(f"relative humidity {hum:.0f} %")
        lines.append(f"Measurements were performed at {' and '.join(env_parts)}. ")
    hw = proj_data.get("hw_serial", "")
    chip = proj_data.get("chip_serial", "")
    if hw or chip:
        hw_str = f"spectrometer S/N {hw}" if hw else ""
        chip_str = f"sensor chip S/N {chip}" if chip else ""
        lines.append(
            "Hardware: " + ", ".join(x for x in [hw_str, chip_str] if x) + ". "
        )
    return "".join(lines)


# ---------------------------------------------------------------------------
# Figure rendering (calibration curve)
# ---------------------------------------------------------------------------

def _calibration_curve_base64(
    concentrations: np.ndarray,
    responses: np.ndarray,
    gas_name: str,
    r2: float | None,
) -> str | None:
    """Render a calibration curve as a base64-encoded PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(concentrations, responses, color="#1a3a5c", zorder=5, s=50, label="Calibration data")

        # Fit line for display
        if len(concentrations) >= 2:
            coeffs = np.polyfit(concentrations, responses, 1)
            x_line = np.linspace(concentrations.min(), concentrations.max(), 200)
            ax.plot(x_line, np.polyval(coeffs, x_line), color="#3b82f6", linewidth=1.5,
                    label=f"Linear fit (R²={r2:.4f})" if r2 else "Linear fit")

        ax.set_xlabel("Concentration (ppm)", fontsize=11)
        ax.set_ylabel("Δλ / Response (nm)", fontsize=11)
        ax.set_title(f"Calibration Curve — {gas_name}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as exc:
        log.warning("Could not render calibration curve: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Software provenance helpers
# ---------------------------------------------------------------------------

def _get_git_hash() -> str:
    """Return the current git commit hash (short form), or 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=str(_REPO_ROOT), timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_library_versions() -> dict[str, str]:
    """Return versions of key scientific libraries."""
    import sys
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in ("numpy", "scipy", "sklearn", "streamlit", "joblib"):
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg if pkg != "sklearn" else "scikit-learn")
        except Exception:
            versions[pkg] = "?"
    return versions


# ---------------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------------

def generate_html_report(proj: Any) -> str:
    """Generate a self-contained HTML analysis report from a ProjectStore.

    Parameters
    ----------
    proj : ProjectStore
        The current project state.

    Returns
    -------
    str
        Complete HTML document.
    """
    perf = proj.performance or {}
    r2 = perf.get("r_squared")
    lod = perf.get("lod_ppm")
    loq = perf.get("loq_ppm")
    sensitivity = perf.get("sensitivity")
    rdiag = perf.get("residual_diagnostics") or {}

    git_hash = _get_git_hash()
    lib_versions = _get_library_versions()

    # Pull environmental/hardware metadata from the project's preprocessing_config
    # or from session metadata if stored there
    _meta = getattr(proj, "_session_meta", {}) or {}
    temperature_c = perf.get("temperature_c") or _meta.get("temperature_c")
    humidity_pct = perf.get("humidity_pct") or _meta.get("humidity_pct")
    hw_serial = perf.get("hw_serial") or _meta.get("hw_serial", "")
    chip_serial = perf.get("chip_serial") or _meta.get("chip_serial", "")

    proj_data = {
        "gas_name": proj.gas_name,
        "model_type": proj.model_type,
        "n_cal_points": len(proj.calibration_concentrations) if proj.calibration_concentrations is not None else 0,
        "r_squared": r2,
        "lod_ppm": lod,
        "loq_ppm": loq,
        "sensitivity": sensitivity,
        "session_id": proj.session_id,
        "preprocessing_config": proj.preprocessing_config,
        "temperature_c": temperature_c,
        "humidity_pct": humidity_pct,
        "hw_serial": hw_serial,
        "chip_serial": chip_serial,
        "git_hash": git_hash,
    }

    methods_text = _build_methods_section(proj_data)

    # Calibration curve figure
    cal_fig_b64 = None
    if proj.calibration_concentrations is not None and proj.calibration_responses is not None:
        cal_fig_b64 = _calibration_curve_base64(
            proj.calibration_concentrations, proj.calibration_responses, proj.gas_name, r2
        )

    # Residual diagnostics rows
    def _diag_row(label: str, value: Any, passed: bool | None) -> str:
        if passed is True:
            badge = '<span class="pass">PASS</span>'
        elif passed is False:
            badge = '<span class="fail">FAIL</span>'
        else:
            badge = "—"
        return f"<tr><td>{label}</td><td>{value}</td><td>{badge}</td></tr>"

    diag_rows = ""
    if rdiag:
        dw = rdiag.get("durbin_watson", {})
        sw = rdiag.get("shapiro_wilk", {})
        bp = rdiag.get("breusch_pagan", {})
        lof = rdiag.get("lack_of_fit", {})
        diag_rows = (
            _diag_row("Durbin-Watson (autocorrelation)",
                      f"DW={dw.get('statistic', '—'):.3f}" if isinstance(dw.get('statistic'), float) else "—",
                      dw.get("pass"))
            + _diag_row("Shapiro-Wilk (normality)",
                        f"W={sw.get('statistic', '—'):.4f}, p={sw.get('p_value', '—'):.4f}"
                        if isinstance(sw.get('statistic'), float) else "—",
                        sw.get("pass"))
            + _diag_row("Breusch-Pagan (homoscedasticity)",
                        f"LM={bp.get('lm_statistic', '—'):.4f}, p={bp.get('p_value', '—'):.4f}"
                        if isinstance(bp.get('lm_statistic'), float) else "—",
                        bp.get("pass"))
            + _diag_row("Lack-of-Fit F-test",
                        f"F={lof.get('f_statistic', '—'):.4f}, p={lof.get('p_value', '—'):.4f}"
                        if isinstance(lof.get('f_statistic'), float) else "—",
                        lof.get("pass"))
        )

    # Calibration table rows
    cal_table_rows = ""
    if proj.calibration_concentrations is not None and proj.calibration_responses is not None:
        for c, r in zip(proj.calibration_concentrations, proj.calibration_responses):
            cal_table_rows += f"<tr><td>{c:.4g}</td><td>{r:.6g}</td></tr>"

    # Prediction history rows
    pred_rows = ""
    for p in proj.predictions:
        q = p.get("quality", "ok")
        q_class = "pass" if q == "OK" else ("fail" if "LOD" in q else "warn")
        pred_rows += (
            f"<tr><td>{p.get('timestamp', '')[:19]}</td>"
            f"<td>{p.get('concentration_ppm', '—'):.4g}</td>"
            f"<td>{p.get('ci_lower_ppm', '—'):.4g}</td>"
            f"<td>{p.get('ci_upper_ppm', '—'):.4g}</td>"
            f"<td><span class='{q_class}'>{q}</span></td></tr>"
        )

    # Metric blocks
    def _metric(label: str, value: str) -> str:
        return (
            f"<div class='metric'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"</div>"
        )

    metrics_html = (
        _metric("Analyte", proj.gas_name)
        + _metric("Model", proj.model_type)
        + _metric("R²", f"{r2:.4f}" if r2 is not None else "—")
        + _metric("LOD", f"{lod:.3g} ppm" if lod else "—")
        + _metric("LOQ", f"{loq:.3g} ppm" if loq else "—")
        + _metric("Sensitivity", f"{sensitivity:.4g} nm/ppm" if sensitivity else "—")
        + _metric("Cal. points", str(proj_data["n_cal_points"]))
        + _metric("Predictions", str(len(proj.predictions)))
    )

    # Figure section
    cal_fig_html = (
        f"<img src='data:image/png;base64,{cal_fig_b64}' alt='Calibration curve' />"
        if cal_fig_b64 else "<p><em>Calibration curve not available (matplotlib required).</em></p>"
    )

    overall_pass = rdiag.get("overall_pass")
    diag_badge = (
        "<span class='pass'>All diagnostics PASS</span>"
        if overall_pass is True else
        ("<span class='fail'>One or more diagnostics FAIL — review before publication</span>"
         if overall_pass is False else "<em>Residual diagnostics not run</em>")
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SpectraAgent Analysis Report — {proj.gas_name} — {proj.session_id}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">

<h1>SpectraAgent Analysis Report</h1>
<p>
  <strong>Analyte:</strong> {proj.gas_name} &nbsp;|&nbsp;
  <strong>Session:</strong> {proj.session_id} &nbsp;|&nbsp;
  <strong>Generated:</strong> {now_str}
</p>
{f"<p><strong>Notes:</strong> {proj.notes}</p>" if proj.notes else ""}

<h2>1. Performance Summary</h2>
<div class="metric-row">{metrics_html}</div>

<h2>2. Calibration Curve</h2>
{cal_fig_html}

{"<h2>3. Calibration Data</h2><table><thead><tr><th>Concentration (ppm)</th><th>Response (Δλ, nm)</th></tr></thead><tbody>" + cal_table_rows + "</tbody></table>" if cal_table_rows else ""}

<h2>4. Residual Diagnostics</h2>
<p>{diag_badge}</p>
{"<table><thead><tr><th>Test</th><th>Statistic</th><th>Result</th></tr></thead><tbody>" + diag_rows + "</tbody></table>" if diag_rows else "<p><em>No residual diagnostics data available.</em></p>"}

<h2>5. Methods Section</h2>
<div class="methods">
<p>{methods_text}</p>
</div>
<p><em>Copy the text above into your manuscript's Materials and Methods section.
Verify instrument-specific details (integration time, temperature, humidity) before submission.</em></p>

{"<h2>6. Prediction History</h2><table><thead><tr><th>Timestamp</th><th>Concentration (ppm)</th><th>CI Lower</th><th>CI Upper</th><th>Quality</th></tr></thead><tbody>" + pred_rows + "</tbody></table>" if pred_rows else ""}

<h2>7. Software Provenance</h2>
<table>
  <thead><tr><th>Item</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Git commit hash</td><td><code>{git_hash}</code></td></tr>
    <tr><td>Python</td><td>{lib_versions.get('python', '?')}</td></tr>
    <tr><td>NumPy</td><td>{lib_versions.get('numpy', '?')}</td></tr>
    <tr><td>SciPy</td><td>{lib_versions.get('scipy', '?')}</td></tr>
    <tr><td>scikit-learn</td><td>{lib_versions.get('sklearn', '?')}</td></tr>
    <tr><td>Streamlit</td><td>{lib_versions.get('streamlit', '?')}</td></tr>
    <tr><td>Generated</td><td>{now_str}</td></tr>
    {"<tr><td>Spectrometer S/N</td><td>" + hw_serial + "</td></tr>" if hw_serial else ""}
    {"<tr><td>Chip S/N</td><td>" + chip_serial + "</td></tr>" if chip_serial else ""}
  </tbody>
</table>
<p><em>This table satisfies the software traceability requirement in Handbook §7.
Archive this report alongside raw data for long-term reproducibility.</em></p>

<div class="footer">
  Generated by SpectraAgent — Chulalongkorn University LSPR Research Platform &nbsp;|&nbsp;
  Session {proj.session_id} &nbsp;|&nbsp; git:{git_hash} &nbsp;|&nbsp; {now_str}
</div>

</div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# Streamlit widget
# ---------------------------------------------------------------------------

def render_report_download_button(proj: Any) -> None:
    """Render a "Generate & Download Report" button in the Streamlit UI.

    Parameters
    ----------
    proj : ProjectStore
        The current project.
    """
    import streamlit as st

    with st.expander("📄 Generate Analysis Report", expanded=False):
        st.markdown(
            "Export a self-contained HTML report with calibration curve, "
            "performance metrics, residual diagnostics, and an auto-generated "
            "Methods section ready for your manuscript."
        )

        if not proj.has_calibration:
            st.info("Complete calibration first (Step 3) to include calibration data in the report.")

        if st.button("Generate report", key="report_gen_btn", type="primary"):
            with st.spinner("Generating report…"):
                try:
                    html = generate_html_report(proj)
                    fname = f"spectraagent_report_{proj.gas_name}_{proj.session_id}.html"
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=html.encode("utf-8"),
                        file_name=fname,
                        mime="text/html",
                        key="report_dl_btn",
                    )
                    st.success(f"Report ready: {fname}")

                    # Also save to session directory
                    try:
                        report_path = proj.session_dir / fname
                        proj.session_dir.mkdir(parents=True, exist_ok=True)
                        report_path.write_text(html, encoding="utf-8")
                        st.caption(f"Also saved to: {report_path}")
                    except Exception as save_exc:
                        log.warning("Could not auto-save report to session dir: %s", save_exc)

                except Exception as exc:
                    st.error(f"Report generation failed: {exc}")
