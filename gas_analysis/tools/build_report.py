import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _md_h2(title: str) -> str:
    return f"\n\n## {title}\n\n"


def _md_h3(title: str) -> str:
    return f"\n\n### {title}\n\n"


def _md_kv_table(d: dict, keys: List[str], titles: Optional[List[str]] = None) -> str:
    if not d:
        return ""
    if not titles:
        titles = keys
    # Header
    lines = ["| Key | Value |", "|---|---|"]
    for k, t in zip(keys, titles):
        v = d.get(k, None)
        if isinstance(v, float):
            try:
                v = f"{float(v):.6g}"
            except Exception:
                pass
        elif isinstance(v, (dict, list)):
            try:
                v = json.dumps(v)
            except Exception:
                v = str(v)
        else:
            v = "" if v is None else str(v)
        lines.append(f"| {t} | {v} |")
    return "\n".join(lines) + "\n"


def _ensure_plsr_calibration(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "plsr_cv_predictions.csv"
    if not csv_path.exists():
        return
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        if df.empty or not set(["y_true", "y_pred"]).issubset(df.columns):
            return
        y_true = df["y_true"].astype(float).values
        y_pred = df["y_pred"].astype(float).values
        resid = y_pred - y_true
        # Residual histogram with quantiles
        try:
            fig, ax = plt.subplots(figsize=(6.4, 3.8))
            ax.hist(resid, bins=20, color="#6baed6", alpha=0.8, edgecolor="#2b8cbe")
            for q, col in [(0.9, "#e6550d"), (0.95, "#31a354")]:
                qv = float(np.quantile(np.abs(resid), q))
                ax.axvline(+qv, color=col, linestyle="--", label=f"+|res| q{int(q*100)}")
                ax.axvline(-qv, color=col, linestyle="--", label=f"-|res| q{int(q*100)}")
            ax.set_title("PLSR Residuals")
            ax.set_xlabel("Residual (y_pred - y_true)")
            ax.set_ylabel("Count")
            ax.legend(loc="upper right", ncols=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(plots_dir / "plsr_residual_hist.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
        # Calibration fit plot
        try:
            # simple OLS fit: y_pred = a + b * y_true
            X = np.vstack([np.ones_like(y_true), y_true]).T
            coef, *_ = np.linalg.lstsq(X, y_pred, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            xx = np.linspace(np.nanmin(y_true), np.nanmax(y_true), 100)
            yy = a + b * xx
            fig, ax = plt.subplots(figsize=(6.0, 4.5))
            ax.scatter(y_true, y_pred, s=18, color="#4daf4a", alpha=0.8, label="CV pred")
            ax.plot(xx, xx, color="#636363", linestyle=":", label="Ideal (slope=1)")
            ax.plot(xx, yy, color="#e41a1c", linestyle="-", label=f"Fit: y={a:.3g}+{b:.3g}x")
            ax.set_xlabel("True concentration (ppm)")
            ax.set_ylabel("Predicted concentration (ppm)")
            ax.set_title("PLSR Calibration Fit (CV)")
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(plots_dir / "plsr_calibration_fit.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
    except Exception:
        pass


def _plsr_calibration_md(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    csv_path = metrics_dir / "plsr_cv_predictions.csv"
    if not csv_path.exists():
        return "PLSR CV predictions not found.\n"
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(csv_path)
        if df.empty or not set(["y_true", "y_pred"]).issubset(df.columns):
            return "PLSR CV predictions not found.\n"
        y_true = df["y_true"].astype(float).values
        y_pred = df["y_pred"].astype(float).values
        resid = y_pred - y_true
        # calibration: fit y_pred = a + b y_true
        X = np.vstack([np.ones_like(y_true), y_true]).T
        coef, *_ = np.linalg.lstsq(X, y_pred, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        yhat = a + b * y_true
        ss_res = float(np.sum((y_pred - yhat) ** 2))
        ss_tot = float(np.sum((y_pred - np.mean(y_pred)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        # empirical symmetric PIs via residual quantiles
        q90 = float(np.quantile(np.abs(resid), 0.90))
        q95 = float(np.quantile(np.abs(resid), 0.95))
        cov90 = float(np.mean(np.abs(resid) <= q90))
        cov95 = float(np.mean(np.abs(resid) <= q95))
        kv = {
            "cal_intercept": a,
            "cal_slope": b,
            "cal_r2": r2,
            "residual_rmse": rmse,
            "pi90_abs_residual": q90,
            "pi95_abs_residual": q95,
            "pi90_coverage": cov90,
            "pi95_coverage": cov95,
        }
        titles = [
            "Calibration intercept (a)",
            "Calibration slope (b)",
            "Calibration R²",
            "Residual RMSE",
            "Empirical PI half-width @90%",
            "Empirical PI half-width @95%",
            "Observed coverage @90%",
            "Observed coverage @95%",
        ]
        table = _md_kv_table(kv, list(kv.keys()), titles)
        notes = [
            f"- Coverage check: target 90% → observed ≈ {cov90:.3f}; target 95% → observed ≈ {cov95:.3f}.",
            f"- Calibration check: slope b ≈ {b:.3g}, intercept a ≈ {a:.3g} (ideal a≈0, b≈1).",
        ]
        return table + "\n" + "\n".join(notes) + "\n"
    except Exception:
        return "(Could not compute calibration metrics.)\n"


def _md_list(items: List[str]) -> str:
    return "\n".join([f"- {it}" for it in items]) + "\n"


def _find_plots(run_dir: Path) -> List[Path]:
    plots = [
        "canonical_overlay.png",
        "slope_to_noise_heatmap.png",
        "concentration_response.png",
        "peak_shift_calibration.png",
        "peak_center_delta_calibration.png",
        "peak_shift_delta_calibration.png",
        "peak_shift_delta_best_calibration.png",
        "peak_center_delta_best_calibration.png",
        "shift_manhattan.png",
        "shift_heatmap.png",
        "fullscan_concentration_response.png",
        "plsr_cv_curves.png",
        "plsr_pred_vs_actual.png",
        "plsr_residuals.png",
        "plsr_residual_hist.png",
        "plsr_calibration_fit.png",
        "mcr_als_components.png",
        "mcr_als_pred_vs_actual.png",
        "mcr_als_components_roi.png",
        "volcano_plot.png",
        "manhattan_plot.png",
        "permutation_summary.png",
        "roi_repeatability.png",
        "gas_dashboard.png",
    ]
    out = []
    plots_dir = run_dir / "plots"
    for p in plots:
        cand = plots_dir / p
        if cand.exists():
            out.append(cand)
    return out


def _plots_md(plots: List[Path], run_dir: Path) -> str:
    if not plots:
        return ""
    captions = {
        "canonical_overlay": "Aggregated spectra per concentration; sanity-check directionality and band shape in the ROI.",
        "concentration_response": "ROI signal vs concentration with regression; slope, R², RMSE relate to sensitivity and linearity.",
        "peak_shift_calibration": "Absolute wavelength (nm) vs concentration with linear fit; peak-center shift (Δλ) calibration.",
        "peak_center_delta_calibration": "Δλ of peak center (relative to baseline λ₀) vs concentration with weighted fit.",
        "peak_shift_delta_calibration": "Δλ (sub‑pixel xcorr, amplitude‑invariant) vs concentration with weighted fit.",
        "peak_shift_delta_best_calibration": "Δλ for auto‑selected best window (scanned across wavelength) with weighted fit.",
        "peak_center_delta_best_calibration": "Δλ of center in auto‑selected best window; baseline λ₀ annotated.",
        "shift_manhattan": "Significance of Δλ across wavelength (BH‑FDR); spikes highlight robust shift regions.",
        "shift_heatmap": "Δλ linearity/score across wavelength and window size; brighter = more linear/robust region.",
        "fullscan_concentration_response": "Effect profiles across all wavelengths; peaks locate candidate bands and support ROI choice.",
        "slope_to_noise_heatmap": "Effect size normalized by noise across spectrum; warmer colors indicate better detectability.",
        "volcano_plot": "Effect magnitude vs −log10(q) under FDR; more significant points appear higher.",
        "manhattan_plot": "−log10(q) across wavelength; contiguous spikes indicate robust significant regions after multiple testing.",
        "permutation_summary": "Observed |slope| vs empirical null percentiles; beyond 95th percentile implies unlikely by chance.",
        "plsr_cv_curves": "Cross-validated performance vs components; use one‑SE rule for parsimonious complexity.",
        "plsr_pred_vs_actual": "Calibration scatter against ideal y=x; spread illustrates prediction error and bias.",
        "plsr_residuals": "Residual diagnostics across folds; patterns can reveal misfit or heteroscedasticity.",
        "plsr_residual_hist": "Residual distribution; supports empirical predictive intervals and coverage checks.",
        "plsr_calibration_fit": "Fit y_pred = a + b·y_true; a≈0 and b≈1 indicate unbiased calibration-in-the-large.",
        "mcr_als_components": "Resolved components; interpret spectral signatures and loadings.",
        "mcr_als_pred_vs_actual": "MCR‑ALS predicted vs actual; overall multivariate calibration quality.",
        "mcr_als_components_roi": "Components overlaid on ROI; alignment supports chemical plausibility of selected bands.",
        "roi_repeatability": "ROI stability across repeats; smaller spread indicates robust behavior.",
        "gas_dashboard": "At‑a‑glance key metrics to track improvements and regressions across runs.",
    }
    blocks = []
    reports_dir = run_dir / "reports"
    for idx, p in enumerate(plots, 1):
        try:
            relp = Path(os.path.relpath(p, reports_dir))
        except Exception:
            relp = p
        stem = p.stem
        cap = captions.get(stem, "")
        img_md = f"![{stem}]({relp.as_posix()})"
        cap_md = f"_Figure {idx}: {cap}_" if cap else f"_Figure {idx}_"

        notes_map = {
            "peak_shift_calibration": [
                "Legend: triangles with error bars = mean±SD of fitted peak centers per concentration; red dotted = linear fit.",
                "How to read: downward/upward slope indicates Δλ per ppm; text box shows y=mx+b and R².",
                "Why it matters: Δλ is drift‑robust and physically tied to refractive index/resonance changes, complementing amplitude.",
            ],
            "peak_shift_delta_calibration": [
                "Legend: triangles with error bars = mean±SD Δλ per concentration; red dotted = weighted linear fit.",
                "How to read: slope is Δλ per ppm; weights down‑weight noisy concentrations (large SD).",
                "Why it matters: sub‑pixel xcorr is amplitude‑invariant and uses the full ROI shape for better sensitivity.",
            ],
            "peak_center_delta_calibration": [
                "Legend: triangles with error bars = mean±SD Δλ(center) per concentration; baseline λ₀ is annotated.",
                "How to read: Δλ(center) tracks feature position change relative to λ₀ (uses ROI wavelength logic).",
                "Why it matters: shows the exact wavelength where the feature starts and how far it shifts.",
            ],
            "peak_shift_delta_best_calibration": [
                "Legend: same as above but for auto‑selected best window from a wavelength scan.",
                "How to read: should show the strongest monotonic Δλ response for this dataset.",
                "Why it matters: auto‑selection avoids manual ROI guesswork when shifts are tiny.",
            ],
            "peak_center_delta_best_calibration": [
                "Legend: Δλ(center) computed inside the auto‑selected best Δλ window; baseline λ₀ is annotated.",
                "How to read: verifies the same window gives a consistent center‑based shift.",
                "Why it matters: aligns ROI selection with wavelength logic (transmittance + reference).",
            ],
            "shift_manhattan": [
                "Legend: points = windows; y = −log10(q) after BH‑FDR; dashed line = significance threshold.",
                "How to read: significant spikes mark wavelength regions with consistent Δλ across repeats.",
                "Why it matters: guards against false positives when scanning many windows.",
            ],
            "shift_heatmap": [
                "Legend: x = window center (nm), y = window width (nm), color = metric (e.g., R²_w).",
                "How to read: bright clusters indicate stable, linear Δλ regions across window sizes.",
                "Why it matters: helps fine‑tune window size and center to maximize Δλ linearity and robustness.",
            ],
            "volcano_plot": [
                "Legend: red = Significant (FDR), gray = Not significant; dashed line marks q = α.",
                "How to read: higher −log10(q) = stronger statistical support; slope sign shows direction of effect.",
                "Why it matters: highlights wavelengths with both effect and statistical support to define/validate the ROI.",
            ],
            "manhattan_plot": [
                "Legend: red = Significant (FDR), blue = Not significant; dashed line marks q = α.",
                "How to read: points above the line are significant after multiple testing; contiguous spikes suggest robust bands.",
                "Why it matters: helps anchor the ROI center and width objectively.",
            ],
            "permutation_summary": [
                "Legend: dashed = empirical 5th/50th/95th percentiles of |slope| (null); solid red = observed |slope|.",
                "How to read: observed beyond 95th percentile → effect unlikely by chance.",
                "Why it matters: nonparametric validation of ROI significance.",
            ],
            "slope_to_noise_heatmap": [
                "How to read: warmer colors = higher slope-to-noise ratio (STN).",
                "Why it matters: prioritizes bands with robust detectability beyond significance alone.",
            ],
            "concentration_response": [
                "How to read: regression slope = sensitivity; R² and RMSE summarize fit quality.",
                "Why it matters: quantifies linear response within the ROI and supports LOD/LOQ estimates.",
            ],
            "canonical_overlay": [
                "How to read: monotonic changes and consistent band shapes across concentrations indicate a physical response.",
                "Why it matters: sanity‑checks preprocessing and ROI directionality.",
            ],
            "fullscan_concentration_response": [
                "How to read: peaks across wavelength indicate candidate bands for the ROI.",
                "Why it matters: seeds ROI selection before confirmatory testing.",
            ],
            "plsr_cv_curves": [
                "How to read: prefer the smallest component count within one‑SE of the minimum CV error.",
                "Why it matters: avoids overfitting while keeping performance.",
            ],
            "plsr_pred_vs_actual": [
                "How to read: points close to y=x imply accurate predictions; spread indicates error magnitude.",
                "Why it matters: checks overall predictive performance.",
            ],
            "plsr_residual_hist": [
                "Legend: dashed lines = ±|residual| quantiles at 90% and 95%.",
                "How to read: narrower distribution and smaller quantiles → tighter predictive intervals.",
                "Why it matters: sets empirical predictive interval half‑widths used in uncertainty reporting.",
            ],
            "plsr_calibration_fit": [
                "Legend: dotted gray = ideal (y=x); red = fitted calibration; green points = CV predictions.",
                "How to read: a≈0 and b≈1 → unbiased; deviations indicate bias.",
                "Why it matters: confirms whether the model is trustworthy for concentration estimation.",
            ],
            "mcr_als_components": [
                "How to read: component spectra should show meaningful peaks.",
                "Why it matters: interpretability and chemical plausibility.",
            ],
            "mcr_als_pred_vs_actual": [
                "How to read: similar to PLSR pred vs actual, but for MCR‑ALS.",
                "Why it matters: compares multivariate methods.",
            ],
            "mcr_als_components_roi": [
                "How to read: component peaks aligning with the ROI support analyte attribution.",
                "Why it matters: strengthens ROI choice.",
            ],
            "roi_repeatability": [
                "How to read: smaller spread across repeats indicates robust behavior.",
                "Why it matters: assesses stability across runs.",
            ],
            "gas_dashboard": [
                "How to read: quick view of key metrics vs thresholds.",
                "Why it matters: informs operational decisions.",
            ],
        }
        note_lines = notes_map.get(stem, [])
        notes_md = ("\n" + "\n".join([f"- {ln}" for ln in note_lines])) if note_lines else ""

        block = "\n\n".join([img_md, cap_md]).strip() + notes_md
        blocks.append(block)
    return "\n\n".join(blocks) + "\n"


def _figure_index_md(run_dir: Path) -> str:
    plots = _find_plots(run_dir)
    if not plots:
        return "No figures available.\n"
    captions = {
        "canonical_overlay": "Aggregated spectra per concentration; sanity-check directionality and band shape in the ROI.",
        "concentration_response": "ROI signal vs concentration with regression; slope, R², RMSE relate to sensitivity and linearity.",
        "fullscan_concentration_response": "Effect profiles across all wavelengths; peaks locate candidate bands and support ROI choice.",
        "slope_to_noise_heatmap": "Effect size normalized by noise across spectrum; warmer colors indicate better detectability.",
        "volcano_plot": "Effect magnitude vs −log10(q) under FDR; more significant points appear higher.",
        "manhattan_plot": "−log10(q) across wavelength; contiguous spikes indicate robust significant regions after multiple testing.",
        "permutation_summary": "Observed |slope| vs empirical null percentiles; beyond 95th percentile implies unlikely by chance.",
        "plsr_cv_curves": "Cross-validated performance vs components; use one‑SE rule for parsimonious complexity.",
        "plsr_pred_vs_actual": "Calibration scatter against ideal y=x; spread illustrates prediction error and bias.",
        "plsr_residuals": "Residual diagnostics across folds; patterns can reveal misfit or heteroscedasticity.",
        "plsr_residual_hist": "Residual distribution; supports empirical predictive intervals and coverage checks.",
        "plsr_calibration_fit": "Fit y_pred = a + b·y_true; a≈0 and b≈1 indicate unbiased calibration-in-the-large.",
        "mcr_als_components": "Resolved components; interpret spectral signatures and loadings.",
        "mcr_als_pred_vs_actual": "MCR‑ALS predicted vs actual; overall multivariate calibration quality.",
        "mcr_als_components_roi": "Components overlaid on ROI; alignment supports chemical plausibility of selected bands.",
        "roi_repeatability": "ROI stability across repeats; smaller spread indicates robust behavior.",
        "gas_dashboard": "At‑a‑glance key metrics to track improvements and regressions across runs.",
    }
    lines = []
    for idx, p in enumerate(plots, 1):
        stem = p.stem
        pretty = stem.replace("_", " ").title()
        cap = captions.get(stem, "")
        if cap:
            lines.append(f"- Figure {idx}: {pretty}. {cap}")
        else:
            lines.append(f"- Figure {idx}: {pretty}.")
    return "\n".join(lines) + "\n"


def _appendix_md(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    md = []

    # QC Summary
    qc = _read_json(metrics_dir / "qc_summary.json")
    md.append(_md_h2("Quality Control Summary"))
    if qc:
        keys = [
            "min_snr", "median_snr", "max_rsd_percent",
            "snr_threshold", "rsd_threshold_percent",
            "snr_pass", "rsd_pass", "overall_pass",
        ]
        titles = [
            "Min SNR", "Median SNR", "Max RSD%",
            "SNR Threshold", "RSD Threshold%",
            "SNR Pass", "RSD Pass", "Overall Pass",
        ]
        md.append(_md_kv_table(qc, keys, titles))
    else:
        md.append("QC summary not found.\n")

    # Calibration Metrics (univariate + PLSR block if present)
    calib = _read_json(metrics_dir / "calibration_metrics.json")
    md.append(_md_h2("Calibration Metrics"))
    if calib:
        keys = ["slope", "intercept", "r2", "rmse", "lod", "loq", "roi_center"]
        titles = ["Slope", "Intercept", "R²", "RMSE", "LOD", "LOQ", "ROI Center"]
        md.append(_md_kv_table(calib, keys, titles))
        # PLSR sub-block
        plsr = calib.get("plsr_model") if isinstance(calib, dict) else None
        if isinstance(plsr, dict):
            md.append(_md_h3("PLSR Summary"))
            pk = ["model", "n_components", "r2_cv", "rmse_cv"]
            pt = ["Model", "Components", "CV R² (selected)", "CV RMSE (selected)"]
            md.append(_md_kv_table(plsr, pk, pt))
    else:
        md.append("Calibration metrics not found.\n")

    # ROI Performance
    roi = _read_json(metrics_dir / "roi_performance.json")
    md.append(_md_h2("ROI Performance"))
    if roi:
        rk = [
            "regression_slope", "regression_intercept", "regression_r2", "regression_rmse",
            "dynamic_range", "dynamic_range_per_ppm", "lod_ppm", "loq_ppm"
        ]
        rt = ["Slope", "Intercept", "R²", "RMSE", "Dynamic range", "Dynamic range/ppm", "LOD (ppm)", "LOQ (ppm)"]
        md.append(_md_kv_table(roi, rk, rt))
        bs = roi.get("bootstrap", {}) if isinstance(roi, dict) else {}
        if bs and bs.get("enabled") and isinstance(bs.get("ci", {}), dict):
            ci = bs.get("ci", {})
            md.append(_md_h3("ROI Bootstrap Confidence Intervals (95%)"))
            def _fmt_ci(d):
                try:
                    lo = float(d.get('lower', np.nan))
                    med = float(d.get('median', np.nan))
                    hi = float(d.get('upper', np.nan))
                    return f"{lo:.6g} – {hi:.6g} (median {med:.6g})"
                except Exception:
                    return "NA"
            lines = [
                f"- Slope: {_fmt_ci(ci.get('slope', {}))}",
                f"- R²: {_fmt_ci(ci.get('r2', {}))}",
                f"- RMSE: {_fmt_ci(ci.get('rmse', {}))}",
                f"- LOD (ppm): {_fmt_ci(ci.get('lod_ppm', {}))}",
                f"- LOQ (ppm): {_fmt_ci(ci.get('loq_ppm', {}))}",
                ""
            ]
            md.append("\n".join(lines))
            # Numeric worked example for LOD/LOQ using current ROI metrics
            try:
                lod_val = float(roi.get("lod_ppm", np.nan))
                loq_val = float(roi.get("loq_ppm", np.nan))
                if np.isfinite(lod_val) and np.isfinite(loq_val):
                    md.append(_md_h3("Worked Example: Detection Limits"))
                    md.append(f"Using the ROI slope and noise estimates, LOD ≈ {lod_val:.3g} ppm and LOQ ≈ {loq_val:.3g} ppm.")
            except Exception:
                pass
    else:
        md.append("ROI performance not found.\n")

    # Multivariate Selection
    mv = _read_json(metrics_dir / "multivariate_selection.json")
    md.append(_md_h2("Multivariate Selection"))
    if mv and isinstance(mv, dict):
        best = mv.get("best_method")
        scores = mv.get("scores", {})
        md.append(f"Best by CV R²: {best}\n\n")
        md.append("| Model | CV R² | RMSE |\n|---|---:|---:|\n")
        for k in ["plsr", "ica", "mcr_als"]:
            sc = scores.get(k, {}) or {}
            r2v = sc.get("r2_cv")
            rmsev = sc.get("rmse_cv")
            r2s = "NA" if r2v is None else f"{float(r2v):.4g}"
            rmses = "NA" if rmsev is None else f"{float(rmsev):.4g}"
            md.append(f"| {k.upper()} | {r2s} | {rmses} |\n")
    else:
        md.append("Multivariate selection not found.\n")

    # Top wavelengths (first 20)
    tw_path = metrics_dir / "top_wavelengths.csv"
    md.append(_md_h2("Top Wavelengths (first 20)"))
    if tw_path.exists():
        try:
            import pandas as pd  # type: ignore
            df = pd.read_csv(tw_path)
            show_cols = [
                c for c in [
                    "wavelength", "slope", "slope_ci_low", "slope_ci_high", "slope_stderr",
                    "residual_std", "slope_to_noise", "slope_to_noise_stderr", "r2",
                    "p_value", "p_adjusted", "significant"
                ]
                if c in df.columns
            ]
            head = df[show_cols].head(20)
            md.append(head.to_markdown(index=False))
            md.append("\n")
            try:
                notes = []
                if "p_adjusted" in head.columns:
                    notes.append("q-value is the FDR-adjusted p-value (Benjamini–Hochberg).")
                if "slope_ci_low" in head.columns and "slope_ci_high" in head.columns:
                    notes.append("slope_ci_low/high are 95% t-based confidence interval bounds for the slope.")
                notes.append("Slope units are ΔT/ppm; higher |slope| and STN indicate stronger, more detectable effects.")
                if notes:
                    md.append("_Notes_: " + " ".join(notes) + "\n")
            except Exception:
                pass
        except Exception:
            md.append("(Could not render table; see CSV.)\n")
    else:
        md.append("Top wavelengths CSV not found.\n")

    # Multiple-testing and significance summary
    cr_path = metrics_dir / "concentration_response.json"
    md.append(_md_h2("Significance & Multiple-Testing"))
    if cr_path.exists():
        cr = _read_json(cr_path)
        if cr and isinstance(cr, dict):
            mt = cr.get("multiple_testing", {})
            if mt and mt.get("enabled"):
                md.append("**Multiple-Testing Control**")
                md.append(_md_kv_table(mt, ["method", "alpha", "significant_count"], ["Method", "Alpha", "Significant wavelengths"]))
                md.append("")
            pt = cr.get("permutation_test", {})
            if pt and pt.get("enabled"):
                md.append("**Permutation Test (ROI validation)**")
                md.append(_md_kv_table(pt, ["n_iterations_effective", "observed_abs_slope", "empirical_p_value"], ["Iterations", "Observed |slope|", "Empirical p-value"]))
                pct = pt.get("percentiles", {})
                if pct:
                    md.append(f"Percentiles (5/50/95): {pct.get('5', 'N/A')}/{pct.get('50', 'N/A')}/{pct.get('95', 'N/A')}")
                md.append("")
                md.append("Note: In the Volcano and Manhattan plots, the dashed horizontal line marks the FDR threshold (q = α).")
                md.append("Note: In the Volcano and Manhattan plots, the dashed horizontal line marks the FDR threshold (q = α).")
                md.append("Note: In the Volcano and Manhattan plots, the dashed horizontal line marks the FDR threshold (q = α).")
    else:
        md.append("Significance summary not found.\n")

    # Provenance
    meta_path = metrics_dir / "run_metadata.json"
    md.append(_md_h2("Provenance & Reproducibility"))
    if meta_path.exists():
        meta = _read_json(meta_path) or {}
        keys = [
            k for k in [
                "run_id", "timestamp", "user", "hostname", "git_commit", "random_seed",
                "software_versions", "notes"
            ] if k in (meta or {})
        ]
        if keys:
            md.append(_md_kv_table(meta, keys, [x.replace('_', ' ').title() for x in keys]))
        else:
            md.append("(run_metadata.json present)\n")
    else:
        md.append("Run metadata not found.\n")

    # PLSR calibration and predictive intervals
    md.append(_md_h2("PLSR Calibration & Predictive Intervals"))
    md.append(_plsr_calibration_md(run_dir))

    # Artifacts list
    md.append(_md_h2("Artifacts"))
    artifacts = [
        "metrics/calibration_metrics.json",
        "metrics/roi_performance.json",
        "metrics/noise_metrics.json",
        "metrics/qc_summary.json",
        "metrics/multivariate_selection.json",
        "metrics/concentration_response.json",
        "metrics/top_wavelengths.csv",
        "metrics/plsr_cv_predictions.csv",
    ]
    md.append(_md_list(artifacts))

    md.append(_md_h2("Figure Index"))
    md.append(_figure_index_md(run_dir))

    # Selected plots
    md.append(_md_h2("Key Plots"))
    md.append(_plots_md(_find_plots(run_dir), run_dir))

    return "".join(md)


def _interpretation_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Results Interpretation Guide"))
    md.append("This guide explains what each figure/table conveys, why it's used, and how to read it.\n\n")
    items = [
        ("Canonical overlay", "Overlay of aggregated spectra per concentration. Why: sanity-check directionality and shape of response within ROI. Read: look for monotonic changes and consistent band shapes across concentrations."),
        ("Concentration response", "ROI signal vs concentration with regression. Why: quantify sensitivity and linearity. Read: slope sign/magnitude, R², RMSE; compare to LOD/LOQ."),
        ("Full-scan concentration response", "Effect profiles across all wavelengths. Why: locate bands with strongest trends. Read: peaks indicate high effect bands; used to seed ROI."),
        ("Slope-to-noise heatmap", "Effect size normalized by noise. Why: emphasizes robust detectability. Read: warmer colors mean higher slope-to-noise; thresholds suggest reliable bands."),
        ("Volcano plot", "Effect (slope) vs −log10(q). Why: combine magnitude and significance under FDR control. Read: upper points are more significant; right/left reflect effect direction."),
        ("Manhattan plot", "−log10(q) across wavelength. Why: identify continuous significant regions. Read: spikes highlight statistically robust bands after multiple-testing control."),
        ("Permutation summary", "Observed |slope| vs empirical null percentiles. Why: nonparametric validation of ROI effect. Read: observed line beyond 95th percentile → unlikely by chance."),
        ("Top wavelengths table", "Band-level stats (slope, r², p/q, significance). Why: tabular view of best bands. Read: use adjusted p (q) and slope CI to prioritize interpretable, stable peaks."),
        ("PLSR CV curves", "R²/RMSE across components. Why: select complexity via one-SE rule. Read: prefer parsimonious component count with near-best CV performance."),
        ("PLSR predicted vs actual", "Calibration scatter with ideal line. Why: assess bias and spread. Read: points hugging y=x and narrow spread indicate good calibration."),
        ("PLSR residuals / histogram", "Error structure and magnitude. Why: check assumptions and define empirical predictive intervals. Read: symmetric, centered residuals imply well-behaved errors."),
        ("PLSR calibration fit", "Fit y_pred = a + b y_true. Why: calibration-in-the-large. Read: a≈0 and b≈1 indicate unbiased predictions; deviations signal bias."),
        ("Multivariate selection table", "CV comparison of models. Why: justify choice (PLSR/ICA/MCR-ALS). Read: selected model balances accuracy and simplicity under CV."),
        ("ROI performance (with bootstrap CIs)", "Sensitivity (slope), fit (R²), precision (RMSE), and detection limits with uncertainty. Why: quantify reliability. Read: narrower CIs → more stable performance."),
        ("MCR-ALS components / ROI overlay", "Resolved components and their alignment with ROI. Why: interpretability and spectral specificity. Read: component peaks aligning with ROI supports chemical plausibility."),
        ("ROI repeatability", "Variation across replicates. Why: assess stability. Read: small spread across repeats indicates robust sensor behavior."),
        ("Gas dashboard", "At-a-glance summary of key metrics. Why: operational decisions. Read: track improvements and regressions run-to-run."),
    ]
    for name, expl in items:
        md.append(f"- **{name}**. {expl}")
    md.append("\n")
    return "".join(md)


def _plain_language_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Plain-Language Summary"))
    metrics_dir = run_dir / "metrics"
    roi = _read_json(metrics_dir / "roi_performance.json") or {}
    try:
        slope = roi.get("regression_slope", None)
        r2 = roi.get("regression_r2", None)
        lod = roi.get("lod_ppm", None)
        loq = roi.get("loq_ppm", None)
        bullets = []
        if slope is not None:
            s = float(slope)
            direction = "decreases" if s < 0 else "increases"
            bullets.append(f"- The sensor signal {direction} as concentration increases; sensitivity (slope) ≈ {s:.3g} ΔT/ppm.")
        if r2 is not None:
            bullets.append(f"- Fit quality R² ≈ {float(r2):.3g}, indicating how much of the variation is explained by concentration.")
        if lod is not None and loq is not None:
            bullets.append(f"- Detection limits: LOD ≈ {float(lod):.3g} ppm and LOQ ≈ {float(loq):.3g} ppm (smaller is better).")
        md.append("This report shows where the spectrum responds to the gas, how strong and reliable the change is, and how accurately we can predict concentration.\n")
        if bullets:
            md.append("\n".join(bullets) + "\n")
    except Exception:
        md.append("This report summarizes the main findings in plain language.\n")
    return "".join(md)


def _glossary_md() -> str:
    items = [
        ("ROI (Region of Interest)", "Wavelength window where the analyte causes the most consistent, strongest change."),
        ("FDR / q-value", "FDR controls false discoveries across many tests; q-value is the adjusted p-value after FDR."),
        ("Permutation test", "Shuffle concentrations to build a 'no-effect' baseline; compare the observed slope against this null."),
        ("PLSR", "Multivariate model that predicts concentration from the whole spectrum using a small number of components."),
        ("Predictive interval", "Range around the prediction expected to contain the true value with a given probability (e.g., 90%)."),
        ("LOD / LOQ", "Smallest levels we can reliably detect/quantify, based on noise and sensitivity."),
        ("Calibration-in-the-large", "Checks overall prediction bias using y_pred = a + b·y_true; unbiased if a≈0 and b≈1."),
    ]
    md = []
    md.append(_md_h2("Glossary"))
    for k, v in items:
        md.append(f"- **{k}**: {v}")
    md.append("\n")
    return "".join(md)


def _assumptions_md() -> str:
    md = []
    md.append(_md_h2("Assumptions and Limitations"))
    bullets = [
        "Linear trend between signal and concentration within the studied range.",
        "Residuals are roughly independent and symmetric; large deviations widen predictive intervals.",
        "SNR and baseline stability affect LOD/LOQ; results depend on current setup and environment.",
        "FDR assumes comparable tests; correlation among wavelengths is mitigated by ROI aggregation.",
        "External generalization depends on drift and transfer; future work will include external validation.",
    ]
    md.append(_md_list(bullets))
    return "".join(md)


def _technical_justification_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Technical Justification and Mathematical Details"))

    md.append(_md_h3("Multiple testing control (Benjamini–Hochberg FDR)"))
    md.append(r"""
We test $m$ wavelengths. Sort p-values $p_{(1)} \le \cdots \le p_{(m)}$. Define
$k = \max\{ i : p_{(i)} \le (i/m)\,\alpha \}$. Then the first $k$ are significant at FDR level $\alpha$ [1].
Adjusted p-values (q-values) are $q_{(i)} = \min_{j \ge i} \frac{m}{j}\, p_{(j)}$, ensuring monotonicity.
This controls $\mathbb{E}[\mathrm{FDR}] \le \alpha$ under independence/PRDS, avoiding false discoveries from scanning many wavelengths.
""")

    md.append(_md_h3("Per-wavelength linear regression and t-based confidence interval"))
    md.append(r"""
Model: $y_i = \beta_0 + \beta_1 c_i + \epsilon_i$, with residual variance $\sigma^2$.
Estimate $\hat{\beta}_1$ and standard error $\operatorname{SE}(\hat{\beta}_1) = s\big/\sqrt{\sum_i (c_i - \bar c)^2}$, where
$s^2 = \mathrm{SSE}/(n-2)$ and $\mathrm{SSE}=\sum_i (y_i - \hat y_i)^2$. A $95\%$ CI is
$\hat{\beta}_1 \pm t_{n-2,0.975}\,\operatorname{SE}(\hat{\beta}_1)$. This is reported per-wavelength and used to filter unstable bands.
""")

    md.append(_md_h3("Permutation test for ROI effect (empirical null)"))
    md.append(r"""
We permute concentrations to break association and compute $|\hat{\beta}_1|$ under the null, $B$ times.
The two-sided empirical p-value is $p = \frac{1 + \#\{b : |\hat{\beta}_1^{(b)}| \ge |\hat{\beta}_1|\}}{B+1}$ [2].
This validates that the ROI slope is unlikely to occur by chance, complementing parametric tests.
""")

    md.append(_md_h3("Bootstrap CIs for ROI performance (slope, R², RMSE, LOD/LOQ)"))
    md.append(r"""
Nonparametric bootstrap resamples $(c_i, y_i)$ pairs with replacement, recomputing metrics over $B$ replicates.
Percentile intervals (2.5/50/97.5) provide robust uncertainty even when residuals are non-normal [2].
We report $95\%$ CIs for slope, $R^2$, RMSE, and detection limits to quantify reliability.
""")

    md.append(_md_h3("PLSR selection and overfitting guard (one-standard-error rule)"))
    md.append(r"""
PLSR projects spectra into latent components to predict concentration [3]. We evaluate CV RMSE across components and
select the smallest number whose RMSE is within one standard error of the minimum (one-SE rule) [4].
This favors simpler, more generalizable models without sacrificing predictive performance.
""")

    md.append(_md_h3("Predictive intervals and coverage"))
    md.append(r"""
Empirical predictive intervals use residual quantiles: half-width $q_{\gamma} = \operatorname{quantile}(|\mathrm{residual}|, \gamma)$.
Coverage is the observed fraction with $|\mathrm{residual}| \le q_{\gamma}$. This is robust and assumption-light compared to
parametric $\hat y \pm t\, s\,\sqrt{1+h}$ intervals, and directly reflects model errors.
""")

    md.append(_md_h3("Calibration-in-the-large (bias check)"))
    md.append(r"""
We regress predictions on truths: $\hat y = a + b\, y$. Unbiased calibration implies $a \approx 0$ and $b \approx 1$.
Systematic deviations indicate under/overestimation or compression/expansion of the scale.
""")

    md.append(_md_h3("Detection limits (IUPAC)"))
    md.append(r"""
Using global noise $\sigma$, $\mathrm{LOD} = 3\sigma/|\hat{\beta}_1|$ and $\mathrm{LOQ} = 10\sigma/|\hat{\beta}_1|$ [8].
Equivalently, with residual SD $s_{y/x}$, one may use $\mathrm{LOD}_x = 3 s_{y/x}/|\hat{\beta}_1|$ and $\mathrm{LOQ}_x = 10 s_{y/x}/|\hat{\beta}_1|$.
""")

    md.append(_md_h3("Slope-to-noise ratio (detectability)"))
    md.append(r"""
We summarize detectability as $\mathrm{STN} = |\hat{\beta}_1|/\sigma_{\mathrm{noise}}$. Larger STN indicates more reliable detection,
guiding ROI choice along with FDR significance and permutation validation.
""")

    md.append(_md_h3("Why the overall approach is scientifically sound"))
    md.append(r"""
- Controls false discoveries across the spectrum (BH-FDR) while preserving power.
- Validates ROI effects with an empirical null (permutation) to avoid model misspecification.
- Quantifies uncertainty with per-band $t$ CIs and bootstrap CIs for ROI performance.
- Selects parsimonious multivariate models via one-SE rule, reducing overfitting risk.
- Communicates real-world uncertainty through empirical predictive intervals and calibration diagnostics.
""")

    return "".join(md)


def _technical_appendix_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Additional Technical Notes and Definitions"))
    md.append(_md_h3("Goodness-of-fit and error metrics"))
    md.append(r"""
We use $R^2 = 1 - \frac{\mathrm{SS}_{\mathrm{res}}}{\mathrm{SS}_{\mathrm{tot}}}$ where
$\mathrm{SS}_{\mathrm{res}}=\sum_i (y_i - \hat y_i)^2$ and $\mathrm{SS}_{\mathrm{tot}}=\sum_i (y_i - \bar y)^2$. RMSE is
$\mathrm{RMSE} = \sqrt{\tfrac{1}{n}\sum_i (y_i - \hat y_i)^2}$. These summarize fit quality and absolute error, respectively.
""")
    md.append(_md_h3("Cross-validation details"))
    md.append(r"""
We use repeated $K$-fold CV (or LOOCV when $K\ge n$). For each component count, we average fold-wise errors to obtain CV RMSE and its standard error (SE).
The one-standard-error rule selects the smallest component count whose CV RMSE is within $\text{min RMSE} + \text{SE}$, improving generalization.
""")
    md.append(_md_h3("p-values vs q-values (FDR)"))
    md.append(r"""
Raw $p$-values quantify evidence per wavelength but inflate false positives under multiplicity.
q-values are FDR-adjusted $p$-values via BH; using $q$ controls expected false discovery rate at level $\alpha$ while preserving power.
""")
    md.append(_md_h3("Effect direction (physical rationale)"))
    md.append(r"""
We often observe negative slopes for transmittance because higher analyte absorption reduces transmitted signal; after absorbance transform ($A=-\log_{10}T$), the effect becomes positive.
Reporting the sign clarifies whether the analyte attenuates or amplifies signal at the ROI.
""")
    md.append(_md_h3("Slope-to-noise ratio (STN)"))
    md.append(r"""
$\mathrm{STN} = \dfrac{|\hat{\beta}_1|}{\sigma_{\mathrm{noise}}}$ ranks detectability across wavelengths, complementing significance (q-values) and stability (slope CI width).
""")
    return "".join(md)


def _annotate_summary_images(summary_md: str, run_dir: Path) -> str:
    if not summary_md:
        return summary_md
    captions = {
        "canonical_overlay": "Aggregated spectra per concentration; sanity-check directionality and band shape in the ROI.",
        "concentration_response": "ROI signal vs concentration with regression; slope, R², RMSE relate to sensitivity and linearity.",
        "fullscan_concentration_response": "Effect profiles across all wavelengths; peaks locate candidate bands and support ROI choice.",
        "slope_to_noise_heatmap": "Effect size normalized by noise across spectrum; warmer colors indicate better detectability.",
        "volcano_plot": "Effect magnitude vs −log10(q) under FDR; more significant points appear higher.",
        "manhattan_plot": "−log10(q) across wavelength; contiguous spikes indicate robust significant regions after multiple testing.",
        "permutation_summary": "Observed |slope| vs empirical null percentiles; beyond 95th percentile implies unlikely by chance.",
        "plsr_cv_curves": "Cross-validated performance vs components; use one‑SE rule for parsimonious complexity.",
        "plsr_pred_vs_actual": "Calibration scatter against ideal y=x; spread illustrates prediction error and bias.",
        "plsr_residuals": "Residual diagnostics across folds; patterns can reveal misfit or heteroscedasticity.",
        "plsr_residual_hist": "Residual distribution; supports empirical predictive intervals and coverage checks.",
        "plsr_calibration_fit": "Fit y_pred = a + b·y_true; a≈0 and b≈1 indicate unbiased calibration-in-the-large.",
        "mcr_als_components": "Resolved components; interpret spectral signatures and loadings.",
        "mcr_als_pred_vs_actual": "MCR‑ALS predicted vs actual; overall multivariate calibration quality.",
        "mcr_als_components_roi": "Components overlaid on ROI; alignment supports chemical plausibility of selected bands.",
        "roi_repeatability": "ROI stability across repeats; smaller spread indicates robust behavior.",
        "gas_dashboard": "At‑a‑glance key metrics to track improvements and regressions across runs.",
    }
    out_lines = []
    for line in summary_md.splitlines():
        out_lines.append(line)
        s = line.strip()
        if s.startswith("![") and ")" in s and "(" in s:
            try:
                url = s[s.find("(") + 1 : s.rfind(")")]
                stem = Path(url).stem
                cap = captions.get(stem, None)
                if cap:
                    out_lines.append(f"_Caption: {cap}_")
            except Exception:
                pass
    return "\n".join(out_lines) + ("\n" if not summary_md.endswith("\n") else "")


def _key_findings_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Key Findings"))
    metrics_dir = run_dir / "metrics"
    bullets = []
    # ROI summary
    roi = _read_json(metrics_dir / "roi_performance.json") or {}
    try:
        s = roi.get("regression_slope", None)
        r2 = roi.get("regression_r2", None)
        rmse = roi.get("regression_rmse", None)
        lod = roi.get("lod_ppm", None)
        loq = roi.get("loq_ppm", None)
        if s is not None:
            s = float(s)
            direction = "decreases" if s < 0 else "increases"
            bullets.append(f"- ROI signal {direction} with concentration; sensitivity (slope) ≈ {s:.3g} ΔT/ppm.")
        if r2 is not None and rmse is not None:
            bullets.append(f"- ROI fit quality: R² ≈ {float(r2):.3g}, RMSE ≈ {float(rmse):.3g} ΔT.")
        if lod is not None and loq is not None:
            bullets.append(f"- Detection limits: LOD ≈ {float(lod):.3g} ppm; LOQ ≈ {float(loq):.3g} ppm.")
    except Exception:
        pass
    # Top wavelengths quick view
    try:
        import pandas as pd  # type: ignore
        tw = pd.read_csv(metrics_dir / "top_wavelengths.csv")
        if not tw.empty and "wavelength" in tw.columns:
            cols = [c for c in ["wavelength", "slope", "p_adjusted", "p_value"] if c in tw.columns]
            for _, r in tw[cols].head(3).iterrows():
                lam = float(r.get("wavelength", float("nan")))
                slope = float(r.get("slope", float("nan")))
                q = r.get("p_adjusted", None)
                p = r.get("p_value", None)
                sig = f"q≈{float(q):.2g}" if q is not None else (f"p≈{float(p):.2g}" if p is not None else "")
                bullets.append(f"- Top band near {lam:.2f} nm: slope {slope:.3g} ΔT/ppm {sig}.")
    except Exception:
        pass
    # PLSR calibration highlight (recompute minimal stats)
    try:
        import pandas as pd  # type: ignore
        df = pd.read_csv(metrics_dir / "plsr_cv_predictions.csv")
        if set(["y_true", "y_pred"]).issubset(df.columns) and not df.empty:
            y_true = df["y_true"].astype(float).values
            y_pred = df["y_pred"].astype(float).values
            resid = y_pred - y_true
            X = np.vstack([np.ones_like(y_true), y_true]).T
            coef, *_ = np.linalg.lstsq(X, y_pred, rcond=None)
            a, b = float(coef[0]), float(coef[1])
            q90 = float(np.quantile(np.abs(resid), 0.90))
            q95 = float(np.quantile(np.abs(resid), 0.95))
            cov90 = float(np.mean(np.abs(resid) <= q90))
            cov95 = float(np.mean(np.abs(resid) <= q95))
            bullets.append(f"- PLSR calibration: slope b ≈ {b:.3g}, intercept a ≈ {a:.3g}; coverage: 90%→{cov90:.3f}, 95%→{cov95:.3f}.")
    except Exception:
        pass
    if bullets:
        md.append("\n".join(bullets) + "\n")
    return "".join(md)


def _dual_channel_calibration_md(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    md = []
    md.append(_md_h2("Dual‑Channel Calibration (Intensity vs Δλ)"))
    try:
        inten = _read_json(metrics_dir / "roi_performance.json") or {}
        cr = _read_json(metrics_dir / "concentration_response.json") or {}
        shift = _read_json(metrics_dir / "peak_shift_metrics.json") or {}
        roi_s = cr.get("roi_start_wavelength", None)
        roi_e = cr.get("roi_end_wavelength", None)
        i_slope = inten.get("regression_slope", None)
        i_int = inten.get("regression_intercept", None)
        i_r2 = inten.get("regression_r2", None)
        i_lod = inten.get("lod_ppm", None)
        i_loq = inten.get("loq_ppm", None)
        try:
            win_i = f"{float(roi_s):.2f}–{float(roi_e):.2f} nm" if roi_s is not None and roi_e is not None else "NA"
        except Exception:
            win_i = "NA"

        s_slope = shift.get("slope_nm_per_ppm_weighted", shift.get("slope_nm_per_ppm", None))
        s_int = shift.get("intercept_nm_weighted", shift.get("intercept_nm", None))
        s_r2 = shift.get("r_squared_weighted", shift.get("r_squared", None))
        sd_arr = shift.get("delta_lambda_sd_nm", [])
        try:
            import numpy as _np  # type: ignore
            gsd = float(_np.nanmean(_np.asarray(sd_arr, dtype=float))) if sd_arr else None
        except Exception:
            gsd = None
        s_lod = (3.0 * gsd / abs(float(s_slope))) if (gsd is not None and s_slope not in (None, 0, 0.0)) else None
        s_loq = (10.0 * gsd / abs(float(s_slope))) if (gsd is not None and s_slope not in (None, 0, 0.0)) else None

        best_win = None
        try:
            import pandas as pd  # type: ignore
            import numpy as np  # type: ignore
            scan_path = metrics_dir / "shift_scan.csv"
            if scan_path.exists():
                df = pd.read_csv(scan_path)
                if not df.empty:
                    cand = df.copy()
                    if "q_value" in cand.columns:
                        sig = cand[(cand["q_value"].astype(float) <= 0.05)]
                        if not sig.empty:
                            cand = sig
                    if "score" in cand.columns:
                        best_win = cand.loc[cand["score"].idxmax()].to_dict()
                    else:
                        r = cand.copy()
                        if "r2_w" in r.columns and "slope_w" in r.columns:
                            r["tmp"] = r["r2_w"].fillna(0).astype(float) * r["slope_w"].abs().fillna(0).astype(float)
                            best_win = r.loc[r["tmp"].idxmax()].to_dict()
        except Exception:
            best_win = None
        if best_win is not None:
            try:
                win_s = float(best_win.get("start_nm", float("nan")))
                win_e = float(best_win.get("end_nm", float("nan")))
                win_shift = f"{win_s:.2f}–{win_e:.2f} nm"
            except Exception:
                win_shift = "NA"
        else:
            win_shift = "NA"

        rows = [
            ["Intensity (ROI)", win_i, i_slope, i_int, i_r2, i_lod, i_loq],
            ["Δλ (xcorr best)", win_shift, s_slope, s_int, s_r2, s_lod, s_loq],
        ]
        head = ["Channel", "Window (nm)", "Slope", "Intercept", "R²", "LOD (ppm)", "LOQ (ppm)"]
        fmt = []
        try:
            out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
            for r in rows:
                rr = []
                for j, v in enumerate(r):
                    if j == 0 or j == 1:
                        rr.append("" if v is None else str(v))
                    else:
                        try:
                            rr.append("" if v is None else f"{float(v):.6g}")
                        except Exception:
                            rr.append(str(v))
                out.append("| " + " | ".join(rr) + " |")
            md.append("\n".join(out) + "\n")
        except Exception:
            pass

        eq_lines = []
        try:
            if i_slope is not None and i_int is not None:
                eq_lines.append(f"- Intensity: T = {float(i_int):.5g} + {float(i_slope):.5g}·c (ΔT/ppm; ROI {win_i}).")
        except Exception:
            pass
        try:
            if s_slope is not None and s_int is not None:
                eq_lines.append(f"- Wavelength shift: Δλ = {float(s_int):.5g} + {float(s_slope):.5g}·c (nm/ppm; window {win_shift}).")
        except Exception:
            pass
        if eq_lines:
            md.append("\n" + "\n".join(eq_lines) + "\n")
    except Exception:
        return ""
    return "".join(md)

def _units_md() -> str:
    md = []
    md.append(_md_h2("Units and Notation"))
    items = [
        ("Concentration (c)", "ppm (parts per million)."),
        ("Signal (T)", "transmittance (unitless); changes reported as ΔT."),
        ("Sensitivity (slope)", "ΔT/ppm; sign indicates direction of change."),
        ("RMSE", "ΔT."),
        ("Wavelength (λ)", "nm."),
        ("Significance axis", "−log10(q), where q is the FDR-adjusted p-value."),
        ("Predictive interval", "Half-width in ppm, derived from residual quantiles (empirical)."),
    ]
    for k, v in items:
        md.append(f"- **{k}**: {v}")
    md.append("\n")
    return "".join(md)


def _markdown_to_html(md_text: str) -> str:
    css = """
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.4; max-width: 900px; margin: auto; }
      h1,h2,h3 { color: #333; }
      img { max-width: 100%; height: auto; display: block; margin: 10px 0; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 6px 8px; }
      th { background: #f5f5f5; }
      code { background: #f2f2f2; padding: 2px 4px; }
    </style>
    """
    mathjax = """
    <script>
    window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] } };
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """
    try:
        import markdown  # type: ignore
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])  # type: ignore
        return f"<html><head>{css}{mathjax}</head><body>{html}</body></html>"
    except Exception:
        # Fallback: wrap as preformatted
        safe = md_text.replace("<", "&lt;").replace(">", "&gt;")
        return f"<html><head>{css}</head><body><pre>{safe}</pre></body></html>"


def _write_pdf_from_markdown(md_path: Path, pdf_path: Path) -> bool:
    # Try pypandoc -> PDF; if pandoc missing, attempt auto-download
    try:
        import pypandoc  # type: ignore
        try:
            pypandoc.convert_file(str(md_path), "pdf", outputfile=str(pdf_path))  # requires pandoc installed
            return pdf_path.exists() and pdf_path.stat().st_size > 0
        except Exception:
            try:
                pypandoc.download_pandoc()  # attempt to fetch pandoc binary
                pypandoc.convert_file(str(md_path), "pdf", outputfile=str(pdf_path))
                return pdf_path.exists() and pdf_path.stat().st_size > 0
            except Exception:
                return False
    except Exception:
        return False


def _write_pdf_from_html(html_path: Path, pdf_path: Path, base_url: Path) -> bool:
    # Try WeasyPrint -> PDF
    try:
        from weasyprint import HTML  # type: ignore
        HTML(filename=str(html_path), base_url=str(base_url)).write_pdf(str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except Exception:
        return False


def _write_pdf_with_playwright(html_path: Path, pdf_path: Path) -> bool:
    # Try Playwright (Chromium) -> PDF
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return False
    try:
        url = Path(html_path).resolve().as_uri()
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url, wait_until="load")
            try:
                page.wait_for_function("() => typeof MathJax !== 'undefined' && MathJax.typesetPromise", timeout=5000)
                page.evaluate("MathJax.typesetPromise && MathJax.typesetPromise()")
            except Exception:
                pass
            page.pdf(path=str(pdf_path), print_background=True, format="A4",
                     margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"})
            browser.close()
        return pdf_path.exists() and pdf_path.stat().st_size > 0
    except Exception:
        return False


def _ensure_significance_plots(run_dir: Path) -> None:
    metrics_dir = run_dir / "metrics"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cr = _read_json(metrics_dir / "concentration_response.json") or {}
    if not cr:
        return
    try:
        wl = np.array(cr.get("wavelengths", []), dtype=float)
        slopes = np.array(cr.get("slopes", []), dtype=float)
        p_adj = np.array(cr.get("p_values_adjusted", []), dtype=float)
        sig = np.array(cr.get("significant_mask", []), dtype=bool)
        eps = 1e-16
        # Volcano plot
        try:
            fig, ax = plt.subplots(figsize=(7, 5))
            y = -np.log10(np.clip(p_adj, eps, 1.0))
            ax.scatter(slopes[~sig], y[~sig], s=12, c="#bbbbbb", label="Not significant", alpha=0.7)
            ax.scatter(slopes[sig], y[sig], s=14, c="#d62728", label="Significant (FDR)", alpha=0.8)
            # Reference line at q = alpha (default 0.05)
            alpha_thr = 0.05
            try:
                mt = cr.get("multiple_testing", {}) or {}
                if mt.get("enabled") and mt.get("alpha") is not None:
                    alpha_thr = float(mt.get("alpha"))
            except Exception:
                pass
            thr_y = -np.log10(np.clip(alpha_thr, eps, 1.0))
            ax.axhline(thr_y, color="#636363", linestyle="--", linewidth=1.0, label=f"q={alpha_thr:g}")
            ax.set_xlabel("Effect size (slope, ΔT/ppm)")
            ax.set_ylabel("-log10(FDR-adjusted p)")
            ax.set_title("Volcano Plot")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(plots_dir / "volcano_plot.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
        # Manhattan plot
        try:
            fig, ax = plt.subplots(figsize=(8, 3.8))
            y = -np.log10(np.clip(p_adj, eps, 1.0))
            ax.scatter(wl[~sig], y[~sig], s=8, c="#1f78b4", alpha=0.85, label="Not significant")
            ax.scatter(wl[sig], y[sig], s=10, c="#e31a1c", alpha=0.9, label="Significant (FDR)")
            # Reference line at q = alpha (default 0.05)
            alpha_thr = 0.05
            try:
                mt = cr.get("multiple_testing", {}) or {}
                if mt.get("enabled") and mt.get("alpha") is not None:
                    alpha_thr = float(mt.get("alpha"))
            except Exception:
                pass
            thr_y = -np.log10(np.clip(alpha_thr, eps, 1.0))
            ax.axhline(thr_y, color="#636363", linestyle="--", linewidth=1.0, label=f"q={alpha_thr:g}")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("-log10(FDR-adjusted p)")
            ax.set_title("Manhattan Spectrum Significance")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(plots_dir / "manhattan_plot.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass
        # Permutation summary
        try:
            pt = cr.get("permutation_test", {}) or {}
            if pt.get("enabled"):
                fig, ax = plt.subplots(figsize=(6.4, 3.2))
                obs = float(pt.get("observed_abs_slope", np.nan))
                pct = pt.get("percentiles", {}) or {}
                for lab, col in [("5th", "#1f77b4"), ("50th", "#2ca02c"), ("95th", "#ff7f0e")]:
                    val = pct.get(lab.split("th")[0], None)
                    if val is not None:
                        try:
                            ax.axvline(float(val), color=col, linestyle="--", label=f"{lab}")
                        except Exception:
                            pass
                if np.isfinite(obs):
                    ax.axvline(obs, color="#d62728", linestyle="-", label="Observed |slope|")
                ax.set_xlabel("|slope| (ΔT/ppm)")
                ax.set_yticks([])
                ax.set_title("Permutation Summary (percentiles & observed)")
                ax.legend(loc="upper right")
                fig.tight_layout()
                fig.savefig(plots_dir / "permutation_summary.png", dpi=150)
                plt.close(fig)
        except Exception:
            pass
    except Exception:
        pass


def _methods_md(run_dir: Path) -> str:
    md = []
    md.append(_md_h2("Methods and Step-by-Step Justification"))
    md.append(
        "We describe each stage of the pipeline with rationale, equations, and references.\n\n"
        "- Data aggregation: Stable windows are averaged per concentration to reduce noise while preserving analyte response.\n"
        "- Preprocessing: Baseline and smoothing (e.g., Savitzky–Golay) balance bias–variance to improve slope detectability [7].\n"
        "- Univariate scans: For each wavelength $\\lambda$, linear regression $y=\\beta_0+\\beta_1 c+\\epsilon$ gives effect $\\beta_1$, $R^2$, and $p$-value.\n"
        "- Multiple testing: FDR (BH) controls expected false discoveries across wavelengths [1]. Adjusted p-values $q_i=\\min_j \\frac{m}{j} p_{(j)}$ enforce monotonicity.\n"
        "- Permutation testing: Under the null (no association), permuting concentrations yields an empirical distribution for $|\\hat{\\beta}_1|$; $p=\\frac{1+\\#\\{b: |\\hat{\\beta}_1^{(b)}|\\ge |\\hat{\\beta}_1|\\}}{B+1}$ [2].\n"
        "- Hybrid ROI scoring: Combine $R^2$, |slope|, derivative/ratio cues, and slope-to-noise constraints to localize physically consistent bands.\n"
        "- Uncertainty quantification: Per-wavelength $\\hat{\\beta}_1$ 95% CI via $t$-critical and $\\mathrm{SE}(\\hat{\\beta}_1)$; ROI performance CIs via bootstrap percentiles (2.5, 50, 97.5).\n"
        "- LOD/LOQ: IUPAC approximations $\\mathrm{LOD}=3\\sigma/|\\beta_1|$, $\\mathrm{LOQ}=10\\sigma/|\\beta_1|$ where $\\sigma$ is global noise [8].\n"
        "- Multivariate (PLSR): Repeated CV with the one-standard-error rule prevents overfitting; residual diagnostics assess assumptions [3,4].\n"
        "- Deconvolution (ICA/MCR-ALS): Component extraction with robustness checks (ICASSO-like stability, SIMPLISMA/EFA inits) [5,6].\n\n"
    )
    return "".join(md)


def _references_md() -> str:
    md = []
    md.append(_md_h2("References"))
    refs = [
        "[1] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. JRSS B.",
        "[2] Efron, B., & Tibshirani, R. (1994). An Introduction to the Bootstrap. CRC.",
        "[3] Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression. Chemometrics and Intelligent Lab Systems.",
        "[4] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.",
        "[5] Hyvärinen, A., & Oja, E. (2000). Independent component analysis: algorithms and applications. Neural Networks.",
        "[6] Tauler, R. (1995). Multivariate curve resolution applied to second order data. Chemometrics and Intelligent Lab Systems.",
        "[7] Savitzky, A., & Golay, M. J. E. (1964). Smoothing and differentiation. Analytical Chemistry.",
        "[8] IUPAC. (1997). Nomenclature in analytical chemistry: detection limit definitions.",
    ]
    md.append("\n".join([f"- {r}" for r in refs]))
    md.append("\n")
    return "".join(md)


def _recommendations_md(run_dir: Path) -> str:
    metrics_dir = run_dir / "metrics"
    md = []
    md.append(_md_h2("Recommendations and Next Steps"))

    qc = _read_json(metrics_dir / "qc_summary.json") or {}
    mv = _read_json(metrics_dir / "multivariate_selection.json") or {}
    dyn = _read_json(metrics_dir / "dynamics_summary.json") or {}

    # Derive an ROI hint from top STN wavelengths (first 10)
    roi_hint = None
    try:
        import pandas as pd  # type: ignore
        tw_path = metrics_dir / "top_wavelengths.csv"
        if tw_path.exists():
            tw = pd.read_csv(tw_path)
            if not tw.empty and "wavelength" in tw.columns:
                top = tw.head(10)["wavelength"].astype(float)
                roi_hint = (float(top.min()), float(top.max()))
    except Exception:
        pass

    snr_msg = None
    try:
        med_snr = qc.get("median_snr", None)
        thr = qc.get("snr_threshold", None)
        if med_snr is not None and thr is not None:
            snr_msg = f"Median SNR ≈ {float(med_snr):.2f} vs threshold {float(thr):.0f}."
    except Exception:
        pass

    best_method = mv.get("best_method", None)

    bullets = []
    if snr_msg:
        bullets.append(f"Improve SNR before claiming tight LOD/LOQ. {snr_msg} Consider optics shielding, thermal control, longer averaging, and illumination alignment.")
    else:
        bullets.append("Improve SNR via optics shielding, temperature control, longer averaging, and illumination alignment.")

    if roi_hint:
        bullets.append(f"Anchor ROI using top STN wavelengths (approx. {roi_hint[0]:.1f}–{roi_hint[1]:.1f} nm). Validate with derivative+SNV and baseline settings.")
    else:
        bullets.append("Use top_wavelengths.csv to anchor ROI center and bandwidth; validate preprocessing (derivative+SNV, baseline).")

    if best_method:
        bullets.append(f"Multivariate: prioritize {best_method.upper()} based on CV R²; keep PLSR as exploratory with one-SE rule and CV ribbons.")
    else:
        bullets.append("Multivariate: compare MCR-ALS, PLSR, ICA; use repeated CV with one-SE rule and report ±1 SE ribbons.")

    try:
        ov = (dyn or {}).get("overall", {})
        t90 = ov.get("mean_T90", None)
        t10 = ov.get("mean_T10", None)
        if t90 is not None and t10 is not None:
            bullets.append(f"Kinetics: T90 ≈ {float(t90):.0f}s, T10 ≈ {float(t10):.0f}s. Set dwell/purge times accordingly; consider preconditioning.")
    except Exception:
        pass

    bullets.extend([
        "P3: Control multiple comparisons in ROI scanning (FDR/Benjamini–Hochberg) and add permutation tests for ROI significance.",
        "P4: Bootstrap CIs for ROI center, slopes, STN, and LOD/LOQ (IUPAC 3σ/10σ and regression-based).",
        "P5: ICA robustness (ICASSO-like stability) and export stability plots/metrics.",
        "P6: MCR-ALS improvements (SIMPLISMA/EFA init; optional unimodality/closure).",
        "P7: Residual diagnostics (normality, heteroscedasticity, autocorrelation) and influence; label outliers in report.",
        "P8: Reproducibility—save git commit, pip freeze, config snapshot, random seeds; add CITATION.cff and LICENSE review.",
        "P9: Expand metrics exports (candidate ROI table, CV fold predictions, mixing/scores matrices, per-wavelength slopes).",
        "P10: CI and tests—unit tests for ROI selection, PLSR CV, heatmap, deconvolution; set up CI workflow.",
    ])

    # Δλ (wavelength‑shift) specific actions
    bullets.extend([
        "Δλ: Enable sliding‑window sub‑pixel xcorr scan and tune parameters for tiny shifts (see YAML snippet below).",
        "Δλ: Increase upsample (30–50), reduce step_nm (0.5–1.0), and enable derivative to emphasize line position.",
        "Δλ: Apply light smoothing (Savitzky–Golay, window≈7, poly=2) to stabilize cross‑correlation.",
        "Δλ: Use BH‑FDR (α≈0.05) and a min |slope| guard (e.g., 1e-4 nm/ppm) when ranking windows.",
        "Δλ: Review shift_manhattan.png and shift_scan.csv; prioritize windows with high R²_w, |slope_w|, and small SD.",
        "Δλ: If R²_w remains low, try narrower/wider window_nm (e.g., 5/10/20 nm), raise upsample, or enable derivative.",
        "Δλ: Next upgrades—add global spectral alignment before local Δλ, monotonicity (Kendall τ/Spearman ρ), and robust/CV selection (Huber/Theil–Sen, LOOCV).",
        "Consensus band: intersect intensity‑significant and Δλ‑significant regions; report both calibrations side‑by‑side.",
    ])

    md.append(_md_list(bullets))

    # Config snippet for Δλ tuning
    md.append("\nRecommended YAML (place under roi.shift):\n\n")
    md.append("""
```yaml
roi:
  shift:
    scan_enabled: true
    window_nm: [5.0, 10.0, 20.0]
    step_nm: 0.5
    upsample: 30
    use_derivative: true
    smooth_window: 7
    smooth_poly: 2
    fdr_alpha: 0.05
    min_slope_nm_per_ppm: 1e-4
```
""")

    return "".join(md)
def build_report(run_dir: Path, title: Optional[str] = None, output_name: str = "report_full") -> Path:
    run_dir = run_dir.resolve()
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Title + summary.md
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = title or f"Gas Analysis Report - {run_dir.name}"

    summary_md_path = reports_dir / "summary.md"
    summary_md = _read_text(summary_md_path)
    if summary_md:
        summary_md = _annotate_summary_images(summary_md, run_dir)
    else:
        summary_md = f"# {title}\n\n(Generated at {now})\n\nSummary not found; continuing with appendix.\n"

    # Ensure significance figures exist
    _ensure_significance_plots(run_dir)
    _ensure_plsr_calibration(run_dir)

    # Compose sections
    plain = _plain_language_md(run_dir)
    key_findings = _key_findings_md(run_dir)
    units = _units_md()
    methods = _methods_md(run_dir)
    technical = _technical_justification_md(run_dir)
    tech_extra = _technical_appendix_md(run_dir)
    interpretation = _interpretation_md(run_dir)
    assumptions = _assumptions_md()
    recommendations = _recommendations_md(run_dir)
    appendix = _appendix_md(run_dir)
    glossary = _glossary_md()

    # Combine
    combined_md = []
    if not summary_md.strip().startswith("# "):
        combined_md.append(f"# {title}\n\nGenerated at {now}.\n")
    combined_md.append(summary_md.strip())
    combined_md.append("\n\n## Contents\n\n[TOC]\n")
    combined_md.append("\n\n---\n")
    combined_md.append(plain)
    combined_md.append("\n\n---\n")
    combined_md.append(key_findings)
    combined_md.append("\n\n---\n")
    combined_md.append(units)
    combined_md.append("\n\n---\n")
    combined_md.append(methods)
    combined_md.append("\n\n---\n")
    combined_md.append(technical)
    combined_md.append("\n\n---\n")
    combined_md.append(tech_extra)
    combined_md.append("\n\n---\n")
    combined_md.append(interpretation)
    combined_md.append("\n\n---\n")
    combined_md.append(assumptions)
    combined_md.append("\n\n---\n")
    combined_md.append(recommendations)
    combined_md.append("\n\n---\n")
    combined_md.append(_md_h2("Appendix"))
    combined_md.append(appendix)
    combined_md.append("\n\n---\n")
    combined_md.append(glossary)
    combined_md.append("\n\n---\n")
    combined_md.append(_references_md())

    combined_md_text = "\n".join(combined_md)

    out_md = reports_dir / f"{output_name}.md"
    out_html = reports_dir / f"{output_name}.html"
    out_pdf = reports_dir / f"{output_name}.pdf"

    out_md.write_text(combined_md_text, encoding="utf-8")

    # HTML
    html_text = _markdown_to_html(combined_md_text)
    out_html.write_text(html_text, encoding="utf-8")

    # Try PDF (Pandoc), then Playwright (Chromium), then WeasyPrint; otherwise, leave HTML only
    pdf_ok = _write_pdf_from_markdown(out_md, out_pdf)
    if not pdf_ok:
        pdf_ok = _write_pdf_with_playwright(out_html, out_pdf)
    if not pdf_ok:
        pdf_ok = _write_pdf_from_html(out_html, out_pdf, base_url=reports_dir)

    if not pdf_ok:
        sys.stderr.write(
            "PDF generation failed. Options: \n"
            " - Install Pandoc + a TeX engine (e.g., MiKTeX) for pypandoc PDF.\n"
            " - Or install WeasyPrint native deps (Cairo/Pango/GDK-PixBuf).\n"
            " - Or install Playwright Chromium: pip install playwright && python -m playwright install chromium.\n"
            "HTML report has been created.\n"
        )

    return out_pdf if out_pdf.exists() else out_html


def main():
    parser = argparse.ArgumentParser(description="Build PDF/HTML report for a gas analysis run directory.")
    parser.add_argument("run_dir", type=str, help="Path to the run output directory (e.g., output/etoh_topavg)")
    parser.add_argument("--title", type=str, default=None, help="Report title override")
    parser.add_argument("--output-name", type=str, default="report_full", help="Base name for the output report")
    args = parser.parse_args()

    run_path = Path(args.run_dir)
    if not run_path.exists():
        print(f"Run directory not found: {run_path}", file=sys.stderr)
        sys.exit(1)

    out_path = build_report(run_path, title=args.title, output_name=args.output_name)
    print(f"Report written to: {out_path}")


if __name__ == "__main__":
    main()
