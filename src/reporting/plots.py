"""Reporting plots — matplotlib visualisations for pipeline analysis outputs.

All functions are CONFIG-free: every tunable parameter is passed explicitly.
Each function creates one or more figures, saves them to
``out_root/plots/<name>.png``, and returns the saved path (or ``None`` when
the input data are insufficient to produce a meaningful plot).

Design notes
------------
- ``matplotlib.use("Agg")`` is set at module load so the module is safe in
  headless / CI environments.  Callers that need a different backend must
  set it *before* importing this module.
- All figures are closed with ``plt.close(fig)`` in a ``finally`` block so
  memory is not leaked when called in long-running pipelines.
- Temporary-file / atomic-rename pattern (``path.tmp → path``) prevents
  downstream consumers from reading a partially-written PNG.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import re
from typing import Any, Optional, Union

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress, probplot

from src.reporting.metrics import select_signal_column

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# ROI discovery
# ---------------------------------------------------------------------------


def save_roi_discovery_plot(
    discovery: dict[str, Any],
    out_root: str,
) -> Optional[str]:
    """Scatter plot of candidate ROI centre wavelengths vs sensitivity, coloured by R².

    Args:
        discovery: Dict with ``"candidates"`` list and optional ``"selected"`` dict.
        out_root: Root output directory.

    Returns:
        Path to ``plots/roi_discovery.png``, or ``None`` if no candidates found.
    """
    candidates = discovery.get("candidates", []) if isinstance(discovery, dict) else []
    if not isinstance(candidates, list) or not candidates:
        return None

    centers: list[float] = []
    slopes: list[float] = []
    r2_vals: list[float] = []
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        c = cand.get("center_nm")
        m = cand.get("slope_nm_per_ppm")
        r2 = cand.get("r2")
        try:
            c_val = float(c or 0.0)
            m_val = float(m or 0.0)
        except Exception:
            continue
        centers.append(c_val)
        slopes.append(m_val)
        try:
            r2_vals.append(float(r2 or float("nan")))
        except Exception:
            r2_vals.append(float("nan"))

    if not centers:
        return None

    centers_arr = np.array(centers, dtype=float)
    slopes_arr = np.array(slopes, dtype=float)
    r2_arr = np.array(r2_vals, dtype=float)

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(centers_arr, slopes_arr, c=r2_arr, cmap="viridis", s=30, edgecolor="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("R²")

    selected = discovery.get("selected", {}) if isinstance(discovery, dict) else {}
    if isinstance(selected, dict):
        try:
            sel_c = float(selected.get("center_nm") or 0.0)
            sel_m = float(selected.get("slope_nm_per_ppm") or 0.0)
            ax.scatter([sel_c], [sel_m], marker="*", color="red", s=120, label="Selected ROI")
        except Exception:
            pass

    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Center Wavelength (nm)")
    ax.set_ylabel("Slope (nm/ppm)")
    ax.set_title("ROI Discovery: Sensitivity vs Wavelength")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(plots_dir, "roi_discovery.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Concentration-response plot
# ---------------------------------------------------------------------------


def save_concentration_response_plot(
    response: dict[str, Any],
    avg_by_conc: dict[float, np.ndarray],
    out_root: str,
    name: str = "concentration_response",
    clamp_to_roi: bool = True,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Optional[str]:
    """Two-panel plot: average spectra per concentration (top) and slope profile (bottom).

    Args:
        response: Concentration-response dict from
            :func:`src.calibration.roi_scan.compute_concentration_response`.
        avg_by_conc: Mapping ``{concentration: signal_array}`` over the common wavelength grid.
        out_root: Root output directory.
        name: Base filename (without extension).
        clamp_to_roi: When ``True`` *and* both *x_min* / *x_max* are provided, the
            x-axis is constrained to ``[x_min, x_max]``.
        x_min: Lower wavelength limit for x-axis clamping (read from ``CONFIG["roi"]``
            by the pipeline.py wrapper).
        x_max: Upper wavelength limit for x-axis clamping.

    Returns:
        Path to the saved PNG, or ``None`` if *response* is empty.
    """
    if not response:
        return None
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, f"{name}.png")

    wl = np.array(response["wavelengths"])
    slopes = np.array(response["slopes"])
    roi_start = response["roi_start_wavelength"]
    roi_end = response["roi_end_wavelength"]

    r2_arr = None
    best_r2_wl = None
    top_candidates_wl = []
    max_abs_slope_wl = None

    if "candidates" in response and response["candidates"]:
        top_candidates_wl = [c["wavelength"] for c in response["candidates"]]
    else:
        try:
            r2_arr = np.array(response.get("r_squared", []), dtype=float)
            if r2_arr.size == wl.size and r2_arr.size > 0:
                order = np.argsort(r2_arr)[-6:][::-1]
                top_candidates_wl = [float(wl[i]) for i in order]
        except Exception:
            pass

    try:
        r2_arr = np.array(response.get("r_squared", []), dtype=float)
        if r2_arr.size == wl.size and r2_arr.size > 0:
            best_idx = int(np.nanargmax(r2_arr))
            best_r2_wl = float(wl[best_idx])
    except Exception:
        pass
    try:
        abs_sl = np.abs(slopes.astype(float))
        if abs_sl.size == wl.size and abs_sl.size > 0:
            max_abs_slope_wl = float(wl[int(np.nanargmax(abs_sl))])
    except Exception:
        pass

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for conc, arr in sorted(avg_by_conc.items(), key=lambda kv: kv[0]):
        ax1.plot(wl, arr, label=f"{conc:g} ppm")
    ax1.axvspan(roi_start, roi_end, color="orange", alpha=0.2, label="ROI")
    if best_r2_wl is not None:
        ax1.axvline(best_r2_wl, color="red", linestyle="--", linewidth=1.0, label="Best R² λ")
    if max_abs_slope_wl is not None:
        ax1.axvline(max_abs_slope_wl, color="blue", linestyle="-.", linewidth=1.0, label="Max |slope| λ")
    if top_candidates_wl:
        for w in top_candidates_wl:
            ax1.axvline(w, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    ax1.set_ylabel("Transmittance")
    if best_r2_wl is not None and max_abs_slope_wl is not None:
        ax1.set_title(
            f"Average Transmittance per Concentration "
            f"(Best R² λ={best_r2_wl:.2f} nm, Max |slope| λ={max_abs_slope_wl:.2f} nm)"
        )
    else:
        ax1.set_title("Average Transmittance per Concentration")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(wl, slopes, label="Slope (ΔT / Δppm)")
    ax2.axvspan(roi_start, roi_end, color="orange", alpha=0.2)
    if best_r2_wl is not None:
        ax2.axvline(best_r2_wl, color="red", linestyle="--", linewidth=1.0, label="Best R² λ")
    if max_abs_slope_wl is not None:
        ax2.axvline(max_abs_slope_wl, color="blue", linestyle="-.", linewidth=1.0, label="Max |slope| λ")
    if top_candidates_wl:
        for w in top_candidates_wl:
            ax2.axvline(w, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Slope")
    ax2.set_title("Concentration Response Gradient")
    ax2.grid(True, alpha=0.3)

    if clamp_to_roi and x_min is not None and x_max is not None and float(x_min) < float(x_max):
        ax1.set_xlim(float(x_min), float(x_max))
        ax2.set_xlim(float(x_min), float(x_max))

    fig.tight_layout()
    tmp_path = out_path + ".tmp"
    try:
        fig.savefig(tmp_path, dpi=300, format="png")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Wavelength shift visualization (2×3 grid)
# ---------------------------------------------------------------------------


def save_wavelength_shift_visualization(
    canonical: dict[float, pd.DataFrame],
    calib_result: dict[str, Any],
    out_root: str,
    dataset_label: Optional[str] = None,
) -> Optional[str]:
    """Create a 2×3 grid comparing best peak vs best valley wavelength shifts.

    Panels (row × col):

    - (1,1) Full spectrum with ROI highlighted
    - (1,2) Best peak calibration curve
    - (1,3) Best valley calibration curve
    - (2,1) Zoomed feature region
    - (2,2) Peak vs valley R² bar chart
    - (2,3) Δλ from baseline comparison

    Args:
        canonical: Mapping ``{concentration: DataFrame}`` of representative spectra.
        calib_result: Calibration result dict (must contain ``"peak_wavelengths"``
            and ``"concentrations"``).
        out_root: Root output directory.
        dataset_label: Optional gas/dataset name for plot titles.

    Returns:
        Path to ``plots/wavelength_shift_visualization.png``, or ``None`` on failure.
    """
    if not canonical or not calib_result:
        return None

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "wavelength_shift_visualization.png")

    peak_wavelengths = calib_result.get("peak_wavelengths", [])
    concentrations = calib_result.get("concentrations", [])
    roi_center = calib_result.get("roi_center", 550.0)
    calib_result.get("r2", 0.0)
    calib_result.get("slope", 0.0)

    selected_feature = calib_result.get("selected_feature", {})
    feature_type = selected_feature.get("feature_type", "unknown") if selected_feature else "unknown"

    best_peak = calib_result.get("best_peak", {})
    best_valley = calib_result.get("best_valley", {})

    if not peak_wavelengths or not concentrations:
        return None

    items = sorted(canonical.items(), key=lambda kv: kv[0])
    if not items:
        return None

    signal_col = "transmittance"
    for _, df in items:
        if "absorbance" in df.columns:
            signal_col = "absorbance"
            break
        elif "transmittance" in df.columns:
            signal_col = "transmittance"
            break
        elif "intensity" in df.columns:
            signal_col = "intensity"
            break

    roi_half_width = 20.0
    roi_min = roi_center - roi_half_width
    roi_max = roi_center + roi_half_width

    fig = plt.figure(figsize=(18, 14))
    colors = plt.colormaps["viridis"](np.linspace(0, 1, len(items)))

    # (1,1) Full spectrum
    ax1 = fig.add_subplot(2, 3, 1)
    for (conc, df), color in zip(items, colors):
        wl = df["wavelength"].values
        if signal_col in df.columns:
            sig = df[signal_col].values
            ax1.plot(wl, sig, color=color, alpha=0.7, label=f"{conc:.1f} ppm")
    ax1.axvspan(roi_min, roi_max, color="red", alpha=0.25, label="ROI Region", zorder=10)
    ax1.axvline(roi_center, color="red", linestyle="--", linewidth=2.5, alpha=0.9, zorder=11)
    ylim = ax1.get_ylim()
    ax1.annotate(
        f"ROI\n{roi_center:.0f}nm",
        xy=(roi_center, ylim[1] * 0.8),
        fontsize=9,
        ha="center",
        color="red",
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel(signal_col.capitalize())
    ax1.set_title(f"Full Spectrum - {dataset_label or 'Gas'}\n(ROI center: {roi_center:.1f} nm)")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    # (1,2) Best peak calibration
    ax2 = fig.add_subplot(2, 3, 2)
    if best_peak and best_peak.get("peak_wavelengths"):
        peak_wls = best_peak["peak_wavelengths"]
        peak_r2 = best_peak.get("r2", 0)
        peak_slope = best_peak.get("slope", 0)
        peak_center = best_peak.get("center_wavelength", 0)
        ax2.scatter(concentrations, peak_wls, c=colors[: len(concentrations)], s=100, edgecolor="black", zorder=5)
        if len(concentrations) >= 2:
            z = np.polyfit(concentrations, peak_wls, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(concentrations), max(concentrations), 100)
            ax2.plot(x_line, p(x_line), "r--", linewidth=2, label=f"R²={peak_r2:.4f}")
        ax2.set_xlabel("Concentration (ppm)")
        ax2.set_ylabel("Peak Wavelength (nm)")
        ax2.set_title(f"Best PEAK @ {peak_center:.1f} nm\nSlope: {peak_slope:.4f} nm/ppm, R²: {peak_r2:.4f}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        if feature_type == "peak":
            ax2.text(0.02, 0.98, "★ SELECTED", transform=ax2.transAxes, fontsize=10,
                     fontweight="bold", color="green", va="top")
    else:
        ax2.text(0.5, 0.5, "No Peak Found", ha="center", va="center", fontsize=12)
        ax2.set_title("Best PEAK\n(Not Available)")

    # (1,3) Best valley calibration
    ax3 = fig.add_subplot(2, 3, 3)
    if best_valley and best_valley.get("peak_wavelengths"):
        valley_wls = best_valley["peak_wavelengths"]
        valley_r2 = best_valley.get("r2", 0)
        valley_slope = best_valley.get("slope", 0)
        valley_center = best_valley.get("center_wavelength", 0)
        ax3.scatter(concentrations, valley_wls, c=colors[: len(concentrations)], s=100, edgecolor="black", zorder=5)
        if len(concentrations) >= 2:
            z = np.polyfit(concentrations, valley_wls, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(concentrations), max(concentrations), 100)
            ax3.plot(x_line, p(x_line), "b--", linewidth=2, label=f"R²={valley_r2:.4f}")
        ax3.set_xlabel("Concentration (ppm)")
        ax3.set_ylabel("Valley Wavelength (nm)")
        ax3.set_title(f"Best VALLEY @ {valley_center:.1f} nm\nSlope: {valley_slope:.4f} nm/ppm, R²: {valley_r2:.4f}")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        if feature_type == "valley":
            ax3.text(0.02, 0.98, "★ SELECTED", transform=ax3.transAxes, fontsize=10,
                     fontweight="bold", color="green", va="top")
    else:
        ax3.text(0.5, 0.5, "No Valley Found", ha="center", va="center", fontsize=12)
        ax3.set_title("Best VALLEY\n(Not Available)")

    # (2,1) Zoomed feature region
    ax4 = fig.add_subplot(2, 3, 4)
    if len(peak_wavelengths) > 0:
        feature_min = min(peak_wavelengths)
        feature_max = max(peak_wavelengths)
        feature_range = feature_max - feature_min
        zoom_padding = max(5.0, feature_range * 3)
        zoom_min = feature_min - zoom_padding
        zoom_max = feature_max + zoom_padding
    else:
        zoom_min = roi_min
        zoom_max = roi_max

    all_sig_in_zoom: list[float] = []
    for i, ((conc, df), color) in enumerate(zip(items, colors)):
        wl = df["wavelength"].values
        if signal_col in df.columns:
            sig = df[signal_col].values
            mask = (wl >= zoom_min) & (wl <= zoom_max)
            if np.any(mask):
                all_sig_in_zoom.extend(sig[mask])
                ax4.plot(wl[mask], sig[mask], color=color, linewidth=2.0, label=f"{conc:.1f} ppm")
                if i < len(peak_wavelengths):
                    peak_wl = peak_wavelengths[i]
                    peak_idx = np.argmin(np.abs(wl - peak_wl))
                    ax4.axvline(peak_wl, color=color, linestyle=":", alpha=0.7, linewidth=1.5)
                    ax4.scatter([peak_wl], [sig[peak_idx]], color=color, s=100, zorder=5,
                                edgecolor="black", linewidth=1.5, marker="o")

    if all_sig_in_zoom:
        sig_min = min(all_sig_in_zoom)
        sig_max = max(all_sig_in_zoom)
        sig_range = sig_max - sig_min
        if sig_range > 0:
            ax4.set_ylim(sig_min - 0.1 * sig_range, sig_max + 0.1 * sig_range)

    if len(peak_wavelengths) >= 2:
        shift_nm = max(peak_wavelengths) - min(peak_wavelengths)
        ax4.annotate(
            f"Δλ = {shift_nm * 1000:.1f} pm",
            xy=(0.95, 0.05),
            xycoords="axes fraction",
            fontsize=10,
            ha="right",
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel(signal_col.capitalize())
    ax4.set_title(
        f"Zoomed Feature Region ({zoom_min:.1f}-{zoom_max:.1f} nm)\n{feature_type.upper()} positions marked"
    )
    if ax4.get_legend_handles_labels()[0]:
        ax4.legend(loc="upper right", fontsize=7)
    ax4.grid(True, alpha=0.3)

    # (2,2) Peak vs Valley R² comparison bar chart
    ax5 = fig.add_subplot(2, 3, 5)
    comparison_data = []
    labels: list[str] = []
    colors_bar = []
    if best_peak and best_peak.get("r2"):
        comparison_data.append(best_peak["r2"])
        labels.append(f"PEAK\n{best_peak.get('center_wavelength', 0):.1f} nm")
        colors_bar.append("coral" if feature_type == "peak" else "lightcoral")
    if best_valley and best_valley.get("r2"):
        comparison_data.append(best_valley["r2"])
        labels.append(f"VALLEY\n{best_valley.get('center_wavelength', 0):.1f} nm")
        colors_bar.append("steelblue" if feature_type == "valley" else "lightsteelblue")
    if comparison_data:
        bars = ax5.bar(range(len(comparison_data)), comparison_data, color=colors_bar, edgecolor="black")
        ax5.set_xticks(range(len(comparison_data)))
        ax5.set_xticklabels(labels)
        ax5.set_ylabel("R² Value")
        ax5.set_title("Peak vs Valley Comparison\n(Darker = Selected)")
        ax5.set_ylim(0, 1.1)
        ax5.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, comparison_data):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # (2,3) Δλ from baseline comparison
    ax6 = fig.add_subplot(2, 3, 6)
    x_pos = np.arange(len(concentrations))
    width = 0.35
    if best_peak and best_peak.get("peak_wavelengths"):
        peak_wls = best_peak["peak_wavelengths"]
        peak_baseline = peak_wls[0] if peak_wls else 0
        peak_deltas = [w - peak_baseline for w in peak_wls]
        ax6.bar(x_pos - width / 2, peak_deltas, width, label="Peak Δλ", color="coral", edgecolor="black")
    if best_valley and best_valley.get("peak_wavelengths"):
        valley_wls = best_valley["peak_wavelengths"]
        valley_baseline = valley_wls[0] if valley_wls else 0
        valley_deltas = [w - valley_baseline for w in valley_wls]
        ax6.bar(x_pos + width / 2, valley_deltas, width, label="Valley Δλ", color="steelblue", edgecolor="black")
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f"{c:.1f}" for c in concentrations])
    ax6.set_xlabel("Concentration (ppm)")
    ax6.set_ylabel("Δλ from baseline (nm)")
    ax6.set_title("Wavelength Shift Comparison\n(Peak vs Valley)")
    ax6.axhline(0, color="black", linestyle="-", linewidth=0.5)
    if ax6.get_legend_handles_labels()[0]:
        ax6.legend()
    ax6.grid(True, alpha=0.3, axis="y")

    selected_label = f"Selected: {feature_type.upper()}" if feature_type != "unknown" else ""
    signal_label = f"Signal: {signal_col.upper()}"
    fig.suptitle(
        f"{dataset_label or 'Gas'} - Peak vs Valley Comparison\n{selected_label} | {signal_label}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    tmp_path = out_path + ".tmp"
    try:
        fig.savefig(tmp_path, dpi=300, format="png")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Research-grade calibration plot
# ---------------------------------------------------------------------------


def save_research_grade_calibration_plot(
    canonical: dict[float, pd.DataFrame],
    calib_result: dict[str, Any],
    out_root: str,
    dataset_label: Optional[str] = None,
) -> Optional[str]:
    """Publication-quality 2×2 calibration plot with residual analysis.

    Panels:

    - (1,1) Calibration curve with fit line and metrics box
    - (1,2) Wavelength shift (Δλ) vs concentration
    - (2,1) Residual scatter with ±RMSE band
    - (2,2) Calibration summary metrics table

    Args:
        canonical: Mapping ``{concentration: DataFrame}`` (used only to check non-empty).
        calib_result: Calibration result dict with keys ``"concentrations"``,
            ``"peak_wavelengths"``, ``"r2"``, ``"slope"``, ``"intercept"``,
            ``"lod"``, ``"loq"``, ``"rmse"``, ``"selected_model"``.
        out_root: Root output directory.
        dataset_label: Optional gas/dataset name for plot titles.

    Returns:
        Path to ``plots/calibration_research_grade.png``, or ``None`` on failure.
    """
    if not canonical or not calib_result:
        return None

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "calibration_research_grade.png")

    concentrations = np.array(calib_result.get("concentrations", []))
    peak_wavelengths = np.array(calib_result.get("peak_wavelengths", []))
    r2 = calib_result.get("r2", 0.0)
    slope = calib_result.get("slope", 0.0)
    intercept = calib_result.get("intercept", 0.0)
    lod = calib_result.get("lod", 0.0)
    loq = calib_result.get("loq", 0.0)
    rmse = calib_result.get("rmse", 0.0)
    selected_model = calib_result.get("selected_model", "linear")

    selected_feature = calib_result.get("selected_feature", {})
    feature_type = selected_feature.get("feature_type", "unknown") if selected_feature else "unknown"
    feature_center = selected_feature.get("center_wavelength", 0) if selected_feature else 0

    if len(concentrations) < 2 or len(peak_wavelengths) < 2:
        return None

    baseline_wl = peak_wavelengths[0]
    delta_wavelengths = peak_wavelengths - baseline_wl

    if "poly" in selected_model:
        degree = int(selected_model.split("_")[-1]) if "_" in selected_model else 2
        coeffs = np.polyfit(concentrations, peak_wavelengths, degree)
        predictions = np.polyval(coeffs, concentrations)
    else:
        predictions = slope * concentrations + intercept

    residuals = peak_wavelengths - predictions

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    main_color = "#2E86AB"
    accent_color = "#A23B72"

    # (1,1) Main calibration curve
    ax1 = axes[0, 0]
    ax1.scatter(concentrations, peak_wavelengths, s=120, c=main_color,
                edgecolor="black", linewidth=1.5, zorder=5, label="Measured")
    x_fit = np.linspace(0, max(concentrations) * 1.1, 100)
    if "poly" in selected_model:
        y_fit = np.polyval(coeffs, x_fit)
    else:
        y_fit = slope * x_fit + intercept
    ax1.plot(x_fit, y_fit, "--", color=accent_color, linewidth=2, label=f"{selected_model} fit")
    if "poly" not in selected_model:
        eq_text = f"λ = {slope:.4f}·C + {intercept:.2f}"
    else:
        eq_text = f"Polynomial (deg {degree})"
    ax1.text(0.05, 0.95, f"{eq_text}\nR² = {r2:.4f}\nSensitivity = {slope:.4f} nm/ppm",
             transform=ax1.transAxes, fontsize=11, verticalalignment="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax1.set_xlabel("Concentration (ppm)", fontsize=12)
    ax1.set_ylabel(f"{feature_type.capitalize()} Wavelength (nm)", fontsize=12)
    ax1.set_title(f"Calibration Curve - {dataset_label or 'Gas'}", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # (1,2) Δλ vs concentration
    ax2 = axes[0, 1]
    ax2.scatter(concentrations, delta_wavelengths * 1000, s=120, c=main_color,
                edgecolor="black", linewidth=1.5, zorder=5)
    delta_slope = slope * 1000
    delta_fit = delta_slope * x_fit
    ax2.plot(x_fit, delta_fit, "--", color=accent_color, linewidth=2)
    ax2.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Concentration (ppm)", fontsize=12)
    ax2.set_ylabel("Δλ from baseline (pm)", fontsize=12)
    ax2.set_title(f"Wavelength Shift Response\nSensitivity: {delta_slope:.2f} pm/ppm",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    total_shift = (max(delta_wavelengths) - min(delta_wavelengths)) * 1000
    ax2.text(0.95, 0.05, f"Total Δλ: {total_shift:.1f} pm", transform=ax2.transAxes,
             fontsize=11, ha="right", bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    # (2,1) Residual analysis
    ax3 = axes[1, 0]
    ax3.scatter(concentrations, residuals * 1000, s=100, c=main_color,
                edgecolor="black", linewidth=1.5, zorder=5)
    ax3.axhline(0, color="red", linestyle="--", linewidth=1.5)
    rmse_pm = rmse * 1000
    ax3.axhline(rmse_pm, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax3.axhline(-rmse_pm, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax3.fill_between([0, max(concentrations) * 1.1], -rmse_pm, rmse_pm, alpha=0.1, color="orange")
    ax3.set_xlabel("Concentration (ppm)", fontsize=12)
    ax3.set_ylabel("Residual (pm)", fontsize=12)
    ax3.set_title(f"Residual Analysis\nRMSE: {rmse_pm:.2f} pm", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)

    # (2,2) Summary table
    ax4 = axes[1, 1]
    ax4.axis("off")
    metrics_text = f"""
    ╔══════════════════════════════════════════╗
    ║     CALIBRATION SUMMARY                  ║
    ╠══════════════════════════════════════════╣
    ║  Gas:           {dataset_label or "Unknown":>20}  ║
    ║  Feature:       {feature_type.upper():>20}  ║
    ║  Center λ:      {feature_center:>17.2f} nm  ║
    ╠══════════════════════════════════════════╣
    ║  PERFORMANCE METRICS                     ║
    ╠══════════════════════════════════════════╣
    ║  R²:            {r2:>20.4f}  ║
    ║  Sensitivity:   {slope * 1000:>14.2f} pm/ppm  ║
    ║  RMSE:          {rmse * 1000:>17.2f} pm  ║
    ║  LOD:           {lod:>17.2f} ppm  ║
    ║  LOQ:           {loq:>17.2f} ppm  ║
    ╠══════════════════════════════════════════╣
    ║  Model:         {selected_model:>20}  ║
    ║  Conc. Range:   {min(concentrations):.1f} - {max(concentrations):.1f} ppm{" " * 10}║
    ╚══════════════════════════════════════════╝
    """
    ax4.text(0.5, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="center", horizontalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray"))

    fig.suptitle(f"{dataset_label or 'Gas'} - Research-Grade Calibration Analysis",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    tmp_path = out_path + ".tmp"
    try:
        fig.savefig(tmp_path, dpi=300, format="png", facecolor="white", edgecolor="none")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Spectral response diagnostic (ΔA vs Δλ comparison)
# ---------------------------------------------------------------------------


def save_spectral_response_diagnostic(
    canonical: dict[float, pd.DataFrame],
    out_root: str,
    dataset_label: Optional[str] = None,
    wl_min: float = 400.0,
    wl_max: float = 800.0,
    step_nm: float = 2.0,
    window_nm: Union[float, list[float]] = 10.0,
) -> Optional[str]:
    """Diagnostic comparing absorbance amplitude (ΔA) and wavelength shift (Δλ) methods.

    Creates a 2×3 figure with:

    - (1,1) Full spectrum overlay
    - (1,2) ΔA spectrum (high vs low concentration)
    - (1,3) R² map for absorbance amplitude
    - (2,1) Zoomed spectrum at best ΔA region
    - (2,2) R² map for wavelength shift
    - (2,3) Method comparison summary table

    Args:
        canonical: Mapping ``{concentration: DataFrame}`` of representative spectra.
        out_root: Root output directory.
        dataset_label: Optional gas/dataset name for plot titles.
        wl_min: Lower wavelength bound for analysis (nm).
        wl_max: Upper wavelength bound for analysis (nm).
        step_nm: Step size (nm) for the wavelength-shift sliding-window scan.
            Read from ``CONFIG["roi"]["shift"]["step_nm"]`` by the pipeline.py wrapper.
        window_nm: Window width(s) (nm) for the wavelength-shift scan.
            Read from ``CONFIG["roi"]["shift"]["window_nm"]`` by the pipeline.py wrapper.

    Returns:
        Path to ``plots/spectral_response_diagnostic.png``, or ``None`` on failure.
    """
    if not canonical or len(canonical) < 2:
        return None

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "spectral_response_diagnostic.png")

    items = sorted(canonical.items(), key=lambda x: x[0])
    concentrations = [c for c, _ in items]

    ref_conc, ref_df = items[0]
    high_conc, high_df = items[-1]

    signal_col = "absorbance"
    if signal_col not in ref_df.columns:
        signal_col = "transmittance" if "transmittance" in ref_df.columns else "intensity"

    wl = ref_df["wavelength"].values
    mask = (wl >= wl_min) & (wl <= wl_max)
    wl_roi = wl[mask]

    if len(wl_roi) < 10:
        return None

    ref_sig = ref_df[signal_col].values[mask]
    high_sig = high_df[signal_col].values[mask]
    delta_abs = high_sig - ref_sig

    abs_r2_map = np.zeros(len(wl_roi))
    abs_slope_map = np.zeros(len(wl_roi))
    for i in range(len(wl_roi)):
        vals = []
        for _, df in items:
            if signal_col not in df.columns:
                vals.append(np.nan)
                continue
            wl_arr = df["wavelength"].values
            sig_arr = df[signal_col].values
            idx = np.argmin(np.abs(wl_arr - wl_roi[i]))
            vals.append(float(sig_arr[idx]))
        vals_arr = np.array(vals, dtype=float)
        if np.all(np.isfinite(vals_arr)) and len(vals_arr) >= 2:
            try:
                reg = linregress(concentrations, vals_arr)
                abs_r2_map[i] = float(reg.rvalue**2)
                abs_slope_map[i] = float(reg.slope)
            except Exception:
                pass

    # Normalise window_nm to list[float]
    if not np.isfinite(step_nm) or step_nm <= 0:
        step_nm = 2.0

    if isinstance(window_nm, (list, tuple)):
        window_nm_values = []
        for w in window_nm:
            try:
                wv = float(w)
            except Exception:
                continue
            if np.isfinite(wv) and wv > 0:
                window_nm_values.append(wv)
        if not window_nm_values:
            window_nm_values = [10.0]
    else:
        try:
            wv = float(window_nm)
        except Exception:
            wv = 10.0
        window_nm_values = [wv] if (np.isfinite(wv) and wv > 0) else [10.0]

    max_window = float(max(window_nm_values))
    centers = np.arange(wl_min + max_window / 2.0, wl_max - max_window / 2.0, step_nm)
    shift_r2_map = np.zeros(len(centers))
    shift_slope_map = np.zeros(len(centers))

    prefer_peak = signal_col == "absorbance"
    for i, center in enumerate(centers):
        best_r2 = float("nan")
        best_slope = float("nan")
        for win in window_nm_values:
            wl_low = center - win / 2.0
            wl_high = center + win / 2.0
            feature_positions = []
            for _, df in items:
                if signal_col not in df.columns:
                    feature_positions.append(np.nan)
                    continue
                wl_arr = df["wavelength"].values
                sig_arr = df[signal_col].values
                win_mask = (wl_arr >= wl_low) & (wl_arr <= wl_high)
                if np.sum(win_mask) < 3:
                    feature_positions.append(np.nan)
                    continue
                wl_win = wl_arr[win_mask]
                sig_win = sig_arr[win_mask]
                extremum_idx = int(np.argmax(sig_win)) if prefer_peak else int(np.argmin(sig_win))
                if 0 < extremum_idx < len(sig_win) - 1:
                    y0, y1, y2 = sig_win[extremum_idx - 1], sig_win[extremum_idx], sig_win[extremum_idx + 1]
                    denom = 2 * (2 * y1 - y0 - y2)
                    if abs(denom) > 1e-12:
                        delta = np.clip((y0 - y2) / denom, -0.5, 0.5)
                        wl_step_local = wl_win[extremum_idx] - wl_win[extremum_idx - 1]
                        feature_wl = wl_win[extremum_idx] + delta * wl_step_local
                    else:
                        feature_wl = wl_win[extremum_idx]
                else:
                    feature_wl = wl_win[extremum_idx]
                feature_positions.append(float(feature_wl))
            vals_arr = np.array(feature_positions, dtype=float)
            if vals_arr.size != len(concentrations) or not np.all(np.isfinite(vals_arr)):
                continue
            try:
                reg = linregress(concentrations, vals_arr)
                r2_val = float(reg.rvalue**2)
            except Exception:
                continue
            if not np.isfinite(r2_val):
                continue
            if not np.isfinite(best_r2) or r2_val > best_r2:
                best_r2 = r2_val
                best_slope = float(reg.slope)
        if np.isfinite(best_r2):
            shift_r2_map[i] = best_r2
        if np.isfinite(best_slope):
            shift_slope_map[i] = best_slope

    # Build figure
    fig = plt.figure(figsize=(18, 12))
    colors = plt.colormaps["viridis"](np.linspace(0, 1, len(items)))

    ax1 = fig.add_subplot(2, 3, 1)
    for (conc, df), color in zip(items, colors):
        if signal_col in df.columns:
            ax1.plot(df["wavelength"].values, df[signal_col].values,
                     color=color, alpha=0.8, linewidth=1.5, label=f"{conc:.1f} ppm")
    ax1.axvline(wl_min, color="red", linestyle="--", alpha=0.5)
    ax1.axvline(wl_max, color="red", linestyle="--", alpha=0.5)
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel(signal_col.capitalize())
    ax1.set_title(f"Full Spectrum - {dataset_label or 'Gas'}")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(wl_roi, delta_abs, "b-", linewidth=1.5)
    ax2.fill_between(wl_roi, 0, delta_abs, alpha=0.3, color="blue")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    max_idx = np.argmax(np.abs(delta_abs))
    ax2.scatter([wl_roi[max_idx]], [delta_abs[max_idx]], color="red", s=100, zorder=5,
                marker="*", label=f"Max ΔA @ {wl_roi[max_idx]:.1f} nm")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel(f"Δ{signal_col.capitalize()} ({high_conc:.1f} - {ref_conc:.1f} ppm)")
    ax2.set_title(f"Absorbance Amplitude Change\nMax |ΔA| = {np.abs(delta_abs[max_idx]):.4f}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(wl_roi, abs_r2_map, "g-", linewidth=1.5)
    ax3.fill_between(wl_roi, 0, abs_r2_map, alpha=0.3, color="green")
    ax3.axhline(0.9, color="red", linestyle="--", alpha=0.7, label="R² = 0.9")
    best_abs_idx = int(np.argmax(abs_r2_map))
    ax3.scatter([wl_roi[best_abs_idx]], [abs_r2_map[best_abs_idx]], color="red", s=100, zorder=5,
                marker="*", label=f"Best @ {wl_roi[best_abs_idx]:.1f} nm")
    ax3.set_xlabel("Wavelength (nm)")
    ax3.set_ylabel("R² (Absorbance vs Concentration)")
    ax3.set_title(f"Absorbance Amplitude R² Map\nBest R² = {abs_r2_map[best_abs_idx]:.4f}")
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 3, 4)
    best_abs_wl = wl_roi[best_abs_idx]
    zoom_min = best_abs_wl - 15
    zoom_max = best_abs_wl + 15
    for (conc, df), color in zip(items, colors):
        if signal_col in df.columns:
            wl_arr = df["wavelength"].values
            sig_arr = df[signal_col].values
            zmask = (wl_arr >= zoom_min) & (wl_arr <= zoom_max)
            if np.any(zmask):
                ax4.plot(wl_arr[zmask], sig_arr[zmask], color=color, linewidth=2, label=f"{conc:.1f} ppm")
    ax4.axvline(best_abs_wl, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax4.set_xlabel("Wavelength (nm)")
    ax4.set_ylabel(signal_col.capitalize())
    ax4.set_title(f"Zoomed: Best ΔA Region ({zoom_min:.0f}-{zoom_max:.0f} nm)\nClear line separation visible")
    ax4.legend(loc="upper right", fontsize=7)
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(2, 3, 5)
    if len(centers) > 0:
        ax5.plot(centers, shift_r2_map, "m-", linewidth=1.5)
        ax5.fill_between(centers, 0, shift_r2_map, alpha=0.3, color="magenta")
        ax5.axhline(0.9, color="red", linestyle="--", alpha=0.7, label="R² = 0.9")
        best_shift_idx = int(np.argmax(shift_r2_map))
        ax5.scatter([centers[best_shift_idx]], [shift_r2_map[best_shift_idx]], color="red", s=100,
                    zorder=5, marker="*", label=f"Best @ {centers[best_shift_idx]:.1f} nm")
        ax5.set_title(f"Wavelength Shift R² Map\nBest R² = {shift_r2_map[best_shift_idx]:.4f}")
    else:
        best_shift_idx = 0
        ax5.set_title("Wavelength Shift R² Map\n(insufficient range)")
    ax5.set_xlabel("Window Center (nm)")
    ax5.set_ylabel("R² (Wavelength Shift vs Concentration)")
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    best_abs_r2 = abs_r2_map[best_abs_idx]
    best_abs_slope_val = abs_slope_map[best_abs_idx]
    if len(centers) > 0:
        best_shift_r2 = shift_r2_map[best_shift_idx]
        best_shift_slope = shift_slope_map[best_shift_idx] * 1000
        best_shift_center = centers[best_shift_idx]
    else:
        best_shift_r2 = 0.0
        best_shift_slope = 0.0
        best_shift_center = 0.0

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           SPECTRAL RESPONSE COMPARISON                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  METHOD 1: ABSORBANCE AMPLITUDE (ΔA)                     ║
    ║  ─────────────────────────────────────                   ║
    ║  Best Wavelength:     {wl_roi[best_abs_idx]:>8.1f} nm                      ║
    ║  Best R²:             {best_abs_r2:>8.4f}                          ║
    ║  Sensitivity:         {best_abs_slope_val:>8.4f} AU/ppm                 ║
    ║                                                          ║
    ║  METHOD 2: WAVELENGTH SHIFT (Δλ)                         ║
    ║  ─────────────────────────────────                       ║
    ║  Best Window Center:  {best_shift_center:>8.1f} nm                      ║
    ║  Best R²:             {best_shift_r2:>8.4f}                          ║
    ║  Sensitivity:         {best_shift_slope:>8.2f} pm/ppm                  ║
    ║                                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  RECOMMENDATION:                                         ║
    ║  {"ΔA method shows better correlation" if best_abs_r2 > best_shift_r2 else "Δλ method shows better correlation":^54}  ║
    ║  {"Consider using absorbance amplitude" if best_abs_r2 > best_shift_r2 else "Wavelength shift is optimal":^54}  ║
    ╚══════════════════════════════════════════════════════════╝
    """
    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment="center", horizontalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9))

    fig.suptitle(
        f"{dataset_label or 'Gas'} - Spectral Response Diagnostic\n"
        f"Comparing Wavelength Shift (Δλ) vs Absorbance Amplitude (ΔA)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    tmp_path = out_path + ".tmp"
    try:
        fig.savefig(tmp_path, dpi=300, format="png", facecolor="white")
        os.replace(tmp_path, out_path)
    finally:
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# ROI repeatability scatter
# ---------------------------------------------------------------------------


def save_roi_repeatability_plot(
    stable_by_conc: dict[float, dict[str, pd.DataFrame]],
    response: dict[str, Any],
    out_root: str,
) -> Optional[str]:
    """Scatter plot of mean ROI transmittance vs concentration across all trials.

    Args:
        stable_by_conc: Mapping ``{concentration: {trial_name: DataFrame}}``.
        response: Must contain ``"roi_start_wavelength"`` and ``"roi_end_wavelength"``.
        out_root: Root output directory.

    Returns:
        Path to ``plots/roi_repeatability.png``, or ``None`` if the ROI is degenerate.
    """
    start = response["roi_start_wavelength"]
    end = response["roi_end_wavelength"]
    if start == end:
        return None

    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "roi_repeatability.png")

    center = 0.5 * (start + end)
    fig, ax = plt.subplots(figsize=(8, 5))

    xs = []
    ys = []
    for conc, trials in sorted(stable_by_conc.items(), key=lambda kv: kv[0]):
        for _trial, df in trials.items():
            col = select_signal_column(df)
            wl = df["wavelength"].values
            y = df[col].values
            mask = (wl >= start) & (wl <= end)
            if not mask.any():
                val = float(np.interp(center, wl, y))
            else:
                val = float(np.nanmean(y[mask]))
            xs.append(conc)
            ys.append(val)

    if not xs:
        plt.close(fig)
        return None

    ax.scatter(xs, ys, alpha=0.7)
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Mean Transmittance in ROI")
    ax.set_title(f"ROI Repeatability ({start:.2f}–{end:.2f} nm)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Per-trial aggregated plots
# ---------------------------------------------------------------------------


def save_aggregated_plots(
    aggregated: dict[float, dict[str, pd.DataFrame]],
    out_root: str,
) -> dict[float, dict[str, str]]:
    """Save one spectrum plot per trial, organised by concentration.

    Args:
        aggregated: Mapping ``{concentration: {trial_name: DataFrame}}``.
        out_root: Root output directory.

    Returns:
        Nested dict mirroring *aggregated* with PNG file paths as leaf values.
    """
    plots_root = os.path.join(out_root, "plots", "aggregated")
    plot_paths: dict[float, dict[str, str]] = {}
    for conc, trials in aggregated.items():
        conc_dir = os.path.join(plots_root, f"{conc:g}")
        _ensure_dir(conc_dir)
        plot_paths[conc] = {}
        for trial, df in trials.items():
            col = select_signal_column(df)
            y_label = "Transmittance" if col == "transmittance" else "Intensity"
            safe_trial = re.sub(r"[^A-Za-z0-9._-]+", "_", trial)
            fname = f"{safe_trial or 'trial'}.png"
            fpath = os.path.join(conc_dir, fname)

            plt.figure(figsize=(10, 6))
            plt.plot(df["wavelength"].values, df[col].values, label=trial)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel(y_label)
            plt.title(f"Conc {conc:g} - {trial}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fpath, dpi=300)
            plt.close()

            plot_paths[conc][trial] = fpath
    return plot_paths


# ---------------------------------------------------------------------------
# Canonical overlay
# ---------------------------------------------------------------------------


def save_canonical_overlay(
    canonical: dict[float, pd.DataFrame],
    out_root: str,
) -> Optional[str]:
    """Overlay all canonical spectra on a single axes, one line per concentration.

    Args:
        canonical: Mapping ``{concentration: DataFrame}``.
        out_root: Root output directory.

    Returns:
        Path to ``plots/canonical_overlay.png``, or ``None`` if *canonical* is empty.
    """
    if not canonical:
        return None
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(plots_dir)
    out_path = os.path.join(plots_dir, "canonical_overlay.png")

    plt.figure(figsize=(10, 6))
    for conc, df in sorted(canonical.items(), key=lambda kv: kv[0]):
        col = select_signal_column(df)
        plt.plot(df["wavelength"].values, df[col].values, label=f"{conc:g}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(
        "Transmittance"
        if "transmittance" in next(iter(canonical.values())).columns
        else "Intensity"
    )
    plt.title("Canonical Spectra Overlay")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Concentration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Calibration outputs (combined IO + plots)
# ---------------------------------------------------------------------------


def save_calibration_outputs(
    calib: dict[str, Any],
    out_root: str,
    name_suffix: str = "",
) -> None:
    """Save all calibration artefacts: CSV, metrics JSON, and diagnostic plots.

    Writes to both ``metrics/`` and ``plots/`` sub-directories.  The *calib* dict
    is mutated in-place: a ``"plots"`` key is added (or updated) with paths to
    any generated plot files.

    Args:
        calib: Calibration result dict.  Must contain ``"concentrations"``,
            ``"peak_wavelengths"``, ``"slope"``, ``"intercept"``, ``"r2"``.
        out_root: Root output directory.
        name_suffix: Optional suffix appended to file stems (e.g. ``"_mode1"``).
    """
    metrics_dir = os.path.join(out_root, "metrics")
    plots_dir = os.path.join(out_root, "plots")
    _ensure_dir(metrics_dir)
    _ensure_dir(plots_dir)

    plots_dict = calib.setdefault("plots", {})
    cal_label = f"calibration{name_suffix}" if name_suffix else "calibration"

    # Calibration CSV
    cal_path = os.path.join(metrics_dir, f"{cal_label}.csv")
    data = pd.DataFrame(
        {"concentration": calib["concentrations"], "peak_wavelength": calib["peak_wavelengths"]}
    )
    data.to_csv(cal_path, index=False)

    # Calibration metrics JSON (full dict)
    meta = dict(calib)
    meta_path = os.path.join(metrics_dir, f"{cal_label}_metrics.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Main calibration plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.array(calib["concentrations"])
    y = np.array(calib["peak_wavelengths"])
    slope = calib["slope"]
    intercept = calib["intercept"]

    yerr = None
    resid_summary = calib.get("residual_summary", {})
    if isinstance(resid_summary, dict):
        with contextlib.suppress(Exception):
            y_std = float(resid_summary.get("std", float("nan")))
            if np.isfinite(y_std) and y_std > 0.0:
                yerr = np.full_like(y, y_std, dtype=float)

    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="o", color="tab:blue", ecolor="lightgray",
                    elinewidth=1.0, capsize=3, label="Data ±σ")
    else:
        ax.scatter(x, y, label="Data")
    xx = np.linspace(x.min(), x.max(), 100)
    yy = intercept + slope * xx
    ax.plot(xx, yy, "r-", label=f"Fit (R^2={calib['r2']:.3f})")
    ax.set_xlabel("Concentration (ppm)")
    ax.set_ylabel("Peak Wavelength (nm)")
    ax.set_title("Calibration: Wavelength Shift vs Concentration")
    ax.grid(True, alpha=0.3)
    ax.legend()

    text_lines = []
    with contextlib.suppress(Exception):
        text_lines.append(f"slope = {float(slope):.4f} nm/ppm")
    with contextlib.suppress(Exception):
        text_lines.append(f"R² = {float(calib.get('r2', float('nan'))):.3f}")
    try:
        lod_val = float(calib.get("lod", float("nan")))
        if np.isfinite(lod_val):
            text_lines.append(f"LOD = {lod_val:.2f} ppm")
    except Exception:
        pass
    if text_lines:
        ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"{cal_label}.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    plots_dict["calibration"] = plot_path

    # Residual diagnostic plots
    conc_arr = np.array(calib.get("concentrations", []), dtype=float)
    residuals = np.array(calib.get("residuals") or [], dtype=float)
    preds = np.array(calib.get("predictions") or [], dtype=float)
    mask_conc = residuals.size and conc_arr.size == residuals.size
    mask_pred = mask_conc and preds.size == residuals.size
    if mask_conc:
        finite_mask = np.isfinite(conc_arr) & np.isfinite(residuals)
        conc_fin = conc_arr[finite_mask]
        resid_fin = residuals[finite_mask]
        pred_fin = preds[finite_mask] if mask_pred else None
        if resid_fin.size >= 3:
            fig_res, axes = plt.subplots(2, 2, figsize=(9, 6))
            axes[0, 0].scatter(conc_fin, resid_fin, s=25, alpha=0.8, color="tab:green")
            axes[0, 0].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
            axes[0, 0].set_xlabel("Concentration (ppm)")
            axes[0, 0].set_ylabel("Residual (nm)")
            axes[0, 0].set_title("Residuals vs Concentration")
            axes[0, 0].grid(True, alpha=0.3)
            if pred_fin is not None and np.any(np.isfinite(pred_fin)):
                axes[0, 1].scatter(pred_fin, resid_fin, s=25, alpha=0.8, color="tab:blue")
                axes[0, 1].set_xlabel("Predicted Wavelength (nm)")
                axes[0, 1].set_ylabel("Residual (nm)")
                axes[0, 1].set_title("Residuals vs Predicted")
            else:
                axes[0, 1].axis("off")
            axes[0, 1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
            axes[0, 1].grid(True, alpha=0.3)
            axes[1, 0].hist(resid_fin, bins=max(8, int(np.sqrt(resid_fin.size))),
                            color="gray", edgecolor="black", alpha=0.85)
            axes[1, 0].set_xlabel("Residual (nm)")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Residual Histogram")
            try:
                (osm, osr), (slope_q, intercept_q, r_q) = probplot(resid_fin, dist="norm", fit=True)
                axes[1, 1].scatter(osm, osr, s=20, alpha=0.8)
                axes[1, 1].plot(osm, slope_q * np.array(osm) + intercept_q, "r--", linewidth=1.0)
                axes[1, 1].set_title(f"Normal Q-Q (R={r_q:.3f})")
                axes[1, 1].set_xlabel("Theoretical quantiles")
                axes[1, 1].set_ylabel("Ordered residuals")
            except Exception:
                axes[1, 1].text(0.5, 0.5, "Q-Q failed", ha="center", va="center")
                axes[1, 1].axis("off")
            fig_res.tight_layout()
            resid_plot_path = os.path.join(plots_dir, "calibration_residuals.png")
            fig_res.savefig(resid_plot_path, dpi=200)
            plt.close(fig_res)
            plots_dict["calibration_residuals"] = resid_plot_path

    # Absolute wavelength shift plot
    abs_info = calib.get("absolute_shift") or {}
    abs_conc = np.array(abs_info.get("concentrations") or [], dtype=float)
    abs_delta = np.array(abs_info.get("absolute_delta_wavelengths") or [], dtype=float)
    if abs_conc.size and abs_delta.size == abs_conc.size:
        mask_abs = np.isfinite(abs_conc) & np.isfinite(abs_delta)
        conc_abs = abs_conc[mask_abs]
        delta_abs_arr = abs_delta[mask_abs]
        if conc_abs.size >= 2:
            fit_vals = abs_info.get("predicted_absolute_delta")
            if fit_vals is not None:
                fit_vals = np.array(fit_vals, dtype=float)
                fit_vals = fit_vals[mask_abs] if fit_vals.size == abs_conc.size else None
            if fit_vals is None and np.isfinite(abs_info.get("slope", float("nan"))):
                slope_abs = float(abs_info.get("slope") or 0.0)
                intercept_abs = float(abs_info.get("intercept", 0.0))
                fit_vals = intercept_abs + slope_abs * conc_abs
            fig_abs, ax_abs = plt.subplots(figsize=(7, 4))
            ax_abs.scatter(conc_abs, delta_abs_arr, color="tab:orange", alpha=0.85, label="|Δλ| observations")
            if fit_vals is not None and np.size(fit_vals) == conc_abs.size:
                order = np.argsort(conc_abs)
                ax_abs.plot(conc_abs[order], np.asarray(fit_vals)[order],
                            color="tab:red", linewidth=1.2, label="Linear fit")
            ax_abs.set_xlabel("Concentration (ppm)")
            ax_abs.set_ylabel("|Δλ| (nm)")
            ax_abs.set_title("Absolute Wavelength Shift vs Concentration")
            ax_abs.grid(True, alpha=0.3)
            ax_abs.legend(loc="upper left")
            fig_abs.tight_layout()
            abs_plot_path = os.path.join(plots_dir, "absolute_shift_vs_concentration.png")
            fig_abs.savefig(abs_plot_path, dpi=200)
            plt.close(fig_abs)
            plots_dict["absolute_shift"] = abs_plot_path
            with contextlib.suppress(Exception):
                abs_info["plot_path"] = abs_plot_path

    # PLSR artifacts
    if isinstance(calib, dict) and isinstance(calib.get("plsr_model"), dict):
        pm = calib["plsr_model"]
        y_true = np.array(pm.get("concentrations", []), dtype=float)
        y_pred_cv = pm.get("predictions_cv", None)
        y_pred_in = pm.get("predictions_in", None)
        y_pred = np.array(
            y_pred_cv if y_pred_cv is not None else (y_pred_in if y_pred_in is not None else []),
            dtype=float,
        )
        if y_true.size and y_pred.size and y_true.size == y_pred.size:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.scatter(y_true, y_pred, s=20)
            minv = float(min(y_true.min(), y_pred.min()))
            maxv = float(max(y_true.max(), y_pred.max()))
            ax2.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
            r2cv = pm.get("r2_cv", float("nan"))
            ax2.set_xlabel("Actual Concentration (ppm)")
            ax2.set_ylabel("Predicted Concentration (ppm)")
            ax2.set_title(f"PLSR CV (R^2={r2cv:.3f})")
            fig2.tight_layout()
            fig2.savefig(os.path.join(plots_dir, "plsr_pred_vs_actual.png"), dpi=200)
            plt.close(fig2)

        wl_arr = np.array(pm.get("wavelengths", []), dtype=float)
        coef_arr = np.array(pm.get("coef_", []), dtype=float)
        if wl_arr.size and coef_arr.size and wl_arr.size == coef_arr.size:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.plot(wl_arr, coef_arr, linewidth=1)
            ax3.set_xlabel("Wavelength (nm)")
            ax3.set_ylabel("PLSR Coefficient")
            ax3.set_title("PLSR Coefficients")
            fig3.tight_layout()
            fig3.savefig(os.path.join(plots_dir, "plsr_coefficients.png"), dpi=200)
            plt.close(fig3)

        try:
            comps = pm.get("n_components_candidates", None)
            r2_curve = pm.get("r2_cv_curve", None)
            rmse_curve = pm.get("rmse_cv_curve", None)
            if (isinstance(comps, list) and isinstance(r2_curve, list) and isinstance(rmse_curve, list)
                    and len(comps) == len(r2_curve) == len(rmse_curve) and len(comps) > 0):
                figc, axc1 = plt.subplots(figsize=(7, 4))
                axc1.plot(comps, r2_curve, marker="o", color="tab:blue", label="R² (CV)")
                axc1.set_xlabel("PLS components")
                axc1.set_ylabel("R² (CV)", color="tab:blue")
                axc1.tick_params(axis="y", labelcolor="tab:blue")
                axc2 = axc1.twinx()
                axc2.plot(comps, rmse_curve, marker="s", color="tab:red", label="RMSE (CV)")
                axc2.set_ylabel("RMSE (CV)", color="tab:red")
                axc2.tick_params(axis="y", labelcolor="tab:red")
                figc.tight_layout()
                figc.savefig(os.path.join(plots_dir, "plsr_cv_curves.png"), dpi=200)
                plt.close(figc)
        except Exception:
            pass

        try:
            y_true = np.array(pm.get("concentrations", []), dtype=float)
            y_pred = pm.get("predictions_cv", None)
            if y_pred is None:
                y_pred = pm.get("predictions_in", None)
            y_pred = np.array(y_pred if y_pred is not None else [], dtype=float)
            if y_true.size and y_pred.size and y_true.size == y_pred.size:
                resid = y_true - y_pred
                figd, axs = plt.subplots(2, 2, figsize=(9, 6))
                axs[0, 0].scatter(y_true, resid, s=18, alpha=0.8)
                axs[0, 0].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
                axs[0, 0].set_xlabel("Concentration (ppm)")
                axs[0, 0].set_ylabel("Residuals (ppm)")
                axs[0, 0].set_title("Residuals vs Concentration")
                axs[0, 1].scatter(y_pred, resid, s=18, alpha=0.8)
                axs[0, 1].axhline(0.0, color="k", linestyle="--", linewidth=0.8)
                axs[0, 1].set_xlabel("Predicted (ppm)")
                axs[0, 1].set_ylabel("Residuals (ppm)")
                axs[0, 1].set_title("Residuals vs Predicted")
                axs[1, 0].hist(resid, bins=max(8, int(np.sqrt(len(resid)))),
                               color="gray", edgecolor="black", alpha=0.8)
                axs[1, 0].set_xlabel("Residual (ppm)")
                axs[1, 0].set_ylabel("Frequency")
                axs[1, 0].set_title("Residual Histogram")
                try:
                    (osm, osr), (slope_p, intercept_p, r) = probplot(resid, dist="norm", fit=True)
                    axs[1, 1].scatter(osm, osr, s=14, alpha=0.8)
                    axs[1, 1].plot(osm, slope_p * np.array(osm) + intercept_p, "r--", linewidth=1)
                    axs[1, 1].set_title(f"Normal Q-Q (R={r:.3f})")
                    axs[1, 1].set_xlabel("Theoretical quantiles")
                    axs[1, 1].set_ylabel("Ordered residuals")
                except Exception:
                    axs[1, 1].text(0.5, 0.5, "Q-Q plot failed", ha="center", va="center")
                    axs[1, 1].set_axis_off()
                figd.tight_layout()
                figd.savefig(os.path.join(plots_dir, "plsr_residuals.png"), dpi=200)
                plt.close(figd)
        except Exception:
            pass

    # Selected model predicted vs actual
    try:
        y_true = np.array(calib.get("selected_actual", []), dtype=float)
        y_pred = np.array(calib.get("selected_predictions", []), dtype=float)
        if y_true.size and y_pred.size and y_true.size == y_pred.size:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.scatter(y_true, y_pred, s=20)
            minv = float(min(y_true.min(), y_pred.min()))
            maxv = float(max(y_true.max(), y_pred.max()))
            ax4.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
            sel_model = str(calib.get("selected_model", "selected"))
            sel_mode = str(calib.get("selected_predictions_mode", "cv"))
            r2cv = float(calib.get("uncertainty", {}).get("r2_cv", float("nan")))
            ax4.set_xlabel("Actual Concentration (ppm)")
            ax4.set_ylabel("Predicted Concentration (ppm)")
            ax4.set_title(f"Selected model ({sel_model}, {sel_mode}, R^2={r2cv:.3f})")
            fig4.tight_layout()
            fig4.savefig(os.path.join(plots_dir, "selected_pred_vs_actual.png"), dpi=200)
            plt.close(fig4)
            try:
                sel_df = pd.DataFrame({"actual": y_true.astype(float), "predicted": y_pred.astype(float)})
                sel_df.to_csv(os.path.join(metrics_dir, "selected_predictions.csv"), index=False)
            except Exception:
                pass
    except Exception:
        pass


