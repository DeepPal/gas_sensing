"""
src.reporting.publication
=========================
Publication-quality figure export for peer-reviewed journals.

Provides journal-specific dimension presets and three high-level figure
generators that cover the most common spectroscopy publication figures:

1. :func:`save_calibration_figure` — calibration curve with LOD/LOQ/LOL
   shading and 95 % confidence band.
2. :func:`save_spectral_overlay_figure` — multi-concentration spectral
   overlay with a perceptual colormap.
3. :func:`save_pls_diagnostics_figure` — 4-panel PLS diagnostic figure
   (T-scores, loadings, VIP scores, parity plot).

Journal presets
---------------
All physical dimensions are in millimetres; DPI follows each journal's
electronic submission guidelines.  Font sizes are the lower end of each
journal's stated range so that axes remain legible at print size.

+-------------+-------------------+--------------------+--------+--------+
| Preset key  | Width (mm)        | Journal family     | DPI    | Font   |
+-------------+-------------------+--------------------+--------+--------+
| acs_single  | 84 (≈ 3.3 in)     | ACS single col     | 600    | 7 pt   |
| acs_double  | 178 (≈ 7.0 in)    | ACS double col     | 600    | 7 pt   |
| nature_s    | 88                | Nature single col  | 300    | 6 pt   |
| nature_d    | 180               | Nature double col  | 300    | 6 pt   |
| rsc_single  | 84                | RSC single col     | 600    | 7 pt   |
| rsc_double  | 170               | RSC double col     | 600    | 7 pt   |
| elsevier_s  | 90                | Elsevier single col| 300    | 8 pt   |
| elsevier_d  | 190               | Elsevier double col| 300    | 8 pt   |
+-------------+-------------------+--------------------+--------+--------+

Usage
-----
::

    from src.reporting.publication import (
        journal_style, save_calibration_figure,
        save_spectral_overlay_figure, save_pls_diagnostics_figure,
    )

    # Basic calibration curve — ACS single column
    save_calibration_figure(
        concentrations=concs,
        responses=responses,
        fit_result=characterization,
        out_path="figures/fig2_calibration.tiff",
        preset="acs_single",
        gas_name="Ethanol",
    )

    # Use the context manager directly for custom figures
    with journal_style("nature_s") as fig_kw:
        fig, ax = plt.subplots(1, 1, **fig_kw)
        ax.plot(...)
        fig.savefig("custom.pdf", ...)

References
----------
- ACS Author Guidelines: https://publish.acs.org/publish/author_guidelines
- Nature figure guide: https://www.nature.com/nature/for-authors/formatting-guide
- RSC guidelines: https://www.rsc.org/journals-books-databases/author-and-reviewer-hub/
- Elsevier artwork guide: https://www.elsevier.com/authors/policies-and-guidelines
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Generator

import matplotlib
import numpy as np

matplotlib.use("Agg")  # safe in headless / CI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ---------------------------------------------------------------------------
# Journal presets
# ---------------------------------------------------------------------------

_MM_PER_INCH: float = 25.4

#: Preset registry — keys map to style dictionaries.
#: Each entry contains ``width_mm``, ``aspect``, ``dpi``, ``font_size``,
#: ``tick_size``, ``line_width``, ``marker_size``, ``pad``.
JOURNAL_PRESETS: dict[str, dict[str, Any]] = {
    # ── ACS Publications ────────────────────────────────────────────────────
    "acs_single": {
        "width_mm": 84, "aspect": 0.80, "dpi": 600,
        "font_size": 7, "tick_size": 6, "line_width": 0.75,
        "marker_size": 3.0, "pad": 0.04,
        "description": "ACS single column (84 mm / 3.33 in)",
    },
    "acs_double": {
        "width_mm": 178, "aspect": 0.50, "dpi": 600,
        "font_size": 7, "tick_size": 6, "line_width": 0.75,
        "marker_size": 3.5, "pad": 0.04,
        "description": "ACS double column (178 mm / 7.0 in)",
    },
    # ── Nature Portfolio ─────────────────────────────────────────────────────
    "nature_s": {
        "width_mm": 88, "aspect": 0.80, "dpi": 300,
        "font_size": 6, "tick_size": 5, "line_width": 0.75,
        "marker_size": 3.0, "pad": 0.04,
        "description": "Nature single column (88 mm)",
    },
    "nature_d": {
        "width_mm": 180, "aspect": 0.50, "dpi": 300,
        "font_size": 6, "tick_size": 5, "line_width": 0.75,
        "marker_size": 3.5, "pad": 0.04,
        "description": "Nature double column (180 mm)",
    },
    # ── Royal Society of Chemistry ───────────────────────────────────────────
    "rsc_single": {
        "width_mm": 84, "aspect": 0.80, "dpi": 600,
        "font_size": 7, "tick_size": 6, "line_width": 0.75,
        "marker_size": 3.0, "pad": 0.04,
        "description": "RSC single column (84 mm)",
    },
    "rsc_double": {
        "width_mm": 170, "aspect": 0.50, "dpi": 600,
        "font_size": 7, "tick_size": 6, "line_width": 0.75,
        "marker_size": 3.5, "pad": 0.04,
        "description": "RSC double column (170 mm)",
    },
    # ── Elsevier ─────────────────────────────────────────────────────────────
    "elsevier_s": {
        "width_mm": 90, "aspect": 0.80, "dpi": 300,
        "font_size": 8, "tick_size": 7, "line_width": 0.80,
        "marker_size": 3.5, "pad": 0.05,
        "description": "Elsevier single column (90 mm)",
    },
    "elsevier_d": {
        "width_mm": 190, "aspect": 0.50, "dpi": 300,
        "font_size": 8, "tick_size": 7, "line_width": 0.80,
        "marker_size": 4.0, "pad": 0.05,
        "description": "Elsevier double column (190 mm)",
    },
}


def preset_info(preset: str = "acs_single") -> dict[str, Any]:
    """Return the metadata dict for *preset*.

    Parameters
    ----------
    preset :
        Key in :data:`JOURNAL_PRESETS`.

    Raises
    ------
    KeyError
        If *preset* is not registered.
    """
    if preset not in JOURNAL_PRESETS:
        known = sorted(JOURNAL_PRESETS)
        raise KeyError(f"Unknown preset {preset!r}.  Known: {known}")
    return dict(JOURNAL_PRESETS[preset])


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def journal_style(
    preset: str = "acs_single",
    extra_rc: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager that applies journal-correct ``rcParams`` for *preset*.

    Yields a ``fig_kw`` dict suitable for passing to ``plt.subplots(**fig_kw)``.
    All figures created inside the block inherit the correct font sizes,
    line widths, tick parameters, and DPI.

    Parameters
    ----------
    preset :
        Journal preset name.  See :data:`JOURNAL_PRESETS`.
    extra_rc :
        Additional ``rcParams`` to merge (override) on top of the preset.

    Yields
    ------
    dict
        ``{"figsize": (w_in, h_in), "dpi": dpi}`` — pass as ``**fig_kw`` to
        ``plt.subplots()``.

    Example
    -------
    ::

        with journal_style("acs_single") as kw:
            fig, ax = plt.subplots(1, 1, **kw)
            ax.plot(x, y)
            fig.savefig("fig1.tiff", bbox_inches="tight", dpi=kw["dpi"])
    """
    p = preset_info(preset)
    width_in = p["width_mm"] / _MM_PER_INCH
    height_in = width_in * p["aspect"]
    dpi = p["dpi"]

    rc: dict[str, Any] = {
        "font.size": p["font_size"],
        "axes.titlesize": p["font_size"],
        "axes.labelsize": p["font_size"],
        "xtick.labelsize": p["tick_size"],
        "ytick.labelsize": p["tick_size"],
        "legend.fontsize": p["tick_size"],
        "lines.linewidth": p["line_width"],
        "lines.markersize": p["marker_size"],
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "pdf.fonttype": 42,   # embed fonts in PDF
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
    if extra_rc:
        rc.update(extra_rc)

    fig_kw: dict[str, Any] = {
        "figsize": (width_in, height_in),
        "dpi": dpi,
    }

    with matplotlib.rc_context(rc):
        yield fig_kw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    """Atomic-rename save: write to .tmp first, then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fmt = path.suffix.lstrip(".").lower() or "png"
    # TIFF → tiff alias accepted by matplotlib
    fig.savefig(str(tmp), format=fmt, dpi=dpi, bbox_inches="tight")
    tmp.rename(path)


def _linear_confidence_band(
    x: np.ndarray,
    slope: float,
    intercept: float,
    noise_std: float,
    x_eval: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """95 % confidence band for a simple linear calibration curve.

    Uses the t-distribution formulation for the expected value:
    ``ŷ ± t_{α/2, n-2} · s · sqrt(1/n + (x-x̄)²/Sxx)``

    Returns
    -------
    tuple
        (lower_band, upper_band)
    """
    from scipy.stats import t as t_dist

    n = len(x)
    if n < 3:
        y_hat = slope * x_eval + intercept
        band = np.full_like(y_hat, noise_std)
        return y_hat - band, y_hat + band

    x_mean = float(np.mean(x))
    sxx = float(np.sum((x - x_mean) ** 2))
    t_val = float(t_dist.ppf(1.0 - alpha / 2, df=n - 2))

    y_hat = slope * x_eval + intercept
    se = noise_std * np.sqrt(1.0 / n + (x_eval - x_mean) ** 2 / max(sxx, 1e-12))
    half_width = t_val * se

    return y_hat - half_width, y_hat + half_width


# ---------------------------------------------------------------------------
# 1. Calibration curve
# ---------------------------------------------------------------------------


def save_calibration_figure(
    concentrations: np.ndarray,
    responses: np.ndarray,
    fit_result: dict[str, Any],
    out_path: str | Path,
    preset: str = "acs_single",
    gas_name: str = "",
    response_label: str = r"$\Delta\lambda$ (nm)",
    concentration_label: str = "Concentration (ppm)",
    show_confidence_band: bool = True,
    annotate_lod: bool = True,
    annotate_loq: bool = True,
    annotate_lol: bool = True,
    color: str = "#2166ac",
    panel_label: str | None = None,
) -> Path:
    """Publication-ready calibration curve with ICH Q2 limit annotations.

    Generates a scatter plot of calibration data with a linear fit,
    95 % confidence band (optional), and shaded/dashed annotations for
    LOD, LOQ, and LOL as computed by
    :func:`~src.reporting.metrics.compute_comprehensive_sensor_characterization`.

    Parameters
    ----------
    concentrations :
        Calibration concentrations, shape ``(n,)``.
    responses :
        Measured responses (Δλ, peak intensity, …), shape ``(n,)``.
    fit_result :
        Dict returned by :func:`compute_comprehensive_sensor_characterization`.
        Expected keys: ``sensitivity``, ``intercept``, ``noise_std``,
        ``lod_ppm``, ``loq_ppm``, ``lol_ppm``.
    out_path :
        Output file.  Extension determines format (png, tiff, pdf, svg, eps).
    preset :
        Journal preset name.
    gas_name :
        Analyte name used in the legend and title.
    response_label :
        Y-axis label.
    concentration_label :
        X-axis label.
    show_confidence_band :
        Overlay a 95 % confidence band around the fit line.
    annotate_lod :
        Draw a dashed vertical line at LOD and shade the LOD region.
    annotate_loq :
        Draw a dashed vertical line at LOQ.
    annotate_lol :
        Mark the LOL with a dotted vertical line (linearity upper bound).
    color :
        Hex colour for data points and fit line.
    panel_label :
        If provided (e.g. ``"(a)"``), placed in the upper-left corner as a
        bold panel label following journal conventions.

    Returns
    -------
    Path
        Resolved path of the written figure file.
    """
    concs = np.asarray(concentrations, dtype=float)
    resps = np.asarray(responses, dtype=float)

    slope = float(fit_result.get("sensitivity", 1.0))
    intercept = float(fit_result.get("intercept", 0.0))
    noise_std = float(fit_result.get("noise_std", 0.0))
    lod = float(fit_result.get("lod_ppm", float("nan")))
    loq = float(fit_result.get("loq_ppm", float("nan")))
    lol = float(fit_result.get("lol_ppm", float("nan")))
    r2 = float(fit_result.get("r_squared", float("nan")))

    out_path = _ensure_parent(out_path)
    p = preset_info(preset)

    with journal_style(preset) as fig_kw:
        fig, ax = plt.subplots(1, 1, **fig_kw)

        # ── Fit line + confidence band ──────────────────────────────────────
        x_max_plot = float(np.nanmax(concs)) * 1.1
        x_fit = np.linspace(0.0, x_max_plot, 300)
        y_fit = slope * x_fit + intercept

        if show_confidence_band and noise_std > 0:
            lo, hi = _linear_confidence_band(
                concs, slope, intercept, noise_std, x_fit
            )
            ax.fill_between(x_fit, lo, hi, alpha=0.15, color=color, linewidth=0)

        r2_str = f", R² = {r2:.4f}" if not np.isnan(r2) else ""
        label_fit = f"{gas_name} fit{r2_str}" if gas_name else f"Linear fit{r2_str}"
        ax.plot(x_fit, y_fit, color=color, linewidth=p["line_width"],
                label=label_fit, zorder=3)

        # ── Data points ─────────────────────────────────────────────────────
        ax.scatter(concs, resps, color=color, s=p["marker_size"] ** 2,
                   zorder=4, linewidths=0.3, edgecolors="white",
                   label=f"Calibration data (n={len(concs)})")

        # ── LOD / LOQ / LOL annotations ─────────────────────────────────────
        y_min, y_max = ax.get_ylim()

        if annotate_lod and not np.isnan(lod) and lod < x_max_plot:
            ax.axvline(lod, linestyle="--", linewidth=0.6, color="#d6604d",
                       label=f"LOD = {lod:.3g} ppm")
            ax.axvspan(0, lod, alpha=0.06, color="#d6604d", linewidth=0)

        if annotate_loq and not np.isnan(loq) and loq < x_max_plot:
            ax.axvline(loq, linestyle="-.", linewidth=0.6, color="#f4a582",
                       label=f"LOQ = {loq:.3g} ppm")

        if annotate_lol and not np.isnan(lol) and lol < x_max_plot * 0.98:
            ax.axvline(lol, linestyle=":", linewidth=0.6, color="#4dac26",
                       label=f"LOL = {lol:.3g} ppm")

        # ── Axes decoration ─────────────────────────────────────────────────
        ax.set_xlabel(concentration_label)
        ax.set_ylabel(response_label)

        title_parts = ["Calibration Curve"]
        if gas_name:
            title_parts.append(gas_name)
        ax.set_title(" — ".join(title_parts))

        ax.legend(frameon=False, handlelength=1.5)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in")

        if panel_label:
            ax.text(
                0.02, 0.97, panel_label, transform=ax.transAxes,
                va="top", ha="left", fontweight="bold",
                fontsize=p["font_size"] + 1,
            )

        fig.tight_layout(pad=p["pad"])
        _save_fig(fig, out_path, p["dpi"])
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# 2. Spectral overlay
# ---------------------------------------------------------------------------


def save_spectral_overlay_figure(
    wavelengths: np.ndarray,
    spectra: np.ndarray,
    concentrations: np.ndarray,
    out_path: str | Path,
    preset: str = "acs_single",
    wl_range: tuple[float, float] | None = None,
    reference_spectrum: np.ndarray | None = None,
    xlabel: str = "Wavelength (nm)",
    ylabel: str = "Intensity (counts)",
    colormap: str = "viridis",
    show_colorbar: bool = True,
    panel_label: str | None = None,
) -> Path:
    """Multi-concentration spectral overlay with perceptual colormap.

    Plots each spectrum in *spectra* as a line coloured by its corresponding
    analyte concentration.  An optional reference (blank) spectrum can be
    overlaid in grey for comparison.

    Parameters
    ----------
    wavelengths :
        Wavelength axis, shape ``(n_pixels,)``.
    spectra :
        Stack of spectra, shape ``(n_concentrations, n_pixels)``.
    concentrations :
        Analyte concentrations for each row of *spectra*, shape
        ``(n_concentrations,)``.
    out_path :
        Output file path.
    preset :
        Journal preset name.
    wl_range :
        ``(lambda_min, lambda_max)`` in nm — zoom to this region if provided.
    reference_spectrum :
        Optional blank/reference spectrum, shape ``(n_pixels,)``.  Plotted as
        a dashed grey line labelled "Reference".
    xlabel, ylabel :
        Axis labels.
    colormap :
        Matplotlib colormap name.  ``"viridis"`` is recommended for
        accessibility and greyscale reproduction.
    show_colorbar :
        Whether to append a concentration colorbar.
    panel_label :
        Panel label, e.g. ``"(b)"``.

    Returns
    -------
    Path
        Resolved path of the written figure file.
    """
    wl = np.asarray(wavelengths, dtype=float)
    sp = np.asarray(spectra, dtype=float)
    concs = np.asarray(concentrations, dtype=float)

    if sp.ndim == 1:
        sp = sp[np.newaxis, :]

    out_path = _ensure_parent(out_path)
    p = preset_info(preset)

    with journal_style(preset) as fig_kw:
        fig, ax = plt.subplots(1, 1, **fig_kw)

        # Restrict wavelength range
        if wl_range is not None:
            mask = (wl >= wl_range[0]) & (wl <= wl_range[1])
        else:
            mask = np.ones(len(wl), dtype=bool)

        wl_plot = wl[mask]

        # ── Reference spectrum ───────────────────────────────────────────────
        if reference_spectrum is not None:
            ref = np.asarray(reference_spectrum, dtype=float)
            ax.plot(
                wl_plot, ref[mask], color="0.55",
                linewidth=p["line_width"] * 0.8,
                linestyle="--", zorder=2, label="Reference",
            )

        # ── Sample spectra ───────────────────────────────────────────────────
        cmap = plt.get_cmap(colormap)
        c_min = float(np.nanmin(concs))
        c_max = float(np.nanmax(concs))
        if c_max == c_min:
            c_max = c_min + 1.0
        norm = Normalize(vmin=c_min, vmax=c_max)

        for i, (spec_row, conc) in enumerate(zip(sp, concs)):
            rgba = cmap(norm(conc))
            ax.plot(
                wl_plot, spec_row[mask],
                color=rgba, linewidth=p["line_width"], alpha=0.85, zorder=3,
            )

        # ── Colorbar ─────────────────────────────────────────────────────────
        if show_colorbar:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
            cb.set_label("Concentration (ppm)", fontsize=p["tick_size"])
            cb.ax.tick_params(labelsize=p["tick_size"])

        # ── Axes decoration ──────────────────────────────────────────────────
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.tick_params(which="both", direction="in")

        if reference_spectrum is not None:
            ax.legend(frameon=False, handlelength=1.5)

        if panel_label:
            ax.text(
                0.02, 0.97, panel_label, transform=ax.transAxes,
                va="top", ha="left", fontweight="bold",
                fontsize=p["font_size"] + 1,
            )

        fig.tight_layout(pad=p["pad"])
        _save_fig(fig, out_path, p["dpi"])
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# 3. PLS diagnostics — 4-panel figure
# ---------------------------------------------------------------------------


def save_pls_diagnostics_figure(
    pls_result: Any,
    concentrations: np.ndarray,
    predicted: np.ndarray,
    out_path: str | Path,
    preset: str = "acs_double",
    wavelengths: np.ndarray | None = None,
    top_n_vip: int = 20,
    color: str = "#2166ac",
    panel_labels: tuple[str, str, str, str] = ("(a)", "(b)", "(c)", "(d)"),
) -> Path:
    """4-panel PLS diagnostic figure for publication.

    Panels
    ------
    (a) **T-scores** — scatter plot of LV1 vs LV2 scores coloured by
        concentration.  Reveals concentration ordering in latent space.
    (b) **P-loadings** — LV1 and LV2 spectral loadings vs wavelength (or
        pixel index when *wavelengths* is not provided).  Highlights which
        spectral regions drive variance.
    (c) **VIP scores** — bar chart of the top-*top_n_vip* important
        wavelengths/pixels.  Dashed reference line at VIP = 1.0 (Chong &
        Jun, 2005).
    (d) **Parity plot** — predicted vs measured concentrations with 1:1
        line and residual context.  Annotated with RMSECV and Q².

    Parameters
    ----------
    pls_result :
        :class:`~src.calibration.pls.PLSFitResult` instance.  Accessed
        fields: ``x_scores``, ``x_loadings``, ``vip_scores``,
        ``rmsecv``, ``q2``, ``n_components``.
    concentrations :
        Measured (reference) concentrations, shape ``(n,)``.
    predicted :
        Model-predicted concentrations, shape ``(n,)``.
    out_path :
        Output file path.
    preset :
        Journal preset.  ``"acs_double"`` gives a 178 mm wide 4-panel figure.
    wavelengths :
        Wavelength axis, shape ``(n_features,)``.  If ``None``, pixel indices
        are used on the x-axis of panels (b) and (c).
    top_n_vip :
        Number of top VIP-ranked features to show in panel (c).
    color :
        Base colour for scatter / bar plots.
    panel_labels :
        4-tuple of panel label strings.

    Returns
    -------
    Path
        Resolved path of the written figure file.
    """
    concs = np.asarray(concentrations, dtype=float)
    preds = np.asarray(predicted, dtype=float)

    out_path = _ensure_parent(out_path)
    p = preset_info(preset)

    # Extract from PLSFitResult (works for both dataclass and duck-typed dict)
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    x_scores = _get(pls_result, "x_scores")          # (n, n_comp)
    x_loadings = _get(pls_result, "x_loadings")      # (n_feat, n_comp)
    vip_scores = _get(pls_result, "vip_scores")      # (n_feat,)
    rmsecv = _get(pls_result, "rmsecv", float("nan"))
    q2 = _get(pls_result, "q2", float("nan"))

    with journal_style(preset) as fig_kw:
        # Override figsize: 4 panels in 2×2 layout — double the height
        w_in, h_in = fig_kw["figsize"]
        fig_kw["figsize"] = (w_in, h_in * 2.0)

        fig, axes = plt.subplots(2, 2, **fig_kw)
        (ax_scores, ax_load), (ax_vip, ax_parity) = axes

        cmap = plt.get_cmap("viridis")
        c_min = float(np.nanmin(concs))
        c_max = float(np.nanmax(concs))
        if c_max == c_min:
            c_max = c_min + 1.0
        norm = Normalize(vmin=c_min, vmax=c_max)

        # ── (a) T-scores ────────────────────────────────────────────────────
        if x_scores is not None and np.asarray(x_scores).shape[1] >= 2:
            sc = np.asarray(x_scores)
            sc_fig = ax_scores.scatter(
                sc[:, 0], sc[:, 1],
                c=concs, cmap=cmap, norm=norm,
                s=p["marker_size"] ** 2, linewidths=0.3, edgecolors="white",
                zorder=3,
            )
            cb_s = fig.colorbar(sc_fig, ax=ax_scores, pad=0.02, fraction=0.046)
            cb_s.set_label("Conc. (ppm)", fontsize=p["tick_size"])
            cb_s.ax.tick_params(labelsize=p["tick_size"] - 1)
            ax_scores.axhline(0, linewidth=0.4, color="0.7")
            ax_scores.axvline(0, linewidth=0.4, color="0.7")
            ax_scores.set_xlabel("LV1 score")
            ax_scores.set_ylabel("LV2 score")
        else:
            ax_scores.text(0.5, 0.5, "Scores unavailable",
                           ha="center", va="center", transform=ax_scores.transAxes,
                           fontsize=p["font_size"])

        if panel_labels[0]:
            ax_scores.text(0.03, 0.97, panel_labels[0], transform=ax_scores.transAxes,
                           va="top", ha="left", fontweight="bold",
                           fontsize=p["font_size"] + 1)

        # ── (b) P-loadings ───────────────────────────────────────────────────
        if x_loadings is not None and np.asarray(x_loadings).shape[1] >= 2:
            ld = np.asarray(x_loadings)
            x_axis = (
                np.asarray(wavelengths, dtype=float)
                if wavelengths is not None
                else np.arange(ld.shape[0])
            )
            x_label_load = "Wavelength (nm)" if wavelengths is not None else "Pixel index"

            ax_load.plot(x_axis, ld[:, 0], linewidth=p["line_width"],
                         color=color, label="LV1")
            ax_load.plot(x_axis, ld[:, 1], linewidth=p["line_width"],
                         color="#d6604d", linestyle="--", label="LV2")
            ax_load.axhline(0, linewidth=0.4, color="0.7")
            ax_load.set_xlabel(x_label_load)
            ax_load.set_ylabel("Loading weight")
            ax_load.legend(frameon=False, handlelength=1.2)
        else:
            ax_load.text(0.5, 0.5, "Loadings unavailable",
                         ha="center", va="center", transform=ax_load.transAxes,
                         fontsize=p["font_size"])

        if panel_labels[1]:
            ax_load.text(0.03, 0.97, panel_labels[1], transform=ax_load.transAxes,
                         va="top", ha="left", fontweight="bold",
                         fontsize=p["font_size"] + 1)

        # ── (c) VIP scores bar chart ─────────────────────────────────────────
        if vip_scores is not None:
            vip = np.asarray(vip_scores, dtype=float)
            top_idx = np.argsort(vip)[::-1][:top_n_vip]
            top_vip = vip[top_idx]

            x_tick_labels: list[str]
            if wavelengths is not None:
                wl_arr = np.asarray(wavelengths, dtype=float)
                x_tick_labels = [f"{wl_arr[i]:.0f}" for i in top_idx]
            else:
                x_tick_labels = [str(i) for i in top_idx]

            bar_x = np.arange(len(top_vip))
            bar_colors = [color if v >= 1.0 else "0.70" for v in top_vip]
            ax_vip.bar(bar_x, top_vip, color=bar_colors,
                       width=0.8, linewidth=0, zorder=3)
            ax_vip.axhline(1.0, linewidth=0.6, color="#d6604d",
                           linestyle="--", label="VIP = 1.0")

            step = max(1, len(x_tick_labels) // 8)
            ax_vip.set_xticks(bar_x[::step])
            ax_vip.set_xticklabels(x_tick_labels[::step], rotation=45, ha="right")
            ax_vip.set_xlabel(
                "Wavelength (nm)" if wavelengths is not None else "Feature index"
            )
            ax_vip.set_ylabel("VIP score")
            ax_vip.legend(frameon=False, handlelength=1.2)
        else:
            ax_vip.text(0.5, 0.5, "VIP scores unavailable",
                        ha="center", va="center", transform=ax_vip.transAxes,
                        fontsize=p["font_size"])

        if panel_labels[2]:
            ax_vip.text(0.03, 0.97, panel_labels[2], transform=ax_vip.transAxes,
                        va="top", ha="left", fontweight="bold",
                        fontsize=p["font_size"] + 1)

        # ── (d) Parity plot ─────────────────────────────────────────────────
        c_lo = min(float(np.nanmin(concs)), float(np.nanmin(preds))) * 0.9
        c_hi = max(float(np.nanmax(concs)), float(np.nanmax(preds))) * 1.05
        unity = np.array([c_lo, c_hi])

        ax_parity.plot(unity, unity, linewidth=0.6, color="0.5",
                       linestyle="--", zorder=1)
        ax_parity.scatter(concs, preds, color=color, s=p["marker_size"] ** 2,
                          zorder=3, linewidths=0.3, edgecolors="white")

        # Annotation: RMSECV and Q²
        ann_parts: list[str] = []
        if not np.isnan(rmsecv):
            ann_parts.append(f"RMSECV = {rmsecv:.4g}")
        if not np.isnan(q2):
            ann_parts.append(f"Q² = {q2:.4f}")
        if ann_parts:
            ax_parity.text(
                0.04, 0.96, "\n".join(ann_parts),
                transform=ax_parity.transAxes,
                va="top", ha="left", fontsize=p["tick_size"],
                linespacing=1.4,
            )

        ax_parity.set_xlabel("Measured concentration (ppm)")
        ax_parity.set_ylabel("Predicted concentration (ppm)")
        ax_parity.set_xlim(c_lo, c_hi)
        ax_parity.set_ylim(c_lo, c_hi)
        ax_parity.set_aspect("equal", adjustable="box")

        if panel_labels[3]:
            ax_parity.text(0.03, 0.97, panel_labels[3], transform=ax_parity.transAxes,
                           va="top", ha="left", fontweight="bold",
                           fontsize=p["font_size"] + 1)

        # ── Global decoration ────────────────────────────────────────────────
        for ax in axes.flat:
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.tick_params(which="both", direction="in")

        fig.tight_layout(pad=p["pad"])
        _save_fig(fig, out_path, p["dpi"])
        plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Convenience: list available presets
# ---------------------------------------------------------------------------


def list_presets() -> list[dict[str, str]]:
    """Return a list of ``{key, description}`` dicts for all available presets."""
    return [
        {"key": k, "description": v["description"]}
        for k, v in JOURNAL_PRESETS.items()
    ]
