"""
Au-MIP LSPR Gas Sensing Platform — Streamlit Dashboard
=======================================================

Four-tab interactive dashboard:

  Tab 1 — Automation Pipeline  : 5-agent workflow (acquire → train → predict → export)
  Tab 2 — Experiment (Guided)  : Step-by-step guided calibration workflow
  Tab 3 — Batch Analysis       : Offline exploration of Joy_Data CSV files
  Tab 4 — Live Sensor          : Real-time CCS200 monitoring and readout

Run from the project root::

    streamlit run dashboard/app.py
    # or
    run_dashboard.bat
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — must happen before any project imports
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Logging (Streamlit runs in a fresh process; initialise here)
# ---------------------------------------------------------------------------
try:
    import os as _os
    from gas_analysis.logging_setup import configure_logging  # noqa: E402

    _log_dir = Path(_os.environ.get("LOG_DIR", str(REPO_ROOT / "logs")))
    _log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        level=logging.INFO,
        log_file=_log_dir / "dashboard.log",
        console=False,  # Streamlit captures stderr; write to file instead
    )
except Exception:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional tab modules — import failures are surfaced in the UI, not hidden
# ---------------------------------------------------------------------------
_import_errors: dict[str, str] = {}

try:
    from dashboard.agentic_pipeline_tab import render as _render_agentic  # type: ignore[import]

    AGENTIC_AVAILABLE = True
except Exception as _exc:
    AGENTIC_AVAILABLE = False
    _import_errors["agentic"] = str(_exc)
    log.warning("Agentic pipeline tab unavailable: %s", _exc)

try:
    from dashboard.experiment_tab import render as _render_experiment  # type: ignore[import]

    EXPERIMENT_AVAILABLE = True
except Exception as _exc:
    EXPERIMENT_AVAILABLE = False
    _import_errors["experiment"] = str(_exc)
    log.warning("Experiment tab unavailable: %s", _exc)

# Core signal-processing imports — prefer src.preprocessing (new canonical location)
try:
    from src.preprocessing.baseline import airpls_baseline, als_baseline
    from src.preprocessing.baseline import correct_baseline as baseline_correction
    from src.preprocessing.denoising import smooth_spectrum, wavelet_denoise
    from src.features.lspr_features import detect_lspr_peak as _detect_peak_app

    SIGNAL_PROC_AVAILABLE = True
except Exception as _exc:
    # Fallback to legacy location during transition
    try:
        from gas_analysis.core.signal_proc import (
            als_baseline,
            baseline_correction,
            smooth_spectrum,
            wavelet_denoise,
        )

        SIGNAL_PROC_AVAILABLE = True
    except Exception as _exc2:
        SIGNAL_PROC_AVAILABLE = False
        _import_errors["signal_proc"] = str(_exc2)
        log.error("signal_proc unavailable: %s", _exc2)

# ---------------------------------------------------------------------------
# Live server — start once per process before any Streamlit calls
# ---------------------------------------------------------------------------
_LIVE_SERVER_PORT = 5006

try:
    from dashboard.live_server import start_live_server as _start_live_server

    _start_live_server(port=_LIVE_SERVER_PORT)
except Exception as _exc:
    log.warning("Live server could not start (live view will be unavailable): %s", _exc)

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Au-MIP LSPR Gas Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab_agentic, tab_exp, tab_batch, tab_live = st.tabs(
    [
        "🤖 Automation Pipeline",
        "🧪 Experiment (Guided)",
        "📊 Batch Analysis",
        "📡 Live Sensor",
    ]
)

# ===========================================================================
# Tab 1 — Automation Pipeline
# ===========================================================================
with tab_agentic:
    if AGENTIC_AVAILABLE:
        _render_agentic()
    else:
        st.error("Agentic Automation workflow is unavailable.")
        with st.expander("Error details"):
            st.code(_import_errors.get("agentic", "Unknown error"))

# ===========================================================================
# Tab 2 — Experiment (Guided)
# ===========================================================================
with tab_exp:
    if EXPERIMENT_AVAILABLE:
        _render_experiment()
    else:
        st.error("Experiment workflow is unavailable.")
        with st.expander("Error details"):
            st.code(_import_errors.get("experiment", "Unknown error"))

# ===========================================================================
# Tab 3 — Batch Analysis
# ===========================================================================


def _render_batch() -> None:
    st.title("🔬 Batch Spectrum Analysis")

    if not SIGNAL_PROC_AVAILABLE:
        st.error(
            "Signal processing module could not be loaded. "
            "Check that `gas_analysis` is importable from the project root."
        )
        with st.expander("Error details"):
            st.code(_import_errors.get("signal_proc", "Unknown error"))
        return

    # ---- Sidebar --------------------------------------------------------
    st.sidebar.header("Data Configuration")
    data_root = st.sidebar.text_input(
        "Data Root",
        str(REPO_ROOT / "data"),
        help="Directory containing gas sub-folders (e.g. data/JOY_Data/)",
    )

    root_path = Path(data_root).resolve()
    # Restrict traversal to paths within the repo root to prevent directory traversal
    try:
        root_path.relative_to(REPO_ROOT)
    except ValueError:
        st.sidebar.error("Data Root must be inside the project directory.")
        return

    if root_path.exists():
        available_gases = sorted(d.name for d in root_path.iterdir() if d.is_dir())
    else:
        available_gases = []

    if not available_gases:
        st.sidebar.warning("No gas directories found. Check the Data Root path.")
        st.info(
            "Expected structure:\n```\ndata/JOY_Data/\n"
            "├── Ethanol/\n│   ├── 0.5 ppm-1/\n│   └── ...\n"
            "└── ref_EtOH.csv\n```"
        )
        return

    gas_type = st.sidebar.selectbox("Select Gas", available_gases)
    gas_path = root_path / gas_type

    # ---- File discovery -------------------------------------------------
    csv_files = sorted(gas_path.rglob("*.csv"))
    st.sidebar.caption(f"{len(csv_files)} CSV files found for {gas_type}")

    if not csv_files:
        st.info(f"No CSV files found in `{gas_path}`.")
        return

    file_map = {f.name: f for f in csv_files}
    selected_filename = st.sidebar.selectbox("Select Spectrum", list(file_map.keys()))
    selected_file = file_map[selected_filename]

    # ---- Reference spectrum ---------------------------------------------
    ref_files = sorted(root_path.glob("ref*.csv"))
    default_ref_idx = 0
    for i, rf in enumerate(ref_files):
        if (
            gas_type in rf.name
            or ("EtOH" in rf.name and "Ethanol" in gas_type)
            or ("IPA" in rf.name and "IPA" in gas_type)
            or ("MeOH" in rf.name and "MeOH" in gas_type)
        ):
            default_ref_idx = i
            break

    st.sidebar.markdown("---")
    st.sidebar.subheader("Reference Spectrum")
    use_ref = st.sidebar.checkbox("Show Reference", value=True)
    selected_ref_file = None
    if ref_files:
        selected_ref_file = st.sidebar.selectbox(
            "Select Reference",
            ref_files,
            index=default_ref_idx,
            format_func=lambda x: x.name,
        )
    else:
        st.sidebar.warning("No reference files (ref*.csv) found in Data Root.")

    # ---- Inner tabs -----------------------------------------------------
    inner_spectrum, inner_science = st.tabs(["📈 Spectrum Analysis", "🔬 Scientific Analysis"])

    # ---------------------------------------------------------------
    # Inner tab A — Spectrum viewer with processing controls
    # ---------------------------------------------------------------
    with inner_spectrum:
        st.subheader(f"Spectrum: {selected_file.name}")

        try:
            df = pd.read_csv(selected_file)
            if "wavelength" not in df.columns:
                df = pd.read_csv(selected_file, header=None, names=["wavelength", "intensity"])
            wl = df["wavelength"].values
            intensity_raw = df["intensity"].values
        except Exception as exc:
            st.error(f"Failed to load spectrum: {exc}")
            log.error("Spectrum load error (%s): %s", selected_file, exc)
            return

        # Load reference
        intensity_ref = None
        if use_ref and selected_ref_file:
            try:
                df_ref = pd.read_csv(selected_ref_file)
                if "wavelength" not in df_ref.columns:
                    df_ref = pd.read_csv(
                        selected_ref_file, header=None, names=["wavelength", "intensity"]
                    )
                wl_ref = df_ref["wavelength"].values
                intensity_ref_raw = df_ref["intensity"].values
                # Interpolate to match experimental wavelength axis
                if len(wl_ref) != len(wl) or not np.allclose(wl_ref, wl, atol=0.01):
                    intensity_ref = np.interp(wl, wl_ref, intensity_ref_raw)
                else:
                    intensity_ref = intensity_ref_raw
            except Exception as exc:
                st.sidebar.error(f"Reference load error: {exc}")
                log.warning("Reference load error: %s", exc)

        fig, ax = plt.subplots(figsize=(10, 5))

        if intensity_ref is not None:
            ax.plot(
                wl,
                intensity_ref,
                label="Reference (Air/N₂)",
                color="black",
                alpha=0.5,
                linestyle=":",
                linewidth=1,
            )
        ax.plot(wl, intensity_raw, label="Raw Signal (Gas)", alpha=0.6, color="gray")

        # ---- Processing controls in sidebar ---------------------------
        st.sidebar.markdown("---")
        st.sidebar.subheader("1. Denoising")
        denoise_method = st.sidebar.radio(
            "Method", ["None", "Savitzky-Golay", "Wavelet (DWT)"], index=1
        )
        processed = intensity_raw.copy()

        if denoise_method == "Savitzky-Golay":
            window = st.sidebar.slider("Window Size", 3, 51, 11, step=2)
            poly = st.sidebar.slider("Poly Order", 1, 5, 2)
            processed = smooth_spectrum(processed, window=window, poly_order=poly)
            ax.plot(wl, processed, label="SG Smoothed", color="blue", linewidth=1.5)

        elif denoise_method == "Wavelet (DWT)":
            wavelet = st.sidebar.selectbox(
                "Wavelet", ["db4", "db8", "sym4", "sym8", "coif4"], index=0
            )
            mode = st.sidebar.selectbox("Threshold Mode", ["soft", "hard"], index=0)
            sigma_est = st.sidebar.selectbox("Noise Estimate", ["mad", "std"], index=0)
            processed = wavelet_denoise(processed, wavelet=wavelet, mode=mode, sigma_est=sigma_est)
            ax.plot(wl, processed, label="DWT Denoised", color="purple", linewidth=1.5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("2. Baseline Correction")
        baseline_method = st.sidebar.radio(
            "Baseline Method", ["None", "Polynomial", "ALS", "airPLS"], index=0
        )

        if baseline_method == "Polynomial":
            poly_base = st.sidebar.slider("Poly Order (Baseline)", 1, 5, 1)
            processed = baseline_correction(
                wl, processed, method="polynomial", poly_order=poly_base
            )
            ax.plot(wl, processed, label="Poly Baseline Corrected", color="green", linestyle="--")

        elif baseline_method == "ALS":
            lam = st.sidebar.number_input(
                "Lambda (λ)",
                value=10_000.0,
                step=1_000.0,
                help="Smoothness penalty — larger = smoother baseline",
            )
            p_val = st.sidebar.number_input(
                "p (Asymmetry)",
                value=0.01,
                step=0.001,
                format="%.4f",
                help="Fraction of points below baseline; small p keeps baseline below signal",
            )
            # als_baseline returns the corrected signal (intensities − baseline).
            # Reconstruct the baseline for overlay by taking the difference.
            corrected = als_baseline(processed, lam=lam, p=p_val)
            baseline_est = processed - corrected
            ax.plot(
                wl, baseline_est, label="ALS Baseline", color="orange", linestyle=":", alpha=0.7
            )
            processed = corrected
            ax.plot(wl, processed, label="ALS Corrected", color="green", linestyle="--")

        elif baseline_method == "airPLS":
            lam_air = st.sidebar.number_input(
                "Lambda (λ)",
                value=1e5,
                step=1e4,
                format="%.0f",
                help="Smoothness — larger = broader baseline (Zhang et al. 2010)",
            )
            # airpls_baseline returns the corrected signal (intensities − baseline).
            # Reconstruct the baseline for overlay by taking the difference.
            corrected = airpls_baseline(processed, lam=lam_air)
            baseline_est = processed - corrected
            ax.plot(
                wl,
                baseline_est,
                label="airPLS Baseline",
                color="darkorange",
                linestyle=":",
                alpha=0.7,
            )
            processed = corrected
            ax.plot(wl, processed, label="airPLS Corrected", color="green", linestyle="--")

        ax.legend(fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Spectrum — {gas_type} / {selected_file.name}", fontsize=11)
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------------------
    # Inner tab B — Scientific analysis (heatmap, calibration curve)
    # ---------------------------------------------------------------
    with inner_science:
        st.header("🧪 Scientific Analysis")
        st.caption(
            "Aggregates all CSV files by concentration level (parsed from folder names) "
            "and generates heatmaps, calibration curves, and LOD estimates."
        )

        # ── Sensor peak search window ─────────────────────────────────────
        with st.expander("Sensor Peak Window (must match your sensor)", expanded=False):
            st.caption(
                "Wavelength window used to locate the sensor response peak for Δλ "
                "computation. Set to the region where your sensor peak sits — "
                "outside this window the peak is ignored."
            )
            _pw_col1, _pw_col2 = st.columns(2)
            _batch_peak_min = _pw_col1.number_input(
                "Peak search min (nm)", min_value=100.0, max_value=2000.0,
                value=480.0, step=10.0, key="batch_peak_min",
            )
            _batch_peak_max = _pw_col2.number_input(
                "Peak search max (nm)", min_value=100.0, max_value=2000.0,
                value=800.0, step=10.0, key="batch_peak_max",
            )

        if st.button("Generate Heatmap & Calibration Curve"):
            with st.spinner(f"Processing {len(csv_files)} files…"):
                try:
                    import plotly.graph_objects as go
                    from scipy.stats import linregress

                    # --- Group files by concentration via canonical data_loader ---
                    from src.batch.data_loader import scan_experiment_root

                    try:
                        exp_scan = scan_experiment_root(gas_path, gas_type=gas_type)
                    except ValueError as _ve:
                        st.warning(str(_ve))
                        return

                    if not exp_scan.concentrations:
                        st.warning(
                            "Could not group files by concentration. "
                            "Check that folder names contain a numeric concentration "
                            "(e.g. `0.5 ppm-1`, `vary-EtOH-0.5-1`)."
                        )
                        return

                    st.success(
                        f"Found {len(exp_scan.concentrations)} concentration levels: "
                        f"{exp_scan.concentrations} ppm"
                    )

                    # --- Stable-plateau representative spectrum per concentration ---
                    # build_canonical_from_scan: temporal tail-gate (last 20 frames)
                    # + longest stable sub-block via normalised frame-to-frame MAD.
                    # This rejects gas-injection transients and late-exposure drift.
                    from src.batch.aggregation import build_canonical_from_scan
                    from src.batch.data_loader import load_frames

                    scan_data: dict[float, dict[str, list]] = {
                        c: {
                            trial: load_frames(paths, max_frames=30)
                            for trial, paths in conc_trials.items()
                        }
                        for c, conc_trials in exp_scan.trials.items()
                    }
                    canonical = build_canonical_from_scan(scan_data, n_tail=20, weight_mode="max")

                    common_wl: np.ndarray | None = None
                    mean_spectra: list[np.ndarray] = []
                    conc_levels: list[float] = []

                    for c in sorted(canonical.keys()):
                        df_c = canonical[c]
                        if df_c.empty:
                            continue
                        _sig = (
                            "intensity"
                            if "intensity" in df_c.columns
                            else [col for col in df_c.columns if col != "wavelength"][0]
                        )
                        w = df_c["wavelength"].values
                        i_arr = df_c[_sig].values
                        if common_wl is None:
                            common_wl = np.linspace(w.min(), w.max(), 1_000)
                        mean_spectra.append(np.interp(common_wl, w, i_arr))
                        conc_levels.append(c)

                    if not mean_spectra or common_wl is None:
                        st.error("No spectra could be loaded.")
                        return

                    Z = np.array(mean_spectra)
                    concs = np.array(conc_levels)

                    # --- Response variable: Δλ (LSPR-correct) or peak intensity ---
                    # LSPR primary signal is peak wavelength SHIFT relative to a
                    # reference (air/blank).  Load ref peak wavelength if available.
                    # Compute search window first so reference peak uses same bounds
                    _bpmin = st.session_state.get("batch_peak_min", _batch_peak_min)
                    _bpmax = st.session_state.get("batch_peak_max", _batch_peak_max)
                    _win_mask_batch = (common_wl >= _bpmin) & (common_wl <= _bpmax)
                    if _win_mask_batch.sum() < 3:
                        # Window covers no points — fall back to full spectrum
                        _win_mask_batch = np.ones(len(common_wl), dtype=bool)
                        st.warning(
                            f"Peak search window {_bpmin:.0f}–{_bpmax:.0f} nm covers no "
                            "interpolated wavelength points — using full spectrum."
                        )

                    _ref_peak_wl: float | None = None
                    if selected_ref_file:
                        try:
                            from src.batch.data_loader import read_spectrum_csv as _rsc

                            _df_r = _rsc(selected_ref_file)
                            _wl_r = _df_r["wavelength"].values
                            _i_r = _df_r["intensity"].values
                            # Lorentzian sub-pixel detection within the search window
                            try:
                                _rp = _detect_peak_app(_wl_r, _i_r, _bpmin, _bpmax)
                                _ref_peak_wl = _rp if _rp is not None else float(_wl_r[np.argmax(_i_r)])
                            except Exception:
                                _ref_peak_wl = float(_wl_r[np.argmax(_i_r)])
                        except Exception:
                            _ref_peak_wl = None

                    if _ref_peak_wl is not None:
                        # Δλ = peak in window(sample) − ref peak
                        # Lorentzian sub-pixel detection; falls back to argmax internally
                        def _batch_peak_wl(wl_arr, int_arr, wmin, wmax):
                            try:
                                p = _detect_peak_app(wl_arr, int_arr, wmin, wmax)
                                return p if p is not None else float(wl_arr[_win_mask_batch][np.argmax(int_arr[_win_mask_batch])])
                            except Exception:
                                return float(wl_arr[_win_mask_batch][np.argmax(int_arr[_win_mask_batch])])

                        _peak_wl_per_conc = np.array(
                            [
                                _batch_peak_wl(common_wl, spec, _bpmin, _bpmax)
                                for spec in Z
                            ]
                        )
                        responses = _peak_wl_per_conc - _ref_peak_wl
                        response_label = "Δλ (nm)"
                        sensitivity_unit = "nm/ppm"
                        st.info(
                            f"Response = Δλ relative to reference peak at "
                            f"**{_ref_peak_wl:.2f} nm** | search window "
                            f"**{_bpmin:.0f}–{_bpmax:.0f} nm**"
                        )
                    else:
                        _peak_idx = int(np.argmax(np.mean(Z[:, _win_mask_batch], axis=0)))
                        # Map windowed index back to full common_wl index
                        _peak_idx_full = int(np.where(_win_mask_batch)[0][_peak_idx])
                        responses = Z[:, _peak_idx_full]
                        response_label = f"Intensity @ {common_wl[_peak_idx_full]:.1f} nm (a.u.)"
                        sensitivity_unit = "a.u./ppm"
                        st.warning(
                            "No reference spectrum — falling back to **peak intensity** "
                            f"(window {_bpmin:.0f}–{_bpmax:.0f} nm). "
                            "Load a reference file for Δλ analysis."
                        )

                    # --- Per-trial spread for error bars and RSD ---
                    # Compute per-trial response for each concentration to get σ across replicates.
                    # Uses the same response variable (Δλ or peak intensity) as the calibration curve.
                    _conc_stds = np.zeros(len(concs))
                    _conc_n_trials: list[int] = []
                    _conc_rsd_pct: list[float] = []
                    for _ci, _c_val in enumerate(concs):
                        _trial_resp: list[float] = []
                        for _frames_list in scan_data.get(_c_val, {}).values():
                            if not _frames_list:
                                continue
                            try:
                                _t_spec = np.mean(
                                    [
                                        np.interp(
                                            common_wl,
                                            _f["wavelength"].values,
                                            _f["intensity"].values,
                                        )
                                        for _f in _frames_list
                                    ],
                                    axis=0,
                                )
                                if _ref_peak_wl is not None:
                                    _trial_resp.append(
                                        _batch_peak_wl(common_wl, _t_spec, _bpmin, _bpmax) - _ref_peak_wl
                                    )
                                else:
                                    _trial_resp.append(float(_t_spec[_peak_idx_full]))
                            except Exception:
                                continue
                        _conc_n_trials.append(len(_trial_resp))
                        if len(_trial_resp) >= 2:
                            _std_v = float(np.std(_trial_resp, ddof=1))
                            _mean_v = float(np.mean(_trial_resp))
                            _conc_stds[_ci] = _std_v
                            _conc_rsd_pct.append(
                                abs(_std_v / _mean_v * 100) if _mean_v != 0 else float("nan")
                            )
                        else:
                            _conc_rsd_pct.append(float("nan"))

                    # --- Spectral overlay ---
                    colors = [f"hsl({h},70%,50%)" for h in np.linspace(0, 240, len(concs))]
                    fig_ov = go.Figure()
                    for idx, spec in enumerate(Z):
                        fig_ov.add_trace(
                            go.Scatter(
                                x=common_wl,
                                y=spec,
                                mode="lines",
                                name=f"{concs[idx]} ppm",
                                line=dict(color=colors[idx]),
                            )
                        )
                    fig_ov.update_layout(
                        title=f"Spectral Evolution — {gas_type} (stable-plateau representative)",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Intensity (a.u.)",
                        height=450,
                    )
                    st.plotly_chart(fig_ov, use_container_width=True)

                    # --- Calibration curve ---
                    slope, intercept, r_val, *_ = linregress(concs, responses)
                    x_fit = np.linspace(concs.min(), concs.max(), 100)

                    fig_cal = go.Figure()
                    _has_error_bars = any(s > 0 for s in _conc_stds)
                    fig_cal.add_trace(
                        go.Scatter(
                            x=concs,
                            y=responses,
                            mode="markers",
                            name="Measured (mean ± σ)",
                            marker=dict(size=12, color="red"),
                            error_y=dict(
                                type="data",
                                array=_conc_stds.tolist(),
                                visible=_has_error_bars,
                                color="rgba(200,50,50,0.6)",
                                thickness=2,
                                width=6,
                            ),
                        )
                    )
                    fig_cal.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=slope * x_fit + intercept,
                            mode="lines",
                            name=f"Linear fit (R²={r_val**2:.4f})",
                            line=dict(color="royalblue", dash="dash"),
                        )
                    )
                    fig_cal.update_layout(
                        title=f"Calibration Curve — {gas_type}",
                        xaxis_title="Concentration (ppm)",
                        yaxis_title=response_label,
                        height=400,
                    )
                    st.plotly_chart(fig_cal, use_container_width=True)

                    # --- Per-concentration reproducibility table ---
                    with st.expander("📋 Per-Concentration Reproducibility Statistics"):
                        _rsd_display = [
                            f"{v:.1f} %" if not np.isnan(v) else "— (1 trial)"
                            for v in _conc_rsd_pct
                        ]
                        _std_display = [
                            f"{s:.4g}" if s > 0 else "—"
                            for s in _conc_stds
                        ]
                        _repro_df = pd.DataFrame(
                            {
                                "Concentration (ppm)": concs,
                                f"Response ({response_label})": np.round(responses, 4),
                                "σ (std across trials)": _std_display,
                                "RSD": _rsd_display,
                                "n trials": _conc_n_trials,
                            }
                        )
                        st.dataframe(_repro_df, use_container_width=True, hide_index=True)
                        if any(not np.isnan(v) for v in _conc_rsd_pct):
                            _valid_rsd = [v for v in _conc_rsd_pct if not np.isnan(v)]
                            st.caption(
                                f"Mean RSD across concentrations: **{np.mean(_valid_rsd):.1f} %** "
                                f"| Max: **{np.max(_valid_rsd):.1f} %** "
                                "(RSD < 5 % indicates good reproducibility for gas sensing)"
                            )

                    # --- Heatmap ---
                    fig_hm = go.Figure(
                        data=go.Heatmap(
                            z=Z,
                            x=common_wl,
                            y=concs,
                            colorscale="Viridis",
                            colorbar=dict(title="Intensity"),
                        )
                    )
                    fig_hm.update_layout(
                        title=f"Sensitivity Heatmap — {gas_type}",
                        xaxis_title="Wavelength (nm)",
                        yaxis_title="Concentration (ppm)",
                        height=420,
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)

                    # --- 3D surface ---
                    fig_3d = go.Figure(data=[go.Surface(z=Z, x=common_wl, y=concs)])
                    fig_3d.update_layout(
                        title=f"3D Response Surface — {gas_type}",
                        scene=dict(
                            xaxis_title="Wavelength (nm)",
                            yaxis_title="Concentration (ppm)",
                            zaxis_title="Intensity (a.u.)",
                        ),
                        height=550,
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                    # --- Metrics ---
                    st.markdown("### Dataset Metrics")
                    # σ noise in response units: for Δλ mode use fit residuals
                    # (captures instrument repeatability); for intensity mode use
                    # the first-50-wavelength baseline region of the blank spectrum.
                    _fit_residuals = responses - (slope * concs + intercept)
                    if _ref_peak_wl is not None:
                        _ddof = min(2, max(1, len(_fit_residuals) - 1))
                        baseline_noise = float(np.std(_fit_residuals, ddof=_ddof))
                    else:
                        baseline_noise = float(np.std(Z[0, :50]))

                    try:
                        from src.scientific.lod import (
                            calculate_lod_3sigma,
                            lod_bootstrap_ci,
                            mandel_linearity_test,
                        )

                        from src.scientific.lod import calculate_loq_10sigma

                        lod = calculate_lod_3sigma(baseline_noise, slope)
                        lod_str = f"{lod:.3f} ppm"
                        loq = calculate_loq_10sigma(baseline_noise, slope)
                        loq_str = f"{loq:.3f} ppm"

                        # Bootstrap 95 % CI on LOD
                        try:
                            _lod_pt, _lod_lo, _lod_hi = lod_bootstrap_ci(
                                concs, responses, n_bootstrap=500
                            )
                            lod_ci_str = (
                                f"{_lod_pt:.3f} ppm "
                                f"[{_lod_lo:.3f}–{_lod_hi:.3f}]"
                            )
                        except Exception:
                            lod_ci_str = "N/A"

                        # Mandel linearity test (ICH Q2(R1))
                        try:
                            mandel = mandel_linearity_test(concs, responses)
                        except Exception:
                            mandel = None

                    except Exception:
                        lod_str = "N/A"
                        lod_ci_str = "N/A"
                        loq_str = "N/A"
                        mandel = None

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Sensitivity", f"{abs(slope):.4g} {sensitivity_unit}")
                    c1.metric("Linearity R²", f"{r_val**2:.4f}")
                    _noise_label = (
                        "Δλ Noise σ (nm)" if _ref_peak_wl is not None else "Baseline Noise σ"
                    )
                    c2.metric(_noise_label, f"{baseline_noise:.2e}")
                    c2.metric("LOD (3σ/S)", lod_str, help="IUPAC LOD = 3σ_noise / sensitivity")
                    c3.metric("LOQ (10σ/S)", loq_str, help="IUPAC LOQ = 10σ_noise / sensitivity — lowest reliably quantifiable concentration")
                    c3.metric("LOD 95 % CI (bootstrap)", lod_ci_str)
                    if mandel:
                        verdict = "PASS" if mandel["is_linear"] else "FAIL"
                        c4.metric(
                            "Mandel Linearity",
                            f"{verdict}  p={mandel['p_value']:.3f}",
                            help="ICH Q2(R1) F-test: PASS = quadratic term not significant (p>0.05)",
                        )

                    # --- Isotherm Fitting ---
                    st.markdown("### Nonlinear Isotherm Fitting")
                    st.caption(
                        "Fits Langmuir, Freundlich, and Hill models by nonlinear least "
                        "squares and selects the best via AIC (lower = better). "
                        "More physically appropriate than linear regression when the "
                        "sensor operates near MIP binding-site saturation."
                    )
                    if st.button("Fit Isotherms (AIC selection)", key="btn_isotherms"):
                        try:
                            from src.calibration.isotherms import select_isotherm

                            iso_sel = select_isotherm(concs, responses)
                            iso = iso_sel["best_result"]

                            i_c1, i_c2 = st.columns([2, 1])
                            with i_c1:
                                fig_iso = go.Figure()
                                fig_iso.add_trace(
                                    go.Scatter(
                                        x=concs,
                                        y=responses,
                                        mode="markers",
                                        name="Measured",
                                        marker=dict(size=12, color="crimson"),
                                    )
                                )
                                fig_iso.add_trace(
                                    go.Scatter(
                                        x=iso.concentrations_fit,
                                        y=iso.responses_fit,
                                        mode="lines",
                                        name=f"{iso.model.capitalize()} (AIC winner)",
                                        line=dict(color="royalblue", dash="dash", width=2),
                                    )
                                )
                                fig_iso.update_layout(
                                    title=(
                                        f"Isotherm Fit — {iso.model.capitalize()} "
                                        f"(R²={iso.r_squared:.4f}, RMSE={iso.rmse:.4g}, "
                                        f"AIC={iso.aic:.2f})"
                                    ),
                                    xaxis_title="Concentration (ppm)",
                                    yaxis_title=response_label,
                                    height=380,
                                )
                                st.plotly_chart(fig_iso, use_container_width=True)

                            with i_c2:
                                st.markdown(f"**Winner: {iso.model.capitalize()}**")
                                st.caption(f"AIC = {iso.aic:.2f}")
                                st.caption(f"R² = {iso.r_squared:.4f}")
                                st.caption(f"RMSE = {iso.rmse:.4g}")
                                st.caption(iso_sel["recommendation"])
                                st.markdown("**Parameters:**")
                                for pname, pval in iso.params.items():
                                    if pname == "sign":
                                        continue
                                    perr = iso.param_stderrs.get(pname, float("nan"))
                                    st.caption(f"`{pname}` = {pval:.4g} ± {perr:.2g}")
                                st.markdown("**AIC table:**")
                                for _mn, _ma, _mr2, _mrmse in iso_sel["aic_table"]:
                                    st.caption(f"{_mn}: AIC={_ma:.2f}, R²={_mr2:.4f}")

                        except Exception as exc:
                            st.error(f"Isotherm fitting failed: {exc}")
                            with st.expander("Traceback"):
                                import traceback as _tb_iso

                                st.code(_tb_iso.format_exc())

                    # --- Cross-Gas Selectivity ---
                    st.markdown("### Cross-Gas Selectivity Matrix")
                    st.caption(
                        "Computes K_{j/i} = S_j / S_i for all gas pairs in the selected "
                        "data root.  |K| > 0.5 (orange) indicates problematic cross-sensitivity."
                    )
                    if st.button("Compute Selectivity Matrix", key="btn_selectivity"):
                        with st.spinner("Loading calibration data for all gases…"):
                            try:
                                from scipy.stats import linregress as _lr

                                from src.scientific.selectivity import selectivity_matrix

                                gas_sensitivities: dict[str, float] = {}
                                for gdir in sorted(root_path.iterdir()):
                                    if not gdir.is_dir():
                                        continue
                                    # Use canonical data_loader for concentration grouping
                                    from src.batch.data_loader import (
                                        load_last_n_frames as _load_last,
                                    )
                                    from src.batch.data_loader import (
                                        scan_experiment_root as _scan_root,
                                    )

                                    try:
                                        g_scan = _scan_root(gdir, gas_type=gdir.name)
                                    except ValueError:
                                        continue
                                    if len(g_scan.concentrations) < 2:
                                        continue
                                    # Peak intensity per concentration
                                    g_concs, g_resp = [], []
                                    g_wl_ref = None
                                    for cv in g_scan.concentrations:
                                        batch_specs = []
                                        for frame_df in _load_last(g_scan.frames_for(cv), n=10):
                                            wg = frame_df["wavelength"].values
                                            ig = frame_df["intensity"].values
                                            if g_wl_ref is None:
                                                g_wl_ref = np.linspace(wg.min(), wg.max(), 500)
                                            batch_specs.append(np.interp(g_wl_ref, wg, ig))
                                        if batch_specs:
                                            m = np.mean(batch_specs, axis=0)
                                            g_concs.append(cv)
                                            # Use peak wavelength (nm) — sensitivity
                                            # in nm/ppm makes K ratios dimensionless
                                            # and physically correct for LSPR sensors.
                                            g_resp.append(float(g_wl_ref[np.argmax(m)]))
                                    if len(g_concs) >= 2:
                                        s, *_ = _lr(g_concs, g_resp)
                                        gas_sensitivities[gdir.name] = float(s)

                                if len(gas_sensitivities) < 2:
                                    st.warning(
                                        "Need calibration data for at least 2 gases "
                                        "(≥2 concentration sub-folders each). "
                                        "Check the Data Root path."
                                    )
                                else:
                                    sel = selectivity_matrix(gas_sensitivities)

                                    import plotly.graph_objects as _go

                                    gases_s = sel.gases
                                    mat = sel.matrix
                                    # Colour scale: white at 0, orange at 0.5, red at 1+
                                    fig_sel = _go.Figure(
                                        data=_go.Heatmap(
                                            z=np.abs(mat).tolist(),
                                            x=gases_s,
                                            y=gases_s,
                                            colorscale=[
                                                [0.0, "#ffffff"],
                                                [0.1, "#ffffcc"],
                                                [0.5, "#fd8d3c"],
                                                [1.0, "#bd0026"],
                                            ],
                                            zmin=0,
                                            zmax=1,
                                            text=[
                                                [f"{mat[i, j]:+.3f}" for j in range(len(gases_s))]
                                                for i in range(len(gases_s))
                                            ],
                                            texttemplate="%{text}",
                                            colorbar=dict(title="|K|"),
                                        )
                                    )
                                    fig_sel.update_layout(
                                        title="Cross-Sensitivity Matrix  K_{interferent/target}",
                                        xaxis_title="Interferent",
                                        yaxis_title="Target",
                                        height=420,
                                    )
                                    st.plotly_chart(fig_sel, use_container_width=True)
                                    st.text(sel.summary_table())

                                    for gas, interp in sel.interpretation.items():
                                        icon = (
                                            "green"
                                            if "HIGH" in interp
                                            else ("orange" if "MODERATE" in interp else "red")
                                        )
                                        st.markdown(
                                            f"**{gas}**: "
                                            f":{icon}[{interp.split(' | ')[0]}]  "
                                            f"{interp.split(' | ')[1] if ' | ' in interp else ''}"
                                        )
                            except Exception as exc:
                                st.error(f"Selectivity computation failed: {exc}")
                                import traceback as _tb

                                with st.expander("Traceback"):
                                    st.code(_tb.format_exc())

                    # -------------------------------------------------------
                    # MCR-ALS Spectral Deconvolution
                    # -------------------------------------------------------
                    st.markdown("### Spectral Deconvolution (MCR-ALS)")
                    st.caption(
                        "Resolves the spectral mixture matrix into k pure-component "
                        "spectra (S) and their abundance profiles (C) using "
                        "Alternating Least Squares with non-negativity constraints. "
                        "Identifies spectral sub-components in mixed-gas systems."
                    )
                    _mcr_k = st.number_input(
                        "Components (k)",
                        min_value=2,
                        max_value=max(2, min(len(concs), 5)),
                        value=min(2, len(concs)),
                        key="mcr_k",
                    )
                    if st.button("Run MCR-ALS", key="btn_mcr"):
                        try:
                            from gas_analysis.advanced.mcr_als import _als_nnls as _mcr_fn

                            # Shift to non-negative baseline (NNLS requirement)
                            _Z_nn = Z - Z.min(axis=1, keepdims=True)
                            _C_mcr, _S_mcr = _mcr_fn(_Z_nn, int(_mcr_k), random_state=42)
                            _mcr_col1, _mcr_col2 = st.columns(2)
                            with _mcr_col1:
                                fig_mcr_s = go.Figure()
                                for _ki in range(int(_mcr_k)):
                                    fig_mcr_s.add_trace(
                                        go.Scatter(
                                            x=common_wl,
                                            y=_S_mcr[:, _ki],
                                            mode="lines",
                                            name=f"Component {_ki + 1}",
                                        )
                                    )
                                fig_mcr_s.update_layout(
                                    title=f"Pure-Component Spectra — {gas_type}",
                                    xaxis_title="Wavelength (nm)",
                                    yaxis_title="Relative Intensity (a.u.)",
                                    height=360,
                                )
                                st.plotly_chart(fig_mcr_s, use_container_width=True)
                            with _mcr_col2:
                                fig_mcr_c = go.Figure()
                                for _ki in range(int(_mcr_k)):
                                    fig_mcr_c.add_trace(
                                        go.Scatter(
                                            x=concs,
                                            y=_C_mcr[:, _ki],
                                            mode="lines+markers",
                                            name=f"Component {_ki + 1}",
                                        )
                                    )
                                fig_mcr_c.update_layout(
                                    title="Abundance vs Concentration",
                                    xaxis_title="Concentration (ppm)",
                                    yaxis_title="Abundance (a.u.)",
                                    height=360,
                                )
                                st.plotly_chart(fig_mcr_c, use_container_width=True)
                        except Exception as _exc_mcr:
                            st.error(f"MCR-ALS failed: {_exc_mcr}")

                    # -------------------------------------------------------
                    # Analysis Report Export (HTML)
                    # -------------------------------------------------------
                    st.markdown("### Export Analysis Report")
                    from datetime import datetime as _dt

                    _rpt_rows = "".join(
                        f"<tr><td style='padding:6px 14px;border:1px solid #ddd'>{k}</td>"
                        f"<td style='padding:6px 14px;border:1px solid #ddd'><b>{v}</b></td></tr>"
                        for k, v in [
                            ("Gas", gas_type),
                            ("Date", _dt.now().strftime("%Y-%m-%d %H:%M")),
                            ("Concentrations (ppm)", ", ".join(f"{c}" for c in concs)),
                            ("Response variable", response_label),
                            ("Sensitivity", f"{abs(slope):.4g} {sensitivity_unit}"),
                            ("Linearity R²", f"{r_val**2:.4f}"),
                            ("LOD (3σ/S)", lod_str),
                            ("LOQ (10σ/S)", loq_str),
                            ("LOD 95% CI", lod_ci_str),
                            ("Stable-plateau frames", "20-frame tail + MAD gating"),
                        ]
                    )
                    _rpt_html = (
                        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                        f"<title>{gas_type} Calibration Report</title>"
                        "<style>body{font-family:Arial,sans-serif;margin:40px;color:#222}"
                        "h1{color:#1a4f8a}table{border-collapse:collapse;width:560px}"
                        "tr:nth-child(even){background:#f7f7f7}</style></head><body>"
                        "<h1>Gas Sensor Calibration Report</h1>"
                        "<p><em>Generated by Au-MIP LSPR Analysis Platform</em></p>"
                        f"<h2>Summary — {gas_type}</h2>"
                        f"<table>{_rpt_rows}</table>"
                        "</body></html>"
                    )
                    st.download_button(
                        label="Download Report (HTML)",
                        data=_rpt_html.encode("utf-8"),
                        file_name=f"{gas_type}_calibration_{_dt.now():%Y%m%d_%H%M}.html",
                        mime="text/html",
                        key="btn_dl_report",
                    )

                    # -------------------------------------------------------
                    # ONNX Model Export
                    # -------------------------------------------------------
                    st.markdown("### Model Export (ONNX)")
                    st.caption(
                        "Export the trained CNN gas classifier to ONNX format for "
                        "edge deployment (ONNX Runtime, TensorRT, embedded C++)."
                    )
                    if st.button("Export CNN → ONNX", key="btn_onnx"):
                        try:
                            from src.models.cnn import CNNGasClassifier
                            from src.models.onnx_export import (
                                export_cnn_to_onnx,
                                validate_onnx_export,
                            )

                            _cnn_pt = REPO_ROOT / "output" / "models" / "cnn_classifier.pt"
                            _cnn_onnx = REPO_ROOT / "output" / "models" / "cnn_classifier.onnx"
                            if not _cnn_pt.exists():
                                st.error(
                                    f"CNN checkpoint not found at `{_cnn_pt}`. "
                                    "Train a model first via Tab 1 (Automation Pipeline)."
                                )
                            else:
                                _clf = CNNGasClassifier.load(str(_cnn_pt))
                                _out = export_cnn_to_onnx(_clf, str(_cnn_onnx))
                                _ok, _delta = validate_onnx_export(_clf, _out)
                                if _ok:
                                    st.success(
                                        f"ONNX export validated — max output Δ = {_delta:.2e}. "
                                        f"Saved to `{_out}`"
                                    )
                                else:
                                    st.warning(
                                        f"Exported to `{_out}` but validation delta is high "
                                        f"({_delta:.2e}). Check opset compatibility."
                                    )
                        except Exception as _exc_onnx:
                            st.error(f"ONNX export failed: {_exc_onnx}")
                            with st.expander("Traceback"):
                                import traceback as _tb_onnx

                                st.code(_tb_onnx.format_exc())

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    log.error("Scientific analysis failed: %s", exc, exc_info=True)
                    with st.expander("Traceback"):
                        import traceback

                        st.code(traceback.format_exc())



with tab_batch:
    _render_batch()

# ===========================================================================
# Tab 4 — Live Sensor
# ===========================================================================
with tab_live:
    try:
        from dashboard.sensor_dashboard import render as _render_live  # type: ignore[import]

        _render_live(live_server_port=_LIVE_SERVER_PORT)
    except ImportError as exc:
        st.error(f"Live sensor module not available: {exc}")
        log.warning("sensor_dashboard import failed: %s", exc)
    except Exception as exc:
        st.error(f"Live sensor error: {exc}")
        log.error("Live sensor error: %s", exc, exc_info=True)
        with st.expander("Traceback"):
            import traceback

            st.code(traceback.format_exc())
