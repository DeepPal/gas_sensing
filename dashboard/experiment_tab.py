"""
Experiment Tab — Guided Research Pipeline
------------------------------------------
4-step guided workflow for gas sensing experiments:

  Step 1  Configure   — gas name, integration time, frames to average, output dir
  Step 2  Reference   — capture baseline (air/blank) from CCS200 or upload CSV
  Step 3  Sample      — capture analyte from CCS200 or upload CSV
  Step 4  Results     — automatic analysis: SNR, peak shift, calibration → ppm

Session state keys used (all prefixed "exp_"):
  exp_step          int 1-4
  exp_config        dict  (gas, integration_ms, n_frames, output_dir, label)
  exp_reference     dict  (wavelengths, intensities, source, timestamp)
  exp_sample        dict  (wavelengths, intensities, source, timestamp)
  exp_result        dict  (analysis outputs)
"""

from __future__ import annotations

import contextlib
from datetime import datetime
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── optional hardware import ──────────────────────────────────────────────────
try:
    import sys

    _REPO = Path(__file__).resolve().parents[1]
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    from src.acquisition import CCS200Spectrometer

    if CCS200Spectrometer is None:
        raise ImportError("CCS200Spectrometer unavailable")
    _HW_AVAILABLE = True
except Exception:
    _HW_AVAILABLE = False

# ── signal processing imports ─────────────────────────────────────────────────
try:
    from src.preprocessing.baseline import als_baseline
    from src.preprocessing.denoising import smooth_spectrum

    _SP_AVAILABLE = True
except Exception:
    try:
        from gas_analysis.core.signal_proc import als_baseline, smooth_spectrum

        _SP_AVAILABLE = True
    except Exception:
        _SP_AVAILABLE = False

# ── sub-pixel peak detection (Lorentzian fit) ─────────────────────────────────
try:
    from src.features.lspr_features import detect_lspr_peak as _detect_peak

    _PEAK_LORENTZ_AVAILABLE = True
except Exception:
    _PEAK_LORENTZ_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _step_badge(n: int, label: str, done: bool, active: bool) -> str:
    colour = "#2ecc71" if done else ("#1f77b4" if active else "#888")
    icon = "✓" if done else str(n)
    return (
        f'<span style="background:{colour};color:white;border-radius:50%;'
        f'padding:2px 8px;font-weight:bold;margin-right:6px">{icon}</span>'
        f'<span style="color:{colour};font-weight:{"bold" if active else "normal"}">{label}</span>'
    )


def _csv_to_spectrum(uploaded_file) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse an uploaded CSV into (wavelengths, intensities)."""
    try:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[1] >= 2:
            wl = df.iloc[:, 0].astype(float).values
            it = df.iloc[:, 1].astype(float).values
        elif df.shape[1] == 1:
            it = df.iloc[:, 0].astype(float).values
            wl = np.linspace(196, 1019, len(it))
        else:
            return None
        return wl, it
    except Exception:
        return None


def _acquire_from_ccs200(
    integration_ms: int, n_frames: int, chart_placeholder=None
) -> tuple[np.ndarray, np.ndarray]:
    """Acquire n_frames from CCS200 and return (wavelengths, mean_intensities).

    Handles stale VISA handle errors (code -1073807339 = VI_ERROR_INV_OBJECT)
    by closing and reinitializing the device once on failure.
    """

    def _open_spec():
        s = CCS200Spectrometer(integration_time_s=integration_ms / 1000.0)
        return s, s.get_wavelengths()

    spec, wl = _open_spec()
    try:
        frames = []
        progress = st.progress(0, text="Acquiring spectra…")

        for i in range(n_frames):
            # --- Attempt get_data with one auto-retry on stale-handle error ---
            for attempt in range(2):
                try:
                    frame_data = spec.get_data()
                    break  # success
                except RuntimeError as exc:
                    if attempt == 0 and "getScanData" in str(exc):
                        # Stale VISA handle — close and reopen
                        with contextlib.suppress(Exception):
                            spec.close()
                        import time as _time

                        _time.sleep(0.5)
                        spec, wl = _open_spec()
                    else:
                        raise  # second failure → surface to user

            frames.append(frame_data)

            # Live plot update
            if chart_placeholder is not None:
                current_mean = np.mean(frames, axis=0)
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=wl,
                        y=current_mean,
                        mode="lines",
                        name=f"Live Average ({i + 1}/{n_frames})",
                        line=dict(color="#FF4B4B", width=1.5),
                    )
                )
                fig.update_layout(
                    title=f"Live Acquisition (Frame {i + 1} of {n_frames})",
                    xaxis_title="Wavelength (nm)",
                    yaxis_title="Intensity (a.u.)",
                    height=300,
                    margin=dict(t=40, b=40, l=40, r=20),
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

            progress.progress((i + 1) / n_frames, text=f"Frame {i + 1}/{n_frames}")

        progress.empty()
        return wl, np.mean(frames, axis=0)
    finally:
        with contextlib.suppress(Exception):
            spec.close()


def _noise_mask_for_region(
    wl: np.ndarray,
    region: str,
    peak_wl: float | None = None,
) -> np.ndarray:
    """Return a boolean mask for the noise-estimation region.

    Parameters
    ----------
    wl : wavelength axis
    region : one of the three UI choices
    peak_wl : sensor peak wavelength (nm); used by 'auto' to pick the tail
        furthest from the peak
    """
    wl_range = float(wl[-1] - wl[0])

    def _low_tail() -> np.ndarray:
        m = wl <= (float(wl[0]) + 0.10 * wl_range)
        return m if m.sum() > 5 else _high_tail()

    def _high_tail() -> np.ndarray:
        m = wl >= (float(wl[-1]) - 0.10 * wl_range)
        return m if m.sum() > 5 else _fallback()

    def _fallback() -> np.ndarray:
        idx = np.concatenate(
            [np.arange(min(5, len(wl))), np.arange(max(0, len(wl) - 5), len(wl))]
        )
        m = np.zeros(len(wl), dtype=bool)
        m[idx] = True
        return m

    if region == "Low-end tail (first 10 %)":
        return _low_tail()
    if region == "High-end tail (last 10 %)":
        return _high_tail()
    # Auto: pick the tail whose centre is furthest from the sensor peak
    if peak_wl is not None:
        low_centre = float(wl[0]) + 0.05 * wl_range
        high_centre = float(wl[-1]) - 0.05 * wl_range
        if abs(low_centre - peak_wl) >= abs(high_centre - peak_wl):
            return _low_tail()
        return _high_tail()
    return _low_tail()


def _analyse(cfg: dict, ref: dict, smp: dict) -> dict:
    """
    General spectrometer sensor analysis — works for any sensor type
    (LSPR, absorbance, fluorescence, etc.) as long as the response is a
    peak wavelength shift.

    Steps:
      1. Baseline-correct both spectra (ALS)
      2. Smooth (Savitzky-Golay)
      3. Absorbance = log10(ref / sample)  [Beer-Lambert extinction change]
      4. Detect sensor response peak within the configured search window
      5. Compute SNR using the user-selected noise region
      6. Compute Δλ (wavelength shift) between ref and sample peaks
      7. Convert Δλ → concentration via signed calibration slope
    """
    wl = ref["wavelengths"]
    r = ref["intensities"].copy().astype(float)
    s = smp["intensities"].copy().astype(float)

    # ── baseline correction (ALS) ─────────────────────────────────────────
    if _SP_AVAILABLE:
        try:
            r_bc = r - als_baseline(r, lam=1e5, p=0.01)
            s_bc = s - als_baseline(s, lam=1e5, p=0.01)
        except Exception:
            r_bc, s_bc = r, s
    else:
        r_bc, s_bc = r, s

    # ── smooth ────────────────────────────────────────────────────────────
    if _SP_AVAILABLE:
        try:
            r_sm = smooth_spectrum(r_bc, window=11, poly_order=2)
            s_sm = smooth_spectrum(s_bc, window=11, poly_order=2)
        except Exception:
            r_sm, s_sm = r_bc, s_bc
    else:
        r_sm, s_sm = r_bc, s_bc

    # ── absorbance (extinction change) ────────────────────────────────────
    eps = 1e-9
    r_pos = np.clip(r_sm, eps, None)
    s_pos = np.clip(s_sm, eps, None)
    absorbance = np.log10(r_pos / s_pos)

    # ── wavelength shift — within configured sensor peak search window ────
    # Use the user-specified window to avoid picking up UV/NIR artefacts
    # from the spectrometer or analyte absorptions outside the sensor band.
    peak_min = float(cfg.get("peak_search_min", wl[0]))
    peak_max = float(cfg.get("peak_search_max", wl[-1]))
    win_mask = (wl >= peak_min) & (wl <= peak_max)

    if win_mask.sum() >= 3:
        wl_win = wl[win_mask]
        # Lorentzian sub-pixel peak detection (±0.01 nm vs ±0.5 nm argmax).
        # detect_lspr_peak internally falls back to argmax when Lorentzian fit fails.
        if _PEAK_LORENTZ_AVAILABLE:
            _rp = _detect_peak(wl, r_sm, peak_min, peak_max)
            _sp = _detect_peak(wl, s_sm, peak_min, peak_max)
            ref_peak_wl = _rp if _rp is not None else float(wl_win[int(np.argmax(r_sm[win_mask]))])
            smp_peak_wl = _sp if _sp is not None else float(wl_win[int(np.argmax(s_sm[win_mask]))])
        else:
            ref_peak_wl = float(wl_win[int(np.argmax(r_sm[win_mask]))])
            smp_peak_wl = float(wl_win[int(np.argmax(s_sm[win_mask]))])
        # Absorbance peak: largest extinction change in window
        abs_peak_idx_win = int(np.argmax(np.abs(absorbance[win_mask])))
        peak_wl = float(wl_win[abs_peak_idx_win])
        peak_abs = float(absorbance[win_mask][abs_peak_idx_win])
    else:
        # Fallback: full spectrum if window covers no points
        if _PEAK_LORENTZ_AVAILABLE:
            _rp = _detect_peak(wl, r_sm)
            _sp = _detect_peak(wl, s_sm)
            ref_peak_wl = _rp if _rp is not None else float(wl[int(np.argmax(r_sm))])
            smp_peak_wl = _sp if _sp is not None else float(wl[int(np.argmax(s_sm))])
        else:
            ref_peak_wl = float(wl[int(np.argmax(r_sm))])
            smp_peak_wl = float(wl[int(np.argmax(s_sm))])
        peak_idx_fb = int(np.argmax(np.abs(absorbance)))
        peak_wl = float(wl[peak_idx_fb])
        peak_abs = float(absorbance[peak_idx_fb])

    wl_shift_nm = smp_peak_wl - ref_peak_wl

    # ── SNR — noise estimated in user-selected spectral region ────────────
    noise_region = cfg.get("noise_region", "Auto (furthest from peak)")
    noise_mask = _noise_mask_for_region(wl, noise_region, peak_wl=ref_peak_wl)
    noise_std = float(np.std(absorbance[noise_mask])) if noise_mask.sum() > 1 else 1e-9
    noise_std = max(noise_std, 1e-9)
    snr = abs(peak_abs) / noise_std

    # ── concentration from wavelength shift (signed slope) ─────────────────
    # Handles both red-shift (positive slope) and blue-shift (negative slope)
    # sensors. slope = Δλ / Δconcentration — negative for blue-shift sensors.
    slope = float(cfg.get("calibration_slope", -0.116))
    if abs(slope) > 1e-12:
        conc_ppm = float(max(0.0, wl_shift_nm / slope))
    else:
        conc_ppm = 0.0

    return dict(
        wavelengths=wl,
        ref_corrected=r_sm,
        smp_corrected=s_sm,
        absorbance=absorbance,
        peak_wavelength_nm=peak_wl,
        peak_absorbance=peak_abs,
        snr=snr,
        noise_std=noise_std,
        ref_peak_wl=ref_peak_wl,
        smp_peak_wl=smp_peak_wl,
        wavelength_shift_nm=wl_shift_nm,
        concentration_ppm=conc_ppm,
        calibration_slope=slope,
        peak_search_min=peak_min,
        peak_search_max=peak_max,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────────────────


def render() -> None:
    st.header("🧪 Gas Sensing Experiment")
    st.caption("Guided pipeline: Configure → Reference → Sample → Results")

    # ── init session state ────────────────────────────────────────────────
    ss = st.session_state
    ss.setdefault("exp_step", 1)
    ss.setdefault("exp_config", {})
    ss.setdefault("exp_reference", None)
    ss.setdefault("exp_sample", None)
    ss.setdefault("exp_result", None)
    ss.setdefault("exp_session_runs", [])  # accumulates runs across experiments

    step = ss["exp_step"]

    # ── progress header ───────────────────────────────────────────────────
    cols = st.columns(4)
    labels = ["1  Configure", "2  Reference", "3  Sample", "4  Results"]
    for i, (col, lbl) in enumerate(zip(cols, labels), 1):
        done = step > i
        active = step == i
        col.markdown(_step_badge(i, lbl, done, active), unsafe_allow_html=True)
    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Configure
    # ════════════════════════════════════════════════════════════════════════
    if step == 1:
        st.subheader("Step 1 — Configure Experiment")

        c1, c2 = st.columns(2)
        with c1:
            gas = st.selectbox(
                "Analyte / Gas",
                ["Ethanol", "Methanol", "Isopropanol", "MixVOC", "Custom"],
                help="Used for labelling saved files",
            )
            if gas == "Custom":
                gas = st.text_input("Enter gas name")
            concentration = st.number_input(
                "Known Concentration (ppm)",
                min_value=0.0,
                value=100.0,
                step=10.0,
                help="The tested concentration of the analyte",
            )
            label = st.text_input(
                "Experiment label (optional)",
                placeholder="e.g. run1_0.5ppm",
                help="Added to output filename for identification",
            )
        with c2:
            integration_ms = st.slider(
                "Integration time (ms)", 10, 5000, 50, 10, help="CCS200 exposure time per frame"
            )
            n_frames = st.slider(
                "Frames to average", 1, 50, 10, help="More frames = lower noise (each ~410 ms)"
            )
            st.info(f"Estimated acquisition time: **{n_frames * 0.41:.0f} s** per measurement")

        _REPO = Path(__file__).resolve().parents[1]
        output_dir = st.text_input(
            "Output directory",
            str(_REPO / "output" / "experiments"),
            help="Where reference, sample, and result CSVs are saved",
        )

        # Sensor & calibration settings
        with st.expander("Sensor & Calibration Settings", expanded=True):
            st.caption(
                "Configure these for your specific sensor. "
                "Works for any spectrometer-based sensor (LSPR, absorbance, fluorescence, etc.)."
            )
            _sc1, _sc2 = st.columns(2)
            with _sc1:
                peak_search_min = st.number_input(
                    "Peak search min (nm)",
                    min_value=100.0,
                    max_value=2000.0,
                    value=480.0,
                    step=10.0,
                    help="Lower bound of the wavelength window to search for the sensor response peak.",
                )
                peak_search_max = st.number_input(
                    "Peak search max (nm)",
                    min_value=100.0,
                    max_value=2000.0,
                    value=800.0,
                    step=10.0,
                    help="Upper bound of the wavelength window to search for the sensor response peak.",
                )
            with _sc2:
                slope = st.number_input(
                    "Calibration slope (nm / ppm)",
                    value=-0.116,
                    format="%.4f",
                    help=(
                        "Slope of your calibration curve (Δλ / ppm). "
                        "**Negative** if your sensor peak shifts to shorter wavelengths on gas exposure "
                        "(blue-shift). **Positive** if the peak shifts to longer wavelengths (red-shift)."
                    ),
                )
                noise_region = st.selectbox(
                    "SNR noise region",
                    ["Low-end tail (first 10 %)", "High-end tail (last 10 %)", "Auto (furthest from peak)"],
                    index=2,
                    help="Spectral region used to estimate baseline noise for SNR. "
                    "Choose a region away from both the sensor peak and any analyte absorptions.",
                )

        if st.button("→ Next: Capture Reference", type="primary", disabled=not gas):
            ss["exp_config"] = dict(
                gas=gas,
                concentration=concentration,
                label=label,
                integration_ms=integration_ms,
                n_frames=n_frames,
                output_dir=output_dir,
                calibration_slope=slope,
                peak_search_min=peak_search_min,
                peak_search_max=peak_search_max,
                noise_region=noise_region,
            )
            ss["exp_step"] = 2
            ss["exp_reference"] = None
            ss["exp_sample"] = None
            ss["exp_result"] = None
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Reference spectrum
    # ════════════════════════════════════════════════════════════════════════
    elif step == 2:
        cfg = ss["exp_config"]
        st.subheader("Step 2 — Capture Reference Spectrum (Air / Blank)")
        st.caption(
            f"Gas: **{cfg['gas']}** | Integration: **{cfg['integration_ms']} ms** | "
            f"Frames: **{cfg['n_frames']}**"
        )

        src = st.radio("Source", ["CCS200 Spectrometer", "Upload CSV file"], horizontal=True)

        if src == "CCS200 Spectrometer":
            if not _HW_AVAILABLE:
                st.error("CCS200 driver not available — check TLCCS_64.dll installation.")
            else:
                st.info(
                    "Ensure the sensing cell contains **air / blank** (no analyte). "
                    "Click **Acquire** when ready."
                )
                if st.button("Acquire Reference from CCS200", type="primary"):
                    try:
                        with st.spinner("Connecting and acquiring…"):
                            live_chart = st.empty()
                            wl, mean_int = _acquire_from_ccs200(
                                cfg["integration_ms"], cfg["n_frames"], chart_placeholder=live_chart
                            )
                        ss["exp_reference"] = dict(
                            wavelengths=wl,
                            intensities=mean_int,
                            source="CCS200",
                            timestamp=datetime.now().isoformat(),
                        )
                        st.success(
                            f"Reference acquired: {len(wl)} pixels, max={mean_int.max():.4f}"
                        )
                    except Exception as exc:
                        st.error(f"Acquisition failed: {exc}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())

        else:  # CSV upload
            f = st.file_uploader("Upload reference CSV (wavelength, intensity)", type="csv")
            if f:
                result = _csv_to_spectrum(f)
                if result:
                    wl, it = result
                    ss["exp_reference"] = dict(
                        wavelengths=wl,
                        intensities=it,
                        source=f.name,
                        timestamp=datetime.now().isoformat(),
                    )
                    st.success(f"Loaded: {len(wl)} pixels, max={it.max():.4f}")
                else:
                    st.error("Could not parse CSV. Expected two columns: wavelength, intensity.")

        # ── preview ───────────────────────────────────────────────────────
        if ss["exp_reference"] is not None:
            ref = ss["exp_reference"]
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=ref["wavelengths"],
                    y=ref["intensities"],
                    name="Reference",
                    line=dict(color="#2196F3", width=1.5),
                )
            )
            fig.update_layout(
                title="Reference Spectrum",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                height=300,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Pixels", len(ref["wavelengths"]))
            col2.metric("Peak intensity", f"{ref['intensities'].max():.4f}")

            st.markdown("---")
            bcol1, bcol2 = st.columns([1, 4])
            with bcol1:
                if st.button("← Back"):
                    ss["exp_step"] = 1
                    st.rerun()
            with bcol2:
                if st.button("→ Next: Capture Sample", type="primary"):
                    ss["exp_step"] = 3
                    st.rerun()

        elif src == "CCS200 Spectrometer":
            pass  # waiting for user to click Acquire

        # Back button when no data yet
        if ss["exp_reference"] is None and st.button("← Back to Configure"):
            ss["exp_step"] = 1
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Sample spectrum
    # ════════════════════════════════════════════════════════════════════════
    elif step == 3:
        cfg = ss["exp_config"]
        st.subheader("Step 3 — Capture Sample Spectrum (With Analyte)")
        st.caption(
            f"Gas: **{cfg['gas']}** | Integration: **{cfg['integration_ms']} ms** | "
            f"Frames: **{cfg['n_frames']}**"
        )

        src = st.radio("Source", ["CCS200 Spectrometer", "Upload CSV file"], horizontal=True)

        if src == "CCS200 Spectrometer":
            if not _HW_AVAILABLE:
                st.error("CCS200 driver not available — check TLCCS_64.dll installation.")
            else:
                st.info(
                    f"Introduce **{cfg['gas']}** into the sensing cell. "
                    "Click **Acquire** when the gas is present."
                )
                if st.button("Acquire Sample from CCS200", type="primary"):
                    try:
                        with st.spinner("Connecting and acquiring…"):
                            live_chart = st.empty()
                            wl, mean_int = _acquire_from_ccs200(
                                cfg["integration_ms"], cfg["n_frames"], chart_placeholder=live_chart
                            )
                        ss["exp_sample"] = dict(
                            wavelengths=wl,
                            intensities=mean_int,
                            source="CCS200",
                            timestamp=datetime.now().isoformat(),
                        )
                        st.success(f"Sample acquired: max={mean_int.max():.4f}")
                    except Exception as exc:
                        st.error(f"Acquisition failed: {exc}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())

        else:
            f = st.file_uploader("Upload sample CSV (wavelength, intensity)", type="csv")
            if f:
                result = _csv_to_spectrum(f)
                if result:
                    wl, it = result
                    ss["exp_sample"] = dict(
                        wavelengths=wl,
                        intensities=it,
                        source=f.name,
                        timestamp=datetime.now().isoformat(),
                    )
                    st.success(f"Loaded: {len(wl)} pixels, max={it.max():.4f}")
                else:
                    st.error("Could not parse CSV.")

        # ── overlay preview ───────────────────────────────────────────────
        if ss["exp_sample"] is not None:
            smp = ss["exp_sample"]
            ref = ss["exp_reference"]
            fig = go.Figure()
            if ref:
                fig.add_trace(
                    go.Scatter(
                        x=ref["wavelengths"],
                        y=ref["intensities"],
                        name="Reference (air)",
                        line=dict(color="#2196F3", width=1.5, dash="dot"),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=smp["wavelengths"],
                    y=smp["intensities"],
                    name=f"Sample ({cfg['gas']})",
                    line=dict(color="#FF5722", width=1.5),
                )
            )
            fig.update_layout(
                title="Reference vs Sample",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity (a.u.)",
                height=320,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            bcol1, bcol2 = st.columns([1, 4])
            with bcol1:
                if st.button("← Back"):
                    ss["exp_step"] = 2
                    st.rerun()
            with bcol2:
                if st.button("→ Analyse Results", type="primary"):
                    # Run analysis
                    try:
                        ss["exp_result"] = _analyse(cfg, ss["exp_reference"], smp)
                        ss["exp_step"] = 4
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analysis error: {exc}")
                        with st.expander("Details"):
                            st.code(traceback.format_exc())

        if ss["exp_sample"] is None and st.button("← Back to Reference"):
            ss["exp_step"] = 2
            st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Results
    # ════════════════════════════════════════════════════════════════════════
    elif step == 4:
        cfg = ss["exp_config"]
        res = ss["exp_result"]
        if res is None:
            st.error("No results — go back to Step 3.")
            if st.button("← Back"):
                ss["exp_step"] = 3
                st.rerun()
            return

        st.subheader("Step 4 — Analysis Results")
        _ref_src = (ss.get("exp_reference") or {}).get("source", "N/A")
        _smp_src = (ss.get("exp_sample") or {}).get("source", "N/A")
        st.caption(
            f"Gas: **{cfg['gas']}** | "
            f"Ref: {_ref_src} | "
            f"Sample: {_smp_src}"
        )

        # ── key metrics row 1 ─────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Estimated concentration",
            f"{res['concentration_ppm']:.2f} ppm",
            help="From calibration: Δλ / slope",
        )
        m2.metric(
            "Wavelength shift Δλ",
            f"{res['wavelength_shift_nm']:+.4f} nm",
            help="Sample peak − Reference peak (within search window)",
        )
        m3.metric(
            "Sample peak",
            f"{res['smp_peak_wl']:.3f} nm",
            help=f"Reference peak: {res['ref_peak_wl']:.3f} nm",
        )
        m4.metric(
            "SNR",
            f"{res['snr']:.1f}",
            delta="Good" if res["snr"] > 10 else ("Marginal" if res["snr"] > 3 else "Low"),
            delta_color="normal" if res["snr"] > 10 else "inverse",
        )

        # ── accuracy row (known vs estimated) ────────────────────────────
        _known_conc = float(cfg.get("concentration", 0.0))
        if _known_conc > 0:
            _delta_conc = res["concentration_ppm"] - _known_conc
            _pct_err = _delta_conc / _known_conc * 100
            a1, a2, a3 = st.columns(3)
            a1.metric("Known concentration", f"{_known_conc:.2f} ppm")
            a2.metric(
                "Estimated concentration",
                f"{res['concentration_ppm']:.2f} ppm",
                delta=f"{_delta_conc:+.3f} ppm",
                delta_color="normal" if abs(_pct_err) < 10 else "inverse",
            )
            a3.metric(
                "Relative error",
                f"{abs(_pct_err):.1f} %",
                delta="< 10 % (acceptable)" if abs(_pct_err) < 10 else "> 10 % (review calibration)",
                delta_color="normal" if abs(_pct_err) < 10 else "inverse",
            )

        # ── LOD / LOQ single-point estimate ──────────────────────────────
        try:
            from src.scientific.lod import calculate_lod_3sigma, calculate_loq_10sigma

            _lod_sp = calculate_lod_3sigma(res["noise_std"], res["calibration_slope"])
            _loq_sp = calculate_loq_10sigma(res["noise_std"], res["calibration_slope"])
            l1, l2 = st.columns(2)
            l1.metric(
                "Est. LOD (3σ/S)",
                f"{_lod_sp:.3f} ppm" if _lod_sp < 1e6 else "N/A",
                help="Single-point IUPAC LOD = 3σ_noise / sensitivity. "
                "For a calibration-based LOD use the Batch Analysis tab.",
            )
            l2.metric(
                "Est. LOQ (10σ/S)",
                f"{_loq_sp:.3f} ppm" if _loq_sp < 1e6 else "N/A",
                help="Single-point IUPAC LOQ = 10σ_noise / sensitivity.",
            )
        except Exception:
            pass

        # ── corrected spectra overlay (primary view) ─────────────────────
        # Show ref vs sample peaks directly — this is the primary signal
        # for any spectrometer-based sensor (the intensity + peak position).
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=res["wavelengths"],
                y=res["ref_corrected"],
                name="Reference (baseline-corrected)",
                line=dict(color="#2196F3", width=1.5, dash="dot"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=res["wavelengths"],
                y=res["smp_corrected"],
                name=f"Sample — {cfg['gas']} (baseline-corrected)",
                line=dict(color="#FF5722", width=1.5),
            )
        )
        # Mark the detected peaks within the search window
        fig2.add_vline(
            x=res["ref_peak_wl"],
            line_dash="dot",
            line_color="#2196F3",
            annotation_text=f"Ref {res['ref_peak_wl']:.2f} nm",
            annotation_position="top left",
        )
        fig2.add_vline(
            x=res["smp_peak_wl"],
            line_dash="dash",
            line_color="#FF5722",
            annotation_text=f"Sample {res['smp_peak_wl']:.2f} nm",
            annotation_position="top right",
        )
        # Shade the configured peak search window
        fig2.add_vrect(
            x0=res["peak_search_min"],
            x1=res["peak_search_max"],
            fillcolor="rgba(100,200,100,0.07)",
            line_width=0,
            annotation_text="Search window",
            annotation_position="top left",
        )
        _shift_sign = "+" if res["wavelength_shift_nm"] >= 0 else ""
        fig2.update_layout(
            title=(
                f"Spectra — Δλ = {_shift_sign}{res['wavelength_shift_nm']:.4f} nm  "
                f"({cfg['gas']})"
            ),
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity (a.u.)",
            height=360,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── absorbance spectrum (secondary — extinction change) ───────────
        with st.expander("Absorbance spectrum  [log₁₀(Reference / Sample)]"):
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=res["wavelengths"],
                    y=res["absorbance"],
                    name="Absorbance",
                    line=dict(color="#9C27B0", width=1.5),
                )
            )
            fig.add_vline(
                x=res["peak_wavelength_nm"],
                line_dash="dash",
                line_color="#FF5722",
                annotation_text=f"Max ΔA {res['peak_wavelength_nm']:.1f} nm",
                annotation_position="top right",
            )
            fig.add_vrect(
                x0=res["peak_search_min"],
                x1=res["peak_search_max"],
                fillcolor="rgba(100,200,100,0.07)",
                line_width=0,
            )
            fig.update_layout(
                xaxis_title="Wavelength (nm)",
                yaxis_title="Absorbance (a.u.)",
                height=280,
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── calibration info ──────────────────────────────────────────────
        with st.expander("Calibration & measurement details"):
            _noise_lbl = cfg.get("noise_region", "Auto")
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Calibration slope | {res["calibration_slope"]:+.4f} nm/ppm |
| Reference peak | {res["ref_peak_wl"]:.3f} nm |
| Sample peak | {res["smp_peak_wl"]:.3f} nm |
| Wavelength shift Δλ | {res["wavelength_shift_nm"]:+.4f} nm |
| Concentration (linear) | {res["concentration_ppm"]:.3f} ppm |
| Noise std ({_noise_lbl}) | {res["noise_std"]:.6f} |
| SNR | {res["snr"]:.1f} |
| Peak search window | {res["peak_search_min"]:.0f} – {res["peak_search_max"]:.0f} nm |
""")
            if res["snr"] < 3:
                st.warning(
                    "SNR < 3 — consider increasing integration time or number of frames. "
                    "Results may not be reliable."
                )
            elif res["snr"] < 10:
                st.info(
                    "SNR 3–10 — acceptable for screening. For publication-grade data, aim for SNR > 20."
                )
            else:
                st.success("SNR > 10 — good signal quality.")

        # ── save results ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Export")

        safe_gas = cfg["gas"].replace(" ", "_")
        safe_conc = f"{cfg.get('concentration', 0.0)}ppm"

        # Nested directory structure: output_dir / gas / concentration /
        out_root = Path(cfg["output_dir"])
        out_dir = out_root / safe_gas / safe_conc

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_lbl = ("_" + cfg["label"].replace(" ", "_")) if cfg.get("label") else ""
        stem = f"{safe_gas}{safe_lbl}_{ts}"

        if st.button("Save Results to Disk", type="primary"):
            try:
                out_dir.mkdir(parents=True, exist_ok=True)

                # Raw spectra CSVs
                ref_df = pd.DataFrame(
                    {
                        "wavelength_nm": ss["exp_reference"]["wavelengths"],
                        "intensity": ss["exp_reference"]["intensities"],
                    }
                )
                smp_df = pd.DataFrame(
                    {
                        "wavelength_nm": ss["exp_sample"]["wavelengths"],
                        "intensity": ss["exp_sample"]["intensities"],
                    }
                )
                abs_df = pd.DataFrame(
                    {
                        "wavelength_nm": res["wavelengths"],
                        "absorbance": res["absorbance"],
                    }
                )
                ref_df.to_csv(out_dir / f"{stem}_reference.csv", index=False)
                smp_df.to_csv(out_dir / f"{stem}_sample.csv", index=False)
                abs_df.to_csv(out_dir / f"{stem}_absorbance.csv", index=False)

                # JSON summary
                summary = {
                    "timestamp": ts,
                    "gas": cfg["gas"],
                    "label": cfg.get("label", ""),
                    "integration_ms": cfg["integration_ms"],
                    "n_frames": cfg["n_frames"],
                    "ref_source": ss["exp_reference"]["source"],
                    "smp_source": ss["exp_sample"]["source"],
                    "ref_peak_wl_nm": res["ref_peak_wl"],
                    "smp_peak_wl_nm": res["smp_peak_wl"],
                    "peak_wavelength_nm": res["peak_wavelength_nm"],
                    "peak_absorbance": res["peak_absorbance"],
                    "wavelength_shift_nm": res["wavelength_shift_nm"],
                    "concentration_ppm": res["concentration_ppm"],
                    "calibration_slope_nm_per_ppm": res["calibration_slope"],
                    "peak_search_min_nm": res["peak_search_min"],
                    "peak_search_max_nm": res["peak_search_max"],
                    "snr": res["snr"],
                    "noise_std": res["noise_std"],
                }
                with open(out_dir / f"{stem}_result.json", "w") as fh:
                    json.dump(summary, fh, indent=2)

                st.success(f"Saved to `{out_dir}/`")
                st.markdown(f"""
Files written:
- `{stem}_reference.csv`
- `{stem}_sample.csv`
- `{stem}_absorbance.csv`
- `{stem}_result.json`
""")
            except Exception as exc:
                st.error(f"Save failed: {exc}")

        # ── in-browser download ───────────────────────────────────────────
        result_json = json.dumps(
            {
                "gas": cfg["gas"],
                "timestamp": ts,
                "concentration_ppm": res["concentration_ppm"],
                "wavelength_shift_nm": res["wavelength_shift_nm"],
                "snr": res["snr"],
                "peak_wavelength_nm": res["peak_wavelength_nm"],
            },
            indent=2,
        )
        st.download_button(
            "Download Result JSON",
            data=result_json,
            file_name=f"{stem}_result.json",
            mime="application/json",
        )

        # ── multi-run session tracker ─────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Multi-Run Session Tracker")
        st.caption(
            "Log each measurement to build a calibration curve and compute "
            "RSD / reproducibility across repeated runs. Runs persist across experiments."
        )

        _known_ppm = float(cfg.get("concentration", 0.0))
        _est_ppm = float(res["concentration_ppm"])
        _run_entry = {
            "gas": cfg["gas"],
            "label": cfg.get("label", ""),
            "conc_known_ppm": _known_ppm,
            "conc_est_ppm": _est_ppm,
            "error_ppm": round(_est_ppm - _known_ppm, 4) if _known_ppm > 0 else None,
            "error_pct": round((_est_ppm - _known_ppm) / _known_ppm * 100, 2) if _known_ppm > 0 else None,
            "wl_shift_nm": float(res["wavelength_shift_nm"]),
            "ref_peak_nm": float(res["ref_peak_wl"]),
            "smp_peak_nm": float(res["smp_peak_wl"]),
            "snr": float(res["snr"]),
            "noise_std": float(res["noise_std"]),
            "peak_win": f"{res['peak_search_min']:.0f}–{res['peak_search_max']:.0f} nm",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        _log_col, _clear_col = st.columns([3, 1])
        with _log_col:
            if st.button("+ Log This Run to Session", type="primary"):
                ss["exp_session_runs"].append(_run_entry)
                st.rerun()
        with _clear_col:
            if st.button("Clear Session"):
                ss["exp_session_runs"] = []
                st.rerun()

        if ss["exp_session_runs"]:
            _session_df = pd.DataFrame(ss["exp_session_runs"])
            _gas_runs = _session_df[_session_df["gas"] == cfg["gas"]].copy()

            if not _gas_runs.empty:
                st.markdown(
                    f"**{cfg['gas']}** — {len(_gas_runs)} run(s) logged "
                    f"({_session_df['gas'].nunique()} gas(es) total in session)"
                )
                # Display log table
                with st.expander("Run log", expanded=True):
                    _display_cols = {
                        "conc_known_ppm": "Known (ppm)",
                        "conc_est_ppm": "Est. (ppm)",
                        "error_ppm": "Err (ppm)",
                        "error_pct": "Err (%)",
                        "wl_shift_nm": "Δλ (nm)",
                        "ref_peak_nm": "Ref peak (nm)",
                        "smp_peak_nm": "Smp peak (nm)",
                        "peak_win": "Search window",
                        "snr": "SNR",
                        "label": "Label",
                        "timestamp": "Time",
                    }
                    st.dataframe(
                        _gas_runs[list(_display_cols)].rename(columns=_display_cols),
                        use_container_width=True,
                        hide_index=True,
                    )

                # Per-concentration statistics
                _grp = (
                    _gas_runs.groupby("conc_known_ppm")["wl_shift_nm"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                _grp.columns = pd.Index(["Conc (ppm)", "Mean Δλ (nm)", "Std Δλ (nm)", "n"])
                _grp["RSD (%)"] = (
                    (_grp["Std Δλ (nm)"] / _grp["Mean Δλ (nm)"].abs() * 100)
                    .where(_grp["n"] > 1)
                    .round(2)
                )
                _grp["Std Δλ (nm)"] = _grp["Std Δλ (nm)"].where(_grp["n"] > 1)

                st.markdown("**Reproducibility per concentration**")
                st.dataframe(_grp.round(4), use_container_width=True, hide_index=True)

                _valid_rsd = _grp["RSD (%)"].dropna()
                if not _valid_rsd.empty:
                    st.caption(
                        f"Mean RSD: **{_valid_rsd.mean():.1f} %** | "
                        f"Max: **{_valid_rsd.max():.1f} %** "
                        "(< 5 % is good reproducibility for LSPR sensing)"
                    )

                # Calibration curve from session if ≥ 2 distinct concentrations
                _unique_concs = _grp["Conc (ppm)"].dropna()
                if len(_unique_concs) >= 2:
                    from scipy.stats import linregress as _lr_sess

                    _cal_concs = _grp["Conc (ppm)"].values
                    _cal_shifts = _grp["Mean Δλ (nm)"].values
                    _cal_stds = _grp["Std Δλ (nm)"].fillna(0).values
                    _s_slope, _s_intercept, _s_r, *_ = _lr_sess(_cal_concs, _cal_shifts)
                    _x_fit = np.linspace(_cal_concs.min(), _cal_concs.max(), 200)

                    _fig_sess_cal = go.Figure()
                    _fig_sess_cal.add_trace(
                        go.Scatter(
                            x=_cal_concs,
                            y=_cal_shifts,
                            mode="markers",
                            name="Mean ± std",
                            marker=dict(size=12, color="crimson"),
                            error_y=dict(
                                type="data",
                                array=_cal_stds.tolist(),
                                visible=any(v > 0 for v in _cal_stds),
                                color="rgba(180,30,30,0.6)",
                                thickness=2,
                                width=6,
                            ),
                        )
                    )
                    _fig_sess_cal.add_trace(
                        go.Scatter(
                            x=_x_fit,
                            y=_s_slope * _x_fit + _s_intercept,
                            mode="lines",
                            name=f"Linear fit (R²={_s_r**2:.4f})",
                            line=dict(color="royalblue", dash="dash", width=2),
                        )
                    )
                    _fig_sess_cal.update_layout(
                        title=f"Session Calibration Curve — {cfg['gas']}",
                        xaxis_title="Known Concentration (ppm)",
                        yaxis_title="Δλ (nm)",
                        height=380,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(_fig_sess_cal, use_container_width=True)

                    # Nonlinear isotherm fitting (requires ≥ 3 distinct concentrations)
                    if len(_unique_concs) >= 3:
                        with st.expander("🔁 Nonlinear Isotherm Fitting (AIC selection)", expanded=False):
                            st.caption(
                                "Fits Langmuir, Freundlich, Hill, and linear models to the "
                                "session calibration data and selects the best by AICc. "
                                "Relevant when the sensor operates near binding-site saturation."
                            )
                            if st.button("Fit Isotherms", key="exp_btn_isotherms"):
                                try:
                                    from src.calibration.isotherms import select_isotherm as _sel_iso

                                    _iso_sel = _sel_iso(_cal_concs, _cal_shifts)
                                    _iso_best = _iso_sel["best_result"]

                                    _iso_c1, _iso_c2 = st.columns([2, 1])
                                    with _iso_c1:
                                        _fig_iso = go.Figure()
                                        _fig_iso.add_trace(
                                            go.Scatter(
                                                x=_cal_concs,
                                                y=_cal_shifts,
                                                mode="markers",
                                                name="Mean Δλ",
                                                marker=dict(size=12, color="crimson"),
                                                error_y=dict(
                                                    type="data",
                                                    array=_cal_stds.tolist(),
                                                    visible=any(v > 0 for v in _cal_stds),
                                                ),
                                            )
                                        )
                                        _fig_iso.add_trace(
                                            go.Scatter(
                                                x=_iso_best.concentrations_fit,
                                                y=_iso_best.responses_fit,
                                                mode="lines",
                                                name=f"{_iso_best.model.capitalize()} (AIC winner)",
                                                line=dict(color="royalblue", dash="dash", width=2),
                                            )
                                        )
                                        _fig_iso.update_layout(
                                            title=(
                                                f"Isotherm Fit — {_iso_best.model.capitalize()} "
                                                f"(R²={_iso_best.r_squared:.4f}, "
                                                f"AIC={_iso_best.aic:.2f})"
                                            ),
                                            xaxis_title="Concentration (ppm)",
                                            yaxis_title="Δλ (nm)",
                                            height=340,
                                            margin=dict(t=40, b=30),
                                        )
                                        st.plotly_chart(_fig_iso, use_container_width=True)

                                    with _iso_c2:
                                        st.markdown(f"**Winner: {_iso_best.model.capitalize()}**")
                                        st.caption(f"AIC = {_iso_best.aic:.2f}")
                                        st.caption(f"R² = {_iso_best.r_squared:.4f}")
                                        st.caption(f"RMSE = {_iso_best.rmse:.4g}")
                                        st.markdown("**Parameters:**")
                                        for _pn, _pv in _iso_best.params.items():
                                            if _pn == "sign":
                                                continue
                                            _pe = _iso_best.param_stderrs.get(_pn, float("nan"))
                                            st.caption(f"`{_pn}` = {_pv:.4g} ± {_pe:.2g}")
                                        st.markdown("**AIC table:**")
                                        for _mn, _ma, _mr2, _mrmse in _iso_sel["aic_table"]:
                                            st.caption(
                                                f"{_mn}: AIC={_ma:.2f}, R²={_mr2:.4f}"
                                            )
                                except Exception as _iso_exc:
                                    st.error(f"Isotherm fit failed: {_iso_exc}")

                    # Session LOD / LOQ from calibration sensitivity + mean noise
                    try:
                        from src.scientific.lod import (
                            calculate_lod_3sigma,
                            calculate_loq_10sigma,
                        )

                        _avg_noise = float(_gas_runs["noise_std"].mean())
                        _sess_lod = calculate_lod_3sigma(_avg_noise, _s_slope)
                        _sess_loq = calculate_loq_10sigma(_avg_noise, _s_slope)
                        _lc1, _lc2, _lc3 = st.columns(3)
                        _lc1.metric("Session Sensitivity", f"{abs(_s_slope):.4g} nm/ppm")
                        _lc2.metric(
                            "Session LOD (3σ/S)",
                            f"{_sess_lod:.3f} ppm" if _sess_lod < 1e6 else "N/A",
                            help="From session calibration slope + mean noise",
                        )
                        _lc3.metric(
                            "Session LOQ (10σ/S)",
                            f"{_sess_loq:.3f} ppm" if _sess_loq < 1e6 else "N/A",
                        )
                    except Exception:
                        pass
            else:
                st.info(
                    f"No runs logged for **{cfg['gas']}** yet. "
                    "Click **+ Log This Run** above to start."
                )
        else:
            st.info("Session is empty. Click **+ Log This Run** to start building a session log.")

        # ── new experiment ────────────────────────────────────────────────
        st.markdown("---")
        col_back, col_new = st.columns(2)
        with col_back:
            if st.button("← Re-capture Sample (same reference)"):
                ss["exp_step"] = 3
                ss["exp_sample"] = None
                ss["exp_result"] = None
                st.rerun()
        with col_new:
            if st.button("Start New Experiment", type="primary"):
                ss["exp_step"] = 1
                ss["exp_reference"] = None
                ss["exp_sample"] = None
                ss["exp_result"] = None
                ss["exp_config"] = {}
                st.rerun()
