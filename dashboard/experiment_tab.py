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


def _analyse(cfg: dict, ref: dict, smp: dict) -> dict:
    """
    Core analysis:
      1. Baseline-correct both spectra (ALS)
      2. Compute absorbance = log10(ref / sample)  [Beer-Lambert]
      3. Find peak in absorbance spectrum
      4. Compute SNR
      5. Apply linear calibration → concentration ppm
    Returns a result dict.
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

    # ── absorbance ────────────────────────────────────────────────────────
    eps = 1e-9
    r_pos = np.clip(r_sm, eps, None)
    s_pos = np.clip(s_sm, eps, None)
    absorbance = np.log10(r_pos / s_pos)

    # ── peak detection ────────────────────────────────────────────────────
    peak_idx = int(np.argmax(np.abs(absorbance)))
    peak_wl = float(wl[peak_idx])
    peak_abs = float(absorbance[peak_idx])

    # ── SNR ───────────────────────────────────────────────────────────────
    # Use lowest 10% of wavelength range as noise reference (adapts to any sensor range)
    wl_range = float(wl[-1] - wl[0])
    noise_mask = wl <= (float(wl[0]) + 0.10 * wl_range)
    if noise_mask.sum() <= 5:
        noise_mask = np.ones(len(wl), dtype=bool)  # fallback: whole spectrum
    noise_std = float(np.std(absorbance[noise_mask])) if noise_mask.sum() > 0 else 1e-9
    noise_std = max(noise_std, 1e-9)
    snr = abs(peak_abs) / noise_std

    # ── wavelength shift (reference peak vs sample peak) ─────────────────
    ref_peak_idx = int(np.argmax(r_sm))
    smp_peak_idx = int(np.argmax(s_sm))
    wl_shift_nm = float(wl[smp_peak_idx] - wl[ref_peak_idx])

    # ── calibration (linear: shift_nm / slope = ppm) ──────────────────────
    slope = float(cfg.get("calibration_slope", 0.116))  # nm/ppm
    conc_ppm = wl_shift_nm / slope if slope != 0 else 0.0
    conc_ppm = float(np.clip(conc_ppm, 0, 10000))

    # ── replicate RSD (if multiple frames stored) ─────────────────────────
    rsd_ref = 0.0
    if "frames" in ref and len(ref["frames"]) > 1:
        peak_vals = [f[peak_idx] for f in ref["frames"]]
        m = np.mean(peak_vals)
        rsd_ref = float(np.std(peak_vals) / m * 100) if m != 0 else 0.0

    return dict(
        wavelengths=wl,
        ref_corrected=r_sm,
        smp_corrected=s_sm,
        absorbance=absorbance,
        peak_wavelength_nm=peak_wl,
        peak_absorbance=peak_abs,
        snr=snr,
        noise_std=noise_std,
        wavelength_shift_nm=wl_shift_nm,
        concentration_ppm=conc_ppm,
        calibration_slope=slope,
        rsd_ref_pct=rsd_ref,
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

        # Calibration override
        with st.expander("Calibration settings (optional override)"):
            slope = st.number_input(
                "Calibration slope (nm / ppm)",
                value=0.116,
                format="%.4f",
                help="From your calibration curve. Default 0.116 nm/ppm",
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

        # ── key metrics ───────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Estimated concentration",
            f"{res['concentration_ppm']:.2f} ppm",
            help="From calibration: Δλ / slope",
        )
        m2.metric(
            "Wavelength shift",
            f"{res['wavelength_shift_nm']:+.4f} nm",
            help="Sample peak − Reference peak",
        )
        m3.metric(
            "Peak absorbance",
            f"{res['peak_absorbance']:.4f}",
            help=f"At {res['peak_wavelength_nm']:.1f} nm",
        )
        m4.metric(
            "SNR",
            f"{res['snr']:.1f}",
            delta="Good" if res["snr"] > 10 else ("Marginal" if res["snr"] > 3 else "Low"),
            delta_color="normal" if res["snr"] > 10 else "inverse",
        )

        # ── absorbance spectrum ───────────────────────────────────────────
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
            annotation_text=f"Peak {res['peak_wavelength_nm']:.1f} nm",
            annotation_position="top right",
        )
        fig.update_layout(
            title="Absorbance Spectrum  [log₁₀(Reference / Sample)]",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance (a.u.)",
            height=320,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── corrected spectra overlay ─────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=res["wavelengths"],
                y=res["ref_corrected"],
                name="Reference (baseline-corrected)",
                line=dict(color="#2196F3", width=1.2, dash="dot"),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=res["wavelengths"],
                y=res["smp_corrected"],
                name=f"Sample — {cfg['gas']} (baseline-corrected)",
                line=dict(color="#FF5722", width=1.2),
            )
        )
        fig2.update_layout(
            title="Baseline-Corrected Spectra",
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity (a.u.)",
            height=300,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── calibration info ──────────────────────────────────────────────
        with st.expander("Calibration details"):
            st.markdown(f"""
| Parameter | Value |
|---|---|
| Calibration slope | {res["calibration_slope"]} nm/ppm |
| Wavelength shift | {res["wavelength_shift_nm"]:+.4f} nm |
| Concentration (linear) | {res["concentration_ppm"]:.3f} ppm |
| Peak wavelength | {res["peak_wavelength_nm"]:.2f} nm |
| Noise std (690–720 nm) | {res["noise_std"]:.6f} |
| SNR | {res["snr"]:.1f} |
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
                    "peak_wavelength_nm": res["peak_wavelength_nm"],
                    "peak_absorbance": res["peak_absorbance"],
                    "wavelength_shift_nm": res["wavelength_shift_nm"],
                    "concentration_ppm": res["concentration_ppm"],
                    "calibration_slope_nm_per_ppm": res["calibration_slope"],
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
