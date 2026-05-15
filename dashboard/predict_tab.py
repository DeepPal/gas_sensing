"""
dashboard.predict_tab
======================
First-class "Predict Unknown Sample" tab for the SpectraAgent dashboard.

**Purpose**: A researcher who has already built a calibration (today or a
previous day) can come here directly to get a concentration estimate for an
unknown spectrum — without re-running the full 4-step pipeline.

**Workflow**:
1. Load calibration — from current ProjectStore, or from a saved session,
   or by uploading calibration CSV files manually.
2. Upload unknown spectrum CSV (or acquire from live hardware).
3. Preprocess using the same config as the original calibration.
4. Predict → display concentration ± CI with quality flag.
5. Accumulate predictions → exportable results table.

The prediction engine uses whatever model was trained in the current or loaded
session (GPR by default, which gives analytic uncertainty intervals).

"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import streamlit as st

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    from dashboard.project_store import ProjectStore, get_project, list_saved_sessions, set_project
    STORE_OK = True
except Exception as _e:
    STORE_OK = False
    log.warning("project_store unavailable: %s", _e)

try:
    from dashboard.calibration_library import render_load_from_library
    CAL_LIB_OK = True
except Exception as _cle:
    CAL_LIB_OK = False
    log.warning("calibration_library unavailable: %s", _cle)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_spectrum_csv(uploaded) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse an uploaded CSV into (wavelengths, intensities).  Header-aware."""
    if uploaded is None:
        return None
    try:
        import pandas as pd
        df = pd.read_csv(uploaded)
        uploaded.seek(0)
        # Try to detect numeric first row — if not, assume header already read
        try:
            float(df.iloc[0, 0])
        except (ValueError, TypeError):
            pass  # has header — df is fine

        wl_col = next(
            (c for c in df.columns if str(c).lower().startswith(("wavel", "nm", "wl"))),
            None,
        )
        int_col = next(
            (c for c in df.columns if str(c).lower().startswith(("intens", "signal", "counts", "i"))),
            None,
        )
        if wl_col and int_col:
            return df[wl_col].astype(float).values, df[int_col].astype(float).values
        if df.shape[1] >= 2:
            return df.iloc[:, 0].astype(float).values, df.iloc[:, 1].astype(float).values
    except Exception as exc:
        st.error(f"Could not parse spectrum CSV: {exc}")
    return None


def _compute_response(
    wavelengths: np.ndarray,
    intensities: np.ndarray,
    reference: np.ndarray | None,
    ref_peak_nm: float | None,
    preprocessing_config: dict,
) -> tuple[float, dict]:
    """Apply the same preprocessing as calibration and return (response, debug_dict).

    Returns
    -------
    response : float
        The single scalar feature sent to the model (Δλ or peak intensity).
    debug : dict
        Intermediate values for display (peak_nm, shift_nm, snr, etc.).
    """
    from src.signal.peak import find_peak_wavelength
    from src.preprocessing.baseline import correct_baseline

    method = preprocessing_config.get("baseline_method", "als")
    try:
        corrected = correct_baseline(intensities, method=method)
    except Exception:
        corrected = intensities

    peak_nm = float(find_peak_wavelength(wavelengths, corrected))

    if reference is not None and ref_peak_nm is not None:
        shift_nm = peak_nm - ref_peak_nm
        response = shift_nm
    else:
        response = peak_nm

    snr = float(np.max(corrected) / (np.std(corrected[:50]) + 1e-10))

    return response, {
        "peak_nm": round(peak_nm, 4),
        "shift_nm": round(response, 4) if reference is not None else None,
        "snr": round(snr, 1),
    }


def _predict_with_gpr(model_path: str, feature: float) -> tuple[float, float, float]:
    """Run GPR prediction.  Returns (mean, lower_95ci, upper_95ci)."""
    import joblib
    model = joblib.load(model_path)
    X = np.array([[feature]])
    try:
        mean, std = model.predict(X, return_std=True)
        mean, std = float(mean[0]), float(std[0])
    except TypeError:
        mean = float(model.predict(X)[0])
        std = 0.0
    ci95 = 1.96 * std
    return mean, mean - ci95, mean + ci95


def _quality_flag(
    concentration: float,
    shift: float | None,
    performance: dict,
    snr: float,
) -> tuple[str, str]:
    """Return (flag_label, explanation) based on quality thresholds."""
    lod = performance.get("lod_ppm") or 0.0
    loq = performance.get("loq_ppm") or 0.0
    lol = performance.get("lol_ppm") or float("inf")

    if snr < 3:
        return "LOW SNR", f"SNR={snr:.1f} < 3 — measurement may be unreliable"
    if concentration < lod:
        return "BELOW LOD", f"{concentration:.3g} ppm < LOD ({lod:.3g} ppm) — not detected"
    if concentration < loq:
        return "BELOW LOQ", f"{concentration:.3g} ppm < LOQ ({loq:.3g} ppm) — detectable but not quantifiable"
    if concentration > lol:
        return "ABOVE LOL", f"{concentration:.3g} ppm > linear range limit ({lol:.3g} ppm) — extrapolation"
    return "OK", f"Within calibrated range — quantifiable"


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Predict Unknown tab."""
    st.title("🎯 Predict Unknown Sample")
    st.markdown(
        "Load a validated calibration and predict the concentration of an unknown spectrum. "
        "Uses the model trained in the current or any saved session."
    )

    if not STORE_OK:
        st.error("ProjectStore unavailable — restart the dashboard.")
        return

    proj = get_project()

    # ── Step 1: Load Calibration ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 1 — Load Calibration")

    _source_opts = ["Current session", "Load saved session", "Calibration Library", "Upload calibration CSV"]
    source = st.radio(
        "Calibration source",
        _source_opts,
        horizontal=True,
        key="pred_cal_source",
    )

    if source == "Calibration Library":
        if not CAL_LIB_OK:
            st.error("Calibration library module unavailable.")
        else:
            result = render_load_from_library()
            if result is not None:
                _lib_model, _lib_concs, _lib_resps, _lib_perf = result
                import pickle, tempfile, os
                _tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
                pickle.dump(_lib_model, _tmp)
                _tmp.close()
                proj.set_calibration(_lib_concs, _lib_resps)
                proj.set_model(
                    model_path=_tmp.name,
                    model_type=_lib_perf.get("model_type", "GPR"),
                    performance=_lib_perf,
                )
                proj.gas_name = _lib_perf.get("gas_name", proj.gas_name)
                st.success("Calibration library entry loaded into the current session.")
                st.rerun()

    elif source == "Load saved session":
        sessions = list_saved_sessions()
        trained = [s for s in sessions if s.get("r_squared") is not None]
        if not trained:
            st.warning("No trained sessions found in `output/sessions/`. Run the Guided Calibration tab first.")
        else:
            opts = [
                f"{s['created_at']}  {s['gas_name']}  R²={s['r_squared']:.4f}  LOD={s['lod_ppm']:.3g} ppm"
                if s.get("r_squared") and s.get("lod_ppm")
                else f"{s['created_at']}  {s['gas_name']}"
                for s in trained
            ]
            sel = st.selectbox("Select session", range(len(opts)), format_func=lambda i: opts[i], key="pred_sess_sel")
            if st.button("Load this session", key="pred_load_sess"):
                try:
                    loaded = ProjectStore.load(trained[sel]["session_id"])
                    set_project(loaded)
                    proj = loaded
                    st.success(f"Loaded: {trained[sel]['session_id']}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load session: {exc}")

    elif source == "Upload calibration CSV":
        st.info(
            "Upload a CSV with columns `concentration_ppm` and `signal` (or `wavelength_shift_nm`). "
            "This will fit a quick GPR and store it in the current session."
        )
        cal_file = st.file_uploader("Calibration CSV", type=["csv"], key="pred_cal_upload")
        gas_input = st.text_input("Analyte name", value=proj.gas_name, key="pred_gas_name")
        if cal_file and st.button("Fit calibration", key="pred_fit_btn"):
            _fit_quick_calibration(cal_file, gas_input, proj)

    # Show calibration status
    _render_calibration_status(proj)

    if not proj.has_calibration or not proj.has_model:
        if source == "Current session" and not proj.has_calibration:
            st.info(
                "No calibration in the current session. "
                "Run the **Guided Calibration** tab first, or load a saved session above."
            )
        return

    # ── Step 2: Upload Unknown Spectrum ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 2 — Upload Unknown Spectrum")

    uploaded = st.file_uploader(
        "Unknown spectrum CSV (wavelength, intensity)",
        type=["csv"],
        key="pred_unknown_csv",
        help="Two-column CSV: wavelength (nm), intensity (counts or normalised).",
    )

    if uploaded is None:
        st.info("Upload a spectrum CSV to continue.")
        return

    parsed = _load_spectrum_csv(uploaded)
    if parsed is None:
        return
    wl_unk, inten_unk = parsed

    # Show the uploaded spectrum
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=wl_unk, y=inten_unk, name="Unknown", line=dict(color="#2196F3")))
        if proj.has_reference and proj.wavelengths is not None:
            # Interpolate reference onto unknown wavelengths
            ref_interp = np.interp(wl_unk, proj.wavelengths, proj.reference_spectrum)
            fig.add_trace(go.Scatter(
                x=wl_unk, y=ref_interp, name="Reference",
                line=dict(color="#9E9E9E", dash="dash"),
            ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)", yaxis_title="Intensity",
            height=300, template="plotly_white", margin=dict(t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Step 3: Predict ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 3 — Predict Concentration")

    response, debug = _compute_response(
        wl_unk, inten_unk,
        proj.reference_spectrum,
        proj.reference_peak_nm,
        proj.preprocessing_config,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Peak wavelength", f"{debug['peak_nm']:.3f} nm")
    with col2:
        if debug["shift_nm"] is not None:
            st.metric("Δλ (shift)", f"{debug['shift_nm']:+.4f} nm")
        else:
            st.metric("Feature (no ref)", f"{response:.4f}")
    with col3:
        st.metric("SNR", f"{debug['snr']:.1f}")

    if st.button("▶ Predict", type="primary", key="pred_run_btn"):
        try:
            conc, ci_lo, ci_hi = _predict_with_gpr(proj.model_path, response)
            flag, explanation = _quality_flag(conc, debug["shift_nm"], proj.performance, debug["snr"])

            # Display result prominently
            flag_color = {"OK": "green", "BELOW LOQ": "orange",
                          "BELOW LOD": "red", "ABOVE LOL": "orange", "LOW SNR": "red"}.get(flag, "grey")
            st.markdown(f"""
<div style="background:#f0f7ff;border-left:5px solid #1565C0;padding:16px 20px;border-radius:4px;margin:8px 0">
<span style="font-size:2em;font-weight:bold;color:#1565C0">{conc:.3f} ppm</span>
<span style="margin-left:16px;color:#555">95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] ppm</span><br>
<span style="display:inline-block;margin-top:6px;padding:2px 10px;background:{flag_color};color:white;border-radius:3px;font-size:0.85em">{flag}</span>
<span style="margin-left:10px;color:#666;font-size:0.9em">{explanation}</span>
</div>
""", unsafe_allow_html=True)

            # Store in session
            proj.add_prediction(
                concentration=conc,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                quality=flag,
                spectrum_label=uploaded.name,
            )
            proj.save()
            st.session_state["pred_last_result"] = {
                "concentration": conc, "ci_lo": ci_lo, "ci_hi": ci_hi,
                "flag": flag, "explanation": explanation,
                "feature": response, "debug": debug,
            }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            log.exception("Prediction error")
            return

    # ── Step 4: Prediction History ────────────────────────────────────────────
    if proj.predictions:
        st.markdown("---")
        st.subheader("Prediction History")
        st.caption(f"{len(proj.predictions)} predictions in this session")

        if PANDAS_OK:
            import pandas as pd
            df = pd.DataFrame(proj.predictions)
            rename = {
                "timestamp": "Time", "spectrum_label": "File",
                "concentration_ppm": "Conc. (ppm)",
                "ci_lower_ppm": "CI lower", "ci_upper_ppm": "CI upper",
                "quality": "Quality",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV",
                data=csv_bytes,
                file_name=f"predictions_{proj.session_id}.csv",
                mime="text/csv",
                key="pred_dl_csv",
            )

        if PLOTLY_OK and len(proj.predictions) >= 2:
            import plotly.express as px
            df_plot = pd.DataFrame(proj.predictions) if PANDAS_OK else None
            if df_plot is not None and "concentration_ppm" in df_plot.columns:
                fig2 = px.scatter(
                    df_plot,
                    x=df_plot.index,
                    y="concentration_ppm",
                    error_y=df_plot["ci_upper_ppm"] - df_plot["concentration_ppm"]
                    if "ci_upper_ppm" in df_plot.columns else None,
                    color="quality" if "quality" in df_plot.columns else None,
                    labels={"x": "Measurement #", "concentration_ppm": "Concentration (ppm)"},
                    title="Prediction History",
                    template="plotly_white",
                )
                st.plotly_chart(fig2, use_container_width=True)

    # ── Analysis Report ──────────────────────────────────────────────────
    try:
        from dashboard.report_generator import render_report_download_button
        render_report_download_button(proj)
    except Exception as _rpt_e:
        log.warning("report_generator unavailable: %s", _rpt_e)

    # ── Mixture Deconvolution ────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🧪 Mixture Deconvolution (multi-analyte)", expanded=False):
        st.markdown(
            "Estimate concentrations of **multiple analytes simultaneously** from a single sensor response. "
            "Requires one sensitivity value per analyte (from individual calibrations). "
            "Uses linear pseudo-inverse for the linear regime; switches to Langmuir solver when saturation is expected."
        )

        # Analyte table input
        st.markdown("##### Define analyte sensitivities")
        st.caption(
            "Enter one row per analyte. Sensitivity = slope of Δλ vs concentration (nm/ppm) "
            "from individual calibration curves."
        )

        _md_n = st.number_input(
            "Number of analytes", min_value=2, max_value=8, value=int(st.session_state.get("md_n_analytes", 2)),
            step=1, key="md_n_analytes",
        )
        _md_analytes, _md_sens, _md_kds = [], [], []
        _md_use_langmuir = st.checkbox(
            "Use Langmuir solver (non-linear / saturation regime)",
            value=False, key="md_use_langmuir",
            help="Enable if any analyte concentration approaches sensor saturation (C ~ Kd).",
        )

        for _i in range(int(_md_n)):
            _mc1, _mc2, _mc3 = st.columns([2, 1, 1])
            _name = _mc1.text_input(
                f"Analyte {_i+1}", value=f"Analyte_{_i+1}", key=f"md_name_{_i}"
            )
            _sens = _mc2.number_input(
                "Sensitivity (nm/ppm)", min_value=0.0, value=0.001, format="%.5f",
                step=0.0001, key=f"md_sens_{_i}",
            )
            _kd = _mc3.number_input(
                "Kd (ppm)", value=1e6, format="%.1f", key=f"md_kd_{_i}",
                help="Langmuir dissociation constant. Set very high (1e6) for linear model.",
                disabled=not _md_use_langmuir,
            )
            _md_analytes.append(_name)
            _md_sens.append(_sens)
            _md_kds.append(_kd)

        st.markdown("##### Observed peak shifts (nm)")
        st.caption("Enter the peak shift observed from each sensor channel (1 value if single-channel sensor).")
        _md_shift_str = st.text_input(
            "Observed Δλ values (nm, comma-separated)",
            value="", key="md_obs_shifts",
            placeholder="e.g. 0.045  (single channel) or  0.045, 0.032  (two channels)",
        )

        if st.button("Deconvolve mixture", key="md_deconvolve_btn", type="primary"):
            try:
                from src.calibration.mixture_deconvolution import deconvolve_mixture
                _obs = np.array([float(x.strip()) for x in _md_shift_str.split(",") if x.strip()])
                if len(_obs) == 0:
                    st.error("Enter at least one observed peak shift.")
                else:
                    # Build S matrix: shape (N_analytes, M_channels)
                    # For single-channel: each analyte contributes its sensitivity to channel 0
                    _M = len(_obs)
                    _N = len(_md_analytes)
                    _S = np.zeros((_N, _M))
                    for _ai in range(_N):
                        _S[_ai, 0] = _md_sens[_ai]  # all analytes share channel 0 in single-channel case
                    _Kd = np.array(_md_kds)[:, np.newaxis] * np.ones((_N, _M)) if _md_use_langmuir else None

                    _result = deconvolve_mixture(
                        delta_lambda=_obs,
                        analytes=_md_analytes,
                        S=_S,
                        Kd=_Kd,
                        use_nonlinear=_md_use_langmuir,
                    )

                    st.success(f"Deconvolution complete (solver: {_result.solver})")
                    _cols_res = st.columns(len(_md_analytes))
                    for _ci, (_an, _conc) in enumerate(_result.concentrations.items()):
                        _cols_res[_ci].metric(_an, f"{_conc:.4g} ppm")

                    _rsd_c1, _rsd_c2 = st.columns(2)
                    _rsd_c1.metric("RMS residual", f"{_result.residual_nm:.4g} nm")
                    _rsd_c2.metric("Solver converged", "Yes" if _result.success else "No")

                    if _result.residual_nm > 0.05:
                        st.warning(
                            f"RMS residual = {_result.residual_nm:.4g} nm is relatively large. "
                            "Consider: wrong analytes, missing components, or out-of-range concentrations."
                        )
            except Exception as _md_e:
                st.error(f"Deconvolution failed: {_md_e}")


def _render_calibration_status(proj: "ProjectStore") -> None:
    """Show a compact calibration status card."""
    perf = proj.performance or {}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Analyte", proj.gas_name)
    with col2:
        r2 = perf.get("r_squared")
        st.metric("Calibration R²", f"{r2:.4f}" if r2 is not None else "—",
                  delta="good" if (r2 or 0) >= 0.99 else "check",
                  delta_color="normal" if (r2 or 0) >= 0.99 else "inverse")
    with col3:
        lod = perf.get("lod_ppm")
        st.metric("LOD", f"{lod:.3g} ppm" if lod is not None else "—")
    with col4:
        st.metric("Model", proj.model_type)

    if proj.has_model:
        st.success(f"Calibration ready — model: `{proj.model_path}`")
    elif proj.has_calibration:
        st.warning("Calibration data loaded but no trained model. Fit a model first.")
    else:
        st.info("No calibration loaded.")


def _fit_quick_calibration(uploaded_file, gas_name: str, proj: "ProjectStore") -> None:
    """Fit a quick GPR from an uploaded calibration CSV and update the project store."""
    try:
        import pandas as pd
        import joblib
        from src.scientific.lod import sensor_performance_summary

        df = pd.read_csv(uploaded_file)
        conc_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["conc", "ppm"])), None
        )
        sig_col = next(
            (c for c in df.columns if any(k in c.lower() for k in
             ["signal", "shift", "response", "intens", "delta"])), None
        )
        if conc_col is None or sig_col is None and df.shape[1] >= 2:
            conc_col, sig_col = df.columns[0], df.columns[1]

        concs = df[conc_col].dropna().astype(float).values
        responses = df[sig_col].dropna().astype(float).values
        n = min(len(concs), len(responses))
        concs, responses = concs[:n], responses[:n]

        if n < 3:
            st.error("Need at least 3 calibration points.")
            return

        proj.gas_name = gas_name
        proj.set_calibration(concs, responses)

        # Fit GPR
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            gpr.fit(concs.reshape(-1, 1), responses)

            model_path = _REPO_ROOT / "output" / "sessions" / proj.session_id / "quick_gpr.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(gpr, model_path)
        except ImportError:
            st.error("scikit-learn required for GPR fitting.")
            return

        perf = sensor_performance_summary(concs, responses, gas_name=gas_name)
        proj.set_model(str(model_path), "GPR", perf)
        proj.save()

        st.success(
            f"Calibration fitted: R²={perf.get('r_squared', 0):.4f}, "
            f"LOD={perf.get('lod_ppm', '?'):.3g} ppm, "
            f"n={n} points"
        )

    except Exception as exc:
        st.error(f"Calibration fitting failed: {exc}")
        log.exception("Quick calibration error")
