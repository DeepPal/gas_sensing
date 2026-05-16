"""Acquisition control, calibration, analyte, and simulation endpoints."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as _np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse

from spectraagent.webapp.routes._models import (
    AcquisitionConfig,
    CalibrationPoint,
    MixtureInferenceRequest,
    SensitivityFitRequest,
    SimGenerateRequest,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["acquisition"])


# ---------------------------------------------------------------------------
# Acquisition config / start / stop / reference
# ---------------------------------------------------------------------------


@router.post("/api/acquisition/config")
async def acq_config(cfg: AcquisitionConfig, request: Request) -> JSONResponse:
    app = request.app
    _acq_config: dict[str, Any] = app.state._acq_config
    _acq_config.update(cfg.model_dump())
    if app.state.driver is not None:
        app.state.driver.set_integration_time_ms(cfg.integration_time_ms)
    return JSONResponse(_acq_config)


@router.post("/api/acquisition/start")
async def acq_start(request: Request) -> JSONResponse:
    import time as _time

    app = request.app
    _acq_config: dict[str, Any] = app.state._acq_config
    _session_active: dict[str, Any] = app.state._session_active

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _session_active["running"] = True
    _session_active["session_id"] = session_id
    app.state.session_running = True
    app.state.last_session_id = session_id
    app.state.session_frame_count = 0
    # Reset per-session agent state so cross-session history doesn't bleed.
    drift_agent = getattr(app.state, "drift_agent", None)
    if drift_agent is not None:
        drift_agent.reset()
    planner_agent = getattr(app.state, "planner_agent", None)
    if planner_agent is not None:
        planner_agent.reset()
    app.state.session_events = []
    app.state.session_start_monotonic = _time.monotonic()
    sw = getattr(app.state, "session_writer", None)
    if sw is not None:
        meta = {
            "gas_label": _acq_config.get("gas_label", "unknown"),
            "target_concentration": _acq_config.get("target_concentration"),
            "hardware": getattr(getattr(app.state, "driver", None), "name", "unknown"),
            "temperature_c": _acq_config.get("temperature_c"),
            "humidity_pct": _acq_config.get("humidity_pct"),
        }
        sw.start_session(session_id, meta)
    return JSONResponse({"status": "started", "session_id": session_id})


@router.post("/api/acquisition/stop")
async def acq_stop(request: Request, background_tasks: BackgroundTasks) -> JSONResponse:
    from spectraagent.webapp.server import (  # type: ignore[attr-defined]
        _is_nan,
        _write_session_manifest,
        _write_session_scientific_summary,
    )

    app = request.app
    _acq_config: dict[str, Any] = app.state._acq_config
    _session_active: dict[str, Any] = app.state._session_active

    _session_active["running"] = False
    app.state.session_running = False
    frame_count = int(app.state.session_frame_count)
    sw = getattr(app.state, "session_writer", None)
    session_id = str(
        _session_active.get("session_id")
        or getattr(app.state, "last_session_id", None)
        or "unknown"
    )
    if sw is not None:
        sw.stop_session(frame_count=frame_count)

    # Snapshot mutable state before returning — background task runs after response
    session_events = list(getattr(app.state, "session_events", []))
    gas_label = _acq_config.get("gas_label", "unknown")
    acq_config_snapshot = dict(_acq_config)
    app.state.session_events = []  # clear immediately so next session starts fresh

    # Capture app reference for the background closure
    app_ref = app

    def _post_session_work() -> None:
        """Heavy post-processing runs after the HTTP response is sent.

        Includes: reproducibility manifest (git subprocess), SessionAnalyzer,
        SensorMemory — all deferred so /stop returns immediately.
        """
        _write_session_manifest(app_ref, session_id)
        analysis = None
        sw_inner = getattr(app_ref.state, "session_writer", None)
        try:
            from src.inference.session_analyzer import SessionAnalyzer
            from src.reporting.scientific_summary import session_analysis_to_dict

            analysis = SessionAnalyzer().analyze(session_events, frame_count)
            app_ref.state.last_session_analysis = analysis
            app_ref.state.last_session_analysis_session_id = session_id
            stored_session = (
                sw_inner.get_session(session_id) if sw_inner is not None else None
            )

            session_data: dict[str, Any] = {
                "session_id": session_id,
                "gas_label": gas_label,
                "hardware": getattr(
                    getattr(app_ref.state, "driver", None), "name", "unknown"
                ),
                "started_at": stored_session.get("started_at")
                if stored_session
                else None,
                "stopped_at": stored_session.get("stopped_at")
                if stored_session
                else None,
                "target_concentration": acq_config_snapshot.get("target_concentration"),
                "lod_ppm": analysis.lod_ppm,
                "lob_ppm": getattr(analysis, "lob_ppm", None),
                "lod_ci_lower": getattr(analysis, "lod_ci_lower", None),
                "lod_ci_upper": getattr(analysis, "lod_ci_upper", None),
                "loq_ppm": analysis.loq_ppm,
                "calibration_r2": analysis.calibration_r2,
                "calibration_rmse_ppm": getattr(analysis, "calibration_rmse_ppm", None),
                "calibration_n_points": getattr(analysis, "calibration_n_points", 0),
                "lol_ppm": getattr(analysis, "lol_ppm", None),
                "mean_snr": analysis.mean_snr,
                "drift_rate_nm_per_frame": analysis.drift_rate_nm_per_frame,
                "frame_count": analysis.frame_count,
                "summary": analysis.summary_text,
                "tau_63_s": getattr(analysis, "tau_63_s", None),
                "tau_95_s": getattr(analysis, "tau_95_s", None),
                "k_on_per_s": getattr(analysis, "k_on_per_s", None),
                "kinetics_fit_r2": getattr(analysis, "kinetics_fit_r2", None),
                "interval_coverage": getattr(analysis, "interval_coverage", None),
                "temperature_c": acq_config_snapshot.get("temperature_c"),
                "humidity_pct": acq_config_snapshot.get("humidity_pct"),
            }
            session_summary_context = {
                **session_data,
                "analysis": session_analysis_to_dict(analysis),
            }
            _write_session_scientific_summary(app_ref, session_id, session_summary_context)
            bus = getattr(app_ref.state, "agent_bus", None)
            if bus is not None:
                from spectraagent.webapp.agent_bus import AgentEvent

                bus.emit(
                    AgentEvent(
                        source="SessionAnalyzer",
                        level="info",
                        type="session_complete",
                        data=session_data,
                        text=analysis.summary_text,
                    )
                )
        except Exception as exc:
            log.warning("Post-session analysis failed: %s", exc)

        try:
            memory = getattr(app_ref.state, "sensor_memory", None)
            if memory is not None:
                import datetime as _dt

                now_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
                memory.record_session(
                    session_id=session_id,
                    analyte=gas_label,
                    frame_count=frame_count,
                    stopped_at=now_utc,
                )
                if analysis is not None and analysis.calibration_r2 is not None:
                    from spectraagent.knowledge.sensor_memory import CalibrationObservation

                    cal_agent = getattr(app_ref.state, "calibration_agent", None)
                    best_model = "unknown"
                    sensitivity: float | None = None
                    if cal_agent is not None:
                        best_model = (
                            getattr(cal_agent, "_last_best_model", "unknown") or "unknown"
                        )
                        sensitivity = getattr(
                            cal_agent, "_last_sensitivity_nm_per_ppm", None
                        )
                    memory.record_calibration(
                        CalibrationObservation(
                            session_id=session_id,
                            timestamp_utc=now_utc,
                            analyte=gas_label,
                            sensitivity_nm_per_ppm=sensitivity,
                            lod_ppm=analysis.lod_ppm
                            if not _is_nan(analysis.lod_ppm)
                            else None,
                            loq_ppm=analysis.loq_ppm
                            if not _is_nan(analysis.loq_ppm)
                            else None,
                            r_squared=analysis.calibration_r2,
                            rmse_ppm=getattr(analysis, "calibration_rmse_ppm", None),
                            calibration_model=best_model,
                            n_calibration_points=getattr(
                                analysis, "calibration_n_points", 0
                            ),
                            reference_peak_nm=getattr(app_ref.state, "ref_peak_nm", None),
                            conformal_coverage=None,
                            tau_63_s=getattr(analysis, "tau_63_s", None),
                            reference_fwhm_nm=getattr(app_ref.state, "ref_fwhm_nm", None),
                        )
                    )
        except Exception as exc:
            log.warning("SensorMemory record failed: %s", exc)

    background_tasks.add_task(_post_session_work)

    return JSONResponse(
        {"status": "stopped", "session_id": _session_active.get("session_id")}
    )


@router.post("/api/acquisition/reference")
async def acq_reference(request: Request) -> JSONResponse:
    from src.features.lspr_features import detect_all_peaks, fit_lorentzian_peak

    app = request.app
    latest_spectrum = getattr(app.state, "latest_spectrum", None)
    intensities = None if latest_spectrum is None else latest_spectrum.get("intensities")
    if intensities is None:
        return JSONResponse(
            {"error": "No spectrum available yet — wait for first frame"},
            status_code=400,
        )
    app.state.reference = intensities

    wl_np = _np.asarray(
        latest_spectrum.get("wl", []) if isinstance(latest_spectrum, dict) else []
    )
    int_np = _np.asarray(intensities)
    plugin = getattr(app.state, "plugin", None)

    # Detect ALL spectral peaks (multi-peak sensor support)
    ref_peak_wls: list[float] = []
    if len(wl_np) > 0:
        try:
            cfg_obj = getattr(app.state, "cfg", None)
            smin = cfg_obj.physics.search_min_nm if cfg_obj else float(wl_np[0])
            smax = cfg_obj.physics.search_max_nm if cfg_obj else float(wl_np[-1])
            ref_peak_wls = detect_all_peaks(wl_np, int_np, search_min=smin, search_max=smax)
        except Exception as exc:
            log.warning("Multi-peak detection failed: %s", exc)

    ref_peak: float | None = ref_peak_wls[0] if ref_peak_wls else None
    app.state.ref_peak_nm = ref_peak
    app.state.ref_peak_wls = ref_peak_wls

    # Update plugin so subsequent extract_features calls use discovered peaks
    if plugin is not None and ref_peak_wls and hasattr(plugin, "update_from_reference"):
        plugin.update_from_reference(ref_peak_wls)

    # Pre-compute cached Lorentzian fit for the primary peak (saves ~5 ms/frame)
    if plugin is not None and ref_peak is not None:
        try:
            app.state.cached_ref = plugin.compute_reference_cache(wl_np, int_np)
        except Exception as exc:
            log.warning("Reference cache build failed: %s", exc)
            app.state.cached_ref = None
    else:
        app.state.cached_ref = None

    # Extract FWHM of primary peak for sensor health tracking (B4)
    ref_fwhm: float | None = None
    if ref_peak is not None:
        try:
            lfit = fit_lorentzian_peak(wl_np, int_np, peak_wl_init=ref_peak)
            if lfit is not None:
                ref_fwhm = lfit[1]
                app.state.ref_fwhm_nm = ref_fwhm
        except Exception as exc:
            log.debug("Reference FWHM extraction failed: %s", exc)

    # Propagate to RealTimePipeline
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None and hasattr(pipeline, "set_reference"):
        try:
            pipeline.set_reference(int_np)
        except Exception as exc:
            log.warning("Pipeline reference set failed: %s", exc)

    return JSONResponse(
        {
            "status": "reference_captured",
            "peak_wavelength": ref_peak,           # primary peak (backward compat)
            "peak_wavelengths": ref_peak_wls,      # all detected peaks
            "n_peaks": len(ref_peak_wls),
            "fwhm_nm": ref_fwhm,
        }
    )


# ---------------------------------------------------------------------------
# Calibration endpoints
# ---------------------------------------------------------------------------


@router.post("/api/calibration/add-point")
async def calibration_add_point(
    point: CalibrationPoint, request: Request
) -> JSONResponse:
    app = request.app
    calib_agent = getattr(app.state, "calibration_agent", None)
    if calib_agent is not None:
        calib_agent.add_point(point.concentration, point.delta_lambda)
    return JSONResponse(
        {
            "status": "added",
            "concentration": point.concentration,
            "delta_lambda": point.delta_lambda,
        }
    )


@router.post("/api/calibration/suggest")
async def calibration_suggest(request: Request) -> JSONResponse:
    app = request.app
    planner = getattr(app.state, "planner_agent", None)
    if planner is None:
        return JSONResponse({"suggestion": None, "reason": "planner_not_initialized"})
    suggested = planner.suggest()
    if suggested is None:
        return JSONResponse({"suggestion": None, "reason": "no_gpr_fitted"})
    return JSONResponse({"suggestion": suggested})


@router.post("/api/calibration/sensitivity-matrix/fit")
async def calibration_sensitivity_fit(req: SensitivityFitRequest) -> JSONResponse:
    """Fit a sensitivity matrix from single-analyte calibration runs."""
    try:
        from src.calibration.sensitivity_matrix import SensitivityMatrix

        sm = SensitivityMatrix(req.analytes, req.n_peaks)
        for entry in req.calibration_data:
            sm.fit_analyte(
                analyte=entry["analyte"],
                peak_idx=int(entry["peak_idx"]),
                conc_ppm=entry["conc_ppm"],
                shifts_nm=entry["shifts_nm"],
            )
        summary = sm.summary()
        lod = sm.compute_lod_mixture()
        return JSONResponse(
            {
                "status": "fitted",
                "S_matrix": summary["S_matrix"],
                "condition_number": summary["condition_number"],
                "rank": summary["rank"],
                "r2_per_entry": [
                    {"analyte": e["analyte"], "peak": e["peak"], "r2": e["r_squared"]}
                    for e in summary["entries"]
                ],
                "lod_mixture_ppm": lod,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/inference/mixture")
async def inference_mixture(req: MixtureInferenceRequest) -> JSONResponse:
    """Estimate all analyte concentrations from a peak-shift observation."""
    try:
        import numpy as np

        from src.calibration.mixture_deconvolution import deconvolve_mixture

        S = np.array(req.S_matrix, dtype=float)
        Kd = np.array(req.Kd_matrix, dtype=float) if req.Kd_matrix else None
        dl = np.array(req.delta_lambda, dtype=float)
        result = deconvolve_mixture(
            dl, req.analytes, S, Kd=Kd, use_nonlinear=req.use_nonlinear
        )
        return JSONResponse(
            {
                "concentrations_ppm": result.concentrations,
                "residual_nm": result.residual_nm,
                "solver": result.solver,
                "success": result.success,
                "predicted_shifts_nm": result.predicted_shifts.tolist(),
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/analytes")
async def list_analytes(request: Request) -> JSONResponse:
    """List analytes registered in the current sensor configuration."""
    app = request.app
    plugin = app.state.plugin
    if plugin is None:
        return JSONResponse({"analytes": [], "n_peaks": 0, "S_matrix": None})
    sensor_cfg = getattr(plugin, "_cfg", None)
    if sensor_cfg is None:
        return JSONResponse({"analytes": [], "n_peaks": 0, "S_matrix": None})
    return JSONResponse(
        {
            "analytes": [a.name for a in sensor_cfg.analytes],
            "n_peaks": len(sensor_cfg.peaks),
            "peak_wavelengths_nm": [p.center_nm for p in sensor_cfg.peaks],
            "S_matrix": sensor_cfg.sensitivity_matrix.tolist()
            if sensor_cfg.analytes
            else None,
        }
    )


@router.post("/api/simulation/generate")
async def simulation_generate(req: SimGenerateRequest) -> JSONResponse:
    """Generate a synthetic calibration dataset from the physics simulation."""
    try:
        from src.simulation.dataset_generator import DatasetConfig, DatasetGenerator
        from src.simulation.gas_response import make_analyte, make_single_peak_sensor

        sensor = make_single_peak_sensor(req.peak_nm, req.fwhm_nm, req.wl_start, req.wl_end)
        sensor.analytes = [
            make_analyte(
                req.analyte_name,
                1,
                req.sensitivity_nm_per_ppm,
                tau_s=req.tau_s,
                kd_ppm=req.kd_ppm,
            )
        ]
        cfg = DatasetConfig(
            sensor_config=sensor,
            analyte_names=[req.analyte_name],
            concentration_levels=req.concentrations,
            n_sessions=req.n_sessions,
            random_seed=req.random_seed,
            domain_randomize=True,
        )
        df = DatasetGenerator(cfg).generate_calibration_dataset()
        summary = (
            df.groupby("concentration_ppm")["peak_shift_0"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "mean_shift_nm",
                    "std": "std_shift_nm",
                    "count": "n",
                }
            )
            .to_dict(orient="records")
        )
        return JSONResponse(
            {
                "status": "ok",
                "analyte": req.analyte_name,
                "n_sessions": req.n_sessions,
                "n_rows": len(df),
                "calibration_summary": summary,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
