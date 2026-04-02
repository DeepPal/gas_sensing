"""spectraagent.__main__ — CLI entry point (Typer)."""
from __future__ import annotations

from importlib.metadata import entry_points
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

import typer

from spectraagent import __version__
from spectraagent.commands.robustness import RobustnessRunner
from spectraagent.drivers.base import AbstractHardwareDriver
from spectraagent.drivers.validation import (
    validate_driver_class,
    validate_driver_instance,
)
from spectraagent.physics.base import AbstractSensorPhysicsPlugin

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"spectraagent {__version__}")
        raise typer.Exit()


cli = typer.Typer(
    name="spectraagent",
    help="Universal Agentic Spectroscopy Platform",
    no_args_is_help=True,
)


@cli.callback()
def _main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


# ---------------------------------------------------------------------------
# Helper: load hardware driver
# ---------------------------------------------------------------------------


def _load_driver(simulate: bool, cfg) -> AbstractHardwareDriver:
    """Load hardware driver: simulation if forced, else try config default, fallback sim."""
    from spectraagent.drivers.simulation import SimulationDriver

    if simulate:
        drv: AbstractHardwareDriver = SimulationDriver(
            integration_time_ms=cfg.hardware.integration_time_ms
        )
        drv.connect()
        issues = validate_driver_instance(drv, require_live_sample=True)
        if issues:
            raise RuntimeError(
                "Simulation driver failed contract validation: " + "; ".join(issues)
            )
        return drv

    hw_eps = {ep.name: ep for ep in entry_points(group="spectraagent.hardware")}
    driver_name = cfg.hardware.default_driver
    if driver_name not in hw_eps:
        typer.echo(
            f"  Driver '{driver_name}' not found -- falling back to simulation",
            err=True,
        )
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv

    try:
        cls = hw_eps[driver_name].load()
        class_issues = validate_driver_class(cls)
        if class_issues:
            raise RuntimeError("; ".join(class_issues))

        # Extension compatibility: only pass kwargs accepted by plugin constructor.
        sig = inspect.signature(cls)
        kwargs = {}
        if "integration_time_ms" in sig.parameters:
            kwargs["integration_time_ms"] = cfg.hardware.integration_time_ms
        drv = cls(**kwargs)  # type: ignore[assignment]
        drv.connect()
        issues = validate_driver_instance(drv, require_live_sample=True)
        if issues:
            raise RuntimeError("; ".join(issues))
        return drv
    except Exception as exc:
        typer.echo(
            f"  Failed to connect to {driver_name}: {exc} -- falling back to simulation",
            err=True,
        )
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
        return drv


# ---------------------------------------------------------------------------
# Helper: load physics plugin
# ---------------------------------------------------------------------------


def _load_physics_plugin(name: str) -> AbstractSensorPhysicsPlugin:
    """Load physics plugin by entry-point name."""
    ph_eps = {ep.name: ep for ep in entry_points(group="spectraagent.sensor_physics")}
    if name not in ph_eps:
        raise typer.BadParameter(
            f"Physics plugin '{name}' not found. Run 'spectraagent plugins list'."
        )
    return cast(AbstractSensorPhysicsPlugin, ph_eps[name].load()())


# ---------------------------------------------------------------------------
# Helper: acquisition broadcast loop
# ---------------------------------------------------------------------------


def _acquisition_loop(
    driver: AbstractHardwareDriver,
    app: FastAPI,
) -> None:
    """Daemon thread: read spectra, run quality/drift agents, broadcast to WS clients."""
    import time

    import numpy as np

    wl_list = driver.get_wavelengths().tolist()
    wl_np = np.array(wl_list)
    frame_num = 0

    while True:
        try:
            intensities = driver.read_spectrum()
        except Exception as exc:
            log.warning("Acquisition error: %s", exc)
            time.sleep(1.0)
            continue

        frame_num += 1
        _process_acquired_frame(app, wl_list, wl_np, frame_num, intensities)


def _process_acquired_frame(
    app: FastAPI,
    wl_list: list[float],
    wl_np,
    frame_num: int,
    intensities,
) -> bool:
    """Process one acquired frame.

    Returns True when the frame continues through the broadcast path, and False
    when it is hard-blocked by the quality gate.
    """
    import asyncio
    import json

    app.state.latest_spectrum = {
        "wl": wl_list,
        "intensities": intensities,
    }
    if getattr(app.state, "session_running", False):
        app.state.session_frame_count += 1

    quality_agent = getattr(app.state, "quality_agent", None)
    if quality_agent is not None:
        passes = quality_agent.process(frame_num, wl_np, intensities)
        if not passes:
            return False

    drift_agent = getattr(app.state, "drift_agent", None)
    plugin = getattr(app.state, "plugin", None)
    direct_peak_wl: float | None = None  # peak_wl from plugin, used as broadcast fallback
    if plugin is not None:
        try:
            direct_peak_wl = plugin.detect_peak(wl_np, intensities)
            if direct_peak_wl is not None and drift_agent is not None:
                drift_agent.update(frame_num, direct_peak_wl)
        except Exception as exc:
            log.debug("DriftAgent.update() failed: %s", exc)

    # Run ML inference pipeline if wired
    pipeline_result = None
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        try:
            pipeline_result = pipeline.process_spectrum(wl_np, intensities)
        except Exception as exc:
            log.debug("RealTimePipeline.process_spectrum() failed: %s", exc)

    # Accumulate session events for post-session SessionAnalyzer
    if pipeline_result is not None and pipeline_result.success:
        session_events = getattr(app.state, "session_events", None)
        if session_events is not None and getattr(app.state, "session_running", False):
            sd = pipeline_result.spectrum
            ev = {"type": "measurement"}
            if sd.concentration_ppm is not None:
                ev["concentration_ppm"] = sd.concentration_ppm
            if sd.ci_low is not None:
                ev["ci_low"] = sd.ci_low
            if sd.ci_high is not None:
                ev["ci_high"] = sd.ci_high
            if sd.wavelength_shift is not None:
                ev["wavelength_shift"] = sd.wavelength_shift
            if sd.snr is not None:
                ev["snr"] = sd.snr
            if sd.peak_wavelength is not None:
                ev["peak_wavelength"] = sd.peak_wavelength
            # Elapsed time (s) from session start — used for kinetics fitting (τ₆₃/τ₉₅)
            session_start_t = getattr(app.state, "session_start_monotonic", None)
            if session_start_t is not None:
                import time as _time
                ev["timestamp_s"] = _time.monotonic() - session_start_t
            session_events.append(ev)

    # Persist per-frame pipeline results to CSV (crash-safe streaming write).
    app_loop = getattr(app.state, "asyncio_loop", None)
    sw = getattr(app.state, "session_writer", None)
    if sw is not None and pipeline_result is not None and getattr(app.state, "session_running", False):
        sd = pipeline_result.spectrum
        frame_row = {
            "frame": frame_num,
            "timestamp": getattr(sd, "timestamp", ""),
            "peak_wavelength": round(sd.peak_wavelength, 4) if sd.peak_wavelength is not None else "",
            "wavelength_shift": round(sd.wavelength_shift, 4) if sd.wavelength_shift is not None else "",
            "concentration_ppm": round(sd.concentration_ppm, 4) if sd.concentration_ppm is not None else "",
            "ci_low": round(sd.ci_low, 4) if sd.ci_low is not None else "",
            "ci_high": round(sd.ci_high, 4) if sd.ci_high is not None else "",
            "snr": round(sd.snr, 2) if sd.snr is not None else "",
            "gas_type": getattr(sd, "gas_type", "") or "",
            "confidence_score": round(sd.confidence_score, 3) if getattr(sd, "confidence_score", None) is not None else "",
        }
        if app_loop is not None:
            app_loop.call_soon_threadsafe(sw.append_frame_result, frame_row)
        else:
            sw.append_frame_result(frame_row)

    spectrum_bc = getattr(app.state, "spectrum_bc", None)
    if app_loop is not None and spectrum_bc is not None and hasattr(spectrum_bc, "broadcast"):
        payload: dict = {"wl": wl_list, "i": intensities.tolist(), "frame": frame_num}
        if pipeline_result is not None and pipeline_result.success:
            sd = pipeline_result.spectrum
            if sd.concentration_ppm is not None:
                payload["concentration_ppm"] = round(sd.concentration_ppm, 4)
            if sd.ci_low is not None:
                payload["ci_low"] = round(sd.ci_low, 4)
            if sd.ci_high is not None:
                payload["ci_high"] = round(sd.ci_high, 4)
            if sd.wavelength_shift is not None:
                payload["peak_shift_nm"] = round(sd.wavelength_shift, 4)
            if sd.snr is not None:
                payload["snr"] = round(sd.snr, 2)
            if sd.peak_wavelength is not None:
                payload["peak_wavelength"] = round(sd.peak_wavelength, 4)
            if getattr(sd, "gas_type", None) is not None:
                payload["gas_type"] = sd.gas_type
            if getattr(sd, "confidence_score", None) is not None:
                payload["confidence_score"] = round(sd.confidence_score, 3)
        # Fallback: include direct plugin peak when pipeline didn't provide it.
        # This ensures the UI always shows a live peak_wavelength, even before
        # a reference spectrum is captured or the ML pipeline is ready.
        if "peak_wavelength" not in payload and direct_peak_wl is not None:
            payload["peak_wavelength"] = round(direct_peak_wl, 4)
        msg = json.dumps(payload)
        broadcast_fn = spectrum_bc.broadcast
        app_loop.call_soon_threadsafe(
            lambda m=msg, fn=broadcast_fn: asyncio.ensure_future(fn(m))
        )

    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@cli.command()
def start(
    simulate: bool = typer.Option(False, "--simulate", help="Force simulation mode"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
    host: str = typer.Option("", "--host", help="Bind host (overrides config)"),
    port: int = typer.Option(0, "--port", help="Port (overrides config)"),
    physics: str = typer.Option("", "--physics", help="Sensor physics plugin name"),
) -> None:
    """Start the SpectraAgent server and open the browser."""
    import os
    import threading
    import webbrowser

    import uvicorn

    from spectraagent.config import load_config

    cfg = load_config()

    bind_host = host or cfg.server.host
    bind_port = port or cfg.server.port
    physics_name = physics or cfg.physics.default_plugin

    # Step 1: Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo(
            "WARNING: ANTHROPIC_API_KEY not set -- Claude agents disabled.",
            err=True,
        )

    # Step 2: Discover and load hardware driver
    driver = _load_driver(simulate, cfg)
    typer.echo(f"Hardware: {driver.name}")

    # Step 3: Discover and load physics plugin
    plugin = _load_physics_plugin(physics_name)
    typer.echo(f"Physics: {plugin.name}")

    # Step 4: Build the FastAPI app
    from spectraagent.webapp.server import create_app

    app = create_app(simulate=simulate)
    app.state.driver = driver
    app.state.plugin = plugin
    app.state.reference = None
    app.state.cached_ref = None

    # Step 5a: Create deterministic agents
    from spectraagent.webapp.agents.calibration import CalibrationAgent
    from spectraagent.webapp.agents.drift import DriftAgent
    from spectraagent.webapp.agents.planner import ExperimentPlannerAgent
    from spectraagent.webapp.agents.quality import QualityAgent
    agent_bus = app.state.agent_bus

    app.state.quality_agent = QualityAgent(agent_bus)
    app.state.drift_agent = DriftAgent(
        agent_bus,
        integration_time_ms=cfg.hardware.integration_time_ms,
    )
    app.state.calibration_agent = CalibrationAgent(agent_bus)
    app.state.planner_agent = ExperimentPlannerAgent(agent_bus)
    typer.echo("Agents ready: Quality, Drift, Calibration, Planner")

    # Step 5aa: Wire RealTimePipeline for per-frame ML inference
    from src.inference.realtime_pipeline import PipelineConfig, RealTimePipeline

    pipeline_cfg = PipelineConfig(
        integration_time_ms=cfg.hardware.integration_time_ms,
        peak_search_min_nm=cfg.physics.search_min_nm,
        peak_search_max_nm=cfg.physics.search_max_nm,
        reference_wavelength=-1.0,  # -1.0 sentinel → auto from midpoint of search window
    )
    app.state.pipeline = RealTimePipeline(pipeline_cfg)
    app.state.session_events = []
    typer.echo("RealTimePipeline wired")

    # Step 5b: Create SensorMemory and wire Claude API agents
    from spectraagent.webapp.agents.calibration_validator import (
        CalibrationValidationOrchestrator,
    )
    from spectraagent.webapp.agents.claude_agents import (
        AnomalyExplainer,
        ClaudeAgentRunner,
        DiagnosticsAgent,
        ExperimentNarrator,
        ReportWriter,
    )
    from spectraagent.webapp.agents.sensor_health import SensorHealthAgent

    # SensorMemory accumulates calibration outcomes, drift stats, and failure
    # events across sessions.  Agents use it for data-driven context instead of
    # hardcoded literature values.
    try:
        from spectraagent.knowledge.sensor_memory import SensorMemory as _SensorMemory
        sensor_memory = _SensorMemory(
            memory_dir=Path("output/memory"),
            sensor_id=driver.name,
        )
        app.state.sensor_memory = sensor_memory
        typer.echo(f"SensorMemory loaded ({sensor_memory.get_sensor_health_summary()['total_sessions']} prior sessions)")
    except Exception as _exc:
        sensor_memory = None
        typer.echo(f"SensorMemory unavailable: {_exc}", err=True)

    # Infer sensor modality from physics plugin name for context builder preamble.
    def _infer_sensor_type(plugin_name: str) -> str:
        name = plugin_name.lower()
        if "lspr" in name:
            return "lspr"
        if "spr" in name:
            return "spr"
        if "fluor" in name:
            return "fluorescence"
        return "optical"

    _sensor_type = _infer_sensor_type(physics_name)

    # get_analyte is a lazy callable so agents always see the current session's
    # gas label at event time, not a snapshot from agent construction time.
    def _get_analyte() -> str:
        _acq_cfg = cast(dict[str, object], getattr(app.state, "_acq_config", {}))
        return str(_acq_cfg.get("gas_label", "unknown"))

    app.state.anomaly_explainer = AnomalyExplainer(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        cooldown_s=cfg.agents.anomaly_explainer_cooldown_s,
        auto_explain=cfg.agents.auto_explain,
        memory=sensor_memory,
        sensor_type=_sensor_type,
        get_analyte=_get_analyte,
    )
    app.state.experiment_narrator = ExperimentNarrator(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        auto_explain=cfg.agents.auto_explain,
        memory=sensor_memory,
        get_analyte=_get_analyte,
    )
    app.state.diagnostics_agent = DiagnosticsAgent(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        cooldown_s=cfg.agents.diagnostics_cooldown_s,
    )
    app.state.report_writer = ReportWriter(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        memory=sensor_memory,
        get_analyte=_get_analyte,
    )
    # SensorHealthAgent: multi-metric sensor lifecycle monitor.
    # Fires on session_complete (every session stop) and model_selected.
    app.state.sensor_health_agent = SensorHealthAgent(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        memory=sensor_memory,
        get_analyte=_get_analyte,
        auto_explain=cfg.agents.auto_explain,
        sensor_type=_sensor_type,
    )

    # CalibrationValidationOrchestrator: ICH Q2(R1) gap tracker.
    # Fires on session_complete and tells researchers what tests remain.
    app.state.calibration_validator = CalibrationValidationOrchestrator(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        memory=sensor_memory,
        get_analyte=_get_analyte,
        auto_explain=cfg.agents.auto_explain,
    )

    app.state.claude_runner = ClaudeAgentRunner(
        agent_bus,
        anomaly_explainer=app.state.anomaly_explainer,
        experiment_narrator=app.state.experiment_narrator,
        diagnostics_agent=app.state.diagnostics_agent,
    )
    # Register lifecycle agents — added after construction so ClaudeAgentRunner
    # does not need to know their types up front.
    app.state.claude_runner.add_agent(app.state.sensor_health_agent)
    app.state.claude_runner.add_agent(app.state.calibration_validator)

    typer.echo(
        f"Claude agents ready: AnomalyExplainer ({_sensor_type}), ExperimentNarrator, "
        "DiagnosticsAgent, ReportWriter, SensorHealthAgent, CalibrationValidationOrchestrator"
    )

    # Step 5: Start acquisition thread after AgentBus is wired (via app lifespan)
    async def _start_runtime_services() -> None:
        """Start acquisition and Claude runner after app lifespan startup begins."""
        acq_thread = threading.Thread(
            target=_acquisition_loop,
            args=(driver, app),
            daemon=True,
            name="spectraagent-acquisition",
        )
        acq_thread.start()
        typer.echo("Acquisition loop started")

        # Start Claude agent runner (requires live event loop)
        claude_runner = getattr(app.state, "claude_runner", None)
        if claude_runner is not None:
            claude_runner.start()
            typer.echo("Claude agent runner started")

    app.state.startup_callbacks.append(_start_runtime_services)

    # Step 6: Open browser
    if not no_browser and cfg.server.open_browser:
        browser_url = f"http://localhost:{bind_port}"
        threading.Timer(1.5, lambda: webbrowser.open(browser_url)).start()
        typer.echo(f"Opening browser at {browser_url}")

    # Step 7: Start uvicorn (blocking)
    typer.echo(f"Serving at http://{bind_host}:{bind_port}")
    uvicorn.run(app, host=bind_host, port=bind_port, log_level="warning")


# ---------------------------------------------------------------------------
# sessions sub-app
# ---------------------------------------------------------------------------

sessions_app = typer.Typer(name="sessions", help="Manage recorded sessions.", no_args_is_help=True)
cli.add_typer(sessions_app)


@sessions_app.command(name="list")
def sessions_list(
    sessions_dir: str = typer.Option("output/sessions", "--dir", help="Sessions directory"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum rows to show"),
) -> None:
    """List recorded sessions, newest first."""
    import contextlib
    import json
    from pathlib import Path

    base = Path(sessions_dir)
    if not base.exists():
        typer.echo(f"No sessions directory found at '{base}'. Run a session first.")
        raise typer.Exit(0)

    metas = []
    for meta_path in base.glob("*/session_meta.json"):
        with contextlib.suppress(json.JSONDecodeError, OSError):
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))
    metas.sort(key=lambda s: s.get("started_at", ""), reverse=True)

    if not metas:
        typer.echo("No sessions recorded yet.")
        raise typer.Exit(0)

    header = f"{'SESSION ID':<22}  {'STARTED':<26}  {'FRAMES':>6}  STATUS"
    typer.echo(header)
    typer.echo("-" * len(header))
    for m in metas[:limit]:
        sid = m.get("session_id", "?")
        started = m.get("started_at", "")[:19].replace("T", " ")
        frames = m.get("frame_count", "?")
        status = "stopped" if m.get("stopped_at") else "active"
        typer.echo(f"{sid:<22}  {started:<26}  {frames!s:>6}  {status}")


@sessions_app.command(name="get")
def sessions_get(
    session_id: str = typer.Argument(..., help="Session ID (YYYYMMDD_HHMMSS)"),
    sessions_dir: str = typer.Option("output/sessions", "--dir", help="Sessions directory"),
    events: bool = typer.Option(False, "--events", help="Include last 20 agent events"),
) -> None:
    """Show metadata for a specific session."""
    import json
    from pathlib import Path

    meta_path = Path(sessions_dir) / session_id / "session_meta.json"
    if not meta_path.exists():
        typer.echo(f"Session '{session_id}' not found in '{sessions_dir}'.", err=True)
        raise typer.Exit(1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(meta, indent=2))

    if events:
        events_path = Path(sessions_dir) / session_id / "agent_events.jsonl"
        if events_path.exists():
            lines = events_path.read_text(encoding="utf-8").splitlines()
            recent = lines[-20:]
            typer.echo(f"\n--- Last {len(recent)} agent events ---")
            for line in recent:
                typer.echo(line)


# ---------------------------------------------------------------------------
# plugins command
# ---------------------------------------------------------------------------


@cli.command(name="plugins")
def plugins_cmd(
    action: str = typer.Argument("list", help="Action: list"),
) -> None:
    """Show discovered plugins and their status."""
    if action != "list":
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(1)

    typer.echo("\nHardware Drivers:")
    hw_eps = entry_points(group="spectraagent.hardware")
    for ep in hw_eps:
        try:
            cls = ep.load()
            class_issues = validate_driver_class(cls)
            status = "✓ loadable" if not class_issues else f"✗ invalid: {'; '.join(class_issues)}"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")

    typer.echo("\nSensor Physics Plugins:")
    ph_eps = entry_points(group="spectraagent.sensor_physics")
    for ep in ph_eps:
        try:
            ep.load()
            status = "✓ loadable"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")


@cli.command(name="robustness")
def robustness_cmd(
    param: str = typer.Option("integration_time", "--param", help="Parameter name to sweep"),
    range_: str = typer.Option("45:55", "--range", help="Sweep range '<start>:<end>'"),
    steps: int = typer.Option(3, "--steps", help="Number of parameter points"),
    runs: int = typer.Option(5, "--runs", help="Runs per parameter point"),
    dataset_dir: str = typer.Option(
        "data/automation_dataset",
        "--dataset-dir",
        help="Directory containing labelled CSV spectra",
    ),
    output_csv: str = typer.Option(
        "",
        "--output-csv",
        help="Optional CSV output path for robustness summary",
    ),
) -> None:
    """Run robustness sweep and print publication-ready R²/LOD comparison table."""
    runner = RobustnessRunner(
        dataset_dir=Path(dataset_dir),
        param_name=param,
        range_spec=range_,
        steps=steps,
        runs_per_step=runs,
    )
    try:
        results = runner.run()
    except Exception as exc:
        typer.echo(f"Robustness run failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo("\nRobustness sweep summary\n")
    typer.echo(runner.format_table(results))

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        runner.to_dataframe(results).to_csv(out_path, index=False)
        typer.echo(f"\nSaved CSV summary: {out_path}")


if __name__ == "__main__":
    cli()
