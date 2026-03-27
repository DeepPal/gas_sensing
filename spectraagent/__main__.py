"""spectraagent.__main__ — CLI entry point (Typer)."""
from __future__ import annotations

import logging
from importlib.metadata import entry_points as entry_points
from typing import TYPE_CHECKING, cast

import typer

from spectraagent.drivers.base import AbstractHardwareDriver
from spectraagent.physics.base import AbstractSensorPhysicsPlugin

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

cli = typer.Typer(
    name="spectraagent",
    help="Universal Agentic Spectroscopy Platform",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helper: load hardware driver
# ---------------------------------------------------------------------------


def _load_driver(simulate: bool, cfg) -> AbstractHardwareDriver:
    """Load hardware driver: simulation if forced, else try config default, fallback sim."""
    from spectraagent.drivers.simulation import SimulationDriver

    if simulate:
        drv: AbstractHardwareDriver = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
        drv.connect()
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
        drv = cls(integration_time_ms=cfg.hardware.integration_time_ms)  # type: ignore[assignment]
        drv.connect()
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
    import asyncio
    import json
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
    if drift_agent is not None and plugin is not None:
        try:
            peak_wl = plugin.detect_peak(wl_np, intensities)
            if peak_wl is not None:
                drift_agent.update(frame_num, peak_wl)
        except Exception as exc:
            log.debug("DriftAgent.update() failed: %s", exc)

    app_loop = getattr(app.state, "asyncio_loop", None)
    spectrum_bc = getattr(app.state, "spectrum_bc", None)
    if app_loop is not None and spectrum_bc is not None and hasattr(spectrum_bc, "broadcast"):
        msg = json.dumps({"wl": wl_list, "i": intensities.tolist()})
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

    # Step 5b: Create Claude API agents
    from spectraagent.webapp.agents.claude_agents import (
        AnomalyExplainer,
        ClaudeAgentRunner,
        DiagnosticsAgent,
        ExperimentNarrator,
        ReportWriter,
    )

    app.state.anomaly_explainer = AnomalyExplainer(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        cooldown_s=cfg.agents.anomaly_explainer_cooldown_s,
        auto_explain=cfg.agents.auto_explain,
    )
    app.state.experiment_narrator = ExperimentNarrator(
        agent_bus,
        model=cfg.claude.model,
        timeout_s=cfg.claude.timeout_s,
        auto_explain=cfg.agents.auto_explain,
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
    )
    app.state.claude_runner = ClaudeAgentRunner(
        agent_bus,
        anomaly_explainer=app.state.anomaly_explainer,
        experiment_narrator=app.state.experiment_narrator,
        diagnostics_agent=app.state.diagnostics_agent,
    )
    typer.echo(
        "Claude agents ready: AnomalyExplainer, ExperimentNarrator, "
        "DiagnosticsAgent, ReportWriter"
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
            ep.load()
            status = "✓ loadable"
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


if __name__ == "__main__":
    cli()
