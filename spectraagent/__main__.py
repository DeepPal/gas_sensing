"""spectraagent.__main__ — CLI entry point (Typer)."""
from __future__ import annotations

import logging
from importlib.metadata import entry_points as entry_points

import typer

log = logging.getLogger(__name__)

cli = typer.Typer(
    name="spectraagent",
    help="Universal Agentic Spectroscopy Platform",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helper: load hardware driver
# ---------------------------------------------------------------------------


def _load_driver(simulate: bool, cfg) -> "AbstractHardwareDriver":
    """Load hardware driver: simulation if forced, else try config default, fallback sim."""
    from spectraagent.drivers.simulation import SimulationDriver

    if simulate:
        drv = SimulationDriver(integration_time_ms=cfg.hardware.integration_time_ms)
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
        drv = cls(integration_time_ms=cfg.hardware.integration_time_ms)
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


def _load_physics_plugin(name: str) -> "AbstractSensorPhysicsPlugin":
    """Load physics plugin by entry-point name."""
    ph_eps = {ep.name: ep for ep in entry_points(group="spectraagent.sensor_physics")}
    if name not in ph_eps:
        raise typer.BadParameter(
            f"Physics plugin '{name}' not found. Run 'spectraagent plugins list'."
        )
    return ph_eps[name].load()()


# ---------------------------------------------------------------------------
# Helper: acquisition broadcast loop
# ---------------------------------------------------------------------------


def _acquisition_loop(driver: "AbstractHardwareDriver", app: "FastAPI") -> None:
    """Daemon thread: read spectra and broadcast to WebSocket clients."""
    import asyncio
    import json
    import time

    wl = driver.get_wavelengths().tolist()

    while True:
        try:
            intensities = driver.read_spectrum()
        except Exception as exc:
            log.warning("Acquisition error: %s", exc)
            time.sleep(1.0)
            continue

        spectrum_bc = getattr(app.state, "spectrum_bc", None)
        if spectrum_bc is not None:
            msg = json.dumps({"wl": wl, "i": intensities.tolist()})
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(
                    lambda m=msg: asyncio.ensure_future(spectrum_bc.broadcast(m))
                )
            except RuntimeError:
                pass


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

    # Step 5: Start the acquisition broadcast loop in a daemon thread
    acq_thread = threading.Thread(
        target=_acquisition_loop,
        args=(driver, app),
        daemon=True,
        name="spectraagent-acquisition",
    )
    acq_thread.start()
    typer.echo("Acquisition loop started")

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
