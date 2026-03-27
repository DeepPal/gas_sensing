"""spectraagent.__main__ — CLI entry point (Typer)."""
from __future__ import annotations

from importlib.metadata import entry_points as _entry_points

import typer

cli = typer.Typer(
    name="spectraagent",
    help="Universal Agentic Spectroscopy Platform",
    no_args_is_help=True,
)


@cli.command()
def start(
    simulate: bool = typer.Option(False, "--simulate", help="Force simulation mode"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
    host: str = typer.Option("", "--host", help="Bind host (overrides config)"),
    port: int = typer.Option(0, "--port", help="Port (overrides config)"),
    physics: str = typer.Option("", "--physics", help="Sensor physics plugin (overrides config)"),
) -> None:
    """Start the SpectraAgent server and open the browser."""
    typer.echo("spectraagent start — not yet implemented (Task 13)")


@cli.command(name="plugins")
def plugins_cmd(
    action: str = typer.Argument("list", help="Action: list"),
) -> None:
    """Show discovered plugins and their status."""
    if action != "list":
        typer.echo(f"Unknown action: {action}", err=True)
        raise typer.Exit(1)

    typer.echo("\nHardware Drivers:")
    hw_eps = _entry_points(group="spectraagent.hardware")
    for ep in hw_eps:
        try:
            ep.load()
            status = "✓ loadable"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")

    typer.echo("\nSensor Physics Plugins:")
    ph_eps = _entry_points(group="spectraagent.sensor_physics")
    for ep in ph_eps:
        try:
            ep.load()
            status = "✓ loadable"
        except Exception as exc:
            status = f"✗ {exc}"
        typer.echo(f"  [{ep.name}]  {ep.value}  —  {status}")


if __name__ == "__main__":
    cli()
