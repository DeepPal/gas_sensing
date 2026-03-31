"""
dashboard/health.py - Health check endpoints for monitoring
===========================================================

Provides simple health check endpoints that can be polled by monitoring systems.
Suitable for research lab environments with manual monitoring or simple scripts.

Health checks include:
- Application startup status
- Hardware availability (spectrometer, live server)
- Disk space for data output
- Log file accessibility
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class HealthCheck:
    """Health status of the Au-MIP LSPR dashboard application."""

    def __init__(self, app_root: Path | None = None) -> None:
        self.app_root = app_root or Path(__file__).resolve().parents[1]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.hostname = platform.node()

    def check_disk_space(self, min_gb: float = 1.0) -> dict:
        """Check available disk space in output directory."""
        output_dir = self.app_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / (1024**3)
        healthy = free_gb >= min_gb

        return {
            "available_gb": round(free_gb, 2),
            "total_gb": round(total / (1024**3), 2),
            "healthy": healthy,
            "status": "OK" if healthy else f"LOW (< {min_gb} GB)",
        }

    def check_logs(self) -> dict:
        """Check if log files are writable."""
        log_dir = self.app_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "dashboard.log"

        try:
            # Try to write a test message
            with open(log_file, "a") as f:
                f.write("")  # Empty write to check permissions
            healthy = True
            status = "OK"
        except (PermissionError, IOError) as e:
            healthy = False
            status = f"NOT WRITABLE: {e}"

        return {
            "log_file": str(log_file),
            "healthy": healthy,
            "status": status,
        }

    def check_hardware(self) -> dict:
        """Check spectrometer and live server availability."""
        results = {
            "spectrometer": {"available": False, "status": "Not checked yet"},
            "live_server": {"available": False, "status": "Not checked yet"},
        }

        # Try to import hardware modules
        try:
            from src.acquisition import CCS200Spectrometer

            results["spectrometer"]["available"] = CCS200Spectrometer is not None
            results["spectrometer"]["status"] = "Importable" if results["spectrometer"][
                "available"
            ] else "Import failed"
        except ImportError as e:
            results["spectrometer"]["status"] = f"Import error: {e}"

        # Check live server connectivity
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("127.0.0.1", 5006))
            sock.close()
            results["live_server"]["available"] = result == 0
            results["live_server"]["status"] = (
                "Connected" if result == 0 else f"Connection failed (code {result})"
            )
        except Exception as e:
            results["live_server"]["status"] = f"Check error: {e}"

        return results

    def get_status(self) -> dict:
        """Get complete health status as a dictionary."""
        return {
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "application": {
                "name": "Au-MIP LSPR Gas Sensing Platform",
                "version": "1.0.0",
                "status": "running",
            },
            "disk_space": self.check_disk_space(),
            "logs": self.check_logs(),
            "hardware": self.check_hardware(),
            "overall_healthy": all(
                [
                    self.check_disk_space()["healthy"],
                    self.check_logs()["healthy"],
                ]
            ),
        }

    def to_json(self) -> str:
        """Serialize health status to JSON."""
        return json.dumps(self.get_status(), indent=2, default=str)

    def print_report(self) -> None:
        """Print a human-readable health report."""
        status = self.get_status()

        print("\n" + "=" * 70)
        print("Au-MIP LSPR HEALTH CHECK REPORT")
        print("=" * 70)
        print(f"Timestamp:  {status['timestamp']}")
        print(f"Hostname:   {status['hostname']}")
        print(f"\nApplication: {status['application']['name']} v{status['application']['version']}")

        print("\n📦 Disk Space:")
        disk = status["disk_space"]
        print(
            f"  Available: {disk['available_gb']:.2f} GB / {disk['total_gb']:.2f} GB"
        )
        print(f"  Status:    {disk['status']}")

        print("\n📝 Logs:")
        logs = status["logs"]
        print(f"  File:  {logs['log_file']}")
        print(f"  Status: {logs['status']}")

        print("\n⚙️  Hardware:")
        hw = status["hardware"]
        spec = hw["spectrometer"]
        print(f"  Spectrometer: {spec['status']} {'✓' if spec['available'] else '✗'}")
        live = hw["live_server"]
        print(f"  Live Server:  {live['status']} {'✓' if live['available'] else '✗'}")

        print(
            f"\n{'✓ OVERALL HEALTHY' if status['overall_healthy'] else '✗ ISSUES DETECTED'}"
        )
        print("=" * 70 + "\n")


def startup_check() -> bool:
    """
    Run health checks on startup. Log warnings for any issues.
    
    Returns
    -------
    bool
        True if all critical checks pass, False otherwise.
    """
    hc = HealthCheck()
    status = hc.get_status()

    critical_ok: bool = status["overall_healthy"]  # type: ignore[assignment]

    if not critical_ok:
        log.warning("⚠️  Health check warnings detected on startup:")
        if not status["disk_space"]["healthy"]:
            log.warning(f"  • Disk space: {status['disk_space']['status']}")
        if not status["logs"]["healthy"]:
            log.warning(f"  • Logs: {status['logs']['status']}")

    hardware = status["hardware"]
    if not hardware["spectrometer"]["available"]:
        log.info("  ℹ️ Spectrometer not available — using simulation mode")
    if not hardware["live_server"]["available"]:
        log.warning("  ⚠️  Live server not available — real-time view disabled")

    return critical_ok


if __name__ == "__main__":
    # CLI usage: python -m dashboard.health
    hc = HealthCheck()
    hc.print_report()
