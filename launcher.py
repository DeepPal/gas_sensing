"""
SpectraAgent unified launcher.
Starts both the SpectraAgent server and Streamlit dashboard,
waits for both to be ready, opens browsers, then monitors until Ctrl+C.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
SPECTRAAGENT_PORT = 8765
DASHBOARD_PORT = 8501
SPECTRAAGENT_URL = f"http://localhost:{SPECTRAAGENT_PORT}/app"
DASHBOARD_URL = f"http://localhost:{DASHBOARD_PORT}"
HEALTH_URL = f"http://localhost:{SPECTRAAGENT_PORT}/api/health"
READY_TIMEOUT = 30  # seconds to wait for services to become ready


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_ready(url: str) -> bool:
    """Return True if the URL responds with HTTP 200."""
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _wait_ready(url: str, label: str, timeout: int = READY_TIMEOUT) -> bool:
    """Poll url until ready or timeout. Returns True if ready."""
    deadline = time.monotonic() + timeout
    dots = 0
    while time.monotonic() < deadline:
        if _is_ready(url):
            print(f"\r  {label}: ready ✓                    ")
            return True
        dots = (dots + 1) % 4
        print(f"\r  Waiting for {label}{'.' * dots}   ", end="", flush=True)
        time.sleep(0.5)
    print(f"\r  WARNING: {label} did not respond within {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
procs: list[subprocess.Popen] = []


def _cleanup(*_) -> None:
    """Stop all child processes cleanly."""
    print("\n\n  Stopping services...")
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    print("  Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    simulate = "--hardware" not in sys.argv
    mode_flag = "--simulate" if simulate else ""
    mode_label = "Simulation" if simulate else "Hardware"

    if not PYTHON.exists():
        print(f"ERROR: {PYTHON} not found. Run the installer first.")
        sys.exit(1)

    # Register cleanup for Ctrl+C and termination signals
    signal.signal(signal.SIGINT, lambda s, f: (_cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (_cleanup(), sys.exit(0)))

    print(f"  Mode: {mode_label}")
    print()

    # ── Build HTTPS args for dashboard ───────────────────────────────────
    cert = ROOT / ".streamlit" / "certs" / "server.crt"
    key  = ROOT / ".streamlit" / "certs" / "server.key"
    https_args: list[str] = []
    if cert.exists() and key.exists():
        https_args = [
            f"--server.sslCertFile={cert}",
            f"--server.sslKeyFile={key}",
        ]

    # ── Start SpectraAgent server ─────────────────────────────────────────
    print("  [1/2] Starting SpectraAgent server...")
    sa_cmd = [str(PYTHON), "-m", "spectraagent", "start",
              "--port", str(SPECTRAAGENT_PORT), "--no-browser"]
    if mode_flag:
        sa_cmd.append(mode_flag)

    sa_proc = subprocess.Popen(
        sa_cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(sa_proc)

    # ── Start Streamlit dashboard ─────────────────────────────────────────
    print("  [2/2] Starting Streamlit dashboard...")
    dash_cmd = [
        str(PYTHON), "-m", "streamlit", "run", "dashboard/app.py",
        "--server.port", str(DASHBOARD_PORT),
        "--server.headless", "true",
        "--server.enableXsrfProtection", "true",
        "--logger.level", "warning",
        "--client.showErrorDetails", "false",
        *https_args,
    ]
    # Forward dashboard password environment variable if set
    env = os.environ.copy()
    dash_proc = subprocess.Popen(
        dash_cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(dash_proc)

    # ── Wait for both to be ready ─────────────────────────────────────────
    print()
    sa_ready = _wait_ready(HEALTH_URL, "SpectraAgent")
    # Streamlit doesn't have a JSON health endpoint — check the main page
    dash_ready = _wait_ready(DASHBOARD_URL, "Dashboard")

    # ── Open browsers ─────────────────────────────────────────────────────
    print()
    if sa_ready:
        webbrowser.open(SPECTRAAGENT_URL)
        print(f"  SpectraAgent  ->  {SPECTRAAGENT_URL}")
    if dash_ready:
        webbrowser.open(DASHBOARD_URL)
        print(f"  Dashboard     ->  {DASHBOARD_URL}")

    print()
    print("  Both services running. Press Ctrl+C to stop.")
    print()

    # ── Monitor until either process dies or Ctrl+C ───────────────────────
    try:
        while True:
            # Check if either process died unexpectedly
            if sa_proc.poll() is not None:
                print("\n  SpectraAgent stopped unexpectedly.")
                break
            if dash_proc.poll() is not None:
                print("\n  Dashboard stopped unexpectedly.")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
