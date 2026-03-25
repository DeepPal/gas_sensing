"""
serve.py — Au-MIP LSPR Inference API Server
=============================================

Launches the FastAPI inference server with uvicorn.

Usage
-----
::

    # Basic (development)
    python serve.py

    # Production
    python serve.py --host 0.0.0.0 --port 8000 --workers 2

    # Via uvicorn directly
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

After starting, open:
  http://localhost:8000/docs   → Swagger UI (interactive API explorer)
  http://localhost:8000/health → Liveness check
  http://localhost:8000/status → Pipeline statistics

Endpoints
---------
  POST /predict        → Run inference on one spectrum
  POST /predict/batch  → Run inference on up to 1000 spectra
  GET  /health         → {"status": "ok", "models_loaded": {...}}
  GET  /status         → Pipeline statistics
"""

from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and launch the uvicorn server."""

    # Windows console UTF-8 fix
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Au-MIP LSPR Gas Sensing Inference API")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (default: 1)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload (development only).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn[standard]")
        sys.exit(1)

    log.info(
        "Starting Au-MIP LSPR API on http://%s:%d  (docs: /docs)",
        args.host,
        args.port,
    )

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
