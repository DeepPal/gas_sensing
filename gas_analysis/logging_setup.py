"""
gas_analysis.logging_setup
==========================
Centralised logging configuration for the Au-MIP LSPR gas sensing platform.

Usage
-----
Call ``configure_logging()`` once at application startup (in ``run.py`` or
``dashboard/app.py``).  Every other module just does::

    import logging
    log = logging.getLogger(__name__)
    log.info("message")

All ``gas_analysis.*`` and ``sensor_app.*`` loggers inherit the root handler
configured here automatically.

Log levels
----------
- DEBUG    : detailed diagnostic traces (hardware register reads, per-frame stats)
- INFO     : normal operational events (session start, calibration update)
- WARNING  : recoverable issues (low SNR frame, simulation fallback)
- ERROR    : failures that affect a result but let the system keep running
- CRITICAL : failures that require operator intervention or a restart
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Default format strings
# ---------------------------------------------------------------------------
_CONSOLE_FMT = "%(levelname)-8s %(name)s: %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Loggers that are too noisy at INFO and should stay at WARNING by default
_NOISY_THIRD_PARTY = [
    "matplotlib",
    "PIL",
    "streamlit",
    "urllib3",
    "asyncio",
    "torch",
]


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    console: bool = True,
    quiet_third_party: bool = True,
) -> logging.Logger:
    """Configure the root logger for the gas sensing platform.

    Parameters
    ----------
    level:
        Minimum log level for the platform's own loggers
        (``gas_analysis.*``, ``sensor_app.*``).
    log_file:
        If given, a rotating file handler is added writing to this path.
        The directory is created automatically.  Log files rotate at 5 MB
        and keep the last 3 backups.
    console:
        Whether to attach a ``StreamHandler`` writing to *stderr*.
    quiet_third_party:
        Suppress chatty third-party loggers (matplotlib, torch, etc.) below
        WARNING.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter level

    # Remove any handlers already attached (idempotent re-initialisation)
    root.handlers.clear()

    # ------------------------------------------------------------------
    # Console handler
    # ------------------------------------------------------------------
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(_CONSOLE_FMT))
        root.addHandler(console_handler)

    # ------------------------------------------------------------------
    # Rotating file handler
    # ------------------------------------------------------------------
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # log everything to file
        file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        root.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Suppress noisy third-party loggers
    # ------------------------------------------------------------------
    if quiet_third_party:
        for name in _NOISY_THIRD_PARTY:
            logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger("gas_analysis")
    logger.debug("Logging initialised (level=%s, file=%s)", logging.getLevelName(level), log_file)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the platform hierarchy.

    Equivalent to ``logging.getLogger(name)`` but documents intent.

    Example
    -------
    >>> log = get_logger(__name__)
    >>> log.info("pipeline started")
    """
    return logging.getLogger(name)
