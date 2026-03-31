"""
src.spectrometer.registry
==========================
Driver registry and factory for the hardware abstraction layer.

The registry is a central catalogue of all available spectrometer drivers.
Drivers register themselves (or are registered by the library at import time)
and are instantiated via :meth:`SpectrometerRegistry.create`.

Built-in registrations
-----------------------
- ``"simulated"`` / ``"sim"`` — :class:`~src.spectrometer.simulated.SimulatedSpectrometer`
- ``"ccs200"`` — Thorlabs CCS200 (requires DLL; optional import)

Registering custom drivers
--------------------------
::

    from src.spectrometer.registry import SpectrometerRegistry
    from src.spectrometer.base import AbstractSpectrometer

    @SpectrometerRegistry.register("my_spectrometer")
    class MySpectrometer(AbstractSpectrometer):
        ...

    # Later:
    spec = SpectrometerRegistry.create("my_spectrometer", port="COM3")
"""

from __future__ import annotations

import logging
from typing import Any, Type

from src.spectrometer.base import AbstractSpectrometer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


class SpectrometerRegistry:
    """Central factory and catalogue for spectrometer drivers.

    All registered drivers are stored in the class-level dict
    ``_drivers``.  Keys are lowercase string aliases.
    """

    _drivers: dict[str, Type[AbstractSpectrometer]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, alias: str) -> Any:
        """Class decorator that registers a driver under *alias*.

        Parameters
        ----------
        alias :
            Case-insensitive string key used when calling
            :meth:`create`.

        Returns
        -------
        Callable
            The decorator (passes the class through unchanged).

        Example
        -------
        ::

            @SpectrometerRegistry.register("usb2000")
            class USB2000Driver(AbstractSpectrometer):
                ...
        """
        def decorator(driver_cls: Type[AbstractSpectrometer]) -> Type[AbstractSpectrometer]:
            key = alias.lower()
            if key in cls._drivers:
                log.warning(
                    "SpectrometerRegistry: overwriting existing driver for alias %r",
                    key,
                )
            cls._drivers[key] = driver_cls
            log.debug("SpectrometerRegistry: registered %r → %s", key, driver_cls.__name__)
            return driver_cls
        return decorator

    @classmethod
    def register_driver(
        cls,
        alias: str,
        driver_cls: Type[AbstractSpectrometer],
    ) -> None:
        """Imperative alternative to the :meth:`register` decorator."""
        cls._drivers[alias.lower()] = driver_cls

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @classmethod
    def available(cls) -> list[str]:
        """Return all registered driver aliases, sorted."""
        _ensure_builtins_registered()
        return sorted(cls._drivers.keys())

    @classmethod
    def discover(cls) -> dict[str, dict[str, str]]:
        """Probe each registered driver and report which are reachable.

        For hardware drivers this calls ``open()`` / ``close()`` in a
        try/except — unreachable hardware is noted as unavailable.
        For the simulated driver it always reports available.

        Returns
        -------
        dict
            ``{alias: {"status": "available" | "unavailable", "reason": ...}}``
        """
        _ensure_builtins_registered()
        report: dict[str, dict[str, str]] = {}
        for alias, driver_cls in cls._drivers.items():
            try:
                inst = driver_cls()
                inst.open()
                inst.close()
                report[alias] = {"status": "available", "reason": "OK"}
                log.info("SpectrometerRegistry.discover: %r is available", alias)
            except Exception as exc:
                report[alias] = {"status": "unavailable", "reason": str(exc)}
                log.debug("SpectrometerRegistry.discover: %r unavailable — %s", alias, exc)
        return report

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, alias: str, **kwargs: Any) -> AbstractSpectrometer:
        """Instantiate a registered driver by alias.

        Parameters
        ----------
        alias :
            Driver alias (case-insensitive).  Use :meth:`available` to
            list registered aliases.
        **kwargs :
            Passed verbatim to the driver constructor.

        Returns
        -------
        AbstractSpectrometer
            An uninitialised driver instance.  Call ``open()`` (or use
            as a context manager) before acquiring spectra.

        Raises
        ------
        KeyError
            If *alias* is not registered.
        """
        _ensure_builtins_registered()
        key = alias.lower()
        if key not in cls._drivers:
            raise KeyError(
                f"Unknown spectrometer alias {alias!r}. "
                f"Registered: {cls.available()}"
            )
        driver_cls = cls._drivers[key]
        inst = driver_cls(**kwargs)
        log.debug("SpectrometerRegistry.create: %r → %s", alias, type(inst).__name__)
        return inst


# ---------------------------------------------------------------------------
# Built-in driver registration
# ---------------------------------------------------------------------------


def _ensure_builtins_registered() -> None:
    """Lazily register built-in drivers (avoids circular imports)."""
    if "simulated" in SpectrometerRegistry._drivers:
        return  # already done

    # ── Simulated spectrometer (always available) ──────────────────────
    from src.spectrometer.simulated import SimulatedSpectrometer  # noqa: F401
    SpectrometerRegistry.register_driver("simulated", SimulatedSpectrometer)
    SpectrometerRegistry.register_driver("sim", SimulatedSpectrometer)

    # ── Thorlabs CCS200 via AbstractSpectrometer adapter ──────────────
    # The adapter wraps the native DLL driver and raises ImportError at
    # construction time (not import time) if the DLL is unavailable.
    try:
        from src.spectrometer.ccs200_adapter import CCS200Adapter
        SpectrometerRegistry.register_driver("ccs200", CCS200Adapter)
        SpectrometerRegistry.register_driver("thorlabs_ccs200", CCS200Adapter)
        log.debug("SpectrometerRegistry: CCS200Adapter registered")
    except Exception:
        pass  # Silently skip on platforms without the native driver
