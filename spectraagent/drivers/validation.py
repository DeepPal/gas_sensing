"""
spectraagent.drivers.validation
================================
Extension-quality validation helpers for hardware drivers.

These checks are intentionally lightweight and deterministic so they can run
at startup or in CI to catch incompatible third-party plugins early.
"""
from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from spectraagent.drivers.base import AbstractHardwareDriver

_MIN_PIXELS = 128


def _is_1d_numeric_array(arr: Any) -> bool:
    return isinstance(arr, np.ndarray) and arr.ndim == 1 and np.issubdtype(arr.dtype, np.number)


def validate_driver_class(driver_cls: type[Any]) -> list[str]:
    """Validate a plugin class before instantiation.

    Returns a list of human-readable issues. Empty list means valid.
    """
    issues: list[str] = []
    if not inspect.isclass(driver_cls):
        return ["entry-point target is not a class"]

    if not issubclass(driver_cls, AbstractHardwareDriver):
        issues.append("driver class must subclass AbstractHardwareDriver")

    if inspect.isabstract(driver_cls):
        issues.append("driver class is abstract (missing required methods)")

    return issues


def validate_driver_instance(
    driver: AbstractHardwareDriver,
    require_live_sample: bool = False,
) -> list[str]:
    """Validate a connected driver instance contract.

    Parameters
    ----------
    driver:
        Connected driver instance.
    require_live_sample:
        If True, validates read_spectrum() shape and finite values.
    """
    issues: list[str] = []

    if not driver.is_connected:
        issues.append("driver reports is_connected=False after connect()")

    try:
        wl = driver.get_wavelengths()
    except Exception as exc:
        return [f"get_wavelengths() failed: {exc}"]

    if not _is_1d_numeric_array(wl):
        issues.append("get_wavelengths() must return a 1D numeric numpy array")
    else:
        if wl.size < _MIN_PIXELS:
            issues.append(f"wavelength axis too short ({wl.size} < {_MIN_PIXELS})")
        if not np.all(np.isfinite(wl)):
            issues.append("wavelength axis contains non-finite values")
        if np.any(np.diff(wl) <= 0):
            issues.append("wavelength axis must be strictly increasing")

    try:
        integration_ms = float(driver.get_integration_time_ms())
        if not np.isfinite(integration_ms) or integration_ms <= 0:
            issues.append("get_integration_time_ms() must return a positive finite value")
    except Exception as exc:
        issues.append(f"get_integration_time_ms() failed: {exc}")

    if require_live_sample:
        try:
            sp = driver.read_spectrum()
            if not _is_1d_numeric_array(sp):
                issues.append("read_spectrum() must return a 1D numeric numpy array")
            elif sp.shape != wl.shape:
                issues.append(
                    f"spectrum shape {sp.shape} does not match wavelengths shape {wl.shape}"
                )
            elif not np.all(np.isfinite(sp)):
                issues.append("spectrum contains non-finite values")
        except Exception as exc:
            issues.append(f"read_spectrum() failed: {exc}")

    return issues
