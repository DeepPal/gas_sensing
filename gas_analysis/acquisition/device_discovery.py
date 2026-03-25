"""Utility helpers for discovering connected CCS200 spectrometers."""

from __future__ import annotations

CCS200_USB_VENDOR = "0x1313"
CCS200_USB_PRODUCT = "0x8089"


def discover_ccs200_resources() -> tuple[list[str], list[str]]:
    """Return (resources, warnings) for connected CCS200 spectrometers.

    We attempt multiple backends (pyvisa and pylablib). Any errors are added to the
    warnings list so the caller can display them to the user. The returned resource
    strings can be fed directly into ``RealtimeAcquisitionService``.
    """

    resources: list[str] = []
    warnings: list[str] = []

    # pyvisa scan
    try:
        import pyvisa  # type: ignore

        rm = pyvisa.ResourceManager()
        # Use wildcard query to list all resources
        all_resources = rm.list_resources("?*")
        for res in all_resources:
            if "USB" in res and CCS200_USB_VENDOR in res and CCS200_USB_PRODUCT in res:
                resources.append(res)
    except Exception as exc:  # pragma: no cover - best-effort discovery
        warnings.append(f"pyvisa scan failed: {exc}")

    # pylablib scan - returns tuples like ('USB', 'M00505929')
    try:
        from pylablib.devices import Thorlabs  # type: ignore

        for dev in Thorlabs.list_devices():
            candidate = None
            if isinstance(dev, str):
                candidate = dev
            elif isinstance(dev, (list, tuple)) and dev:
                serial = dev[-1]
                candidate = f"USB0::{CCS200_USB_VENDOR}::{CCS200_USB_PRODUCT}::{serial}::RAW"
            if candidate and candidate not in resources:
                resources.append(candidate)
    except Exception as exc:  # pragma: no cover - best-effort discovery
        warnings.append(f"pylablib scan failed: {exc}")

    return resources, warnings


def format_resource_list(resources: list[str]) -> str:
    """Return a human-readable string of detected resources."""
    if not resources:
        return "No CCS200 spectrometers detected. Specify --resources manually."

    lines = ["Detected CCS200 spectrometers:"]
    for idx, res in enumerate(resources, start=1):
        lines.append(f"  {idx}. {res}")
    return "\n".join(lines)
