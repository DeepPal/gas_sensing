"""Concentration-axis transforms for calibration curve fitting.

Linearises the sensor response for robust curve fitting — e.g. log10 for
Langmuir-type adsorption where signal saturates at high concentrations.
"""
from __future__ import annotations

import numpy as np


def transform_concentrations(
    concs: np.ndarray, mode: str
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a non-linear transform to the concentration axis.

    The returned *meta* dict carries any parameters (e.g. a clipping offset)
    needed to invert the transform at inference time.

    Args:
        concs: Concentration values (ppm), 1-D array.
        mode: Transform to apply.  One of:

            - ``"linear"`` (or any unrecognised string) – pass-through
            - ``"log10"`` – base-10 logarithm with automatic positive offset
            - ``"log"`` – natural logarithm with automatic positive offset
            - ``"sqrt"`` – square-root (negative values clamped to zero)

            For log transforms the offset is ``min_positive * 0.1``,
            falling back to ``1e-3`` if all values are non-positive.

    Returns:
        ``(transformed_concs, meta)`` where *meta* contains any parameters
        required for inversion.  Currently only ``"offset"`` is populated
        (for log-family transforms).

    Example:
        >>> import numpy as np
        >>> t, meta = transform_concentrations(np.array([0.5, 1.0, 2.0]), "log10")
        >>> "offset" in meta
        True
        >>> t, meta = transform_concentrations(np.array([1.0, 2.0]), "linear")
        >>> meta
        {}
    """
    meta: dict[str, float] = {}
    mode = mode.lower()
    if mode == "log10":
        safe = np.where(concs <= 0, np.nan, concs)
        min_positive = float(np.nanmin(safe))
        offset = min_positive * 0.1 if np.isfinite(min_positive) and min_positive > 0 else 1e-3
        meta["offset"] = offset
        transformed = np.log10(np.clip(concs, offset, None))
    elif mode == "log":
        safe = np.where(concs <= 0, np.nan, concs)
        min_positive = float(np.nanmin(safe))
        offset = min_positive * 0.1 if np.isfinite(min_positive) and min_positive > 0 else 1e-3
        meta["offset"] = offset
        transformed = np.log(np.clip(concs, offset, None))
    elif mode == "sqrt":
        transformed = np.sqrt(np.clip(concs, 0.0, None))
    else:
        transformed = concs
    return transformed, meta
