"""
src.advanced
============
Re-export bridge for spectral deconvolution algorithms.

Import from this namespace so callers are decoupled from the internal
``gas_analysis.advanced.*`` layout::

    from src.advanced import fit_mcrals_from_canonical, fit_ica_from_canonical
"""

from __future__ import annotations

from gas_analysis.advanced.deconvolution_ica import (
    fit_ica_from_canonical,
    save_ica_components_overlay,
    save_ica_outputs,
)
from gas_analysis.advanced.mcr_als import (
    fit_mcrals_from_canonical,
    save_mcr_components_overlay,
    save_mcrals_outputs,
)

__all__ = [
    "fit_mcrals_from_canonical",
    "save_mcrals_outputs",
    "save_mcr_components_overlay",
    "fit_ica_from_canonical",
    "save_ica_outputs",
    "save_ica_components_overlay",
]
