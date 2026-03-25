"""
src.batch
=========
Batch analysis pipeline for Au-MIP LSPR gas sensing.

Modules
-------
data_loader
    Discover and load raw spectrum CSV files from Joy_Data directory layout.
    Key functions: ``scan_experiment_root``, ``read_spectrum_csv``,
    ``load_last_n_frames``.
aggregation
    Temporal-gating and weighted averaging to produce canonical spectra.
    Key functions: ``find_stable_block``, ``average_stable_block``,
    ``build_canonical_from_scan``.
"""

from src.batch.aggregation import (
    average_stable_block,
    build_canonical_from_scan,
    find_stable_block,
    select_canonical_per_concentration,
)
from src.batch.data_loader import (
    ExperimentScan,
    load_frames,
    load_last_n_frames,
    read_spectrum_csv,
    scan_experiment_root,
    sort_frame_paths,
)

__all__ = [
    # data_loader
    "ExperimentScan",
    "read_spectrum_csv",
    "sort_frame_paths",
    "scan_experiment_root",
    "load_frames",
    "load_last_n_frames",
    # aggregation
    "find_stable_block",
    "average_stable_block",
    "select_canonical_per_concentration",
    "build_canonical_from_scan",
]
