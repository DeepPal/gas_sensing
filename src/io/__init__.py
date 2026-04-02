"""
src.io
======
I/O utilities for the SpectraAgent — Spectrometer-Based Sensing Platform.

Modules
-------
hdf5
    HDF5 session archiving using h5py.
    Writes complete measurement sessions (frames, calibration, results) to
    a self-describing HDF5 file that is readable by Thorlabs ThorSpectra,
    Origin, MATLAB, and Python h5py.
universal_loader
    Sensor-agnostic spectral dataset loader.  Handles long-format CSV,
    wide-format CSV, HDF5, and SpectraAgent session directories.
"""

from src.io.hdf5 import (
    ArchiveReader,
    ArchiveWriter,
    open_archive_reader,
    open_archive_writer,
)
from src.io.universal_loader import (
    SpectralDataset,
    load_dataset,
    load_session_csv,
    list_sessions,
    load_timeseries_features,
    merge_datasets,
)

__all__ = [
    "ArchiveWriter",
    "ArchiveReader",
    "open_archive_writer",
    "open_archive_reader",
    "SpectralDataset",
    "load_dataset",
    "load_session_csv",
    "list_sessions",
    "load_timeseries_features",
    "merge_datasets",
]
