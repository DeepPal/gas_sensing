"""
src.io
======
I/O utilities for the Au-MIP LSPR Gas Sensing Platform.

Modules
-------
hdf5
    HDF5 session archiving using h5py.
    Writes complete measurement sessions (frames, calibration, results) to
    a self-describing HDF5 file that is readable by Thorlabs ThorSpectra,
    Origin, MATLAB, and Python h5py.
"""

from src.io.hdf5 import (
    ArchiveReader,
    ArchiveWriter,
    open_archive_reader,
    open_archive_writer,
)

__all__ = [
    "ArchiveWriter",
    "ArchiveReader",
    "open_archive_writer",
    "open_archive_reader",
]
