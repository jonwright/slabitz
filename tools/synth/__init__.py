"""T00 synthetic-data generator (Python package, numpy + h5py + hdf5plugin)."""

from . import cli, encode, fit, geometry, metrics, models  # noqa: F401

__all__ = ["cli", "encode", "fit", "geometry", "metrics", "models"]
