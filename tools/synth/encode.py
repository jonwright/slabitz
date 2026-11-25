"""Write synthetic frames as filter-32008 HDF5 in the encodings the real files use.

The encoding is the bitshuffle HDF5 chunk format, written through hdf5plugin so
the chunk layout (total, block bytes, per-block frames) matches what the
decoders read.  ``encodings`` gives, for each wanted variant, an
``(encoder_name, gap_kind)`` pair; ``encoder_name`` is one of metrics.ENCODERS.

Each frame becomes one HDF5 chunk (chunk shape = (1,) + frame_shape), exactly
like the real files.
"""

import os

import numpy as np

from . import metrics
from . import geometry  # noqa: F401

try:
    import h5py
    import hdf5plugin
    _HAVE_H5 = True
except Exception:  # pragma: no cover
    _HAVE_H5 = False


def _mask_frames(frames, gap_kind, dtype, gap_cols, gap_rows, gap_value=None, neg_mask=0):
    """Return a copy of `frames` with the gap lines set to the masked value."""
    out = frames.copy()
    dt = np.dtype(dtype)
    if gap_value is None:
        gap_value = geometry.gap_value(gap_kind, dtype)
    if gap_rows:
        out[..., list(gap_rows), :] = gap_value
    if gap_cols:
        out[..., :, list(gap_cols)] = gap_value
    return out


def write_h5(frames, dtype, out_path, encoder_name, gap_kind="zero", gap_cols=(), gap_rows=(),
             gap_value=None):
    """Write `frames` (n, rows, cols) to `out_path` with the given encoder,
    applying the gap mask.  Returns the output path."""
    if not _HAVE_H5:
        raise RuntimeError("hdf5plugin not available")
    enc = metrics.ENCODERS[encoder_name]
    kind, block_bytes, cname, clevel = enc
    elem_size = np.dtype(dtype).itemsize
    nelems = metrics._nelems_for(block_bytes, elem_size)
    bp = hdf5plugin.Bitshuffle(nelems=nelems, cname=cname, clevel=clevel) if clevel is not None \
        else hdf5plugin.Bitshuffle(nelems=nelems, cname=cname)
    arr = np.ascontiguousarray(frames, dtype)
    arr = _mask_frames(arr, gap_kind, dtype, gap_cols, gap_rows, gap_value)
    chunk = (1,) + arr.shape[1:]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=arr, chunks=chunk, compression=bp)
    return out_path
