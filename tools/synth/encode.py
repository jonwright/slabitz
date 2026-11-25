"""Write frames as filter-32008 HDF5 (one frame per chunk, like the real files)."""

import os

import numpy as np

from . import metrics


def write_h5(frames, out_path, enc_name, dataset="data"):
    """frames: an array (1-D: one chunk; n-D: one chunk per index of axis 0), or
    a dict name -> array for several datasets in one file."""
    import h5py
    _, bb, cname, lvl = metrics.ENCODERS[enc_name]
    arrays = frames if isinstance(frames, dict) else {dataset: frames}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for name, a in arrays.items():
            arr = np.ascontiguousarray(a)
            f.create_dataset(name, data=arr, chunks=arr.shape if arr.ndim == 1 else (1,) + arr.shape[1:],
                             compression=metrics._filter(bb, arr.dtype.itemsize, cname, lvl))
    return out_path
