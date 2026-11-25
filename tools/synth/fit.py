"""Estimate generator model parameters from a real file.

fit.py reads a few frames of a real file, measures P1 (value class fractions,
mean, variance) and the gap line pattern (columns/rows that are always masked
across the sampled frames), and writes a small, reviewable JSON parameter file
that ``generate`` then feeds to the model.

The parameter file records: the model name, the dtype, the detector geometry,
the P1 targets the twin must reproduce, and the gap pattern.  The model's own
free parameters (background mean, spot density, ring positions, ...) are kept
small; the generator calibrates the overall intensity scale to the measured
mean in ``generate``, so clicking "fit then generate" reproduces the dominant
P1 statistics of the real file.
"""

import json
import os

import numpy as np

from . import geometry
from . import metrics


def read_frames(path, dsname, n=8):
    """Read up to n frames from a real HDF5 file (channel of the filter)."""
    import h5py  # noqa: E402
    import hdf5plugin  # noqa: F401,E402
    with h5py.File(path, "r") as f:
        ds = f[dsname]
        nn = min(n, ds.shape[0])
        return [np.asarray(ds[i]) for i in range(nn)]


def detect_gaps(frames, dtype):
    """Columns/rows that are always zero across frames = masked gap lines."""
    arrs = [np.asarray(fr) for fr in frames]
    if arrs[0].dtype.kind == "i" and arrs[0].min() < 0:
        # Pilatus: gaps are the negative pixels; handled via neg_mask instead
        return [], []
    zero_pat = np.ones(arrs[0].shape, dtype=bool)
    for fr in arrs:
        zero_pat = zero_pat & (np.asarray(fr) == 0)
    cols = np.where(zero_pat.all(axis=0))[0]
    rows = np.where(zero_pat.all(axis=1))[0]
    return list(cols), list(rows)


def fit(key_data, out_path, n=8):
    """key_data: dict with real_path, dsname, dtype, model, geometry, gap_kind.
    Writes params JSON.  Returns the dict written."""
    frames = read_frames(key_data["real_path"], key_data["dsname"], n)
    dtype = key_data["dtype"]
    first = np.asarray(frames[0])
    shape = first.shape
    gv = geometry.gap_value(key_data.get("gap_kind", "zero"), dtype)
    p1 = metrics.compute_P1(np.stack([np.asarray(fr) for fr in frames]), dtype, masked_value=gv)
    cols, rows = detect_gaps(frames, dtype)
    cols = [int(c) for c in cols]
    rows = [int(r) for r in rows]
    params = {
        "version": 1,
        "model": key_data["model"],
        "dtype": dtype,
        "geometry": key_data["geom"],
        "shape": list(shape),
        "gap_kind": key_data.get("gap_kind", "zero"),
        "gap_cols": cols,
        "gap_rows": rows,
        "neg_mask": key_data.get("neg_mask", 0.0),
        "targets": {
            "mean": p1["mean"],
            "var": p1["var"],
            "nonzero_frac": p1["nonzero_frac"],
            "masked_frac": p1["masked_frac"],
            "hist": p1["hist"].tolist(),
        },
        "parameters": {
            # small, tunable model knobs; the intensity scale is calibrated in
            # generate to hit the target mean.
            "field": key_data.get("field", [0.1, 0.1, 0.1]),
            "n_spots": key_data.get("n_spots", 2200),
            "spot_sigma": key_data.get("spot_sigma", 4.0),
            "alpha": key_data.get("alpha", 2.2),
            "intensity": key_data.get("intensity", 9.0),
            "max_intensity": key_data.get("max_intensity", 30.0),
            "n_rings": key_data.get("n_rings", 6),
            "radii": key_data.get("radii", []),
            "widths": key_data.get("widths", []),
            "heights": key_data.get("heights", []),
            "az": key_data.get("az", 0.0),
            "cx": key_data.get("cx", 0.0),
            "cy": key_data.get("cy", 0.0),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(params, fh, indent=2, sort_keys=True)
    return params
