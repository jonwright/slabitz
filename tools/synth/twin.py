"""The generator: frames from a fitted parameter set.

A frame is made from the pieces ``fit`` stored for a key:

  1. ``lam``     a coarse map of the expected count per pixel (block means of
                 the real frames, clipped so single hot pixels do not leak in),
                 interpolated to the detector grid and scaled by a per-frame
                 factor (``scales``: real scans vary in brightness);
  2. ``k_vals``  the shape of a mean-one gamma multiplier on every pixel, one
                 per range of expected count ``k_edges`` (Poisson-gamma counts:
                 the speckle and texture of bright rings; null: plain Poisson);
  3. counts      Poisson(lam * gamma);
  4. ``hot``     a Poisson number of pixels take values drawn from the stored
                 empirical sample of the real frames' tail (Bragg peaks,
                 saturated pixels, ring speckle far above the map).  They come
                 in clusters of ``hot_cluster`` pixels on average: centres
                 with probability proportional to ``lam``, the biggest values
                 at the centres, the others a few pixels around;
  5. ``mask``    the measured masked pixels are overwritten with the masked
                 value of the real file, or with 2**n-1 or 0 (``mask_kind``).

Everything is a function of (params, npz, seed, frame index).
"""

import json
import os

import numpy as np

VERSION = 1
HOT_SIGMA = 1.5          # pixels: spread of a hot cluster (a Bragg peak and its skirt)


def load_params(key, params_dir):
    with open(os.path.join(params_dir, key + ".json")) as fh:
        p = json.load(fh)
    z = np.load(os.path.join(params_dir, key + ".npz"))
    return p, {name: z[name] for name in z.files}


def _axis_interp(n, B, m):
    """Index/weight pair for linear interpolation of block centres."""
    pos = (np.arange(n) + 0.5) / B - 0.5
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, m - 1)
    i1 = np.clip(i0 + 1, 0, m - 1)
    w = np.clip(pos - i0, 0.0, 1.0).astype(np.float32)
    return i0, i1, w


def upsample(lam, B, shape):
    """Bilinear interpolation of a block map back to pixel resolution."""
    r0, r1, wr = _axis_interp(shape[0], B, lam.shape[0])
    c0, c1, wc = _axis_interp(shape[1], B, lam.shape[1])
    lam = lam.astype(np.float32)
    tmp = lam[r0, :] * (1 - wr)[:, None] + lam[r1, :] * wr[:, None]
    return tmp[:, c0] * (1 - wc)[None, :] + tmp[:, c1] * wc[None, :]


def resize_labels(idx, shape):
    """Nearest-neighbour resize of a label map (masks for other detector sizes)."""
    r = (np.arange(shape[0]) * idx.shape[0] // shape[0])
    c = (np.arange(shape[1]) * idx.shape[1] // shape[1])
    return idx[np.ix_(r, c)]


def mask_values(params, kind, dtype):
    """Masked value per mask class for a mask kind: 'native' (as the real
    file), 'max' (2**n-1) or 'zero'."""
    dt = np.dtype(dtype)
    out = []
    for v in params["mask_values"]:
        if kind == "native":
            out.append(v)
        elif kind == "max":
            out.append(int(np.iinfo(dt).max) if dt.kind == "u" else int(np.iinfo(dt).max))
        elif kind == "zero":
            out.append(0)
        else:
            raise ValueError(kind)
    return out


def frames(params, arrays, n, seed, dtype=None, shape=None, mask_kind="native",
           lam_scale=1.0, first=0, ids=None):
    """n frames (n, rows, cols) of dtype, deterministic in (params, seed, index).

    ``shape``/``dtype``/``lam_scale`` build the extensions (another detector
    size, another dtype, a brighter or darker version of the same pattern).
    """
    dtype = np.dtype(dtype or params["dtype"])
    base_shape = tuple(params["shape"])
    shape = tuple(shape or base_shape)
    B = int(params["B"])
    lam_map = arrays["lam"].astype(np.float32)
    if shape != base_shape:
        # resample the coarse map to the new size, keeping the pattern's extent
        lam0 = upsample(lam_map, B, base_shape)
        r = (np.arange(shape[0]) * base_shape[0] // shape[0])
        c = (np.arange(shape[1]) * base_shape[1] // shape[1])
        lam0 = lam0[np.ix_(r, c)]
        base_lam = lam0
    else:
        base_lam = upsample(lam_map, B, shape)
    idx = arrays["mask_idx"]
    if shape != base_shape:
        idx = resize_labels(idx, shape)
    live = idx == 0
    base_lam = np.where(live, base_lam, 0.0).astype(np.float32) * np.float32(lam_scale)
    scales = params["scales"]
    kmap = None
    if params.get("k_vals") and any(v for v in params["k_vals"]):
        # gamma shape per pixel, by the pixel's expected count (piecewise constant)
        kv = np.array([v if v else 1e6 for v in params["k_vals"]], dtype=np.float32)
        kmap = kv[np.clip(np.searchsorted(np.array(params["k_edges"]), base_lam, side="right") - 1,
                          0, len(kv) - 1)]
    hot = arrays["hot_values"]
    nhot = float(params["hot_per_frame"]) * (shape[0] * shape[1]) / float(base_shape[0] * base_shape[1])
    mvals = mask_values(params, mask_kind, dtype)
    info = np.iinfo(dtype)
    ids = list(ids) if ids is not None else [first + i for i in range(n)]
    n = len(ids)
    out = np.empty((n,) + shape, dtype=dtype)
    cdf = None
    csize = float(params.get("hot_cluster", 1.0))
    for i in range(n):
        fi = ids[i]
        rng = np.random.default_rng([seed, fi])
        lam = base_lam * np.float32(scales[fi % len(scales)])
        if kmap is not None:
            lam = lam * rng.gamma(kmap, 1.0 / kmap).astype(np.float32)
        cnt = rng.poisson(lam)
        if len(hot) and nhot > 0:
            if cdf is None:
                hm = arrays.get("hot_map")
                if hm is not None:
                    w = np.where(live, upsample(hm, int(params["Bh"]), shape if shape == base_shape else base_shape)
                                 if shape == base_shape else
                                 upsample(hm, int(params["Bh"]), base_shape)[np.ix_(
                                     np.arange(shape[0]) * base_shape[0] // shape[0],
                                     np.arange(shape[1]) * base_shape[1] // shape[1])], 0.0)
                else:
                    w = base_lam
                cdf = np.cumsum(w.reshape(-1), dtype=np.float64)
            m = int(rng.poisson(nhot))
            vals = np.sort(hot[rng.integers(0, len(hot), m)])[::-1]
            nc = max(1, int(np.ceil(m / csize)))
            cen = np.minimum(np.searchsorted(cdf, rng.random(nc) * cdf[-1]), cdf.size - 1)
            which = np.arange(m) % nc                 # the biggest values sit at the centres
            cy, cx = np.divmod(cen[which], shape[1])
            off = np.rint(rng.normal(0.0, HOT_SIGMA, (2, m))).astype(np.int64)
            off[:, :nc] = 0
            yy = np.clip(cy + off[0], 0, shape[0] - 1)
            xx = np.clip(cx + off[1], 0, shape[1] - 1)
            cnt[yy, xx] = vals
        cnt = np.clip(cnt, info.min if info.min < 0 else 0, info.max)
        fr = cnt.astype(dtype)
        for cls, v in enumerate(mvals, start=1):
            if v < info.min or v > info.max:      # e.g. 65535 in a u8 extension, -1 unsigned
                v = info.max
            fr[idx == cls] = v
        out[i] = fr
    return out


def flat_params(shape=(2162, 2068), dtype="uint32", mean=70.0):
    """A structureless detector: constant Poisson mean, no mask, no tail.  Built
    in memory (there is no real file behind it)."""
    B = 64
    lam = np.full((-(-shape[0] // B), -(-shape[1] // B)), mean, dtype=np.float32)
    params = {"version": VERSION, "key": "flat", "dtype": dtype, "shape": list(shape), "B": B,
              "mask_values": [], "mask_source": "none", "scales": [1.0], "hot_per_frame": 0.0,
              "k_edges": [0.0, 1e12], "k_vals": [None]}
    arrays = {"lam": lam, "mask_idx": np.zeros(shape, dtype=np.uint8),
              "hot_values": np.zeros(0, dtype=np.int64)}
    return params, arrays
