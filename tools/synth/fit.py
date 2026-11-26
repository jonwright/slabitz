"""Fit a key's generator parameters from its real file.

``fit`` reads K frames spread over the scan and measures

  * the masks: pixels that hold the same masked value in every frame (for
    silicon and ceo2 from the file that writes the gaps as 2**n-1);
  * ``lam``: block means of the frames, clipped at the 99.99th percentile of
    live values, on blocks big enough to hold a few hundred counts;
  * ``scales``: the clipped live mean of each frame over the mean of all;
  * ``frame_ratio``: per frame, its block sums over the map's expectation on
    32-pixel blocks (brightness and pattern changes during a scan);
  * ``hot``: the live values above the clip, kept as an empirical sample, and
    their count in each frame (``hot_counts``);
  * ``k``: the gamma shape that makes the twin's value histogram closest to the
    real one (Jensen-Shannon distance), or none for plain Poisson.

The result is ``params/<key>.json`` (scalars, small, reviewable) and
``params/<key>.npz`` (the maps and the sample).
"""

import json
import os

import numpy as np

from . import keys as K
from . import metrics, twin

NFRAMES = 12
K_CANDIDATES = [None, 60.0, 20.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
BF = 32               # block size of the per-frame ratio maps
RATIO_PSEUDO = 50.0   # counts: a block with fewer expected counts leans to the frame's mean factor


def frame_indices(n, nfit=NFRAMES):
    return np.unique(np.linspace(0, n - 1, min(nfit, n)).round().astype(int))


def read_frames(path, ds, idx):
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        d = f[ds]
        return [np.asarray(d[int(i)]) for i in idx], d.shape[0]


def block_sum(a, B):
    H, W = a.shape
    h, w = -(-H // B) * B, -(-W // B) * B
    if (h, w) != (H, W):
        a = np.pad(a, ((0, h - H), (0, w - W)))
    return a.reshape(h // B, B, w // B, B).sum(axis=(1, 3))


def _fill(m, valid):
    """Fill invalid blocks from valid neighbours (gap lines wider than a block)."""
    m = m.copy()
    valid = valid.copy()
    for _ in range(64):
        if valid.all():
            break
        acc = np.zeros_like(m)
        cnt = np.zeros(m.shape)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sv = np.roll(valid, (dy, dx), axis=(0, 1))
            sm = np.roll(m, (dy, dx), axis=(0, 1))
            acc += np.where(sv, sm, 0.0)
            cnt += sv
        new = (~valid) & (cnt > 0)
        m[new] = acc[new] / cnt[new]
        valid = valid | new
    return m


def _box(a, r):
    """Sum of a over the (2r+1)^2 neighbourhood (edge blocks: what exists)."""
    p = np.pad(a, r)
    c = p.cumsum(0).cumsum(1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]


def _adaptive_smooth(lam, num, den, events=800.0, rmax=3):
    """A block's mean from a (2r+1)^2 neighbourhood, the smallest r for which the
    neighbourhood holds `events` counts: low-count blocks would otherwise carry
    their Poisson noise into every twin frame as extra overdispersion."""
    out = lam.copy()
    done = num >= events
    for r in range(1, rmax + 1):
        sn, sd = _box(num, r), _box(den, r)
        use = (~done) & ((sn >= events) | (r == rmax)) & (sd > 0)
        out[use] = sn[use] / sd[use]
        done = done | use
    return out


def find_masks(frames, mask_values, mask_frames=None):
    """Label map: 0 live, i>0 the i-th masked value (always held, every frame)."""
    shape = frames[0].shape
    idx = np.zeros(shape, dtype=np.uint8)
    if mask_frames is not None:
        mx = np.iinfo(mask_frames[0].dtype).max
        always = np.ones(shape, dtype=bool)
        for f in mask_frames:
            always &= (f == mx)
        idx[always] = 1
        return idx, [int(mx)]
    for cls, v in enumerate(mask_values, start=1):
        always = np.ones(shape, dtype=bool)
        for f in frames:
            always &= (f == v)
        idx[always & (idx == 0)] = cls
    return idx, list(mask_values)


def pick_block(lam_mean, nframes):
    for B in (4, 8, 16, 32, 64):
        if nframes * B * B * max(lam_mean, 1e-9) >= 400:
            return B
    return 64


def fit(key, params_dir=K.PARAMS_DIR, nfit=NFRAMES, log=print):
    kd = K.KEYS[key]
    path = K.real_path(key)
    import h5py
    with h5py.File(path, "r") as f:
        nfr = f[kd["ds"]].shape[0]
    idx_f = frame_indices(nfr, nfit)
    fr, _ = read_frames(path, kd["ds"], idx_f)
    dtype = fr[0].dtype
    shape = fr[0].shape
    mfr = None
    if kd.get("mask_file"):
        mi = frame_indices(nfr, 6)
        mpath = K.real_path(key, kd["mask_file"])
        mfr, _ = read_frames(mpath, kd["ds"], mi)
    mask_idx, mvals = find_masks(fr, kd["mask_values"], mfr)
    # what the real plain file holds in the masked pixels (the mask file only
    # locates them: it writes them as 2**n-1)
    mvals_native = list(kd["mask_values"]) if kd["mask_values"] else ([0] if mfr is not None else mvals)
    live = mask_idx == 0
    nlive = int(live.sum())
    log("%s: %d frames %s %s, masked %.4f" % (key, len(fr), shape, dtype, 1 - nlive / live.size))

    # clip: the 99.99th percentile of live values (at least 2)
    rng = np.random.default_rng(12345)
    samp = np.concatenate([f[live][rng.integers(0, nlive, 400000)] for f in fr]).astype(np.float64)
    clip = max(2.0, float(np.percentile(samp, 99.99)))
    # block means of the clipped frames
    lam_mean = float(np.mean([np.maximum(f[live & (f <= clip)], 0).mean() for f in fr]))
    B = pick_block(lam_mean, len(fr))
    num = 0
    den = 0
    nmean = []
    hot = []
    nhot = 0
    for f in fr:
        # hot pixels (above the clip) are left out of the map: the tail layer owns them
        ok = live & (f <= clip)
        num = num + block_sum(np.where(ok, np.maximum(f, 0), 0).astype(np.float64), B)
        den = den + block_sum(ok.astype(np.float64), B)
        nmean.append(float(np.maximum(f[ok], 0).mean()))
        h = f[live][f[live] > clip]
        nhot += h.size
        hot.append(h)
    valid = den > 0
    lam = np.where(valid, num / np.maximum(den, 1), 0.0)
    lam = _adaptive_smooth(lam, num, den)
    lam = _fill(lam, valid).astype(np.float32)
    scales = np.array(nmean) / np.mean(nmean)
    hot_raw = hot
    hot = np.concatenate(hot).astype(np.int64)
    if hot.size > 20000:
        hot = np.sort(rng.choice(hot, 20000, replace=False))
    params = {
        "version": twin.VERSION, "key": key, "real": os.path.basename(path), "frames_used": [int(i) for i in idx_f],
        "dtype": str(dtype), "shape": list(shape), "B": int(B), "clip": clip,
        "mask_values": [int(v) for v in mvals_native], "mask_source": kd.get("mask_file") or "always",
        "scales": [round(float(s), 6) for s in scales],
        "hot_per_frame": nhot / len(fr),
        "live_mean_clipped": lam_mean,
    }
    arrays = {"lam": lam, "mask_idx": mask_idx, "hot_values": hot}
    # each frame's pattern: its block sums over what the map expects there, on
    # BF-pixel blocks (pseudo-counts pull dim blocks to the frame's brightness
    # factor).  Scans move the pattern between frames (kevlar), which one
    # averaged map cannot hold.
    base_fit = twin.upsample(lam, B, shape)
    ratio = []
    for f, s in zip(fr, scales):
        ok = live & (f <= clip)
        n_ = block_sum(np.where(ok, np.maximum(f, 0), 0).astype(np.float64), BF)
        g_ = block_sum(np.where(ok, base_fit, 0.0), BF)
        ratio.append((n_ + RATIO_PSEUDO * s) / (g_ + RATIO_PSEUDO))
    arrays["frame_ratio"] = np.array(ratio, dtype=np.float16)
    params["Bf"] = BF
    # gamma shape per range of expected count: the twin whose value histogram of
    # the clipped live values is closest to the real one in that range
    edges = [0.0, 0.05, 0.5, 2.0, 8.0, 30.0, 100.0, 300.0, 1000.0, 1e12]
    params["k_edges"] = edges
    params["k_vals"] = [None] * (len(edges) - 1)
    base = twin.upsample(lam, B, shape)
    binid = np.clip(np.searchsorted(np.array(edges), base, side="right") - 1, 0, len(edges) - 2)
    real_bins = []
    for j in range(len(edges) - 1):
        sel = live & (binid == j)
        real_bins.append((sel, metrics.clipped_hist(fr, sel, clip), int(sel.sum())))
    best = [(9.0, None)] * (len(edges) - 1)
    real_over = [float(np.mean([(f[sel] > clip).sum() for f in fr])) for sel, _, _ in real_bins]
    ids = sorted({int(round(x)) for x in np.linspace(0, len(fr) - 1, 4)})   # dim and bright frames
    for k in K_CANDIDATES:
        t = twin.frames(dict(params, k_vals=[k] * (len(edges) - 1)), arrays, 0, 99, mask_kind="native", ids=ids)
        for j, (sel, rh, n) in enumerate(real_bins):
            if n < 20000 or edges[j] < 0.05:
                continue
            d = metrics.js_distance(rh, metrics.clipped_hist(list(t), sel, clip))
            # the noise must not make more values above the clip than the real frames hold
            over = float(np.mean([(np.asarray(x)[sel] > clip).sum() for x in t]))
            if over > 1.3 * real_over[j] + 5:
                continue
            if d < best[j][0] - 1e-4:
                best[j] = (d, k)
    params["k_vals"] = [b_[1] for b_ in best]
    # the gamma noise makes some values above the clip by itself: the tail layer
    # only adds what the real frames have beyond that
    t = twin.frames(dict(params, hot_per_frame=0.0), arrays, len(fr), 98, mask_kind="native")
    natural = float(np.mean([(np.asarray(x)[live] > clip).sum() for x in t]))
    # hot pixels per cluster: hot pixels over the 8x8 tiles that hold any
    sizes = []
    for f in fr:
        hm = live & (f > clip)
        tiles = int((block_sum(hm.astype(np.float64), 8) > 0).sum())
        if tiles:
            sizes.append(hm.sum() / tiles)
    params["hot_cluster"] = round(float(np.mean(sizes)), 3) if sizes else 1.0
    # where the hot pixels are: counts on a coarse grid, smoothed to >= 20 events
    Bh = 32
    hn = sum(block_sum((live & (f > clip)).astype(np.float64), Bh) for f in fr)
    hd = block_sum(live.astype(np.float64), Bh) * len(fr)
    hmap = _adaptive_smooth(np.where(hd > 0, hn / np.maximum(hd, 1), 0.0), hn, hd, events=20.0, rmax=6)
    params["Bh"] = Bh
    arrays["hot_map"] = hmap.astype(np.float32)
    params["hot_natural"] = natural
    params["hot_per_frame"] = max(0.0, nhot / len(fr) - natural)
    # real frames differ a lot in how many hot pixels they hold (a rotating
    # crystal lights up peaks in some frames only): keep each frame's count
    params["hot_counts"] = [int(h.size) for h in hot_raw]
    for j, (_, _, n) in enumerate(real_bins):
        log("  lam %-8g..%-8g px %-9d k=%s (JSD %.4f)" % (edges[j], edges[j + 1], n, params["k_vals"][j], best[j][0]))
    log("  B=%d clip=%.0f hot/frame real %.1f natural %.1f added %.1f" % (
        B, clip, nhot / len(fr), natural, params["hot_per_frame"]))
    os.makedirs(params_dir, exist_ok=True)
    with open(os.path.join(params_dir, key + ".json"), "w") as fh:
        json.dump(params, fh, indent=1, sort_keys=True)
        fh.write("\n")
    np.savez_compressed(os.path.join(params_dir, key + ".npz"),
                        lam=lam, mask_idx=mask_idx, hot_values=hot, hot_map=arrays["hot_map"],
                        frame_ratio=arrays["frame_ratio"])
    return params
