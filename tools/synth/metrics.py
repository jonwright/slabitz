"""Metrics P1-P4 for synthetic/real detector data.

P1  value distribution per pixel class
P3  bit-plane occupancy (after bitshuffle), by block size
P4  codec-level statistics: compression ratios (hdf5plugin LZ4 / zstd, pac
    zstd / LZ4), zstd block-type and literals mix, sequence counts, and
    match/literal byte fractions.

The module is import-only; the CLI drives it.
"""

import ctypes
import hashlib
import os
import tempfile

import numpy as np

from . import geometry  # noqa: F401  (used for gap_value at the call site)

try:
    import h5py  # noqa: F401
    import hdf5plugin
    _HAVE_H5 = True
except Exception:  # pragma: no cover
    _HAVE_H5 = False

# ENCODERS: name -> (kind, block_bytes, codec, clevel), kind in {"h5plugin","pac"}.
# block_bytes is the raw block size in bytes; nelems = block_bytes//elem_size.
ENCODERS = {
    "lz4_8k": ("h5plugin", 8192, "lz4", None),
    "zstd2_256k": ("h5plugin", 262144, "zstd", 2),
    "zstd3_256k": ("h5plugin", 262144, "zstd", 3),
    "pac_lz4": ("pac", 57344, "lz4", None),
    "pac_zstd": ("pac", 253952, "zstd", None),
}


# ---------------------------------------------------------------- bitshuffle
def _pad_chunk(bytes_, block_elems, elem_size):
    """Split a raw block of `block_elems` elements into `8*elem_size` bit planes.

    Returns a uint8 array of shape (planes, plane_len) where plane_len =
    block_elems // 8 (bitshuffle packs 8 elements of one bit plane per byte).
    `bytes_` must be a C-contiguous block of block_elems*elem_size bytes, as a
    uint8 array with the *untransposed* (row-major element, then byte within
    element) layout.
    """
    block = bytes_  # uint8 array, len = block_elems*elem_size
    nelem = block_elems
    bpl = nelem // 8  # bytes per bit plane
    planes = np.zeros((8 * elem_size, bpl), dtype=np.uint8)
    # element j's byte b sits at block[j*elem_size + b]
    el = block.reshape(nelem, elem_size)
    for b in range(elem_size):
        col = el[:, b]  # value of byte b across all elements
        # spread bits: bit i of each element -> a plane of nelem bits
        for i in range(8):
            bit = (col >> i) & 1
            # pack 8 bits -> 1 byte (bit 0 is the least significant of each element)
            packed = np.zeros((nelem // 8,), dtype=np.uint8)
            bit8 = bit.reshape(nelem // 8, 8)
            for k in range(8):
                packed = packed | (bit8[:, k].astype(np.uint8) << k)
            planes[b * 8 + i] = packed
    return planes


def bit_plane_metrics(bytes_, block_elems, elem_size):
    """P3 for one untransposed block: all-zero plane fraction, equal-to-plane-
    above fraction, and byte entropy of non-trivial planes."""
    planes = _pad_chunk(bytes_, block_elems, elem_size)
    total = planes.shape[0]
    nz = planes.reshape(planes.shape[0], -1)
    all_zero = (nz.sum(axis=1) == 0)
    # plane equal to the plane above (full bytes equal)
    eq_above = np.zeros(total, dtype=bool)
    pm = planes.reshape(total, -1)
    eq_above[1:] = np.all(pm[1:] == pm[:-1], axis=1)
    nontrivial = ~all_zero
    ent = []
    for p in pm[nontrivial]:
        v, c = np.unique(p, return_counts=True)
        c = c.astype(np.float64) / p.size
        ent.append(float(-np.sum(c * np.log2(c))))
    return {
        "planes": int(total),
        "all_zero_frac": float(all_zero.mean()),
        "eq_above_frac": float(eq_above.mean()),
        "entropy": float(np.mean(ent)) if ent else 0.0,
        "entropy_max": float(np.log2(256)),
    }


def compute_P3(frames, elem_size, block_bytes):
    """P3 over a set of frames for a given block size.

    `frames` is a uint8 array (n, rows, cols*elem_size) of byte-serialised
    frames, or a (n, rows, cols) array of the dtype.  We serialise to bytes,
    split into blocks of block_bytes, and aggregate the plane metrics.
    """
    arr = frames
    block_elems = block_bytes // elem_size
    blocks = []
    for f in arr:
        b = np.ascontiguousarray(f).view(np.uint8).reshape(-1)
        blocks.append(b)
    allb = np.concatenate(blocks)
    nblocks = len(allb) // block_bytes
    out = {"n_blocks": int(nblocks), "all_zero": [], "eq_above": [], "entropy": []}
    for i in range(nblocks):
        seg = allb[i * block_bytes:(i + 1) * block_bytes]
        m = bit_plane_metrics(seg, block_elems, elem_size)
        out["all_zero"].append(m["all_zero_frac"])
        out["eq_above"].append(m["eq_above_frac"])
        out["entropy"].append(m["entropy"])
    if out["all_zero"]:
        out["all_zero_frac"] = float(np.mean(out["all_zero"]))
        out["eq_above_frac"] = float(np.mean(out["eq_above"]))
        out["entropy"] = float(np.mean(out["entropy"]))
    else:
        out["all_zero_frac"] = float("nan")
        out["eq_above_frac"] = float("nan")
        out["entropy"] = float("nan")
    return out


# ------------------------------------------------------------- compression
def _nelems_for(block_bytes, elem_size):
    ne = block_bytes // elem_size
    if ne % 8:
        raise ValueError("block_bytes %s not a multiple of 8*elem_size" % block_bytes)
    return ne


def h5plugin_ratio(frames, dtype, encoder):
    """Compression ratio (raw/compressed) writing `frames` through hdf5plugin."""
    if not _HAVE_H5:
        raise RuntimeError("hdf5plugin not available")
    _, block_bytes, cname, clevel = encoder
    elem_size = np.dtype(dtype).itemsize
    nelems = _nelems_for(block_bytes, elem_size)
    bp = hdf5plugin.Bitshuffle(nelems=nelems, cname=cname, clevel=clevel) if clevel is not None \
        else hdf5plugin.Bitshuffle(nelems=nelems, cname=cname)
    n = frames.shape[0]
    chunk = (1,) + frames.shape[1:]
    td = tempfile.mkdtemp(prefix="sbz_ratio_")
    p = os.path.join(td, "r.h5")
    try:
        with h5py.File(p, "w") as f:
            ds = f.create_dataset("data", data=frames, chunks=chunk, compression=bp)
            raw = int(round(frames.nbytes))
            comp = int(ds.id.get_storage_size())
        return raw / comp if comp else float("nan")
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def pac_ratio(frames, dtype, encoder, pac_lib):
    """Compression ratio through libpac (pac_compress_chunk)."""
    _, block_bytes, codec, _ = encoder
    elem_size = np.dtype(dtype).itemsize
    lib = ctypes.CDLL(pac_lib)
    lib.pac_create.argtypes = [ctypes.c_int]
    lib.pac_create.restype = ctypes.c_void_p
    lib.pac_free.argtypes = [ctypes.c_void_p]
    lib.pac_bound.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t]
    lib.pac_bound.restype = ctypes.c_size_t
    lib.pac_compress_chunk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t]
    lib.pac_compress_chunk.restype = ctypes.c_size_t
    code_id = 2 if codec == "zstd" else 1
    enc = lib.pac_create(code_id)
    if not enc:
        raise RuntimeError("pac_create failed")
    try:
        raw = np.ascontiguousarray(frames)
        src = raw.ctypes.data
        src_size = raw.nbytes
        bound = lib.pac_bound(src_size, elem_size, block_bytes)
        dst = ctypes.create_string_buffer(bound or 1)
        csz = lib.pac_compress_chunk(enc, dst, bound, src, src_size, elem_size, block_bytes)
        if csz == 0:
            return float("nan")
        return src_size / csz
    finally:
        lib.pac_free(enc)


def compression_ratio(frames, dtype, encoder, pac_lib=None):
    kind = encoder[0]
    if kind == "h5plugin":
        return h5plugin_ratio(frames, dtype, encoder)
    if kind == "pac":
        if pac_lib is None or not os.path.exists(pac_lib):
            raise RuntimeError("pac lib not available: %s" % pac_lib)
        return pac_ratio(frames, dtype, encoder, pac_lib)
    raise ValueError(kind)


# ---------------------------------------------------------------- P1 hist
_LOG_BINS = None


def _log_bins(lo, hi, n=64):
    b = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi + 1.0), n)
    return b


def value_hist(values, dtype):
    """Log-binned histogram (normalised) of a frame's values for dtype."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    v = np.asarray(values).astype(np.float64)
    if info.min < 0:  # signed: histogram the non-negative part only
        v = v[v >= 0]
    hi = max(1.0, float(info.max))
    bins = _log_bins(1.0, hi)
    c, _ = np.histogram(v, bins=bins)
    s = c.sum()
    if s == 0:
        return np.full(len(bins), 0.0)
    return c.astype(np.float64) / s


def js_distance(h1, h2):
    """Jensen-Shannon distance between two normalised histograms."""
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    h1 = h1 / (h1.sum() + 1e-12)
    h2 = h2 / (h2.sum() + 1e-12)
    m = 0.5 * (h1 + h2)
    kld = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / np.clip(b, 1e-15, None)), 0.0)))
    return float(np.sqrt(0.5 * (kld(h1, m) + kld(h2, m))))


def compute_P1(frames, dtype, masked_value=0):
    """P1 for a set of frames: class fractions, mean, and a value histogram."""
    info = np.iinfo(np.dtype(dtype))
    vals = np.asarray(frames).astype(np.int64)
    n = vals.size
    nonzero = vals != 0
    masked = vals == masked_value  # value written for gap/masked pixels
    mean = float(vals.mean())
    var = float(vals.var())
    hist = value_hist(vals, dtype)
    return {
        "nonzero_frac": float(nonzero.mean()),
        "masked_frac": float(masked.mean()),
        "mean": mean,
        "var": var,
        "hist": hist,
    }


# ------------------------------------------------------- zstd frame walker
def zstd_block_mix(data):
    """Walk a plain zstd frame and report the count and total bytes of each
    block type (0=raw, 1=RLE, 2=compressed, 3=reserved).  Returns a dict.
    Only single concatenated frames followed by optional skippable frames are
    handled; anything else raises.
    """
    import struct
    out = {"raw": 0, "rle": 0, "compressed": 0, "reserved": 0,
           "raw_bytes": 0, "rle_bytes": 0, "compressed_bytes": 0,
           "blocks": 0, "frames": 0, "bytes": len(data)}
    pos = 0
    while pos + 4 <= len(data):
        magic = struct.unpack("<I", data[pos:pos + 4])[0]
        if (magic & 0xFFFFFFF0) == 0x184D2A50:  # skippable frame
            if pos + 8 > len(data):
                break
            size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
            pos += 8 + size
            continue
        if magic != 0xFD2FB528:
            break  # not a standard frame; stop
        pos += 4
        # Frame_Header_Descriptor
        if pos >= len(data):
            break
        fhd = data[pos]; pos += 1
        fcs_flag = (fhd >> 6) & 3
        single_segment = (fhd >> 5) & 1
        dictid_flag = fhd & 3
        if not single_segment:
            if pos >= len(data):
                break
            pos += 1  # Window_Descriptor
        if dictid_flag == 1:
            pos += 1
        elif dictid_flag == 2:
            pos += 2
        elif dictid_flag == 3:
            pos += 4
        fcs_size = {0: 0, 1: 2, 2: 4, 3: 8}[fcs_flag]
        # single-segment with FCS 0 means a 1-byte FCS (small content)
        if single_segment and fcs_flag == 0:
            fcs_size = 1
        pos += fcs_size
        out["frames"] += 1
        # walk the blocks
        while pos + 3 <= len(data):
            hdr = data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16)
            last = hdr & 1
            btype = (hdr >> 1) & 3
            bsize = (hdr >> 3) & 0x1FFFFF
            pos += 3
            # the block payload
            if btype == 1:
                pos += 1  # RLE: one byte (the repeated value)
            else:
                pos += bsize
            out["blocks"] += 1
            if btype == 0:
                out["raw"] += 1; out["raw_bytes"] += bsize
            elif btype == 1:
                out["rle"] += 1; out["rle_bytes"] += bsize
            elif btype == 2:
                out["compressed"] += 1; out["compressed_bytes"] += bsize
            else:
                out["reserved"] += 1
            if last:
                break
        if last:
            # done with this frame
            break
        # could be another concatenated frame
    return out
