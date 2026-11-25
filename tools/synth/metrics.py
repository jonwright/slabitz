"""Metrics P1-P4 for detector frames (real or synthetic).

P1  value distribution of the live (not masked) pixels, by class
P3  bit-plane occupancy after bitshuffle, by block size
P4  codec level: compression ratios (hdf5plugin LZ4 and zstd, pac), the zstd
    block-type mix, and the fractions of bytes that are literals, matches at
    offset 1, matches at offset P (one bit plane) and other matches

Everything takes arrays of frames (n, rows, cols); masks are label maps (0 live).
"""

import ctypes
import os
import struct
import tempfile

import numpy as np

try:
    import h5py
    import hdf5plugin
    _HAVE_H5 = True
except Exception:  # pragma: no cover
    _HAVE_H5 = False

# name -> (kind, block_bytes, codec, level)
ENCODERS = {
    "lz4_8k": ("h5plugin", 8192, "lz4", None),
    "zstd2_256k": ("h5plugin", 262144, "zstd", 2),
    "zstd3_256k": ("h5plugin", 262144, "zstd", 3),
    "lz4_128k": ("h5plugin", 131072, "lz4", None),
    "lz4_56k": ("h5plugin", 57344, "lz4", None),
    "zstd_248k": ("h5plugin", 253952, "zstd", 3),
    "pac_lz4": ("pac", 57344, "lz4", None),
    "pac_zstd": ("pac", 253952, "zstd", None),
}

# ------------------------------------------------------------------- P1
# classes of live values (inclusive lower bound, exclusive upper bound)
CLASSES = [("0", 0, 1), ("1", 1, 2), ("2-3", 2, 4), ("4-15", 4, 16), ("16-255", 16, 256),
           ("256-4095", 256, 4096), (">=4096", 4096, 1 << 63)]
HOT_FROM = 4096                       # default: values above `clip` are the "hot" class (fit.py)

_EDGES = np.unique(np.concatenate([[0, 1, 2, 3], np.round(2.0 ** (np.arange(4, 66) / 2.0))]))


def value_hist(values):
    """Normalised histogram on 0, 1, 2, 3 and half-octave bins above."""
    v = np.maximum(np.asarray(values).reshape(-1).astype(np.float64), 0)
    c, _ = np.histogram(v, bins=np.append(_EDGES, np.inf))
    return c / max(c.sum(), 1)


def clipped_hist(frames, live, clip):
    v = np.concatenate([np.minimum(np.asarray(f)[live], clip).astype(np.float64) for f in frames])
    return value_hist(v)


def js_distance(h1, h2):
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    h1 = h1 / max(h1.sum(), 1e-300)
    h2 = h2 / max(h2.sum(), 1e-300)
    m = 0.5 * (h1 + h2)

    def kl(a):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / m[nz])))
    return float(np.sqrt(max(0.0, 0.5 * (kl(h1) + kl(h2)))))


def p1(frames, mask_idx, mask_values, clip=HOT_FROM):
    """Class fractions of live pixels, the live mean below HOT_FROM, the hot
    fraction, the masked fraction (of mask pixels holding their value) and a
    histogram of all live values."""
    live = mask_idx == 0
    v = np.concatenate([np.asarray(f)[live].reshape(-1) for f in frames]).astype(np.int64)
    out = {"n_live": int(v.size)}
    out["class_frac"] = {name: float(((v >= lo) & (v < hi)).mean()) for name, lo, hi in CLASSES}
    cold = v[(v <= clip) & (v >= 0)]
    out["mean_cold"] = float(cold.mean()) if cold.size else 0.0
    out["hot_frac"] = float((v > clip).mean())
    out["live_mean"] = float(v.mean())
    held = []
    for cls, mv in enumerate(mask_values, start=1):
        sel = mask_idx == cls
        if sel.any():
            held.append(float(np.mean([(np.asarray(f)[sel] == mv).mean() for f in frames])))
    out["mask_held"] = held
    out["masked_frac"] = float(1 - live.mean())
    out["hist"] = value_hist(v)
    return out


# ------------------------------------------------------------------- P3
def bitshuffle_planes(block):
    """Bit planes (8*itemsize, nelem/8 bytes) of one block of elements, in the
    order bitshuffle writes them: byte 0 bits 0-7, byte 1 bits 0-7, ..."""
    a = np.ascontiguousarray(block)
    bits = np.unpackbits(a.view(np.uint8).reshape(a.size, a.itemsize), axis=1, bitorder="little")
    return np.packbits(bits.T, axis=1, bitorder="little")


def p3(frames, block_bytes, max_blocks=64):
    """Per-plane all-zero fraction, equal-to-previous-plane fraction and mean
    byte entropy of the other planes, over up to max_blocks full blocks."""
    fr0 = np.asarray(frames[0])
    es = fr0.dtype.itemsize
    ne = block_bytes // es
    zero, eq, ent = [], [], []
    done = 0
    for f in frames:
        flat = np.ascontiguousarray(f).reshape(-1)
        nb = flat.size // ne
        step = max(1, nb // max(1, max_blocks // len(frames)))
        for b in range(0, nb, step):
            pl = bitshuffle_planes(flat[b * ne:(b + 1) * ne])
            z = ~pl.any(axis=1)
            zero.append(z.mean())
            e = np.zeros(len(pl), dtype=bool)
            e[1:] = (pl[1:] == pl[:-1]).all(axis=1) & ~z[1:]
            eq.append(e.mean())
            for p in pl[~z]:
                c = np.bincount(p, minlength=256) / p.size
                c = c[c > 0]
                ent.append(float(-(c * np.log2(c)).sum()))
            done += 1
            if done >= max_blocks:
                break
        if done >= max_blocks:
            break
    return {"blocks": done, "zero_frac": float(np.mean(zero)), "eq_prev_frac": float(np.mean(eq)),
            "entropy": float(np.mean(ent)) if ent else 0.0}


# ------------------------------------------------------------------- P4
def _nelems(block_bytes, elem_size):
    ne = block_bytes // elem_size
    if ne % 8:
        raise ValueError("block of %d bytes is not a multiple of 8 elements" % block_bytes)
    return ne


def _filter(block_bytes, elem_size, cname, clevel):
    ne = _nelems(block_bytes, elem_size)
    if clevel is not None:
        return hdf5plugin.Bitshuffle(nelems=ne, cname=cname, clevel=clevel)
    return hdf5plugin.Bitshuffle(nelems=ne, cname=cname)


def h5_chunks(frames, enc_name):
    """Encode each frame as one filter-32008 chunk with hdf5plugin and return the
    list of chunk byte strings."""
    _, bb, cname, lvl = ENCODERS[enc_name]
    fr = np.ascontiguousarray(frames)
    td = tempfile.mkdtemp(prefix="sbz_p4_")
    p = os.path.join(td, "r.h5")
    try:
        with h5py.File(p, "w") as f:
            ds = f.create_dataset("data", data=fr, chunks=(1,) + fr.shape[1:],
                                  compression=_filter(bb, fr.dtype.itemsize, cname, lvl))
            return [ds.id.read_direct_chunk((i,) + (0,) * (fr.ndim - 1))[1] for i in range(fr.shape[0])]
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def pac_ratio(frames, encoder, pac_lib):
    _, block_bytes, codec, _ = encoder
    es = frames.dtype.itemsize
    lib = ctypes.CDLL(pac_lib)
    lib.pac_create.argtypes = [ctypes.c_int]
    lib.pac_create.restype = ctypes.c_void_p
    lib.pac_free.argtypes = [ctypes.c_void_p]
    lib.pac_bound.argtypes = [ctypes.c_size_t] * 3
    lib.pac_bound.restype = ctypes.c_size_t
    lib.pac_compress_chunk.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                       ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                       ctypes.c_size_t]
    lib.pac_compress_chunk.restype = ctypes.c_size_t
    enc = lib.pac_create(2 if codec == "zstd" else 1)
    if not enc:
        raise RuntimeError("pac_create failed")
    try:
        total_raw = total_c = 0
        for f in frames:
            raw = np.ascontiguousarray(f)
            bound = lib.pac_bound(raw.nbytes, es, block_bytes)
            dst = ctypes.create_string_buffer(bound or 1)
            c = lib.pac_compress_chunk(enc, dst, bound, raw.ctypes.data, raw.nbytes, es, block_bytes)
            if c == 0:
                return float("nan")
            total_raw += raw.nbytes
            total_c += c
        return total_raw / total_c
    finally:
        lib.pac_free(enc)


def compression_ratio(frames, enc_name, pac_lib=None):
    enc = ENCODERS[enc_name]
    if enc[0] == "h5plugin":
        ch = h5_chunks(frames, enc_name)
        return np.ascontiguousarray(frames).nbytes / sum(len(c) for c in ch)
    if pac_lib is None or not os.path.exists(pac_lib):
        raise RuntimeError("pac library not found: %s" % pac_lib)
    return pac_ratio(np.ascontiguousarray(frames), enc, pac_lib)


def chunk_blocks(chunk, elem_size):
    """Split a filter-32008 chunk (8-byte size, 4-byte block bytes, then
    4-byte big-endian length + payload per block, then up to 7 raw tail
    elements) into the block payloads."""
    total, = struct.unpack(">Q", chunk[:8])
    bs, = struct.unpack(">I", chunk[8:12])
    be = bs // elem_size
    nel = total // elem_size
    nblocks = -(-(nel - nel % 8) // be) if nel >= 8 else 0
    pos = 12
    out = []
    while pos < len(chunk) and len(out) < nblocks:
        n, = struct.unpack(">I", chunk[pos:pos + 4])
        out.append(chunk[pos + 4:pos + 4 + n])
        pos += 4 + n
    return total, bs, out


def seq_fractions(frames_zst, plane_bytes, expect_len):
    """Byte fractions from the educational-decoder dump of zstd frames: literal
    bytes, matches at offset 1, at offset plane_bytes, other matches."""
    from tools.edu_oracle import run_oracle
    lit = m1 = mp = mo = 0
    nseq = 0
    for z, nbytes in zip(frames_zst, expect_len):
        d = run_oracle(z)
        seqs = [s for v in d.get("SEQ", []) for s in [v]]
        offs = [v[0] for v in d.get("OFFS", [])]
        ml = []
        for v in seqs:
            ml.extend(v[1::3])
        ll = []
        for v in seqs:
            ll.extend(v[0::3])
        nseq += len(ml)
        mb = sum(ml)
        for m, o in zip(ml, offs):
            if o == 1:
                m1 += m
            elif o == plane_bytes:
                mp += m
            else:
                mo += m
        lit += nbytes - mb
    tot = max(lit + m1 + mp + mo, 1)
    return {"literal": lit / tot, "off1": m1 / tot, "offP": mp / tot, "other": mo / tot,
            "sequences": nseq}


def zstd_block_mix_frame(z):
    """Count block types in one zstd frame (see zstd_block_mix for several)."""
    from . import zstdlib
    return zstdlib.block_mix(z)


def p4(frames, enc_name, pac_lib=None, nblocks=4):
    """Ratio for the encoder; for the zstd encoders also the zstd block-type
    mix and the byte-class fractions of the first nblocks blocks of frame 0."""
    out = {"ratio": compression_ratio(frames, enc_name, pac_lib)}
    enc = ENCODERS[enc_name]
    if enc[0] == "h5plugin" and enc[2] == "zstd":
        ch = h5_chunks(np.ascontiguousarray(frames[:2]), enc_name)
        total, bs, blocks = chunk_blocks(ch[0], frames.dtype.itemsize)
        out["block_mix"] = _mix_over(blocks)
        out["seq"] = seq_fractions(blocks[:nblocks], bs // (8 * frames.dtype.itemsize), [bs] * nblocks)
    return out


def _mix_over(blocks):
    from . import zstdlib
    mix = {"raw": 0, "rle": 0, "compressed": 0}
    for z in blocks:
        m = zstdlib.block_mix(z)
        for k in mix:
            mix[k] += m[k]
    return mix
