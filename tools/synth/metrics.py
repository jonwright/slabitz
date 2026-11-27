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


def spread(n, k):
    """k indices spread evenly over range(n) (all of them when k >= n)."""
    if k >= n:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(np.int64))


def p3(frames, block_bytes, max_blocks=2048):
    """Per-plane all-zero fraction and equal-to-previous-plane fraction over
    the full blocks of all frames (at most max_blocks, spread evenly over
    frames and positions: a fixed stride per frame would visit the same
    detector region in every frame)."""
    fr0 = np.asarray(frames[0])
    es = fr0.dtype.itemsize
    ne = block_bytes // es
    nb = fr0.size // ne
    pick = spread(len(frames) * nb, max_blocks)
    zero, eq = [], []
    for g in pick:
        f, b = divmod(int(g), nb)
        flat = np.asarray(frames[f]).reshape(-1)
        pl = bitshuffle_planes(flat[b * ne:(b + 1) * ne])
        z = ~pl.any(axis=1)
        zero.append(z.mean())
        e = np.zeros(len(pl), dtype=bool)
        e[1:] = (pl[1:] == pl[:-1]).all(axis=1) & ~z[1:]
        eq.append(e.mean())
    return {"blocks": len(pick), "zero_frac": float(np.mean(zero)), "eq_prev_frac": float(np.mean(eq))}


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


SHORT_OFF = 16      # matches with 1 < offset < 16 overlap a 16-byte copy (the decoder's uint4 path)


def _off_class(o):
    return "fill" if o == 1 else ("short" if o < SHORT_OFF else "long")


def seq_fractions(frames_zst, plane_bytes):
    """Byte classes of the compressed blocks of zstd frames, from the
    educational-decoder dump, as fractions of the bytes those blocks decode to.

    literal: literal bytes.  zero: match bytes that copy zero bytes only (the
    all-zero runs of sparse bit planes); any earlier zero byte would serve as
    the source, and libzstd picks the offset from its repeat-offset history,
    so their offset split (zero_fill, zero_short, zero_long) is reported but
    is not a property of the data.  fill, short, long: match bytes that copy
    data, by the copy the decoder makes: offset 1, offset 2-15 (overlapping a
    16-byte copy), 16 and more.  offP: data copies at one bit-plane distance
    (part of long)."""
    from tools.edu_oracle import run_oracle
    from . import zstdlib
    acc = dict.fromkeys(("literal", "zero", "fill", "short", "long", "offP",
                         "zero_fill", "zero_short", "zero_long"), 0)
    nseq = comp = 0
    for z in frames_zst:
        d = run_oracle(z)
        out = np.frombuffer(d["_output"], np.uint8)
        seqs = iter(d.get("SEQ", []))
        offs = iter(v[0] for v in d.get("OFFS", []))
        pos = 0
        for fr in zstdlib.analyse(z):
            for b in fr.get("blocks", []):
                if b["type"] in ("raw", "rle"):
                    pos += b["size"]
                    continue
                n = b["lit_regen"]
                acc["literal"] += n
                if b["nseq"]:                     # the oracle prints no SEQ line for 0 sequences
                    s = next(seqs)
                    for ll, ml in zip(s[0::3], s[1::3]):
                        o = next(offs)
                        pos += ll
                        cls = _off_class(o)
                        if out[pos:pos + ml].any():
                            acc[cls] += ml
                            if o == plane_bytes:
                                acc["offP"] += ml
                        else:
                            acc["zero"] += ml
                            acc["zero_" + cls] += ml
                        pos += ml
                        n += ml
                    nseq += b["nseq"]
                    pos += b["lit_regen"] - sum(s[0::3])
                else:
                    pos += n
                comp += n
        if pos != len(out):
            raise RuntimeError("oracle dump walk: %d bytes, decoded %d" % (pos, len(out)))
    c = max(comp, 1)
    res = {k: v / c for k, v in acc.items()}
    res.update(seq_per_kb=1024.0 * nseq / c, blocks=len(frames_zst))
    return res


def zstd_block_mix_frame(z):
    """Count block types in one zstd frame (see zstd_block_mix for several)."""
    from . import zstdlib
    return zstdlib.block_mix(z)


def p4(frames, enc_name, pac_lib=None, nblocks=128):
    """Ratio for the encoder; for the zstd encoders also the zstd block-type
    mix (all blocks) and the byte classes of nblocks blocks spread evenly over
    all frames and positions."""
    enc = ENCODERS[enc_name]
    if not (enc[0] == "h5plugin" and enc[2] == "zstd"):
        return {"ratio": compression_ratio(frames, enc_name, pac_lib)}
    fr = np.ascontiguousarray(frames)
    ch = h5_chunks(fr, enc_name)
    out = {"ratio": fr.nbytes / sum(len(c) for c in ch)}
    blocks = []
    for c in ch:
        _, bs, bl = chunk_blocks(c, fr.dtype.itemsize)
        blocks.extend(bl)
    out["block_mix"] = _mix_over(blocks)
    pick = spread(len(blocks), nblocks)
    out["seq"] = seq_fractions([blocks[i] for i in pick], bs // (8 * fr.dtype.itemsize))
    out["seq"]["rle_raw"] = _rle_raw_frac(blocks)
    return out


def _rle_raw_frac(blocks):
    """Bytes in RLE and raw zstd blocks over all bytes, over every block (in
    bitshuffled detector data mostly the all-zero high planes)."""
    from . import zstdlib
    rr = tot = 0
    for z in blocks:
        for fr in zstdlib.analyse(z):
            if fr.get("skippable"):
                continue
            tot += fr["fcs"]                     # hdf5plugin frames carry their content size
            rr += sum(b["size"] for b in fr["blocks"] if b["type"] in ("raw", "rle"))
    return rr / max(tot, 1)


def _mix_over(blocks):
    from . import zstdlib
    mix = {"raw": 0, "rle": 0, "compressed": 0}
    for z in blocks:
        m = zstdlib.block_mix(z)
        for k in mix:
            mix[k] += m[k]
    return mix
