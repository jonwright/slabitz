#!/usr/bin/env python3
"""Generate the crunch reference data: bitshuffle-zstd chunks + expected bytes.

The crunch check script (check_bshuf.py) is deliberately dependency-free
(ctypes only), so the reference data has to be baked here, on a machine with
hdf5plugin/bitshuffle/zstandard/numpy. Each case is one bitshuffle-zstd chunk:
its raw bytes, the expected untransposed bytes (the original array), and the
element size. The cases cover a broad ZSTD surface so the on-card decode is
exercised across more than the single hdf5plugin default:

  baseline (written with hdf5plugin.Bitshuffle(cname="zstd")): real filter output
  z-* (built here): one chunk per zstd LEVEL x data characterisation, plus a
    checksum-bearing frame and a multi-block (large) frame. Levels select
    different literal/sequence/FSE-table strategies; content characterisation
    selects raw/RLE/compressed blocks; a 256k+ frame is multi-block in zstd.

`--report` decodes every case here (needs the GPU + built kernels + lib) and
appends the per-case decode counters (block types, Huffman, sequences) to
data/features.tsv, so the README can say exactly what each case exercises.

Writes, per case, into --data:
  <name>.chunk.bin   the raw chunk bytes
  <name>.expected.bin the expected decoded bytes
  <name>.elem        the element size (one integer)

Usage:
  python3 tools/gen_crunch_data.py --data DIR [--report]
"""
import argparse
import ctypes as ct
import os
import pathlib
import struct

import bitshuffle
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import zstandard as zstd


# ------------------------------------------------------------ data generators
def det(n, seed, dt):
    rng = np.random.default_rng(seed)
    return (rng.poisson(3, n) + (np.arange(n) % 97 == 0) * 500).astype(dt)


def text(n, seed, dt):
    """Compressible: printable ASCII from a small vocabulary."""
    rng = np.random.default_rng(seed)
    alpha = np.frombuffer(b"the quick brown fox jumps over the lazy dog ", np.uint8)
    a = alpha[rng.integers(0, alpha.size, n)]
    return a.astype(dt)


def noise(n, seed, dt):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, n, dtype=np.uint8).astype(dt)


def const(n, seed, dt):
    return np.full(n, 0x0D, dtype=dt)


def mixed(n, seed, dt):
    """cmp (text) + raw (noise) + RLE (constant) blocks in one frame."""
    rng = np.random.default_rng(seed)
    third = n // 3
    a = np.concatenate([text(third, seed, dt), noise(third, seed + 1, dt),
                        np.full(n - 2 * third, 0x0D, dtype=dt)])
    return a.astype(dt)


def manual_chunk(a, B, level, checksum):
    """Build a bitshuffle-zstd chunk for array `a` (block = B elements) at a
    chosen zstd level / checksum. Each block is one zstd frame of the block's
    bit-transposed bytes; the (S%8) raw tail elements follow untransposed."""
    e = a.dtype.itemsize
    n = a.size
    S = n
    bs = bitshuffle.bitshuffle(a, B).view(np.uint8).tobytes()  # bit-transposed blocks
    nfull = S // B
    last = S - nfull * B
    last2 = last - last % 8
    out = struct.pack(">QI", n * e, B * e)
    pos = 0
    comp = zstd.ZstdCompressor(level=level, write_content_size=True,
                               write_checksum=checksum)
    for _ in range(nfull):
        fr = comp.compress(bs[pos:pos + B * e])
        pos += B * e
        out += struct.pack(">I", len(fr)) + fr
    if last2:
        fr = comp.compress(bs[pos:pos + last2 * e])
        pos += last2 * e
        out += struct.pack(">I", len(fr)) + fr
    return out + a[nfull * B + last2:].view(np.uint8).tobytes()


# ------------------------------------------------------------ the cases
# baseline via hdf5plugin (real filter output): (name, dtype, n, nelems)
BASELINE = [
    ("u8-256k", "uint8", 700000, 256 * 1024),
    ("u16-256k", "uint16", 400000, 256 * 1024 // 2),
    ("u32-256k", "uint32", 300000, 256 * 1024 // 4),
    ("u64-256k", "uint64", 100000, 256 * 1024 // 8),        # e=8: no fast kernel
    ("u32-256k-partial", "uint32", 132075, 256 * 1024 // 4),  # fast full + slow last + tail
    ("u32-tail3", "uint32", 3, 0),                          # all raw tail
]

# zstd sweep: (name, dtype, n, nelems, data_fn, level, checksum)
SWEEP = [
    ("z-l1-text", "uint16", 500000, 262144, text, 1, False),
    ("z-l3-text", "uint16", 500000, 262144, text, 3, False),
    ("z-l9-text", "uint16", 500000, 262144, text, 9, False),
    ("z-l19-text", "uint16", 500000, 262144, text, 19, False),
    ("z-l22-text", "uint16", 500000, 262144, text, 22, False),
    ("z-l9-multiblock", "uint16", 1048576, 1048576, text, 9, False),  # 1 MB frame
    ("z-l9-checksum", "uint16", 500000, 262144, text, 9, True),        # checksum in frame
    ("z-l3-noise", "uint16", 400000, 262144, noise, 3, False),         # raw blocks
    ("z-l3-const", "uint16", 400000, 262144, const, 3, False),         # RLE blocks
    ("z-l9-mixed", "uint16", 700000, 262144, mixed, 9, False),         # cmp+raw+rle
    ("z-l9-detector", "uint32", 300000, 262144, det, 9, False),        # pac-like
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", action="store_true",
                    help="decode each case here (GPU) and write data/features.tsv")
    a = ap.parse_args(argv)
    data = pathlib.Path(a.data)
    data.mkdir(parents=True, exist_ok=True)
    scratch = os.path.join(os.path.dirname(str(data)), "_crunch.h5")

    # baseline via hdf5plugin (real filter output)
    with h5py.File(scratch, "w") as f:
        for name, dt, n, nel in BASELINE:
            arr = det(n, 1000 + hash(name) % 10000, dt)
            f.create_dataset(name, data=arr, chunks=(n,),
                             **hdf5plugin.Bitshuffle(nelems=nel, cname="zstd"))
    with h5py.File(scratch, "r") as f:
        for name, dt, n, nel in BASELINE:
            ds = f[name]
            e = np.dtype(dt).itemsize
            _, chunk = ds.id.read_direct_chunk((0,))
            (data / (name + ".chunk.bin")).write_bytes(bytes(chunk))
            (data / (name + ".expected.bin")).write_bytes(ds[()].tobytes())
            (data / (name + ".elem")).write_text(str(e))

    # zstd sweep (manual construction: control level/checksum)
    for name, dt, n, nel, fn, level, ck in SWEEP:
        e = np.dtype(dt).itemsize
        arr = fn(n, 2000 + hash(name) % 10000, dt)
        chunk = manual_chunk(arr, nel, level, ck)
        (data / (name + ".chunk.bin")).write_bytes(chunk)
        (data / (name + ".expected.bin")).write_bytes(arr.tobytes())
        (data / (name + ".elem")).write_text(str(e))

    print("wrote %d cases to %s" % (len(BASELINE) + len(SWEEP), data))

    if a.report:
        names = [n for n, *_ in BASELINE] + [n for n, *_ in SWEEP]
        report(data, names)
    return 0


# ------------------------------------------------------------ optional report
class Counters(ct.Structure):
    _fields_ = [(n, ct.c_uint32) for n in (
        "FseHufW FseLLen FseOffs FseMLen DecompressLiteralsGroups HUF_WgtStreams "
        "Seq_Streams_DecodedItems HUF_Streams_DecodedBytes Seq_Streams HUF_Streams HufLit "
        "RAW_Streams RLE_Streams Blocks_RAW Blocks_RLE Blocks_CMP BlocksBytes_RAW "
        "BlocksBytes_RLE Frames Frames_UncompressedByteSize "
        "Frames_AcceptedUncompressedByteSize Frames_RequestTooLarge").split()]


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def report(data, names):
    LIB = str(pathlib.Path(__file__).resolve().parent.parent / "build" / "libslabitz.so")
    kdir = str(pathlib.Path(__file__).resolve().parent.parent / "build")
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                             ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_counters.argtypes = [ct.c_void_p, ct.POINTER(Counters)]
    err = ct.create_string_buffer(512)
    lines = ["name\te\traw\trle\tcmp\thufStreams\tseqStreams\tfse\tcmpBlocks_tot"]
    for name in names:
        chunk = (data / (name + ".chunk.bin")).read_bytes()
        elem = int((data / (name + ".elem")).read_text())
        want = (data / (name + ".expected.bin")).read_bytes()
        span = (Span * 1)(Span(0, len(chunk)))
        # fresh context per case: the chain is not safe to reuse across calls.
        dc = lib.sbz_dctx_create_cuda(0, kdir.encode(), err, len(err))
        if not dc:
            print("  report skipped: %s" % err.value.decode())
            return
        rc = lib.sbz_dctx_bshuf_decompress(dc, chunk, len(chunk), span, 1, elem)
        if rc != 0:
            print("  %-18s FAIL rc=%d %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
            lib.sbz_dctx_destroy(dc)
            continue
        c = Counters()
        lib.sbz_dctx_counters(dc, ct.byref(c))
        lib.sbz_dctx_destroy(dc)
        lines.append("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d" % (
            name, elem, c.Blocks_RAW, c.Blocks_RLE, c.Blocks_CMP, c.HUF_Streams,
            c.Seq_Streams, c.FseHufW, c.Blocks_CMP))
    (data / "features.tsv").write_text("\n".join(lines) + "\n")
    print("  wrote %s with the decode counters" % (data / "features.tsv"))


if __name__ == "__main__":
    import sys
    sys.exit(main())
