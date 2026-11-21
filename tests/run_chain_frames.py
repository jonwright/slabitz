#!/usr/bin/env python3
"""P2/P3 prep: sbz_dctx_decompress_frames() on real bitshuffle-zstd HDF5 chunks.

A bitshuffle-zstd chunk is [BE u64 total bytes][BE u32 block bytes] then, per block,
[BE u32 c][c bytes = one zstd frame], then a raw tail (see ROADMAP P3). The chunks
here are written by hdf5plugin.Bitshuffle(cname="zstd") and read back raw with
h5py's read_direct_chunk. The test walks the block records on the host, passes the
frame spans to the C chain, and checks UnCompressedFramesData against libzstd
decoding each frame (the still bit-transposed blocks, back to back). The
unshuffle itself is P3.

Negative cases: a wrong frame size and a span outside the buffer must give
SBZ_ERR_CORRUPT / SBZ_ERR_ARGS, not a crash or a hang.

Usage:  python3 tests/run_chain_frames.py      (builds with `make` first)
"""
import ctypes as ct
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))  # e.g. build/sm_35
TIMEOUT = 120
SBZ_ERR_ARGS, SBZ_ERR_CORRUPT = -3, -5


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                              ct.POINTER(Span), ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]
    return lib


def walk_chunk(chunk, elem):
    """Host walk of the block records -> (spans, raw tail bytes, total bytes, block bytes)."""
    total, bbytes = struct.unpack_from(">QI", chunk, 0)
    S, B = total // elem, bbytes // elem
    sizes = [B] * (S // B)
    last = (S % B) - (S % B) % 8
    if last:
        sizes.append(last)
    pos, spans = 12, []
    for _ in sizes:
        c = struct.unpack_from(">I", chunk, pos)[0]
        spans.append((pos + 4, c))
        pos += 4 + c
    tail = chunk[pos:pos + (S % 8) * elem]
    return spans, tail, total, bbytes, sizes


def make_chunks():
    """Yields (name, chunk bytes, elem size) from hdf5plugin-written datasets."""
    import h5py
    import hdf5plugin
    import numpy as np

    rng = np.random.default_rng(7)
    specs = [  # name, dtype, n elements, nelems per block (0 = bitshuffle default)
        ("u16-default", np.uint16, 300000, 0),
        ("u32-default", np.uint32, 200003, 0),        # S % 8 != 0 -> raw tail
        ("u16-248k", np.uint16, 400000, 248 * 1024 // 2),
        ("u8-248k-odd", np.uint8, 600011, 248 * 1024),  # last block S%B%32 != 0
        ("u32-248k", np.uint32, 150000, 248 * 1024 // 4),
    ]
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        fn = os.path.join(td, "c.h5")
        with h5py.File(fn, "w") as f:
            for name, dt, n, nel in specs:
                # detector-like: mostly small counts, some structure
                a = (rng.poisson(3, n) + (np.arange(n) % 97 == 0) * 500).astype(dt)
                f.create_dataset(name, data=a, chunks=(n,),
                                 **hdf5plugin.Bitshuffle(nelems=nel, cname="zstd"))
        with h5py.File(fn, "r") as f:
            for name, dt, n, nel in specs:
                _, chunk = f[name].id.read_direct_chunk((0,))
                yield name, bytes(chunk), np.dtype(dt).itemsize


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:n.value]


def run_one(name):
    import zstandard as zstd

    chunk, elem = next((c, e) for n, c, e in make_chunks() if n == name)
    spans, tail, total, bbytes, sizes = walk_chunk(chunk, elem)
    want = b"".join(zstd.ZstdDecompressor().decompress(chunk[o:o + c]) for o, c in spans)
    assert len(want) == sum(sizes) * elem
    arr = (Span * len(spans))(*[Span(o, c) for o, c in spans])

    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        print("FAIL %s: create: %s" % (name, err.value.decode()))
        return 1
    try:
        rc = lib.sbz_dctx_decompress_frames(dc, chunk, len(chunk), arr, len(spans))
        if rc != 0:
            print("FAIL %s: rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
            return 1
        got = read_buf(lib, dc, "UnCompressedFramesData")[:len(want)]
        if got != want:
            i = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), len(got))
            print("FAIL %s: mismatch at byte %d (got %d want %d bytes)" % (name, i, len(got),
                                                                            len(want)))
            return 1
        # negative cases on the same context (it must stay usable)
        bad = (Span * len(spans))(*[Span(o, c) for o, c in spans])
        bad[0].size -= 1
        rc1 = lib.sbz_dctx_decompress_frames(dc, chunk, len(chunk), bad, len(spans))
        bad[0].size = len(chunk)
        rc2 = lib.sbz_dctx_decompress_frames(dc, chunk, len(chunk), bad, len(spans))
        rc3 = lib.sbz_dctx_decompress_frames(dc, chunk, len(chunk), arr, len(spans))
        if (rc1, rc2, rc3) != (SBZ_ERR_CORRUPT, SBZ_ERR_ARGS, 0):
            print("FAIL %s: negative cases rc=%r, want (%d, %d, 0)" % (name, (rc1, rc2, rc3),
                                                                     SBZ_ERR_CORRUPT, SBZ_ERR_ARGS))
            return 1
        print("ok   %s (e=%d block=%d B, frames=%d, last block %d elems, tail %d B, out %d B)"
              % (name, elem, bbytes, len(spans), sizes[-1], len(tail), len(want)))
        return 0
    finally:
        lib.sbz_dctx_destroy(dc)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        return run_one(sys.argv[2])
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    bad = 0
    for name, _, _ in make_chunks():
        try:
            r = subprocess.run([sys.executable, __file__, "--one", name], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL %s: timeout (%ds) -- GPU hang?" % (name, TIMEOUT))
            bad += 1
    print("%s: %d case(s) failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
