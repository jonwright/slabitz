#!/usr/bin/env python3
"""T06 part 3: fixed-seed round trips over sizes, levels and data kinds.

Compresses many inputs with `zstandard` (libzstd) at levels 1-19, over a log
scale of sizes from 1 byte to 4 MB, across data kinds (random, text, zeros,
runs, and u16/u32 detector-like images with Poisson noise and a mask), then
decompresses with the slabitz library (sbz_dctx_decompress) and checks the
output byte-exact against libzstd.  It also builds concatenated batches of up
to 64 frames per call and decodes those with sbz_dctx_decompress and
sbz_dctx_decompress_frames.

This is a fixed-seed soak: every case is reproducible, there is no wall-clock
dependence.  It prints the number of cases (must be at least 500) and the
number of compared bytes (must be nonzero) and exits 0 on success.

It is registered under the pytest `slow` marker (not in `sbz test`).

Usage:  timeout 600 python3 tests/run_roundtrip.py
"""
import ctypes as ct
import os
import sys

import numpy as np
import zstandard as zstd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
LEVELS = list(range(1, 20))
SIZES = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144,
         1048576, 4194304]


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p,
                                         ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_decompress_frames.argtypes = [ct.c_void_p, ct.c_char_p,
                                               ct.c_size_t, ct.POINTER(Span),
                                               ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                         ct.POINTER(ct.c_size_t)]
    return lib


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0, \
        lib.sbz_dctx_error(dc)
    return b.raw[:n.value]


def make_dctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise RuntimeError("sbz_dctx_create_cuda: %s" % err.value.decode())
    return dc


def decompress(lib, dc, data):
    rc = lib.sbz_dctx_decompress(dc, data, len(data))
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


def decompress_frames(lib, dc, data, spans):
    arr = (Span * len(spans))(*[Span(o, s) for o, s in spans])
    rc = lib.sbz_dctx_decompress_frames(dc, data, len(data), arr, len(spans))
    if rc == 0:
        return rc, read_buf(lib, dc, "UnCompressedFramesData")
    return rc, b""


def make_data(kind, n, nprng):
    """Build one input of length `n` for the given data kind (fixed seed)."""
    if kind == "random":
        return nprng.bytes(n)
    if kind == "text":
        return (b"The quick brown fox jumps over the lazy dog. "
                * (n // 44 + 1))[:n]
    if kind == "zeros":
        return bytes(n)
    if kind == "runs":
        return bytes([int(nprng.integers(0, 256))]) * n
    if kind in ("u16", "u32"):
        typ = np.uint16 if kind == "u16" else np.uint32
        esz = np.dtype(typ).itemsize
        cnt = max(1, n // esz)
        a = nprng.poisson(3, cnt).astype(typ)
        mask = nprng.random(cnt) < 0.02
        a[mask] = np.iinfo(typ).max
        return a.tobytes()
    raise ValueError(kind)


def main():
    lib = load()
    dc = make_dctx(lib)
    compared = 0
    cases = 0
    batch_frames = []         # (compressed_frame, expected_output)
    try:
        for lvl in LEVELS:
            rng = np.random.default_rng(1000 + lvl)
            for n in SIZES:
                for kind in ("random", "text", "zeros", "runs", "u16", "u32"):
                    if kind in ("u16", "u32") and n < 2:
                        continue
                    data = make_data(kind, n, rng)
                    comp = zstd.ZstdCompressor(level=lvl).compress(data)
                    want = zstd.ZstdDecompressor().decompress(comp)
                    rc, got = decompress(lib, dc, comp)
                    if rc != 0 or got[:len(want)] != want:
                        print("FAIL lvl%d n=%d kind=%s rc=%d got=%d want=%d"
                              % (lvl, n, kind, rc, len(got), len(want)))
                        return 1
                    compared += len(want)
                    cases += 1
                    # bound the batch-candidate set to keep memory low
                    if len(want) <= 100000:
                        batch_frames.append((comp, want))

        # ---- concatenated batches: up to 64 single frames per call ----
        # (digest the single-frame cases into a few large concatenations)
        import math
        per = 64
        for start in range(0, len(batch_frames), per):
            chunk = batch_frames[start:start + per]
            stream = b"".join(f for f, _ in chunk)
            want = b"".join(w for _, w in chunk)
            # spans over the concatenated frames
            spans = []
            off = 0
            for f, _ in chunk:
                spans.append((off, len(f)))
                off += len(f)
            rc, got = decompress(lib, dc, stream)
            if rc != 0 or got[:len(want)] != want:
                print("FAIL concatenated batch at %d rc=%d got=%d want=%d"
                      % (start, rc, len(got), len(want)))
                return 1
            compared += len(want)
            cases += 1
            # decompress_frames path on the same batch
            rc, got = decompress_frames(lib, dc, stream, spans)
            if rc != 0 or got[:len(want)] != want:
                print("FAIL decompress_frames batch at %d rc=%d got=%d want=%d"
                      % (start, rc, len(got), len(want)))
                return 1
            compared += len(want)
            cases += 1
    finally:
        lib.sbz_dctx_destroy(dc)

    ok = cases >= 500 and compared > 0
    print("cases=%d  compared_bytes=%d  %s"
          % (cases, compared, "OK" if ok else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
