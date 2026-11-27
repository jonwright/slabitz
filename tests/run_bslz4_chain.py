#!/usr/bin/env python3
"""P3: sbz_dctx_bslz4_decompress() on real bitshuffle-LZ4 HDF5 chunks.

Default mode writes datasets with hdf5plugin.Bitshuffle(cname="lz4"), reads each
chunk raw with read_direct_chunk, decodes with sbz_dctx_bslz4_decompress (one chunk
per call and several chunks per call), and compares BshufOutput byte-exactly with
h5py reading the dataset. Covers u8/u16/u32/u64, 8 kB / 248 kB / 256 kB / a
non-power-of-two block, a raw tail, a last block with N%32 != 0, a 1-element and
an all-tail (< 8 elements) dataset, a multi-chunk dataset, and four data shapes
(zeros, random/incompressible, smooth detector-like, many short-offset repeats).

Corrupt cases must give SBZ_ERR_CORRUPT, never a crash or a hang: a truncated
chunk, a block-size record past the end, and crafted LZ4 blocks with a 0 offset,
an offset before the block start, a match running past the block end, and a block
that decodes short.

--h5 FILE [--frames N] decodes the first N (default 4) chunks of the first chunked
bitshuffle-LZ4 dataset in FILE and compares with h5py; prints GB/s of the decode
call (wall clock, after a warm-up).

Each GPU case runs in a subprocess under a timeout, as the other tests do.

Usage: python3 tests/run_bslz4_chain.py
"""
import ctypes as ct
import math
import os
import struct
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))  # e.g. build/sm_35
TIMEOUT = 180
SBZ_ERR_ARGS, SBZ_ERR_CORRUPT, SBZ_ERR_LIMIT = -3, -5, -6


def data_required(label):
    """T02: with SLABITZ_REQUIRE_DATA=1 a missing data file is a failure, not a skip.

    Returns 1 if the requirement was turned into a failure (caller should count it
    as a bad case), else 0. `label` is the missing file's path."""
    if os.environ.get("SLABITZ_REQUIRE_DATA") == "1":
        print("FAIL: data required but %s is missing (set SLABITZ_EXAMPLES)" % label)
        return 1
    print("ok   %s: not readable, skipped (set SLABITZ_EXAMPLES)" % label)
    return 0


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_error.restype = ct.c_char_p
    lib.sbz_dctx_error.argtypes = [ct.c_void_p]
    lib.sbz_dctx_bslz4_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                             ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]
    lib.sbz_dctx_lz4_blocks.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint32),
                                       ct.POINTER(ct.c_uint32)]
    lib.sbz_dctx_lz4_blocks.restype = ct.c_int
    lib.sbz_dctx_bslz4_decompress_async.argtypes = lib.sbz_dctx_bslz4_decompress.argtypes
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    return lib


def read_buf(lib, dc, name):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:n.value]


def decode(lib, dc, raw_chunks, elem_size):
    """Decode a list of raw chunk byte-strings (concatenated) with one call."""
    src = b"".join(raw_chunks)
    arr = (Span * len(raw_chunks))(*[
        Span(sum(map(len, raw_chunks[:i])), len(c)) for i, c in enumerate(raw_chunks)])
    rc = lib.sbz_dctx_bslz4_decompress(dc, src, len(src), arr, len(raw_chunks), elem_size)
    return rc


def gen_data(kind, dt, n, seed):
    """A data array exercising a different LZ4 regime for `kind`."""
    import numpy as np
    rng = np.random.default_rng(seed)
    if kind == "zeros":
        return np.zeros(n, dtype=dt)
    if kind == "rand":
        if dt == np.uint8:
            return rng.integers(0, 256, n, dtype=dt)
        return rng.integers(0, np.iinfo(dt).max, n, dtype=dt)
    if kind == "smooth":
        # smooth detector-like: slow ramp + gentle noise, clipped to the dtype
        base = np.arange(n, dtype=np.float64) * 0.01 + rng.normal(0, 4, n)
        return np.clip(base, 0, 4000).astype(dt)
    if kind == "repeat":
        # many short-offset repeats: cycle random small blocks through periods 1..40,
        # so LZ4 finds overlapping matches with offsets 1..40
        a = np.empty(n, dtype=dt)
        pos = 0
        while pos < n:
            p = (pos % 40) + 1
            k = min(n - pos, p * 3)
            sym = rng.integers(0, 16, p, dtype=np.uint8).astype(dt)
            a[pos:pos + k] = np.tile(sym, k // p + 1)[:k]
            pos += k
        return a
    raise ValueError(kind)


def make_h5(specs, dirpath):
    """Write the given specs into one h5 with cname="lz4"; yield per spec
    (name, dtype, raw_chunks, ref). """
    import h5py
    import hdf5plugin
    import numpy as np
    fn = os.path.join(dirpath, "c.h5")
    with h5py.File(fn, "w") as f:
        for name, dt, n, nel, ces, seed, kind in specs:
            a = gen_data(kind, dt, n, seed)
            chunk = (ces,) if ces else (n,)
            f.create_dataset(name, data=a, chunks=chunk,
                             **hdf5plugin.Bitshuffle(nelems=nel, cname="lz4"))
    out = []
    with h5py.File(fn, "r") as f:
        for name, dt, n, nel, ces, seed, kind in specs:
            ds = f[name]
            e = np.dtype(dt).itemsize
            csz = ces or n
            nch = int(math.ceil(n / csz))
            raw = []
            for i in range(nch):
                _, chunk = ds.id.read_direct_chunk((i,))
                raw.append(bytes(chunk))
            out.append((name, dt, raw, ds[()].tobytes(), csz, n))
    return out


SPECS = [
    ("u16-default", "uint16", 300000, 0, 0, 7, "rand"),
    ("u32-tail", "uint32", 200003, 0, 0, 9, "smooth"),
    ("u16-248k", "uint16", 400000, 248 * 1024 // 2, 0, 11, "repeat"),
    ("u8-248k-odd", "uint8", 600011, 248 * 1024, 0, 13, "rand"),
    ("u32-248k", "uint32", 150000, 248 * 1024 // 4, 0, 17, "smooth"),
    ("u64-256k", "uint64", 100000, 256 * 1024 // 8, 0, 19, "zeros"),
    ("u8-256k", "uint8", 700000, 256 * 1024, 0, 23, "repeat"),
    ("u16-nonpow2", "uint16", 300000, 24 * 1024, 0, 41, "rand"),
    ("u8-manyrepeat", "uint8", 500000, 0, 0, 47, "repeat"),
    ("u16-one", "uint16", 1, 0, 0, 29, "rand"),
    ("u32-tail3", "uint32", 3, 0, 0, 31, "rand"),
]
# A multi-chunk case built from single-chunk datasets (read_direct_chunk only returns
# the first chunk's raw bytes for a dataset), bundled into one decode call.
MULTI = ("u16-multi", (((300000, 0), (250003, 0), (100000, 0)),), "uint16", 53, "repeat")
# 40 chunks in one call, each with a different partial last block (100000 + 8i elements:
# last blocks of 1696 + 8i elements, a mix of %32 == 0 (warp kernel) and not (slow
# kernel)): more distinct unshuffle buckets than the first block-table version allowed.
MULTI_MANY = ("u16-many", (tuple((100000 + 8 * i, 0) for i in range(40)),), "uint16", 71,
              "repeat")
MULTIS = {MULTI[0]: MULTI, MULTI_MANY[0]: MULTI_MANY}


def buf_eq(got, want):
    """BshufOutput is a raw buffer padded to 4 bytes; compare the real bytes."""
    return len(got) >= len(want) and got[:len(want)] == want


def run_one(name):
    import numpy as np
    spec = next((s for s in SPECS if s[0] == name), None)
    if spec is None:
        print("FAIL %s: unknown spec" % name)
        return 1
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        (entry,) = make_h5([spec], td)
        _, dt, raw, ref, csz, n = entry
        e = np.dtype(dt).itemsize
        lib = load()
        err = ct.create_string_buffer(512)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL %s: create: %s" % (name, err.value.decode()))
            return 1
        try:
            rc = decode(lib, dc, raw, e)
            if rc != 0:
                print("FAIL %s: rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            got = read_buf(lib, dc, "BshufOutput")
            if not buf_eq(got, ref):
                print("FAIL %s: mismatch (got %d want %d bytes)" % (name, len(got), len(ref)))
                return 1
            print("ok   %s (e=%d, chunks=%d, %d B compared, matches h5py)"
                  % (name, e, len(raw), len(ref)))
            return 0
        finally:
            lib.sbz_dctx_destroy(dc)


def run_multi(name):
    import numpy as np
    if name != MULTI[0]:
        print("FAIL %s: unknown multi spec" % name)
        return 1
    _, specs, dtype, seed, kind = MULTI
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        entries = make_h5([("%s-%d" % (name, i), dtype, s, nel, 0, seed + i, kind)
                           for i, (s, nel) in enumerate(specs[0])], td)
        e = np.dtype(dtype).itemsize
        raws = [x[2][0] for x in entries]          # first (only) chunk of each dataset
        refs = [x[3] for x in entries]
        lib = load()
        err = ct.create_string_buffer(512)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL %s: create: %s" % (name, err.value.decode()))
            return 1
        try:
            rc = decode(lib, dc, raws, e)  # several chunks in ONE call
            if rc != 0:
                print("FAIL %s: multi rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            got = read_buf(lib, dc, "BshufOutput")
            want = b"".join(refs)
            if not buf_eq(got, want):
                print("FAIL %s: multi mismatch (got %d want %d bytes)" % (name, len(got), len(want)))
                return 1
            for i, (r, ref) in enumerate(zip(raws, refs)):  # one chunk per call
                rc = decode(lib, dc, [r], e)
                if rc != 0 or not buf_eq(read_buf(lib, dc, "BshufOutput"), ref):
                    print("FAIL %s: single chunk %d rc=%d" % (name, i, rc))
                    return 1
            print("ok   %s (e=%d, %d chunks in one call, %d B compared, matches h5py)"
                  % (name, e, len(raws), len(want)))
            return 0
        finally:
            lib.sbz_dctx_destroy(dc)
    return 0


def lz4_counts(lib, dc):
    fb, ub = ct.c_uint32(), ct.c_uint32()
    lib.sbz_dctx_lz4_blocks(dc, ct.byref(fb), ct.byref(ub))
    return fb.value, ub.value


def eligible_full(rawd, e):
    """Whether rawd has at least one full block that takes the fused path (Phase 3,
    step 2): a full block with e in {1,2,4}, blockElems % 32 == 0 and blockElems*e
    <= 8192. Used to assert the fused kernel is actually used on the =1 run."""
    if len(rawd) < 12:
        return False
    total = int.from_bytes(rawd[0:8], "big")
    block_bytes = int.from_bytes(rawd[8:12], "big")
    if e not in (1, 2, 4) or block_bytes == 0 or block_bytes % e:
        return False
    bo = block_bytes // e
    if bo % 32 or block_bytes > 8192:
        return False
    return total // e >= bo


def run_both(name):
    """Run the case with SLABITZ_LZ4_FUSED=0 and =1 in one process, compare each with
    h5py, compare them with each other, and print how many blocks took the fused path.
    The 8 kB cases with e in {1,2,4} must report a nonzero fused count on =1 and zero
    on =0 (the debugging path / oracle)."""
    import numpy as np
    spec = next((s for s in SPECS if s[0] == name), None)
    if spec is None:
        print("FAIL %s: unknown spec" % name)
        return 1
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        (entry,) = make_h5([spec], td)
        _, dt, raw, ref, csz, n = entry
        e = np.dtype(dt).itemsize
        lib = load()
        err = ct.create_string_buffer(512)
        got = {}
        for fused in (0, 1):
            os.environ["SLABITZ_LZ4_FUSED"] = str(fused)
            dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
            if not dc:
                print("FAIL %s: create: %s" % (name, err.value.decode()))
                return 1
            try:
                rc = decode(lib, dc, raw, e)
                if rc != 0:
                    print("FAIL %s fused=%d rc=%d: %s" % (name, fused, rc,
                                                          lib.sbz_dctx_error(dc).decode()))
                    return 1
                got[fused] = (read_buf(lib, dc, "BshufOutput"), lz4_counts(lib, dc))
            finally:
                lib.sbz_dctx_destroy(dc)
        os.environ.pop("SLABITZ_LZ4_FUSED", None)
        g0, (f0, u0) = got[0]
        g1, (f1, u1) = got[1]
        el = eligible_full(raw[0], e)
        ok = (g0[:len(ref)] == ref and g1[:len(ref)] == ref and g0[:len(ref)] == g1[:len(ref)] and
              f0 == 0 and (not el or f1 > 0))
        print("%s %s (e=%d, %d chunks, %d B compared; fused=0: %d fused/%d unfused, "
              "fused=1: %d fused/%d unfused, %s)" % ("ok  " if ok else "FAIL", name, e, len(raw),
                                                     len(ref), f0, u0, f1, u1,
                                                     "match h5py and each other" if ok else "MISMATCH"))
        return 0 if ok else 1


def run_both_multi(name):
    import numpy as np
    if name not in MULTIS:
        print("FAIL %s: unknown multi spec" % name)
        return 1
    _, specs, dtype, seed, kind = MULTIS[name]
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        entries = make_h5([("%s-%d" % (name, i), dtype, s, nel, 0, seed + i, kind)
                           for i, (s, nel) in enumerate(specs[0])], td)
        e = np.dtype(dtype).itemsize
        raws = [x[2][0] for x in entries]          # first (only) chunk of each dataset
        refs = [x[3] for x in entries]
        lib = load()
        err = ct.create_string_buffer(512)
        got = {}
        for fused in (0, 1):
            os.environ["SLABITZ_LZ4_FUSED"] = str(fused)
            dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
            if not dc:
                print("FAIL %s: create: %s" % (name, err.value.decode()))
                return 1
            try:
                rc = decode(lib, dc, raws, e)  # several chunks in ONE call (mixed path)
                if rc != 0:
                    print("FAIL %s fused=%d rc=%d: %s" % (name, fused, rc,
                                                          lib.sbz_dctx_error(dc).decode()))
                    return 1
                got[fused] = (read_buf(lib, dc, "BshufOutput"), lz4_counts(lib, dc))
            finally:
                lib.sbz_dctx_destroy(dc)
        os.environ.pop("SLABITZ_LZ4_FUSED", None)
        want = b"".join(refs)
        g0, (f0, u0) = got[0]
        g1, (f1, u1) = got[1]
        ok = (g0[:len(want)] == want and g1[:len(want)] == want and g0[:len(want)] == g1[:len(want)]
              and f0 == 0 and f1 > 0)  # these 8 kB e=2 chunks mix fused full blocks + tails
        nlast = len({s % 4096 for s, _ in specs[0]})  # 8 kB u16 blocks: 4096 elements
        ok = ok and len(want) > 0 and nlast == len(specs[0])
        print("%s %s (e=%d, %d chunks/call, %d distinct last blocks, %d B compared; fused=0: %d "
              "fused, fused=1: %d fused/%d unfused, %s)"
              % ("ok  " if ok else "FAIL", name, e, len(raws), nlast, len(want), f0, f1, u1,
                 "match h5py and each other" if ok else "MISMATCH"))
        return 0 if ok else 1


def run_both_corrupt(case):
    """A corrupt LZ4 block inside an eligible (fused-path) block: SBZ_ERR_CORRUPT, no hang."""
    payloads = {"offset0f": bytes([0x40, 0x41, 0x42, 0x43, 0x44, 0x00, 0x00]),
                "shortf": bytes([0x20, 0x41, 0x42])}
    if case not in payloads:
        print("FAIL corrupt %s: unknown case" % case)
        return 1
    data = make_chunk(32, 32, 1, [payloads[case]])  # 32 elements, e=1, eligible (32%32==0, <=8192)
    lib = load()
    err = ct.create_string_buffer(512)
    os.environ["SLABITZ_LZ4_FUSED"] = "1"
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    os.environ.pop("SLABITZ_LZ4_FUSED", None)
    if not dc:
        print("FAIL corrupt %s: create: %s" % (case, err.value.decode()))
        return 1
    try:
        arr = (Span * 1)(Span(0, len(data)))
        rc = lib.sbz_dctx_bslz4_decompress(dc, data, len(data), arr, 1, 1)
        if rc != SBZ_ERR_CORRUPT:
            print("FAIL corrupt %s (fused): rc=%d (want %d): %s"
                  % (case, rc, SBZ_ERR_CORRUPT, lib.sbz_dctx_error(dc).decode()))
            return 1
        print("ok   corrupt %s (fused path, rc=%d %s)" % (case, rc, lib.sbz_dctx_error(dc).decode()))
        return 0
    finally:
        lib.sbz_dctx_destroy(dc)


def run_async(name):
    """P4 step 2a: two contexts in one CUDA context, alternated from one thread with
    sbz_dctx_bslz4_decompress_async + sbz_dctx_wait, three rounds each, on different
    inputs (MULTI on one, MULTI_MANY on the other); each output vs h5py after its
    wait. A second call before the wait is refused (SBZ_ERR_ARGS) and leaves the
    queued call intact; a corrupt chunk is reported by sbz_dctx_wait."""
    import numpy as np  # noqa: F401
    if name != "two-ctx":
        print("FAIL async %s: unknown case" % name)
        return 1
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        inputs = []
        for spec in (MULTI, MULTI_MANY):
            mname, specs, dtype, seed, kind = spec
            entries = make_h5([("%s-%d" % (mname, i), dtype, sz, nel, 0, seed + i, kind)
                               for i, (sz, nel) in enumerate(specs[0])], td)
            raws = [x[2][0] for x in entries]
            src = b"".join(raws)
            offs = [0]
            for r in raws:
                offs.append(offs[-1] + len(r))
            spans = (Span * len(raws))(*[Span(offs[i], len(r)) for i, r in enumerate(raws)])
            inputs.append((mname, src, spans, len(raws), np.dtype(dtype).itemsize,
                           b"".join(x[3] for x in entries)))
        lib = load()
        cu = ct.CDLL("libcuda.so.1")
        cu.cuInit(0)
        dev = ct.c_int()
        cu.cuDeviceGet(ct.byref(dev), 0)
        prim = ct.c_void_p()
        # one CUDA context for both decoder contexts: the primary context (CUDA
        # 7.0+), else (Kepler's driver 340, CUDA 6.5) a context of our own
        if hasattr(cu, "cuDevicePrimaryCtxRetain"):
            rc = cu.cuDevicePrimaryCtxRetain(ct.byref(prim), dev)
        else:
            rc = cu.cuCtxCreate_v2(ct.byref(prim), 0, dev)
        if rc or cu.cuCtxSetCurrent(prim):
            print("FAIL async: no CUDA context (rc=%d)" % rc)
            return 1
        err = ct.create_string_buffer(512)
        dcs = [lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err)) for _ in inputs]
        if not all(dcs):
            print("FAIL async: create: %s" % err.value.decode())
            return 1
        bad = 0
        compared = 0
        try:
            for rnd in range(3):
                for dc, (mname, src, spans, n, e, _want) in zip(dcs, inputs):
                    rc = lib.sbz_dctx_bslz4_decompress_async(dc, src, len(src), spans, n, e)
                    if rc != 0:
                        print("FAIL async %s round %d: rc=%d %s"
                              % (mname, rnd, rc, lib.sbz_dctx_error(dc).decode()))
                        return 1
                if rnd == 0:  # a second call before the wait must be refused
                    mname, src, spans, n, e, _ = inputs[0]
                    rc = lib.sbz_dctx_bslz4_decompress_async(dcs[0], src, len(src), spans, n, e)
                    if rc != SBZ_ERR_ARGS:
                        print("FAIL async: call before wait gave rc=%d (want %d)"
                              % (rc, SBZ_ERR_ARGS))
                        bad += 1
                for dc, (mname, src, spans, n, e, want) in zip(dcs, inputs):
                    rc = lib.sbz_dctx_wait(dc)
                    got = read_buf(lib, dc, "BshufOutput")[:len(want)]
                    if rc != 0 or got != want:
                        print("FAIL async %s round %d: rc=%d, output %s"
                              % (mname, rnd, rc, "matches" if got == want else "DIFFERS"))
                        bad += 1
                    compared += len(want)
            # corrupt chunk: queued fine, reported by the wait, context usable after
            data = make_chunk(32, 32, 1, [bytes([0x40, 0x41, 0x42, 0x43, 0x44, 0x00, 0x00])])
            arr = (Span * 1)(Span(0, len(data)))
            rc1 = lib.sbz_dctx_bslz4_decompress_async(dcs[1], data, len(data), arr, 1, 1)
            rc2 = lib.sbz_dctx_wait(dcs[1])
            if rc1 != 0 or rc2 != SBZ_ERR_CORRUPT:
                print("FAIL async corrupt: async rc=%d, wait rc=%d (want 0, %d)"
                      % (rc1, rc2, SBZ_ERR_CORRUPT))
                bad += 1
            mname, src, spans, n, e, want = inputs[1]
            rc = lib.sbz_dctx_bslz4_decompress(dcs[1], src, len(src), spans, n, e)
            got = read_buf(lib, dcs[1], "BshufOutput")[:len(want)]
            if rc != 0 or got != want:
                print("FAIL async: context not usable after a corrupt chunk (rc=%d)" % rc)
                bad += 1
            compared += len(want)
        finally:
            for dc in dcs:
                lib.sbz_dctx_destroy(dc)
        ok = bad == 0 and compared > 0
        print("%s async two-ctx (3 rounds x 2 contexts, %d B compared, refused call before "
              "wait, corrupt chunk reported by wait)" % ("ok  " if ok else "FAIL", compared))
        return 0 if ok else 1


def run_h5_both(path, frames):
    """The real file: decode the first `frames` chunks with both paths, compare each to
    h5py and to each other. Skipped (with a notice) if the file is not readable."""
    import h5py
    import hdf5plugin  # noqa: F401  (registers the bitshuffle filter for the h5py read)
    import numpy as np
    if not os.path.exists(path):
        if os.environ.get("SLABITZ_REQUIRE_DATA") == "1":
            print("FAIL: data required but %s is missing (set SLABITZ_EXAMPLES)" % path)
            return 1
        print("ok   h5 %s: file not readable, skipped" % path)
        return 0
    got = {}
    with h5py.File(path, "r") as f:
        ds = find_chunked(f)
        if ds is None:
            print("FAIL h5: no chunked dataset in %s" % path)
            return 1
        e = ds.dtype.itemsize
        nframes = min(frames, ds.shape[0])
        nchunks = int(np.ceil(ds.shape[0] / ds.chunks[0]))
        raw = []
        for i in range(min(nframes, nchunks)):
            _, chunk = ds.id.read_direct_chunk(tuple([i] + [0] * (ds.ndim - 1)))
            raw.append(bytes(chunk))
        nout = sum(int.from_bytes(r[0:8], "big") for r in raw)
        want = ds[0:nframes].reshape(-1).tobytes()[:nout]
    lib = load()
    err = ct.create_string_buffer(512)
    for fused in (0, 1):
        os.environ["SLABITZ_LZ4_FUSED"] = str(fused)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL h5 %s: create: %s" % (path, err.value.decode()))
            return 1
        try:
            rc = decode(lib, dc, raw, e)
            if rc != 0:
                print("FAIL h5 %s fused=%d rc=%d: %s" % (path, fused, rc,
                                                         lib.sbz_dctx_error(dc).decode()))
                return 1
            got[fused] = (read_buf(lib, dc, "BshufOutput"), lz4_counts(lib, dc))
        finally:
            lib.sbz_dctx_destroy(dc)
    os.environ.pop("SLABITZ_LZ4_FUSED", None)
    g0, (f0, u0) = got[0]
    g1, (f1, u1) = got[1]
    ok = g0[:len(want)] == want and g1[:len(want)] == want and g0[:len(want)] == g1[:len(want)]
    print("%s h5 %s: %d chunks, %d B compared, fused=0 %d/%d, fused=1 %d/%d, %s"
          % ("ok  " if ok else "FAIL", os.path.basename(path), len(raw), len(want), f0, u0, f1,
             u1, "match h5py and each other" if ok else "MISMATCH"))
    return 0 if ok else 1


def make_chunk(total_elements, block_bytes, elem_size, payloads, tail=b""):
    """A bitshuffle-LZ4 chunk: [BE u64 total][BE u32 block bytes], then per block
    [BE u32 c][c bytes], then the raw tail. Used to hand-craft corrupt blocks."""
    out = bytearray()
    out += struct.pack(">Q", total_elements * elem_size)
    out += struct.pack(">I", block_bytes)
    for p in payloads:
        out += struct.pack(">I", len(p))
        out += p
    out += tail
    return bytes(out)


def run_corrupt(case):
    import numpy as np
    if case in ("trunc", "fsize"):
        spec = next(s for s in SPECS if s[0] == "u16-default")
        with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
            (entry,) = make_h5([spec], td)
            _, dt, raw, ref, csz, n = entry
            e = np.dtype(dt).itemsize
            chunk = bytearray(raw[0])
            if case == "trunc":
                data = bytes(chunk[:-5])
            else:  # fsize: the first block-size record (u32 BE at byte 12) is huge
                chunk[12] ^= 0xFF
                data = bytes(chunk)
            return corrupt_case(case, data, e)
    # hand-crafted corrupt LZ4 blocks, all in one 8-element (e=1) full block. The
    # container is valid (the walk passes); only the LZ4 payload is corrupt.
    if case == "offset0":
        payload = bytes([0x40, 0x41, 0x42, 0x43, 0x44, 0x00, 0x00])
    elif case == "offsetstart":
        payload = bytes([0x40, 0x41, 0x42, 0x43, 0x44, 0x05, 0x00])
    elif case == "matchover":
        payload = bytes([0x26, 0x41, 0x42, 0x01, 0x00])
    elif case == "decodeshort":
        payload = bytes([0x20, 0x41, 0x42])
    else:
        print("FAIL corrupt %s: unknown case" % case)
        return 1
    data = make_chunk(8, 8, 1, [payload])
    return corrupt_case(case, data, 1)


def corrupt_case(case, data, e):
    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        print("FAIL corrupt %s: create: %s" % (case, err.value.decode()))
        return 1
    try:
        arr = (Span * 1)(Span(0, len(data)))
        rc = lib.sbz_dctx_bslz4_decompress(dc, data, len(data), arr, 1, e)
        if rc != SBZ_ERR_CORRUPT:
            print("FAIL corrupt %s: rc=%d (want %d): %s"
                  % (case, rc, SBZ_ERR_CORRUPT, lib.sbz_dctx_error(dc).decode()))
            return 1
        print("ok   corrupt %s (rc=%d %s)" % (case, rc, lib.sbz_dctx_error(dc).decode()))
        return 0
    finally:
        lib.sbz_dctx_destroy(dc)


def find_chunked(node):
    """Return the first chunked dataset in a (possibly nested) h5 file."""
    import h5py
    if isinstance(node, h5py.Dataset):
        return node if node.chunks is not None else None
    for k in node:
        found = find_chunked(node[k])
        if found is not None:
            return found
    return None


def run_h5(path, frames):
    import h5py
    import hdf5plugin  # noqa: F401  (registers the bitshuffle filter for the h5py read)
    import numpy as np
    with h5py.File(path, "r") as f:
        ds = find_chunked(f)
        if ds is None:
            print("FAIL h5: no chunked dataset in %s" % path)
            return 1
        e = ds.dtype.itemsize
        nframes = min(frames, ds.shape[0])
        nchunks = int(np.ceil(ds.shape[0] / ds.chunks[0]))
        raw = []
        for i in range(min(nframes, nchunks)):
            _, chunk = ds.id.read_direct_chunk(tuple([i] + [0] * (ds.ndim - 1)))
            raw.append(bytes(chunk))
        nout = sum(int.from_bytes(r[0:8], "big") for r in raw)
        want = ds[0:nframes].reshape(-1).tobytes()[:nout]
        lib = load()
        err = ct.create_string_buffer(512)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL h5: create: %s" % err.value.decode())
            return 1
        try:
            rc = decode(lib, dc, raw, e)  # warm-up
            if rc != 0:
                print("FAIL h5 %s: rc=%d: %s" % (path, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            if not buf_eq(read_buf(lib, dc, "BshufOutput"), want):
                print("FAIL h5 %s: mismatch in first %d chunks" % (path, len(raw)))
                return 1
            t0 = time.monotonic()
            rc = decode(lib, dc, raw, e)
            dt = time.monotonic() - t0
            if rc != 0:
                print("FAIL h5 %s: timed rc=%d: %s" % (path, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            gbs = (nout / dt) / 1e9
            print("ok   h5 %s: %d chunks (%d bytes) decoded in %.4f s (%.2f GB/s)"
                  % (path, len(raw), nout, dt, gbs))
            return 0
        finally:
            lib.sbz_dctx_destroy(dc)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--h5":
        path = argv[1]
        frames = 4
        if len(argv) >= 4 and argv[2] == "--frames":
            frames = int(argv[3])
        subprocess.run(["make", "-s", "-C", ROOT], check=True)
        return run_h5(path, frames)
    if len(argv) == 2 and argv[0] == "--one":
        return run_one(argv[1])
    if len(argv) == 2 and argv[0] == "--both":
        return run_both(argv[1])
    if len(argv) == 2 and argv[0] == "--multi":
        return run_multi(argv[1])
    if len(argv) == 2 and argv[0] == "--both-multi":
        return run_both_multi(argv[1])
    if len(argv) == 2 and argv[0] == "--corrupt":
        return run_corrupt(argv[1])
    if len(argv) == 2 and argv[0] == "--async":
        return run_async(argv[1])
    if len(argv) == 2 and argv[0] == "--both-corrupt":
        return run_both_corrupt(argv[1])
    if len(argv) >= 3 and argv[0] == "--h5-both":
        return run_h5_both(argv[1], int(argv[2]))
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    bad = 0
    for name, *_ in SPECS:
        try:
            r = subprocess.run([sys.executable, __file__, "--both", name], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL %s: timeout (%ds) -- GPU hang?" % (name, TIMEOUT))
            bad += 1
    for mname in MULTIS:
        try:
            r = subprocess.run([sys.executable, __file__, "--both-multi", mname], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL %s: timeout (%ds) -- GPU hang?" % (mname, TIMEOUT))
            bad += 1
    for case in ("trunc", "fsize", "offset0", "offsetstart", "matchover", "decodeshort"):
        try:
            r = subprocess.run([sys.executable, __file__, "--corrupt", case], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL corrupt %s: timeout (%ds) -- GPU hang?" % (case, TIMEOUT))
            bad += 1
    for case in ("offset0f", "shortf"):
        try:
            r = subprocess.run([sys.executable, __file__, "--both-corrupt", case], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL corrupt %s: timeout (%ds) -- GPU hang?" % (case, TIMEOUT))
            bad += 1
    try:
        r = subprocess.run([sys.executable, __file__, "--async", "two-ctx"], timeout=TIMEOUT)
        bad += r.returncode != 0
    except subprocess.TimeoutExpired:
        print("FAIL async two-ctx: timeout (%ds) -- GPU hang?" % TIMEOUT)
        bad += 1
    examples = os.environ.get("SLABITZ_EXAMPLES") or os.path.join(ROOT, "workspace", "examples")
    paths = []
    for name in ("silicon_lz4.h5", "ceo2_lz4.h5"):
        path = os.path.join(examples, name)
        if os.path.exists(path):
            paths.append(path)
        else:
            bad += data_required(path)  # T02: REQUIRE_DATA turns the skip into a failure
    for path in paths:
        try:
            r = subprocess.run([sys.executable, __file__, "--h5-both", path, "32"], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL h5-both %s: timeout (%ds) -- GPU hang?" % (path, TIMEOUT))
            bad += 1
    print("%s: %d case(s) failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
