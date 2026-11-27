#!/usr/bin/env python3
"""P3: sbz_dctx_bshuf_decompress() on real bitshuffle-zstd HDF5 chunks.

Default mode writes datasets with hdf5plugin.Bitshuffle(cname="zstd"), reads each
chunk raw with read_direct_chunk, decodes with sbz_dctx_bshuf_decompress (one chunk
per call and several chunks per call), and compares BshufOutput byte-exactly with
h5py reading the dataset. Covers u8/u16/u32/u64, 8 kB/248 kB/256 kB blocks, a raw
tail, a last block with N%32 != 0, a 1-element and an all-tail (< 8 elements)
dataset, and a multi-chunk dataset. Corrupt cases (truncated chunk, flipped size
record, wrong elemSize) must give a SBZ_ERR_* code, never a crash or a hang.

--h5 FILE [--frames N] decodes the first N (default 4) chunks of the first chunked
bitshuffle-zstd dataset in FILE and compares with h5py; prints GB/s of the decode
call (wall clock, after a warm-up).

Each GPU case runs in a subprocess under a timeout, as the other tests do.

Usage: python3 tests/run_bshuf_chain.py
"""
import ctypes as ct
import os
import struct
import subprocess
import sys
import tempfile
import time
import math

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
    lib.sbz_dctx_bshuf_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t,
                                             ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p,
                                        ct.POINTER(ct.c_size_t)]
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
    rc = lib.sbz_dctx_bshuf_decompress(dc, src, len(src), arr, len(raw_chunks), elem_size)
    return rc


def make_h5(specs, dirpath):
    """Write the given specs into one h5; yield (name, dtype, raw_chunks, ref) per spec."""
    import h5py
    import hdf5plugin
    import numpy as np
    fn = os.path.join(dirpath, "c.h5")
    with h5py.File(fn, "w") as f:
        for name, dt, n, nel, ces, seed in specs:
            rng = np.random.default_rng(seed)
            a = (rng.poisson(3, n) + (np.arange(n) % 97 == 0) * 500).astype(dt)
            chunk = (ces,) if ces else (n,)
            f.create_dataset(name, data=a, chunks=chunk,
                             **hdf5plugin.Bitshuffle(nelems=nel, cname="zstd"))
    out = []
    with h5py.File(fn, "r") as f:
        for name, dt, n, nel, ces, seed in specs:
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
    ("u16-default", "uint16", 300000, 0, 0, 7),
    ("u32-tail", "uint32", 200003, 0, 0, 9),
    ("u16-248k", "uint16", 400000, 248 * 1024 // 2, 0, 11),
    ("u8-248k-odd", "uint8", 600011, 248 * 1024, 0, 13),
    ("u32-248k", "uint32", 150000, 248 * 1024 // 4, 0, 17),
    ("u64-256k", "uint64", 100000, 256 * 1024 // 8, 0, 19),
    ("u8-256k", "uint8", 700000, 256 * 1024, 0, 23),
    ("u16-one", "uint16", 1, 0, 0, 29),
    ("u32-tail3", "uint32", 3, 0, 0, 31),
]
# The multi-chunk case uses several single-chunk datasets (read_direct_chunk only
# returns raw bytes for a dataset's first chunk), bundled into one decode call.
MULTI = ("u16-multi", (((300000, 0), (250003, 0), (100000, 0)),), "uint16", 37)
MULTI_DTYPES = {"uint8": ct.c_uint8, "uint16": ct.c_uint16, "uint32": ct.c_uint32,
                "uint64": ct.c_uint64}


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
            # corrupt: wrong elemSize on the same context (must stay usable)
            rc1 = decode(lib, dc, raw, 3 if e != 3 else 5)
            rc2 = decode(lib, dc, raw, e)
            if rc1 == 0 or rc2 != 0:
                print("FAIL %s: corrupt/wrong-elem rc=%d,%d (want !=0, 0)" % (name, rc1, rc2))
                return 1
            print("ok   %s (e=%d, chunks=%d, out %d B, matches h5py)"
                  % (name, e, len(raw), len(ref)))
            return 0
        finally:
            lib.sbz_dctx_destroy(dc)


def run_multi(name):
    import numpy as np
    if name != MULTI[0]:
        print("FAIL %s: unknown multi spec" % name)
        return 1
    _, specs, dtype, seed = MULTI
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        entries = make_h5([("%s-%d" % (name, i), dtype, s, nel, 0, seed + i)
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
            # several chunks in ONE call
            rc = decode(lib, dc, raws, e)
            if rc != 0:
                print("FAIL %s: multi rc=%d: %s" % (name, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            got = read_buf(lib, dc, "BshufOutput")
            want = b"".join(refs)
            if not buf_eq(got, want):
                print("FAIL %s: multi mismatch (got %d want %d bytes)" % (name, len(got), len(want)))
                return 1
            # one chunk per call, each must match its own dataset
            for i, (r, ref) in enumerate(zip(raws, refs)):
                rc = decode(lib, dc, [r], e)
                if rc != 0 or not buf_eq(read_buf(lib, dc, "BshufOutput"), ref):
                    print("FAIL %s: single chunk %d rc=%d" % (name, i, rc))
                    return 1
            print("ok   %s (e=%d, %d chunks in one call, out %d B, matches h5py)"
                  % (name, e, len(raws), len(want)))
            return 0
        finally:
            lib.sbz_dctx_destroy(dc)
    return 0


def run_corrupt(case):
    """Tamper a valid u16-default chunk and check a SBZ_ERR_* code, no crash/hang."""
    import numpy as np
    spec = SPECS[0]
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "build")) as td:
        (entry,) = make_h5([spec], td)
        _, dt, raw, ref, csz, n = entry
        e = np.dtype(dt).itemsize
        chunk = bytearray(raw[0])
        if case == "trunc":
            data = bytes(chunk[:-5])
        elif case == "flip":
            # corrupt the first block size record (u32 BE at offset 12)
            cs = struct.unpack_from(">I", chunk, 12)[0]
            chunk[12] ^= 0xFF  # make the size huge -> record past the end
            data = bytes(chunk)
        elif case == "badblock0":
            cs = struct.unpack_from(">I", chunk, 12)[0]
            struct.pack_into(">I", chunk, 12, max(1, cs - 1))  # frame too small overall
            data = bytes(chunk)
        else:
            print("FAIL corrupt %s: unknown case" % case)
            return 1
        lib = load()
        err = ct.create_string_buffer(512)
        dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
        if not dc:
            print("FAIL corrupt %s: create: %s" % (case, err.value.decode()))
            return 1
        try:
            arr = (Span * 1)(Span(0, len(data)))
            rc = lib.sbz_dctx_bshuf_decompress(dc, data, len(data), arr, 1, e)
            # must be an error, not 0; a success on corrupt input is also a failure
            if rc == 0 or rc not in (SBZ_ERR_CORRUPT, SBZ_ERR_ARGS, SBZ_ERR_LIMIT):
                print("FAIL corrupt %s: rc=%d (want a SBZ_ERR_*): %s"
                      % (case, rc, lib.sbz_dctx_error(dc).decode()))
                return 1
            # the context must stay usable
            arr2 = (Span * 1)(Span(0, len(raw[0])))
            rc2 = lib.sbz_dctx_bshuf_decompress(dc, bytes(raw[0]), len(raw[0]), arr2, 1, e)
            if rc2 != 0:
                print("FAIL corrupt %s: context broken after error: rc=%d" % (case, rc2))
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
        if nout == 0:
            print("FAIL h5 %s: zero bytes compared" % path)
            return 1
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
            print("ok   h5 %s: %d chunks (%d B compared) decoded in %.4f s (%.2f GB/s)"
                  % (os.path.basename(path), len(raw), nout, dt, gbs))
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
    if len(argv) == 2 and argv[0] == "--multi":
        return run_multi(argv[1])
    if len(argv) == 2 and argv[0] == "--corrupt":
        return run_corrupt(argv[1])
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    bad = 0
    for name, *_ in SPECS:
        try:
            r = subprocess.run([sys.executable, __file__, "--one", name], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL %s: timeout (%ds) -- GPU hang?" % (name, TIMEOUT))
            bad += 1
    try:
        r = subprocess.run([sys.executable, __file__, "--multi", MULTI[0]], timeout=TIMEOUT)
        bad += r.returncode != 0
    except subprocess.TimeoutExpired:
        print("FAIL %s: timeout (%ds) -- GPU hang?" % (MULTI[0], TIMEOUT))
        bad += 1
    for case in ("trunc", "flip", "badblock0"):
        try:
            r = subprocess.run([sys.executable, __file__, "--corrupt", case], timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL corrupt %s: timeout (%ds) -- GPU hang?" % (case, TIMEOUT))
            bad += 1
    examples = os.environ.get("SLABITZ_EXAMPLES") or os.path.join(ROOT, "workspace", "examples")
    paths = []
    for name in ("silicon_u16_zstd2_256k.h5", "ceo2_u32_zstd2_256k.h5"):
        path = os.path.join(examples, name)
        if os.path.exists(path):
            paths.append(path)
        else:
            bad += data_required(path)  # T02: REQUIRE_DATA turns the skip into a failure
    for path in paths:
        try:
            r = subprocess.run([sys.executable, __file__, "--h5", path, "--frames", "16"],
                               timeout=TIMEOUT)
            bad += r.returncode != 0
        except subprocess.TimeoutExpired:
            print("FAIL h5 %s: timeout (%ds) -- GPU hang?" % (path, TIMEOUT))
            bad += 1
    print("%s: %d case(s) failed" % ("FAIL" if bad else "PASS", bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
