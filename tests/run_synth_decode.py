#!/usr/bin/env python3
"""Decode the synthetic data through slabitz and check it byte for byte.

The committed subset (tests/data/synth, listed with sha256 of the expected
decoded bytes in MANIFEST.json): every chunk of every .h5 goes through the GPU
decoder (sbz_dctx_bslz4_decompress or sbz_dctx_bshuf_decompress, by the
filter's codec) and every .zst through sbz_dctx_decompress.  The output is
compared with the expected sha256 and, for .h5, with what hdf5plugin reads.

With $SLABITZ_SYNTH set, every .h5 and .zst of the full set (twins, ext, worst)
is decoded the same way and compared with the dataset read by h5py, or with
raw.bin of the case.

The GPU decoder rejects some valid zstd (see AGENTS.md, known limits); a case
listed in EXPECTED_REJECT must be rejected with an SBZ_ERR code, and fails the
test if the decoder accepts it or crashes.  Compared byte counts are printed
and must be nonzero.
"""
import ctypes as ct
import glob
import subprocess
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "data", "synth")
LIB = os.path.join(ROOT, "build", "libslabitz.so")
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))
SYNTH = os.environ.get("SLABITZ_SYNTH", "")

# A case whose manifest says expected_reject must be rejected with an SBZ_ERR
# code (known limits: frames without Frame_Content_Size, T09/T10).


class Span(ct.Structure):
    _fields_ = [("offs", ct.c_uint64), ("size", ct.c_uint64)]


def new_ctx(lib):
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("FAIL: cannot create the decoder context: %s" % err.value.decode())
    return dc


class Ctx:
    """One decoder context, reused; a failed decode leaves it unusable, so a
    failure replaces it."""

    def __init__(self, lib):
        self.lib, self.dc = lib, None

    def get(self):
        if self.dc is None:
            self.dc = new_ctx(self.lib)
        return self.dc

    def reset(self):
        if self.dc is not None:
            self.lib.sbz_dctx_destroy(self.dc)
            self.dc = None


def load():
    lib = ct.CDLL(LIB)
    lib.sbz_dctx_create_cuda.restype = ct.c_void_p
    lib.sbz_dctx_create_cuda.argtypes = [ct.c_int, ct.c_char_p, ct.c_char_p, ct.c_size_t]
    lib.sbz_dctx_destroy.argtypes = [ct.c_void_p]
    lib.sbz_dctx_decompress.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t]
    for name in ("sbz_dctx_bshuf_decompress", "sbz_dctx_bslz4_decompress"):
        getattr(lib, name).argtypes = [ct.c_void_p, ct.c_char_p, ct.c_size_t, ct.POINTER(Span),
                                       ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_read_buffer.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_void_p, ct.POINTER(ct.c_size_t)]
    return lib


def read_buf(lib, dc, name, n_want):
    n = ct.c_size_t(0)
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), None, ct.byref(n)) == 0
    b = ct.create_string_buffer(max(n.value, 1))
    assert lib.sbz_dctx_read_buffer(dc, name.encode(), b, ct.byref(n)) == 0
    return b.raw[:min(n.value, n_want)]


def codec_of(ds):
    opts = ds._filters.get("32008")
    return "lz4" if opts and opts[4] == 2 else "zstd"


def chunks_of(ds):
    """(raw chunk bytes, expected decoded bytes) for every chunk of a dataset."""
    out = []
    for k in range(ds.id.get_num_chunks()):
        info = ds.id.get_chunk_info(k)
        _, raw = ds.id.read_direct_chunk(info.chunk_offset)
        sel = tuple(slice(o, min(o + c, s)) for o, c, s in zip(info.chunk_offset, ds.chunks, ds.shape))
        out.append((bytes(raw), np.ascontiguousarray(ds[sel]).tobytes()))
    return out


def gpu_chunks(lib, ds):
    """Decode all chunks of ds with one call per chunk; return (bytes compared, failures)."""
    import h5py  # noqa: F401
    codec = codec_of(ds)
    fn = lib.sbz_dctx_bslz4_decompress if codec == "lz4" else lib.sbz_dctx_bshuf_decompress
    es = ds.dtype.itemsize
    n, bad = 0, []
    for i, (raw, want) in enumerate(chunks_of(ds)):
        dc = CTX.get()
        sp = (Span * 1)(Span(0, len(raw)))
        rc = fn(dc, raw, len(raw), sp, 1, es)
        if rc != 0:
            bad.append("chunk %d: rc %d" % (i, rc))
            CTX.reset()
            continue
        got = read_buf(lib, dc, "BshufOutput", len(want))
        if got != want:
            bad.append("chunk %d: output differs" % i)
        n += len(want)
    return n, bad


def check_h5(lib, path, expect, failures):
    import h5py
    import hdf5plugin  # noqa: F401
    total = 0
    with h5py.File(path, "r") as f:
        for name in f:
            ds = f[name]
            want_sha = None
            ref = hashlib.sha256(np.ascontiguousarray(ds[()]).tobytes()).hexdigest()
            if expect:
                e = expect.get("datasets", {}).get(name) if "datasets" in expect else expect.get("raw_sha256")
                if e and e != ref:
                    failures.append("%s/%s: hdf5plugin output != manifest sha256" % (os.path.basename(path), name))
            n, bad = gpu_chunks(lib, ds)
            total += n
            for b in bad:
                failures.append("%s/%s: %s" % (os.path.basename(path), name, b))
    return total


def check_zst(lib, path, want, failures, expect_reject=False):
    name = os.path.basename(path)
    z = open(path, "rb").read()
    if expect_reject:
        # In a subprocess: a rejected frame has been seen to fault the CUDA
        # context (error 700), which would poison every later case.
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--reject", path],
                           capture_output=True, text=True, timeout=300)
        print("  %s: %s" % (name, r.stdout.strip().replace("\n", "; ")))
        if r.returncode != 0:
            failures.append("%s: expected a reject, got: %s %s" % (name, r.stdout.strip(), r.stderr.strip()[-200:]))
        return 0
    dc = CTX.get()
    rc = lib.sbz_dctx_decompress(dc, z, len(z))
    got = read_buf(lib, dc, "UnCompressedFramesData", len(want)) if rc == 0 else b""
    if rc != 0:
        CTX.reset()
    if rc != 0:
        failures.append("%s: sbz_dctx_decompress rc %d" % (name, rc))
        return 0
    if got != want:
        failures.append("%s: decoded bytes differ" % name)
    return len(want)


def subset(lib, failures):
    import subprocess
    man = json.load(open(os.path.join(DATA, "MANIFEST.json")))
    total = 0
    for name in sorted(man):
        path = os.path.join(DATA, name)
        ent = man[name]
        if name.endswith(".h5"):
            n = check_h5(lib, path, ent, failures)
        else:
            z = open(path, "rb").read()
            ref = subprocess.run(["/bin/zstd", "-d", "-q", "-c"], input=z, capture_output=True)
            if ref.returncode != 0 or hashlib.sha256(ref.stdout).hexdigest() != ent["raw_sha256"]:
                failures.append("%s: zstd -d output != manifest sha256" % name)
                continue
            n = check_zst(lib, path, ref.stdout, failures, ent.get("expected_reject", False))
        total += n
        print("%-44s %9d bytes %s" % (name, n, "" if n or not ent.get("expected_reject") else "(rejected, as expected)"))
    return total


def full(lib, failures):
    total = 0
    for sub in ("twins", "ext", "worst"):
        base = os.path.join(SYNTH, sub)
        for path in sorted(glob.glob(os.path.join(base, "*", "*.h5"))):
            total += check_h5(lib, path, None, failures)
            print("%-60s ok so far" % os.path.relpath(path, SYNTH))
        for path in sorted(glob.glob(os.path.join(base, "*", "data.zst"))):
            raw = os.path.join(os.path.dirname(path), "raw.bin")
            if os.path.exists(raw):
                man = json.load(open(os.path.join(os.path.dirname(path), "manifest.json")))
                total += check_zst(lib, path, open(raw, "rb").read(), failures, man.get("expected_reject", False))
    return total


CTX = None


def reject_main(path):
    """Child: the frame must be rejected with an SBZ_ERR code (rc < 0).  Also
    reports whether the CUDA context survived the rejection."""
    lib = load()
    z = open(path, "rb").read()
    dc = new_ctx(lib)
    rc = lib.sbz_dctx_decompress(dc, z, len(z))
    lib.sbz_dctx_destroy(dc)
    state = "usable"
    try:
        lib.sbz_dctx_destroy(new_ctx(lib))
    except SystemExit as e:
        state = "FAULTED (%s)" % str(e).split(": ")[-1]
    print("rc %d, CUDA context after the rejection: %s" % (rc, state))
    return 0 if rc < 0 else 1


def main():
    global CTX
    lib = load()
    CTX = Ctx(lib)
    failures = []
    total = subset(lib, failures)
    if SYNTH:
        total += full(lib, failures)
    elif os.environ.get("SLABITZ_REQUIRE_DATA") == "1":  # T02
        failures.append("data required but SLABITZ_SYNTH is not set: the full set was not run")
    else:
        print("SLABITZ_SYNTH not set: the full set was not run")
    CTX.reset()
    print("\ncompared bytes: %d" % total)
    if total == 0:
        failures.append("compared zero bytes")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  " + f)
        return 1
    print("all decodes passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reject":
        sys.exit(reject_main(sys.argv[2]))
    sys.exit(main())
