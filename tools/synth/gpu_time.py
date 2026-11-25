#!/usr/bin/env python3
"""GPU decode time per frame of a filter-32008 HDF5 file, through slabitz.

    gpu-run python3 tools/synth/gpu_time.py FILE [FILE ...] [--frames 16] [--reps 9]

One call decodes `--frames` chunks (one frame per chunk), the way
bench/old_cards.py does: the measured time is upload + decode + unshuffle +
sync.  The median of `--reps` calls, in milliseconds per frame, is printed as
`TIME <file> <ms_per_frame> <compressed bytes per frame>`.  The first call is
checked byte for byte against what h5py reads.
"""
import argparse
import ctypes as ct
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests"))
KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))


def time_file(path, nframes, reps):
    import h5py
    import hdf5plugin  # noqa: F401
    from run_bshuf_chain import find_chunked
    with h5py.File(path, "r") as f:
        ds = find_chunked(f)
        opts = ds._filters.get("32008")
        lz4 = bool(opts and opts[4] == 2)
        n = min(nframes, ds.shape[0])
        es = ds.dtype.itemsize
        raw = [bytes(ds.id.read_direct_chunk((i,) + (0,) * (ds.ndim - 1))[1]) for i in range(n)]
        want = ds[0:n].tobytes()
    if lz4:
        from run_bslz4_chain import load, read_buf, Span
        entry = "sbz_dctx_bslz4_decompress"
    else:
        from run_bshuf_chain import load, read_buf, Span
        entry = "sbz_dctx_bshuf_decompress"
    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("create: %s" % err.value.decode())
    src = b"".join(raw)
    offs, pos = [], 0
    for c in raw:
        offs.append(Span(pos, len(c)))
        pos += len(c)
    spans = (Span * n)(*offs)
    fn = getattr(lib, entry)
    times = []
    try:
        for r in range(reps + 1):
            t0 = time.perf_counter()
            rc = fn(dc, src, len(src), spans, n, es)
            dt = time.perf_counter() - t0
            if rc != 0:
                raise SystemExit("decode rc %d" % rc)
            if r == 0:
                got = bytes(read_buf(lib, dc, "BshufOutput")[:len(want)])
                if got != want:
                    raise SystemExit("%s: GPU output differs from h5py" % path)
            else:
                times.append(dt)
    finally:
        lib.sbz_dctx_destroy(dc)
    return statistics.median(times) / n * 1e3, len(src) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    for p in a.files:
        ms, cb = time_file(p, a.frames, a.reps)
        print("TIME %s %.4f %.0f" % (p, ms, cb), flush=True)


if __name__ == "__main__":
    main()
