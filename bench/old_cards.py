#!/usr/bin/env python3
"""Is GPU decode worth it on this machine? Times both ways of getting frames into GPU memory.

  CPU path: decode on the CPU (h5py + hdf5plugin), then copy the pixels over PCIe.
  GPU path: copy the compressed chunks, decode on the GPU (sbz_dctx_bshuf_decompress;
            the call includes the upload of the compressed bytes, output stays on the GPU).

For one bitshuffle-zstd HDF5 dataset (one frame per chunk) it prints, per frame: CPU
decode time (1 process and N processes), host->device copy time of the decoded frame
(pageable and pinned), and GPU path time for batches of frames. Run it on each machine:

  SLABITZ_KERNEL_DIR=build/sm_35 python3 bench/old_cards.py FILE.h5 [--frames 32] [--procs N]
"""
import argparse
import ctypes as ct
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_bshuf_chain import find_chunked  # noqa: E402

KDIR = os.environ.get("SLABITZ_KERNEL_DIR", os.path.join(ROOT, "build"))


def lz4_chain():
    """The bslz4 test harness (load/decode/read_buf/find_chunked/Span) + entry."""
    from run_bslz4_chain import load, decode, read_buf, find_chunked, Span  # noqa: F401
    return load, decode, read_buf, find_chunked, "sbz_dctx_bslz4_decompress", Span


def zstd_chain():
    """The bshuf-zstd test harness + its entry name."""
    from run_bshuf_chain import load, decode, read_buf, find_chunked, Span  # noqa: F401
    return load, decode, read_buf, find_chunked, "sbz_dctx_bshuf_decompress", Span


def chain_for(ds):
    """Pick the harness for a dataset's bitshuffle compression (filter 32008)."""
    import hdf5plugin  # noqa: F401
    opts = ds._filters.get('32008')
    # the compression code is the 5th option (2 = lz4, 3 = zstd)
    return lz4_chain() if (opts and opts[4] == 2) else zstd_chain()


def prep_calls(entry, lib, dc, raw, batch, e, Span):
    """Join each call's chunks once into one contiguous src (a real caller keeps
    the compressed data in one buffer), returning [(src, spans, nchunks)]. The
    host-side join is done here, before the timed loop, so it is not measured."""
    calls = []
    for i in range(0, len(raw), batch):
        bt = raw[i:i + batch]
        src = b"".join(bt)
        offs, cum = [], 0
        for c in bt:
            offs.append((cum, len(c)))
            cum += len(c)
        spans = (Span * len(bt))(*[Span(o, l) for o, l in offs])
        calls.append((src, spans, len(bt)))
    return calls


def run_calls(entry, lib, dc, calls, e):
    for src, spans, nc in calls:
        rc = getattr(lib, entry)(dc, src, len(src), spans, nc, e)
        if rc != 0:
            raise SystemExit("decode: %s" % lib.sbz_dctx_error(dc).decode())



def cpu_worker(args):
    path, idx = args
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    with h5py.File(path, "r") as f:
        ds = find_chunked(f)
        out = np.empty(ds.shape[1:], ds.dtype)
        t0 = time.perf_counter()
        for i in idx:
            ds.read_direct(out, np.s_[i])
        return time.perf_counter() - t0


def cpu_rate(path, nframes, procs):
    """Seconds per frame decoding nframes with `procs` processes (page cache warm)."""
    cpu_worker((path, range(nframes)))  # warm the page cache and the plugin
    if procs == 1:
        return cpu_worker((path, range(nframes))) / nframes
    reps = max(1, (procs * 4) // nframes)  # give every process work
    work = [(path, [i % nframes for i in range(p, nframes * reps, procs)]) for p in range(procs)]
    with mp.Pool(procs) as pool:
        pool.map(cpu_worker, work[:procs])  # start-up
        t0 = time.perf_counter()
        pool.map(cpu_worker, work)
        return (time.perf_counter() - t0) / (nframes * reps)


def h2d_rate(nbytes, reps=20):
    """(pageable, pinned) seconds for one host->device copy of nbytes."""
    cu = ct.CDLL(os.environ.get("CUDA_DRIVER", "libcuda.so.1"))
    ctx, dptr, hptr = ct.c_void_p(), ct.c_uint64(), ct.c_void_p()
    assert cu.cuInit(0) == 0
    dev = ct.c_int()
    cu.cuDeviceGet(ct.byref(dev), 0)
    assert cu.cuCtxCreate_v2(ct.byref(ctx), 0, dev) == 0
    assert cu.cuMemAlloc_v2(ct.byref(dptr), ct.c_size_t(nbytes)) == 0
    page = ct.create_string_buffer(nbytes)
    assert cu.cuMemAllocHost_v2(ct.byref(hptr), ct.c_size_t(nbytes)) == 0
    out = []
    for src in (ct.cast(page, ct.c_void_p), hptr):
        cu.cuMemcpyHtoD_v2(dptr, src, ct.c_size_t(nbytes))
        cu.cuCtxSynchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            cu.cuMemcpyHtoD_v2(dptr, src, ct.c_size_t(nbytes))
        cu.cuCtxSynchronize()
        out.append((time.perf_counter() - t0) / reps)
    cu.cuMemFreeHost(hptr)
    cu.cuMemFree_v2(dptr)
    cu.cuCtxDestroy_v2(ctx)
    return out


def gpu_rate(path, batch, total):
    """Seconds per frame through the GPU decode, `batch` frames per call, one frame
    per chunk. The host-side join of each batch is done once before the timed loop;
    the measured cost is upload + decode + unshuffle + sync (all on the GPU)."""
    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as f:
        ds = find_chunked(f)
        e = ds.dtype.itemsize
        load, decode, read_buf, _, entry, Span = chain_for(ds)
        raw = [bytes(ds.id.read_direct_chunk((i, 0, 0))[1]) for i in range(total)]
        want0 = ds[0:batch].tobytes()
    lib = load()
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR.encode(), err, len(err))
    if not dc:
        raise SystemExit("create: %s" % err.value.decode())
    try:
        calls = prep_calls(entry, lib, dc, raw[:batch], batch, e, Span)
        run_calls(entry, lib, dc, calls, e)
        got = read_buf(lib, dc, "BshufOutput")
        if bytes(got[:len(want0)]) != want0:
            raise SystemExit("GPU output differs from h5py (batch %d)" % batch)
        calls = prep_calls(entry, lib, dc, raw, batch, e, Span)
        t0 = time.perf_counter()
        run_calls(entry, lib, dc, calls, e)
        return (time.perf_counter() - t0) / (len(calls) * batch)
    finally:
        lib.sbz_dctx_destroy(dc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("h5")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--batches", default="1,4,16")
    a = ap.parse_args()

    import h5py
    import hdf5plugin  # noqa: F401
    with h5py.File(a.h5, "r") as f:
        ds = find_chunked(f)
        n = min(a.frames, ds.shape[0])
        fbytes = ds.dtype.itemsize * (ds.size // ds.shape[0])
        cbytes = sum(ds.id.get_chunk_info(i).size for i in range(n)) / n
    ms = lambda s: "%7.2f ms" % (s * 1e3)  # noqa: E731
    gbs = lambda s: "%6.2f GB/s" % (fbytes / s / 1e9)  # noqa: E731
    print("%s: %d frames, %.2f MB/frame, ratio %.0f, %d CPU cores"
          % (os.path.basename(a.h5), n, fbytes / 1e6, fbytes / cbytes, os.cpu_count()))

    c1 = cpu_rate(a.h5, n, 1)
    cn = cpu_rate(a.h5, n, a.procs)
    pg, pin = h2d_rate(fbytes)
    print("CPU decode, 1 process   %s  %s" % (ms(c1), gbs(c1)))
    print("CPU decode, %2d procs    %s  %s" % (a.procs, ms(cn), gbs(cn)))
    print("H2D copy, pageable      %s  %s" % (ms(pg), gbs(pg)))
    print("H2D copy, pinned        %s  %s" % (ms(pin), gbs(pin)))
    print("CPU path (1 proc + pageable)   %s per frame" % ms(c1 + pg))
    print("CPU path (%d procs + pinned)   %s per frame" % (a.procs, ms(cn + pin)))
    for b in [int(x) for x in a.batches.split(",") if int(x) <= n]:
        g = gpu_rate(a.h5, b, n)
        print("GPU path, %2d frames/call %s  %s per frame" % (b, ms(g), gbs(g)))


if __name__ == "__main__":
    sys.exit(main())
