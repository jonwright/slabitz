"""Does decoding batch n overlap uploading batch n+1 with today's synchronous API?

Real files, 16 frames per call, input built once. Modes:
  serial : one sbz_dctx, calls back to back
  2ctx   : two threads, each creates its own sbz_dctx with NO current CUDA context
           (so each gets its own CUDA context: the GPU time-slices contexts)
  shared : two threads, both with the device's primary context current before
           sbz_dctx_create_cuda (one context, two streams)
  async  : one thread, two contexts in the primary context, alternated with
           sbz_dctx_bslz4_decompress_async (or _bshuf_ for zstd) / sbz_dctx_wait,
           input in ordinary (pageable) host memory
  pinned : as async, with the input in pinned host memory (cuMemHostAlloc), one
           buffer per context
  device : (LZ4) the input already in device memory (uploaded once, not timed):
           sbz_dctx_bslz4_decompress_device, two contexts alternated: the decode
           alone, what a caller gets when its own uploads overlap (P4 step 2c)
Timing starts after every thread is warm (barrier), and counts calls/second.

  GZROOT=<repo> SLABITZ_KERNEL_DIR=<repo>/build python3 bench/p4_overlap.py FILE.h5 [calls]

Time only on an idle GPU (check nvidia-smi; on the shared L40S node use a Slurm
GPU job). Results 2026-11-20: docs/p4_readback_audit.md, section 5.
"""
import ctypes as ct
import os
import sys
import threading
import time

ROOT = os.environ["GZROOT"]
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_bslz4_chain import load, find_chunked, Span  # noqa: E402
import h5py, hdf5plugin  # noqa: E402,F401

path = sys.argv[1]
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
BATCH = 16
KDIR = os.environ["SLABITZ_KERNEL_DIR"].encode()

with h5py.File(path, "r") as f:
    ds = find_chunked(f)
    e = ds.dtype.itemsize
    raw = [bytes(ds.id.read_direct_chunk((i, 0, 0))[1]) for i in range(BATCH)]
    fb = ds.shape[1] * ds.shape[2] * e
src = b"".join(raw)
offs = [0]
for r in raw:
    offs.append(offs[-1] + len(r))
spans = (Span * BATCH)(*[Span(offs[i], len(r)) for i, r in enumerate(raw)])
codec = "lz4" if "lz4" in os.path.basename(path) else "zstd"
lib = load()
lib.sbz_dctx_bshuf_decompress.argtypes = lib.sbz_dctx_bslz4_decompress.argtypes
fn = lib.sbz_dctx_bshuf_decompress if codec == "zstd" else lib.sbz_dctx_bslz4_decompress

cu = ct.CDLL("libcuda.so.1")
cu.cuInit(0)
dev = ct.c_int()
cu.cuDeviceGet(ct.byref(dev), 0)
cu.cuMemHostAlloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_uint]
cu.cuMemFreeHost.argtypes = [ct.c_void_p]
prim = ct.c_void_p()
assert cu.cuDevicePrimaryCtxRetain(ct.byref(prim), dev) == 0


def make(shared):
    if shared:
        assert cu.cuCtxSetCurrent(prim) == 0
    err = ct.create_string_buffer(512)
    dc = lib.sbz_dctx_create_cuda(0, KDIR, err, len(err))
    assert dc, err.value
    return dc


def call(dc):
    rc = fn(dc, src, len(src), spans, BATCH, e)
    assert rc == 0, lib.sbz_dctx_error(dc)


def async_mode(pinned, nctx=2):
    afn = lib.sbz_dctx_bslz4_decompress_async if codec == "lz4" else \
        lib.sbz_dctx_bshuf_decompress_async
    afn.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                    ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    dcs = [make(True) for _ in range(nctx)]
    bufs = []
    for _ in dcs:
        if pinned:
            p = ct.c_void_p()
            assert cu.cuMemHostAlloc(ct.byref(p), ct.c_size_t(len(src)), 0) == 0
            ct.memmove(p, src, len(src))
            bufs.append(p.value)
        else:
            b = ct.create_string_buffer(src, len(src))
            bufs.append(ct.addressof(b))
            bufs.append(b)  # keep alive
    ptrs = [b for b in bufs if isinstance(b, int)]

    tgo = [0.0]
    twait = [0.0]

    def go(k):
        t = time.perf_counter()
        assert afn(dcs[k], ptrs[k], len(src), spans, BATCH, e) == 0
        tgo[0] += time.perf_counter() - t

    def wait(k):
        t = time.perf_counter()
        rc = lib.sbz_dctx_wait(dcs[k])
        twait[0] += time.perf_counter() - t
        assert rc == 0, lib.sbz_dctx_error(dcs[k])

    for k in range(nctx):  # warm up
        go(k)
        wait(k)
    tgo[0] = twait[0] = 0.0
    t0 = time.perf_counter()
    for i in range(CALLS):
        k = i % nctx
        if i >= nctx:
            wait(k)
        go(k)
    for k in range(nctx):
        wait((CALLS + k) % nctx)
    dt = time.perf_counter() - t0
    for dc in dcs:
        lib.sbz_dctx_destroy(dc)
    if pinned:
        for p in ptrs:
            cu.cuMemFreeHost(ct.c_void_p(p))
    ASYNC_CALL[(pinned, nctx)] = (tgo[0] / CALLS, twait[0] / CALLS)
    return dt


ASYNC_CALL = {}


class Geom(ct.Structure):
    _fields_ = [("elemSize", ct.c_uint32), ("chunkBytes", ct.c_uint64),
                ("blockBytes", ct.c_uint32)]


def device_mode():
    import struct
    lib.sbz_dctx_bslz4_decompress_device.argtypes = [
        ct.c_void_p, ct.c_uint64, ct.c_size_t, ct.POINTER(Span), ct.c_uint32,
        ct.POINTER(Geom)]
    lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
    cu.cuMemAlloc_v2.argtypes = [ct.POINTER(ct.c_uint64), ct.c_size_t]
    cu.cuMemcpyHtoD_v2.argtypes = [ct.c_uint64, ct.c_char_p, ct.c_size_t]
    dcs = [make(True), make(True)]
    d = ct.c_uint64()
    assert cu.cuMemAlloc_v2(ct.byref(d), len(src)) == 0
    assert cu.cuMemcpyHtoD_v2(d.value, src, len(src)) == 0
    g = Geom(e, fb, struct.unpack(">I", raw[0][8:12])[0])

    def go(k):
        assert lib.sbz_dctx_bslz4_decompress_device(dcs[k], d.value, len(src), spans, BATCH,
                                                   ct.byref(g)) == 0

    def wait(k):
        assert lib.sbz_dctx_wait(dcs[k]) == 0, lib.sbz_dctx_error(dcs[k])

    for k in (0, 1):
        go(k)
        wait(k)
    t0 = time.perf_counter()
    for i in range(CALLS):
        k = i % 2
        if i >= 2:
            wait(k)
        go(k)
    wait(0)
    wait(1)
    dt = time.perf_counter() - t0
    for dc in dcs:
        lib.sbz_dctx_destroy(dc)
    return dt


def serial():
    dc = make(True)
    for _ in range(3):
        call(dc)
    t0 = time.perf_counter()
    for _ in range(CALLS):
        call(dc)
    dt = time.perf_counter() - t0
    lib.sbz_dctx_destroy(dc)
    return dt


def threaded(shared):
    bar = threading.Barrier(3)
    done = threading.Barrier(3)

    def run(idx):
        dc = make(shared)
        for _ in range(3):
            call(dc)
        bar.wait()
        for _ in range(idx, CALLS, 2):
            call(dc)
        done.wait()
        lib.sbz_dctx_destroy(dc)

    ts = [threading.Thread(target=run, args=(i,)) for i in (0, 1)]
    for t in ts:
        t.start()
    bar.wait()
    t0 = time.perf_counter()
    done.wait()
    dt = time.perf_counter() - t0
    for t in ts:
        t.join()
    return dt


print("%s: %s, %d frames/call, %.1f MB compressed/call, %.1f MB out/call"
      % (os.path.basename(path), codec, BATCH, len(src) / 1e6, BATCH * fb / 1e6))
base = serial()
modes = [("serial", base), ("2ctx", threaded(False)), ("shared", threaded(True))]
KEYS = {"async": (False, 2), "pinned": (True, 2), "pinned3": (True, 3)}
modes += [(n, async_mode(*k)) for n, k in KEYS.items()]
if codec == "lz4":
    modes.append(("device", device_mode()))
for name, dt in modes:
    extra = ""
    if name in KEYS:
        g, w = ASYNC_CALL[KEYS[name]]
        extra = "  (host per call: async call %.2f ms, wait %.2f ms)" % (g * 1e3, w * 1e3)
    print("  %-7s %7.2f ms/call  %6.0f frames/s  x%.2f%s"
          % (name, dt / CALLS * 1e3, CALLS * BATCH / dt, base / dt, extra))
