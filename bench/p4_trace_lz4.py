"""SLABITZ_TRACE timeline of the async LZ4 pipeline: one thread, two contexts in the
primary context, pinned input, depth 2. Prints the merged device timeline of the last
calls and, per upload, how much of it ran alongside the other context's kernels.

  GZROOT=<repo> SLABITZ_KERNEL_DIR=<repo>/build python3 bench/p4_trace_lz4.py FILE.h5 [calls]

Run on an idle GPU (Slurm low-gpu job). 2026-11-20, CeO2: the two contexts'
uploads ran at the same time (sharing PCIe), then both contexts' kernels ran
together with PCIe idle: about upload + kernels/2 per call.
"""
import ctypes as ct
import os
import re
import subprocess
import sys

if os.environ.get("SLABITZ_TRACE") != "1":
    env = dict(os.environ, SLABITZ_TRACE="1")
    p = subprocess.run([sys.executable] + sys.argv, env=env, capture_output=True, text=True)
    sys.stdout.write(p.stdout)
    ev = []
    for line in p.stderr.splitlines():
        m = re.match(r"gztrace (\S+)\s+([\d.]+)\s+([\d.]+) (\S+)", line)
        if m:
            ev.append((float(m.group(2)), float(m.group(3)), m.group(1), m.group(4)))
        elif "Warn" not in line and line.strip():
            print("stderr:", line)
    ctxs = sorted({e[2] for e in ev})
    name = {c: "AB"[i] for i, c in enumerate(ctxs)}
    ev.sort()
    big = [e for e in ev if e[3] == "upload" and e[1] - e[0] > 500]  # the input uploads
    kern = [e for e in ev if e[3] not in ("upload", "download", "fill32")]
    print("\nlast 45 ops (us, ctx, op):")
    t0 = ev[-45][0]
    for a, b, c, n in ev[-45:]:
        print("  %9.1f %9.1f  %s  %-24s %8.1f" % (a - t0, b - t0, name[c], n, b - a))
    print("\ninput uploads: duration, and time overlapped by the OTHER context's kernels:")
    for a, b, c, n in big[2:]:
        ov = sum(max(0.0, min(b, kb) - max(a, ka)) for ka, kb, kc, kn in kern if kc != c)
        print("  %s upload %7.1f us, overlapped %7.1f us" % (name[c], b - a, ov))
    sys.exit(p.returncode)

ROOT = os.environ["GZROOT"]
sys.path.insert(0, os.path.join(ROOT, "tests"))
from run_bslz4_chain import load, find_chunked, Span  # noqa: E402
import h5py, hdf5plugin  # noqa: E402,F401

path = sys.argv[1]
CALLS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
BATCH = 16
with h5py.File(path, "r") as f:
    ds = find_chunked(f)
    e = ds.dtype.itemsize
    raw = [bytes(ds.id.read_direct_chunk((i, 0, 0))[1]) for i in range(BATCH)]
src = b"".join(raw)
offs = [0]
for r in raw:
    offs.append(offs[-1] + len(r))
spans = (Span * BATCH)(*[Span(offs[i], len(r)) for i, r in enumerate(raw)])
lib = load()
lib.sbz_dctx_bslz4_decompress_async.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                               ct.POINTER(Span), ct.c_uint32, ct.c_uint32]
lib.sbz_dctx_wait.argtypes = [ct.c_void_p]
cu = ct.CDLL("libcuda.so.1")
cu.cuInit(0)
cu.cuMemHostAlloc.argtypes = [ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_uint]
dev = ct.c_int()
cu.cuDeviceGet(ct.byref(dev), 0)
prim = ct.c_void_p()
assert cu.cuDevicePrimaryCtxRetain(ct.byref(prim), dev) == 0 and cu.cuCtxSetCurrent(prim) == 0
err = ct.create_string_buffer(512)
dcs = [lib.sbz_dctx_create_cuda(0, os.environ["SLABITZ_KERNEL_DIR"].encode(), err, 512)
       for _ in range(2)]
assert all(dcs), err.value
ptrs = []
for _ in dcs:
    p = ct.c_void_p()
    assert cu.cuMemHostAlloc(ct.byref(p), len(src), 0) == 0
    ct.memmove(p, src, len(src))
    ptrs.append(p.value)
for i in range(CALLS):
    k = i % 2
    if i >= 2:
        assert lib.sbz_dctx_wait(dcs[k]) == 0
    assert lib.sbz_dctx_bslz4_decompress_async(dcs[k], ptrs[k], len(src), spans, BATCH, e) == 0
for k in (0, 1):
    assert lib.sbz_dctx_wait(dcs[k]) == 0
for dc in dcs:
    lib.sbz_dctx_destroy(dc)
print("%s: %d calls traced" % (os.path.basename(path), CALLS))
