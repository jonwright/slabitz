"""Time the bslz4_to_sparse CUDA reference decoder (the target for slabitz's LZ4 path).

The reference is chunk2sparse(mask, dtype).decompress(chunks): upload, unblock, LZ4 decode with
the register unshuffle fused in, output left on the GPU. (The top-level bslz4_to_sparse.decompress
is the CPU decoder.) Its driver backend needs only libcuda and the prebuilt fatbin: no cupy, no
venv; it runs in this workspace's pixi python.

Set-up: a package of symlinks to the reference's sources plus its built extension:

  B=/path/to/bslz4_to_sparse; P=scratch_work/pkg/bslz4_to_sparse; mkdir -p $P
  for f in $B/src/*; do ln -sfn $f $P/; done
  ln -sfn $B/build/lib.linux-x86_64-cpython-312/bslz4_to_sparse/_bslz4_to_sparse.c2py23-linux_x86_64.so $P/
  PYTHONPATH=scratch_work/pkg BSLZ4_CUDA_BACKEND=driver python3 bench/ref_bslz4_cuda.py FILE.h5 \
      [--frames 1,4,16] [--reps 40] [--kernels] [--check 0]

--kernels: per-kernel device time from CUDA events around each launch (ncu needs performance
counters, admin-only on the L40S node). The "unblock" time excludes the upload: the upload is
queued before its start event. "decompress" is the fused decode + unshuffle to compare with.
"""
import argparse
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

import bslz4_to_sparse.cuda as bc


def find_ds(f):
    out = []
    f.visititems(lambda n, o: out.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
    return out[0]


def install_kernel_timer():
    """Wrap the driver backend's kernel launch in CUDA events (ncu needs perf counters, which
    are admin-only on this node: RmProfilingAdminOnly=1).  Returns {kernel: [ms, ...]}."""
    import collections
    import ctypes
    from bslz4_to_sparse.cuda import _backend_drv as d
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
    times = collections.defaultdict(list)
    orig = d._Kernel.__call__

    def call(self, grid, block, args, shared_mem=0):
        names = {v: k for k, v in d._Module._names.items() if isinstance(v, str)}
        e0, e1 = d._P(), d._P()
        d._check(d.cuEventCreate(ctypes.byref(e0), 0))
        d._check(d.cuEventCreate(ctypes.byref(e1), 0))
        s = d._current_stream()
        d._check(d.cuEventRecord(e0, s))
        orig(self, grid, block, args, shared_mem)
        d._check(d.cuEventRecord(e1, s))
        lib.cuEventSynchronize(e1)
        ms = ctypes.c_float()
        lib.cuEventElapsedTime(ctypes.byref(ms), e0, e1)
        times[names.get(self.symbol.decode(), self.symbol.decode())].append(ms.value)
        d.cuEventDestroy(e0)
        d.cuEventDestroy(e1)

    d._Kernel.__call__ = call
    return times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5")
    p.add_argument("--frames", default="1,4,16")
    p.add_argument("--reps", type=int, default=40)
    p.add_argument("--check", type=int, default=1)
    p.add_argument("--kernels", action="store_true", help="per-kernel device time (CUDA events)")
    a = p.parse_args()
    timed = install_kernel_timer() if a.kernels else None
    with h5py.File(a.h5, "r") as f:
        name = find_ds(f)
        ds = f[name]
        nmax = max(int(x) for x in a.frames.split(","))
        chunks = [np.frombuffer(ds.id.read_direct_chunk((i, 0, 0))[1], np.uint8) for i in range(nmax)]
        ref = ds[:nmax] if a.check else None
        shape, dtype = ds.shape, ds.dtype
    fbytes = shape[1] * shape[2] * dtype.itemsize
    print("%s %s %s %s, %.2f MB/frame, backend %s, %s"
          % (a.h5, name, shape, dtype, fbytes / 1e6, bc.backend, bc.device_name(0)))
    dec = bc.chunk2sparse(np.zeros(shape[1:], np.uint8), dtype=dtype)
    for nf in (int(x) for x in a.frames.split(",")):
        sub = chunks[:nf]
        out = dec.decompress(sub)  # warm up (module load, allocations)
        if ref is not None:
            got = np.asarray(out.get())
            n = got.size
            assert n > 0 and np.array_equal(got.reshape(ref[:nf].shape), ref[:nf]), "mismatch"
            print("  %2d frames: checked %d pixels vs h5py" % (nf, n))
        del out
        t = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            out = dec.decompress(sub)  # complete when returned (per the docstring)
            t.append(time.perf_counter() - t0)
            del out
        t = np.array(t)
        if timed is not None:
            tot = 0.0
            for k, v in timed.items():
                us = 1e3 * np.median(v[-a.reps * (len(v) // (a.reps + 1)):] if len(v) > a.reps else v)
                calls = len(v) / (a.reps + 1)
                tot += us * calls
                print("    kernel %-28s %8.1f us/launch x %.0f/call = %8.1f us/call, %6.1f us/frame"
                      % (k, us, calls, us * calls, us * calls / nf))
            print("    kernels total %.1f us/call, %.1f us/frame" % (tot, tot / nf))
            timed.clear()
        print("  %2d frames/call: median %.1f us/call, %.1f us/frame, %.1f GB/s (min %.1f us/call)"
              % (nf, np.median(t) * 1e6, np.median(t) * 1e6 / nf, nf * fbytes / np.median(t) / 1e9,
                 t.min() * 1e6))


if __name__ == "__main__":
    main()
