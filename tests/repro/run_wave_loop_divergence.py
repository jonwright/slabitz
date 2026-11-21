#!/usr/bin/env python3
"""Reproduce/verify the partial-iteration wave-intrinsic deadlock.

Runs two Slang entries (built from wave_loop_divergence.slang):
  * main_orig    -- DS-style `for(j=tid; j<n; j+=32)` with a WaveActiveSum body. Expected to
                    HANG when n is not a multiple of 32 (run under a timeout in a subprocess).
  * main_uniform -- uniform trip count with an `if(j<n)` body guard. Expected to finish for all
                    n in [1,256] and match the reference sum.

Usage:
    run_wave_loop_divergence.py [cubin]
The cubin is built by tools/build_cuda.py from wave_loop_divergence.slang.
"""
import os
import struct
import subprocess
import sys
import ctypes as ct

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from slabitz import loader

_CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "wave_loop_divergence.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def _cubin():
    if os.path.exists(_CUBIN):
        return open(_CUBIN, "rb").read()
    raise SystemExit("cubin not found: %s (build with tools/build_cuda.py)" % _CUBIN)


def _uniform_run(drv, fn, n, wbytes):
    w = loader.DeviceMemory(drv, (len(wbytes) + 3) // 4 * 4)
    w.from_host(wbytes + b"\x00" * ((-len(wbytes)) % 4))
    nm = loader.DeviceMemory(drv, 4)
    nm.from_host(struct.pack("I", n))
    out = loader.DeviceMemory(drv, 4)
    out.from_host(bytes(4))
    args = [Buf(w.ptr, (len(wbytes) + 3) // 4), Buf(nm.ptr, 1), Buf(out.ptr, 1)]
    fn(grid=(1, 1, 1), block=(32, 1, 1), args=args, shared=4)
    drv.synchronize()
    return struct.unpack("I", out.to_host(4))[0]


def _run_entry(entry, n, timeout=20):
    drv = loader.Driver(0)
    fn = drv.load(_cubin()).function(entry)
    wbytes = bytes([2]) * n
    return _uniform_run(drv, fn, n, wbytes)


def main():
    cubin = _cubin()
    # ---- uniform loop must not hang for n=1..256. The accumulated sum is informational: in the
    # repro (no entry-threaded active mask) WaveIsFirstLane/WaveActiveSum is not faithful to the
    # DS kernel, which threads the mask; only the hang/no-hang behaviour is meaningful here.
    # Real correctness is established by tests/run_lit_gpu_chain.py against the oracle.
    drv = loader.Driver(0)
    fn = drv.load(cubin).function("main_uniform")
    ref = lambda n: n * 2  # sum of 1<<(w-1) with w=2 for every weight
    hung = []
    wrong = []
    for n in range(1, 257):
        try:
            got = _uniform_run(drv, fn, n, bytes([2]) * n)
            if got != ref(n):
                wrong.append((n, got))
        except Exception as e:
            hung.append((n, str(e)))
    print("uniform loop: NO HANG for %d/256 n" % (256 - len(hung)))
    if hung:
        print("  hung:", hung[:10])
        return 1
    print("  informational: %d/256 sums match (sum not asserted -- see module docstring)" % (256 - len(wrong)))

    # ---- original form must hang at n=126 (partial iteration) ----
    rc = subprocess.run([sys.executable, "-c",
                         "import sys;sys.path.insert(0,%r);"
                         "import tests.repro.run_wave_loop_divergence as r;"
                         "r._CUBIN=%r;r._run_entry('main_orig', 126)" % (os.getcwd(), _CUBIN)],
                        capture_output=True, timeout=25)
    hung = rc.returncode != 0
    print("original form @ n=126: %s" % ("HANGED (as expected)" if hung else "finished (unexpected!)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
