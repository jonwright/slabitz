#!/usr/bin/env python3
"""P3: the general (slow) bit-unshuffle Slang kernel vs the bitshuffle library.

The input is bitshuffle.bitshuffle(a, block) without its raw tail (the S % 8
trailing elements, stored untransposed); expected = a without the tail. Covers
8 kB and 248 kB blocks, e = 1/2/4/8, a last block whose element count is not
a multiple of 32, and lengths that leave a raw tail.

Usage: python3 tests/run_unshuffle_slow.py [build/unshuf/unshuf.cubin]
"""
import ctypes as ct
import os
import sys

import bitshuffle
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader  # noqa: E402
import kparams  # tests/kparams.py

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/unshuf/unshuf.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def case(drv, fn, dtype, n, block_bytes, seed):
    e = np.dtype(dtype).itemsize
    B = block_bytes // e
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << min(8 * e, 62), n, dtype=np.uint64).astype(dtype)
    t = bitshuffle.bitshuffle(a, B).view(np.uint8)
    full, rem = n // B, n % B
    last = rem - rem % 8
    body = (full * B + last) * e
    inp = t[:body].tobytes()
    want = a.view(np.uint8)[:body].tobytes()

    dIn = loader.DeviceMemory(drv, max(len(inp), 4))
    dIn.from_host(inp + b"\0" * (dIn.size - len(inp)))
    dOut = loader.DeviceMemory(drv, max(len(want), 4))
    # the kernel now takes a block table (uint2 {inOff, outOff} per block). The full
    # blocks and the partial last block have different element counts, so they go in
    # two launches (as the host does: one bucket per (kernel, blockElems)).
    full_elems = full * B
    if full > 0:
        tbl = np.array([(b * B * e, b * B * e) for b in range(full)],
                       dtype=np.uint32).tobytes()
        dTbl = loader.DeviceMemory(drv, max(len(tbl), 4))
        dTbl.from_host(tbl)
        # parameter order of kernels/BitUnshuffleSlow.slang
        args = [Buf(dIn.ptr, len(inp)), Buf(dOut.ptr, len(want)), Buf(dTbl.ptr, len(tbl) // 8),
                ct.c_uint32(e), ct.c_uint32(B), ct.c_uint32(full), ct.c_uint32(0)]
        kparams.check_args("BitUnshuffleSlow", args)
        fn(grid=(max(1, (full_elems + 255) // 256), 1, 1), block=(256, 1, 1), args=args)
    if last > 0:
        tblog = (full_elems * e).to_bytes(4, "little") * 2
        dTbl2 = loader.DeviceMemory(drv, max(len(tblog), 4))
        dTbl2.from_host(tblog)
        args2 = [Buf(dIn.ptr, len(inp)), Buf(dOut.ptr, len(want)), Buf(dTbl2.ptr, 1),
                 ct.c_uint32(e), ct.c_uint32(last), ct.c_uint32(1), ct.c_uint32(0)]
        kparams.check_args("BitUnshuffleSlow", args2)
        fn(grid=(max(1, (last + 255) // 256), 1, 1), block=(256, 1, 1), args=args2)
    drv.synchronize()
    got = dOut.to_host(len(want))
    ok = got == want
    print("%s %-7s n=%-7d block=%-6d B full=%d last=%d (last%%32=%d) tail=%d" % (
        "ok  " if ok else "FAIL", np.dtype(dtype).name, n, block_bytes, full, last, last % 32,
        n % 8))
    return ok


def main():
    drv = loader.Driver()
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")
    good = True
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        for n, bb in ((100000, 8192), (300001, 253952), (5000, 253952), (65543, 8192)):
            good &= case(drv, fn, dtype, n, bb, n + np.dtype(dtype).itemsize)
    print("PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
