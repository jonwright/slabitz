#!/usr/bin/env python3
"""P3: the fast warp bit-unshuffle Slang kernel vs the bitshuffle library.

The warp kernel unshuffles whole blocks whose element count is a multiple of 32
(full blocks only; the partial last block is the slow kernel's job, and the raw
tail is handled by sbz_dctx_bshuf_decompress). So each case here uses N = nfull*B
elements with B a multiple of 32, e in {1,2,4}, expects the whole N-element block
set back untransposed, and checks the in/out byte offsets too.

Usage: python3 tests/run_unshuffle_warp.py [build/unshufw/unshufw.cubin]
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

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/unshufw/unshufw.cubin")


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


WARPS = 8  # warps per thread group (256 threads), as in kernels/BitUnshuffleWarp.slang / sbz_chain.c


def case(drv, fn, dtype, nfull, B, in_off, out_off, seed):
    e = np.dtype(dtype).itemsize
    block_bytes = B * e
    n = nfull * B
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 1 << min(8 * e, 62), n, dtype=np.uint64).astype(dtype)
    t = bitshuffle.bitshuffle(a, B).view(np.uint8)
    body = n * e
    inp = t.tobytes()
    want = a.view(np.uint8).tobytes()

    # place input at in_off, output at out_off (so the offsets are exercised).
    # the kernel now takes a block table (uint2 {inOff, outOff} per block), so build
    # one entry per block.
    inbuf = bytearray(in_off) + inp
    outbuf = bytearray(body + out_off)
    dIn = loader.DeviceMemory(drv, max(len(inbuf), 4))
    dIn.from_host(bytes(inbuf))
    dOut = loader.DeviceMemory(drv, max(len(outbuf), 4))
    dOut.from_host(b"\0" * dOut.size)
    block_bytes = B * e
    tbl = np.array([(in_off + b * block_bytes, out_off + b * block_bytes)
                    for b in range(nfull)], dtype=np.uint32).tobytes()
    dTbl = loader.DeviceMemory(drv, max(len(tbl), 4))
    dTbl.from_host(tbl)

    # the kernel writes element data starting at each block's outOff
    # parameter order of kernels/BitUnshuffleWarp.slang: the byte views, then the uint32
    # and uint4 views of the same memory (counts in their own elements), then the block
    # table (uint2 entries), then scalars
    args = [Buf(dIn.ptr, len(inbuf)), Buf(dIn.ptr, len(inbuf) // 4),
            Buf(dOut.ptr, len(outbuf)), Buf(dOut.ptr, len(outbuf) // 4),
            Buf(dOut.ptr, len(outbuf) // 16), Buf(dTbl.ptr, len(tbl) // 8),
            ct.c_uint32(e), ct.c_uint32(B), ct.c_uint32(nfull), ct.c_uint32(0)]
    groups = B // 32
    nslices = (groups + 31) // 32
    kparams.check_args("BitUnshuffleWarp", args)
    fn(grid=((nfull * nslices + WARPS - 1) // WARPS, 1, 1), block=(WARPS * 32, 1, 1), args=args)
    drv.synchronize()
    got = dOut.to_host(len(outbuf))[out_off:out_off + body]
    ok = got == want and len(got) == body and body > 0
    TOTAL[0] += len(got)
    print("%s %-7s n=%-8d block=%-6d B=%-6d nfull=%-3d inOff=%d outOff=%d" % (
        "ok  " if ok else "FAIL", np.dtype(dtype).name, n, block_bytes, B, nfull, in_off, out_off))
    return ok


TOTAL = [0]


def main():
    drv = loader.Driver()
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")
    good = True
    for dtype in (np.uint8, np.uint16, np.uint32):
        e = np.dtype(dtype).itemsize
        for bb, nfull in ((8192, 12), (253952, 2), (262144, 5), (8192, 1)):
            good &= case(drv, fn, dtype, nfull, bb // e, 0, 0, nfull + e + bb)
        # small / partial blocks (multiples of 32 elements): 1, 3, 33 groups,
        # i.e. a partial slice, a full slice plus one group, several blocks
        for B, nfull in ((32, 1), (96, 1), (96, 7), (1056, 1), (1056, 3), (2048, 9)):
            good &= case(drv, fn, dtype, nfull, B, 0, 0, B + nfull + e)
        # offsets: 4-aligned, 8-aligned, and unaligned in/out (byte paths)
        for in_off, out_off in ((0, 4), (8, 0), (11, 7), (3, 1), (4, 2), (13, 0), (0, 3)):
            good &= case(drv, fn, dtype, 3, 8192 // e, in_off, out_off, in_off * 16 + out_off)
            good &= case(drv, fn, dtype, 2, 1056, in_off, out_off, in_off * 16 + out_off + 1)
    # an offset case (both in and out), e=2 248 kB block
    good &= case(drv, fn, np.uint16, 3, 124 * 1024, 11, 7, 42)
    print("compared %d bytes" % TOTAL[0])
    good &= TOTAL[0] > 0
    print("PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
