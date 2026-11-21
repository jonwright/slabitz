#!/usr/bin/env python3
"""GPU-vs-CPU comparison for the ComputePrefixSum (literal-group ends) kernel.

Builds nothing; expects a prebuilt cubin (tools/build_all.py -> build/cps/cps.cubin).
For each Huffman-lit table id computes the group count from its lit-stream count,
then the inclusive group prefix (decoupled-lookback == mathematical oracle). Also
checks the last thread's DecompressLiteralsGroups counter update.

Usage:
    PYTHONPATH=$PWD python3 tests/run_cps_chain.py build/cps/cps.cubin
"""
import ctypes as ct
import random
import struct
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from slabitz import loader
from kernels.cpu_ref_initresources import _FIELD_ORDER

CUBIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build/cps/cps.cubin")
TG = 32
UNUSED = 0xFFFFFFFF


class Buf(ct.Structure):
    _fields_ = [("data", ct.c_uint64), ("count", ct.c_uint64)]


def dm(drv, src):
    m = loader.DeviceMemory(drv, len(src) * 4)
    m.from_host(struct.pack(f"{len(src)}I", *src))
    return m


def cpu_ref(hufw_to_huflit, huflit_to_stream, huf_lit, huf_streams, literals_per_group):
    """inclusive group prefix; returns list and the last group-end (== DecompressLiteralsGroups)."""
    res = []
    acc = 0
    for huf_lit_id in hufw_to_huflit:
        cnt = 0
        if UNUSED != huf_lit_id:
            start = huflit_to_stream[huf_lit_id]
            if huf_lit_id + 1 < huf_lit:
                cnt = huflit_to_stream[huf_lit_id + 1] - start
            else:
                cnt = huf_streams - start
        g = (cnt + literals_per_group - 1) // literals_per_group
        acc += g
        res.append(acc)
    return res, res[-1]


def run_gpu(drv, fn, hufw_to_huflit, huflit_to_stream, huf_lit, huf_streams, literals_per_group):
    n = len(hufw_to_huflit)
    nb = (n + TG - 1) // TG
    hwM = dm(drv, hufw_to_huflit)
    hsM = dm(drv, huflit_to_stream)
    outM = dm(drv, hufw_to_huflit[:])   # in-place output, start = copy of input
    lkM = dm(drv, [0] * nb)
    ctr = [0] * 22
    ctr[_FIELD_ORDER.index("HufLit")] = huf_lit
    ctr[_FIELD_ORDER.index("HUF_Streams")] = huf_streams
    cM = dm(drv, ctr)
    daM = dm(drv, [0] * 64)
    args = [Buf(hwM.ptr, n), Buf(hsM.ptr, len(huflit_to_stream)), Buf(outM.ptr, n),
            Buf(lkM.ptr, nb), Buf(cM.ptr, 22), Buf(daM.ptr, 64),
            ct.c_uint32(0), ct.c_uint32(n), ct.c_uint32(literals_per_group)]
    fn(grid=(nb, 1, 1), block=(TG, 1, 1), args=args)
    drv.synchronize()
    out = list(struct.unpack(f"{n}I", outM.to_host(n * 4)))
    ctr_out = list(struct.unpack("22I", cM.to_host(22 * 4)))
    dlg = ctr_out[_FIELD_ORDER.index("DecompressLiteralsGroups")]
    return out, dlg


def main():
    random.seed(11)
    drv = loader.Driver(0)
    fn = drv.load(open(CUBIN, "rb").read()).function("main_0")

    def make(huf_lit, huf_streams, literals_per_group, n, includes_unused=True):
        # assign stream counts to tables; starts[t] = first lit-stream id of table t.
        starts = [0] * (huf_lit + 1)
        cum = 0
        for t in range(huf_lit):
            starts[t] = cum
            cum += [3, 2, 3, 2, 4][t % 5]
        starts[huf_lit] = cum
        huf_streams = cum
        h2l = [random.randrange(huf_lit) if (not includes_unused or random.random() > 0.3)
               else UNUSED for _ in range(n)]
        return h2l, starts, huf_lit, huf_streams

    cases = []
    for huf_lit, n in ((3, 4), (3, 64), (5, 65), (5, 128)):
        lg = (8, 16)[huf_lit % 2]
        h2l, starts, hl, hs = make(huf_lit, 0, lg, n)
        cases.append((f"huf_lit={huf_lit} n={n} lg={lg}", h2l, starts, hl, hs, lg))

    all_ok = True
    for name, h2l, starts, hl, hs, lg in cases:
        cpu_res, cpu_dlg = cpu_ref(h2l, starts, hl, hs, lg)
        gpu_res, gpu_dlg = run_gpu(drv, fn, h2l, starts, hl, hs, lg)
        bad = [(j, gpu_res[j], cpu_res[j]) for j in range(len(cpu_res)) if gpu_res[j] != cpu_res[j]]
        dlg_ok = (gpu_dlg == cpu_dlg)
        print(f"  {name}: {'PASS' if not bad and dlg_ok else 'FAIL'} "
              f"({len(bad)} mismatch, dlg {'PASS' if dlg_ok else 'FAIL'})")
        if bad:
            print("    first few:", bad[:6])
        all_ok &= (not bad and dlg_ok)

    drv.destroy()
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
